"""
Functions for setting up and parsing competing phase calculations, in order to
determine and analyse elemental chemical potential limits for a given host
system.
"""

import contextlib
import copy
import itertools
import os
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from functools import cache, partial
from pathlib import Path
from re import sub
from types import ModuleType
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labellines import labelLines
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.tri import Triangulation
from monty.json import MontyDecoder, MSONable
from monty.serialization import loadfn
from mp_api.client.mprester import DEFAULT_THERMOTYPE_CRITERIA, MPRester, MPRestError
from numpy.typing import NDArray
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.entries import (
    ComputedEntry,
    ComputedStructureEntry,
    ConstantEnergyAdjustment,
    ManualEnergyAdjustment,
)
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning
from pymatgen.util.string import latexify, latexify_spacegroup
from pymatgen.util.typing import PathLike
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from doped.generation import _element_sort_func
from doped.io import get_backend
from doped.io import utils as _io_utils
from doped.utils import _doped_obj_properties_methods, _ignore_pmg_warnings, get_mp_context, pool_manager
from doped.utils.efficiency import StructureMatcher_scan_stol
from doped.utils.plotting import doped_plot_style, get_colormap
from doped.utils.symmetry import _custom_round, _round_floats, get_primitive_structure

# globally ignore:
_ignore_pmg_warnings()

elemental_diatomic_bond_lengths = {"H": 0.74, "O": 1.21, "N": 1.10, "F": 1.42, "Cl": 1.99}

# TODO: Update chemical potentials tutorial notebook for new code/function names. Show example of
#  combining entries from a previously parsed CPA with a new one (i.e. similar functionality to previous
#  ``combine_extrinsic`` function, but more appropriate), and example adjusting entry energy as noted below
#  and maybe note the possibility of using ``single_extrinsic_phase_limits`` for generation but not parsing
#  as a reasonable approach to boosting efficiency without major accuracy loss.
# Check tutorial links working!
# Show example of generating `NKRED` folders for competing phases, and mention in docstrings.
# TODO: Use Codex to review the new module (and ask Claude to review all as well)

MPRESTER_PROPERTY_DATA = (  # properties to pull for Materials Project entries
    "material_id",  # populated in ``entry.data``; needed to map entries to summary docs
    "formula_pretty",
    "energy_above_hull",
    "nsites",
    "volume",
    "formation_energy_per_atom",  # note that this is the corrected formation energy
    "energy_per_atom",  # note that this is the corrected energy per atom
    "nelements",
    "elements",
)
MPRESTER_SUMMARY_DATA = (
    "material_id",  # required so we can map docs back to entries via MPID
    "band_gap",
    "total_magnetization",
    "theoretical",
    "database_IDs",  # dict, possibly with an "icsd" key with list of ICSD entry codes
)

default_get_entries_kwargs: dict[str, Any] = {"property_data": list(MPRESTER_PROPERTY_DATA)}


def _attach_summary_data_to_entries(
    entries: list[ComputedEntry],
    mpr: MPRester,
    summary_data: Iterable[str] = MPRESTER_SUMMARY_DATA,
) -> None:
    """
    Fetch ``summary_data`` fields for ``entries`` via
    ``mpr.materials.summary.search(fields=summary_data)`` and attach them in-
    place to ``entry.data["summary"]``.

    Stores values as JSON-safe dicts (via
    ``SummaryDoc.model_dump(mode="json")``) so entries round-trip cleanly
    through ``MontyEncoder`` (and ``ComputedEntry.to_json()``).
    """
    material_ids = [mpid for mpid in (entry.data.get("material_id") for entry in entries) if mpid]
    if material_ids:  # otherwise mp-api treats material_ids=[] as "no filter" and returns the entire DB...
        docs_by_mpid = {
            doc.material_id: doc.model_dump(mode="json")
            for doc in mpr.materials.summary.search(material_ids=material_ids, fields=list(summary_data))
        }
        for entry in entries:
            if doc := docs_by_mpid.get(entry.data.get("material_id")):
                # drop ``material_id`` (already in the entry data):
                entry.data["summary"] = {k: v for k, v in doc.items() if k != "material_id"}


def make_molecule_in_a_box(element: str) -> Structure:
    """
    Generate an X2 'molecule-in-a-box' structure for the input element X, (i.e.
    a 30 Å cuboid supercell with a single X2 molecule in the centre).

    This is the natural state of several elemental competing phases, such as
    O2, N2, H2, F2 and Cl2. Initial bond lengths are set to the experimental
    bond lengths of these gaseous molecules.

    Args:
        element (str):
            Element symbol of the molecule to generate.

    Returns:
        Structure:
            ``pymatgen`` |Structure| of the molecule in a box.
    """
    element = sub(r"\d+$", "", element)  # remove digits in case provided as X2 etc
    if element not in elemental_diatomic_bond_lengths:
        from shakenbreak.distortions import get_dimer_bond_length

        bond_length = get_dimer_bond_length(element, element)

    else:
        bond_length = elemental_diatomic_bond_lengths[element]

    lattice = np.array([[30.01, 0, 0], [0, 30.00, 0], [0, 0, 29.99]])
    return Structure(
        lattice=lattice,
        species=[element, element],
        coords=[[15, 15, 15], [15, 15, 15 + bond_length]],
        coords_are_cartesian=True,
    )


def make_molecular_entry(computed_entry: ComputedEntry) -> ComputedStructureEntry:
    """
    Generate a |ComputedStructureEntry| for a diatomic 'molecule-in-a-box' (X2)
    structure (e.g. O2, H2, Cl2, etc.), with composition and energy-per-atom
    set to match the input elemental |ComputedEntry|.

    The magnetization of the output molecular entry (as in
    ``entry.data["summary"]["total_magnetization"]``) is set to 0 for all X2
    except O2, which has a triplet ground state (S = 1) -- this is then used to
    set spin polarisation input settings (``ISPIN`` and ``NUPDOWN`` in
    ``VASP``) appropriately, with
    ``doped.io.vasp.inputs._set_spin_polarisation()``.

    Args:
        computed_entry (|ComputedEntry|):
            |ComputedEntry| object for a single-element entry, for which to
            generate a corresponding diatomic 'molecule-in-a-box' entry (with
            matching composition and energy-per-atom).

    Returns:
        |ComputedStructureEntry|:
            |ComputedStructureEntry| for the diatomic (X2) 'molecule-in-a-box'.
    """
    assert len(computed_entry.composition.elements) == 1  # Elemental!
    formula = computed_entry.data.get("formula_pretty", "N/A")
    element = computed_entry.composition.elements[0].symbol
    struct = make_molecule_in_a_box(element)
    molecular_entry = ComputedStructureEntry(
        structure=struct,
        energy=computed_entry.energy_per_atom * 2,  # set entry energy to be hull energy
        composition=Composition(formula),
        parameters=None,
    )
    molecular_entry.data["formula_pretty"] = formula
    molecular_entry.data["energy_above_hull"] = 0.0
    molecular_entry.data["nsites"] = 2
    molecular_entry.data["volume"] = 27000
    molecular_entry.data["formation_energy_per_atom"] = 0.0
    molecular_entry.data["energy_per_atom"] = computed_entry.data["energy_per_atom"]
    molecular_entry.data["nelements"] = 1
    molecular_entry.data["elements"] = [formula]
    molecular_entry.data["molecule"] = True
    molecular_entry.data["material_id"] = "mp-0"
    molecular_entry.data["summary"] = {
        "band_gap": None,
        "total_magnetization": 0 if element != "O" else 2,  # O2 has a triplet ground state (S = 1)
        "theoretical": False,
        "database_IDs": {},
    }

    return molecular_entry


def _renormalise_entry(
    entry: ComputedEntry | ComputedStructureEntry,
    renormalisation_energy_per_atom: float,
    name: str | None = None,
    description: str | None = None,
) -> ComputedEntry | ComputedStructureEntry:
    """
    Regenerate the input entry (|ComputedEntry|/|ComputedStructureEntry|) with
    an energy per atom `decreased` by ``renormalisation_energy_per_atom``, by
    appending an ``EnergyAdjustment`` object to ``entry.energy_adjustments``.

    Args:
        entry (|ComputedEntry|/|ComputedStructureEntry|):
            Input entry to renormalise.
        renormalisation_energy_per_atom (float):
            Energy to subtract from the entry's energy per atom.
        name (str):
            Name for the ``EnergyAdjustment`` object to be added to the entry.
            Default is ``None``.
        description (str):
            Description for the ``EnergyAdjustment`` object to be added to the
            entry. Default is ``None``.

    Returns:
        |ComputedEntry|/|ComputedStructureEntry|: Renormalised entry.
    """
    renormalisation_energy = -renormalisation_energy_per_atom * sum(entry.composition.values())
    if name is not None or description is not None:
        energy_adjustment = ConstantEnergyAdjustment(
            renormalisation_energy, name=name, description=description, cls=None
        )
    else:
        energy_adjustment = ManualEnergyAdjustment(renormalisation_energy)
    renormalised_entry = deepcopy(entry)
    renormalised_entry.energy_adjustments += [energy_adjustment]  # includes MP corrections as desired

    return renormalised_entry


def get_chempots_from_phase_diagram(
    bulk_computed_entry: ComputedEntry, phase_diagram: PhaseDiagram
) -> dict[str, dict[Element, float]]:
    """
    Get the chemical potential limits for the bulk computed entry in the
    supplied phase diagram.

    Args:
        bulk_computed_entry (|ComputedEntry|):
            |ComputedEntry|/|ComputedStructureEntry| object for the host
            composition.
        phase_diagram (PhaseDiagram):
            ``PhaseDiagram`` object for the system of interest.

    Returns:
        dict:
            Dictionary of the chemical potential limits for the bulk
            computed entry in the phase diagram.
    """
    # append bulk_computed_entry to phase diagram, if not present
    entries = phase_diagram.all_entries.copy()
    if not any(
        (ent.composition == bulk_computed_entry.composition and ent.energy == bulk_computed_entry.energy)
        for ent in entries
    ):
        entries.append(
            PDEntry(
                bulk_computed_entry.composition,
                bulk_computed_entry.energy,
                attribute="Bulk Material",
            )
        )
        phase_diagram = PhaseDiagram(entries)

    return phase_diagram.get_all_chempots(bulk_computed_entry.composition.reduced_composition)


def _get_all_chemsyses(chemsys: str | list[str]) -> list[str]:
    """
    Convert a chemical system (``chemsys``) string (in the old/usual "A-B-C" or
    ["A", "B", "C"] formats, used on Materials Project website) to format
    required for the new MP API (i.e. only returns AxBy phases for "A-B", and
    no A-only or B-only phases).

    See https://github.com/materialsproject/api/issues/918
    """
    if isinstance(chemsys, str):
        chemsys = chemsys.split("-")
    elements_set = set(chemsys)  # remove duplicate elements
    all_chemsyses: list[str] = []
    for i in range(len(elements_set)):
        all_chemsyses.extend("-".join(sorted(els)) for els in itertools.combinations(elements_set, i + 1))

    return all_chemsyses


def get_entries_in_chemsys(
    chemsys: str | list[str],
    api_key: str | None = None,
    energy_above_hull: float | None = None,
    bulk_composition: str | Composition | None = None,
    **kwargs: Any,
) -> list[ComputedStructureEntry]:
    r"""
    Convenience function to get a list of |ComputedStructureEntry|\s for an
    input chemical system, using ``MPRester.get_entries_in_chemsys()``.

    Automatically uses the appropriate format and syntax required for the (new)
    Materials Project (MP) API. ``chemsys = ["Li", "Fe", "O"]`` will return a
    list of all entries in the Li-Fe-O chemical system, i.e., all LixOy, FexOy,
    LixFey, LixFeyOz, Li, Fe and O phases. Extremely useful for creating phase
    diagrams of entire chemical systems.

    If ``energy_above_hull`` is supplied, then only entries with energies above
    hull (according to the MP-computed phase diagram) less than this value (in
    eV/atom) will be returned.

    The output entries list is sorted by energy above hull, then by the number
    of elements in the formula, then by the position of elements in the
    periodic table (main group elements, then transition metals, sorted by
    row).

    Args:
        chemsys (str, list[str]):
            Chemical system to get entries for, in the format "A-B-C" or
            ["A", "B", "C"]. E.g. "Li-Fe-O" or ["Li", "Fe", "O"].
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            to obtain the corresponding |ComputedStructureEntry|\s. If not
            supplied, will attempt to read from ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml`` (under ``PMG_MAPI_KEY``) or from the
            ``MP_API_KEY`` environment variable -- see the ``doped``
            :ref:`Installation docs <setup_potcars_mp_api>`.
        energy_above_hull (float):
            If supplied, only entries with energies above hull (according to
            the MP-computed phase diagram) less than this value (in eV/atom)
            will be returned. Set to 0 to only return phases on the MP convex
            hull. Default is ``None`` (i.e. all entries are returned).
        bulk_composition (str/|Composition|):
            Optional input; formula of the bulk host material, to use for
            sorting the output entries (with all those matching the bulk
            composition first). Default is ``None``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            ``get_entries_in_chemsys()`` query.

    Returns:
        list[|ComputedStructureEntry|]:
            List of |ComputedStructureEntry| objects for the input chemical
            system.
    """
    with MPRester(api_key) as mpr:
        # get all entries in the chemical system
        MP_full_pd_entries = mpr.get_entries_in_chemsys(
            elements=chemsys,
            **default_get_entries_kwargs,
            **kwargs,
        )
        _attach_summary_data_to_entries(MP_full_pd_entries, mpr)

    temp_phase_diagram = PhaseDiagram(MP_full_pd_entries)
    for entry in MP_full_pd_entries:
        # reparse energy above hull, to avoid mislabelling issues noted in Materials Project database
        # (mostly only the legacy database, so should be resolved with the new MP API database now);
        # e.g. search "F", or ZnSe2 on Zn-Se convex hull from MP PD, but EaH = 0.147 eV/atom?
        # or Immm phases for Br, I..., Na2FePO4F... # TODO: Check if needed, if not can cut this function?
        # only other thing is sorting, can be done after
        entry.data["energy_above_hull"] = temp_phase_diagram.get_e_above_hull(entry)

    if energy_above_hull is not None:
        MP_full_pd_entries = [
            entry
            for entry in MP_full_pd_entries
            if entry.data.get("energy_above_hull", 0) <= energy_above_hull
        ]

    # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
    MP_full_pd_entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=bulk_composition))

    return MP_full_pd_entries


def get_entries(
    chemsys_formula_id_criteria: str | dict[str, Any],
    api_key: str | None = None,
    bulk_composition: str | Composition | None = None,
    **kwargs: Any,
) -> list[ComputedStructureEntry]:
    r"""
    Convenience function to get a list of |ComputedStructureEntry|\s for an
    input single composition/formula, chemical system, MPID or full criteria,
    using ``MPRester.get_entries()``.

    The output entries list is sorted by energy per atom (equivalent sorting as
    energy above hull), then by the number of elements in the formula, then by
    the position of elements in the periodic table (main group elements, then
    transition metals, sorted by row).

    Args:
        chemsys_formula_id_criteria (str/dict):
            A formula (e.g., Fe2O3), chemical system (e.g., Li-Fe-O) or MPID
            (e.g., mp-1234) or full Mongo-style dict criteria.
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            to obtain the corresponding |ComputedStructureEntry|\s. If not
            supplied, will attempt to read from ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml`` (under ``PMG_MAPI_KEY``) or from the
            ``MP_API_KEY`` environment variable -- see the ``doped``
            :ref:`Installation docs <setup_potcars_mp_api>`.
        bulk_composition (str/|Composition|):
            Optional input; formula of the bulk host material, to use for
            sorting the output entries (with all those matching the bulk
            composition first). Default is ``None``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            ``get_entries()`` query.

    Returns:
        list[|ComputedStructureEntry|]:
            List of |ComputedStructureEntry| objects for the input chemical
            system.
    """
    with MPRester(api_key) as mpr:
        entries = mpr.get_entries(
            chemsys_formula_id_criteria,
            **default_get_entries_kwargs,
            **kwargs,
        )
        _attach_summary_data_to_entries(entries, mpr)

    # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
    entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=bulk_composition))

    return entries


def _check_MP_API_key(api_key: str | None = None) -> str | None:
    """
    Check that the Materials Project (MP) API key is valid, throwing an
    informative error if not.

    Args:
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            (for phase diagram analysis, competing phase generation etc). If
            not supplied, will attempt to read from ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml`` (under ``PMG_MAPI_KEY``) or from the
            ``MP_API_KEY`` environment variable -- see the ``doped``
            :ref:`Installation docs <setup_potcars_mp_api>`.

    Returns:
        str | None:
            The Materials Project (MP) API key, if valid.
    """
    try:
        with MPRester(api_key) as mpr:
            mpr.get_entry_by_material_id("mp-1")  # check if API key is valid
    except MPRestError as mp_exc:
        raise ValueError(
            f"The supplied API key (either ``api_key={api_key}``, the ``MP_API_KEY`` environment "
            f"variable, or 'PMG_MAPI_KEY' in ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``; in order of "
            f"precedence) is not a valid Materials Project API key, which is required for automatic "
            f"competing phase generation. See the doped installation instructions for details:\n"
            f"https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
            f"-project-api"
        ) from mp_exc

    return api_key


def get_MP_summary_dicts(
    entries: list[ComputedEntry] | None = None,
    chemsys: str | list[str] | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> dict[str, dict]:
    r"""
    Get the corresponding Materials Project (MP) summary dictionaries for
    computed entries in the input ``entries`` list or ``chemsys`` chemical
    system.

    If ``entries`` is provided (which should be a list of |ComputedEntry|\s
    from the Materials Project), then only summary dictionaries in this
    chemical system which match one of these entries (based on the MPIDs given
    in ``ComputedEntry.entry_id``/``ComputedEntry.data["material_id"]`` and
    ``summary_dict["material_id"]``) are returned.

    Args:
        entries (list[|ComputedEntry|]):
            Optional input; list of |ComputedEntry| objects for the input
            chemical system. If provided, only summary dictionaries which match
            one of these entries (based on the MPIDs given in
            ``ComputedEntry.entry_id``/``ComputedEntry.data["material_id"]``
            and ``summary_dict["material_id"]``) are returned.
        chemsys (str, list[str]):
            Optional input; chemical system to get entries for, in the format
            ``"A-B-C"`` or ``["A", "B", "C"]``. E.g. ``"Li-Fe-O"`` or
            ``["Li", "Fe", "O"]``. Either ``entries`` or ``chemsys`` must be
            provided!
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            to obtain the corresponding summary dictionaries. If not
            supplied, will attempt to read from ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml`` (under ``PMG_MAPI_KEY``) or from the
            ``MP_API_KEY`` environment variable -- see the ``doped``
            :ref:`Installation docs <setup_potcars_mp_api>`.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            query, e.g. ``MPRester.materials.summary.search()``.

    Returns:
        dict[str, dict]:
            Dictionary of summary dictionaries for the input chemical system.
            The keys are the MPIDs (e.g. ``"mp-1234"``) and the values are the
            corresponding summary dictionaries.
    """
    _check_MP_API_key(api_key)
    if entries is None and chemsys is None:
        raise ValueError("Either `entries` or `chemsys` must be provided!")

    summary_search_kwargs = {**kwargs}
    if entries:
        summary_search_kwargs["material_ids"] = [entry.data["material_id"] for entry in entries]
    elif chemsys is not None:  # convert to ``MPRester.summary.search`` chemsys format:
        summary_search_kwargs["chemsys"] = _get_all_chemsyses("-".join(chemsys))

    with MPRester(api_key) as mpr:  # ``SummaryDoc`` -> JSON-safe dict via ``model_dump(mode="json")``
        MP_doc_dicts = {
            doc.material_id: doc.model_dump(mode="json")
            for doc in mpr.materials.summary.search(**summary_search_kwargs)
        }

    if not entries:
        return MP_doc_dicts

    for entry in entries:
        entry.MP_doc_dict = MP_doc_dicts.get(entry.data["material_id"])
        if entry.MP_doc_dict is None and entry.data["material_id"] != "mp-0":
            warnings.warn(
                f"No matching summary dictionary found for entry {entry.name, entry.data['material_id']} "
                f"in the Materials Project API database."
            )

    return MP_doc_dicts


def _entries_sort_func(
    entry: ComputedEntry,
    use_e_per_atom: bool = False,
    bulk_composition: str | Composition | dict | list | None = None,
) -> tuple[float, bool, int, list[tuple[int, int]], str]:
    r"""
    Function to sort |ComputedEntry|\s by energy above hull, then if
    composition matches ``bulk_composition`` (if provided), then by the number
    of elements in the formula, then by the position of elements in the
    periodic table (main group elements, then transition metals, sorted by
    row), then alphabetically.

    Usage: ``entries_list.sort(key=_entries_sort_func)``

    Args:
        entry (|ComputedEntry|):
            |ComputedEntry| object to sort.
        use_e_per_atom (bool):
            If ``True``, sort by energy per atom rather than energy above hull.
            Default is ``False``.
        bulk_composition (str/|Composition|/dict/list):
            Bulk composition; to sort entries matching this composition first.
            Default is ``None`` (don't sort according to this).

    Returns:
        tuple[float, bool, int, list[tuple[int, int]], str]:
            Sort key: energy above hull (or energy per atom if
            ``use_e_per_atom``), whether the composition differs from the bulk
            composition (when ``bulk_composition`` is set), number of species
            in ``entry.name``, periodic-table ordering tuples for each element,
            then ``entry.name``.
    """
    bulk_reduced_comp = Composition(bulk_composition).reduced_composition if bulk_composition else None
    return (
        entry.energy_per_atom if use_e_per_atom else entry.data.get("energy_above_hull", 0),
        entry.composition.reduced_composition != bulk_reduced_comp,  # goes from False to True
        len(Composition(entry.name).as_dict()),
        sorted([_element_sort_func(i.symbol) for i in Composition(entry.name).elements]),
        entry.name,
    )


def prune_entries_to_border_candidates(
    entries: list[ComputedEntry],
    bulk_computed_entry: ComputedEntry,
    phase_diagram: PhaseDiagram | None = None,
    energy_above_hull: float = 0.05,
) -> list[ComputedEntry]:
    r"""
    Given an input list of |ComputedEntry|/|ComputedStructureEntry|\s
    (``entries``) and a single entry for the host material
    (``bulk_computed_entry``), returns the subset of entries which `could`
    border the host on the phase diagram (and therefore be a competing phase
    which determines the host chemical potential limits), allowing for an error
    tolerance (i.e. for semi-local DFT database energies
    (``energy_above_hull``, set to ``self.energy_above_hull`` -- 0.05 eV/atom
    by default)).

    If ``phase_diagram`` is provided then this is used as the reference phase
    diagram, otherwise it is generated from ``entries`` and
    ``bulk_computed_entry``.

    Args:
        entries (list[|ComputedEntry|]):
            List of |ComputedEntry| objects to prune down to potential host
            border candidates on the phase diagram.
        bulk_computed_entry (|ComputedEntry|):
            |ComputedEntry| object for the host material.
        phase_diagram (PhaseDiagram):
            Optional input; ``PhaseDiagram`` object for the system of interest.
            If provided, this is used as the reference phase diagram from which
            to determine the (potential) chemical potential limits, otherwise
            it is generated from ``entries`` and ``bulk_computed_entry``.
        energy_above_hull (float):
            Maximum energy above hull (in eV/atom) of Materials Project entries
            to be considered as competing phases. This is an uncertainty range
            for the MP-calculated formation energies, which may not be accurate
            due to functional choice (e.g. GGA vs hybrid DFT / GGA+U / RPA etc.),
            lack of vdW corrections etc. All phases that would border the host
            material on the phase diagram, if their relative energy was
            downshifted by ``energy_above_hull``, are included.
            (Default is 0.05 eV/atom).

    Returns:
        list[|ComputedEntry|]:
            List of all |ComputedEntry| objects in ``entries`` which could
            border the host material on the phase diagram (and thus set the
            chemical potential limits), if their relative energy was
            downshifted by ``energy_above_hull`` eV/atom.
    """
    if not phase_diagram:
        phase_diagram = PhaseDiagram({bulk_computed_entry, *entries})

    # cull to only include any phases that would border the host material on the phase
    # diagram, if their relative energy was downshifted by ``energy_above_hull``:
    MP_chempots = get_chempots_from_phase_diagram(bulk_computed_entry, phase_diagram)
    MP_bordering_phases = {phase for limit in MP_chempots for phase in limit.split("-")}
    bordering_entries = [
        entry for entry in entries if entry.name in MP_bordering_phases or entry.is_element
    ]
    bordering_entry_names = [
        bordering_entry.name for bordering_entry in bordering_entries
    ]  # compositions which border the host with EaH=0, according to MP, so we include all phases with
    # these compositions up to EaH=energy_above_hull (which we've already pruned to)
    # for determining phases which alter the chemical potential limits when renormalised, only need to
    # retain the EaH=0 entries from above, so we use this reduced PD to save compute time when looping
    # below:
    reduced_pd_entries = {
        entry for entry in bordering_entries if entry.data.get("energy_above_hull", 0) == 0
    }

    # then add any other phases that would border the host material on the phase diagram, if their
    # relative energy was downshifted by ``energy_above_hull``:
    # only check if not already bordering; can just use names for this:
    entries_to_test = [entry for entry in entries if entry.name not in bordering_entry_names]
    entries_to_test.sort(key=_entries_sort_func)  # sort by energy above hull
    # to save unnecessary looping, whenever we encounter a phase that is not being added to the border
    # candidates list, skip all following phases with this composition (because they have higher
    # energies above hull (because we've sorted by this) and so will also not border the host):
    compositions_to_skip = []
    for entry in entries_to_test:
        if entry.name not in compositions_to_skip:
            # decrease entry energy per atom by ``energy_above_hull`` eV/atom
            renormalised_entry = _renormalise_entry(entry, energy_above_hull)
            new_phase_diagram = PhaseDiagram(
                [*reduced_pd_entries, bulk_computed_entry, renormalised_entry]
            )
            shifted_MP_chempots = get_chempots_from_phase_diagram(bulk_computed_entry, new_phase_diagram)
            shifted_MP_bordering_phases = {
                phase for limit in shifted_MP_chempots for phase in limit.split("-")
            }

            if shifted_MP_bordering_phases != MP_bordering_phases:  # new bordering phase, add to list
                bordering_entries.append(entry)
            else:
                compositions_to_skip.append(entry.name)

    return bordering_entries


@cache
def _known_MP_ground_states() -> dict[Composition, frozenset[str]]:
    """
    Load the ``known_MP_ground_states.yaml`` data file as a mapping of reduced
    |Composition| to the set of Materials Project IDs to retain when
    ``expected_polymorphs=True`` in |CompetingPhases|.
    """
    raw: dict[str, list[str]] = loadfn(
        str(Path(__file__).parent / "utils" / "known_MP_ground_states.yaml")
    )
    return {Composition(formula).reduced_composition: frozenset(mpids) for formula, mpids in raw.items()}


def prune_to_expected_polymorphs(entries: list[ComputedEntry]) -> list[ComputedEntry]:
    r"""
    For each composition listed in ``doped/utils/known_MP_ground_states.yaml``,
    retain only the lowest energy-above-hull entry plus any entries whose
    Materials Project ID is included in the listed set; entries for
    compositions not listed in the data file are returned unchanged.

    Args:
        entries (list[|ComputedEntry|]):
            Input list of |ComputedEntry| objects to prune.

    Returns:
        list[|ComputedEntry|]:
            Pruned list of entries.
    """
    known = _known_MP_ground_states()
    grouped: dict[Composition, list[ComputedEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.composition.reduced_composition].append(entry)

    pruned: list[ComputedEntry] = []
    for reduced_comp, comp_entries in grouped.items():
        if reduced_comp not in known:
            pruned.extend(comp_entries)
            continue

        allowed_mpids = known[reduced_comp]  # Materials Project IDs to retain
        lowest = min(comp_entries, key=lambda e: e.data.get("energy_above_hull", float("inf")))
        pruned.extend(e for e in comp_entries if e is lowest or e.data.get("material_id") in allowed_mpids)

    return pruned


def _warn_if_many_polymorphs_per_composition(entries: list[ComputedEntry], threshold: int = 5) -> None:
    """
    Warn if any composition in ``entries`` has more than ``threshold``
    polymorphs, suggesting that the user prune them using knowledge of the
    expected ground-state / room-temperature phase(s).
    """
    comp_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        comp_counts[entry.composition.reduced_formula] += 1
    many_polymorphs = {comp: n for comp, n in comp_counts.items() if n > threshold}
    if not many_polymorphs:
        return

    details = ", ".join(f"{comp} ({n})" for comp, n in many_polymorphs.items())
    warnings.warn(
        f"The following competing phase composition(s) have more than {threshold} entries to consider: "
        f"{details}. While often unavoidable (e.g. for chemical systems with many low-energy "
        f"polymorphs/allotropes), in some cases knowledge of the expected ground-state phase(s) can be "
        f"used to prune these entries and minimise costs. See the discussion and examples at "
        f"https://doped.readthedocs.io/en/latest/chemical_potentials_tutorial.html"
        f"#chemical-systems-with-many-polymorphs"
    )


def get_and_set_competing_phase_name(
    entry: ComputedStructureEntry | ComputedEntry,
    regenerate: bool = False,
    ndigits: int = 3,
) -> str:
    """
    Get the ``doped`` name for a competing phase entry from the Materials
    Project (MP) database.

    The default naming convention in ``doped`` for competing phases is:
    ``"{Chemical Formula}_{Space Group}_EaH_{MP Energy above Hull}"``. This is
    stored in the ``entry.data["doped_name"]`` key-value pair. If this value is
    already set, then this function just returns the previously-generated
    ``doped`` name, unless ``regenerate=True``.

    Args:
        entry (|ComputedStructureEntry|, |ComputedEntry|):
            ``pymatgen`` |ComputedStructureEntry| object for the competing
            phase.
        regenerate (bool):
            Whether to regenerate the ``doped`` name for the competing phase,
            if ``entry.data["doped_name"]`` already set. Default is ``False``.
        ndigits (int):
            Number of digits to round the energy above hull value (in eV/atom)
            to. Default is 3.

    Returns:
        doped_name (str):
            The ``doped`` name for the competing phase.
    """
    if not entry.data.get("doped_name") or regenerate:  # not set, so generate
        rounded_eah = round(entry.data.get("energy_above_hull", 0), ndigits)

        if np.isclose(rounded_eah, 0):
            rounded_eah = 0

        if entry.data.get("molecule"):
            space_group = "mmm"  # just point group
        elif hasattr(entry, "structure"):
            space_group = entry.structure.get_space_group_info(symprec=0.1)[0]
        else:
            space_group = "NA"
        entry.data["doped_name"] = f"{entry.name}_{space_group}_EaH_{rounded_eah}"

    return entry.data.get("doped_name")


def _nominal_structure_for_input_writing(composition: Composition) -> Structure:
    """
    Build a sparse cubic cell with the correct stoichiometry so VASP input sets
    can be written when an entry has no crystal structure (e.g. MP-missing host
    composition represented only by a hull-energy |ComputedEntry|).
    """
    reduced_comp, _factor = composition.get_reduced_composition_and_factor()
    species: list[Element | str] = []
    for element in sorted(reduced_comp.elements, key=lambda el: el.symbol):
        species.extend([element] * int(reduced_comp[element]))

    lattice_parameter = 100.0
    site_spacing = lattice_parameter / len(species)
    cart_coords = [
        [site_spacing * (i + 1), lattice_parameter / 2, lattice_parameter / 2] for i in range(len(species))
    ]
    return Structure(
        lattice=Lattice.cubic(lattice_parameter),
        species=species,
        coords=cart_coords,
        coords_are_cartesian=True,
        properties={"_is_nominal_structure": True},
    )


def _get_competing_phase_folder_name(
    entry: ComputedStructureEntry | ComputedEntry,
    regenerate: bool = False,
    ndigits: int = 3,
) -> str:
    """
    Get the ``doped`` `folder` name for a competing phase entry from the
    Materials Project (MP) database, handling the possibility of slashes ("/")
    in the formula name (due to space group symbols such as C2/m, P2_1/c etc.)
    by removing them (-> C2m, P2_1c etc.).

    The default naming convention in ``doped`` for competing phases is:
    ``"{Chemical Formula}_{Space Group}_EaH_{MP Energy above Hull}"``, which is
    stored in the ``entry.data["doped_name"]`` key-value pair. If this value is
    already set, then this function just returns the previously-generated
    ``doped`` name, unless ``regenerate=True``.

    Args:
        entry (|ComputedStructureEntry|, |ComputedEntry|):
            ``pymatgen`` |ComputedStructureEntry| object for the competing
            phase.
        regenerate (bool):
            Whether to regenerate the ``doped`` name for the competing phase,
            if ``entry.data["doped_name"]`` already set. Default is ``False``.
        ndigits (int):
            Number of digits to round the energy above hull value (in eV/atom)
            to. Default is 3.

    Returns:
        folder_name (str):
            The ``doped`` `folder` name for the competing phase, to use
            when generating calculation inputs.
    """
    return get_and_set_competing_phase_name(entry, regenerate=regenerate, ndigits=ndigits).replace("/", "")


def _name_entries_and_handle_duplicates(
    entries: list[ComputedStructureEntry | ComputedEntry],
) -> None:
    """
    Set ``entry.data["doped_name"]`` for each |ComputedEntry| in ``entries``,
    using ``get_and_set_competing_phase_name``, increasing ``ndigits``
    (rounding for energy above hull in name) dynamically from 3 -> 4 -> 5 on
    any entries with duplicate names, to ensure unique naming.
    """
    ndigits = 3
    entry_names = [get_and_set_competing_phase_name(entry, ndigits=ndigits) for entry in entries]
    while duplicate_entries := [
        entries[i] for i, name in enumerate(entry_names) if entry_names.count(name) > 1
    ]:
        ndigits += 1
        if ndigits == 5:
            warnings.warn(
                f"Duplicate entry names found for generated competing phases: "
                f"{get_and_set_competing_phase_name(duplicate_entries[0])}!"
            )
            break
        # regenerate names for duplicates only; ``entry_names`` then picks up the set values via
        # ``regenerate=False`` (which returns the already-updated ``entry.data["doped_name"]``):
        for entry in duplicate_entries:
            get_and_set_competing_phase_name(entry, regenerate=True, ndigits=ndigits)
        entry_names = [get_and_set_competing_phase_name(entry, regenerate=False) for entry in entries]


class CompetingPhases(MSONable):
    def __init__(
        self,
        composition: str | Composition | Structure,
        energy_above_hull: float = 0.05,
        extrinsic: str | Element | Iterable[str] | Iterable[Element] | None = None,
        full_phase_diagram: bool = False,
        single_extrinsic_phase_limits: bool = False,
        codoping: bool = False,
        expected_polymorphs: bool = False,
        MP_doc_dicts: bool = False,
        api_key: str | None = None,
        calculator: str = "vasp",
        **kwargs: Any,
    ):
        """
        Class to generate calculation input files for competing phases on the
        phase diagram for the host material (and any extrinsic
        (dopant/impurity) elements), which determine the chemical potential
        limits for elements in that compound.

        For this, the Materials Project (MP) database is queried using the
        ``MPRester`` API, and any calculated compounds which `could` border the
        host material within an error tolerance for the semi-local DFT database
        energies (``energy_above_hull``, 0.05 eV/atom by default) are
        generated, along with the elemental reference phases. Diatomic gaseous
        molecules are generated as molecules-in-a-box as appropriate (e.g. for
        O2, F2, H2 etc).

        Often ``energy_above_hull`` can be lowered (e.g. to ``0``) to reduce
        the number of calculations while retaining good accuracy relative to
        the typical error of defect calculations.

        The default ``energy_above_hull`` of 50 meV/atom works well in
        accounting for MP formation energy inaccuracies in most known cases.
        However, some critical thinking is key (as always!) and so if there are
        any obvious missing phases or known failures of the Materials Project
        energetics in your chemical space of interest, you should adjust this
        parameter to account for this (or alternatively manually include these
        known missing phases in your competing phase calculations, to be
        included in parsing and chemical potential analysis later on).

        Particular attention should be paid for materials containing transition
        metals, (inter)metallic systems, mixed oxidation states, van der Waals
        (vdW) binding and/or large spin-orbit coupling (SOC) effects, for which
        the Materials Project energetics are typically less reliable.

        Args:
            composition (str, |Composition|, |Structure|):
                Composition of the host material (e.g. ``'LiFePO4'``, or
                ``Composition('LiFePO4')``, or
                ``Composition({"Li":1, "Fe":1, "P":1, "O":4})``).
                Alternatively a ``pymatgen`` |Structure| object for the
                host material can be supplied (recommended), in which case
                the primitive structure will be used as the only host
                composition phase, reducing the number of calculations.
            energy_above_hull (float):
                Maximum energy above hull (in eV/atom) of Materials Project
                entries to be considered as competing phases. This is an
                uncertainty range for the MP-calculated formation energies,
                which may not be accurate due to functional choice (e.g. GGA
                vs hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
                All phases that would border the host material on the phase
                diagram, if their relative energy was downshifted by
                ``energy_above_hull``, are included.
                Often ``energy_above_hull`` can be lowered (e.g. to ``0``) to
                reduce the number of calculations while retaining good accuracy
                relative to the typical error of defect calculations.
                (Default is 0.05 eV/atom).
            extrinsic (str | Element | Iterable[str] | Iterable[Element] | None):
                Extrinsic dopant/impurity species to consider, to generate the
                relevant competing phases to additionally determine their
                chemical potential limits within the host. Can be a single
                element as a string (e.g. "Mg") or an iterable of element
                strings (list, set, tuple, dict) (e.g. ["Mg", "Na"]).
            full_phase_diagram (bool):
                If ``True``, include all phases on the MP phase diagram (with
                energy above hull < ``energy_above_hull`` eV/atom) for the
                chemical system of the input composition and any extrinsic
                species. Typically not recommended. If ``False`` (default),
                only includes phases that `would` border the host material on
                the phase diagram (and thus set the chemical potential limits),
                if their relative energy was downshifted by
                ``energy_above_hull`` eV/atom. (Default is ``False``).
            single_extrinsic_phase_limits (bool):
                If ``True``, and ``extrinsic`` is not ``None``, only consider
                chemical potential limits where the host composition borders a
                maximum of 1 extrinsic phase (composition with extrinsic
                element(s)). This is an approximation that can reduce the
                number of extrinsic phases to consider (to reduce computational
                costs, e.g. when considering many possible extrinsic dopants /
                impurities), while often (but not always) being `reasonably`
                accurate for dilute extrinsic concentrations. Default is
                ``False``.
            codoping (bool):
                Whether to consider extrinsic competing phases containing
                `multiple` extrinsic species (e.g. ``KInO2`` as a potential
                competing phase for ``BaSnO3`` with ``K`` and ``In`` extrinsic
                species). Usually only relevant under high (non-dilute)
                co-doping concentrations. Ignored if ``extrinsic`` is a single
                element. If set to ``True`` (and multiple extrinsic species
                present), then ``single_extrinsic_phase_limits`` is forced to
                be ``False``. Default is ``False``.
            expected_polymorphs (bool):
                If ``True``, prune competing phase entries to expected ground-
                state / room-temperature polymorphs for compositions listed in
                ``doped/utils/known_MP_ground_states.yaml`` (common elements
                and oxides with many low-energy polymorphs on the MP database,
                and known ground-state phases). For each listed composition,
                only the lowest MP energy-above-hull entry plus any entries
                whose Materials Project ID is in the listed set are retained.
                See ``docs/known_MP_ground_states.md`` for the rationale and
                citations behind each composition's curated set. See the
                :ref:`chemical_potentials_tutorial:Chemical Systems with Many Polymorphs`
                section of the competing phases tutorial for discussion and
                examples. Default is ``False``.
            MP_doc_dicts (bool):
                If ``True``, also queries the Materials Project (MP) for
                summary doc dicts with ``MPRester.summary_search()`` for the
                competing phase entries, and stores them in
                ``CompetingPhases.MP_doc_dicts``. Default is ``False``.
            api_key (str):
                Materials Project (MP) API key, needed to access the MP
                database for competing phase generation. If not supplied,
                will attempt to read from ``~/.pmgrc.yaml`` or
                ``~/.config/.pmgrc.yaml`` (under ``PMG_MAPI_KEY``) or from
                the ``MP_API_KEY`` environment variable -- see the ``doped``
                :ref:`Installation docs <setup_potcars_mp_api>`.
            calculator (str):
                Name of the calculator to generate input files for (matching
                a ``doped.io.<calculator>`` subpackage with an ``inputs``
                module). Default: "vasp".
            **kwargs:
                Additional keyword arguments to pass to the Materials Project
                API ``get_entries_in_chemsys()`` / ``get_entries()`` queries
                used to pull competing phase entries.

        Key Attributes:
            entries (list[|ComputedEntry|]):
                Final list of competing phase entries (intrinsic and, if
                requested, extrinsic) to consider for chemical potential
                analysis.
            intrinsic_entries (list[|ComputedEntry|]):
                Subset of ``entries`` corresponding to phases in the host
                (intrinsic) chemical system only (no extrinsic species).
            extrinsic_entries (list[|ComputedEntry|]):
                Subset of ``entries`` corresponding to phases containing one
                or more extrinsic species. Empty if ``extrinsic`` is ``None``.
            metallic_entries / nonmetallic_entries / molecular_entries (list[|ComputedEntry|]):
                Properties returning the subsets of ``entries`` that are
                metallic, non-metallic (according to the Materials Project band
                gap values), or diatomic molecules (generated by ``doped``),
                respectively (used to set smearing parameters, `k`-point
                convergence ranges etc. when writing calculation input files).
            composition (|Composition|):
                ``pymatgen`` |Composition| of the host material.
            bulk_structure (|Structure| | None):
                Primitive ``pymatgen`` |Structure| of the host (if a
                |Structure| was supplied as ``composition``), else ``None``.
            intrinsic_elements (list[str]):
                Element symbols of the host composition.
            extrinsic_elements (list[str]):
                Element symbols of the extrinsic species (only set when
                ``extrinsic`` is not ``None``).
            MP_full_pd_entries (list[|ComputedStructureEntry|]):
                All Materials Project entries in the (intrinsic + extrinsic)
                chemical system with energy above hull <
                ``energy_above_hull``, used to build ``MP_full_pd``.
            MP_full_pd (|PhaseDiagram|):
                ``pymatgen`` |PhaseDiagram| built from ``MP_full_pd_entries``.
            MP_intrinsic_full_pd_entries (list[|ComputedStructureEntry|]):
                ``MP_full_pd_entries`` restricted to the intrinsic chemical
                system (only set when ``extrinsic`` is ``None``).
            MP_bulk_computed_entry (|ComputedStructureEntry|):
                Materials Project |ComputedStructureEntry| for the host
                composition (downshifted to the convex hull if MP-unstable, or
                a hull-energy placeholder if not present on MP).
            MP_doc_dicts (dict[str, dict]):
                Materials Project summary dictionaries for ``self.entries``,
                keyed by entry name (empty unless ``MP_doc_dicts=True``).
            intrinsic_MP_doc_dicts (dict[str, dict]):
                Materials Project summary dictionaries for
                ``self.intrinsic_entries`` only (set when ``MP_doc_dicts=True``
                and extrinsic species are present).
            X (Any):
                Stored constructor (initialisation) arguments, for reference.
                e.g. ``CompetingPhases.energy_above_hull``. See ``Args``.
            _get_entries_kwargs (dict):
                ``**kwargs`` passed to MP ``get_entries*`` queries; see
                ``Args``.
        """
        if "full_sub_approach" in kwargs:  # TODO: remove in v4.1
            raise ValueError(
                "`full_sub_approach` was renamed to `single_extrinsic_phase_limits` in doped v4.0, "
                "with inverted polarity (``full_sub_approach=True`` corresponds to "
                "``single_extrinsic_phase_limits=False``). The default has also changed: "
                "``single_extrinsic_phase_limits=False`` is now the default and recommended approach."
            )
        self.energy_above_hull = energy_above_hull  # store parameters for reference
        self.full_phase_diagram = full_phase_diagram
        self.extrinsic = extrinsic
        self.codoping = codoping
        self.single_extrinsic_phase_limits = single_extrinsic_phase_limits
        self.expected_polymorphs = expected_polymorphs
        _check_MP_API_key(api_key)
        self.api_key = api_key
        self.calculator = calculator
        self._get_entries_kwargs = kwargs

        if isinstance(composition, Structure):
            primitive_structure = get_primitive_structure(composition)
            self.bulk_structure = (  # use primitive structure if input is not primitive
                primitive_structure if len(primitive_structure) < len(composition) else composition
            )
            self.composition = self.bulk_structure.composition

        else:
            self.bulk_structure = None
            self.composition = Composition(composition)

        self.intrinsic_elements = [s.symbol for s in self.composition.reduced_composition.elements]

        # get all entries in the chemical system with EaH < ``energy_above_hull``:
        self.MP_full_pd_entries = get_entries_in_chemsys(
            self.intrinsic_elements,
            api_key=self.api_key,
            energy_above_hull=self.energy_above_hull,
            bulk_composition=self.composition.reduced_formula,  # for sorting
            **self._get_entries_kwargs,
        )
        additional_criteria = self._get_entries_kwargs.setdefault("additional_criteria", {})
        if not additional_criteria.get("thermo_types"):
            # matching mp-api default thermo type criteria, to ensure ``get_entries()`` thermo_type matches
            # ``get_entries_in_chemsys()`` default handling (placed after ``get_entries_in_chemsys`` to
            # show mp-api warning about updated default thermo type criteria)
            additional_criteria["thermo_types"] = DEFAULT_THERMOTYPE_CRITERIA["thermo_types"]

        self.MP_full_pd = PhaseDiagram(self.MP_full_pd_entries)

        # convert any gaseous elemental entries to molecules in a box
        formatted_entries = self._generate_elemental_diatomic_phases(self.MP_full_pd_entries)

        # get bulk entry, and warn if not stable or not present on MP database:
        bulk_entries = [
            entry
            for entry in formatted_entries  # sorted by energy_above_hull above in get_entries_in_chemsys
            if entry.composition.reduced_composition == self.composition.reduced_composition
        ]
        if zero_eah_bulk_entries := [
            entry for entry in bulk_entries if entry.data.get("energy_above_hull", 0) == 0.0
        ]:
            self.MP_bulk_computed_entry = bulk_computed_entry = zero_eah_bulk_entries[
                0
            ]  # lowest energy entry for bulk (after sorting)
        else:  # no EaH=0 bulk entries in pruned phase diagram, check first if present (but unstable)
            if bulk_entries := get_entries(  # composition present in MP, but not stable
                self.composition.reduced_formula,
                api_key=self.api_key,
                bulk_composition=self.composition.reduced_formula,  # for sorting
                **self._get_entries_kwargs,
            ):
                self.MP_bulk_computed_entry = bulk_computed_entry = bulk_entries[
                    0
                ]  # already sorted by energy in get_entries()
                eah = PhaseDiagram(formatted_entries).get_e_above_hull(bulk_computed_entry)
                warnings.warn(
                    f"Note that the Materials Project (MP) database entry for "
                    f"{self.composition.reduced_formula} is not stable with respect to competing "
                    f"phases, having an energy above hull of {eah:.4f} eV/atom.\n"
                    f"Formally, this means that the host material is unstable and so has no chemical "
                    f"potential limits; though in reality there may be errors in the MP energies (GGA, "
                    f"no vdW, SOC...), the host may be stabilised by temperature effects etc, or just a "
                    f"metastable phase.\n"
                    f"Here we downshift the host compound entry to the convex hull energy, "
                    f"and then determine the possible competing phases with the same approach as usual."
                )
                # decrease bulk_computed_entry energy per atom by ``energy_above_hull`` + 0.1 meV/atom
                name = description = (
                    "Manual energy adjustment to move the host composition to the MP convex hull"
                )
                bulk_computed_entry = _renormalise_entry(
                    bulk_computed_entry, eah + 1e-4, name=name, description=description
                )
                bulk_computed_entry.data["energy_above_hull"] = 0.0

            else:  # composition not on MP, warn and add shifted bulk entry to entries
                warnings.warn(
                    f"Note that no Materials Project (MP) database entry exists for "
                    f"{self.composition.reduced_formula}. Here we assume the host material has an "
                    f"energy equal to the MP convex hull energy at the corresponding point in chemical "
                    f"space, and then determine the possible competing phases with the same approach as "
                    f"usual."
                )
                self.MP_bulk_computed_entry = bulk_computed_entry = ComputedEntry(
                    self.composition,
                    self.MP_full_pd.get_hull_energy(self.composition) - 1e-4,
                    data={
                        "energy_above_hull": 0.0,
                        "material_id": "mp-0",
                        "molecule": False,
                        "summary": {
                            "band_gap": None,
                            "total_magnetization": None,
                            "database_IDs": {},
                        },
                    },
                )

            if self.MP_bulk_computed_entry not in formatted_entries:
                formatted_entries.append(self.MP_bulk_computed_entry)

        if self.bulk_structure:  # prune all bulk phases to this structure
            manual_bulk_entry = None

            if bulk_entries := [
                entry
                for entry in formatted_entries  # sorted by energy_above_hull in ``get_entries_in_chemsys``
                if entry.composition.reduced_composition == self.composition.reduced_composition
            ]:
                candidate_bulk_entries = [
                    (
                        entry,
                        StructureMatcher_scan_stol(
                            self.bulk_structure, entry.structure, func_name="get_rms_dist", max_stol=0.5
                        )
                        or float("inf"),
                    )
                    for entry in bulk_entries
                    if hasattr(entry, "structure")
                ]
                matching_bulk_entries = sorted(  # those with non-inf RMS (i.e. matching), sorted:
                    [entry for entry in candidate_bulk_entries if entry[1] != float("inf")],
                    key=lambda x: x[1],
                )
                if matching_bulk_entries:
                    matching_bulk_entry = matching_bulk_entries[0][0]
                    manual_bulk_entry = matching_bulk_entry
                    manual_bulk_entry._structure = self.bulk_structure

            if manual_bulk_entry is None:  # take the lowest energy bulk entry
                manual_bulk_entry_dict = self.MP_bulk_computed_entry.as_dict()
                manual_bulk_entry_dict["structure"] = self.bulk_structure.as_dict()
                manual_bulk_entry = ComputedStructureEntry.from_dict(manual_bulk_entry_dict)

            formatted_entries = [  # remove bulk entries from formatted_entries and add the new bulk entry
                entry
                for entry in formatted_entries
                if entry.composition.reduced_composition != self.composition.reduced_composition
            ]
            formatted_entries.append(manual_bulk_entry)

        if not self.full_phase_diagram:  # default, prune to only phases that would border the host
            # on the phase diagram, if their relative energy was downshifted by ``energy_above_hull``:
            self.entries: list[ComputedEntry] = prune_entries_to_border_candidates(
                entries=formatted_entries,
                bulk_computed_entry=bulk_computed_entry,
                energy_above_hull=self.energy_above_hull,
            )

        else:  # self.full_phase_diagram = True
            self.entries = formatted_entries

        if self.expected_polymorphs:
            self.entries = prune_to_expected_polymorphs(self.entries)

        # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
        self.entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=self.composition))
        _name_entries_and_handle_duplicates(self.entries)  # set entry names

        if not self.extrinsic:  # extrinsic path emits its own warning after extrinsic entries added
            _warn_if_many_polymorphs_per_composition(self.entries)

        if MP_doc_dicts:
            self.MP_doc_dicts = get_MP_summary_dicts(entries=self.entries, api_key=self.api_key)
        else:
            self.MP_doc_dicts = {}

        if not self.extrinsic:
            self.intrinsic_entries: list[ComputedEntry] = self.entries
            self.extrinsic_entries: list[ComputedEntry] = []
            self.MP_intrinsic_full_pd_entries = self.MP_full_pd_entries  # includes molecules-in-boxes
            return

        # otherwise, we have extrinsic species present:
        self.intrinsic_entries = copy.deepcopy(self.entries)
        self.extrinsic_elements = (
            [self.extrinsic] if isinstance(self.extrinsic, str) else list(self.extrinsic)
        )
        self.extrinsic_elements = [Element(el).symbol for el in self.extrinsic_elements]
        if extrinsic_in_intrinsic := [
            ext for ext in self.extrinsic_elements if ext in self.intrinsic_elements
        ]:
            raise ValueError(
                f"Extrinsic species {extrinsic_in_intrinsic} are already present in the host composition "
                f"({self.composition}), and so cannot be considered as extrinsic species!"
            )

        if self.codoping:
            self.single_extrinsic_phase_limits = False  # codoping implies no single-ext-phase restriction
            self.MP_full_pd_entries = get_entries_in_chemsys(  # can be time-consuming; high(er)-D chemsys
                chemsys=self.intrinsic_elements + self.extrinsic_elements,
                api_key=self.api_key,
                energy_above_hull=self.energy_above_hull,
                bulk_composition=self.composition.reduced_formula,  # for sorting
                **self._get_entries_kwargs,
            )
            self.entries = self._generate_elemental_diatomic_phases(self.MP_full_pd_entries)

            if not self.full_phase_diagram:
                self.entries = prune_entries_to_border_candidates(
                    entries=self.entries,
                    bulk_computed_entry=self.MP_bulk_computed_entry,
                    energy_above_hull=self.energy_above_hull,
                )  # prune using phase diagram with all extrinsic species

        else:  # build full set of candidate entries first (for ``self.entries``); not co-doping
            candidate_extrinsic_entries = []
            for ext_elt in self.extrinsic_elements:
                ext_elt_MP_full_pd_entries = get_entries_in_chemsys(
                    [*self.intrinsic_elements, ext_elt],
                    api_key=self.api_key,
                    energy_above_hull=self.energy_above_hull,
                    bulk_composition=self.composition.reduced_formula,  # for sorting
                    **self._get_entries_kwargs,
                )
                ext_elt_pd_entries = self._generate_elemental_diatomic_phases(ext_elt_MP_full_pd_entries)
                self.MP_full_pd_entries.extend(
                    [entry for entry in ext_elt_MP_full_pd_entries if entry not in self.MP_full_pd_entries]
                )

                if not self.full_phase_diagram:  # default, prune to only phases that would border the host
                    # material on the phase diagram, if their relative energy was downshifted by
                    # `energy_above_hull`; prune using phase diagrams for one extrinsic species at a time:
                    ext_elt_pd_entries = prune_entries_to_border_candidates(
                        entries=ext_elt_pd_entries,
                        bulk_computed_entry=self.MP_bulk_computed_entry,
                        energy_above_hull=self.energy_above_hull,
                    )

                candidate_extrinsic_entries += [
                    entry for entry in ext_elt_pd_entries if ext_elt in entry.composition
                ]

            if not self.single_extrinsic_phase_limits:
                self.entries += candidate_extrinsic_entries

            else:  # single_extrinsic_phase_limits: only retain extrinsic competing phases when they border
                # the host with no other extrinsic phases in equilibrium at the same limit
                for ext_elt in self.extrinsic_elements:
                    ext_elt_entries = [
                        entry for entry in candidate_extrinsic_entries if ext_elt in entry.composition
                    ]

                    if not ext_elt_entries:
                        raise ValueError(
                            f"No Materials Project entries found for the given chemical system: "
                            f"{[*self.intrinsic_elements, ext_elt]}"
                        )

                    ext_elt_phase_diagram = PhaseDiagram([*self.intrinsic_entries, *ext_elt_entries])
                    MP_extrinsic_gga_chempots = get_chempots_from_phase_diagram(
                        self.MP_bulk_computed_entry, ext_elt_phase_diagram
                    )
                    MP_extrinsic_bordering_phases: list[str] = []

                    for limit in MP_extrinsic_gga_chempots:
                        # note that the number of phases in equilibria at each vertex (limit) is equal
                        # to the number of elements in the chemical system (here being the host
                        # composition plus the extrinsic species)
                        extrinsic_bordering_phases = {
                            phase for phase in limit.split("-") if ext_elt in phase
                        }
                        if len(  # only add when <=1 extrinsic bordering phase
                            extrinsic_bordering_phases
                        ) == 1 and not extrinsic_bordering_phases.issubset(MP_extrinsic_bordering_phases):
                            MP_extrinsic_bordering_phases.extend(extrinsic_bordering_phases)

                    self.entries += [
                        entry
                        for entry in candidate_extrinsic_entries
                        if entry.name in MP_extrinsic_bordering_phases
                        or (entry.is_element and ext_elt in entry.name)
                    ]

        if self.expected_polymorphs:  # also prune extrinsic entries added above
            self.entries = prune_to_expected_polymorphs(self.entries)
            self.intrinsic_entries = prune_to_expected_polymorphs(self.intrinsic_entries)

        # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
        self.MP_full_pd_entries.sort(
            key=lambda x: _entries_sort_func(x, bulk_composition=self.composition.reduced_composition)
        )
        self.MP_full_pd = PhaseDiagram(self.MP_full_pd_entries)
        self.entries.sort(
            key=lambda x: _entries_sort_func(x, bulk_composition=self.composition.reduced_composition)
        )
        _name_entries_and_handle_duplicates(self.entries)  # set entry names
        self.extrinsic_entries = [entry for entry in self.entries if entry not in self.intrinsic_entries]
        assert len(self.intrinsic_entries) + len(self.extrinsic_entries) == len(self.entries), (
            "Error in extrinsic entries generation, please report this to the doped developers!"
        )

        _warn_if_many_polymorphs_per_composition(self.entries)

        if MP_doc_dicts:
            self.intrinsic_MP_doc_dicts = deepcopy(self.MP_doc_dicts)
            self.MP_doc_dicts = get_MP_summary_dicts(entries=self.entries, api_key=self.api_key)

    @property
    def metallic_entries(self) -> list[ComputedEntry]:
        """
        Returns a list of entries in ``self.entries`` which are metallic (i.e.
        have a band gap = 0) according to the Materials Project database.
        """
        return [entry for entry in self.entries if entry.data.get("summary", {}).get("band_gap") == 0]

    @property
    def nonmetallic_entries(self) -> list[ComputedEntry]:
        """
        Returns a list of entries in ``self.entries`` which are non-metallic
        (i.e. have a band gap > 0, or no recorded band gap) according to the
        Materials Project database.

        Note that the ``doped``-generated diatomic molecule phases, which are
        insulators, are not included here.
        """
        return [
            entry
            for entry in self.entries
            if (
                entry.data.get("summary", {}).get("band_gap") is None
                or entry.data.get("summary", {}).get("band_gap") > 0
            )
            and not entry.data.get("molecule")
        ]

    @property
    def molecular_entries(self) -> list[ComputedEntry]:
        """
        Returns a list of entries in ``self.entries`` which are diatomic
        molecules generated by ``doped`` (i.e. O2, N2, H2, F2, or Cl2).
        """
        return [entry for entry in self.entries if entry.data.get("molecule")]

    def _iter_entries_with_categories(self) -> Iterator[tuple[ComputedEntry, str, Structure, str]]:
        """
        Yield ``(entry, category, structure, folder_name)`` tuples for all
        entries in ``self.entries``, where ``category`` is one of ``"non-
        metals"``, ``"metals"`` or ``"molecules"``, and ``folder_name`` is the
        ``doped`` output folder name for the entry.

        When no structure exists in the entry (e.g. MP-missing bulk represented
        by a hull-energy |ComputedEntry|), emits a ``UserWarning`` and supplies
        a nominal large-cell ``Structure`` so that ``INCAR`` and ``POTCAR``
        files can still be written.
        """
        categorised_entries = [
            (self.nonmetallic_entries, "non-metals"),
            (self.metallic_entries, "metals"),
            (self.molecular_entries, "molecules"),
        ]
        for entry_list, category in categorised_entries:
            for entry in entry_list:
                if hasattr(entry, "structure"):
                    structure = entry.structure
                else:
                    warnings.warn(
                        f"No structure is available for '{entry.name}'. This is likely because the phase "
                        f"is a placeholder (e.g. a hull-energy estimate for a composition with no "
                        f"Materials Project crystal structure). Structure-independent calculation inputs "
                        f"(e.g. INCAR and POTCAR files with VASP) will be written assuming a "
                        f"non-metallic, non-magnetic material (when determining smearing / magnetic "
                        f"moment input settings), and element ordering according to the printed "
                        f"composition. One should of course check that these choices are appropriate!"
                    )
                    structure = _nominal_structure_for_input_writing(entry.composition)
                yield entry, category, structure, _get_competing_phase_folder_name(entry)

    @property
    def _input_backend(self) -> ModuleType:
        """
        The ``doped.io`` input-generation backend module
        (``doped.io.<calculator>.inputs``) for the calculator being used.
        """
        return get_backend(getattr(self, "calculator", "vasp"), module="inputs")

    def get_kpoint_convergence_sets(self, **kwargs) -> dict[str, Any]:
        r"""
        Generate calculation input sets for `k`-point convergence testing of
        the competing phases, using PBEsol (GGA) DFT by default.

        Dispatches to the ``get_kpoint_convergence_sets()`` function of the
        ``doped.io`` input-generation backend for ``self.calculator`` -- see
        :func:`doped.io.vasp.inputs.get_kpoint_convergence_sets` for the
        default (VASP) behaviour and full list of options.

        Args:
            **kwargs:
                Keyword arguments for the calculator backend\'s
                ``get_kpoint_convergence_sets()`` function; e.g.
                ``kpoints_metals``, ``kpoints_nonmetals``, ``extrinsic_only``
                and ``output_path``, plus calculator-specific settings such
                as ``user_incar_settings`` / ``user_potcar_settings`` with
                VASP.

        Returns:
            dict[str, Any]:
                Mapping of output folder paths to generated calculation
                input sets.
        """
        return self._input_backend.get_kpoint_convergence_sets(self, **kwargs)

    def write_kpoint_convergence_files(self, **kwargs) -> dict[str, Any]:
        r"""
        Generate and write calculation input files for `k`-point convergence
        testing of the competing phases.

        Dispatches to the ``write_kpoint_convergence_files()`` function of
        the ``doped.io`` input-generation backend for ``self.calculator`` --
        see :func:`doped.io.vasp.inputs.write_kpoint_convergence_files` for
        the default (VASP) behaviour and full list of options.

        Args:
            **kwargs:
                Keyword arguments for the calculator backend\'s
                ``write_kpoint_convergence_files()`` function (see
                :meth:`get_kpoint_convergence_sets`, plus any file-writing
                options such as ``poscar`` / ``potcar_spec`` with VASP).

        Returns:
            dict[str, Any]:
                Mapping of output folder paths to written calculation input
                sets.
        """
        return self._input_backend.write_kpoint_convergence_files(self, **kwargs)

    def convergence_setup(self, **kwargs) -> dict[str, Any]:
        r"""
        Deprecated alias for :meth:`write_kpoint_convergence_files`.

        .. deprecated:: 4.0
            Use :meth:`write_kpoint_convergence_files` instead; this name will
            be removed in v4.1.
        """
        warnings.warn(
            "`CompetingPhases.convergence_setup` is deprecated and will be removed in the next "
            "minor release (v4.1); use `CompetingPhases.write_kpoint_convergence_files` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.write_kpoint_convergence_files(**kwargs)

    def get_relaxation_sets(self, **kwargs) -> dict[str, Any]:
        r"""
        Generate calculation input sets for geometry relaxations of the
        competing phases, using HSE06 by default.

        Dispatches to the ``get_relaxation_sets()`` function of the
        ``doped.io`` input-generation backend for ``self.calculator`` -- see
        :func:`doped.io.vasp.inputs.get_relaxation_sets` for the default
        (VASP) behaviour and full list of options.

        Args:
            **kwargs:
                Keyword arguments for the calculator backend\'s
                ``get_relaxation_sets()`` function; e.g. ``kpoints_metals``,
                ``kpoints_nonmetals``, ``extrinsic_only``, ``output_path``
                and ``subfolder``, plus calculator-specific settings such as
                ``user_incar_settings`` / ``user_potcar_settings`` with VASP.

        Returns:
            dict[str, Any]:
                Mapping of output folder paths to generated calculation
                input sets.
        """
        return self._input_backend.get_relaxation_sets(self, **kwargs)

    def write_relaxation_files(self, **kwargs) -> dict[str, Any]:
        r"""
        Generate and write calculation input files for geometry relaxations of
        the competing phases.

        Dispatches to the ``write_relaxation_files()`` function of the
        ``doped.io`` input-generation backend for ``self.calculator`` -- see
        :func:`doped.io.vasp.inputs.write_relaxation_files` for the default
        (VASP) behaviour and full list of options.

        Args:
            **kwargs:
                Keyword arguments for the calculator backend\'s
                ``write_relaxation_files()`` function (see
                :meth:`get_relaxation_sets`, plus any file-writing options
                such as ``poscar`` / ``potcar_spec`` with VASP).

        Returns:
            dict[str, Any]:
                Mapping of output folder paths to written calculation input
                sets.
        """
        return self._input_backend.write_relaxation_files(self, **kwargs)

    def vasp_std_setup(self, **kwargs) -> dict[str, Any]:
        r"""
        Deprecated alias for :meth:`write_relaxation_files`.

        .. deprecated:: 4.0
            Use :meth:`write_relaxation_files` instead; this name will be
            removed in v4.1.
        """
        warnings.warn(
            "`CompetingPhases.vasp_std_setup` is deprecated and will be removed in the next minor "
            "release (v4.1); use `CompetingPhases.write_relaxation_files` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.write_relaxation_files(**kwargs)

    def get_singlepoint_sets(self, **kwargs) -> dict[str, Any]:
        r"""
        Generate calculation input sets for final single-point (static) energy
        calculations of the competing phases.

        Dispatches to the ``get_singlepoint_sets()`` function of the
        ``doped.io`` input-generation backend for ``self.calculator`` -- see
        :func:`doped.io.vasp.inputs.get_singlepoint_sets` for the default
        (VASP) behaviour and full list of options.

        Args:
            **kwargs:
                Keyword arguments for the calculator backend\'s
                ``get_singlepoint_sets()`` function; e.g. ``soc``,
                ``kpoints_metals``, ``kpoints_nonmetals``,
                ``extrinsic_only``, ``output_path`` and ``subfolder``, plus
                calculator-specific settings such as ``user_incar_settings``
                / ``user_potcar_settings`` with VASP.

        Returns:
            dict[str, Any]:
                Mapping of output folder paths to generated calculation
                input sets.
        """
        return self._input_backend.get_singlepoint_sets(self, **kwargs)

    def write_singlepoint_files(self, **kwargs) -> dict[str, Any]:
        r"""
        Generate and write calculation input files for final single-point
        (static) energy calculations of the competing phases.

        Dispatches to the ``write_singlepoint_files()`` function of the
        ``doped.io`` input-generation backend for ``self.calculator`` -- see
        :func:`doped.io.vasp.inputs.write_singlepoint_files` for the default
        (VASP) behaviour and full list of options.

        Args:
            **kwargs:
                Keyword arguments for the calculator backend\'s
                ``write_singlepoint_files()`` function (see
                :meth:`get_singlepoint_sets`, plus any file-writing options
                such as ``poscar`` / ``potcar_spec`` with VASP).

        Returns:
            dict[str, Any]:
                Mapping of output folder paths to written calculation input
                sets.
        """
        return self._input_backend.write_singlepoint_files(self, **kwargs)

    def _generate_elemental_diatomic_phases(self, entries: list[ComputedEntry]) -> list[ComputedEntry]:
        r"""
        Given an input list of |ComputedEntry| objects, adds a
        |ComputedStructureEntry| for each diatomic elemental phase (O2, N2, H2,
        F2, Cl2) to ``entries`` using ``make_molecular_entry``, and generates
        an output list of |ComputedEntry| / |ComputedStructureEntry|\s
        containing all entries in ``entries``, with `all` elemental phases of
        O, N, H, F, Cl replaced by the single corresponding molecule-in-a-box
        entry.

        Also sets the ``ComputedEntry.data["molecule"]`` flag for each entry in
        ``entries`` (``True`` for diatomic gases, ``False`` for all others).

        The output entries list is sorted by energy above hull, then by the
        number of elements in the formula, then by the position of elements in
        the periodic table (main group elements, then transition metals, sorted
        by row).

        Args:
            entries (list[|ComputedEntry|]):
                List of |ComputedEntry|/|ComputedStructureEntry| objects
                for the input chemical system.

        Returns:
            list[|ComputedEntry|]:
                List of |ComputedEntry|/|ComputedStructureEntry| objects
                for the input chemical system, with diatomic elemental phases
                replaced by the single molecule-in-a-box entry.
        """
        formatted_entries: list[ComputedEntry] = []

        for entry in entries.copy():
            if entry.is_element and str(next(iter(entry.elements))) in elemental_diatomic_bond_lengths:
                if entry.data.get("energy_above_hull", 0) == 0.0 and not any(
                    other_entry.data["molecule"] and other_entry.elements == entry.elements
                    for other_entry in formatted_entries
                ):  # only once for each matching gaseous elemental entry
                    molecular_entry = make_molecular_entry(entry)
                    entries.append(molecular_entry)
                    formatted_entries.append(molecular_entry)
            else:
                entry.data["molecule"] = False
                formatted_entries.append(entry)

        # sort by energy above hull, num_species, then by periodic table positioning:
        formatted_entries.sort(key=_entries_sort_func)

        return formatted_entries

    @property
    def entries_dict(self) -> dict[str, ComputedEntry]:
        """
        Mapping of ``doped`` competing phase names to entries.
        """
        entries_dict: dict[str, ComputedEntry] = {}
        for entry in self.entries:
            doped_name = get_and_set_competing_phase_name(entry, regenerate=False)
            if doped_name in entries_dict:
                raise KeyError(
                    f"Duplicate competing phase key encountered in `self.entries`: {doped_name}. "
                    "Please regenerate entries / entry names to ensure uniqueness."
                )
            entries_dict[doped_name] = entry
        return entries_dict

    def __getattr__(self, attr: str) -> Any:
        """
        Redirect unknown attribute/method lookups to the entries dictionary.
        """
        # ``__getattr__`` is only called when normal lookup has already failed; ``entries`` is
        # accessed by ``entries_dict`` so guard against infinite recursion during partially-
        # initialised states:
        if attr == "entries":
            raise AttributeError(attr)
        return getattr(self.entries_dict, attr)

    @overload
    def __getitem__(self, key: str | int) -> ComputedEntry | ComputedStructureEntry: ...

    @overload
    def __getitem__(self, key: slice) -> list[ComputedEntry | ComputedStructureEntry]: ...

    def __getitem__(self, key: str | int | slice) -> ComputedEntry | list[ComputedEntry]:
        """
        Make the object subscriptable.

        String keys index by ``entry.data["doped_name"]`` (dict-like), while
        integer / slice keys use list-style indexing on ``self.entries``.
        """
        if isinstance(key, str):
            return self.entries_dict[key]
        return self.entries[key]

    def __contains__(self, item: str | ComputedEntry | ComputedStructureEntry) -> bool:
        """
        Return ``True`` if ``item`` is in the entries.

        For string inputs this checks ``entry.data["doped_name"]`` keys, while
        for non-strings this falls back to list-style membership in
        ``self.entries``.
        """
        if isinstance(item, str):
            return item in self.entries_dict
        return item in self.entries

    def __len__(self) -> int:
        """
        Return the number of competing phase entries.
        """
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over ``entry.data["doped_name"]`` keys.
        """
        return iter(self.entries_dict)

    def __repr__(self) -> str:
        """
        Returns a string representation of the |CompetingPhases| object.
        """
        formula = self.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        joined_entry_list = "\n".join([entry.data["doped_name"] for entry in self.entries])
        return (
            f"doped CompetingPhases for bulk composition {formula} with {len(self.entries)} entries "
            f"(in self.entries):\n{joined_entry_list}\n\n"
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )

    def as_dict(self) -> dict:
        """
        Returns:
            JSON-serializable dict representation of |CompetingPhases|.
        """
        cp_dict = self.__dict__
        if isinstance(cp_dict.get("extrinsic"), set):
            cp_dict["extrinsic"] = sorted(cp_dict["extrinsic"])
        return {"@module": type(self).__module__, "@class": type(self).__name__, **cp_dict}

    @classmethod
    def from_dict(cls, d: dict) -> "CompetingPhases":
        """
        Reconstitute a |CompetingPhases| object from a dict representation
        created using ``as_dict()``.

        Args:
            d (dict): dict representation of |CompetingPhases|.

        Returns:
            |CompetingPhases| object
        """
        competing_phases = cls.__new__(cls)  # skip __init__ and avoid MP re-querying
        d = dict(d)  # copy, avoiding mutation of the caller's dict
        if "full_sub_approach" in d:  # TODO: remove ``full_sub_approach`` translation in v4.1
            warnings.warn(
                "Loading a `CompetingPhases` saved with `full_sub_approach`; this attribute was renamed "
                "to `single_extrinsic_phase_limits` (with inverted polarity) in doped v4.0. Translating "
                "on load — re-save to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            d["single_extrinsic_phase_limits"] = not d.pop("full_sub_approach", True)
        if "extrinsic_species" in d:  # TODO: remove ``extrinsic_species`` translation in v4.1
            warnings.warn(
                "Loading a `CompetingPhases` saved with `extrinsic_species`; this attribute was renamed "
                "to `extrinsic_elements` in doped v4.0 (for consistency with "
                "`CompetingPhasesAnalyzer.extrinsic_elements`). Translating on load — re-save to silence "
                "this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            d["extrinsic_elements"] = d.pop("extrinsic_species", [])
        for key, value in d.items():
            if key not in {"@module", "@class", "@version"}:
                setattr(competing_phases, key, MontyDecoder().process_decoded(value))
        return competing_phases


def get_doped_chempots_from_entries(
    entries: Sequence[ComputedEntry | ComputedStructureEntry | PDEntry],
    composition: str | Composition | ComputedEntry,
    single_chempot_limit: bool = False,
) -> dict:
    r"""
    Given a list of |ComputedEntry|\s / |ComputedStructureEntry|\s /
    ``PDEntry``\s and the bulk ``composition``, returns the chemical potential
    limits dictionary in the ``doped`` format (i.e. ``{"limits": [{'limit':
    [chempot_dict]}], ...}``) for the host material.

    Args:
        entries (list[|ComputedEntry|]):
            List of |ComputedEntry|\s / |ComputedStructureEntry|\s /
            ``PDEntry``\s for the chemical system, from which to determine
            the chemical potential limits for the host material
            (``composition``).
        composition (str, |Composition|, |ComputedEntry|):
            Composition of the host material either as a string
            (e.g. 'LiFePO4') a ``pymatgen`` |Composition| object (e.g.
            ``Composition('LiFePO4')``), or a |ComputedEntry| object.
        single_chempot_limit (bool):
            If set to ``True``, only returns the first chemical potential limit
            in the calculated chemical potentials dictionary. Mainly intended
            for internal ``doped`` usage when the host material is calculated
            to be unstable with respect to the competing phases.

    Returns:
        dict:
            Dictionary of chemical potential limits in the ``doped`` format.
    """
    if isinstance(composition, str | Composition):
        composition = Composition(composition)
    else:
        composition = composition.composition

    phase_diagram = PhaseDiagram(entries)
    chem_lims = phase_diagram.get_all_chempots(composition.reduced_composition)
    chem_lims_iterator = list(chem_lims.items())[:1] if single_chempot_limit else chem_lims.items()

    chempots = {  # convert ``Element``\s to ``str`` to make dict ``JSONable``:
        "limits": {k: {str(kk): vv for kk, vv in v.items()} for k, v in chem_lims_iterator},
        "elemental_refs": {str(el): ent.energy_per_atom for el, ent in phase_diagram.el_refs.items()},
    }
    chempots["limits_wrt_el_refs"] = {  # relate the limits to the elemental energies:
        limit: {el: mu - chempots["elemental_refs"][el] for el, mu in chempot_dict.items()}
        for limit, chempot_dict in chempots["limits"].items()
    }

    # round all floats to 4 decimal places (0.1 meV/atom) for cleanliness (well below DFT accuracy):
    return _round_floats(chempots, 4)


class ChemicalPotentialGrid(MSONable):
    """
    A class to represent a grid in chemical potential space and to perform
    operations such as generating a grid within the convex hull of given
    vertices (chemical potential limits).

    This class provides methods for handling and manipulating chemical
    potential data, including the creation of a grid that spans a specified
    chemical potential space.
    """

    def __init__(self, chempots: dict[str, Any], format_chempot_labels: bool = True):
        r"""
        Initializes the |ChemicalPotentialGrid| with chemical potential data.

        This constructor takes a dictionary of chemical potentials and sets up
        the initial vertices of the grid.

        Args:
            chempots (dict):
                Dictionary of chemical potentials for the grid. This can have
                the form of ``{"limits": [{'limit': [chempot_dict]}], ...}``
                (the format generated by ``doped``\'s chemical potential
                parsing functions), or alternatively can be a dictionary of the
                form ``{'limit': [chempot_dict, ...]}`` (i.e. matching the
                format of ``chempots["limits_wrt_el_refs"]`` from the ``doped``
                ``chempots`` dict) where the keys are the limit names (e.g.
                "Cd-CdTe", "Cd-rich" etc) and the values are dictionaries of a
                single chemical potential limit in the format:
                ``{element symbol: chemical potential}``.

                If ``chempots`` in the ``doped`` format is supplied, then the
                chemical potentials `with respect to the elemental reference
                energies` will be used (i.e.
                ``chempots["limits_wrt_el_refs"]``)!
            format_chempot_labels (bool):
                Whether to format the elemental chemical potential labels as
                "µ_{elt} (eV)" (default) or just "{elt}" (if ``False``).
        """
        chempots_dict = chempots.get("limits_wrt_el_refs", chempots)  # unformatted chempots dict
        if format_chempot_labels:
            chempots_dict = {
                limit: {
                    f"μ_{k} (eV)" if Element.is_valid_symbol(k) else k: v
                    for (k, v) in unformatted_chempots_subdict.items()
                }
                for limit, unformatted_chempots_subdict in chempots_dict.items()
            }

        self.vertices = pd.DataFrame.from_dict(chempots_dict, orient="index")

    @classmethod
    def from_dataframe(cls, vertices: pd.DataFrame) -> "ChemicalPotentialGrid":
        """
        Create a |ChemicalPotentialGrid| object from a dataframe of the
        chemical potential limits (i.e. vertices).

        Args:
            vertices (pd.DataFrame):
                A ``DataFrame`` of the chemical potential limits (i.e. vertices).
                The columns should be the chemical potentials of the elements,
                and the rows should be the limits.

        Returns:
            ChemicalPotentialGrid:
                A |ChemicalPotentialGrid| object.
        """
        self = cls.__new__(cls)  # skip init function
        self.vertices = vertices
        return self

    def as_dict(self) -> dict:
        """
        Returns:
            JSON-serializable dict representation of |ChemicalPotentialGrid|.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "vertices": {
                "index": self.vertices.index.tolist(),
                "columns": self.vertices.columns.tolist(),
                "data": self.vertices.to_numpy().tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChemicalPotentialGrid":
        """
        Reconstitute a |ChemicalPotentialGrid| object from a dict
        representation created using ``as_dict()``.

        Args:
            d (dict): dict representation of |ChemicalPotentialGrid|.

        Returns:
            |ChemicalPotentialGrid| object
        """
        vertices = pd.DataFrame(
            d["vertices"]["data"],
            index=d["vertices"]["index"],
            columns=d["vertices"]["columns"],
        )
        return cls.from_dataframe(vertices)

    def get_grid(
        self,
        n_points: int = 1000,
        fixed_elements: dict[str, float] | None = None,
        cartesian: bool = False,
        decimal_places: int = 4,
        drop_duplicates: bool = True,
        include_vertices: bool = True,
    ) -> pd.DataFrame:
        r"""
        Generates a grid of points that spans the chemical potential space
        defined by the vertices. It ensures that the generated points lie
        within the convex hull of the provided vertices and interpolates the
        chemical potential values at these points.

        By default, the grid is generated in barycentric coordinates (i.e. in
        'relative' coordinates, as weighted averages of the chemical potential
        limits). This is far more efficient than generating a Cartesian grid,
        but means that the grid is evenly spaced in barycentric ('relative')
        coordinates, and not necessarily in Cartesian coordinates. If
        ``cartesian`` is set to ``True``, then a regular grid in Cartesian
        coordinates is generated (however this can be much slower).

        Args:
            n_points (int | None):
                `Minimum` number of grid points to generate. The output grid
                will contain at least this many points (regularly spaced in
                barycentric or Cartesian space depending ``cartesian``), in
                addition to the vertices themselves (if ``include_vertices`` is
                ``True``; default), then with any duplicate / overlapping
                points dropped. Default is 1000. Note that large values
                (>= 1e5) with multinary systems can explode, crashing system
                memory (especially with ``cartesian=True``).
            fixed_elements (dict | None):
                A dictionary of chemical potentials to fix (in the format:
                ``{column_name: value}``; e.g. ``{"Li": -2}``), if a reduced /
                constrained chemical potential grid is desired. If provided,
                the ``get_constrained_grid`` method is used.
            cartesian (bool):
                Whether to generate the grid in Cartesian coordinates. If
                ``False`` (default), the grid is generated in barycentric
                coordinates, which is far more efficient, but means that the
                grid is evenly spaced in barycentric ('relative') coordinates,
                and not necessarily in Cartesian coordinates.
            decimal_places (int):
                The number of decimal places to round the grid coordinates to.
                Note that the tolerance for grid-point matching is
                10^[-decimal_places]. Default is 4.
            drop_duplicates (bool):
                Whether to drop duplicate points in the generated grid. With
                barycentric coordinate generation, there can be duplicate
                points in the generated grid from overlapping simplices. If
                duplicates are acceptable (likely true for most downstream
                usages; e.g. plotting etc) then this can be set to ``False`` to
                speed up runtime. Default is ``True``.
            include_vertices (bool):
                Whether to include the vertices themselves in the generated
                grid. Default is ``True``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the points within the convex hull.
                Each row represents a point in the grid.
        """
        if fixed_elements:
            return self.get_constrained_grid(
                fixed_elements, n_points, cartesian, decimal_places, drop_duplicates, include_vertices
            )

        dependent_variable = self.vertices.columns[-1]
        dependent_var = self.vertices[dependent_variable].to_numpy()
        independent_vars = self.vertices.drop(columns=dependent_variable)

        n_dims = independent_vars.shape[1]  # number of independent variables (dimensions)
        if n_dims < 2:
            raise ValueError(
                "Chemical potential grid generation is only possible for systems with "
                "two or more independent variables (chemical potentials), i.e. ternary or "
                "higher-dimensional systems. Stable chemical potential ranges are just a line for binary "
                "systems, for which ``FermiSolver.interpolate_chempots()`` can be used."
            )

        hull = ConvexHull(independent_vars.to_numpy())  # convex hull of the vertices
        # ensure vertices and dependent values are aligned:
        hull_idx = hull.vertices  # indices of the hull points
        coords_hull = independent_vars.to_numpy()[hull_idx]
        values_hull = dependent_var[hull_idx]

        if cartesian:  # Create a dense grid that covers the entire range of the vertices
            # hull volume (in N-D) times grid density = num points:
            req_grid_density = n_points / hull.volume  # points per N-D volume
            req_density_per_dim = req_grid_density ** (1 / n_dims)  # points per dim, inverse system units
            grid_ranges = [
                np.arange(
                    independent_vars.iloc[:, i].min(),
                    independent_vars.iloc[:, i].max(),
                    1 / req_density_per_dim,  # step size in system units = 1/points-per-dimension
                )
                for i in range(n_dims)
            ]
            grid = np.meshgrid(*grid_ranges, indexing="ij")  # Create N-dimensional grid
            grid_points = np.vstack([g.ravel() for g in grid]).T  # Flatten the grid to points

            # Delaunay triangulation to get points inside the convex hull
            grid_with_values = _griddata_linear_in_hull(
                coords_hull, values_hull, grid_points, tol=10 ** (-decimal_places)
            )

        else:  # efficiently generate a grid of points inside the convex hull, using barycentric coords:
            grid_with_values = _lattice_in_hull(coords_hull, values_hull, n_points=n_points)

        if include_vertices:  # Ensure vertices are in the grid
            grid_with_values = np.vstack((grid_with_values, self.vertices.to_numpy()))

        grid_df = pd.DataFrame(
            grid_with_values,
            columns=[*list(independent_vars.columns), dependent_variable],
        ).round(decimal_places)

        return grid_df if not drop_duplicates else grid_df.drop_duplicates()

    def get_constrained_grid(
        self,
        fixed_elements: dict[str, float],
        n_points: int = 1000,
        cartesian: bool = False,
        decimal_places: int = 4,
        drop_duplicates: bool = True,
        include_vertices: bool = True,
    ) -> pd.DataFrame:
        r"""
        Generates a grid of points that spans the chemical potential space
        defined by the vertices, constrained by a set of fixed chemical
        potentials.

        By default, the grid is generated in barycentric coordinates (i.e. in
        'relative' coordinates, as weighted averages of the chemical potential
        limits). This is far more efficient than generating a Cartesian grid,
        but means that the grid is evenly spaced in barycentric ('relative')
        coordinates, and not necessarily in Cartesian coordinates. If
        ``cartesian`` is set to ``True``, then a regular grid in Cartesian
        coordinates is generated (however this can be much slower).

        Args:
            fixed_elements (dict):
                A dictionary of chemical potentials to fix (in the format:
                ``{column_name: value}``; e.g. ``{"Li": -2}``).
            n_points (int | None):
                `Minimum` number of grid points to generate, within the
                constrained subspace. The output grid will contain at least
                this many points (regularly spaced in barycentric or Cartesian
                space depending ``cartesian``), in addition to the vertices of
                the constrained subspace (if ``include_vertices`` is ``True``;
                default), then with any duplicate / overlapping points dropped.
                Default is 1000. Note that large values (>= 1e5) with
                multinary systems can explode, crashing system memory
                (especially with ``cartesian=True``).
            cartesian (bool):
                Whether to generate the grid in Cartesian coordinates. If
                ``False`` (default), the grid is generated in barycentric
                coordinates, which is far more efficient, but means that the
                grid is evenly spaced in barycentric ('relative') coordinates,
                and not necessarily in Cartesian coordinates.
            decimal_places (int):
                The number of decimal places to round the grid coordinates to.
                Default is 4.
            drop_duplicates (bool):
                Whether to drop duplicate points in the generated grid. With
                barycentric coordinate generation, there can be duplicate
                points in the generated grid from overlapping simplices. If
                duplicates are acceptable (likely true for most downstream
                usages; e.g. plotting etc) then this can be set to ``False`` to
                speed up runtime. Default is ``True``.
            include_vertices (bool):
                Whether to include the vertices themselves in the generated
                grid. Default is ``True``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the points within the convex hull,
                constrained by the fixed chemical potentials. Each row
                represents a point in the grid.
        """
        fixed_elements = {
            k if k in self.vertices.columns else f"μ_{k} (eV)": v for k, v in fixed_elements.items()
        }
        variables = [col for col in self.vertices.columns if col not in fixed_elements]
        dependent_variable = variables[-1]
        dependent_var = self.vertices[dependent_variable].to_numpy()
        independent_vars = self.vertices.drop(columns=dependent_variable)
        vertices = independent_vars.to_numpy()

        for element, value in fixed_elements.items():
            try:
                vertices = _intersect_hull_with_plane(
                    vertices,
                    list(independent_vars.columns).index(element),
                    value,
                    tol=10 ** (-decimal_places),
                )
            except ValueError as e:
                if "does not meet the hull" in str(e):
                    raise ValueError(
                        f"The input set of fixed chemical potentials does not intersect with the convex "
                        f"hull (i.e. stable chemical potential range) of the host material. Convex hull "
                        f"vertices:\n{self.vertices}\n"
                    ) from e

                raise e

        # Interpolate the values to get the dependent chemical potential
        grid_with_values = _griddata_linear_in_hull(
            independent_vars.to_numpy(), dependent_var, vertices, tol=10 ** (-decimal_places)
        )

        constrained_vertices = pd.DataFrame(
            grid_with_values,
            columns=[*list(independent_vars.columns), dependent_variable],
        )
        # these are our new constrained vertices, now we generate the grid (without the fixed element):
        input_constrained_vertices = constrained_vertices.drop(columns=list(fixed_elements.keys()))
        constrained_grid = ChemicalPotentialGrid.from_dataframe(input_constrained_vertices)
        grid_df = constrained_grid.get_grid(
            n_points=n_points,
            cartesian=cartesian,
            decimal_places=decimal_places,
            drop_duplicates=drop_duplicates,
            include_vertices=include_vertices,
        ).dropna()

        for element_col_name, value in fixed_elements.items():  # add fixed-element values to the grid
            grid_df[element_col_name] = [value] * len(grid_df)

        return grid_df


def _intersect_hull_with_plane(
    vertices: np.ndarray,
    axis: int,
    value: float,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Intersect the convex hull with a plane defined by a fixed value of a given
    axis.

    Args:
        vertices (np.ndarray):
            Coordinates of the convex-hull vertices.
        axis (int):
            Index of the coordinate you want to fix (0-based).
        value (float):
            The plane, such that ``x[axis] = value``.
        tol (float):
            Numerical tolerance for deciding if points lie on the plane.

    Returns:
        np.ndarray:
            Coordinates of intersections between the convex hull and the plane.
    """
    axis_min, axis_max = vertices[:, axis].min(), vertices[:, axis].max()
    if value < axis_min - tol or value > axis_max + tol:
        raise ValueError(f"The plane {axis} = {value} does not meet the hull.")

    intersection_points = []
    # for each edge, check if it crosses the plane:
    for i, j in itertools.combinations(range(len(vertices)), 2):
        vertex_i, vertex_j = vertices[i], vertices[j]
        signed_dist_i = vertex_i[axis] - value  # signed distance from plane along ``axis``
        signed_dist_j = vertex_j[axis] - value

        if signed_dist_i * signed_dist_j < 0:  # opposite signs -> edge crosses the plane
            # fraction along the edge at which the crossing occurs (0 < frac < 1):
            frac = signed_dist_i / (signed_dist_i - signed_dist_j)
            intersection_points.append(vertex_i + frac * (vertex_j - vertex_i))
        else:  # edge does not cross; include any endpoint that lies on the plane:
            for signed_dist, vertex in ((signed_dist_i, vertex_i), (signed_dist_j, vertex_j)):
                if abs(signed_dist) <= tol:
                    intersection_points.append(vertex)

    return np.asarray(intersection_points)


def _lattice_in_hull(
    vertices: np.ndarray,
    Y: np.ndarray | None = None,
    n_points: int = 1000,
    qhull_options: str = "QJ Qbb Qc",
) -> np.ndarray:
    """
    Generate a grid of points inside the convex hull of the given vertices,
    using barycentric coordinates (i.e. weighted averages of the vertices).

    This is a far more efficient way to generate a grid of points inside the
    convex hull than the brute-force approach of generating a Cartesian grid of
    points in the range of the vertices and then filtering out points outside
    the convex hull, but means that the grid is evenly spaced in barycentric
    ('relative') coordinates, but not necessarily in Cartesian coordinates.
    If greater precision is needed in output predictions, one can just scale
    ``n_points`` until convergence.

    Args:
        vertices (np.ndarray):
            (n, k) float array of data points to interpolate between.
        Y (np.ndarray):
            (n,) float array of values at ``vertices``. This should be the
            values of the dependent variable (e.g. chemical potential) at the
            given vertices. If provided, the function will also interpolate the
            dependent variable values at the generated points inside the convex
            hull. Default is ``None``.
        n_points (int):
            `Minimum` number of grid points to generate. The output grid will
            contain at least this many points, regularly spaced in barycentric
            space. Default is 1000.
        qhull_options (str):
            Options to pass to ``QHull`` via ``~scipy.spatial.Delaunay``.
            Default is "QJ Qbb Qc", where "QJ" means joggled input to avoid
            precision problems, "Qbb" scales coordinates for better
            conditioning, and "Qc" keeps coplanar points.

    Returns:
        np.ndarray:
            A grid of points inside the convex hull, in Cartesian coordinates.
            The shape of the array is (M, k), where M is the number of points
            in the grid.
    """
    if vertices.ndim != 2:
        raise ValueError("`vertices` must be a 2-D array (N_points, N_dimensions)")

    k = vertices.shape[-1]  # dimensionality (k ≥ 2)
    # vertices defines the polytope (k-D polyhedron) of the convex hull; shape (N, k)
    # we then tessellate the hull with Delaunay triangulation, which breaks the polytope into a set of
    # k-D simplices (e.g. triangles in 2D, tetrahedra in 3D; simplest possible polytope in k-D space),
    # which each have k+1 vertices (e.g. 3 vertices for triangles, 4 vertices for tetrahedra, etc)
    if vertices.shape[0] == k + 1:  # Input is already a single simplex; no triangulation needed
        simplices = np.array([np.arange(k + 1)])
    else:  # setup Delaunay triangulation
        simplices = Delaunay(vertices, qhull_options=qhull_options).simplices

    # generate a grid of barycentric coordinates (i.e. weighted averages of the vertices) which are inside
    # the convex hull; for this the total weight should sum to 1, so generate tuples (n0,..,nk) with
    # sum = ``n_points_per_dim``, and then divide by ``n_points_per_dim``:
    def _compositions(r: int, k: int) -> Iterator[tuple[int, ...]]:
        """
        Yield all tuples (n0,...,nk) such that  ∑ n_i = r and n_i ≥ 0.

        Stars-and-bars: choose k cut positions in r+k slots.
        """
        # choose k cut indices; add sentinels at -1 and r+k
        for cuts in itertools.combinations(range(r + k), k):
            prev = -1
            parts = []
            for c in (*cuts, r + k):
                parts.append(c - prev - 1)
                prev = c
            yield tuple(parts)  # (L, k+1); where L depends on binomial(n_points_per_dim + k, k)

    # Note: In theory one could skip this loop and directly predict the required number of points along
    # each simplex dimension, with some fitting of scaling, but the individual computation is very fast for
    # reasonable to large ``n_points``, so shouldn't be an issue in practice.
    n_points_per_dim = 0
    unscaled_bary_coords: list[tuple[int, ...]] = []
    while len(unscaled_bary_coords) * len(simplices) < n_points:
        n_points_per_dim += 1
        unscaled_bary_coords = list(_compositions(n_points_per_dim, k))
        if n_points_per_dim > max(n_points, 1):  # should never happen
            raise RuntimeError(
                "Barycentric coordinate generation failed! Please check your inputs, and report this "
                "issue to the developers if they are reasonable."
            )

    bary_coords = np.array(unscaled_bary_coords) / n_points_per_dim  # (L, k+1); L >= n_points

    # Note: If one really wanted a regular(ish) grid spacing in Cartesian (i.e. energy) coordinates,
    # the barycentric coordinate grid spacing could be scaled by the Euclidean distance between the simplex
    # vertices (for each simplex), but this should not be necessary/important for most use cases (if
    # more precision is needed in output predictions, can just scale ``n_points`` as needed). Can always
    # be implemented if needed

    verts_per_simplex = vertices[simplices]  # shape (S, k+1, k); where S is the number of simplices

    # Vectorised Cartesian coordinates (and interpolated values if dependent_var is provided):
    # points_inside: (S, L, k) -> reshape -> (S*L, k)
    points_inside = np.einsum("LK,SKk->SLk", bary_coords, verts_per_simplex).reshape(-1, k)  # K = k+1
    if Y is None:
        return points_inside

    vals_per_simplex = Y[simplices]  # (S, k+1)
    # values_inside: (S, L) -> reshape -> (S*L,)
    Y_inside = np.einsum("LK,SK->SL", bary_coords, vals_per_simplex).ravel()
    return np.hstack((points_inside, Y_inside.reshape(-1, 1)))


def _griddata_linear_in_hull(
    X: np.ndarray, Y: np.ndarray, xi: np.ndarray, tol: float = 1e-6, qhull_options: str = "QJ Qbb Qc"
):
    """
    Linear ND interpolation of ``xi``, using input data ``X`` and ``Y``, which
    also returns values `on` the convex hull boundary (which ``griddata`` often
    fails to do), and NaN for points truly outside the convex hull.

    Args:
        X (np.ndarray):
            (n, k) float array of data points to interpolate between.
        Y (np.ndarray):
            (n,) float array of values at ``X``.
        xi (np.ndarray):
            (L, k) float array of query points to interpolate values for.
        tol (float):
            Tolerance for including boundary points as inside. Default is
            1e-6.
        qhull_options (str):
            Options to pass to ``QHull`` via ``~scipy.spatial.Delaunay``.
            Default is "QJ Qbb Qc", where "QJ" means joggled input to avoid
            precision problems, "Qbb" scales coordinates for better
            conditioning, and "Qc" keeps coplanar points.

    Returns:
         np.ndarray:
            (N, k+1) float array of interpolated values within the
            convex hull, with NaN values outside the hull, and the input
            query points (xi) concatenated to the end.
    """
    n, k = np.shape(X)
    # Delaunay triangulation breaks our k-D polyhedron (polytope) of the convex hull into k-D
    # simplices (e.g. triangles in 2D, tetrahedra in 3D; simplest possible polytope in k-D space),
    # which each have k+1 vertices (e.g. 3 vertices for triangles, 4 vertices for tetrahedra, etc).

    if n == k + 1:  # Qhull's lifted-hull Delaunay needs N >= k+2 points, so handle single-simplex cases:
        simplices = np.arange(k + 1)[None, :]  # (1, k+1); S = 1
        Tinv = np.linalg.inv((X[:k] - X[k]).T)  # (k, k); T has columns (v_i - v_k)
        transform = np.empty((1, k + 1, k))  # (1, k+1, k)
        transform[0, :k, :], transform[0, k, :] = Tinv, X[k]  # (k, k) and (k,)
        lam_xi = (xi - X[k]) @ Tinv.T  # (L, k)
        bary = np.concatenate([lam_xi, 1.0 - lam_xi.sum(axis=1, keepdims=True)], axis=1)  # (L, k+1)
        simplex_indices = np.where(np.all(bary >= -tol, axis=1), 0, -1)  # (L,)
    else:  # setup Delaunay triangulation
        delaunay_tri = Delaunay(X, qhull_options=qhull_options)
        simplex_indices = delaunay_tri.find_simplex(xi, tol=tol)  # simplex indices; (L,) -> L is len(xi)
        simplices = delaunay_tri.simplices  # (S, k+1)
        transform = delaunay_tri.transform  # (S, k+1, k)

    inside_hull = simplex_indices >= 0  # outside = -1; tol treats near-edge as inside; (L,)
    if not inside_hull.any():  # no inside points, return array of NaNs of shape (L,)
        raise ValueError("No points found inside convex hull (of chemical potentials)")

    X_inside = xi[inside_hull]  # (N, k) where N is number of points inside hull; N <= L
    # k is the xi dimension (k-D chemical potential space)
    # inside_hull_simplex_indices = simplex_indices[inside_hull]  # (N,)

    # Linear interpolation via barycentric coordinates:
    # SciPy exposes an affine map from x to barycentric coords via tri.transform:
    #   For each simplex i:  T_i c = x - r_i, with c[:-1] first d barycentric coordinates,
    #   and c_last = 1 - sum(c[:-1])
    # (This mirrors what ``LinearNDInterpolator`` does under the hood)
    Ti = transform[simplex_indices[inside_hull]]  # (N, k+1, k)
    Xdif = X_inside - Ti[:, -1, :]  # x - r, shape (N, k)
    lam = np.einsum("Nij,Nj->Ni", Ti[:, :k, :], Xdif)  # first k barycentrics; (N, k)
    bary_coords = np.concatenate([lam, 1.0 - lam.sum(axis=1, keepdims=True)], axis=1)  # (N, k+1)

    vertex_indices_of_simplices = simplices[simplex_indices[inside_hull]]  # (N, k+1)
    vertex_values_of_simplices = Y[vertex_indices_of_simplices]  # (N, k+1)
    values_inside = np.einsum("Ni,Ni->N", bary_coords, vertex_values_of_simplices)  # N
    # combine input xi points (which are inside hull) with interpolated values for returned output:
    return np.hstack((X_inside, values_inside.reshape(-1, 1)))  # (N, k+1)


def entries_from_chempot_limits(
    chempots_dict: dict[str, dict],
) -> list[ComputedEntry]:
    """
    Generate a list of |ComputedEntry| objects from a ``doped`` dictionary of
    chemical potential limits.

    These entries will correspond to the ground-state phase for each
    chemically-stable composition in the chemical system, and can be used to
    generate a ``ChemicalPotentialDiagram`` object (used in chemical potential
    heatmap plotting).

    Args:
        chempots_dict (dict[str, dict]):
            ``doped`` chemical potential limits dictionary, as output by
            ``CompetingPhasesAnalyzer.chempots``, having the keys:

                - "limits": {"phaseA-phaseB-phaseC":
                    {"Li": mu_Li, "P": mu_P, ...}, ... }
                - "elemental_refs":
                    {"Li": mu_Li_ref, "P": mu_P_ref, ...}

    Returns:
        list[|ComputedEntry|]:
            List of |ComputedEntry| objects, with energies per formula unit.
    """
    phase_energies: dict[str, list[float]] = defaultdict(list)

    for limit, chempots in chempots_dict["limits"].items():
        for phase in limit.split("-"):
            composition = Composition(phase)
            phase_energies[phase].append(
                sum(composition[el] * float(chempots[str(el)]) for el in composition.elements)
            )  # total energy per formula unit from summed chemical potentials:

    # average duplicates (should all be the same energy anyway)
    entries = [
        ComputedEntry(phase, sum(energies) / len(energies)) for phase, energies in phase_energies.items()
    ]

    # ensure elemental references are included as entries:
    for element, elemental_chempot in chempots_dict.get("elemental_refs", {}).items():
        if element not in phase_energies:
            entries.append(ComputedEntry(element, float(elemental_chempot)))

    return entries


class CompetingPhasesAnalyzer(MSONable):
    def __init__(
        self,
        composition: str | Composition,
        entries: (
            PathLike | list[PathLike] | list[ComputedEntry] | list[ComputedStructureEntry]
        ) = "CompetingPhases",
        subfolder: PathLike | None = None,
        single_extrinsic_phase_limits: bool = False,
        verbose: bool = True,
        processes: int | None = None,
        check_compatibility: bool = True,
        calculator: str = "vasp",
    ):
        r"""
        Class for post-processing competing phases calculations, to determine
        the corresponding chemical potentials for the host ``composition``.

        This class can be initialised from calculation outputs
        (``vasprun.xml``\s with the default ``vasp`` ``calculator``) by
        specifying the path to the directory containing the outputs (e.g.
        ``"CompetingPhases"``) or a list of directories, or from a list of of
        |ComputedEntry|\s / |ComputedStructureEntry|\s (e.g. for use with
        high-throughput computing architectures such as ``atomate2`` or
        ``AiiDA``).

        Multiprocessing is used by default to speed up parsing of calculation
        outputs, which can be controlled with ``processes``. If parsing hangs,
        this may be due to memory issues, in which case you should reduce
        ``processes`` (e.g. 4 or less).

        If extrinsic (dopant/impurity) elements are present, their chemical
        potential limits will also be calculated, where every facet (limit)
        jointly determines ``μ_host`` and ``μ_extrinsic``.

        Args:
            composition (str, |Composition|):
                Composition of the host material (e.g. ``'LiFePO4'``, or
                ``Composition('LiFePO4')``, or
                ``Composition({"Li":1, "Fe":1, "P":1, "O":4})``).
            entries (PathLike, list[PathLike], list[|ComputedEntry|], list[|ComputedStructureEntry|]):
                Either a path to the base folder containing the calculation
                outputs (e.g. ``"CompetingPhases"``; default), which is
                searched recursively for calculation output files
                (``vasprun.xml(.gz)`` files with the default ``vasp``
                ``calculator``), or a list of paths to calculation output
                files / directories. Alternatively, can be a list of
                |ComputedEntry|\s / |ComputedStructureEntry|\s.
            subfolder (PathLike):
                Restrict parsing to ``vasprun.xml(.gz)`` files inside
                directories with this name (e.g. ``"vasp_std"``; default) --
                i.e. with a file-structure like:
                ``{Formula}_{spg}_EaH_{EaH}/{subfolder}/vasprun.xml(.gz)``)
                If ``None`` (default), then auto-detects the subfolder: the
                highest-priority name (case-insensitive) among
                ``SUBFOLDER_PRIORITY`` (``vasp_ncl``, ``singlepoint``,
                ``final``, ``relax``, ``vasp_std``, ``vasp_nkred_std``,
                ``vasp_gam``), `with calculation outputs` (``vasprun.xml(.gz)``
                files) and present in the discovered paths, is used. If none
                match, all found vaspruns are parsed. Set ``subfolder = "."``
                to parse calculation files from the competing phase directory
                with no subfolder.
            single_extrinsic_phase_limits (bool):
                If ``True``, pin ``μ_host`` at the intrinsic chemical potential
                limits, with the corresponding ``μ_extrinsic`` then determined
                at these points -- equivalent to only considering the chemical
                potential limits with a `single` extrinsic competing phase
                (as with ``single_extrinsic_phase_limits`` in
                :class:`CompetingPhases` initialisation). This typically
                returns fewer limits, which can simplify analysis when
                considering multiple extrinsic species at once, but can result
                in substantial portions of the chemical stability region being
                overlooked, and so is rarely recommended. Setting
                ``single_extrinsic_phase_limits=True`` in
                :class:`CompetingPhases` initialisation (to reduce the number
                of extrinsic phases to calculate), but keeping it as ``False``
                in parsing here is a significantly improved (though still
                imperfect) approximation, compared to ``True`` for parsing.
                Default is ``False`` (recommended; parse as normal with no
                restriction on the number of extrinsic phases in equilibrium at
                each facet (limit)).
            verbose (bool):
                Whether to print out information about directories that were
                skipped (due to no ``vasprun.xml(.gz)`` files being found),
                when parsing VASP outputs.
                Default is ``True``.
            processes (int):
                Number of processes to use for multiprocessing for expedited
                parsing of VASP outputs. If ``None`` (default), then the
                parsing time with multiprocessing is estimated based on
                ``vasprun.xml(.gz)`` file sizes, and used if predicted to be
                faster than serial processing. Set to 1 to prevent
                multiprocessing.
            check_compatibility (bool):
                Whether to check the compatibility of the parsed entries,
                by comparing their ``INCAR`` and ``POTCAR`` settings (if
                available). Default is ``True``.
            calculator (str):
                Name of the calculator used for the competing phase
                calculations (matching a ``doped.io.<calculator>`` parsing
                backend), when parsing from calculation outputs. Default:
                "vasp".

        Key attributes:
            composition (str):
                The bulk (host) composition.
            chempots (dict):
                Dictionary of the chemical potential limits for the host
                material, in the ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``). This can be
                directly used with the |DefectThermodynamics| plotting &
                analysis methods, and saved to file with ``dumpfn`` from
                ``monty.serialization``.
            chempots_df (DataFrame):
                ``DataFrame`` of the chemical potential limits for the host.
            intrinsic_chempots (dict):
                Dictionary of the chemical potential limits for the host
                material considering only intrinsic competing phases (i.e.
                ignoring extrinsic species), in the ``doped`` format. Equal
                to ``chempots`` when no extrinsic species are present.
            intrinsic_chempots_df (DataFrame):
                ``DataFrame`` of the intrinsic chemical potential limits for
                the host (i.e. ignoring extrinsic species).
            elements (list):
                List of all elements in the chemical system (host + extrinsic),
                from all parsed calculations.
            extrinsic_elements (list[str]):
                List of extrinsic elements in the chemical system (not present
                in ``composition``).
            intrinsic_elements (list[str]):
                List of intrinsic elements (i.e. those in ``composition``).
            bulk_entry (|ComputedStructureEntry|):
                The lowest energy computed entry for the host material.
            unstable_host (bool):
                Whether the host material is unstable with respect to competing
                phases (i.e. has an energy above hull > 0).
            entries (list[|ComputedEntry|, |ComputedStructureEntry|]):
                List of all parsed |ComputedEntry|\s /
                |ComputedStructureEntry|\s.
            extrinsic_entries (list[|ComputedEntry|, |ComputedStructureEntry|]):
                Subset of ``entries`` containing one or more extrinsic
                species. Empty if no extrinsic species are present.
            phase_diagram (PhaseDiagram):
                A ``pymatgen`` phase diagram generated from the parsed entries.
                Note that this phase diagram is likely not a full phase diagram
                for this chemical space, as we typically only generate the
                nearby competing phases for the host material to reduce the
                number of calculations.
            intrinsic_phase_diagram (PhaseDiagram):
                A ``pymatgen`` phase diagram containing only entries with
                elements from the host material (i.e. no extrinsic elements).
            elemental_energies (dict):
                Dictionary of the lowest energy elemental phases for each
                element in the chemical system.
            calc_output_paths (list[str]):
                List of paths to the calculation output files that were
                parsed (``vasprun.xml(.gz)`` files with the default ``vasp``
                calculator; empty if initialised directly from entries).
            parsed_folders (list):
                List of folders from which calculation outputs were parsed,
                if ``entries`` was given as a path / paths to directories.
            single_extrinsic_phase_limits (bool):
                Stored input parameter (see ``Args`` above).
        """
        self.composition = Composition(composition)
        self.elements: list[str] = [c.symbol for c in self.composition.elements]
        self.intrinsic_elements = self.elements.copy()
        self.extrinsic_elements: list[str] = []
        self.single_extrinsic_phase_limits = single_extrinsic_phase_limits

        # _from_calc_outputs or _from_entries depending on input
        if not isinstance(entries, str | PathLike | list):
            raise TypeError(
                f"`entries` must be either a path to a directory containing calculation outputs, "
                f"a list of paths, or a list of ComputedEntry/ComputedStructureEntry objects, "
                f"got type {type(entries)} instead!"
            )

        self.calculator = calculator
        self.calc_output_paths: list[str] = []
        self.parsed_folders: list[str] = []

        if isinstance(entries, str | PathLike) or isinstance(entries[0], str | PathLike):
            self._from_calc_outputs(
                path=entries,
                subfolder=subfolder,
                verbose=verbose,
                processes=processes,
                check_compatibility=check_compatibility,
            )
        else:
            self._from_entries(entries, check_compatibility=check_compatibility)

    def _from_entries(
        self, entries: list[ComputedEntry | ComputedStructureEntry], check_compatibility: bool = True
    ) -> None:
        r"""
        Initialises the |CompetingPhasesAnalyzer| object from a list of
        ``pymatgen`` |ComputedEntry|\s / |ComputedStructureEntry|\s.

        Args:
            entries (list[|ComputedEntry|, |ComputedStructureEntry|]):
                List of |ComputedEntry|\s / |ComputedStructureEntry|\s,
                from which to compute the phase diagram and chemical
                potential limits.
            check_compatibility (bool):
                Whether to check the compatibility of the parsed entries,
                by comparing their ``INCAR`` and ``POTCAR`` settings.
                Default is ``True``.
        """
        self.entries = entries
        intrinsic_entries: list[ComputedEntry | ComputedStructureEntry] = []
        extrinsic_entries: list[ComputedEntry | ComputedStructureEntry] = []
        bulk_comp_entries: list[ComputedEntry | ComputedStructureEntry] = []
        self.elemental_energies: dict[str, float] = {}

        for entry in entries:
            if len(entry.composition.elements) == 1:  # check if elemental
                el = next(iter(entry.composition.elements)).symbol
                if el not in self.elemental_energies:
                    self.elemental_energies[el] = entry.energy_per_atom
                    if el not in self.elements + self.extrinsic_elements:  # new (extrinsic) element
                        self.extrinsic_elements.append(el)

                elif entry.energy_per_atom < self.elemental_energies[el]:
                    # only include lowest energy elemental polymorph
                    self.elemental_energies[el] = entry.energy_per_atom

            if set(entry.composition.elements).issubset(self.composition.elements):
                intrinsic_entries.append(entry)  # intrinsic phase
                if entry.composition.reduced_composition == self.composition.reduced_composition:  # bulk
                    bulk_comp_entries.append(entry)

            else:  # extrinsic
                extrinsic_entries.append(entry)

        # check elemental references (both intrinsic and extrinsic) have been parsed; and error/warn if not
        all_parsed_entry_element_symbols = {
            el.symbol for entry in entries for el in entry.composition.elements
        }
        missing_elemental_refs = sorted(all_parsed_entry_element_symbols - set(self.elemental_energies))
        if missing_elemental_refs:
            missing_intrinsic = sorted(set(self.intrinsic_elements) & set(missing_elemental_refs))
            if missing_intrinsic:
                raise ValueError(
                    f"No elemental reference phase was parsed for host element(s): {missing_intrinsic}, "
                    f"which is required for chemical potential analyses (phase diagram construction). "
                    f"Please ensure unary (reference) calculations are included for each element in "
                    f"{self.composition.reduced_formula}."
                )
            warnings.warn(  # otherwise we are (just) missing extrinsic elemental references
                f"No elemental reference phase (required for chemical potential analysis) was parsed for "
                f"element(s): {missing_elemental_refs}, but these appear in some parsed (extrinsic) "
                f"competing phase entries. Entries containing these elements will be excluded from "
                f"analysis. Add the missing unary reference calculations to include them."
            )

            def _entry_contains_missing_elemental_ref(
                entry: ComputedEntry | ComputedStructureEntry,
            ) -> bool:
                return any(el.symbol in missing_elemental_refs for el in entry.composition.elements)

            filtered_entries = [e for e in entries if not _entry_contains_missing_elemental_ref(e)]
            if not filtered_entries:
                raise ValueError(
                    "After excluding phases containing elements without elemental reference entries, "
                    "no computed entries remained!"
                )
            extrinsic_entries = [
                e for e in extrinsic_entries if not _entry_contains_missing_elemental_ref(e)
            ]
            self.entries = filtered_entries

        if not bulk_comp_entries:
            intrinsic_compositions = (
                {entry.composition.reduced_formula for entry in intrinsic_entries}
                if intrinsic_entries
                else None
            )
            raise ValueError(
                f"Could not find bulk phase for {self.composition.reduced_formula} in the supplied "
                f"data. Found intrinsic phase diagram entries for: {intrinsic_compositions}"
            )

        # lowest energy bulk phase
        self.bulk_entry = sorted(bulk_comp_entries, key=lambda x: x.energy_per_atom)[0]
        self.unstable_host = False

        # check entry compatibilities (calculation settings), with the calculator backend (if it
        # implements this check); taking the first of bulk entry, bulk comp entries, intrinsic, all
        # entries `with` the relevant settings data as the reference (template) entry:
        if check_compatibility and (
            check_entry_compatibility := getattr(self._backend, "check_entry_compatibility", None)
        ):
            check_entry_compatibility(
                entries,
                template_candidates=[self.bulk_entry, *bulk_comp_entries, *intrinsic_entries, *entries],
            )

        # sort extrinsic elements and energies dict by periodic table positioning (deterministically),
        # and add to self.elements:
        self.extrinsic_elements = sorted(self.extrinsic_elements, key=_element_sort_func)
        self.elemental_energies = dict(
            sorted(self.elemental_energies.items(), key=lambda x: _element_sort_func(x[0]))
        )  # comprises all elements for which at least 1 elemental (unary) phase has been parsed
        self.elements += self.extrinsic_elements

        self.intrinsic_phase_diagram = PhaseDiagram(
            intrinsic_entries,
            list(map(Element, self.composition.elements)),  # preserve bulk comp element ordering
        )

        # check if it's stable and if not, warn user and downshift to get _least_ unstable point on convex
        # hull for the host material
        if self.bulk_entry not in self.intrinsic_phase_diagram.stable_entries:
            self.unstable_host = True
            eah = self.intrinsic_phase_diagram.get_e_above_hull(self.bulk_entry)
            warnings.warn(
                f"{self.composition.reduced_formula} is not stable with respect to competing phases, "
                f"having an energy above hull of {eah:.4f} eV/atom.\n"
                f"Formally, this means that (based on the supplied athermal calculation data) the host "
                f"material is unstable and so has no chemical potential limits; though in reality the "
                f"host may be stabilised by temperature effects etc, or just a metastable phase.\n"
                f"Here we will determine a single chemical potential 'limit' corresponding to the least "
                f"unstable (i.e. closest) point on the convex hull for the host material, "
                f"as an approximation for the true chemical potentials."
            )  # TODO: Add example of adjusting the entry energy after loading (if user has calculated
            # e.g. temperature effects) and link in this warning
            # decrease bulk_pde energy per atom by ``energy_above_hull`` + 0.1 meV/atom
            name = description = (
                "Manual energy adjustment to move the host composition to the calculated convex hull"
            )
            renormalised_bulk_entry = _renormalise_entry(
                self.bulk_entry, eah + 1e-4, name=name, description=description
            )
            self.intrinsic_phase_diagram = PhaseDiagram(
                [*self.intrinsic_phase_diagram.entries, renormalised_bulk_entry],
                list(map(Element, self.composition.elements)),  # preserve bulk comp element ordering
            )

        self.phase_diagram = PhaseDiagram(
            [*self.intrinsic_phase_diagram.entries, *extrinsic_entries],
            list(map(Element, self.composition.elements + self.extrinsic_elements)),  # preserve ordering
        )

        for entry in self.phase_diagram.entries:
            formation_energy = self.phase_diagram.get_form_energy_per_atom(entry)
            if np.isinf(formation_energy) or np.isnan(formation_energy):
                warnings.warn(
                    f"Entry for {entry.reduced_formula} has an infinite/NaN calculated formation energy, "
                    f"indicating an issue with parsing. This may cause failures in chemical potential "
                    f"analyses, and should be checked!"
                )

            # set energies above hull of entries in the intrinsic phase diagram
            entry.data["energy_above_hull"] = self.phase_diagram.get_e_above_hull(entry)

        # then sort:
        self.entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=self.composition))
        _name_entries_and_handle_duplicates(self.entries)  # set entry names

        self.extrinsic_entries = [
            entry
            for entry in self.entries
            if not set(entry.composition.elements).issubset(self.composition.elements)
        ]

        # update ordering in PhaseDiagram entries to match:
        self.intrinsic_phase_diagram.entries = sorted(
            self.intrinsic_phase_diagram.entries,
            key=lambda x: _entries_sort_func(x, bulk_composition=self.composition),
        )
        self.phase_diagram.entries = sorted(
            self.phase_diagram.entries,
            key=lambda x: _entries_sort_func(x, bulk_composition=self.composition),
        )
        self.chempots_df = self.calculate_chempots(
            verbose=False, single_extrinsic_phase_limits=self.single_extrinsic_phase_limits
        )

    def _from_calc_outputs(
        self,
        path: PathLike | list[PathLike] = "CompetingPhases",
        subfolder: PathLike | None = None,
        verbose: bool = True,
        processes: int | None = None,
        check_compatibility: bool = True,
    ) -> None:
        r"""
        Parses competing phase energies from calculation outputs
        (``vasprun.xml(.gz)`` files with the default ``vasp`` calculator),
        generating |ComputedStructureEntry|\s (with the
        ``get_competing_phase_entry()`` function of the ``doped.io`` backend
        for ``self.calculator``) and then continuing initialisation with the
        ``CompetingPhasesAnalyzer._from_entries`` method.

        Recursively searches for ``vasprun.xml(.gz)`` files under ``path``.
        When ``subfolder`` is set, only vaspruns inside directories with that
        name are used.  When ``subfolder`` is ``None`` (default), the
        highest-priority subfolder present among ``SUBFOLDER_PRIORITY``
        (``vasp_ncl``, ``singlepoint``, ``final``, ``relax``, ``vasp_std``,
        ``vasp_nkred_std``, ``vasp_gam``; case-insensitive),
        `with calculation outputs` (``vasprun.xml(.gz)`` files) and present in
        the discovered paths, is used. If none of these are found, all
        discovered ``vasprun.xml(.gz)`` files are parsed.

        If ``path`` is a list, each element may be a direct path to a
        ``vasprun.xml(.gz)`` file **or** to a directory containing one,
        which will be located via a recursive search.

        Args:
            path (PathLike or list):
                Either a path to the base folder containing competing phase
                calculation outputs (searched recursively for
                ``vasprun.xml(.gz)`` files), or a list of paths to
                ``vasprun.xml(.gz)`` files / directories.
            subfolder (PathLike):
                Restrict to ``vasprun.xml(.gz)`` files in directories with
                this name.  Default is ``None``, in which case the subfolder is
                auto-detected (see above).
            verbose (bool):
                Whether to print out information about directories that were
                skipped (due to no ``vasprun.xml(.gz)`` files being found).
                Default is ``True``.
            processes (int):
                Number of processes to use for multiprocessing for expedited
                parsing of VASP outputs. If ``None`` (default), then the
                parsing time with multiprocessing is estimated based on
                ``vasprun.xml(.gz)`` file sizes, and used if predicted to be
                faster than serial processing. Set to 1 to prevent
                multiprocessing.
            check_compatibility (bool):
                Whether to check the compatibility of the parsed entries,
                by comparing their ``INCAR`` and ``POTCAR`` settings.
                Default is ``True``.
        """
        output_file = self._backend.CALC_OUTPUT_MASK[0]  # e.g. "vasprun.xml" with VASP
        if isinstance(path, list):
            self._collect_calc_outputs_from_list(path)
        elif isinstance(path, PathLike):
            self._collect_calc_outputs_from_directory(path, subfolder, verbose)
        else:
            raise ValueError(
                f"`path` should either be a path to a folder (with competing phase "
                f"calculations), or a list of paths to {output_file}(.gz) files."
            )

        if not self.calc_output_paths:
            raise FileNotFoundError(
                f"No {output_file}(.gz) files found under '{path}'. Please check that the folder "
                f"structure and input parameters are in the correct format (see docs/tutorials)."
            )

        # Ignore POTCAR warnings when loading vasprun.xml
        # pymatgen assumes the default PBE with no way of changing this
        _ignore_pmg_warnings()

        self.entries = []
        failed_parsing_dict: dict[str, list] = {}
        parse_entry = partial(_parse_entry_and_catch_exception, calculator=self.calculator)
        if processes is None:  # multiprocessing?
            # from quick tests; Pool takes about 2.75s to initialise, with negligible additional cost per
            # process, and vasprun parsing with pymatgen v2025.1.9 takes ~0.025 s/MB (uncompressed file
            # size; with gzip compressing large vasprun.xml files by ~15-20x) -- (~1.5-3x faster after SK's
            # updates to vasprun parsing in pymatgen v2025.4.16).
            # So for multiprocessing to be worth it, we need at least 2 outputs to parse, with a summed
            # uncompressed output file size, excluding the largest one, of around >100 Mb
            def _estimate_uncompressed_size(calc_output_path: PathLike) -> float:
                return (os.path.getsize(calc_output_path) / 1e6) * (
                    20 if calc_output_path.endswith(".gz") else 1
                )

            calc_output_sizes_MB = [
                _estimate_uncompressed_size(calc_output_path)
                for calc_output_path in self.calc_output_paths
            ] or [0]
            mp_context = get_mp_context()
            if sum(calc_output_sizes_MB) - max(calc_output_sizes_MB) > 100:
                # only multiprocess as much as makes sense:
                num_large_outputs = sum(1 for size in calc_output_sizes_MB if size > 20)
                processes = min(max(1, mp_context.cpu_count() - 1), num_large_outputs - 1)
            else:
                processes = 1

        parsing_results = []
        desc = f"Parsing {output_file}s..."
        with warnings.catch_warnings():
            # VASP-specific warning class; simply never raised with other calculator backends:
            warnings.filterwarnings("ignore", category=UnconvergedVASPWarning)  # checked and warned later
            if processes > 1:  # multiprocessing
                with pool_manager(processes) as pool:  # result is parsed entry
                    for result in tqdm(
                        pool.imap_unordered(parse_entry, self.calc_output_paths),
                        total=len(self.calc_output_paths),
                        desc=desc,
                    ):
                        parsing_results.append(result)
            else:
                for calc_output_path in tqdm(self.calc_output_paths, desc=desc):
                    parsing_results.append(parse_entry(calc_output_path))

        electronic_unconverged_paths = []
        ionic_unconverged_paths = []
        for result in parsing_results:
            if isinstance(result[0], ComputedEntry | ComputedStructureEntry):
                # successful parse; result is entry, parsed folder, converged electronic and ionic
                self.entries.append(result[0])
                self.parsed_folders.append(result[1])
                if not result[2]:
                    electronic_unconverged_paths.append(result[1])
                if not result[3]:
                    ionic_unconverged_paths.append(result[1])
            else:  # failed parse; result is error message and path
                if str(result[0]) in failed_parsing_dict:
                    failed_parsing_dict[str(result[0])] += [result[1]]
                else:
                    failed_parsing_dict[str(result[0])] = [result[1]]

        if failed_parsing_dict:
            warning_string = (
                f"Failed to parse the following `{output_file}` files:\n(files: error)\n"
                + "\n".join([f"{paths}: {error}" for error, paths in failed_parsing_dict.items()])
            )
            warnings.warn(warning_string)

        # check if any calculations are unconverged, and warn together:
        for unconverged_paths, unconverged_type in zip(
            [electronic_unconverged_paths, ionic_unconverged_paths],
            ["Electronic", "Ionic"],
            strict=False,
        ):
            if unconverged_paths:
                warnings.warn(
                    f"{unconverged_type} convergence was not reached for {output_file}s in:\n"
                    + "\n".join(unconverged_paths)
                )

        if not self.entries:
            raise FileNotFoundError(
                f"No {output_file} files have been parsed, suggesting issues with parsing! Please check "
                f"that folders and input parameters are in the correct format (see docstrings/tutorials)."
            )

        return self._from_entries(self.entries, check_compatibility=check_compatibility)

    @property
    def _backend(self):
        """
        The ``doped.io`` output-parsing backend module for ``self.calculator``.
        """
        return get_backend(getattr(self, "calculator", "vasp"))

    def _collect_calc_outputs_from_list(self, path_list: list[PathLike]) -> None:
        """
        Populate ``self.calc_output_paths`` from a list of paths, where each
        entry can be a direct calculation output file path
        (``vasprun.xml(.gz)`` with ``VASP``) or a directory containing one.
        """
        output_file = self._backend.CALC_OUTPUT_MASK[0]
        for entry_path in path_list:
            if str(output_file) in str(entry_path) and not str(entry_path).startswith("."):
                self.calc_output_paths.append(str(entry_path))
                continue

            calc_output_path = self._find_calc_output_in_directory(entry_path)
            if calc_output_path is not None:
                self.calc_output_paths.append(str(calc_output_path))

    def _collect_calc_outputs_from_directory(
        self,
        path: PathLike,
        subfolder: PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Recursively search ``path`` for calculation output files
        (``vasprun.xml(.gz)`` with VASP), using the
        :func:`~doped.io.utils._find_calc_outputs` helper for file discovery
        and subfolder auto-detection (parameterised by the calculator backend's
        ``CALC_OUTPUT_MASK`` / ``SUBFOLDER_PRIORITY``).
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"No such file or directory: '{path}'")

        backend = self._backend
        output_file = backend.CALC_OUTPUT_MASK[0]
        calc_files_df, _folders, resolved_subfolder = _io_utils._find_calc_outputs(
            root,
            subfolder,
            calc_output_mask=backend.CALC_OUTPUT_MASK,
            subfolder_priority=getattr(backend, "SUBFOLDER_PRIORITY", []),
        )
        if calc_files_df.empty:
            return

        if resolved_subfolder != ".":  # actual subfolder found
            filtered = calc_files_df[
                calc_files_df["folder_path"].apply(lambda p: p.name == resolved_subfolder)
            ]
            if not filtered.empty:
                calc_files_df = filtered
            elif verbose:
                warnings.warn(
                    f"No {output_file} files found in '{resolved_subfolder}' subfolders under {path}. "
                    f"Using all {len(calc_files_df)} {output_file} files found."
                )

        for directory in sorted(calc_files_df["folder_path"].unique()):
            calc_output_path, multiple = _io_utils._get_output_files_and_check_if_multiple(
                output_file, str(directory)
            )
            if calc_output_path and os.path.exists(calc_output_path):
                if multiple:
                    self._multiple_files_warning(output_file, directory, calc_output_path)
                self.calc_output_paths.append(calc_output_path)

    def _find_calc_output_in_directory(self, directory: PathLike) -> str | None:
        """
        Find a single calculation output file (``vasprun.xml(.gz)`` with VASP)
        in ``directory``.

        Returns the path as a string, or ``None`` if not found.
        """
        output_file = self._backend.CALC_OUTPUT_MASK[0]
        calc_output_path, multiple = None, False
        with contextlib.suppress(FileNotFoundError):
            calc_output_path, multiple = _io_utils._get_output_files_and_check_if_multiple(
                output_file, str(directory)
            )
        if calc_output_path and os.path.exists(calc_output_path):
            if multiple:
                self._multiple_files_warning(output_file, directory, calc_output_path)
            return str(calc_output_path)
        return None

    def _multiple_files_warning(self, output_file, directory, chosen_filepath) -> None:
        """
        Warn that multiple calculation output files were found in
        ``directory``, using the backend's ``FILE_PARSING_ACTIONS`` for the
        (informative) description of what the file is parsed for.
        """
        action = getattr(self._backend, "FILE_PARSING_ACTIONS", {}).get(
            output_file, "parse the calculation energy and metadata."
        )
        _io_utils._multiple_files_warning(
            output_file, directory, chosen_filepath, action, dir_type="competing phase"
        )

    def get_formation_energy_df(
        self,
        prune_polymorphs: bool = False,
        include_raw_energies: bool = False,
        skip_rounding: bool = False,
    ) -> pd.DataFrame:
        """
        Generate a ``DataFrame`` of the formation energies of parsed competing
        phases calculations.

        Useful for quick summary and analysis of results, e.g. to include in
        the SI of journal articles or a thesis -- to aid open-science and
        reproducibility, or for quick data sharing, or for use with other
        codes. Can be saved to file with ``formation_energy_df.to_csv()`` etc.

        Rows are sorted according to ``CompetingPhasesAnalyzer.entries``, which
        by default is sorted by energy above hull (with the host composition
        first), then by the number of elements in the formula, then by the
        position of elements in the periodic table (main group elements, then
        transition metals, sorted by row), then alphabetically.

        Args:
            prune_polymorphs (bool):
                Whether to only write the lowest energy polymorphs for each
                composition. Doesn't affect chemical potential limits (only the
                ground-state polymorphs matter for this).
                Default is False.
            include_raw_energies (bool):
                Whether to include the raw (QM/DFT) energies in the output
                ``DataFrame``. Default is ``False``.
            skip_rounding (bool):
                Whether to skip rounding the energies to 3 decimal places
                (1 meV/atom or meV/fu) for cleanliness. Default is ``False``.
        """
        rows = []
        for entry in self.entries:
            composition = entry.composition
            formula_units = composition.get_reduced_composition_and_factor()[1]
            kpoints_data = entry.data.get("kpoints")
            kpoints_string = (
                "x".join(str(k) for k in kpoints_data[0])
                if (kpoints_data and len(kpoints_data) == 1)
                else "N/A"
            )
            space_group = (
                entry.data.get("doped_name", f"{entry.name}_N/A_EaH_")
                .split(f"{entry.name}_")[1]
                .split("_EaH_")[0]
            )

            rows.append(
                {
                    "Formula": composition.reduced_formula,
                    "Space Group": space_group,
                    "Energy above Hull (eV/atom)": entry.data.get("energy_above_hull", "N/A"),
                    "Formation Energy (eV/fu)": self.phase_diagram.get_form_energy(entry) / formula_units,
                    "Formation Energy (eV/atom)": self.phase_diagram.get_form_energy_per_atom(entry),
                    "Raw Energy (eV/fu)": entry.energy / formula_units,
                    "Raw Energy (eV/atom)": entry.energy_per_atom,
                    "k-points": kpoints_string,
                }
            )

        formation_energy_df = pd.DataFrame(rows)

        if prune_polymorphs:  # only keep the lowest energy polymorphs
            indices = formation_energy_df.groupby("Formula")["Raw Energy (eV/atom)"].idxmin()
            formation_energy_df = formation_energy_df.loc[sorted(indices)]  # retain ordering

        if not include_raw_energies:
            formation_energy_df = formation_energy_df.drop(
                columns=["Raw Energy (eV/fu)", "Raw Energy (eV/atom)"]
            )
        formation_energy_df = formation_energy_df.round(3) if not skip_rounding else formation_energy_df

        return formation_energy_df.set_index("Formula")

    def calculate_chempots(
        self,
        extrinsic: str | Element | Iterable[str] | Iterable[Element] | None = None,
        single_extrinsic_phase_limits: bool = False,
        sort_by: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Calculates the chemical potential limits for the host composition
        (``self.composition``) and any ``extrinsic`` elements (which defaults
        to ``self.extrinsic_elements`` -- i.e. all extrinsic elements present
        in ``self.entries``).

        If extrinsic (dopant/impurity) elements are present, their chemical
        potential limits will also be calculated, where every facet (limit)
        jointly determines ``μ_host`` and ``μ_extrinsic``.

        Args:
            extrinsic (str | Element | Iterable[str] | Iterable[Element] | None):
                If set, will calculate the chemical potential limits for these
                extrinsic elements (and all intrinsic elements) only. Can be a
                single element (``str`` or ``Element``), or a list of elements.
                If ``None`` (default), uses ``self.extrinsic_elements`` (all
                extrinsic elements present).
            single_extrinsic_phase_limits (bool):
                If ``True``, pin ``μ_host`` at the intrinsic chemical potential
                limits, with the corresponding ``μ_extrinsic`` then determined
                at these points -- equivalent to only considering the chemical
                potential limits with a `single` extrinsic competing phase
                (as with ``single_extrinsic_phase_limits`` in
                :class:`CompetingPhases` initialisation). This typically
                returns fewer limits, which can simplify analysis when
                considering multiple extrinsic species at once, but can result
                in substantial portions of the chemical stability region being
                overlooked, and so is rarely recommended. Setting
                ``single_extrinsic_phase_limits=True`` in
                :class:`CompetingPhases` initialisation (to reduce the number
                of extrinsic phases to calculate), but keeping it as ``False``
                in parsing here is a significantly improved (though still
                imperfect) approximation, compared to ``True`` for parsing.
                Default is ``False`` (recommended; parse as normal with no
                restriction on the number of extrinsic phases in equilibrium at
                each facet (limit)).
            sort_by (str):
                If set, will sort the chemical potential limits in
                ``self.(intrinsic_)chempots`` and the output ``DataFrame``
                according to the chemical potential of the specified element
                (from element-rich to element-poor conditions).
            verbose (bool):
                If ``True`` (default), will print the parsed chemical potential
                limits.

        Returns:
            ``pandas`` ``DataFrame``, optionally saved to csv.
        """
        if extrinsic is None:
            extrinsic_iter: Iterable[str | Element] = self.extrinsic_elements
        elif isinstance(extrinsic, str | Element):
            extrinsic_iter = [extrinsic]
        else:
            extrinsic_iter = list(extrinsic)
        extrinsic_elements = [Element(e) for e in extrinsic_iter]

        if missing_extrinsic := [
            elt for elt in extrinsic_elements if elt.symbol not in self.elemental_energies
        ]:
            raise ValueError(
                f"Elemental reference phase for the specified extrinsic species "
                f"{[elt.symbol for elt in missing_extrinsic]} was not parsed, but is necessary for "
                f"chemical potential calculations. Please ensure that this phase is present in the "
                f"calculation directory and is being correctly parsed."
            )

        self.chempots = self.intrinsic_chempots = get_doped_chempots_from_entries(
            self.intrinsic_phase_diagram.entries,
            self.composition,
            single_chempot_limit=self.unstable_host,
        )  # self.chempots overwritten below when extrinsic elements present

        def _get_chempots_df_from_chempots(chempots: dict) -> pd.DataFrame:
            return pd.DataFrame.from_dict(  # chemical potentials as pandas dataframe
                {k: list(v.values()) for k, v in chempots["limits_wrt_el_refs"].items()},
                orient="index",
                columns=[str(k) for k in next(iter(chempots["limits_wrt_el_refs"].values()))],
            ).rename_axis("Limit")

        chempots_df = self.intrinsic_chempots_df = _get_chempots_df_from_chempots(self.intrinsic_chempots)

        if extrinsic_elements:
            if not single_extrinsic_phase_limits:
                allowed_symbols = set(self.intrinsic_elements) | {elt.symbol for elt in extrinsic_elements}
                self.chempots = get_doped_chempots_from_entries(
                    [
                        entry
                        for entry in self.phase_diagram.entries
                        if {el.symbol for el in entry.composition.elements}.issubset(allowed_symbols)
                    ],
                    self.composition,
                    single_chempot_limit=self.unstable_host,
                )
                chempots_df = _get_chempots_df_from_chempots(self.chempots)
            else:  # single extrinsic phase limits
                for extrinsic_element in extrinsic_elements:
                    chempots_df = self._calculate_dilute_extrinsic_chempot_lims(
                        extrinsic_element=extrinsic_element,
                        chempots_df=chempots_df,
                    )  # updates self.chempots, returns the updated chempots_df

        if sort_by is not None:  # sort chempots_df, self.chempots and self.intrinsic_chempots:
            chempots_df = chempots_df.sort_values(by=sort_by, ascending=False)

            def _sort_chempots_dict_by(chempots_dict: dict, sort_by: str) -> dict:
                for limit_key in ["limits_wrt_el_refs", "limits"]:
                    chempots_dict[limit_key] = dict(
                        sorted(chempots_dict[limit_key].items(), key=lambda x: x[1][sort_by], reverse=True)
                    )
                return chempots_dict

            _sort_chempots_dict_by(self.chempots, sort_by)
            if sort_by in self.intrinsic_elements:  # extrinsic species are absent from intrinsic_chempots
                self.intrinsic_chempots_df = self.intrinsic_chempots_df.sort_values(
                    by=sort_by, ascending=False
                )
                _sort_chempots_dict_by(self.intrinsic_chempots, sort_by)

        # re-sort chempots_df columns and self.chempots inner dicts to match self.elements ordering:
        chempots_df = chempots_df[[el for el in self.elements if el in chempots_df.columns]]
        self.intrinsic_chempots_df = self.intrinsic_chempots_df[
            [el for el in self.intrinsic_elements if el in chempots_df.columns]
        ]
        for limit_key in ["limits", "limits_wrt_el_refs"]:
            self.chempots[limit_key] = {
                limit: {el: chempot_dict[el] for el in self.elements if el in chempot_dict}
                for limit, chempot_dict in self.chempots[limit_key].items()
            }
            self.intrinsic_chempots[limit_key] = {
                limit: {el: chempot_dict[el] for el in self.intrinsic_elements if el in chempot_dict}
                for limit, chempot_dict in self.intrinsic_chempots[limit_key].items()
            }

        if verbose:
            print("Calculated chemical potential limits (in eV wrt elemental reference phases): \n")
            print(chempots_df)

        return chempots_df

    def _calculate_dilute_extrinsic_chempot_lims(
        self, extrinsic_element: Element, chempots_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This function calculates chemical potential limits for extrinsic /
        dopant elements, using the single-extrinsic-phase-limits approximation
        (``full_sub_approach=False`` in PyCDT): host (intrinsic) element
        chemical potentials are pinned by the intrinsic phase diagram limits
        (computed without any extrinsic phases), and for each intrinsic limit
        we then find the extrinsic phase that most strictly bounds
        ``μ_extrinsic_element``.

        This is the ``single_extrinsic_phase_limits=True`` procedure, where
        only chemical potential limits with 1 extrinsic phase in equilibrium
        are considered.

        As discussed in the :class:`CompetingPhasesAnalyzer` docstring, this
        approach is very rarely recommended, as it can result in substantial
        portions of the chemical stability region being neglected.

        For an extrinsic phase with formula ``{host}X_n``, its stability
        constraint is: ``n_X·μ_X + Σ_{i ∈ host} n_i·μ_i <= E_form(phase)``
        (with ``μ_i`` relative to elemental refs). So at fixed ``μ_host``
        (which is the case for a chemical potential limit with only 1 extrinsic
        phase; the other N phases in equilibrium fully determine ``μ_host``):
        ``μ_X <= (E_form - Σ n_i·μ_i) / n_X``. The maximum allowed ``μ_X`` is
        the minimum of this RHS inequality over all extrinsic entries; the
        corresponding entry being the (extrinsic) limiting phase.

        This function assumes ``single_extrinsic_phase_limits=True`` and
        therefore skips any co-doping competing phases (i.e. those containing
        more than one distinct extrinsic species), which would otherwise
        mis-state the ``μ_X`` bound (since here only ``μ_host`` is pinned,
        while the other extrinsic species' chempots are not).
        """
        host_element_symbols = {elt.symbol for elt in self.composition.elements}
        for limit, chempot_series in list(chempots_df.iterrows()):
            assert isinstance(limit, str)  # typing
            chempots_df.loc[limit, extrinsic_element.symbol] = np.nan
            potential_limiting_extrinsic_entries: list[tuple[ComputedEntry, float]] = []
            for entry in self.extrinsic_entries:
                n_extrinsic_entry = entry.composition[extrinsic_element]  # n_X

                # skip phases with only other extrinsic species, as they don't constrain μ_X in the
                # ``single_extrinsic_phase_limits=True`` approach:
                if any(
                    el
                    for el in entry.composition.elements
                    if el != extrinsic_element and el.symbol not in host_element_symbols
                ):
                    continue

                formation_energy = self.phase_diagram.get_form_energy(entry)  # E_form(phase)
                intrinsic_chempot_contribution = sum(  # Σ_{i ∈ host} n_i·μ_i
                    chempot_series[elt.symbol] * entry.composition[elt]
                    for elt in self.composition.elements
                )
                extrinsic_chempot = _custom_round(  # μ_X <= (E_form - Σ n_i·μ_i) / n_X
                    (formation_energy - intrinsic_chempot_contribution) / n_extrinsic_entry, 4
                )
                if np.isfinite(extrinsic_chempot):
                    potential_limiting_extrinsic_entries.append((entry, extrinsic_chempot))

            # most-restrictive (smallest) finite μ_extrinsic_element -> limiting phase:
            # in rare cases of numerical degeneracy, two or more extrinsic phases can tie at the host limit
            # (giving the same extrinsic chemical potential). Thus for determinism we favour the phase with
            # the simplest (reduced) composition and (if still tied) the lowest formation energy:
            potential_limiting_extrinsic_entries.sort(
                key=lambda x: (
                    x[1],
                    sum(x[0].composition.reduced_composition.get_el_amt_dict().values()),
                    self.phase_diagram.get_form_energy(x[0]),
                )
            )
            limiting_extrinsic_entry, limiting_extrinsic_chempot = potential_limiting_extrinsic_entries[0]
            chempots_df.loc[limit, extrinsic_element.symbol] = _custom_round(limiting_extrinsic_chempot, 4)
            chempots_df = chempots_df.rename(  # Append the limiting phase to the limit key
                index={limit: f"{limit}-{limiting_extrinsic_entry.name}"}
            )

        limits_wrt_el_refs = {
            limit: {el: chempots_df.loc[limit, el] for el in chempots_df.columns}
            for limit in chempots_df.index
        }
        self.chempots = _round_floats(  # round all floats to 4 decimal places (0.1 meV/atom)
            {
                "limits": {
                    limit: {  # convert limits wrt el refs to absolute chempots
                        el: mu_wrt_el_ref + self.elemental_energies[el]
                        for el, mu_wrt_el_ref in limits_wrt_el_refs[limit].items()
                    }
                    for limit in limits_wrt_el_refs
                },
                "elemental_refs": self.elemental_energies,
                "limits_wrt_el_refs": limits_wrt_el_refs,
            },
            4,
        )
        return chempots_df

    def to_LaTeX_table(self, splits: int = 1, prune_polymorphs: bool = True) -> str:
        """
        A very simple function to print out the competing phase formation
        energies in a LaTeX table format, showing the formula, space group,
        energy above hull, kpoints (if present in the parsed data) and
        formation energy.

        Needs the ``mhchem`` package to work and does `not` use the
        ``booktabs`` package; change ``hline`` to ``toprule``, ``midrule`` and
        ``bottomrule`` if you want to use ``booktabs`` style.

        Args:
            splits (int):
                Number of splits for the table; either 1 (default) or 2 (with
                two large columns, each with the formula, kpoints (if present)
                and formation energy (sub-)columns).
            prune_polymorphs (bool):
                Whether to only print out the lowest energy polymorphs for each
                composition.
                Default is True.

        Returns:
            str: LaTeX table string
        """
        if splits not in [1, 2]:
            raise ValueError("`splits` must be either 1 or 2")
        # done in the pyscfermi report style
        form_e_df = self.get_formation_energy_df(prune_polymorphs)
        form_e_df["Formula"] = form_e_df.index
        formation_energy_data = form_e_df.to_dict(orient="records")

        kpoints_col = any("k-points" in item for item in formation_energy_data)

        string = "\\begin{table}[h]\n\\centering\n"
        string += (
            "\\caption{Formation energies per formula unit ($\\Delta E_f$) of \\ce{"
            + self.composition.reduced_formula
            + "} and all competing phases"
            + (", with k-meshes used in calculations." if kpoints_col else ".")
            + (" Only the lowest energy polymorphs are included.}\n" if prune_polymorphs else "}\n")
        )
        string += "\\label{tab:competing_phase_formation_energies}\n"
        column_names_string = (
            "Formula & Space Group & E$_{\\textrm{Hull}}$ (eV/atom)"
            + (" & k-mesh" if kpoints_col else "")
            + " & $\\Delta E_f$ (eV/fu)"
        )

        def _format_row_cells(row: dict) -> str:
            r"""
            Render one competing-phase entry as LaTeX table cells (no trailing
            ``\\\\``).
            """
            formula_cell = "\\ce{" + row["Formula"] + "}"
            space_group_cell = latexify_spacegroup(row.get("Space Group", "N/A"))
            eah_cell = f"{row['Energy above Hull (eV/atom)']:.3f}"
            formation_energy_cell = f"{row['Formation Energy (eV/fu)']:.3f}"
            cells = [formula_cell, space_group_cell, eah_cell]
            if kpoints_col:
                k1, k2, k3 = row.get("k-points", "0x0x0").split("x")
                cells.append(f"{k1}$\\times${k2}$\\times${k3}")
            cells.append(formation_energy_cell)
            return " & ".join(cells)

        if splits == 1:
            string += "\\begin{tabular}" + ("{ccccc}" if kpoints_col else "{cccc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " \\\\ \\hline\n"
            for row in formation_energy_data:
                string += _format_row_cells(row) + " \\\\\n"

        elif splits == 2:
            string += "\\begin{tabular}" + ("{ccccc|ccccc}" if kpoints_col else "{cccc|cccc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " & " + column_names_string + " \\\\ \\hline\n"

            mid = len(formation_energy_data) // 2
            first_half = formation_energy_data[:mid]
            last_half = formation_energy_data[mid:]
            for left_row, right_row in zip(first_half, last_half, strict=False):
                string += _format_row_cells(left_row) + " & " + _format_row_cells(right_row) + " \\\\\n"

        string += "\\hline\n"
        string += "\\end{tabular}\n"
        string += "\\end{table}\n"

        return string

    def plot_chempot_heatmap(
        self,
        dependent_element: str | Element | None = None,
        fixed_elements: dict[str, float] | None = None,
        bordering_phases: bool = True,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        cbar_range: tuple[float, float] | None = None,
        colormap: str | colors.Colormap | None = None,
        padding: float | None = None,
        title: str | bool = False,
        label_positions: bool | dict[str, tuple[float, float]] | list[tuple[float, float]] = True,
        filename: PathLike | None = None,
        style_file: PathLike | None = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot a heatmap of the chemical potentials for a multinary system,
        showing bordering secondary phases.

        In this plot, the ``dependent_element`` chemical potential is plotted
        as a heatmap over the stability region of the host composition, as a
        function of two other elemental chemical potentials on the x and y
        axes.

        3-D data is required to plot a 2-D heatmap, and so this function can be
        applied as-is for ternary systems (or binary systems with an extrinsic
        element), but for higher-dimensional systems a set of chemical
        potential constraints must be provided (as ``fixed_elements``) to
        project the chemical stability region to 3-D; see the competing phases
        tutorial section on
        :ref:`chemical_potentials_tutorial:Analysing and visualising the chemical potential limits`.

        Extrinsic chemical potentials are also supported; added as additional
        dimensions to the chemical potential diagram and can be used as plot
        axes (see ``dependent_element`` / ``fixed_elements``).

        Note that due to an issue with ``matplotlib`` ``Stroke`` path effects,
        sometimes there can be odd holes in the whitespace around the chemical
        formula labels (see:
        https://github.com/matplotlib/matplotlib/issues/25669).
        This is only the case for ``png`` output, so saving to e.g. ``svg``
        or ``pdf`` instead will avoid this issue.

        If the heatmap interpolation looks odd (e.g. striation effects),
        generally this can be easily solved by setting ``n_points`` (via
        ``**kwargs``) to a higher value (default = 1000).

        If using the default colour map (``batlow``) in publications, please
        consider citing: https://zenodo.org/records/8409685

        Tip: https://github.com/frssp/cplapy can be used to generate 3D plots
        of chemical stability regions, to show the bordering competing phases
        in quaternary systems.

        Args:
            dependent_element (str or Element):
                The element for which the chemical potential is plotted as a
                heatmap. If None (default), the last element in the bulk
                composition formula is used (which corresponds to the most
                electronegative element present).
            fixed_elements (dict):
                A dictionary of chemical potentials to fix (in the format:
                ``{element: value}``; e.g. ``{"Li": -2}``) if the chemical
                system is >3-D. Constraining chemical potentials is required
                for multinary systems, in order to reduce the dimensionality to
                3-D for plotting a 2-D heatmap. For a system with N elements,
                N-3 fixed chemical potentials should be specified. If ``None``
                (default), the chemical potentials of the first N-3 elements in
                the bulk composition are fixed to their mean values in the
                stability region (i.e. the centroid of the stability region).
            bordering_phases (bool):
                Whether to plot the competing/secondary phases which border the
                host composition in the chemical potential diagram.
                Default is ``True``.
            xlim (tuple):
                The x-axis limits for the plot. If None (default), the limits
                are set to the minimum and maximum values of the x-axis data,
                with padding equal to ``padding`` (default = 10% of the range).
            ylim (tuple):
                The y-axis limits for the plot. If None (default), the limits
                are set to the minimum and maximum values of the y-axis data,
                with padding equal to ``padding`` (default = 10% of the range).
            cbar_range (tuple):
                The range for the colourbar. If None (default), the range is
                set to the minimum and maximum values of the data.
            colormap (str, matplotlib.colors.Colormap):
                Colormap to use for the heatmap, either as a string (which can
                be a colormap name from https://www.fabiocrameri.ch/colourmaps
                / https://matplotlib.org/stable/users/explain/colors/colormaps)
                or a ``Colormap`` / ``ListedColormap`` object. If ``None``
                (default), uses ``batlow`` from
                https://www.fabiocrameri.ch/colourmaps.

                Append "S" to the colormap name if using a sequential colormap
                from https://www.fabiocrameri.ch/colourmaps.
            padding (float):
                The padding to add to the x and y axis limits. If ``None``
                (default), the padding is set to 10% of the range.
            title (str or bool):
                The title for the plot. If ``False`` (default), no title is
                added. If ``True``, the title is set to the bulk composition
                formula, or if ``str``, the title is set to the provided
                string.
            label_positions (bool, dict or list):
                The positions for the chemical formula line labels. If ``True``
                (default), the labels are placed using a custom ``doped``
                algorithm which attempts to find the best possible positions
                (minimising overlap). If ``False``, no labels are added.
                Alternatively a dictionary can be provided, where the keys are
                the chemical formulae and the values are tuples of
                ``(x_coord, y-offset)`` at which to place the line labels
                (where y-offset is the offset from the line at ``x=x_coord``).
                A list of tuples can also be provided, where the order is
                assumed to match the competing phase lines.
            filename (PathLike):
                The filename to save the plot to. If ``None`` (default), the
                plot is not saved.
            style_file (PathLike):
                Path to a mplstyle file to use for the plot. If ``None``
                (default), uses the default ``doped`` style (from
                ``doped/utils/doped.mplstyle``).
            **kwargs:
                Additional keyword arguments to pass to
                ``ChemicalPotentialGrid.get_grid()``, such as ``n_points``
                (default = 1000) and ``cartesian`` (default = ``True`` for
                heatmap plotting, to ensure smooth interpolation).

        Returns:
            plt.Figure: The ``matplotlib`` ``Figure`` object.
        """
        return plot_chempot_heatmap(
            self.chempots,
            composition=self.composition,
            dependent_element=dependent_element,
            fixed_elements=fixed_elements,
            bordering_phases=bordering_phases,
            xlim=xlim,
            ylim=ylim,
            cbar_range=cbar_range,
            colormap=colormap,
            padding=padding,
            title=title,
            label_positions=label_positions,
            filename=filename,
            style_file=style_file,
            **kwargs,
        )

    @property
    def entries_dict(self) -> dict[str, ComputedEntry]:
        """
        Mapping of ``doped`` competing phase names to entries.
        """
        entries_dict: dict[str, ComputedEntry] = {}
        for entry in self.entries:
            doped_name = get_and_set_competing_phase_name(entry, regenerate=False)
            if doped_name in entries_dict:
                raise KeyError(
                    f"Duplicate competing phase key encountered in `self.entries`: {doped_name}. "
                    "Please regenerate entries / entry names to ensure uniqueness."
                )
            entries_dict[doped_name] = entry
        return entries_dict

    def __getattr__(self, attr: str) -> Any:
        """
        Redirect unknown attribute/method lookups to the entries dictionary.
        """
        # ``__getattr__`` is only called when normal lookup has already failed; ``entries`` is
        # accessed by ``entries_dict`` so guard against infinite recursion during partially-
        # initialised states:
        if attr == "entries":
            raise AttributeError(attr)
        return getattr(self.entries_dict, attr)

    @overload
    def __getitem__(self, key: str) -> ComputedEntry: ...

    @overload
    def __getitem__(self, key: int) -> ComputedEntry: ...

    @overload
    def __getitem__(self, key: slice) -> list[ComputedEntry]: ...

    def __getitem__(self, key: str | int | slice) -> ComputedEntry | list[ComputedEntry]:
        """
        Make the object subscriptable.

        String keys index by ``entry.data["doped_name"]`` (dict-like), while
        integer / slice keys use list-style indexing on ``self.entries``.
        """
        if isinstance(key, str):
            return self.entries_dict[key]
        return self.entries[key]

    def __contains__(self, item: str | ComputedEntry | ComputedStructureEntry) -> bool:
        """
        Return ``True`` if ``item`` is in the entries.

        For string inputs this checks ``entry.data["doped_name"]`` keys, while
        for non-strings this falls back to list-style membership in
        ``self.entries``.
        """
        if isinstance(item, str):
            return item in self.entries_dict
        return item in self.entries

    def __len__(self) -> int:
        """
        Return the number of competing phase entries.
        """
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over ``entry.data["doped_name"]`` keys.
        """
        return iter(self.entries_dict)

    def as_dict(self) -> dict:
        """
        Returns:
            JSON-serializable dict representation of
            |CompetingPhasesAnalyzer|.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "composition": self.composition.as_dict(),
            "entries": self.entries,
            "unstable_host": self.unstable_host,
            "bulk_entry": self.bulk_entry,
            "parsed_folders": self.parsed_folders,
            "calc_output_paths": self.calc_output_paths,
            "calculator": self.calculator,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompetingPhasesAnalyzer":
        """
        Reconstitute a |CompetingPhasesAnalyzer| object from a dict
        representation created using ``as_dict()``.

        Args:
            d (dict): dict representation of |CompetingPhasesAnalyzer|.

        Returns:
            |CompetingPhasesAnalyzer| object
        """
        entries = d["entries"]

        def get_entry(entry_or_dict):
            if isinstance(entry_or_dict, dict):
                try:
                    return ComputedStructureEntry.from_dict(entry_or_dict)
                except Exception:
                    return ComputedEntry.from_dict(entry_or_dict)
            return entry_or_dict

        cpa = cls(
            composition=Composition.from_dict(d["composition"]),
            entries=[get_entry(entry) for entry in entries],
        )
        cpa.unstable_host = d.get("unstable_host", cpa.unstable_host)
        cpa.bulk_entry = get_entry(d.get("bulk_entry", cpa.bulk_entry))
        cpa.parsed_folders = d.get("parsed_folders", cpa.parsed_folders)
        cpa.calc_output_paths = d.get("calc_output_paths", cpa.calc_output_paths)
        cpa.calculator = d.get("calculator", cpa.calculator)
        return cpa

    @property
    def chempot_grid(self) -> ChemicalPotentialGrid:
        """
        |ChemicalPotentialGrid| object for the chemical potential limits of the
        host composition.

        This object can be used for plotting and numerical analyses of chemical
        stability regions. See the |ChemicalPotentialGrid| class for more
        details.
        """
        return ChemicalPotentialGrid(self.chempots)

    def __repr__(self) -> str:
        """
        Returns a string representation of the |CompetingPhasesAnalyzer|
        object.
        """
        formula = self.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        joined_entry_list = "\n".join([entry.data.get("doped_name", "N/A") for entry in self.entries])
        return (
            f"doped CompetingPhasesAnalyzer for bulk composition {formula} with {len(self.entries)} "
            f"entries (in self.entries):\n{joined_entry_list}\n\n"
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )


def plot_chempot_heatmap(
    chempots: dict,
    composition: str | Composition | ComputedEntry,
    dependent_element: str | Element | None = None,
    fixed_elements: dict[str, float] | None = None,
    bordering_phases: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cbar_range: tuple[float, float] | None = None,
    colormap: str | colors.Colormap | None = None,
    padding: float | None = None,
    title: str | bool = False,
    label_positions: bool | dict[str, tuple[float, float]] | list[tuple[float, float]] = True,
    filename: PathLike | None = None,
    style_file: PathLike | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a heatmap of the chemical potentials (``chempots``) for a multinary
    system (``composition``), showing bordering secondary phases.

    In this plot, the ``dependent_element`` chemical potential is plotted
    as a heatmap over the stability region of the host composition, as a
    function of two other elemental chemical potentials on the x and y
    axes.

    3-D data is required to plot a 2-D heatmap, and so this function can be
    applied as-is for ternary systems (or binary systems with an extrinsic
    element), but for higher-dimensional systems a set of chemical potential
    constraints must be provided (as ``fixed_elements``) to project the
    chemical stability region to 3-D; see the competing phases tutorial section
    on :ref:`chemical_potentials_tutorial:Analysing and visualising the chemical potential limits`.

    Extrinsic chemical potentials are also supported; added as additional
    dimensions to the chemical potential diagram and can be used as plot axes
    (see ``dependent_element`` / ``fixed_elements``). Note that extrinsic
    chemical potentials are unbounded at the poor (low) chemical potential
    limit, and so by default the lower limit is set equal to 3 eV lower than
    the lowest chemical potential bound, rounded down to the nearest integer.
    This behavior can be controlled by setting ``xlim``, ``ylim`` and/or
    ``cbar_range``.

    Note that due to an issue with ``matplotlib`` ``Stroke`` path effects,
    sometimes there can be odd holes in the whitespace around the chemical
    formula labels (see:
    https://github.com/matplotlib/matplotlib/issues/25669).
    This is only the case for ``png`` output, so saving to e.g. ``svg``
    or ``pdf`` instead will avoid this issue.

    If the heatmap interpolation looks odd (e.g. striation effects), generally
    this can be easily solved by setting ``n_points`` (via ``**kwargs``) to a
    higher value (default = 1000).

    If using the default colour map (``batlow``) in publications, please
    consider citing: https://zenodo.org/records/8409685

    Tip: https://github.com/frssp/cplapy can be used to generate 3D plots
    of chemical stability regions, to show the bordering competing phases
    in quaternary systems.

    Args:
        chempots (dict):
            Chemical potential limits dictionary in the ``doped`` format (i.e.
            ``{"limits": [{'limit': [chempot_dict]}], ...}``) for the host
            material (``composition``).
        composition (str or |Composition| or |ComputedEntry|):
            Host material composition as a string, |Composition| object or
            |ComputedEntry|, for which to plot the chemical stability region
            (and for which ``chempots`` corresponds to).
        dependent_element (str or Element):
            The element for which the chemical potential is plotted as a
            heatmap. If None (default), the last element in the bulk
            composition formula is used (which corresponds to the most
            electronegative element present).
        fixed_elements (dict):
            A dictionary of chemical potentials to fix (in the format:
            ``{element: value}``; e.g. ``{"Li": -2}``) if the chemical
            system is >3-D. Constraining chemical potentials is required for
            multinary systems, in order to reduce the dimensionality to 3-D
            for plotting a 2-D heatmap. For a system with N elements, N-3
            fixed chemical potentials should be specified. If ``None``
            (default), the chemical potentials of the first N-3 elements in
            the bulk composition are fixed to their mean values in the
            stability region (i.e. the centroid of the stability region).
        bordering_phases (bool):
            Whether to plot the competing/secondary phases which border the
            host composition in the chemical potential diagram.
            Default is ``True``.
        xlim (tuple):
            The x-axis limits for the plot. If None (default), the limits
            are set to the minimum and maximum values of the x-axis data,
            with padding equal to ``padding`` (default = 10% of the range).
        ylim (tuple):
            The y-axis limits for the plot. If None (default), the limits
            are set to the minimum and maximum values of the y-axis data,
            with padding equal to ``padding`` (default = 10% of the range).
        cbar_range (tuple):
            The range for the colourbar. If None (default), the range is
            set to the minimum and maximum values of the data.
        colormap (str, matplotlib.colors.Colormap):
            Colormap to use for the heatmap, either as a string (which can
            be a colormap name from https://www.fabiocrameri.ch/colourmaps
            / https://matplotlib.org/stable/users/explain/colors/colormaps)
            or a ``Colormap`` / ``ListedColormap`` object. If ``None``
            (default), uses ``batlow`` from
            https://www.fabiocrameri.ch/colourmaps.

            Append "S" to the colormap name if using a sequential colormap
            from https://www.fabiocrameri.ch/colourmaps.
        padding (float):
            The padding to add to the x and y axis limits. If ``None``
            (default), the padding is set to 10% of the range.
        title (str or bool):
            The title for the plot. If ``False`` (default), no title is
            added. If ``True``, the title is set to the bulk composition
            formula, or if ``str``, the title is set to the provided
            string.
        label_positions (bool, dict or list):
            The positions for the chemical formula line labels. If ``True``
            (default), the labels are placed using a custom ``doped``
            algorithm which attempts to find the best possible positions
            (minimising overlap). If ``False``, no labels are added.
            Alternatively a dictionary can be provided, where the keys are
            the chemical formulae and the values are tuples of
            ``(x_coord, y-offset)`` at which to place the line labels
            (where y-offset is the offset from the line at ``x=x_coord``).
            A list of tuples can also be provided, where the order is
            assumed to match the competing phase lines.
        filename (PathLike):
            The filename to save the plot to. If ``None`` (default), the
            plot is not saved.
        style_file (PathLike):
            Path to a mplstyle file to use for the plot. If ``None``
            (default), uses the default ``doped`` style (from
            ``doped/utils/doped.mplstyle``).
        **kwargs:
            Additional keyword arguments to pass to
            ``ChemicalPotentialGrid.get_grid()``, such as ``n_points``
            (default = 1000) and ``cartesian`` (default = ``True`` for
            heatmap plotting, to ensure smooth interpolation).

    Returns:
        plt.Figure: The ``matplotlib`` ``Figure`` object.
    """
    # Note that we could also add option to instead plot competing phases lines coloured,
    # with a legend added giving the composition of each competing phase line (as in the SI of
    # 10.1021/acs.jpcc.3c05204; Cs2SnTiI6 notebooks), but this isn't as nice/clear, and the same effect
    # can be achieved by the user by saving to PDF without labels, and manually colouring and adding
    # a legend in a vector graphics editor (e.g. Inkscape, Affinity Designer, Adobe Illustrator, etc.).
    composition = Composition(composition)
    entries = entries_from_chempot_limits(chempots)  # intrinsic and extrinsic entries
    limits_wrt_el_refs = chempots.get("limits_wrt_el_refs", chempots.get("limits", {}))
    element_wise_min_limit = {
        el: min(limit_dict[el] for limit_dict in limits_wrt_el_refs.values())
        for el in next(iter(limits_wrt_el_refs.values()))
    }
    default_min_limit = np.floor(
        min(
            *element_wise_min_limit.values(),
            min(xlim) if xlim else 0,
            min(ylim) if ylim else 0,
            min(cbar_range) if cbar_range else 0,
        )
        - 3
    )  # floor to account for ``ChemicalPotentialDiagram`` bug; requires integer min limit
    cpd = ChemicalPotentialDiagram(entries, default_min_limit=default_min_limit)
    host_domains = cpd.domains[composition.reduced_formula]
    cpg = ChemicalPotentialGrid.from_dataframe(
        pd.DataFrame(host_domains, columns=[el.symbol for el in cpd.elements])
    )

    fixed_elements = fixed_elements or {}
    host_elements = list(composition.elements)
    extrinsic_elements = [el for el in cpd.elements if el not in host_elements]
    # ordered host-first, then extrinsic; used both for default ``dependent_element`` selection and
    # (preferentially) for auto-fixing additional chempots when needed:
    ordered_variable_elements = [
        el for el in (*host_elements, *extrinsic_elements) if el.symbol not in fixed_elements
    ]
    if dependent_element is None:  # set to last element in ``ordered_variable_elements``, either an
        # extrinsic element (if present) or the most electronegative anion in the bulk composition:
        dependent_element = next(el for el in reversed(host_elements) if el.symbol not in fixed_elements)
    elif isinstance(dependent_element, str):
        dependent_element = Element(dependent_element)
    assert isinstance(dependent_element, Element)  # typing
    ordered_variable_elements.remove(dependent_element)

    # check dimensionality:
    if len(cpd.elements) == 2:  # switch to line plot
        raise ValueError(
            "Chemical potential heatmap (i.e. 2D) plotting is not possible for a binary system! You "
            "can use ``cpd = ChemicalPotentialDiagram(cpa.entries / "
            "entries_from_chempot_limits(chempots)); cpd.get_plot()`` to generate a line plot of the "
            "chemical potentials as shown in the doped competing phases tutorial."
        )
    if len(cpd.elements) - len(fixed_elements) != 3:  # auto fix to centroid of stability region:
        info_message = (
            f"Chemical potential heatmap plotting requires 3-D data, requiring fixed chemical "
            f"potential constraints for >ternary systems; such that the number of elements in the "
            f"chemical system ({len(cpd.elements)}) minus the number of fixed chemical potentials "
            f"({len(fixed_elements)}) must be equal to 3."
        )
        if len(cpd.elements) - len(fixed_elements) < 3:
            raise ValueError(info_message)

        # use centroid in the user-constrained subspace to auto-fix additional elements as necessary:
        centroid = cpg.get_grid(
            cartesian=True, include_vertices=False, fixed_elements=fixed_elements or None
        ).mean(axis=0)
        # additional constraints required to reduce to 3-D (after existing ``fixed_elements``):
        req_num_constraints = len(cpd.elements) - len(fixed_elements) - 3
        additional_fixed_elements = {
            el.symbol: round(centroid[el.symbol], 4)
            for el in ordered_variable_elements[:req_num_constraints]
        }
        print(
            f"{info_message} The following chemical potentials will additionally be constrained to "
            f"their mean (centroid) values in the chemical stability region: "
            f"{additional_fixed_elements}"
        )
        fixed_elements = {**fixed_elements, **additional_fixed_elements}

    independent_elts = [
        el for el in cpd.elements if el.symbol not in fixed_elements and el != dependent_element
    ]

    # Generate grid data. Use a Cartesian (uniform) grid by default: barycentric grid sampling places
    # points unevenly across narrow / elongated regions of the host stability polygon, which can cause
    # streaking artifacts under triangle-based interpolation in ``tripcolor``:
    grid_kwargs: dict[str, Any] = {"cartesian": True, "fixed_elements": fixed_elements}
    grid_kwargs.update(kwargs)
    grid_data = cpg.get_grid(**grid_kwargs)
    values_inside = grid_data[dependent_element.symbol].to_numpy()
    points_inside = grid_data.drop(  # only independent (X) points, no dependent or fixed elements
        columns=[*list(fixed_elements.keys()), dependent_element.symbol]
    ).to_numpy()
    tri = Triangulation(points_inside[:, 0], points_inside[:, 1])

    # Create plot
    with doped_plot_style(style_file):
        fig, ax = plt.subplots()
        vmin = cbar_range[0] if cbar_range else None
        vmax = cbar_range[1] if cbar_range else None
        if vmax is None and np.isclose(values_inside.max(), 0, atol=3e-2):  # extend to 0 if close
            vmax = 0

        cmap = get_colormap(colormap, default="batlow")
        dep_mu = ax.tripcolor(
            tri,
            values_inside,
            rasterized=True,
            cmap=cmap,
            shading="gouraud",  # smooth
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(dep_mu)

        # Set plot limits and labels
        xmax, ymax = points_inside.max(axis=0)
        orig_xmin, orig_ymin = points_inside.min(axis=0)

        if xlim is None:
            x_padding = padding or 0.1 * abs(xmax - orig_xmin)
            if np.isclose(orig_xmin, default_min_limit):  # default_min_limit -> unbounded extrinsic
                xmin = element_wise_min_limit[independent_elts[0].symbol] - 3  # 3 eV below lowest limit
                x_padding = padding or 0.1 * abs(xmax - xmin)  # padding based on updated xmin
            else:
                xmin = orig_xmin - x_padding
            xlim = (float(xmin), float(xmax + x_padding))

        if ylim is None:
            y_padding = padding or 0.1 * abs(ymax - orig_ymin)
            if np.isclose(orig_ymin, default_min_limit):  # default_min_limit -> unbounded extrinsic
                ymin = element_wise_min_limit[independent_elts[1].symbol] - 3  # 3 eV below lowest limit
                y_padding = padding or 0.1 * abs(ymax - ymin)  # padding based on updated ymin
            else:
                ymin = orig_ymin - y_padding
            ylim = (float(ymin), float(ymax + y_padding))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        cbar.set_label(rf"$\Delta\mu$ ({dependent_element.symbol}) (eV)")
        ax.set_xlabel(rf"$\Delta\mu$ ({independent_elts[0].symbol}) (eV)")
        ax.set_ylabel(rf"$\Delta\mu$ ({independent_elts[1].symbol}) (eV)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        if title:  # add title
            title_str = title if isinstance(title, str) else latexify(f"{composition.reduced_formula}")
            ax.set_title(title_str)

        if bordering_phases:
            _plot_competing_phase_lines(  # plot competing phase lines and labels
                composition, ax, cpd, fixed_elements, independent_elts, label_positions
            )
            _nudge_labels_inside_axes(ax, padding)  # adjust label positions to stay within plot bounds

        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=600)

    return fig


def _plot_competing_phase_lines(
    composition: Composition,
    ax: plt.Axes,
    cpd: ChemicalPotentialDiagram,
    fixed_elements: dict[str, float],
    independent_elts: list[Element],
    label_positions: bool | dict[str, tuple[float, float]] | list[tuple[float, float]] = True,
) -> None:
    """
    Plot competing phase lines and add labels.
    """
    lines = {}  # {formula: matplotlib line object}
    intersections = []
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for formula, pts in cpd.domains.items():
        x = np.linspace(-50, 50, 1000)
        if formula == composition.reduced_formula or set(Composition(formula).elements).issubset(
            {Element(el) for el in fixed_elements}
        ):  # skip host or fixed elemental phases
            continue

        # Get domain points that match host domains
        host_domains = cpd.domains[composition.reduced_formula]
        domain_pts = np.array(
            [
                chempot_coords
                for chempot_coords in pts
                if np.any(np.all(np.isclose(host_domains, chempot_coords), axis=1))  # (M, k)
            ]
        )
        if domain_pts.size < 2:
            continue  # not a stable bordering phase

        try:
            for element, value in fixed_elements.items():
                domain_pts = _intersect_hull_with_plane(
                    domain_pts,
                    cpd.elements.index(Element(element)),
                    value,
                )  # handle fixed elements by intersecting with planes
        except ValueError:
            continue  # no intersection with plane, skip to next phase

        # Fit line function, plot and get intersections:
        formula_x_vals = domain_pts[:, cpd.elements.index(independent_elts[0])]
        formula_y_vals = domain_pts[:, cpd.elements.index(independent_elts[1])]
        intersection: np.ndarray | None  # typing
        if np.isclose(min(formula_x_vals), max(formula_x_vals), atol=5e-5):  # vertical line
            x = formula_x_vals[0]
            line = ax.axvline(x, label=latexify(formula), color="k")
            intersection = np.array(((x, ymin), (x, ymax))) if (x < xmax and x > xmin) else None
        else:
            m, b = np.polyfit(formula_x_vals, formula_y_vals, 1)

            def f(xx, m=m, b=b):  # line function for the fitted line
                return m * xx + b

            (line,) = ax.plot(x, f(x), label=latexify(formula), color="k")
            intersection = _get_line_intersections(f, (xmin, xmax), (ymin, ymax))

        if intersection is not None and np.size(intersection) >= 4:
            if np.size(intersection) == 4:
                # if 4 intersections, check if the midpoint of intersections is at an edge (skip if so)
                midpoint = np.mean(intersection, axis=0)
                if np.any(np.abs(midpoint - np.array([xmin, ymin])) < 1e-4) or np.any(
                    np.abs(midpoint - np.array([xmax, ymax])) < 1e-4
                ):
                    continue

            intersections.append(intersection)
            lines[formula] = line

    if label_positions:  # add labels to lines
        _add_line_labels(
            intersections=intersections,
            lines=lines,  # {formula: matplotlib line object}
            x_range=abs(xmax - xmin),
            y_range=abs(ymax - ymin),
            label_positions=label_positions,
        )


def _get_line_intersections(
    f: Callable, xlim: tuple[float, float], ylim: tuple[float, float]
) -> np.ndarray:
    """
    Get intersections of a line with the plot bounding box.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    x_intersections = []
    y_intersections = []

    # Check intersections with vertical bounds (xmin and xmax)
    y_at_xmin = f(xmin)
    y_at_xmax = f(xmax)
    if ymin <= y_at_xmin <= ymax:
        x_intersections.append((xmin, float(y_at_xmin)))
    if ymin <= y_at_xmax <= ymax:
        x_intersections.append((xmax, float(y_at_xmax)))

    # Check intersections with horizontal bounds (ymin and ymax)
    if not np.isclose(float(y_at_xmin), float(y_at_xmax), atol=1e-4):  # not a horizontal line

        def inv_f(yy):  # inverse of the line function
            return (yy - f(0)) / (f(1) - f(0)) if f(1) != f(0) else xmin

        x_at_ymin = inv_f(ymin)
        x_at_ymax = inv_f(ymax)
        if xmin <= x_at_ymin <= xmax:
            y_intersections.append((float(x_at_ymin), ymin))
        if xmin <= x_at_ymax <= xmax:
            y_intersections.append((float(x_at_ymax), ymax))

    # take unique points in case intersects at x/y corner (which would give a duplicate):
    return np.unique(np.round((x_intersections + y_intersections), 4), axis=0)


def _add_line_labels(
    intersections: Sequence[np.ndarray],
    lines: dict[str, plt.Line2D],
    x_range: float,
    y_range: float,
    label_positions: bool | dict[str, tuple[float, float]] | list[tuple[float, float]] = True,
) -> None:
    """
    Add labels to the competing phase lines.
    """
    if label_positions is True:  # use custom doped algorithm
        poss_label_positions = _possible_label_positions_from_bbox_intersections(intersections)
        plot_label_positions, best_norm_min_dist = _find_best_label_positions(
            poss_label_positions, x_range=x_range, y_range=y_range, return_best_norm_dist=True
        )
        if best_norm_min_dist < 0.1:  # bump positions_per_line to 5 to try improve:
            poss_label_positions = _possible_label_positions_from_bbox_intersections(
                intersections, positions_per_line=5
            )
            plot_label_positions, best_norm_min_dist = _find_best_label_positions(
                poss_label_positions, x_range=x_range, y_range=y_range, return_best_norm_dist=True
            )

    elif isinstance(label_positions, dict):  # pre-set label positions, match formula (key) to line:
        lines = {k: lines[k] for k in lines if k in label_positions}  # drop any without positions
        plot_label_positions = [label_positions[k] for k in lines]  # reorder to match lines

    if isinstance(label_positions, list):
        plot_label_positions = np.array(label_positions, dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The value at position")
        labelLines(
            list(lines.values()),  # must be in same order as plotting order...
            xvals=plot_label_positions[:, 0],
            yoffsets=plot_label_positions[:, 1],
            align=False,
            color="black",
        )


def _nudge_labels_inside_axes(ax: plt.Axes, padding: float | None) -> None:
    """
    Adjust label positions to ensure they stay within plot bounds.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_padding = padding or (xlim[1] - xlim[0]) * 0.1
    y_padding = padding or (ylim[1] - ylim[0]) * 0.1

    latexified_labels = {
        artist.get_text(): artist for artist in ax.get_children() if isinstance(artist, plt.Text)
    }
    for text in latexified_labels.values():
        bbox = text.get_window_extent().transformed(ax.transData.inverted())
        new_position = text.get_position()
        delta_x = delta_y = 0.0

        if bbox.xmin < xlim[0] or bbox.xmax > xlim[1] or bbox.ymin < ylim[0] or bbox.ymax > ylim[1]:
            if bbox.xmin < xlim[0]:  # shift right if label starts before xmin
                delta_x = (xlim[0] - bbox.xmin) + x_padding * 0.25
            elif bbox.xmax > xlim[1]:  # shift left if label ends after xmax
                delta_x = (xlim[1] - bbox.xmax) - x_padding * 0.25

            if bbox.ymin < ylim[0]:  # shift up if label starts before ymin
                delta_y = (ylim[0] - bbox.ymin) + y_padding * 0.25
            if bbox.ymax > ylim[1]:  # shift down if label ends after ymax
                delta_y = (ylim[1] - bbox.ymax) - y_padding * 0.25

            # Apply adjustments while keeping within bounds
            if delta_x != 0 and new_position[0] + delta_x < xlim[1]:
                new_position = (new_position[0] + delta_x, new_position[1])
            if delta_y != 0 and new_position[1] + delta_y < ylim[1]:
                new_position = (new_position[0], new_position[1] + delta_y)

        text.set_position(new_position)


def _parse_entry_and_catch_exception(
    calc_output_path: PathLike, calculator: str = "vasp"
) -> tuple[str | ComputedStructureEntry, PathLike, bool, bool]:
    r"""
    Parse a competing phase calculation output into a
    |ComputedStructureEntry|\s (using the ``get_competing_phase_entry()``
    function of the ``doped.io`` backend for ``calculator``), catching any
    exceptions and returning the error message and the calculation output path
    if an exception is raised.

    Returns the entry, the folder it was parsed from, and whether electronic
    and ionic convergence were reached.
    """
    try:
        entry = get_backend(calculator).get_competing_phase_entry(calc_output_path)
        return (
            entry,
            entry.data.get("folder", str(calc_output_path)),
            entry.data.get("converged_electronic", True),  # default True (i.e. no warning) if the
            entry.data.get("converged_ionic", True),  # backend does not report convergence info
        )
    except Exception as e:
        return str(e), calc_output_path, False, False


def _possible_label_positions_from_bbox_intersections(
    intersections: Sequence[NDArray[np.floating[Any]]] | NDArray[np.floating[Any]],
    positions_per_line: int = 3,
) -> NDArray[np.floating[Any]]:
    """
    From a list or array of ``intersections``, which contains the intersections
    of lines with a plot bounding box (limits) and thus has shape ``(N_lines,
    2, 2)`` (because 2 intersections with 2 (x,y) coordinates per line),
    returns ``positions_per_line`` uniformly-spaced possible x,y coordinates
    for labels of those lines.

    e.g. if ``positions_per_line = 3`` (default), then returns the x,y
    coordinates for the positions which are 1/4, 2/4 and 3/4 along the line
    between the two bbox intersections.

    Args:
        intersections (list or np.ndarray):
            A list or array of intersections of lines with the plot bounding
            box. Should have shape ``(N_lines, 2, 2)``.
        positions_per_line (int):
            The number of possible label positions per line to return. Default
            is 3, which returns positions at 1/4, 2/4 and 3/4 along the line
            between the two bbox intersections.

    Returns:
        np.ndarray:
            The possible label positions, with shape
            ``(N_lines, positions_per_line, 2)``.
    """
    poss_label_positions = np.zeros((len(intersections), positions_per_line, 2))
    for label_idx, points in enumerate(intersections):  # get possible label positions
        for line_pos_idx in range(positions_per_line):
            first_pt_factor = ((positions_per_line + 1) - (line_pos_idx + 1)) / (positions_per_line + 1)
            second_pt_factor = 1 - first_pt_factor
            poss_label_positions[label_idx, line_pos_idx, 0] = (points[0][0] * first_pt_factor) + (
                points[1][0] * second_pt_factor
            )
            poss_label_positions[label_idx, line_pos_idx, 1] = (points[0][1] * first_pt_factor) + (
                points[1][1] * second_pt_factor
            )

    return poss_label_positions


def _find_best_label_positions(
    poss_label_positions: NDArray[np.floating[Any]],
    x_range: float = 1,
    y_range: float = 1,
    return_best_norm_dist: bool = False,
) -> NDArray[np.floating[Any]] | tuple[NDArray[np.floating[Any]], float]:
    """
    From an array of possible label positions, find the best possible
    combination of label positions which maximises the distance between labels
    (i.e. minimises overlap).

    Args:
        poss_label_positions (np.ndarray):
            The possible label positions, with shape
            ``(N_lines, positions_per_line, 2)``.
        x_range (float):
            The range of the x-axis to use for normalisation. Default is 1.
        y_range (float):
            The range of the y-axis to use for normalisation. Default is 1.
        return_best_norm_dist (bool):
            Whether to return the best normalised minimum distance between
            labels. Default is ``False``.

    Returns:
        np.ndarray:
            The best possible label positions, with shape ``(N_lines, 2)``,
            where the second dimension is (x,y-offset) coordinates.
        float:
            The best normalised minimum distance between labels,
            if ``return_best_norm_dist`` is True.
    """
    # Get all possible combinations of indices, for the first two dimensions (N_labels,
    # N_possibilities_per_label):
    N_labels, N_possibilities_per_label, N_xy = poss_label_positions.shape
    combinations = list(itertools.product(range(N_possibilities_per_label), repeat=N_labels))

    # Prepare an empty array to store the results
    all_combos = np.zeros((len(combinations), N_labels, N_xy))  # N_xy should be 2

    # Fill the result array with the corresponding coordinates
    for i, combo in enumerate(combinations):
        all_combos[i] = poss_label_positions[np.arange(N_labels), combo]

    #  all_combos.shape should be (N_possibilities_per_label**N_labels, N_labels, N_xy = 2)
    all_combos[:, :, 0] /= x_range
    all_combos[:, :, 1] /= y_range
    dists = np.linalg.norm(all_combos[:, :, np.newaxis] - all_combos[:, np.newaxis, :], axis=-1)
    # get upper diagonal of distances (removes self-distances = 0 and duplicates ((i,j) = (j,i))
    mask = np.triu(np.ones((N_labels, N_labels)), k=1).astype(bool)
    unique_dists = dists[:, mask]
    dists_list = [sorted(sublist) for sublist in unique_dists.tolist()]
    max_idx = dists_list.index(sorted(dists_list, reverse=True)[0])
    best_combo = all_combos[max_idx]
    best_combo[:, 0] *= x_range
    best_combo[:, 1] *= y_range  # reverse normalisation

    # set y-offset to 0 except for vertical lines:
    for i in range(N_labels):
        if poss_label_positions[i, 0, 0] == poss_label_positions[i, -1, 0]:  # vertical line
            best_combo[i, 1] -= np.mean(
                poss_label_positions[i, :, 1]
            )  # y_offset relative to midpoint (default for vertical lines)
        else:
            best_combo[i, 1] = 0  # zero y-offset

    if return_best_norm_dist:
        return best_combo, dists_list[max_idx][0]

    return best_combo


def get_X_rich_poor_limit(
    X: str,
    chempots: dict,
    rich: bool = True,
    warn_if_multiple: bool = True,
    tol: float = 0.01,
    bulk_composition: str | Composition | None = None,
) -> str:
    """
    Determine the chemical potential limit of the input ``chempots`` dict with
    extremal μ_X -- maximum μ_X for X-rich (``rich = True``; default) or
    minimum μ_X for X-poor (``rich = False``).

    If there are multiple chemical potential limits with the same extremal μ_X
    within an energy tolerance ``tol`` in eV (0.01 eV by default), a warning is
    thrown (if ``warn_if_multiple`` is ``True``, default) and the tie is broken
    by firstly sorting the other elements by:

    - Whether or not they are in the bulk composition, prioritising elements in
      the bulk composition (this aids determinism in this function output).
    - Electronegativity similarity to the extremal element ``X``, with more
      electronegatively-similar elements prioritised (such that in e.g. a
      multi-cation composition ``(A,B)Z``, 'A-rich' here will resolve to the
      A-rich limit with the most B-rich chemical potential -- i.e. the most
      'cation-rich' limit).

    Then, for each element Y in the sorted list, the ``max`` / ``min`` (for
    rich / poor respectively) of the μ_Y values among the still-tied limits is
    taken as μ_Y^*, and only limits with ``|μ_Y - μ_Y^*| < tol`` are kept. The
    process repeats until one limit remains, which is then returned. This is
    mostly equivalent to choosing the lexicographic ``max`` / ``min``
    (μ_Y, ...) tuple for X-rich/poor respectively, with a numerical tolerance
    ``tol``.

    This implementation is similar to that of
    ``FormationEnergyDiagram.get_chempots()`` in ``pymatgen.analysis.defects``,
    but using electronegativity instead of electron affinity.

    Args:
        X (str):
            Element symbol for the extremum species (e.g. "Te"), or a string as
            ``"X-rich"`` or ``"X-poor"`` where ``X`` is the extremum element --
            ``rich`` will be set automatically in this latter case (e.g.
            ``rich`` will be set to ``False`` for ``"Te-poor"``).
        chempots (dict):
            The chemical potential limits dict, as returned by
            ``CompetingPhasesAnalyzer.chempots`` (i.e. containing
            ``"limits"`` mapping limit names to ``{element: absolute μ}``).
        rich (bool):
            Whether to use the maximum μ_X (X-rich; ``True``) or minimum μ_X
            (X-poor; ``False``). Default is ``True``.
            Note that this setting will be overwritten if the ``X`` input is
            given as a ``"X-rich"`` / ``"X-poor"`` string (e.g. ``rich`` will
            be set to ``False`` for ``X = "Te-poor"``).
        warn_if_multiple (bool):
            Whether to warn if there are multiple chemical potential limits
            with the same extremal μ_X within an energy tolerance ``tol`` in
            eV. Default is ``True``.
        tol (float):
            Energy tolerance in eV. Limits whose μ_X satisfies
            ``|μ_X - μ_X^*| < tol``, with μ_X^* being the extremal value, are
            treated as tied. Default is 0.01 eV.
        bulk_composition (str | |Composition| | None):
            Host composition for intrinsic-vs-extrinsic ordering in ties. If
            ``None`` (default), auto-determines the bulk composition from the
            ``chempots`` dict (i.e. the composition which is present for each
            limit).

    Returns:
        str:
            The key in ``chempots["limits"]`` for the chosen limit.
    """
    # parse X:
    if "rich" in X.lower() or "poor" in X.lower():
        rich = "rich" in X.lower()
        X = X.split("-", maxsplit=1)[0]
    else:
        try:
            _ref = Element(X)
        except Exception as exc:
            raise ValueError(
                f"Invalid input for X: {X}. Must be an element symbol or a string as 'X-rich' or 'X-poor' "
                f"where X is the extremum element (e.g. 'Te-rich' or 'Te-poor')."
            ) from exc

    limits: dict[str, dict[str, float]] = chempots["limits"]  # same result with limits_wrt_el_refs
    X_chempots = [(limit, limit_dict[X]) for limit, limit_dict in limits.items() if X in limit_dict]
    if not X_chempots:
        raise ValueError(f"Could not find {X} in the chemical potential limits dict:\n{chempots}")

    ref = Element(X)

    def pauling_similar_first(sym: str) -> float:
        a, b = Element(sym).X, ref.X  # electronegativities
        if a is None or b is None:
            return np.inf  # no electronegativity data available
        af, bf = float(a), float(b)
        return np.inf if np.isnan(af) or np.isnan(bf) else abs(af - bf)

    target = (max if rich else min)(chempot for _limit, chempot in X_chempots)
    tied = [limit for limit, chempot in X_chempots if abs(chempot - target) < tol]
    if len(tied) == 1:
        return next(iter(tied))  # only one limit with the same extremal μ_X, return it

    symbols = set().union(*(limits[limit] for limit in tied))
    if bulk_composition is None and len(chempots["limits"]) > 1:  # auto-determine bulk composition
        # (outside of the edge case of only one limit -- e.g. chemically unstable materials; can't
        # auto-determine in those cases so we skip bulk composition filtering)
        # bulk composition is the only one present in every limit (limits are "X-Y-Z" strings):
        limit_comps = [set(limit.split("-")) for limit in chempots["limits"]]
        bulk_composition = next(iter(set.intersection(*limit_comps)))

    # sort by pauling EN similarity first, then alphabetically (to aid determinism):
    def _sort_key(sym: str) -> tuple[float, str]:
        return pauling_similar_first(sym), sym

    if bulk_composition is not None:
        bulk = {element.symbol for element in Composition(bulk_composition).elements}
        intr = sorted((element for element in symbols if element in bulk and element != X), key=_sort_key)
        extr = sorted((element for element in symbols if element not in bulk), key=_sort_key)
        el_order = intr + extr
    else:
        el_order = sorted((element for element in symbols if element != X), key=_sort_key)

    orig_tied = tied.copy()
    for element in el_order:
        extremal = (max if rich else min)(limits[lim][element] for lim in tied)
        tied = [lim for lim in tied if abs(limits[lim][element] - extremal) < tol]  # overwrites ``tied``
        if len(tied) == 1:
            break

    if len(tied) > 1:
        # edge case handling; should very rarely get to this point (unless dealing w/tiny chempot ranges)
        def mus_tuple(limit: str) -> tuple[float, ...]:
            return tuple(limits[limit][element] for element in el_order)

        return_limit = (max if rich else min)(sorted(tied), key=lambda lim: (mus_tuple(lim), lim))
    else:
        return_limit = next(iter(tied))

    if warn_if_multiple:
        tied_list = ", ".join(sorted(orig_tied))
        min_max_str = "maximum" if rich else "minimum"
        rich_poor_str = "rich" if rich else "poor"
        warnings.warn(
            f"Multiple chemical potential limits are degenerate within tol={tol:.3f} eV for the "
            f"{min_max_str} ({rich_poor_str}) μ_{X}: [{tied_list}]. Choosing the most {rich_poor_str} "
            f"limit for the most electronegatively-similar element(s) to {X} of all tied limits; giving "
            f"{return_limit} (see ``get_X_rich_poor_limit`` docstring for details). If using the "
            f"thermodynamics functions, one can input the specific limit (e.g. 'Cd-CdTe' instead of "
            f"'Cd-rich') to avoid this warning."
        )

    return return_limit


def get_X_rich_limit(X: str, chempots: dict, **kwargs) -> str:
    """
    Deprecated alias for ``get_X_rich_poor_limit(..., rich=True)``.

    Will be removed in ``doped`` 4.1; use :func:`get_X_rich_poor_limit` (with
    ``rich=True``) instead.
    """
    warnings.warn(
        "`get_X_rich_limit` is deprecated and will be removed in doped 4.1; use "
        "`get_X_rich_poor_limit(X, chempots, rich=True, ...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_X_rich_poor_limit(X, chempots, rich=True, **kwargs)


def get_X_poor_limit(X: str, chempots: dict, **kwargs) -> str:
    """
    Deprecated alias for ``get_X_rich_poor_limit(..., rich=False)``.

    Will be removed in ``doped`` 4.1; use :func:`get_X_rich_poor_limit` (with
    ``rich=False``) instead.
    """
    warnings.warn(
        "`get_X_poor_limit` is deprecated and will be removed in doped 4.1; use "
        "`get_X_rich_poor_limit(X, chempots, rich=False, ...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_X_rich_poor_limit(X, chempots, rich=False, **kwargs)
