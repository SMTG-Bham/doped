"""
Functions for setting up and parsing competing phase calculations in order to
determine and analyse the elemental chemical potentials for defect formation
energies.
"""

import contextlib
import copy
import itertools
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from pathlib import Path
from re import sub
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labellines import labelLines
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.tri import Triangulation
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    ComputedStructureEntry,
    ConstantEnergyAdjustment,
    ManualEnergyAdjustment,
)
from pymatgen.ext.matproj import MPRester, MPRestError
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning, Vasprun
from pymatgen.util.string import latexify, latexify_spacegroup
from pymatgen.util.typing import PathLike
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from doped import _doped_obj_properties_methods, _ignore_pmg_warnings, get_mp_context, pool_manager
from doped.generation import _element_sort_func
from doped.utils.parsing import (
    _compare_incar_tags,
    _compare_potcar_symbols,
    _format_mismatching_incar_warning,
    _get_output_files_and_check_if_multiple,
    get_magnetization_from_vasprun,
    get_vasprun,
)
from doped.utils.plotting import get_colormap
from doped.utils.symmetry import _custom_round, _round_floats, get_primitive_structure
from doped.vasp import MODULE_DIR, DopedDictSet, default_HSE_set, default_relax_set

# globally ignore:
_ignore_pmg_warnings()

pbesol_convrg_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/PBEsol_ConvergenceSet.yaml"))  # just INCAR

elemental_diatomic_bond_lengths = {"H": 0.74, "O": 1.21, "N": 1.10, "F": 1.42, "Cl": 1.99}

# TODO: Need to recheck all functionality from old `_chemical_potentials.py` is now present here.

MPRester_property_data = [  # properties to pull for Materials Project entries
    "formula_pretty",
    "energy_above_hull",
    "nsites",
    "volume",
    "formation_energy_per_atom",  # note that this is the corrected formation energy
    "energy_per_atom",  # note that this is the corrected energy per atom
    "nelements",
    "elements",
]
MPRester_summary_data = [
    "band_gap",
    "total_magnetization",
    "theoretical",
    "database_IDs",  # dict, possibly with an "icsd" key with list of ICSD entry codes
]

default_get_entries_kwargs = {
    "property_data": MPRester_property_data,
    "summary_data": MPRester_summary_data,
}


def make_molecule_in_a_box(element: str):
    """
    Generate an X2 'molecule-in-a-box' structure for the input element X, (i.e.
    a 30 Å cuboid supercell with a single X2 molecule in the centre).

    This is the natural state of several elemental competing phases, such as
    O2, N2, H2, F2 and Cl2. Initial bond lengths are set to the experimental
    bond lengths of these gaseous molecules.

    Args:
        element (str):
            Element symbol of the molecule to generate.

    Returns:
        Structure, formula and total magnetization:

        structure (Structure):
            ``pymatgen`` ``Structure`` object of the molecule in a box.
        formula (str):
            Chemical formula of the molecule in a box.
        total_magnetization (int):
            Total magnetization of the molecule in a box
            (0 for all X2 except O2 which has a triplet ground state (S = 1)).
    """
    element = sub(r"\d+$", "", element)  # remove digits in case provided as X2 etc
    if element not in elemental_diatomic_bond_lengths:
        raise ValueError(
            f"Element {element} is not currently supported for molecule-in-a-box structure generation."
        )

    lattice = np.array([[30.01, 0, 0], [0, 30.00, 0], [0, 0, 29.99]])
    bond_length = elemental_diatomic_bond_lengths[element]
    structure = Structure(
        lattice=lattice,
        species=[element, element],
        coords=[[15, 15, 15], [15, 15, 15 + bond_length]],
        coords_are_cartesian=True,
    )
    total_magnetization = 0 if element != "O" else 2  # O2 has a triplet ground state (S = 1)

    return structure, total_magnetization


def make_molecular_entry(computed_entry: ComputedEntry) -> ComputedStructureEntry:
    """
    Generate a new ``ComputedStructureEntry`` for a molecule in a box, for the
    input elemental ``ComputedEntry``.

    The formula of the input ``computed_entry`` must be one of the supported
    diatomic molecules (O2, N2, H2, F2, Cl2).

    Args:
        computed_entry (ComputedEntry):
            ``ComputedEntry`` object for the elemental entry.

    Returns:
        ComputedStructureEntry:
            ``ComputedStructureEntry`` for the molecule in a box.
    """
    assert len(computed_entry.composition.elements) == 1  # Elemental!
    formula = computed_entry.data.get("formula_pretty", "N/A")
    element = computed_entry.composition.elements[0].symbol
    struct, total_magnetization = make_molecule_in_a_box(element)
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
        "total_magnetization": total_magnetization,
        "theoretical": False,
        "database_IDs": {},
    }

    return molecular_entry


def _renormalise_entry(entry, renormalisation_energy_per_atom, name=None, description=None):
    """
    Regenerate the input entry (``ComputedEntry``/``ComputedStructureEntry``)
    with an energy per atom `decreased` by ``renormalisation_energy_per_atom``,
    by appending an ``EnergyAdjustment`` object to
    ``entry.energy_adjustments``.

    Args:
        entry (ComputedEntry/ComputedStructureEntry):
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
        ComputedEntry/ComputedStructureEntry: Renormalised entry.
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
) -> dict:
    """
    Get the chemical potential limits for the bulk computed entry in the
    supplied phase diagram.

    Args:
        bulk_computed_entry (ComputedEntry):
            ``ComputedEntry``/``ComputedStructureEntry`` object for the host
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


def _get_all_chemsyses(chemsys: str | list[str]):
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
    **kwargs,
):
    r"""
    Convenience function to get a list of ``ComputedStructureEntry``\s for an
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
            to obtain the corresponding ``ComputedStructureEntry``s. If not
            supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml``) -- see the ``doped`` Installation docs:
            doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials-project-api
        energy_above_hull (float):
            If supplied, only entries with energies above hull (according to
            the MP-computed phase diagram) less than this value (in eV/atom)
            will be returned. Set to 0 to only return phases on the MP convex
            hull. Default is ``None`` (i.e. all entries are returned).
        bulk_composition (str/Composition):
            Optional input; formula of the bulk host material, to use for
            sorting the output entries (with all those matching the bulk
            composition first). Default is ``None``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            ``get_entries_in_chemsys()`` query.

    Returns:
        list[ComputedStructureEntry], dict, list:
            List of ``ComputedStructureEntry`` objects for the input chemical
            system.
    """
    with MPRester(api_key) as mpr:
        # get all entries in the chemical system
        MP_full_pd_entries = mpr.get_entries_in_chemsys(
            chemsys,
            **default_get_entries_kwargs,
            **kwargs,
        )

    temp_phase_diagram = PhaseDiagram(MP_full_pd_entries)
    for entry in MP_full_pd_entries:
        # reparse energy above hull, to avoid mislabelling issues noted in Materials Project database
        # (mostly only the legacy database, so should be resolved with the new MP API database now);
        # e.g. search "F", or ZnSe2 on Zn-Se convex hull from MP PD, but EaH = 0.147 eV/atom?
        # or Immm phases for Br, I..., Na2FePO4F...
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
    **kwargs,
):
    r"""
    Convenience function to get a list of ``ComputedStructureEntry``\s for an
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
            to obtain the corresponding ``ComputedStructureEntry``s. If not
            supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml``) -- see the ``doped`` Installation docs:
            doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials-project-api
        bulk_composition (str/Composition):
            Optional input; formula of the bulk host material, to use for
            sorting the output entries (with all those matching the bulk
            composition first). Default is ``None``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            ``get_entries()`` query.

    Returns:
        list[ComputedStructureEntry]:
            List of ``ComputedStructureEntry`` objects for the input chemical system.
    """
    with MPRester(api_key) as mpr:
        entries = mpr.get_entries(
            chemsys_formula_id_criteria,
            **default_get_entries_kwargs,
            **kwargs,
        )

    # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
    entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=bulk_composition))

    return entries


def _parse_MP_API_key(api_key: str | None = None):
    """
    Parse the Materials Project (MP) API key, either from the input argument or
    from the environment variable ``PMG_MAPI_KEY``, throwing an error if it is
    not valid.

    Args:
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            (for phase diagram analysis, competing phase generation etc). If
            not supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml``) -- see the ``doped`` Installation docs:
            doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials-project-api

    Returns:
        api_key (str):
            Materials Project API key, as supplied in the input argument or
            read from the environment variable.
    """
    api_key = api_key or SETTINGS.get("PMG_MAPI_KEY")

    # check api_key:
    if api_key is None:  # no API key supplied or set in ``.pmgrc.yaml``
        raise ValueError(
            "No API key (either ``api_key`` parameter or 'PMG_MAPI_KEY' in ``~/.pmgrc.yaml`` or "
            "``~/.config/.pmgrc.yaml``) was supplied. This is required for automatic competing "
            "phase generation in doped, as detailed on the installation instructions:\n"
            "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
            "-project-api"
        )
    if len(api_key) < 15 or len(api_key) > 20:  # looks like an invalid API key; check:
        try:
            with MPRester(api_key) as mpr:
                mpr.get_entry_by_material_id("mp-1")  # check if API key is valid
        except MPRestError as mp_exc:
            raise ValueError(
                f"The supplied API key (either ``api_key`` or 'PMG_MAPI_KEY' in ``~/.pmgrc.yaml`` or "
                f"``~/.config/.pmgrc.yaml``; {api_key}) is not a valid Materials Project API "
                f"key, which is required by doped for competing phase generation. See the doped "
                f"installation instructions for details:\n"
                "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
                "-project-api"
            ) from mp_exc

    return api_key


def get_MP_summary_dicts(
    entries: list[ComputedEntry] | None = None,
    chemsys: str | list[str] | None = None,
    api_key: str | None = None,
    **kwargs,
) -> dict[str, dict]:
    r"""
    Get the corresponding Materials Project (MP) summary dictionaries for
    computed entries in the input ``entries`` list or ``chemsys`` chemical
    system.

    If ``entries`` is provided (which should be a list of ``ComputedEntry``s
    from the Materials Project), then only summary dictionaries in this
    chemical system which match one of these entries (based on the MPIDs given
    in ``ComputedEntry.entry_id``/``ComputedEntry.data["material_id"]`` and
    ``summary_dict["material_id"]``) are returned.

    Args:
        entries (list[ComputedEntry]):
            Optional input; list of ``ComputedEntry`` objects for the input
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
            supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or
            ``~/.config/.pmgrc.yaml``) -- see the ``doped`` Installation docs:
            doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials-project-api
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            query, e.g. ``MPRester.summary_search()``.

    Returns:
        dict[str, dict]:
            Dictionary of summary dictionaries for the input chemical system.
            The keys are the MPIDs (e.g. ``"mp-1234"``) and the values are the
            corresponding summary dictionaries.
    """
    api_key = _parse_MP_API_key(api_key)
    if entries is None and chemsys is None:
        raise ValueError("Either `entries` or `chemsys` must be provided!")

    if entries:
        summary_search_kwargs = {
            "material_ids": [entry.data["material_id"] for entry in entries],
            **kwargs,
        }
    else:
        assert chemsys is not None  # typing
        summary_search_kwargs = {"chemsys": _get_all_chemsyses("-".join(chemsys)), **kwargs}

    with MPRester(api_key) as mpr:
        MP_doc_dicts = {
            doc_dict["material_id"]: doc_dict for doc_dict in mpr.summary_search(**summary_search_kwargs)
        }

    if not entries:
        return MP_doc_dicts

    for entry in entries:
        entry.MP_doc_dict = MP_doc_dicts.get(entry.data["material_id"], None)
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
):
    r"""
    Function to sort ``ComputedEntry``\s by energy above hull, then if
    composition matches ``bulk_composition`` (if provided), then by the number
    of elements in the formula, then by the position of elements in the
    periodic table (main group elements, then transition metals, sorted by
    row), then alphabetically.

    Usage: ``entries_list.sort(key=_entries_sort_func)``

    Args:
        entry (ComputedEntry):
            ComputedEntry object to sort.
        use_e_per_atom (bool):
            If ``True``, sort by energy per atom rather than energy above hull.
            Default is ``False``.
        bulk_composition (str/Composition/dict/list):
            Bulk composition; to sort entries matching this composition first.
            Default is ``None`` (don't sort according to this).

    Returns:
        tuple:
            Tuple of ``True``/``False`` (if composition matches bulk
            composition), the energy above hull (or energy per atom), number of
            elements in the formula, and sorted (group, row) list of elements
            in the formula, and the formula name.
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
):
    """
    Given an input list of ``ComputedEntry``/``ComputedStructureEntry``s
    (``entries``) and a single entry for the host material
    (``bulk_computed_entry``), returns the subset of entries which `could`
    border the host on the phase diagram (and therefore be a competing phase
    which determines the host chemical potential limits), allowing for an error
    tolerance for the semi-local DFT database energies (``energy_above_hull``,
    set to ``self.energy_above_hull`` -- 0.05 eV/atom by default).

    If ``phase_diagram`` is provided then this is used as the reference phase
    diagram, otherwise it is generated from ``entries`` and
    ``bulk_computed_entry``.

    Args:
        entries (list[ComputedEntry]):
            List of ``ComputedEntry`` objects to prune down to potential host
            border candidates on the phase diagram.
        bulk_computed_entry (ComputedEntry):
            ``ComputedEntry`` object for the host material.
        phase_diagram (PhaseDiagram):
            Optional input; ``PhaseDiagram`` object for the system of interest.
            If provided, this is used as the reference phase diagram from which
            to determine the (potential) chemical potential limits, otherwise
            it is generated from ``entries`` and ``bulk_computed_entry``.
        energy_above_hull (float):
            Maximum energy above hull (in eV/atom) of Materials Project entries
            to be considered as competing phases. This is an uncertainty range
            for the MP-calculated formation energies, which may not be accurate
            due to functional choice (GGA vs hybrid DFT / GGA+U / RPA etc.),
            lack of vdW corrections etc. All phases that would border the host
            material on the phase diagram, if their relative energy was
            downshifted by ``energy_above_hull``, are included.
            (Default is 0.05 eV/atom).

    Returns:
        list[ComputedEntry]:
            List of all ``ComputedEntry`` objects in ``entries`` which could
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


def get_and_set_competing_phase_name(
    entry: ComputedStructureEntry | ComputedEntry, regenerate=False, ndigits=3
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
        entry (ComputedStructureEntry, ComputedEntry):
            ``pymatgen`` ``ComputedStructureEntry`` object for the competing
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
            space_group = entry.structure.get_space_group_info()[0]
        else:
            space_group = "NA"
        entry.data["doped_name"] = f"{entry.name}_{space_group}_EaH_{rounded_eah}"

    return entry.data.get("doped_name")


def _get_competing_phase_folder_name(
    entry: ComputedStructureEntry | ComputedEntry, regenerate=False, ndigits=3
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
        entry (ComputedStructureEntry, ComputedEntry):
            ``pymatgen`` ``ComputedStructureEntry`` object for the competing
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


def _name_entries_and_handle_duplicates(entries: list[ComputedStructureEntry]):
    """
    Given an input list of ``ComputedStructureEntry`` objects, sets the
    ``entry.data["doped_name"]`` values using
    ``get_and_set_competing_phase_name``, and increases ``ndigits`` (rounding
    for energy above hull in name) dynamically from 3 -> 4 -> 5 to ensure no
    duplicate names.
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
        _duplicate_entry_names = [
            get_and_set_competing_phase_name(entry, regenerate=True, ndigits=ndigits)
            for entry in duplicate_entries
        ]
        entry_names = [get_and_set_competing_phase_name(entry, regenerate=False) for entry in entries]


# TODO: Make these classes MSONable
# TODO: Make entries sub-selectable using dict indexing like DefectsGenerator
class CompetingPhases:
    def __init__(
        self,
        composition: str | Composition | Structure,
        energy_above_hull: float = 0.05,
        extrinsic: str | Iterable | None = None,
        full_phase_diagram: bool = False,
        MP_doc_dicts: bool = False,
        full_sub_approach: bool = False,
        codoping: bool = False,
        api_key: str | None = None,
    ):
        """
        Class to generate VASP input files for competing phases on the phase
        diagram for the host material (and any extrinsic (dopant/impurity)
        elements), which determine the chemical potential limits for elements
        in that compound.

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
            composition (str, ``Composition``, ``Structure``):
                Composition of the host material (e.g. ``'LiFePO4'``, or
                ``Composition('LiFePO4')``, or
                ``Composition({"Li":1, "Fe":1, "P":1, "O":4})``).
                Alternatively a ``pymatgen`` ``Structure`` object for the
                host material can be supplied (recommended), in which case
                the primitive structure will be used as the only host
                composition phase, reducing the number of calculations.
            energy_above_hull (float):
                Maximum energy above hull (in eV/atom) of Materials Project
                entries to be considered as competing phases. This is an
                uncertainty range for the MP-calculated formation energies,
                which may not be accurate due to functional choice (GGA vs
                hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
                All phases that would border the host material on the phase
                diagram, if their relative energy was downshifted by
                ``energy_above_hull``, are included.
                Often ``energy_above_hull`` can be lowered (e.g. to ``0``) to
                reduce the number of calculations while retaining good accuracy
                relative to the typical error of defect calculations.
                (Default is 0.05 eV/atom).
            extrinsic (str, Iterable):
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
            full_sub_approach (bool):
                Generate competing phases by considering the full phase
                diagram, including chemical potential limits with multiple
                extrinsic phases. Only recommended when looking at high
                (non-dilute) doping concentrations.
                The default approach (``full_sub_approach = False``) for
                extrinsic elements is to only consider chemical potential
                limits where the host composition borders a maximum of 1
                extrinsic phase (composition with extrinsic element(s)). This
                is a valid approximation for the case of dilute dopant/impurity
                concentrations. For high (non-dilute) concentrations of
                extrinsic species, use ``full_sub_approach = True``.
            codoping (bool):
                Whether to consider extrinsic competing phases containing
                multiple extrinsic species. Usually only relevant under high
                (non-dilute) co-doping concentrations. If set to True, then
                ``full_sub_approach`` is also set to ``True``. Default is
                ``False``.
            api_key (str):
                Materials Project (MP) API key, needed to access the MP
                database for competing phase generation. If not supplied, will
                attempt to read from environment variable ``PMG_MAPI_KEY`` (in
                ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``) -- see the
                ``doped`` Installation docs page:
                https://doped.readthedocs.io/en/latest/Installation.html
            MP_doc_dicts (bool):
                If ``True``, also queries the Materials Project (MP) for
                summary doc dicts with ``MPRester.summary_search()`` for the
                competing phase entries, and stores them in
                ``CompetingPhases.MP_doc_dicts``. Default is ``False``.
        """
        # TODO: Give quick attribute summary at end of docstring above, following chempots code overhaul
        self.energy_above_hull = energy_above_hull  # store parameters for reference
        self.full_phase_diagram = full_phase_diagram
        self.extrinsic = extrinsic
        self.codoping = codoping
        self.full_sub_approach = full_sub_approach
        self.api_key = _parse_MP_API_key(api_key)

        # TODO: Should hard code S (solid + S8 (mp-994911), + S2 (molecule in a box)), P, Te and Se in
        #  here too. Common anions with a lot of unnecessary polymorphs on MP. Should at least scan over
        #  elemental phases and hard code any particularly bad cases. E.g. P_EaH=0 is red phosphorus
        #  (HSE06 groundstate), P_EaH=0.037 is black phosphorus (thermo stable at RT), so only need to
        #  generate these. Same for all alkali and alkaline earth metals (ask the battery boys), TiO2,
        #  SnO2, WO3 (particularly bad cases).
        # Can have a data file with a list of known, common cases?
        # With Materials Project querying, can check if the structure has a database ID (i.e. is
        # experimentally observed) with icsd_id(s) / theoretical (same thing). Could have 'lean' option
        # which only outputs phases which are either on the MP-predicted hull or have an ICSD ID?
        # Would want to test this to see if it is sufficient in most cases, then can recommend its use
        # with a caution... From a quick test, this does cut down a good chunk of unnecessary phases,
        # but still not all as often there are several ICSD phases for e.g. metals with a load of known
        # polymorphs (at different temperatures/pressures).
        # for new MP API; see https://github.com/materialsproject/api/issues/625 &
        # https://github.com/materialsproject/api/issues/675 &
        # https://github.com/materialsproject/api/issues/857 for accessing ICSD etc IDs (also
        # doc.database_IDs etc)

        # Strategies for dealing with these cases where MP has many low energy polymorphs in general?
        # Will mention some good practice in the docs anyway. -> Have an in-built warning when many
        # entries for the same composition, warn the user (that if the groundstate phase at low/room
        # temp is well-known, then likely best to prune to that) and direct to relevant section on the
        # docs discussing this
        # -- Could have two optional EaH tolerances, a tight one (0.02 eV/atom?) that applies to all,
        # and a looser one (0.1 eV/atom?) that applies to phases with ICSD IDs?

        if isinstance(composition, Structure):
            # if structure is not primitive, reduce to primitive:
            primitive_structure = get_primitive_structure(composition)
            if len(primitive_structure) < len(composition):
                self.bulk_structure = primitive_structure
            else:
                self.bulk_structure = composition
            self.composition = self.bulk_structure.composition

        else:
            self.bulk_structure = None
            self.composition = Composition(composition)

        self.chemsys = list(self.composition.get_el_amt_dict().keys())

        # get all entries in the chemical system with EaH < ``energy_above_hull``:
        self.MP_full_pd_entries = get_entries_in_chemsys(
            self.chemsys,
            api_key=self.api_key,
            energy_above_hull=self.energy_above_hull,
            bulk_composition=self.composition.reduced_formula,  # for sorting
        )
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
                )  # TODO: Later need to add handling for file writing for this (POTCAR and INCAR assuming
                # non-metallic, non-magnetic, with warning and recommendations

            if self.MP_bulk_computed_entry not in formatted_entries:
                formatted_entries.append(self.MP_bulk_computed_entry)

        if self.bulk_structure:  # prune all bulk phases to this structure
            manual_bulk_entry = None

            if bulk_entries := [
                entry
                for entry in formatted_entries  # sorted by energy_above_hull in ``get_entries_in_chemsys``
                if entry.composition.reduced_composition == self.composition.reduced_composition
            ]:
                sm = StructureMatcher()
                matching_bulk_entries = [
                    entry
                    for entry in bulk_entries
                    if hasattr(entry, "structure") and sm.fit(self.bulk_structure, entry.structure)
                ]
                matching_bulk_entries.sort(key=lambda x: sm.get_rms_dist(self.bulk_structure, x.structure))
                if matching_bulk_entries:
                    matching_bulk_entry = matching_bulk_entries[0]
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

        # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
        self.entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=self.composition))
        _name_entries_and_handle_duplicates(self.entries)  # set entry names

        if MP_doc_dicts:
            self.MP_doc_dicts = get_MP_summary_dicts(entries=self.entries, api_key=self.api_key)
        else:
            self.MP_doc_dicts = {}

        self.intrinsic_species = [s.symbol for s in self.composition.reduced_composition.elements]

        if not self.extrinsic:
            self.intrinsic_entries: list[ComputedEntry] = self.entries
            self.extrinsic_entries: list[ComputedEntry] = []
            self.MP_intrinsic_full_pd_entries = self.MP_full_pd_entries  # includes molecules-in-boxes
            return

        # otherwise, we have extrinsic species present:
        self.intrinsic_entries = copy.deepcopy(self.entries)

        if not isinstance(self.extrinsic, str | Iterable):
            raise TypeError(
                f"`extrinsic` must be a string (i.e. the extrinsic species symbol, e.g. 'Mg') or an "
                f"iterable object (list, set, tuple or dict; e.g. ['Mg', 'Na']), got type "
                f"{type(self.extrinsic)} instead!"
            )

        self.extrinsic_species = (
            [self.extrinsic] if isinstance(self.extrinsic, str) else list(self.extrinsic)
        )
        if extrinsic_in_intrinsic := [
            ext for ext in self.extrinsic_species if ext in self.intrinsic_species
        ]:
            raise ValueError(
                f"Extrinsic species {extrinsic_in_intrinsic} are already present in the host composition "
                f"({self.composition}), and so cannot be considered as extrinsic species!"
            )

        if self.codoping:  # if codoping is True, should have multiple extrinsic species:
            if len(self.extrinsic_species) < 2:
                warnings.warn(
                    "`codoping` is set to True, but `extrinsic_species` only contains 1 element, "
                    "so `codoping` will be set to False."
                )
                self.codoping = False

            elif not self.full_sub_approach:
                self.full_sub_approach = True

        if self.full_sub_approach and self.codoping:
            # can be time-consuming if several extrinsic_species supplied
            self.MP_full_pd_entries = get_entries_in_chemsys(
                chemsys=self.intrinsic_species + self.extrinsic_species,
                api_key=self.api_key,
                energy_above_hull=self.energy_above_hull,
                bulk_composition=self.composition.reduced_formula,  # for sorting
            )
            self.entries = self._generate_elemental_diatomic_phases(self.MP_full_pd_entries)

            if not self.full_phase_diagram:
                self.entries = prune_entries_to_border_candidates(
                    entries=self.entries,
                    bulk_computed_entry=self.MP_bulk_computed_entry,
                    energy_above_hull=self.energy_above_hull,
                )  # prune using phase diagram with all extrinsic species

        else:  # full_sub_approach (for now, to get self.entries) but not co-doping
            candidate_extrinsic_entries = []
            for sub_el in self.extrinsic_species:
                sub_el_MP_full_pd_entries = get_entries_in_chemsys(
                    [*self.intrinsic_species, sub_el],
                    api_key=self.api_key,
                    energy_above_hull=self.energy_above_hull,
                    bulk_composition=self.composition.reduced_formula,  # for sorting
                )
                sub_el_pd_entries = self._generate_elemental_diatomic_phases(sub_el_MP_full_pd_entries)
                self.MP_full_pd_entries.extend(
                    [entry for entry in sub_el_MP_full_pd_entries if entry not in self.MP_full_pd_entries]
                )

                if not self.full_phase_diagram:  # default, prune to only phases that would border the host
                    # material on the phase diagram, if their relative energy was downshifted by
                    # `energy_above_hull`; prune using phase diagrams for one extrinsic species at a time:
                    sub_el_pd_entries = prune_entries_to_border_candidates(
                        entries=sub_el_pd_entries,
                        bulk_computed_entry=self.MP_bulk_computed_entry,
                        energy_above_hull=self.energy_above_hull,
                    )

                candidate_extrinsic_entries += [
                    entry for entry in sub_el_pd_entries if sub_el in entry.composition
                ]

            if self.full_sub_approach:
                self.entries += candidate_extrinsic_entries

            else:  # not full_sub_approach; recommended approach for extrinsic species (assumes dilute
                # concentrations). Here we only retain extrinsic competing phases when they border the
                # host composition on the phase diagram with no other extrinsic phases in equilibrium
                # at this limit. This is essentially the assumption that the majority of elements in
                # the total composition will be from the host composition rather than the extrinsic
                # species (a good approximation for dilute concentrations)
                for sub_el in self.extrinsic_species:
                    sub_el_entries = [
                        entry for entry in candidate_extrinsic_entries if sub_el in entry.composition
                    ]

                    if not sub_el_entries:
                        raise ValueError(
                            f"No Materials Project entries found for the given chemical "
                            f"system: {[*self.intrinsic_species, sub_el]}"
                        )

                    sub_el_phase_diagram = PhaseDiagram([*self.intrinsic_entries, *sub_el_entries])
                    MP_extrinsic_gga_chempots = get_chempots_from_phase_diagram(
                        self.MP_bulk_computed_entry, sub_el_phase_diagram
                    )
                    MP_extrinsic_bordering_phases: list[str] = []

                    for limit in MP_extrinsic_gga_chempots:
                        # note that the number of phases in equilibria at each vertex (limit) is equal
                        # to the number of elements in the chemical system (here being the host
                        # composition plus the extrinsic species)
                        extrinsic_bordering_phases = {
                            phase for phase in limit.split("-") if sub_el in phase
                        }
                        # only add to MP_extrinsic_bordering_phases when only 1 extrinsic bordering phase
                        # (i.e. ``full_sub_approach=False`` behaviour):
                        if len(  # this should always give the same number of facets as the bulk PD
                            extrinsic_bordering_phases
                            # TODO: Explicitly test this for all cases in tests
                        ) == 1 and not extrinsic_bordering_phases.issubset(MP_extrinsic_bordering_phases):
                            MP_extrinsic_bordering_phases.extend(extrinsic_bordering_phases)

                    single_bordering_sub_el_entries = [
                        entry
                        for entry in candidate_extrinsic_entries
                        if entry.name in MP_extrinsic_bordering_phases
                        or (entry.is_element and sub_el in entry.name)
                    ]

                    # check that extrinsic competing phases list is not empty (according to PyCDT
                    # chemical potential handling this can happen (despite purposely neglecting these
                    # "over-dependent" facets above), but no known cases... (apart from when `extrinsic`
                    # actually contains an intrinsic element, which we handle above anyway)
                    if not single_bordering_sub_el_entries:
                        # self.entries += sub_el_entries
                        raise RuntimeError(
                            f"Determined chemical potentials to be over-dependent on the extrinsic "
                            f"species {sub_el} despite `full_sub_approach=False`, which shouldn't happen. "
                            f"Please report this to the developers on the GitHub issues page: "
                            f"https://github.com/SMTG-Bham/doped/issues"
                        )  # Revert to making this a warning if we find a case of this actually happening

                    self.entries += single_bordering_sub_el_entries

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
        assert len(self.intrinsic_entries) + len(self.extrinsic_entries) == len(
            self.entries
        ), "Error in extrinsic entries generation, please report this to the doped developers!"

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

    # TODO: Return dict of DictSet objects for this and vasp_std_setup() functions, as well as
    #  write_files option, for ready integration with high-throughput workflows
    # TODO: Have option to only write extrinsic files to output (in case regenerated when adding calcs
    #  for dopants etc, here and with vasp_std_setup)?
    # TODO: Smart file/folder overwriting handling (like in SnB?)
    def convergence_setup(
        self,
        kpoints_metals=(40, 1000, 5),
        kpoints_nonmetals=(5, 120, 5),
        user_potcar_functional="PBE",
        user_potcar_settings=None,
        user_incar_settings=None,
        **kwargs,
    ):
        """
        Generates VASP input files for k-points convergence testing for
        competing phases, using PBEsol (GGA) DFT by default. Automatically sets
        the ``ISMEAR`` ``INCAR`` tag to 2 (if metallic) or 0 if not.
        Recommended to use with https://github.com/kavanase/vaspup2.0.

        Args:
            kpoints_metals (tuple):
                Kpoint density per inverse volume (Å^-3) to be tested in
                ``(min, max, step)`` format for metals
            kpoints_nonmetals (tuple):
                Kpoint density per inverse volume (Å^-3) to be tested in
                ``(min, max, step)`` format for nonmetals
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR``
                set.
            user_incar_settings (dict):
                Override the default INCAR settings e.g.
                ``{"EDIFF": 1e-5, "LDAU": False, "ALGO": "All"}``. Note that
                any non-numerical or non-``True``/``False`` flags need to be
                input as strings with quotation marks. See
                ``doped/VASP_sets/PBEsol_ConvergenceSet.yaml`` for the default
                settings.
            **kwargs:
                Additional kwargs to pass to ``DictSet.write_input()``
        """
        # by default uses PBEsol, but easy to switch to PBE or PBE+U using user_incar_settings
        base_incar_settings = copy.deepcopy(pbesol_convrg_set["INCAR"])
        base_incar_settings.update(user_incar_settings or {})  # user_incar_settings override defaults

        for entry_list, type in [
            (self.nonmetallic_entries, "non-metals"),
            (self.metallic_entries, "metals"),
        ]:  # no molecular entries as they don't need convergence testing
            # kpoints should be set as (min, max, step):
            min_k, max_k, step_k = {"non-metals": kpoints_nonmetals, "metals": kpoints_metals}[type]
            for entry in entry_list:
                uis = copy.deepcopy(base_incar_settings or {})
                self._set_spin_polarisation(uis, user_incar_settings or {}, entry)
                if type == "metals":
                    self._set_default_metal_smearing(uis, user_incar_settings or {})

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="KPOINTS are Γ")  # Γ only KPAR warning
                    dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                        structure=entry.structure,
                        user_incar_settings=uis,
                        user_kpoints_settings={"reciprocal_density": min_k},
                        user_potcar_settings=user_potcar_settings or {},
                        user_potcar_functional=user_potcar_functional,
                        force_gamma=True,
                    )

                    for kpoint in range(min_k, max_k, step_k):
                        dict_set.user_kpoints_settings = {"reciprocal_density": kpoint}
                        kname = (
                            "k"
                            + ("_" * (dict_set.kpoints.kpts[0][0] // 10))
                            + ",".join(str(k) for k in dict_set.kpoints.kpts[0])
                        )
                        fname = (
                            f"CompetingPhases/{_get_competing_phase_folder_name(entry)}/kpoint_converge"
                            f"/{kname}"
                        )
                        dict_set.write_input(fname, **kwargs)

        if self.molecular_entries:
            print(
                f"Note that diatomic molecular phases, calculated as molecules-in-a-box "
                f"({', '.join([e.name for e in self.molecular_entries])} in this case), do not require "
                f"k-point convergence testing, as Γ-only sampling is sufficient."
            )

    # TODO: Add vasp_ncl_setup()
    def vasp_std_setup(
        self,
        kpoints_metals=200,
        kpoints_nonmetals=64,  # MPRelaxSet default
        user_potcar_functional="PBE",
        user_potcar_settings=None,
        user_incar_settings=None,
        **kwargs,
    ):
        """
        Generates VASP input files for ``vasp_std`` relaxations of the
        competing phases, using HSE06 (hybrid DFT) DFT by default.
        Automatically sets the ``ISMEAR`` ``INCAR`` tag to 2 (if metallic) or 0
        if not. Note that any changes to the default ``INCAR``/``POTCAR``
        settings should be consistent with those used for the defect supercell
        calculations.

        Args:
            kpoints_metals (int):
                Kpoint density per inverse volume (Å^-3) for metals.
                Default is 200.
            kpoints_nonmetals (int):
                Kpoint density per inverse volume (Å^-3) for nonmetals
                (default is 64, the default for ``MPRelaxSet``).
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR``
                set.
            user_incar_settings (dict):
                Override the default INCAR settings e.g.
                ``{"EDIFF": 1e-5, "LDAU": False, "ALGO": "All"}``. Note that
                any non-numerical or non-``True``/``False`` flags need to be
                input as strings with quotation marks. See
                ``doped/VASP_sets/RelaxSet.yaml`` and ``HSESet.yaml`` for the
                default settings.
            **kwargs:
                Additional kwargs to pass to ``DictSet.write_input()``
        """
        base_incar_settings = copy.deepcopy(default_relax_set["INCAR"])

        lhfcalc = (
            True if user_incar_settings is None else user_incar_settings.get("LHFCALC", True)
        )  # True (hybrid) by default for vasp_std relaxations
        if lhfcalc or (isinstance(lhfcalc, str) and lhfcalc.lower().startswith("t")):
            base_incar_settings.update(default_HSE_set["INCAR"])

        base_incar_settings.update(user_incar_settings or {})  # user_incar_settings override defaults

        for entry_list, type in [
            (self.nonmetallic_entries, "non-metals"),
            (self.metallic_entries, "metals"),
            (self.molecular_entries, "molecules"),
        ]:
            if type == "molecules":
                user_kpoints_settings = Kpoints().from_dict(
                    {
                        "comment": "Gamma-only kpoints for molecule-in-a-box",
                        "generation_style": "Gamma",
                    }
                )
            elif type == "non-metals":
                user_kpoints_settings = {"reciprocal_density": kpoints_nonmetals}
            else:  # metals
                user_kpoints_settings = {"reciprocal_density": kpoints_metals}

            for entry in entry_list:
                uis = copy.deepcopy(base_incar_settings or {})
                if type == "molecules":
                    uis["ISIF"] = 2  # can't change the volume
                    uis["KPAR"] = 1  # can't use k-point parallelization, gamma only
                self._set_spin_polarisation(uis, user_incar_settings or {}, entry)
                if type == "metals":
                    self._set_default_metal_smearing(uis, user_incar_settings or {})

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="KPOINTS are Γ")  # Γ only KPAR warning
                    dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                        structure=entry.structure,
                        user_incar_settings=uis,
                        user_kpoints_settings=user_kpoints_settings,
                        user_potcar_settings=user_potcar_settings or {},
                        user_potcar_functional=user_potcar_functional,
                        force_gamma=True,
                    )

                    fname = f"CompetingPhases/{_get_competing_phase_folder_name(entry)}/vasp_std"
                    dict_set.write_input(fname, **kwargs)

    def _set_spin_polarisation(self, incar_settings, user_incar_settings, entry):
        """
        If the entry has a non-zero total magnetization (greater than the
        default tolerance of 0.1), set ``ISPIN`` to 2 (allowing spin
        polarisation) and ``NUPDOWN`` equal to the integer-rounded total
        magnetization.

        Otherwise ``ISPIN`` is not set, so spin polarisation is not allowed (as
        typically desired for non-magnetic phases, for efficiency).

        See https://doped.readthedocs.io/en/latest/Tips.html#spin
        """
        magnetization = entry.data.get("summary", {}).get("total_magnetization")
        with contextlib.suppress(TypeError):  # if magnetization is None, fine, skip
            if magnetization > 0.1:  # account for magnetic moment
                incar_settings["ISPIN"] = user_incar_settings.get("ISPIN", 2)
                if "NUPDOWN" not in incar_settings and int(magnetization) > 0:
                    incar_settings["NUPDOWN"] = int(magnetization)

        # otherwise ISPIN not set, so no spin polarisation

    def _set_default_metal_smearing(self, incar_settings, user_incar_settings):
        """
        Set the smearing parameters to the ``doped`` defaults for metallic
        phases (i.e. ``ISMEAR`` = 2 (Methfessel-Paxton) and ``SIGMA`` = 0.2
        eV).
        """
        incar_settings["ISMEAR"] = user_incar_settings.get("ISMEAR", 2)
        incar_settings["SIGMA"] = user_incar_settings.get("SIGMA", 0.2)

    def _generate_elemental_diatomic_phases(self, entries: list[ComputedEntry]):
        r"""
        Given an input list of ``ComputedEntry`` objects, adds a
        ``ComputedStructureEntry`` for each diatomic elemental phase (O2, N2,
        H2, F2, Cl2) to ``entries`` using ``make_molecular_entry``, and
        generates an output list of ``ComputedEntry`` /
        ``ComputedStructureEntry``\s containing all entries in ``entries``,
        with `all` elemental phases of O, N, H, F, Cl replaced by the single
        corresponding molecule-in-a-box entry.

        Also sets the ``ComputedEntry.data["molecule"]`` flag for each entry in
        ``entries`` (``True`` for diatomic gases, ``False`` for all others).

        The output entries list is sorted by energy above hull, then by the
        number of elements in the formula, then by the position of elements in
        the periodic table (main group elements, then transition metals, sorted
        by row).

        Args:
            entries (list[ComputedEntry]):
                List of ``ComputedEntry``/``ComputedStructureEntry`` objects
                for the input chemical system.

        Returns:
            list[ComputedEntry]:
                List of ``ComputedEntry``/``ComputedStructureEntry`` objects
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
        formatted_entries.sort(key=lambda x: _entries_sort_func(x))

        return formatted_entries

    def __repr__(self):
        """
        Returns a string representation of the ``CompetingPhases`` object.
        """
        formula = self.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        joined_entry_list = "\n".join([entry.data["doped_name"] for entry in self.entries])
        return (
            f"doped CompetingPhases for bulk composition {formula} with {len(self.entries)} entries "
            f"(in self.entries):\n{joined_entry_list}\n\n"
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )


def get_doped_chempots_from_entries(
    entries: Sequence[ComputedEntry | ComputedStructureEntry | PDEntry],
    composition: str | Composition | ComputedEntry,
    sort_by: str | None = None,
    single_chempot_limit: bool = False,
) -> dict:
    r"""
    Given a list of ``ComputedEntry``\s / ``ComputedStructureEntry``\s /
    ``PDEntry``\s and the bulk ``composition``, returns the chemical potential
    limits dictionary in the ``doped`` format (i.e. ``{"limits": [{'limit':
    [chempot_dict]}], ...}``) for the host material.

    Args:
        entries (list[ComputedEntry]):
            List of ``ComputedEntry``\s / ``ComputedStructureEntry``\s /
            ``PDEntry``\s for the chemical system, from which to determine
            the chemical potential limits for the host material
            (``composition``).
        composition (str, Composition, ComputedEntry):
            Composition of the host material either as a string
            (e.g. 'LiFePO4') a ``pymatgen`` ``Composition`` object (e.g.
            ``Composition('LiFePO4')``), or a ``ComputedEntry`` object.
        sort_by (str):
            If set, will sort the chemical potential limits in the output
            ``DataFrame`` according to the chemical potential of the specified
            element (from element-rich to element-poor conditions).
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

    phase_diagram = PhaseDiagram(
        entries,
        list(map(Element, composition.elements)),  # preserve bulk comp element ordering
    )
    chem_lims = phase_diagram.get_all_chempots(composition.reduced_composition)
    chem_lims_iterator = list(chem_lims.items())[:1] if single_chempot_limit else chem_lims.items()

    # remove Element to make it JSONable:
    no_element_chem_lims = {k: {str(kk): vv for kk, vv in v.items()} for k, v in chem_lims_iterator}

    if sort_by is not None:
        no_element_chem_lims = dict(
            sorted(no_element_chem_lims.items(), key=lambda x: x[1][sort_by], reverse=True)
        )

    chempots = {
        "limits": no_element_chem_lims,
        "elemental_refs": {str(el): ent.energy_per_atom for el, ent in phase_diagram.el_refs.items()},
        "limits_wrt_el_refs": {},
    }

    # relate the limits to the elemental energies
    for limit, chempot_dict in chempots["limits"].items():
        relative_chempot_dict = copy.deepcopy(chempot_dict)
        for e in relative_chempot_dict:
            relative_chempot_dict[e] -= chempots["elemental_refs"][e]
        chempots["limits_wrt_el_refs"].update({limit: relative_chempot_dict})

    # round all floats to 4 decimal places (0.1 meV/atom) for cleanliness (well below DFT accuracy):
    return _round_floats(chempots, 4)


class ChemicalPotentialGrid:
    """
    A class to represent a grid in chemical potential space and to perform
    operations such as generating a grid within the convex hull of given
    vertices (chemical potential limits).

    This class provides methods for handling and manipulating chemical
    potential data, including the creation of a grid that spans a specified
    chemical potential space.
    """

    def __init__(self, chempots: dict[str, Any]):
        r"""
        Initializes the ``ChemicalPotentialGrid`` with chemical potential data.

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
        """
        unformatted_chempots_dict = chempots.get("limits_wrt_el_refs", chempots)
        test_elt = Element("H")
        formatted_chempots_dict = {
            limit: {
                f"μ_{k} (eV)" if test_elt.is_valid_symbol(k) else k: v
                for (k, v) in unformatted_chempots_subdict.items()
            }
            for limit, unformatted_chempots_subdict in unformatted_chempots_dict.items()
        }

        self.vertices = pd.DataFrame.from_dict(formatted_chempots_dict, orient="index")

    @classmethod
    def from_dataframe(cls, vertices: pd.DataFrame) -> "ChemicalPotentialGrid":
        """
        Create a ``ChemicalPotentialGrid`` object from a dataframe of the
        chemical potential limits (i.e. vertices).

        Args:
            vertices (pd.DataFrame):
                A ``DataFrame`` of the chemical potential limits (i.e. vertices).
                The columns should be the chemical potentials of the elements,
                and the rows should be the limits.

        Returns:
            ChemicalPotentialGrid:
                A ``ChemicalPotentialGrid`` object.
        """
        self = cls.__new__(cls)  # skip init function
        self.vertices = vertices
        return self

    def get_grid(
        self,
        n_points: int | None = None,
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
                points dropped. The default is 1000 when barycentric
                coordinates are used (``cartesian=False``), and 100 otherwise
                (as Cartesian grid generation and sub-selection to the stable
                polytope is much slower).
                Note that large values (>= 1e5) with multinary systems can
                explode, crashing system memory.
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
                speed up runtime.
            include_vertices (bool):
                Whether to include the vertices themselves in the generated
                grid. Default is ``True``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the points within the convex hull.
                Each row represents a point in the grid.
        """
        n_points = n_points or (1000 if not cartesian else 100)
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
        delaunay_tri = Delaunay(coords_hull)

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
            inside_hull = delaunay_tri.find_simplex(grid_points) >= 0
            points_inside = grid_points[inside_hull]
            values_inside = griddata(  # interpolate values to get the dependent chemical potential
                independent_vars.to_numpy(), dependent_var, points_inside, method="linear"
            )
            # Combine points with their corresponding interpolated values:
            grid_with_values = np.hstack((points_inside, values_inside.reshape(-1, 1)))

        else:  # efficiently generate a grid of points inside the convex hull, using barycentric coords:
            grid_with_values = _lattice_in_hull(delaunay_tri, values_hull, n_points=n_points)

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
        n_points: int | None = None,
        cartesian: bool = False,
        decimal_places: int = 4,
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
                The default is 1000 when barycentric coordinates are used
                (``cartesian=False``), and 100 otherwise (as Cartesian grid
                generation and sub-selection to the stable polytope is much
                slower). Note that large values (>= 1e5) with multinary systems
                can explode, crashing system memory.
            cartesian (bool):
                Whether to generate the grid in Cartesian coordinates. If
                ``False`` (default), the grid is generated in barycentric
                coordinates, which is far more efficient, but means that the
                grid is evenly spaced in barycentric ('relative') coordinates,
                and not necessarily in Cartesian coordinates.
            decimal_places (int):
                The number of decimal places to round the grid coordinates to.
                Default is 4.
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
        values_inside = griddata(independent_vars.to_numpy(), dependent_var, vertices, method="linear")

        # Combine points with their corresponding interpolated values
        grid_with_values = np.hstack((vertices, values_inside.reshape(-1, 1)))

        # remove nan values and round
        grid_with_values = grid_with_values[~np.isnan(grid_with_values[:, -1])].round(decimal_places + 1)

        constrained_vertices = pd.DataFrame(
            grid_with_values,
            columns=[*list(independent_vars.columns), dependent_variable],
        )
        # these are our new constrained vertices, now we generate the grid (without the fixed element):
        input_constrained_vertices = constrained_vertices.drop(columns=list(fixed_elements.keys()))
        constrained_grid = ChemicalPotentialGrid.from_dataframe(input_constrained_vertices)
        return constrained_grid.get_grid(
            n_points=n_points,
            cartesian=cartesian,
            decimal_places=decimal_places,
            include_vertices=include_vertices,
        ).dropna()


def _intersect_hull_with_plane(
    vertices: np.ndarray,
    axis: int,
    value: float,
    tol: float = 1e-10,
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
    lo, hi = vertices[:, axis].min(), vertices[:, axis].max()
    if value < lo - tol or value > hi + tol:
        raise ValueError(f"The plane {axis} = {value} does not meet the hull.")

    pts = []
    # for each edge, check if it crosses the plane:
    for i, j in itertools.combinations(range(len(vertices)), 2):
        v_i, v_j = vertices[i], vertices[j]
        d_i = v_i[axis] - value  # vector from plane to vertex i, along axis
        d_j = v_j[axis] - value  # vector from plane to vertex j, along axis

        if d_i * d_j < 0:  # opposite signs in vectors -> crossing
            t = d_i / (d_i - d_j)  # 0 < t < 1
            pts.append(v_i + t * (v_j - v_i))  # intersection point

        else:  # check if endpoint is on plane:
            for d, v in zip([d_i, d_j], [v_i, v_j], strict=False):
                if abs(d) <= tol:
                    pts.append(v)

    return np.asarray(pts)


def _lattice_in_hull(
    delaunay_tri: Delaunay, dependent_var: np.ndarray | None = None, n_points: int = 1000
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
        delaunay_tri (Delaunay):
            A Delaunay triangulation object representing the convex hull of
            the vertices.
        dependent_var (np.ndarray | None):
            Values of the dependent variable (e.g. chemical potential) at the
            vertices. If provided, the function will also interpolate the
            dependent variable values at the generated points inside the convex
            hull -- must have the same order as the vertices in
            ``delaunay_tri``! Default is ``None``.
        n_points (int):
            `Minimum` number of grid points to generate. The output grid will
            contain at least this many points, regularly spaced in barycentric
            space. Default is 1000.

    Returns:
        np.ndarray:
            A grid of points inside the convex hull, in Cartesian coordinates.
            The shape of the array is (M, k), where M is the number of points
            in the grid.
    """
    vertices = delaunay_tri.points  # vertices of the convex hull
    if vertices.ndim != 2:
        raise ValueError("`vertices` must be a 2-D array (N_points, N_dimensions)")

    # vertices defines the polytope (k-D polyhedron) of the convex hull; shape (N, k)
    # we then tessellate the hull with Delaunay triangulation, which breaks the polytope into a set of
    # k-D simplices (e.g. triangles in 2D, tetrahedra in 3D; simplest possible polytope in k-D space),
    # which each have k+1 vertices (e.g. 3 vertices for triangles, 4 vertices for tetrahedra, etc)
    k = vertices.shape[-1]  # dimensionality (k ≥ 2)

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
    while len(unscaled_bary_coords) * len(delaunay_tri.simplices) < n_points:
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

    verts_per_simplex = vertices[
        delaunay_tri.simplices
    ]  # shape (S, k+1, k); where S is the number of simplices

    # Vectorised Cartesian coordinates (and interpolated values if dependent_var is provided):
    # points_inside: (S, L, k) -> reshape -> (S*L, k)
    points_inside = np.einsum("lK,sKm->slm", bary_coords, verts_per_simplex).reshape(-1, k)  # K = k+1
    if dependent_var is None:
        return points_inside

    vals_per_simplex = dependent_var[delaunay_tri.simplices]  # (S, k+1)
    # values_inside: (S, L) -> reshape -> (S*L,)
    values_inside = np.einsum("lK,sK->sl", bary_coords, vals_per_simplex).ravel()
    return np.hstack((points_inside, values_inside.reshape(-1, 1)))


class CompetingPhasesAnalyzer(MSONable):
    def __init__(
        self,
        composition: str | Composition,
        entries: (
            PathLike | list[PathLike] | list[ComputedEntry] | list[ComputedStructureEntry]
        ) = "CompetingPhases",
        subfolder: PathLike | None = "vasp_std",
        verbose: bool = True,
        processes: int | None = None,
        check_compatibility: bool = True,
    ):
        r"""
        Class for post-processing competing phases calculations, to determine
        the corresponding chemical potentials for the host ``composition``.

        This class can be initialised from VASP outputs (``vasprun.xml``\s) by
        specifying the path to the directory containing the outputs (e.g.
        ``"CompetingPhases"``) or a list of directories, or from a list of of
        ``ComputedEntry``\s / ``ComputedStructureEntry``\s (e.g. for use with
        high-throughput computing architectures such as ``atomate2`` or
        ``AiiDA``).

        Multiprocessing is used by default to speed up parsing of VASP outputs,
        which can be controlled with ``processes``. If parsing hangs, this may
        be due to memory issues, in which case you should reduce ``processes``
        (e.g. 4 or less).

        Args:
            composition (str, ``Composition``):
                Composition of the host material (e.g. ``'LiFePO4'``, or
                ``Composition('LiFePO4')``, or
                ``Composition({"Li":1, "Fe":1, "P":1, "O":4})``).
            entries (PathLike, list[PathLike], list[ComputedEntry], list[ComputedStructureEntry]):
                Either a path to the base folder containing the VASP outputs
                (e.g. ``"CompetingPhases"``; default; with subfolders like:
                ``formula_EaH_X/vasp_std/vasprun.xml(.gz)``, or
                ``formula_EaH_X/vasprun.xml(.gz)``) or a list of paths to
                ``vasprun.xml(.gz)`` files. Alternatively, can be a list of
                ``ComputedEntry``\s / ``ComputedStructureEntry``\s.
            subfolder (PathLike):
                The subfolder in which your vasprun.xml(.gz) output files
                are located (e.g. a file-structure like:
                ``formula_EaH_X/{subfolder}/vasprun.xml(.gz)``), if ``entries``
                is a path to a base folder with VASP outputs. Default is to
                search for ``vasp_std`` subfolders, or directly in the
                ``formula_EaH_X`` folders.
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

        Key attributes:
            composition (str):
                The bulk (host) composition.
            chempots (dict):
                Dictionary of the chemical potential limits for the host
                material, in the ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``). This can be
                directly used with the ``DefectThermodynamics`` plotting &
                analysis methods, and saved to file with ``dumpfn`` from
                ``monty.serialization``.
            chempots_df (DataFrame):
                ``DataFrame`` of the chemical potential limits for the host.
            elements (list):
                List of all elements in the chemical system (host + extrinsic),
                from all parsed calculations.
            extrinsic_elements (str):
                List of extrinsic elements in the chemical system (not present
                in ``composition``).
            bulk_entry (ComputedStructureEntry):
                The lowest energy computed entry for the host material.
            unstable_host (bool):
                Whether the host material is unstable with respect to competing
                phases (i.e. has an energy above hull > 0).
            entries (list[Union[ComputedEntry, ComputedStructureEntry]]):
                List of all parsed ``ComputedEntry``\s /
                ``ComputedStructureEntry``\s.
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
            parsed_folders (list):
                List of folders from which VASP calculation outputs were
                parsed, if ``entries`` was given as a path / paths to
                directories.
        """
        # TODO: Use smart subfolder detection as in DefectsParser, and update docstring!
        self.composition = Composition(composition)
        self.elements: list[str] = [c.symbol for c in self.composition.elements]
        self.extrinsic_elements: list[str] = []

        # _from_vaspruns or _from_entries depending on input
        if not isinstance(entries, str | PathLike | list):
            raise TypeError(
                f"`entries` must be either a path to a directory containing VASP outputs, "
                f"a list of paths, or a list of ComputedEntry/ComputedStructureEntry objects, "
                f"got type {type(entries)} instead!"
            )

        self.vasprun_paths: list[str] = []
        self.parsed_folders: list[str] = []

        if isinstance(entries, str | PathLike) or isinstance(entries[0], str | PathLike):
            self._from_vaspruns(
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
    ):
        r"""
        Initialises the ``CompetingPhasesAnalyzer`` object from a list of
        ``pymatgen`` ``ComputedEntry``\s / ``ComputedStructureEntry``\s.

        Args:
            entries (list[Union[ComputedEntry, ComputedStructureEntry]]):
                List of ``ComputedEntry``\s / ``ComputedStructureEntry``\s,
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

        if check_compatibility:
            # check entry compatibilities:
            # take first of bulk entry, bulk comp entries, intrinsic, all entries which have INCAR/POTCAR
            # data as template entries for compatibility checking:
            sorted_entries_with_incar_data = [
                entry
                for entry in [self.bulk_entry, *bulk_comp_entries, *intrinsic_entries, *entries]
                if entry.data.get("incar")
            ]
            sorted_entries_with_potcar_data = [
                entry
                for entry in [self.bulk_entry, *bulk_comp_entries, *intrinsic_entries, *entries]
                if entry.data.get("potcar_symbols")
            ]
            if sorted_entries_with_incar_data:
                incar_template_entry = sorted_entries_with_incar_data[0]
                for entry in entries:
                    incar_mismatches = _compare_incar_tags(
                        incar_template_entry.data["incar"],
                        entry.data["incar"],
                        ignore_tags={"NKRED"},  # no NKRED mismatch warnings for competing phases
                        warn=False,
                    )  # warned collectively below if any mismatches
                    # ignore ISIF warnings in cases of supercell calculations (i.e. either gas calculations
                    # or bulk supercell -- assumed to be the correct volume):
                    if not isinstance(incar_mismatches, bool):
                        incar_mismatches = [
                            i
                            for i in incar_mismatches
                            if i[0] != "ISIF"
                            or all(ent.structure.volume < 800 for ent in [incar_template_entry, entry])
                        ]
                    incar_mismatches = incar_mismatches if incar_mismatches else False
                    entry.data["mismatching_INCAR_tags"] = (
                        incar_mismatches if not (isinstance(incar_mismatches, bool)) else False
                    )

                mismatching_INCAR_warnings = sorted(
                    [
                        (entry.name, set(entry.data.get("mismatching_INCAR_tags")))
                        for entry in entries
                        if entry.data.get("mismatching_INCAR_tags")
                    ],
                    key=lambda x: (len(x[1]), x[0]),
                    reverse=True,
                )  # sort by number of mismatches, reversed
                if mismatching_INCAR_warnings:
                    warnings.warn(
                        f"There are mismatching INCAR tags for (some of) your competing phases "
                        f"calculations which are likely to cause errors in the parsed results (energies "
                        f"& thus chemical potential limits). Found the following differences:\n"
                        f"(in the format: 'Entries: (INCAR tag, value in entry calculation, "
                        f"value in reference calculation))':"
                        f"\n{_format_mismatching_incar_warning(mismatching_INCAR_warnings)}\n"
                        f"Where {incar_template_entry.name} was used as the reference entry calculation.\n"
                        f"In general, the same INCAR settings should be used in all final calculations "
                        f"for these tags which can affect energies!"
                    )

            if sorted_entries_with_potcar_data:
                potcar_template_entry = sorted_entries_with_potcar_data[0]
                for entry in entries:
                    potcar_mismatches = _compare_potcar_symbols(
                        potcar_template_entry.data["potcar_symbols"],
                        entry.data["potcar_symbols"],
                        warn=False,
                        only_matching_elements=True,
                    )  # warned collectively below if any mismatches
                    entry.data["mismatching_POTCAR_symbols"] = (
                        potcar_mismatches if not (isinstance(potcar_mismatches, bool)) else False
                    )

                mismatching_potcars_warnings = sorted(
                    [
                        (entry.name, entry.data.get("mismatching_POTCAR_symbols"))
                        for entry in entries
                        if entry.data.get("mismatching_POTCAR_symbols")
                    ],
                    key=lambda x: (len(x[1]), x[0]),
                    reverse=True,
                )  # sort by number of mismatches, reversed
                if mismatching_potcars_warnings:
                    joined_info_string = "\n".join(
                        [f"{name}: {mismatching}" for name, mismatching in mismatching_potcars_warnings]
                    )
                    warnings.warn(
                        f"There are mismatching POTCAR symbols for (some of) your competing phases "
                        f"calculations which are likely to cause errors in the parsed results (energies & "
                        f"thus chemical potential limits). Found the following differences:\n"
                        f"(in the format: (entry POTCARs, reference POTCARs)):"
                        f"\n{joined_info_string}\n"
                        f"Where {potcar_template_entry.name} was used as the reference entry "
                        f"calculation.\n"
                        f"In general, the same POTCAR settings should be used in all final calculations "
                        f"for these tags which can affect energies!"
                    )

        # sort extrinsic elements and energies dict by periodic table positioning (deterministically),
        # and add to self.elements:
        self.extrinsic_elements = sorted(self.extrinsic_elements, key=_element_sort_func)
        self.elemental_energies = dict(
            sorted(self.elemental_energies.items(), key=lambda x: _element_sort_func(x[0]))
        )
        self.elements += self.extrinsic_elements

        # TODO: Warn if any missing elemental phases and remove any entries with them in composition,
        #  and remove from element lists?
        # set(Composition(d["Formula"]).elements).issubset(self.composition.elements)
        # or (
        #         extrinsic_elements
        #         and any(
        #     elt in Composition(d["Formula"]).elements for elt in extrinsic_elements)
        # )

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
                    f"indicating an issue with parsing, and so will be skipped for calculating the "
                    f"chemical potential limits."
                )
                # TODO: Update to remove this entry
                continue

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
        self.chempots_df = self.calculate_chempots(verbose=False)

    def _from_vaspruns(
        self,
        path: PathLike | list[PathLike] = "CompetingPhases",
        subfolder: PathLike | None = "vasp_std",
        verbose: bool = True,
        processes: int | None = None,
        check_compatibility: bool = True,
    ):
        r"""
        Parses competing phase energies from ``vasprun.xml(.gz)`` outputs,
        generating ``ComputedStructureEntry``\s and then continuing
        initialisation with the ``CompetingPhasesAnalyzer._from_entries``
        method.

        Args:
            path (PathLike or list):
                Either a path to the base folder containing competing phase
                calculation outputs (e.g.
                ``path/formula_EaH_X/{subfolder}/vasprun.xml(.gz)``, or
                ``path/formula_EaH_X/vasprun.xml(.gz)``), or a list of
                strings/Paths to ``vasprun.xml(.gz)`` files.
            subfolder (PathLike):
                The subfolder containing vasprun.xml(.gz) output files in
                each calculation directory within ``path`` (e.g. a file
                hierarchy like:
                ``path/formula_EaH_X/{subfolder}/vasprun.xml(.gz)``). Default
                is to search for ``vasp_std`` subfolders, or directly in the
                ``formula_EaH_X`` folder.
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
        # TODO: Change this to just recursively search for vaspruns within the specified path (also
        #  currently doesn't seem to revert to searching for vaspruns in the base folder if no vasp_std
        #  subfolders are found) -- see how this is done in DefectsParser in analysis.py, using the
        #  default glob criteria and multiple vaspruns warnings
        skipped_folders = []

        if isinstance(path, list):  # if path is just a list of all competing phases
            for p in path:
                if "vasprun.xml" in str(p) and not str(p).startswith("."):
                    self.vasprun_paths.append(p)

                # try to find the file -- will always pick the first match for vasprun.xml*
                # TODO: Deal with this:
                elif len(list(Path(p).glob("vasprun.xml*"))) > 0:
                    vsp = next(iter(Path(p).glob("vasprun.xml*")))
                    self.vasprun_paths.append(str(vsp))

                else:
                    skipped_folders.append(p)

        elif isinstance(path, PathLike):
            for p in os.listdir(path):
                if os.path.isdir(os.path.join(path, p)) and not str(p).startswith("."):
                    vr_path = "null_directory"
                    with contextlib.suppress(FileNotFoundError):
                        vr_path, multiple = _get_output_files_and_check_if_multiple(
                            "vasprun.xml", f"{os.path.join(path, p)}/{subfolder}"
                        )
                        if multiple:
                            folder_name = (
                                f"{os.path.join(path, p)}/{subfolder}"
                                if subfolder
                                else os.path.join(path, p)
                            )
                            warnings.warn(
                                f"Multiple `vasprun.xml` files found in directory: {folder_name}. Using "
                                f"{vr_path} to parse the calculation energy and metadata."
                            )

                    if os.path.exists(vr_path):
                        self.vasprun_paths.append(vr_path)

                    else:
                        with contextlib.suppress(FileNotFoundError):
                            vr_path, multiple = _get_output_files_and_check_if_multiple(
                                "vasprun.xml", os.path.join(path, p)
                            )
                            if multiple:
                                warnings.warn(
                                    f"Multiple `vasprun.xml` files found in directory: "
                                    f"{os.path.join(path, p)}. Using {vr_path} to parse the calculation "
                                    f"energy and metadata."
                                )

                        if os.path.exists(vr_path):
                            self.vasprun_paths.append(vr_path)

                        else:
                            skipped_folders += [f"{p} or {p}/{subfolder}"]
        else:
            raise ValueError(
                "`path` should either be a path to a folder (with competing phase "
                "calculations), or a list of paths to vasprun.xml(.gz) files."
            )

        # only warn about skipped folders that are recognised calculation folders (containing a material
        # composition in the name, or 'EaH' in the name)
        skipped_folders_for_warning = []
        for folder_name in skipped_folders:
            comps = []
            for i in folder_name.split(" or ")[0].split("_"):
                with contextlib.suppress(ValueError):
                    comps.append(Composition(i))
            if "EaH" in folder_name or comps:
                skipped_folders_for_warning.append(folder_name)

        if skipped_folders_for_warning and verbose:
            parent_folder_string = f" (in {path})" if isinstance(path, PathLike) else ""
            warnings.warn(
                f"vasprun.xml files could not be found in the following "
                f"directories{parent_folder_string}, and so they will be skipped for parsing:\n"
                + "\n".join(skipped_folders_for_warning)
            )

        # Ignore POTCAR warnings when loading vasprun.xml
        # pymatgen assumes the default PBE with no way of changing this
        _ignore_pmg_warnings()

        self.entries = []
        failed_parsing_dict: dict[str, list] = {}
        if processes is None:  # multiprocessing?
            # from quick tests; Pool takes about 2.75s to initialise, with negligible additional cost per
            # process, and vasprun parsing with pymatgen v2025.1.9 takes ~0.025 s/MB (uncompressed file
            # size; with gzip compressing large vasprun.xml files by ~15-20x) -- (~1.5-3x faster after SK's
            # updates to vasprun parsing in pymatgen v2025.4.16).
            # So for multiprocessing to be worth it, we need at least 2 vaspruns to parse, with a summed
            # uncompressed vasprun file size, excluding the largest one, of around >100 Mb
            def _estimate_uncompressed_vasprun_size(vasprun_path: PathLike) -> float:
                return (os.path.getsize(vasprun_path) / 1e6) * (20 if vasprun_path.endswith(".gz") else 1)

            vasprun_sizes_MB = [_estimate_uncompressed_vasprun_size(v) for v in self.vasprun_paths] or [0]
            mp = get_mp_context()
            if sum(vasprun_sizes_MB) - max(vasprun_sizes_MB) > 100:
                # only multiprocess as much as makes sense:
                processes = min(max(1, mp.cpu_count() - 1), sum(1 for s in vasprun_sizes_MB if s > 20) - 1)
            else:
                processes = 1

        parsing_results = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnconvergedVASPWarning)  # checked and warned later
            if processes > 1:  # multiprocessing
                with pool_manager(processes) as pool:  # result is parsed vasprun
                    for result in tqdm(
                        pool.imap_unordered(
                            _parse_entry_from_vasprun_and_catch_exception, self.vasprun_paths
                        ),
                        total=len(self.vasprun_paths),
                        desc="Parsing vaspruns...",
                    ):
                        parsing_results.append(result)
            else:
                for vasprun_path in tqdm(self.vasprun_paths, desc="Parsing vaspruns..."):
                    parsing_results.append(_parse_entry_from_vasprun_and_catch_exception(vasprun_path))

        electronic_unconverged_vaspruns = []
        ionic_unconverged_vaspruns = []
        for result in parsing_results:
            if isinstance(result[0], ComputedEntry | ComputedStructureEntry):
                # successful parse; result is entry, parsed folder, converged electronic and ionic
                self.entries.append(result[0])
                self.parsed_folders.append(result[1])
                if not result[2]:
                    electronic_unconverged_vaspruns.append(result[1])
                if not result[3]:
                    ionic_unconverged_vaspruns.append(result[1])
            else:  # failed parse; result is error message and path
                if str(result[0]) in failed_parsing_dict:
                    failed_parsing_dict[str(result[0])] += [result[1]]
                else:
                    failed_parsing_dict[str(result[0])] = [result[1]]

        if failed_parsing_dict:
            warning_string = (
                "Failed to parse the following `vasprun.xml` files:\n(files: error)\n"
                + "\n".join([f"{paths}: {error}" for error, paths in failed_parsing_dict.items()])
            )
            warnings.warn(warning_string)

        # check if any vaspruns are unconverged, and warn together:
        for unconverged_vaspruns, unconverged_type in zip(
            [electronic_unconverged_vaspruns, ionic_unconverged_vaspruns],
            ["Electronic", "Ionic"],
            strict=False,
        ):
            if unconverged_vaspruns:
                warnings.warn(
                    f"{unconverged_type} convergence was not reached for vaspruns in:\n"
                    + "\n".join(unconverged_vaspruns)
                )

        if not self.entries:
            raise FileNotFoundError(
                "No vasprun files have been parsed, suggesting issues with parsing! Please check that "
                "folders and input parameters are in the correct format (see docstrings/tutorials)."
            )

        return self._from_entries(self.entries, check_compatibility=check_compatibility)

    def as_dict(self) -> dict:
        """
        Returns:
            JSON-serializable dict representation of
            ``CompetingPhasesAnalyzer``.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "composition": self.composition.as_dict(),
            "entries": self.entries,
            "unstable_host": self.unstable_host,
            "bulk_entry": self.bulk_entry,
            "parsed_folders": self.parsed_folders,
            "vasprun_paths": self.vasprun_paths,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompetingPhasesAnalyzer":
        """
        Reconstitute a ``CompetingPhasesAnalyzer`` object from a dict
        representation created using ``as_dict()``.

        Args:
            d (dict): dict representation of ``CompetingPhasesAnalyzer``.

        Returns:
            ``CompetingPhasesAnalyzer`` object
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
        cpa.vasprun_paths = d.get("vasprun_paths", cpa.vasprun_paths)
        return cpa

    def get_formation_energy_df(
        self,
        prune_polymorphs: bool = False,
        include_dft_energies: bool = False,
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
            include_dft_energies (bool):
                Whether to include the raw DFT energies in the output
                ``DataFrame``. Default is ``False``.
            skip_rounding (bool):
                Whether to skip rounding the energies to 3 decimal places
                (1 meV/atom or meV/fu) for cleanliness. Default is ``False``.
        """
        data = []
        for entry in self.entries:
            comp = entry.composition
            formulas_per_unit = comp.get_reduced_composition_and_factor()[1]
            kpoints_data = entry.data.get("kpoints", None)
            kpoints = (
                "x".join(str(x) for x in kpoints_data[0])
                if (kpoints_data and len(kpoints_data) == 1)
                else "N/A"
            )
            space_group = (
                entry.data.get("doped_name", f"{entry.name}_N/A_EaH_")
                .split(f"{entry.name}_")[1]
                .split("_EaH_")[0]
            )

            d = {
                "Formula": comp.reduced_formula,
                "Space Group": space_group,
                "Energy above Hull (eV/atom)": entry.data.get("energy_above_hull", "N/A"),
                "Formation Energy (eV/fu)": self.phase_diagram.get_form_energy(entry) / formulas_per_unit,
                "Formation Energy (eV/atom)": self.phase_diagram.get_form_energy_per_atom(entry),
                "DFT Energy (eV/fu)": entry.energy / formulas_per_unit,
                "DFT Energy (eV/atom)": entry.energy_per_atom,
                "k-points": kpoints,
            }
            data.append(d)

        formation_energy_df = pd.DataFrame(data)

        if prune_polymorphs:  # only keep the lowest energy polymorphs
            indices = formation_energy_df.groupby("Formula")["DFT Energy (eV/atom)"].idxmin()
            formation_energy_df = formation_energy_df.loc[sorted(indices)]  # retain ordering

        if not include_dft_energies:
            formation_energy_df = formation_energy_df.drop(
                columns=["DFT Energy (eV/fu)", "DFT Energy (eV/atom)"]
            )
        formation_energy_df = formation_energy_df.round(3) if not skip_rounding else formation_energy_df

        return formation_energy_df.set_index("Formula")

    def calculate_chempots(
        self,
        extrinsic_species: str | Element | list[str] | list[Element] | None = None,
        sort_by: str | None = None,
        verbose: bool = True,
    ):
        """
        Calculates the chemical potential limits for the host composition
        (``self.composition``).

        If ``extrinsic_species`` (i.e. dopant/impurity elements) is specified,
        then the limiting chemical potential for ``extrinsic_species`` at the
        `intrinsic` chemical potential limits is calculated and also returned
        (corresponds to ``full_sub_approach=False`` in pycdt).
        ``extrinsic_species`` is set to ``self.extrinsic_elements`` if not
        specified.

        Args:
            extrinsic_species (str, Element, list):
                If set, will calculate the limiting chemical potential for the
                specified extrinsic species at the intrinsic chemical potential
                limits. Can be a single element (str or ``Element``), or a list
                of elements. If ``None`` (default), uses
                ``self.extrinsic_elements``.
            sort_by (str):
                If set, will sort the chemical potential limits in the output
                ``DataFrame`` according to the chemical potential of the
                specified element (from element-rich to element-poor
                conditions).
            verbose (bool):
                If ``True`` (default), will print the parsed chemical potential
                limits.

        Returns:
            ``pandas`` ``DataFrame``, optionally saved to csv.
        """
        if extrinsic_species is None:
            extrinsic_species = self.extrinsic_elements
        if not isinstance(extrinsic_species, list):
            extrinsic_species = [extrinsic_species]
        extrinsic_elements: list[Element] = [Element(e) for e in extrinsic_species]

        self.intrinsic_chempots = get_doped_chempots_from_entries(
            self.intrinsic_phase_diagram.entries,
            self.composition,
            sort_by=sort_by,
            single_chempot_limit=self.unstable_host,
        )

        chempots_df = pd.DataFrame.from_dict(  # chemical potentials as pandas dataframe
            {k: list(v.values()) for k, v in self.intrinsic_chempots["limits_wrt_el_refs"].items()},
            orient="index",
            columns=[str(k) for k in next(iter(self.intrinsic_chempots["limits_wrt_el_refs"].values()))],
        ).rename_axis("Limit")

        missing_extrinsic = [
            elt for elt in extrinsic_elements if elt.symbol not in self.elemental_energies
        ]
        if not extrinsic_elements:  # intrinsic only
            self.chempots = self.intrinsic_chempots
        elif missing_extrinsic:
            raise ValueError(  # TODO: Test this
                f"Elemental reference phase for the specified extrinsic species "
                f"{[elt.symbol for elt in missing_extrinsic]} was not parsed, but is necessary for "
                f"chemical potential calculations. Please ensure that this phase is present in the "
                f"calculation directory and is being correctly parsed."
            )
        else:
            self._calculate_extrinsic_chempot_lims(  # updates self.chempots and chempots_df
                extrinsic_elements=extrinsic_elements,
                chempots_df=chempots_df,
            )

        if verbose:
            print("Calculated chemical potential limits (in eV wrt elemental reference phases): \n")
            print(chempots_df)

        return chempots_df  # TODO: Test chempots df as a property

    @property
    def chempot_grid(self) -> ChemicalPotentialGrid:
        """
        ``ChemicalPotentialGrid`` object for the chemical potential limits of
        the host composition.

        This object can be used for plotting and numerical analyses of chemical
        stability regions. See the ``ChemicalPotentialGrid`` class for more
        details.
        """
        return ChemicalPotentialGrid(self.chempots)

    # TODO: This code (in all this module) should be rewritten to be more readable (re-used and
    #  uninformative variable names, missing informative comments, typing...)
    def _calculate_extrinsic_chempot_lims(self, extrinsic_elements, chempots_df):
        # TODO: At present, this does not work for codoping I believe?
        # for each intrinsic chemical potential limit, find the most stable extrinsic competing phase
        # (equivalent to most negative μ_extrinsic_elt):
        for extrinsic_elt in extrinsic_elements:
            for limit, chempot_series in chempots_df.iterrows():
                chempots_df.loc[limit, extrinsic_elt.symbol] = np.inf
                chempots_df.loc[limit, f"{extrinsic_elt.symbol}-Limiting Phase"] = "N/A"
                for entry in self.extrinsic_entries:
                    formation_energy = self.phase_diagram.get_form_energy(entry)
                    mu_extrinsic = (
                        formation_energy
                        - sum(
                            [
                                chempot_series[elt.symbol] * entry.composition[elt]
                                for elt in self.composition.elements
                            ]
                        )
                    ) / entry.composition[extrinsic_elt]
                    if mu_extrinsic < chempots_df.loc[limit, extrinsic_elt.symbol] and (
                        mu_extrinsic not in [-np.inf, np.inf, np.nan]
                    ):  # lower energy entry & μ_extrinsic_elt, and finite
                        chempots_df.loc[limit, extrinsic_elt.symbol] = _custom_round(mu_extrinsic, 4)
                        chempots_df.loc[limit, f"{extrinsic_elt.symbol}-Limiting Phase"] = entry.name

        # move limiting phase columns to the end (for cases with multiple extrinsic elements)
        chempots_df = chempots_df[
            [c for c in chempots_df.columns if "-Limiting Phase" not in c]
            + [c for c in chempots_df.columns if "-Limiting Phase" in c]
        ]

        # reverse engineer chemical potential limits dict with extrinsic entries
        chempot_lim_dict_list = chempots_df.copy().to_dict(orient="records")
        chempot_lims_w_extrinsic = {
            "elemental_refs": self.elemental_energies,
            "limits_wrt_el_refs": {},
            "limits": {},
        }

        for i, d in enumerate(chempot_lim_dict_list):
            key = (
                list(self.intrinsic_chempots["limits_wrt_el_refs"].keys())[i]
                + "-"
                + "-".join(d[col_name] for col_name in d if "Limiting Phase" in col_name)
            )
            new_vals = list(self.intrinsic_chempots["limits_wrt_el_refs"].values())[i]
            for extrinsic_elt in extrinsic_elements:
                new_vals[f"{extrinsic_elt.symbol}"] = d[f"{extrinsic_elt.symbol}"]
            chempot_lims_w_extrinsic["limits_wrt_el_refs"][key] = new_vals

        # relate the limits to the elemental energies
        for limit, chempot_dict in chempot_lims_w_extrinsic["limits_wrt_el_refs"].items():
            relative_chempot_dict = copy.deepcopy(chempot_dict)
            for e in relative_chempot_dict:
                relative_chempot_dict[e] += chempot_lims_w_extrinsic["elemental_refs"][e]
            chempot_lims_w_extrinsic["limits"].update({limit: relative_chempot_dict})

        # round all floats to 4 decimal places (0.1 meV/atom) for cleanliness (well below DFT accuracy):
        self.chempots = _round_floats(chempot_lims_w_extrinsic, 4)

    def to_LaTeX_table(self, splits=1, prune_polymorphs=True):
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

        if splits == 1:
            string += "\\begin{tabular}" + ("{ccccc}" if kpoints_col else "{cccc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " \\\\ \\hline \n"
            for i in formation_energy_data:
                kpoints = i.get("k-points", "0x0x0").split("x")
                fe = i["Formation Energy (eV/fu)"]
                string += (
                    "\\ce{"
                    + i["Formula"]
                    + "}"
                    + " & "
                    + latexify_spacegroup(i.get("Space Group", "N/A"))
                    + " & "
                    + f"{i['Energy above Hull (eV/atom)']:.3f}"
                    + (f" & {kpoints[0]}$\\times${kpoints[1]}$\\times${kpoints[2]}" if kpoints_col else "")
                    + " & "
                    + f"{fe:.3f} \\\\ \n"
                )

        elif splits == 2:
            string += "\\begin{tabular}" + ("{ccccc|ccccc}" if kpoints_col else "{cccc|cccc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " & " + column_names_string + " \\\\ \\hline \n"

            mid = len(formation_energy_data) // 2
            first_half = formation_energy_data[:mid]
            last_half = formation_energy_data[mid:]

            for i, j in zip(first_half, last_half, strict=False):
                kpoints1 = i.get("k-points", "0x0x0").split("x")
                fe1 = i["Formation Energy (eV/fu)"]
                kpoints2 = j.get("k-points", "0x0x0").split("x")
                fe2 = j["Formation Energy (eV/fu)"]
                string += (
                    "\\ce{"
                    + i["Formula"]
                    + "}"
                    + " & "
                    + latexify_spacegroup(i.get("Space Group", "N/A"))
                    + " & "
                    + f"{i['Energy above Hull (eV/atom)']:.3f}"
                    + (
                        f" & {kpoints1[0]}$\\times${kpoints1[1]}$\\times${kpoints1[2]}"
                        if kpoints_col
                        else ""
                    )
                    + " & "
                    + f"{fe1:.3f} & "
                    + "\\ce{"
                    + j["Formula"]
                    + "}"
                    + " & "
                    + latexify_spacegroup(j.get("Space Group", "N/A"))
                    + " & "
                    + f"{j['Energy above Hull (eV/atom)']:.3f}"
                    + (
                        f" & {kpoints2[0]}$\\times${kpoints2[1]}$\\times${kpoints2[2]}"
                        if kpoints_col
                        else ""
                    )
                    + " & "
                    + f"{fe2:.3f} \\\\ \n"
                )

        string += "\\hline\n"
        string += "\\end{tabular}\n"
        string += "\\end{table}"

        print(string)

    def plot_chempot_heatmap(
        self,
        dependent_element: str | Element | None = None,
        fixed_elements: dict[str, float] | None = None,
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
        Plot a heatmap of the chemical potentials for a multinary system.

        In this plot, the ``dependent_element`` chemical potential is plotted
        as a heatmap over the stability region of the host composition, as a
        function of two other elemental chemical potentials on the x and y
        axes.

        3-D data is required to plot a 2-D heatmap, and so this function can be
        applied as-is for ternary systems, but for higher-dimensional systems
        a set of chemical potential constraints must be provided (as
        ``fixed_elements``) to project the chemical stability region to 3-D;
        see the competing phases tutorial:
        https://doped.readthedocs.io/en/latest/chemical_potentials_tutorial.html#analysing-and-visualising-the-chemical-potential-limits

        Note that due to an issue with ``matplotlib`` ``Stroke`` path effects,
        sometimes there can be odd holes in the whitespace around the chemical
        formula labels (see: github.com/matplotlib/matplotlib/issues/25669).
        This is only the case for ``png`` output, so saving to e.g. ``svg``
        or ``pdf`` instead will avoid this issue.

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
                system is >3-D. Constraining chemical potentials is required for
                multinary systems, in order to reduce the dimensionality to 3-D
                for plotting a 2-D heatmap. For a system with N elements, N-3
                fixed chemical potentials should be specified. If ``None``
                (default), the chemical potentials of the first N-3 elements in
                the bulk composition are fixed to their mean values in the
                stability region (i.e. the centroid of the stability region).
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
                ``ChemicalPotentialDiagram.get_grid()``, such as ``n_points``
                (default = 1000) and ``cartesian`` (default = ``False``).

        Returns:
            plt.Figure: The ``matplotlib`` ``Figure`` object.
        """
        # TODO: Option to show _all_ calculated competing phases? (Not just bordering)
        # TODO: Plot extrinsic too? (after full_sub_approach etc re-checked)
        # Note that we could also add option to instead plot competing phases lines coloured,
        # with a legend added giving the composition of each competing phase line (as in the SI of
        # 10.1021/acs.jpcc.3c05204; Cs2SnTiI6 notebooks), but this isn't as nice/clear, and the same effect
        # can be achieved by the user by saving to PDF without labels, and manually colouring and adding
        # a legend in a vector graphics editor (e.g. Inkscape, Affinity Designer, Adobe Illustrator, etc.).
        from shakenbreak.plotting import _install_custom_font

        _install_custom_font()
        cpd = ChemicalPotentialDiagram(list(self.intrinsic_phase_diagram.entries))  # change to
        # self.entries when extrinsic plotting functionality added

        host_domains = cpd.domains[self.composition.reduced_formula]
        cpg = ChemicalPotentialGrid.from_dataframe(
            pd.DataFrame(host_domains, columns=[el.symbol for el in cpd.elements])
        )
        fixed_elements = fixed_elements or {}
        input_variable_elements = [
            el for el in self.composition.elements if el.symbol not in fixed_elements
        ]
        if dependent_element is None:  # set to last element in bulk comp, usually the anion as desired
            dependent_element = input_variable_elements[-1]
        elif isinstance(dependent_element, str):
            dependent_element = Element(dependent_element)
        assert isinstance(dependent_element, Element)  # typing
        input_variable_elements.remove(dependent_element)

        # check dimensionality:
        if len(cpd.elements) == 2:  # switch to line plot
            raise ValueError(
                "Chemical potential heatmap (i.e. 2D) plotting is not possible for a binary system!"
            )  # TODO: Change to warning and auto switch to line plot?
        if len(cpd.elements) - len(fixed_elements) != 3:  # auto fix to centroid of stability region:
            info_message = (
                f"Chemical potential heatmap plotting requires 3-D data, requiring fixed chemical "
                f"potential constraints for >ternary systems; such that the number of elements in the "
                f"chemical system ({len(cpd.elements)}) minus the number of fixed chemical potentials "
                f"({len(fixed_elements)}) must be equal to 3."
            )
            if len(cpd.elements) - len(fixed_elements) < 3:
                raise ValueError(info_message)
            centroid = cpg.get_grid(cartesian=True, include_vertices=False).mean(axis=0)
            req_num_constraints = len(cpd.elements) - 3
            additional_fixed_elements = {
                el.symbol: round(centroid[el.symbol], 4)
                for i, el in enumerate(input_variable_elements)
                if i < req_num_constraints
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

        # Generate grid data
        grid_kwargs: dict[str, Any] = {"n_points": 1000, "cartesian": False}
        grid_kwargs.update(kwargs)
        if fixed_elements:
            grid_data = cpg.get_constrained_grid(fixed_elements, **grid_kwargs)
        else:
            grid_data = cpg.get_grid(**grid_kwargs)
        values_inside = grid_data[dependent_element.symbol].to_numpy()
        points_inside = grid_data.drop(columns=[dependent_element.symbol]).to_numpy()
        tri = Triangulation(points_inside[:, 0], points_inside[:, 1])

        # Create plot
        style_file = style_file or f"{os.path.dirname(__file__)}/utils/doped.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
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
        x_max, y_max = points_inside.max(axis=0)
        x_min, y_min = points_inside.min(axis=0)
        x_padding = padding or abs(x_max - x_min) * 0.1
        y_padding = padding or abs(y_max - y_min) * 0.1

        if xlim is None:
            xlim = (float(x_min - x_padding), float(x_max + x_padding))

        if ylim is None:
            ylim = (float(y_min - y_padding), float(y_max + y_padding))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        cbar.set_label(rf"$\Delta\mu$ ({dependent_element.symbol}) (eV)")
        ax.set_xlabel(rf"$\Delta\mu$ ({independent_elts[0].symbol}) (eV)")
        ax.set_ylabel(rf"$\Delta\mu$ ({independent_elts[1].symbol}) (eV)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        if title:  # add title
            if not isinstance(title, str):
                title = latexify(f"{self.composition.reduced_formula}")
            ax.set_title(title)

        self._plot_competing_phase_lines(  # plot competing phase lines and labels
            ax, cpd, host_domains, fixed_elements, independent_elts, label_positions
        )

        self._nudge_labels_inside_axes(ax, padding)  # adjust label positions to stay within plot bounds

        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=600)

        return fig

    def _plot_competing_phase_lines(
        self,
        ax: plt.Axes,
        cpd: ChemicalPotentialDiagram,
        host_domains: np.ndarray,
        fixed_elements: dict[str, float],
        independent_elts: list[Element],
        label_positions: bool | dict[str, tuple[float, float]] | list[tuple[float, float]] = True,
    ) -> None:
        """
        Plot competing phase lines and add labels.
        """
        lines = {}  # {formula: matplotlib line object}
        labels = {}  # {formula: line function}
        intersections = []
        x = np.linspace(-50, 50, 1000)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        for formula, pts in cpd.domains.items():
            if formula == self.composition.reduced_formula:
                continue

            # Get domain points that match host domains
            domain_pts = [
                chempot_coords
                for chempot_coords in pts
                if np.any(np.all(np.isclose(host_domains, chempot_coords), axis=1))  # (M, k)
            ]
            if len(domain_pts) < 2:
                continue  # not a stable bordering phase

            try:
                domain_pts = self._apply_fixed_element_constraints(
                    domain_pts, fixed_elements, cpd.elements
                )  # handle fixed elements by intersecting with planes
            except ValueError:
                continue  # no intersection with plane, skip to next phase

            # Fit line function
            formula_x_vals = np.array(domain_pts)[:, cpd.elements.index(independent_elts[0])]
            formula_y_vals = np.array(domain_pts)[:, cpd.elements.index(independent_elts[1])]
            if np.isclose(min(formula_x_vals), max(formula_x_vals), atol=5e-5):  # vertical line
                m = np.inf
                b = formula_x_vals[0]
            else:
                m, b = np.polyfit(formula_x_vals, formula_y_vals, 1)

            def f(xx, m=m, b=b):  # line function for the fitted line
                return m * xx + b

            # Check for vertical lines
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "divide by zero")
                vertical_line = (np.abs(f(x)) == np.inf).any() or (
                    f(x).max() - f(x).min() > 1e5  # basically a vertical line, account for rounding
                )

            # Plot line and get intersections
            line, intersection = self._plot_single_phase_line(
                ax, formula, domain_pts, cpd, independent_elts, vertical_line, f, x
            )

            if intersection is not None and np.size(intersection) >= 4:
                if np.size(intersection) == 4:
                    # if 4 intersections, check if the midpoint of intersections is at an edge (skip if so)
                    midpoint = np.mean(intersection, axis=0)
                    if np.any(np.abs(midpoint - np.array([x_min, y_min])) < 1e-4) or np.any(
                        np.abs(midpoint - np.array([x_max, y_max])) < 1e-4
                    ):
                        continue

                intersections.append(intersection)
                lines[formula] = line
                labels[formula] = f

        if label_positions:  # add labels to lines
            self._add_line_labels(
                intersections=intersections,
                lines=lines,
                x_range=abs(x_max - x_min),
                y_range=abs(y_max - y_min),
                label_positions=label_positions,
            )

    def _apply_fixed_element_constraints(
        self, domain_pts: list, fixed_elements: dict[str, float], elements: list[Element]
    ) -> np.ndarray:
        """
        Apply fixed element constraints by intersecting with planes.
        """
        domain_pts = np.array(domain_pts)
        for element, value in fixed_elements.items():
            domain_pts = _intersect_hull_with_plane(
                domain_pts,
                elements.index(Element(element)),
                value,
            )
        return domain_pts

    def _plot_single_phase_line(
        self,
        ax: plt.Axes,
        formula: str,
        domain_pts: np.ndarray,
        cpd: ChemicalPotentialDiagram,
        independent_elts: list[Element],
        vertical_line: bool,
        f: Callable,
        x: np.ndarray,
    ) -> tuple[plt.Line2D, np.ndarray | None]:
        """
        Plot a single competing phase line and return line object and
        intersections.
        """
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        if vertical_line:
            x_val = domain_pts[0][cpd.elements.index(independent_elts[0])]
            line = ax.axvline(x_val, label=latexify(formula), color="k")
            if x_val < xlim[1] and x_val > xlim[0]:
                intersection = ((x_val, ylim[0]), (x_val, ylim[1]))
            else:
                intersection = None
        else:
            (line,) = ax.plot(x, f(x), label=latexify(formula), color="k")
            intersection = self._get_line_intersections(f, xlim, ylim)

        return line, intersection

    def _get_line_intersections(
        self, f: Callable, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> np.ndarray:
        """
        Get intersections of a line with the plot bounding box.
        """
        x_min, x_max = xlim
        y_min, y_max = ylim
        x_intersections = []
        y_intersections = []

        # Check intersections with vertical bounds (x_min and x_max)
        y_at_x_min = f(x_min)
        y_at_x_max = f(x_max)
        if y_min <= y_at_x_min <= y_max:
            x_intersections.append((x_min, float(y_at_x_min)))
        if y_min <= y_at_x_max <= y_max:
            x_intersections.append((x_max, float(y_at_x_max)))

        # Check intersections with horizontal bounds (y_min and y_max)
        if not np.isclose(float(y_at_x_min), float(y_at_x_max), atol=1e-4):  # not a horizontal line

            def inv_f(yy):  # inverse of the line function
                return (yy - f(0)) / (f(1) - f(0)) if f(1) != f(0) else x_min

            x_at_y_min = inv_f(y_min)
            x_at_y_max = inv_f(y_max)
            if x_min <= x_at_y_min <= x_max:
                y_intersections.append((float(x_at_y_min), y_min))
            if x_min <= x_at_y_max <= x_max:
                y_intersections.append((float(x_at_y_max), y_max))

        # take unique points in case intersects at x/y corner (which would give a duplicate):
        return np.unique(np.round((x_intersections + y_intersections), 4), axis=0)

    def _add_line_labels(
        self,
        intersections: list,
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
            label_positions, best_norm_min_dist = _find_best_label_positions(
                poss_label_positions, x_range=x_range, y_range=y_range, return_best_norm_dist=True
            )
            if best_norm_min_dist < 0.1:  # bump positions_per_line to 5 to try improve:
                poss_label_positions = _possible_label_positions_from_bbox_intersections(
                    intersections, positions_per_line=5
                )
                label_positions, best_norm_min_dist = _find_best_label_positions(
                    poss_label_positions, x_range=x_range, y_range=y_range, return_best_norm_dist=True
                )

        elif isinstance(label_positions, dict):  # pre-set label positions, match formula (key) to line:
            lines = {k: lines[k] for k in lines if k in label_positions}  # drop any without positions
            label_positions = [label_positions[k] for k in lines]  # reorder to match lines

        if isinstance(label_positions, list):
            label_positions = np.array(label_positions, dtype=float)

        assert isinstance(label_positions, np.ndarray)  # typing; converted to array now
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The value at position")
            labelLines(
                list(lines.values()),  # must be in same order as plotting order...
                xvals=label_positions[:, 0],
                yoffsets=label_positions[:, 1],
                align=False,
                color="black",
            )

    def _nudge_labels_inside_axes(self, ax: plt.Axes, padding: float | None) -> None:
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
            delta_x = delta_y = 0

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

    def __repr__(self):
        """
        Returns a string representation of the ``CompetingPhasesAnalyzer``
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


def _parse_entry_from_vasprun_and_catch_exception(
    vasprun_path: PathLike,
) -> tuple[str | Vasprun, PathLike, bool, bool]:
    """
    Parse a VASP ``vasprun.xml`` file into a ``ComputedStructureEntry``,
    catching any exceptions and returning the error message and the path to the
    ``vasprun.xml`` file if an exception is raised.
    """
    try:
        vasprun = get_vasprun(vasprun_path)
        entry = vasprun.get_computed_entry()
        unique_symbols = sorted(set(vasprun.atomic_symbols))
        summary_dict = {}
        with contextlib.suppress(Exception):  # non-essential properties, can fail with incomplete vasprun
            summary_dict["band_gap"] = vasprun.eigenvalue_band_properties[0]
            summary_dict["total_magnetization"] = get_magnetization_from_vasprun(vasprun)

        entry.data.update(
            {
                "formula_pretty": entry.composition.reduced_formula,
                "nsites": len(entry.structure),
                "volume": entry.structure.volume,
                "energy_per_atom": entry.energy_per_atom,
                "elements": unique_symbols,
                "nelements": len(unique_symbols),
                "kpoints": vasprun.kpoints.kpts,
                "incar": {k: v for k, v in vasprun.incar.as_dict().items() if "@" not in k},
                "potcar_symbols": vasprun.potcar_spec,
                "summary": summary_dict,
            }
        )
        electronic_converged = vasprun.converged_electronic
        ionic_converged = vasprun.converged_ionic
        folder = vasprun_path.rstrip(".gz").rstrip("vasprun.xml")
        return entry, folder, electronic_converged, ionic_converged
    except Exception as e:
        return str(e), vasprun_path, False, False


def _possible_label_positions_from_bbox_intersections(
    intersections: list[float] | np.ndarray[float], positions_per_line=3
) -> np.ndarray[float]:
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
    poss_label_positions, x_range=1, y_range=1, return_best_norm_dist=False
) -> np.ndarray | tuple[np.ndarray, float]:
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


def get_X_rich_limit(X: str, chempots: dict):
    """
    Determine the chemical potential limit of the input chempots dict which
    corresponds to the most X-rich conditions.

    Args:
        X (str):
            Elemental species (e.g. "Te")
        chempots (dict):
            The chemical potential limits dict, as returned by
            ``CompetingPhasesAnalyzer.chempots``
    """
    X_rich_limit = None
    X_rich_limit_chempot = None
    for limit, chempot_dict in chempots["limits"].items():
        if X in chempot_dict and (X_rich_limit is None or chempot_dict[X] > X_rich_limit_chempot):
            X_rich_limit = limit
            X_rich_limit_chempot = chempot_dict[X]

    if X_rich_limit is None:
        raise ValueError(f"Could not find {X} in the chemical potential limits dict:\n{chempots}")

    return X_rich_limit


def get_X_poor_limit(X: str, chempots: dict):
    """
    Determine the chemical potential limit of the input chempots dict which
    corresponds to the most X-poor conditions.

    Args:
        X (str):
            Elemental species (e.g. "Te")
        chempots (dict):
            The chemical potential limits dict, as returned by
            ``CompetingPhasesAnalyzer.chempots``
    """
    X_poor_limit = None
    X_poor_limit_chempot = None
    for limit, chempot_dict in chempots["limits"].items():
        if X in chempot_dict and (X_poor_limit is None or chempot_dict[X] < X_poor_limit_chempot):
            X_poor_limit = limit
            X_poor_limit_chempot = chempot_dict[X]

    if X_poor_limit is None:
        raise ValueError(f"Could not find {X} in the chemical potential limits dict:\n{chempots}")

    return X_poor_limit


def combine_extrinsic(first: dict, second: dict, extrinsic_species: str) -> dict:
    # TODO: Can we just integrate this to `CompetingPhaseAnalyzer`, so you just pass in a list of
    # extrinsic species and it does the right thing?
    """
    Combines chemical potential limits for different extrinsic species.

    Usage explained in the chemical potentials tutorial. To be removed in
    following versions.

    Args:
        first (dict):
            First chemical potential dictionary, it can contain extrinsic
            species other than the set extrinsic species.
        second (dict):
            Second chemical potential dictionary, it must contain the
            extrinsic species.
        extrinsic_species (str):
            Extrinsic species in the second dictionary.

    Returns:
        dict:
            Combined dictionary with limits for the extrinsic species.
    """
    keys = ["elemental_refs", "limits", "limits_wrt_el_refs"]
    if any(key not in first for key in keys):
        raise KeyError(
            "the first dictionary doesn't contain the correct keys -- it should include "
            "elemental_refs, limits and limits_wrt_el_refs"
        )

    if any(key not in second for key in keys):
        raise KeyError(
            "the second dictionary doesn't contain the correct keys -- it should include "
            "elemental_refs, limits and limits_wrt_el_refs"
        )

    if extrinsic_species not in second["elemental_refs"]:
        raise ValueError("extrinsic species is not present in the second dictionary")

    cpa1 = copy.deepcopy(first)
    cpa2 = copy.deepcopy(second)
    new_limits = {}
    for (k1, v1), (k2, v2) in zip(
        list(cpa1["limits"].items()), list(cpa2["limits"].items()), strict=False
    ):
        if k2.rsplit("-", 1)[0] in k1:
            new_key = k1 + "-" + k2.rsplit("-", 1)[1]
        else:
            raise ValueError("The limits aren't matching, make sure you've used the correct dictionary")

        v1[extrinsic_species] = v2.pop(extrinsic_species)
        new_limits[new_key] = v1

    new_limits_wrt_el_refs = {}
    for (k1, v1), (k2, v2) in zip(
        list(cpa1["limits_wrt_el_refs"].items()),
        list(cpa2["limits_wrt_el_refs"].items()),
        strict=False,
    ):
        if k2.rsplit("-", 1)[0] in k1:
            new_key = k1 + "-" + k2.rsplit("-", 1)[1]
        else:
            raise ValueError("The limits aren't matching, make sure you've used the correct dictionary")

        v1[extrinsic_species] = v2.pop(extrinsic_species)
        new_limits_wrt_el_refs[new_key] = v1

    new_elements = copy.deepcopy(cpa1["elemental_refs"])
    new_elements[extrinsic_species] = copy.deepcopy(cpa2["elemental_refs"])[extrinsic_species]

    return {
        "elemental_refs": new_elements,
        "limits": new_limits,
        "limits_wrt_el_refs": new_limits_wrt_el_refs,
    }
