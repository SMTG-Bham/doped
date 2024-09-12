"""
Functions for setting up and parsing competing phase calculations in order to
determine and analyse the elemental chemical potentials for defect formation
energies.
"""

import contextlib
import copy
import importlib.util
import itertools
import os
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labellines import labelLines
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
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
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning, Vasprun
from pymatgen.util.string import latexify
from pymatgen.util.typing import PathLike
from scipy.interpolate import griddata, interp1d
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from doped import _doped_obj_properties_methods, _ignore_pmg_warnings
from doped.generation import _element_sort_func
from doped.utils.parsing import _get_output_files_and_check_if_multiple, get_vasprun
from doped.utils.plotting import get_colormap
from doped.utils.symmetry import _round_floats, get_primitive_structure
from doped.vasp import MODULE_DIR, DopedDictSet, default_HSE_set, default_relax_set

# globally ignore:
_ignore_pmg_warnings()

pbesol_convrg_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/PBEsol_ConvergenceSet.yaml"))  # just INCAR

elemental_diatomic_gases = ["H2", "O2", "N2", "F2", "Cl2"]
elemental_diatomic_bond_lengths = {"O": 1.21, "N": 1.10, "H": 0.74, "F": 1.42, "Cl": 1.99}

old_MPRester_property_data = [  # properties to pull for Materials Project entries, if using legacy MP API
    "pretty_formula",  # otherwise uses all fields in ``mpr.materials.thermo.available_fields``
    "e_above_hull",
    "band_gap",
    "nsites",
    "volume",
    "icsd_id",
    "icsd_ids",  # some entries have icsd_id and some have icsd_ids
    "theoretical",
    "formation_energy_per_atom",  # uncorrected with legacy MP API, corrected with new API
    "energy_per_atom",  # note that with legacy MP API this is uncorrected, but is corrected with new API
    "energy",  # note that with legacy MP API this is uncorrected, but is corrected with new API
    "total_magnetization",
    "nelements",
    "elements",
]  # note that, because the energy values in the ``data`` dict are uncorrected with legacy MP API and
# corrected with the new MP API, we should refrain from using these values when possible. The ``energy``
# and ``energy_per_atom`` attributes are consistent (corrected in both cases)

MP_API_property_keys = {
    "legacy": {
        "energy_above_hull": "e_above_hull",
        "pretty_formula": "pretty_formula",
    },
    "new": {
        "energy_above_hull": "energy_above_hull",
        "pretty_formula": "formula_pretty",
    },
}


# TODO: Need to recheck all functionality from old `_chemical_potentials.py` is now present here.


def _get_pretty_formula(entry_data: dict):
    return entry_data.get("pretty_formula", entry_data.get("formula_pretty", "N/A"))


def _get_e_above_hull(entry_data: dict):
    return entry_data.get("e_above_hull", entry_data.get("energy_above_hull", 0.0))


def make_molecule_in_a_box(element: str):
    """
    Generate an X2 'molecule-in-a-box' structure for the input element X, (i.e.
    a 30 Å cuboid supercell with a single X2 molecule in the centre).

    This is the natural state of several elemental competing phases, such
    as O2, N2, H2, F2 and Cl2. Initial bond lengths are set to the experimental
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
    element = element[0].upper()  # in case provided as X2 etc
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


def make_molecular_entry(computed_entry, legacy_MP=False):
    """
    Generate a new ``ComputedStructureEntry`` for a molecule in a box, for the
    input elemental ``ComputedEntry``.

    The formula of the input ``computed_entry`` must be one of the
    supported diatomic molecules (O2, N2, H2, F2, Cl2).

    Args:
        computed_entry (ComputedEntry):
            ``ComputedEntry`` object for the elemental entry.
        legacy_MP (bool):
            If ``True``, use the legacy Materials Project property data fields
            (i.e. ``"e_above_hull"``, ``"pretty_formula"`` etc.), rather than
            the new Materials Project API format (``"energy_above_hull"``,
            ``"formula_pretty"`` etc.).
            Default is ``False``.
    """
    property_key_dict = MP_API_property_keys["legacy"] if legacy_MP else MP_API_property_keys["new"]
    assert len(computed_entry.composition.elements) == 1  # Elemental!
    formula = _get_pretty_formula(computed_entry.data)
    element = formula[0].upper()
    struct, total_magnetization = make_molecule_in_a_box(element)
    molecular_entry = ComputedStructureEntry(
        structure=struct,
        energy=computed_entry.energy_per_atom * 2,  # set entry energy to be hull energy
        composition=Composition(formula),
        parameters=None,
    )
    molecular_entry.data[property_key_dict["pretty_formula"]] = formula
    molecular_entry.data[property_key_dict["energy_above_hull"]] = 0.0
    molecular_entry.data["nsites"] = 2
    molecular_entry.data["volume"] = 27000
    molecular_entry.data["formation_energy_per_atom"] = 0.0
    molecular_entry.data["energy_per_atom"] = computed_entry.data["energy_per_atom"]
    molecular_entry.data["energy"] = computed_entry.data["energy_per_atom"] * 2
    molecular_entry.data["nelements"] = 1
    molecular_entry.data["elements"] = [formula]
    molecular_entry.data["molecule"] = True
    molecular_entry.data["band_gap"] = None  # not included by default in new MP entries
    molecular_entry.data["database_IDs"] = "N/A"
    molecular_entry.data["material_id"] = "mp-0"
    molecular_entry.data["icsd_id"] = None
    molecular_entry.data["total_magnetization"] = total_magnetization

    return molecular_entry


def _calculate_formation_energies(data: list, elemental: dict):
    """
    Calculate formation energies for a list of dictionaries, using the input
    elemental reference energies.

    Args:
        data (list):
            List of dictionaries containing the energy data of the
            phases to calculate formation energies for.
        elemental (dict): Dictionary of elemental reference energies.

    Returns:
        pd.DataFrame: ``DataFrame`` of formation energies of the input phases.
    """
    for d in data:
        for el in elemental:
            d[el] = Composition(d["Formula"]).as_dict().get(el, 0)

    formation_energy_df = pd.DataFrame(data)
    formation_energy_df["num_atoms_in_fu"] = formation_energy_df["Formula"].apply(
        lambda x: Composition(x).num_atoms
    )
    formation_energy_df["num_species"] = formation_energy_df["Formula"].apply(
        lambda x: len(Composition(x).as_dict())
    )
    formation_energy_df["periodic_group_ordering"] = formation_energy_df["Formula"].apply(
        lambda x: tuple(sorted([_element_sort_func(i.symbol) for i in Composition(x).elements]))
    )

    # get energy per fu then subtract elemental energies later, to get formation energies
    if "DFT Energy (eV/fu)" in formation_energy_df.columns:
        formation_energy_df["formation_energy_calc"] = formation_energy_df["DFT Energy (eV/fu)"]
        if "DFT Energy (eV/atom)" not in formation_energy_df.columns:
            formation_energy_df["DFT Energy (eV/atom)"] = formation_energy_df["DFT Energy (eV/fu)"] / (
                formation_energy_df["num_atoms_in_fu"]
            )

    elif "DFT Energy (eV/atom)" in formation_energy_df.columns:
        formation_energy_df["formation_energy_calc"] = (
            formation_energy_df["DFT Energy (eV/atom)"] * formation_energy_df["num_atoms_in_fu"]
        )
        formation_energy_df["DFT Energy (eV/fu)"] = formation_energy_df["DFT Energy (eV/atom)"] * (
            formation_energy_df["num_atoms_in_fu"]
        )

    else:
        raise ValueError(
            "No energy data (DFT Energy (eV/atom) or per Formula Unit (eV/fu)) found in input "
            "data to calculate formation energies!"
        )

    for k, v in elemental.items():
        formation_energy_df["formation_energy_calc"] -= formation_energy_df[k] * v

    formation_energy_df["Formation Energy (eV/fu)"] = formation_energy_df["formation_energy_calc"]
    formation_energy_df["Formation Energy (eV/atom)"] = (
        formation_energy_df["formation_energy_calc"] / formation_energy_df["num_atoms_in_fu"]
    )
    formation_energy_df = formation_energy_df.drop(columns=["formation_energy_calc"])

    # sort by num_species, then by periodic group/row of elements, then by num_atoms_in_fu,
    # then by formation_energy
    formation_energy_df = formation_energy_df.sort_values(
        by=["num_species", "periodic_group_ordering", "num_atoms_in_fu", "Formation Energy (eV/fu)"],
    )
    # drop num_atoms_in_fu and num_species
    return formation_energy_df.drop(columns=["num_atoms_in_fu", "num_species", "periodic_group_ordering"])


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
            Description for the ``EnergyAdjustment`` object to be added to the entry.
            Default is ``None``.

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


def get_chempots_from_phase_diagram(bulk_computed_entry, phase_diagram):
    """
    Get the chemical potential limits for the bulk computed entry in the
    supplied phase diagram.

    Args:
        bulk_computed_entry: ``ComputedStructureEntry`` object for the host composition.
        phase_diagram: ``PhaseDiagram`` object for the system of interest
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


def _get_all_chemsyses(chemsys: Union[str, list[str]]):
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


def _get_property_key_dict(legacy_MP: bool):
    """
    Get the appropriate property key dictionary, property data fields and
    kwargs for ``MPRester.get_entries()`` or
    ``MPRester.get_entries_in_chemsys()`` for the new or legacy Materials
    Project API (as given by ``legacy_MP``).

    Args:
        legacy_MP (bool):
            ``True`` if the API key corresponds to the legacy Materials Project
            API, ``False`` if it corresponds to the new Materials Project API.

    Returns:
        property_key_dict, property_data_fields, get_entries_kwargs:
            Tuple of the appropriate property key dictionary, property data fields
            and keyword arguments for the new or legacy Materials Project API
            functions.
    """
    if legacy_MP:
        property_key_dict = MP_API_property_keys["legacy"]
        property_data_fields = old_MPRester_property_data
        get_entries_kwargs = {
            "property_data": property_data_fields,
            "inc_structure": "initial",
        }

    else:
        from emmet.core.thermo import ThermoDoc  # emmet only required if mp-api installed

        property_key_dict = MP_API_property_keys["new"]
        property_data_fields = list(ThermoDoc.model_json_schema()["properties"].keys())
        get_entries_kwargs = {"property_data": property_data_fields}

    return property_key_dict, property_data_fields, get_entries_kwargs


def get_entries_in_chemsys(
    chemsys: Union[str, list[str]],
    api_key: Optional[str] = None,
    e_above_hull: Optional[float] = None,
    return_all_info: bool = False,
    bulk_composition: Optional[Union[str, Composition]] = None,
    **kwargs,
):
    """
    Convenience function to get a list of ``ComputedStructureEntry``s for an
    input chemical system, using ``MPRester.get_entries_in_chemsys()``.

    Automatically uses the appropriate format and syntax required for the
    new or legacy Materials Project (MP) APIs, depending on the type of API
    key supplied/present.
    ``chemsys = ["Li", "Fe", "O"]`` will return a list of all entries in
    the Li-Fe-O chemical system, i.e., all LixOy, FexOy, LixFey, LixFeyOz,
    Li, Fe and O phases. Extremely useful for creating phase diagrams of
    entire chemical systems.

    If ``e_above_hull`` is supplied, then only entries with energies above
    hull (according to the MP-computed phase diagram) less than this value
    (in eV/atom) will be returned.

    The output entries list is sorted by energy above hull, then by the number
    of elements in the formula, then by the position of elements in the
    periodic table (main group elements, then transition metals, sorted by row).

    Args:
        chemsys (str, list[str]):
            Chemical system to get entries for, in the format "A-B-C" or
            ["A", "B", "C"]. E.g. "Li-Fe-O" or ["Li", "Fe", "O"].
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            to obtain the corresponding ``ComputedStructureEntry``s. If not
            supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``)
            - see the ``doped`` Installation docs page:
            https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials
            -project-api
        e_above_hull (float):
            If supplied, only entries with energies above hull (according to the
            MP-computed phase diagram) less than this value (in eV/atom) will be
            returned. Set to 0 to only return phases on the MP convex hull.
            Default is ``None`` (i.e. all entries are returned).
        return_all_info (bool):
            If ``True``, also returns the ``property_key_dict`` and
            ``property_data_fields`` objects, which list the appropriate keys and data
            field names for the new or legacy Materials Project API (corresponding to
            the current API key). Mainly intended for internal ``doped`` usage for
            provenance tracking. Default is ``False``.
        bulk_composition (str/Composition):
            Optional input; formula of the bulk host material, to use for sorting
            the output entries (with all those matching the bulk composition first).
            Default is ``None``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            ``get_entries_in_chemsys()`` query.

    Returns:
        list[ComputedStructureEntry], dict, list:
            List of ``ComputedStructureEntry`` objects for the input chemical system.
            If ``return_all_info`` is ``True``, also returns the ``property_key_dict`` and
            ``property_data_fields`` objects, which list the appropriate keys and data field
            names for the new or legacy Materials Project API (corresponding to the current API key).
    """
    api_key, legacy_MP = _parse_MP_API_key(api_key)
    property_key_dict, property_data_fields, get_entries_kwargs = _get_property_key_dict(legacy_MP)

    with MPRester(api_key) as mpr:
        # get all entries in the chemical system
        MP_full_pd_entries = mpr.get_entries_in_chemsys(
            chemsys,
            **get_entries_kwargs,
            **kwargs,
        )

    temp_phase_diagram = PhaseDiagram(MP_full_pd_entries)
    for entry in MP_full_pd_entries:
        # reparse energy above hull, to avoid mislabelling issues noted in (legacy) Materials Project
        # database; e.g. search "F", or ZnSe2 on Zn-Se convex hull from MP PD, but EaH = 0.147 eV/atom?
        # or Immm phases for Br, I...
        entry.data[property_key_dict["energy_above_hull"]] = temp_phase_diagram.get_e_above_hull(entry)

    if e_above_hull is not None:
        MP_full_pd_entries = [
            entry for entry in MP_full_pd_entries if _get_e_above_hull(entry.data) <= e_above_hull
        ]

    # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
    MP_full_pd_entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=bulk_composition))

    if return_all_info:
        return MP_full_pd_entries, property_key_dict, property_data_fields

    return MP_full_pd_entries


def get_entries(
    chemsys_formula_id_criteria: Union[str, dict[str, Any]],
    api_key: Optional[str] = None,
    bulk_composition: Optional[Union[str, Composition]] = None,
    **kwargs,
):
    """
    Convenience function to get a list of ``ComputedStructureEntry``s for an
    input single composition/formula, chemical system, MPID or full criteria,
    using ``MPRester.get_entries()``.

    Automatically uses the appropriate format and syntax required for the
    new or legacy Materials Project (MP) APIs, depending on the type of API
    key supplied/present.

    The output entries list is sorted by energy per atom (equivalent sorting as
    energy above hull), then by the number of elements in the formula, then
    by the position of elements in the periodic table (main group elements,
    then transition metals, sorted by row).

    Args:
        chemsys_formula_id_criteria (str/dict):
            A formula (e.g., Fe2O3), chemical system (e.g., Li-Fe-O) or MPID
            (e.g., mp-1234) or full Mongo-style dict criteria.
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            to obtain the corresponding ``ComputedStructureEntry``s. If not
            supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``)
            - see the ``doped`` Installation docs page:
            https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials
            -project-api
        bulk_composition (str/Composition):
            Optional input; formula of the bulk host material, to use for sorting
            the output entries (with all those matching the bulk composition first).
            Default is ``None``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            ``get_entries()`` query.

    Returns:
        list[ComputedStructureEntry]:
            List of ``ComputedStructureEntry`` objects for the input chemical system.
    """
    api_key, legacy_MP = _parse_MP_API_key(api_key)
    property_key_dict, property_data_fields, get_entries_kwargs = _get_property_key_dict(legacy_MP)

    with MPRester(api_key) as mpr:
        entries = mpr.get_entries(
            chemsys_formula_id_criteria,
            **get_entries_kwargs,
            **kwargs,
        )

    # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
    entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=bulk_composition))

    return entries


def _parse_MP_API_key(api_key: Optional[str] = None, legacy_MP_info: bool = False):
    """
    Parse the Materials Project (MP) API key, either from the input argument or
    from the environment variable ``PMG_MAPI_KEY``, and determine if it
    corresponds to the new or legacy Materials Project API.

    Args:
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            (for phase diagram analysis, competing phase generation etc). If
            not supplied, will attempt to read from environment variable
            ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``)
            - see the ``doped`` Installation docs page:
            https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials
            -project-api
        legacy_MP_info (bool):
            If ``True``, also prints a message about ``doped``'s updated compatibility
            with the new Materials Project API (if a legacy API key is being used).
            Default is ``False``.

    Returns:
        api_key (str):
            Materials Project API key, as supplied in the input argument or
            read from the environment variable.
        legacy_MP (bool):
            ``True`` if the API key corresponds to the legacy Materials Project
            API, ``False`` if it corresponds to the new Materials Project API
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
    legacy_MP = True
    if len(api_key) == 32:  # new API
        legacy_MP = False
    elif len(api_key) < 15 or len(api_key) > 20:  # looks like an invalid API key; check:
        try:
            with MPRester(api_key) as mpr:
                mpr.get_entry_by_material_id("mp-1")  # check if API key is valid
        except Exception as mp_exc:
            if "MPRestError" in str(type(mp_exc)):  # can't control directly as may be legacy or new API
                raise ValueError(
                    f"The supplied API key (either ``api_key`` or 'PMG_MAPI_KEY' in ``~/.pmgrc.yaml`` or "
                    f"``~/.config/.pmgrc.yaml``; {api_key}) is not a valid Materials Project API "
                    f"key, which is required by doped for competing phase generation. See the doped "
                    f"installation instructions for details:\n"
                    "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
                    "-project-api"
                ) from mp_exc

            raise

    if legacy_MP and legacy_MP_info:
        print(
            "Note that doped now supports the new Materials Project API, which can be used by updating "
            "your API key in ~/.pmgrc.yaml or ~/.config/.pmgrc.yaml: "
            "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
            "-project-api"
        )

    if not legacy_MP and not (
        importlib.util.find_spec("emmet.core") and importlib.util.find_spec("mp_api")
    ):  # new MP, check that required dependencies are installed
        warnings.warn(
            "Your Materials Project (MP) API key (either ``api_key`` parameter or 'PMG_MAPI_KEY' in "
            "``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``) corresponds to the new MP API, "
            "which requires emmet-core and mp-api as dependencies, but these are not both installed. "
            "Please install with ``pip install -U mp-api emmet-core``!"
        )

    return api_key, legacy_MP


def get_MP_summary_docs(
    entries: Optional[list[ComputedEntry]] = None,
    chemsys: Optional[Union[str, list[str]]] = None,
    api_key: Optional[str] = None,
    data_fields: Optional[list[str]] = None,
    **kwargs,
):
    """
    Get the corresponding Materials Project (MP) ``SummaryDoc`` documents for
    computed entries in the input ``entries`` list or ``chemsys`` chemical
    system.

    If ``entries`` is provided (which should be a list of ``ComputedEntry``s
    from the Materials Project), then only ``SummaryDoc``s in this chemical
    system which match one of these entries (based on the MPIDs given in
    ``ComputedEntry.entry_id``/``ComputedEntry.data["material_id"]`` and
    ``SummaryDoc.material_id``) are returned.
    Moreover, all data fields listed in ``data_fields`` (set to ``"band_gap"``,
    ``"total_magnetization"`` and ``"database_IDs"`` by default) will be copied
    from the corresponding ``SummaryDoc`` attribute to ``ComputedEntry.data``
    for the matching ``ComputedEntry`` in ``entries``

    Note that this function can only be used with the new Materials Project API,
    as the legacy API does not have the ``SummaryDoc`` functionality (but most of
    the same data is available through the ``property_data`` arguments for the
    legacy-API-compatible functions).

    Args:
        entries (list[ComputedEntry]):
            Optional input; list of ``ComputedEntry`` objects for the input chemical
            system. If provided, only ``SummaryDoc``s which match one of these entries
            (based on the MPIDs given in ``ComputedEntry.entry_id``/
            ``ComputedEntry.data["material_id"]`` and ``SummaryDoc.material_id``) are
            returned. Moreover, all data fields listed in ``data_fields`` will be copied
            from the corresponding ``SummaryDoc`` attribute to ``ComputedEntry.data`` for
            the matching ``ComputedEntry`` in ``entries``.
        chemsys (str, list[str]):
            Optional input; chemical system to get entries for, in the format "A-B-C" or
            ["A", "B", "C"]. E.g. "Li-Fe-O" or ["Li", "Fe", "O"]. Either ``entries`` or
            ``chemsys`` must be provided!
        api_key (str):
            Materials Project (MP) API key, needed to access the MP database
            to obtain the corresponding ``SummaryDoc`` documents. Must be
            a new (not legacy) MP API key! If not supplied, will attempt to
            read from environment variable ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml``
            or ``~/.config/.pmgrc.yaml``) - see the ``doped`` Installation docs page:
            https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials
            -project-api
        data_fields (list[str]):
            List of data fields to copy from the corresponding ``SummaryDoc``
            attributes to the ``ComputedEntry.data`` objects, if ``entries`` is supplied.
            If not set, defaults to ``["band_gap", "total_magnetization", "database_IDs"]``.
        **kwargs:
            Additional keyword arguments to pass to the Materials Project API
            query, e.g. ``mpr.materials.summary.search()``.

    Returns:
        dict[str, SummaryDoc]:
            Dictionary of ``SummaryDoc`` documents with MPIDs as keys.
    """
    api_key, legacy_MP = _parse_MP_API_key(api_key)
    if legacy_MP:
        raise ValueError(
            "`get_MP_summary_docs` can only be used with the new Materials Project (MP) API (see "
            "https://next-gen.materialsproject.org/api), but a legacy MP API key was supplied!"
        )
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
        MP_docs = {doc.material_id: doc for doc in mpr.materials.summary.search(**summary_search_kwargs)}

    if not entries:
        return MP_docs

    if data_fields is None:
        data_fields = ["band_gap", "total_magnetization", "database_IDs"]  # ICSD IDs and possibly others

    for entry in entries:
        doc = MP_docs.get(entry.data["material_id"])
        if doc:
            entry.MP_doc = doc  # for user convenience, can query the MP doc later
            for data_field in data_fields:
                if (
                    data_field not in entry.data
                ):  # don't overwrite existing data (e.g. our molecular entries)
                    entry.data[data_field] = getattr(doc, data_field, "N/A")

        elif entry.data["material_id"] != "mp-0":  # these are skipped, band_gap and total_mag already set
            warnings.warn(
                f"No matching SummaryDoc found for entry {entry.name, entry.data['material_id']} in the "
                f"(new) Materials Project API database. Assuming that it is an insulating (non-metallic) "
                f"and non-magnetic compound."
            )
            entry.data["band_gap"] = None
            entry.data["total_magnetization"] = None
            entry.data["database_IDs"] = "N/A"

    return MP_docs


def _entries_sort_func(
    entry: ComputedEntry,
    use_e_per_atom: bool = False,
    bulk_composition: Optional[Union[str, Composition, dict, list]] = None,
):
    """
    Function to sort ``ComputedEntry``s by energy above hull, then by the
    number of elements in the formula, then by the position of elements in the
    periodic table (main group elements, then transition metals, sorted by
    row), then alphabetically.

    If ``bulk_composition`` is provided, then entries matching the bulk
    composition are sorted first, followed by all other entries.

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
            Tuple of ``True``/``False`` (if composition matches bulk composition),
            the energy above hull (or energy per atom), number of elements in the
            formula, and sorted (group, row) list of elements in the formula, and
            the formula name.
    """
    bulk_reduced_comp = Composition(bulk_composition).reduced_composition if bulk_composition else None
    return (
        entry.composition.reduced_composition == bulk_reduced_comp,
        entry.energy_per_atom if use_e_per_atom else _get_e_above_hull(entry.data),
        len(Composition(entry.name).as_dict()),
        sorted([_element_sort_func(i.symbol) for i in Composition(entry.name).elements]),
        entry.name,
    )


def prune_entries_to_border_candidates(
    entries: list[ComputedEntry],
    bulk_computed_entry: ComputedEntry,
    phase_diagram: Optional[PhaseDiagram] = None,
    e_above_hull: float = 0.05,
):
    """
    Given an input list of ``ComputedEntry``/``ComputedStructureEntry``s
    (``entries``) and a single entry for the host material
    (``bulk_computed_entry``), returns the subset of entries which `could`
    border the host on the phase diagram (and therefore be a competing phase
    which determines the host chemical potential limits), allowing for an error
    tolerance for the semi-local DFT database energies (``e_above_hull``, set
    to ``self.e_above_hull`` 0.05 eV/atom by default).

    If ``phase_diagram`` is provided then this is used as the reference
    phase diagram, otherwise it is generated from ``entries`` and
    ``bulk_computed_entry``.

    Args:
        entries (list[ComputedEntry]):
            List of ``ComputedEntry`` objects to prune down to potential host
            border candidates on the phase diagram.
        bulk_computed_entry (ComputedEntry):
            ``ComputedEntry`` object for the host material.
        phase_diagram (PhaseDiagram):
            Optional input; ``PhaseDiagram`` object for the system of
            interest. If provided, this is used as the reference phase
            diagram from which to determine the (potential) chemical
            potential limits, otherwise it is generated from ``entries``
            and ``bulk_computed_entry``.
        e_above_hull (float):
            Maximum energy above hull (in eV/atom) of Materials Project
            entries to be considered as competing phases. This is an
            uncertainty range for the MP-calculated formation energies,
            which may not be accurate due to functional choice (GGA vs
            hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
            All phases that would border the host material on the phase
            diagram, if their relative energy was downshifted by
            ``e_above_hull``, are included.
            (Default is 0.05 eV/atom).

    Returns:
        list[ComputedEntry]:
            List of all ``ComputedEntry`` objects in ``entries`` which could border
            the host material on the phase diagram (and thus set the chemical
            potential limits), if their relative energy was downshifted by
            ``e_above_hull`` eV/atom.
    """
    if not phase_diagram:
        phase_diagram = PhaseDiagram({bulk_computed_entry, *entries})

    # cull to only include any phases that would border the host material on the phase
    # diagram, if their relative energy was downshifted by ``e_above_hull``:
    MP_chempots = get_chempots_from_phase_diagram(bulk_computed_entry, phase_diagram)
    MP_bordering_phases = {phase for limit in MP_chempots for phase in limit.split("-")}
    bordering_entries = [
        entry for entry in entries if entry.name in MP_bordering_phases or entry.is_element
    ]
    bordering_entry_names = [
        bordering_entry.name for bordering_entry in bordering_entries
    ]  # compositions which border the host with EaH=0, according to MP, so we include all phases with
    # these compositions up to EaH=e_above_hull (which we've already pruned to)
    # for determining phases which alter the chemical potential limits when renormalised, only need to
    # retain the EaH=0 entries from above, so we use this reduced PD to save compute time when looping
    # below:
    reduced_pd_entries = {
        entry
        for entry in bordering_entries
        if entry.data.get("energy_above_hull", entry.data.get("e_above_hull", 0)) == 0
    }

    # then add any other phases that would border the host material on the phase diagram, if their
    # relative energy was downshifted by ``e_above_hull``:
    # only check if not already bordering; can just use names for this:
    entries_to_test = [entry for entry in entries if entry.name not in bordering_entry_names]
    entries_to_test.sort(key=_entries_sort_func)  # sort by energy above hull
    # to save unnecessary looping, whenever we encounter a phase that is not being added to the border
    # candidates list, skip all following phases with this composition (because they have higher
    # energies above hull (because we've sorted by this) and so will also not border the host):
    compositions_to_skip = []
    for entry in entries_to_test:
        if entry.name not in compositions_to_skip:
            # decrease entry energy per atom by ``e_above_hull`` eV/atom
            renormalised_entry = _renormalise_entry(entry, e_above_hull)
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
    entry: Union[ComputedStructureEntry, ComputedEntry], regenerate=False, ndigits=3
) -> str:
    """
    Get the ``doped`` name for a competing phase entry from the Materials
    Project (MP) database.

    The default naming convention in ``doped`` for competing phases is:
    ``"{Chemical Formula}_{Space Group}_EaH_{MP Energy above Hull}"``.
    This is stored in the ``entry.data["doped_name"]`` key-value pair.
    If this value is already set, then this function just returns the
    previously-generated ``doped`` name, unless ``regenerate=True``.

    Args:
        entry (ComputedStructureEntry, ComputedEntry):
            ``pymatgen`` ``ComputedStructureEntry`` object for the
            competing phase.
        regenerate (bool):
            Whether to regenerate the ``doped`` name for the competing
            phase, if ``entry.data["doped_name"]`` already set.
            Default is False.
        ndigits (int):
            Number of digits to round the energy above hull value (in
            eV/atom) to. Default is 3.

    Returns:
        doped_name (str):
            The ``doped`` name for the competing phase.
    """
    if not entry.data.get("doped_name") or regenerate:  # not set, so generate
        rounded_eah = round(_get_e_above_hull(entry.data), ndigits)

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
    entry: Union[ComputedStructureEntry, ComputedEntry], regenerate=False, ndigits=3
) -> str:
    """
    Get the ``doped`` `folder` name for a competing phase entry from the
    Materials Project (MP) database, handling the possibility of slashes ("/")
    in the formula name (due to space group symbols such as C2/m, P2_1/c etc.)
    by removing them (-> C2m, P2_1c etc.).

    The default naming convention in ``doped`` for competing phases is:
    ``"{Chemical Formula}_{Space Group}_EaH_{MP Energy above Hull}"``,
    which is stored in the ``entry.data["doped_name"]`` key-value pair.
    If this value is already set, then this function just returns the
    previously-generated ``doped`` name, unless ``regenerate=True``.

    Args:
        entry (ComputedStructureEntry, ComputedEntry):
            ``pymatgen`` ``ComputedStructureEntry`` object for the
            competing phase.
        regenerate (bool):
            Whether to regenerate the ``doped`` name for the competing
            phase, if ``entry.data["doped_name"]`` already set.
            Default is False.
        ndigits (int):
            Number of digits to round the energy above hull value (in
            eV/atom) to. Default is 3.

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
        composition: Union[str, Composition, Structure],
        e_above_hull: float = 0.05,
        api_key: Optional[str] = None,
        full_phase_diagram: bool = False,
    ):
        """
        Class to generate VASP input files for competing phases on the phase
        diagram for the host material, which determine the chemical potential
        limits for that compound.

        For this, the Materials Project (MP) database is queried using the
        ``MPRester`` API, and any calculated compounds which `could` border
        the host material within an error tolerance for the semi-local DFT
        database energies (``e_above_hull``, 0.05 eV/atom by default) are
        generated, along with the elemental reference phases.
        Diatomic gaseous molecules are generated as molecules-in-a-box as
        appropriate (e.g. for O2, F2, H2 etc).

        Often ``e_above_hull`` can be lowered (e.g. to ``0``) to reduce the
        number of calculations while retaining good accuracy relative to the
        typical error of defect calculations.

        The default ``e_above_hull`` of 50 meV/atom works well in accounting for
        MP formation energy inaccuracies in most known cases. However, some
        critical thinking is key (as always!) and so if there are any obvious
        missing phases or known failures of the Materials Project energetics in
        your chemical space of interest, you should adjust this parameter to
        account for this (or alternatively manually include these known missing
        phases in your competing phase calculations, to be included in parsing
        and chemical potential analysis later on).

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
            e_above_hull (float):
                Maximum energy above hull (in eV/atom) of Materials Project
                entries to be considered as competing phases. This is an
                uncertainty range for the MP-calculated formation energies,
                which may not be accurate due to functional choice (GGA vs
                hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
                All phases that would border the host material on the phase
                diagram, if their relative energy was downshifted by
                ``e_above_hull``, are included.
                Often ``e_above_hull`` can be lowered (e.g. to ``0``) to reduce
                the number of calculations while retaining good accuracy relative
                to the typical error of defect calculations.
                (Default is 0.05 eV/atom).
            api_key (str):
                Materials Project (MP) API key, needed to access the MP
                database for competing phase generation. If not supplied, will
                attempt to read from environment variable ``PMG_MAPI_KEY`` (in
                ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``) - see the ``doped``
                Installation docs page:
                https://doped.readthedocs.io/en/latest/Installation.html
            full_phase_diagram (bool):
                If ``True``, include all phases on the MP phase diagram (with energy
                above hull < ``e_above_hull`` eV/atom) for the chemical system of
                the input composition (not recommended). If ``False``, only includes
                phases that would border the host material on the phase diagram (and
                thus set the chemical potential limits), if their relative energy was
                downshifted by ``e_above_hull`` eV/atom.
                (Default is ``False``).
        """
        self.e_above_hull = e_above_hull  # store parameters for reference
        self.full_phase_diagram = full_phase_diagram
        # get API key, and print info message if it corresponds to legacy MP -- remove this and legacy
        # MP API warning filter in future versions (TODO)
        self.api_key, self.legacy_MP = _parse_MP_API_key(api_key, legacy_MP_info=True)
        warnings.filterwarnings(  # Remove in future when users have been given time to transition
            "ignore", message="You are using the legacy MPRester"
        )  # previously relied on this so shouldn't show warning, `message` only needs to match start

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
        # - Could have two optional EaH tolerances, a tight one (0.02 eV/atom?) that applies to all,
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

        self.chemsys = list(self.composition.as_dict().keys())

        # get all entries in the chemical system:
        self.MP_full_pd_entries, self.property_key_dict, self.property_data_fields = (
            get_entries_in_chemsys(  # get all entries in the chemical system, with EaH<``e_above_hull``
                self.chemsys,
                api_key=self.api_key,
                e_above_hull=self.e_above_hull,
                return_all_info=True,
                bulk_composition=self.composition.reduced_formula,  # for sorting
            )
        )
        self.MP_full_pd = PhaseDiagram(self.MP_full_pd_entries)

        # convert any gaseous elemental entries to molecules in a box
        formatted_entries = self._generate_elemental_diatomic_phases(self.MP_full_pd_entries)

        # get bulk entry, and warn if not stable or not present on MP database:
        bulk_entries = [
            entry
            for entry in formatted_entries  # sorted by e_above_hull above in get_entries_in_chemsys
            if entry.composition.reduced_composition == self.composition.reduced_composition
        ]
        if zero_eah_bulk_entries := [
            entry for entry in bulk_entries if _get_e_above_hull(entry.data) == 0.0
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
                # decrease bulk_computed_entry energy per atom by ``e_above_hull`` + 0.1 meV/atom
                name = description = (
                    "Manual energy adjustment to move the host composition to the MP convex hull"
                )
                bulk_computed_entry = _renormalise_entry(
                    bulk_computed_entry, eah + 1e-4, name=name, description=description
                )
                bulk_computed_entry.data[self.property_key_dict["energy_above_hull"]] = 0.0

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
                        self.property_key_dict["energy_above_hull"]: 0.0,
                        "band_gap": None,
                        "total_magnetization": None,
                        "database_IDs": "N/A",
                        "material_id": "mp-0",
                        "molecule": False,
                    },
                )  # TODO: Later need to add handling for file writing for this (POTCAR and INCAR assuming
                # non-metallic, non-magnetic, with warning and recommendations

            if self.MP_bulk_computed_entry not in formatted_entries:
                formatted_entries.append(self.MP_bulk_computed_entry)

        if self.bulk_structure:  # prune all bulk phases to this structure
            manual_bulk_entry = None

            if bulk_entries := [
                entry
                for entry in formatted_entries  # sorted by e_above_hull above in get_entries_in_chemsys
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
            # material on the phase diagram, if their relative energy was downshifted by ``e_above_hull``:
            self.entries: list[ComputedEntry] = prune_entries_to_border_candidates(
                entries=formatted_entries,
                bulk_computed_entry=bulk_computed_entry,
                e_above_hull=self.e_above_hull,
            )

        else:  # self.full_phase_diagram = True
            self.entries = formatted_entries

        # sort by host composition?, energy above hull, num_species, then by periodic table positioning:
        self.entries.sort(key=lambda x: _entries_sort_func(x, bulk_composition=self.composition))
        _name_entries_and_handle_duplicates(self.entries)  # set entry names

        if not self.legacy_MP:  # need to pull ``SummaryDoc``s to get band_gap and magnetization info
            self.MP_docs = get_MP_summary_docs(
                entries=self.entries,  # sets "band_gap", "total_magnetization" and "database_IDs" fields
                api_key=self.api_key,
            )

    @property
    def metallic_entries(self) -> list[ComputedEntry]:
        """
        Returns a list of entries in ``self.entries`` which are metallic (i.e.
        have a band gap = 0) according to the Materials Project database.
        """
        return [entry for entry in self.entries if entry.data.get("band_gap") == 0]

    @property
    def nonmetallic_entries(self) -> list[ComputedEntry]:
        """
        Returns a list of entries in ``self.entries`` which are non-metallic
        (i.e. have a band gap > 0, or no recorded band gap) according to the
        Materials Project database.

        Note that the ``doped``-generated diatomic molecule phases,
        which are insulators, are not included here.
        """
        return [
            entry
            for entry in self.entries
            if (entry.data.get("band_gap") is None or entry.data.get("band_gap") > 0)
            and not entry.data.get("molecule")
        ]

    @property
    def molecular_entries(self) -> list[ComputedEntry]:
        """
        Returns a list of entries in ``self.entries`` which are diatomic
        molecules generated by ``doped`` (i.e. O2, N2, H2, F2 or Cl2).
        """
        return [entry for entry in self.entries if entry.data.get("molecule")]

    # TODO: Return dict of DictSet objects for this and vasp_std_setup() functions, as well as
    #  write_files option, for ready integration with high-throughput workflows
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
        the ``ISMEAR`` ``INCAR`` tag to 2 (if metallic) or 0 if not. Recommended to
        use with https://github.com/kavanase/vaspup2.0.

        Args:
            kpoints_metals (tuple):
                Kpoint density per inverse volume (Å^-3) to be tested in
                (min, max, step) format for metals
            kpoints_nonmetals (tuple):
                Kpoint density per inverse volume (Å^-3) to be tested in
                (min, max, step) format for nonmetals
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR`` set.
            user_incar_settings (dict):
                Override the default INCAR settings e.g. {"EDIFF": 1e-5, "LDAU": False,
                "ALGO": "All"}. Note that any non-numerical or non-True/False flags
                need to be input as strings with quotation marks. See
                ``doped/VASP_sets/PBEsol_ConvergenceSet.yaml`` for the default settings.
            **kwargs: Additional kwargs to pass to ``DictSet.write_input()``
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
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR`` set.
            user_incar_settings (dict):
                Override the default INCAR settings e.g. {"EDIFF": 1e-5, "LDAU": False,
                "ALGO": "All"}. Note that any non-numerical or non-True/False flags
                need to be input as strings with quotation marks. See
                ``doped/VASP_sets/RelaxSet.yaml`` and ``HSESet.yaml`` for the default settings.
            **kwargs: Additional kwargs to pass to ``DictSet.write_input()``
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
        If the entry has a non-zero total magnetisation (greater than the
        default tolerance of 0.1), set ``ISPIN`` to 2 (allowing spin
        polarisation) and ``NUPDOWN`` equal to the integer-rounded total
        magnetisation.

        Otherwise ``ISPIN`` is not set, so spin polarisation is not allowed
        (as typically desired for non-magnetic phases, for efficiency).

        See
        https://doped.readthedocs.io/en/latest/Tips.html#spin-polarisation
        """
        magnetization = entry.data.get("total_magnetization")
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
        """
        Given an input list of ``ComputedEntry`` objects, adds a
        ``ComputedStructureEntry`` for each diatomic elemental phase (O2, N2,
        H2, F2, Cl2) to ``entries`` using ``make_molecular_entry``, and
        generates an output list of
        ``ComputedEntry``/``ComputedStructureEntry``s containing all entries in
        ``entries``, with all diatomic elemental phases replaced by the single
        molecule-in-a-box entry.

        Also sets the ``ComputedEntry.data["molecule"]`` flag for each entry
        in ``entries`` (``True`` for diatomic gases, ``False`` for all others).

        The output entries list is sorted by energy above hull, then by the number
        of elements in the formula, then by the position of elements in the
        periodic table (main group elements, then transition metals, sorted by row).

        Args:
            entries (list[ComputedEntry]):
                List of ``ComputedEntry``/``ComputedStructureEntry`` objects for
                the input chemical system.

        Returns:
            list[ComputedEntry]:
                List of ``ComputedEntry``/``ComputedStructureEntry`` objects for the
                input chemical system, with diatomic elemental phases replaced by
                the single molecule-in-a-box entry.
        """
        formatted_entries: list[ComputedEntry] = []

        for entry in entries.copy():
            if (
                _get_pretty_formula(entry.data) in elemental_diatomic_gases
                and _get_e_above_hull(entry.data) == 0.0
            ):  # only once for each matching gaseous elemental entry
                molecular_entry = make_molecular_entry(entry, legacy_MP=self.legacy_MP)
                if not any(
                    entry.data["molecule"]
                    and _get_pretty_formula(entry.data) == _get_pretty_formula(molecular_entry.data)
                    for entry in formatted_entries
                ):  # first entry only
                    entries.append(molecular_entry)
                    formatted_entries.append(molecular_entry)
            elif _get_pretty_formula(entry.data) not in elemental_diatomic_gases:
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


# TODO: Merge this to `CompetingPhases`, and just have options to only write extrinsic files to
#   output? And smart file/folder overwriting handling (like in SnB?)
class ExtrinsicCompetingPhases(CompetingPhases):
    def __init__(
        self,
        composition: Union[str, Composition, Structure],
        extrinsic_species: Union[str, Iterable],
        e_above_hull: float = 0.05,
        full_sub_approach: bool = False,
        codoping: bool = False,
        api_key: Optional[str] = None,
        full_phase_diagram: bool = False,
    ):
        """
        Class to generate VASP input files for competing phases involving
        extrinsic (dopant/impurity) elements, which determine the chemical
        potential limits for those elements in the host compound.

        Only extrinsic competing phases are contained in the
         ``ExtrinsicCompetingPhases.entries`` list (used for input file
         generation), while the `intrinsic` competing phases for the host
         compound are stored in ``ExtrinsicCompetingPhases.intrinsic_entries``.

        For this, the Materials Project (MP) database is queried using the
        ``MPRester`` API, and any calculated compounds which `could` border
        the host material within an error tolerance for the semi-local DFT
        database energies (``e_above_hull``, 0.05 eV/atom by default) are
        generated, along with the elemental reference phases.
        Diatomic gaseous molecules are generated as molecules-in-a-box as
        appropriate (e.g. for O2, F2, H2 etc).

        Often ``e_above_hull`` can be lowered (e.g. to ``0``) to reduce the
        number of calculations while retaining good accuracy relative to the
        typical error of defect calculations.

        The default ``e_above_hull`` of 50 meV/atom works well in accounting for
        MP formation energy inaccuracies in most known cases. However, some
        critical thinking is key (as always!) and so if there are any obvious
        missing phases or known failures of the Materials Project energetics in
        your chemical space of interest, you should adjust this parameter to
        account for this (or alternatively manually include these known missing
        phases in your competing phase calculations, to be included in parsing
        and chemical potential analysis later on).

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
            extrinsic_species (str, Iterable):
                Extrinsic dopant/impurity species to consider, to generate
                the relevant competing phases to additionally determine their
                chemical potential limits within the host. Can be a single
                element as a string (e.g. "Mg") or an iterable of element
                strings (list, set, tuple, dict) (e.g. ["Mg", "Na"]).
            e_above_hull (float):
                Maximum energy-above-hull of Materials Project entries to be
                considered as competing phases. This is an uncertainty range
                for the MP-calculated formation energies, which may not be
                accurate due to functional choice (GGA vs hybrid DFT / GGA+U
                / RPA etc.), lack of vdW corrections etc.
                Any phases that would border the host material on the phase
                diagram, if their relative energy was downshifted by
                ``e_above_hull``, are included.

                Often ``e_above_hull`` can be lowered (e.g. to ``0``) to reduce
                the number of calculations while retaining good accuracy relative
                to the typical error of defect calculations.

                Default is 0.05 eV/atom.
            full_sub_approach (bool):
                Generate competing phases by considering the full phase
                diagram, including chemical potential limits with multiple
                extrinsic phases. Only recommended when looking at high
                (non-dilute) doping concentrations. Default is ``False``.

                The default approach (``full_sub_approach = False``) for
                extrinsic elements is to only consider chemical potential
                limits where the host composition borders a maximum of 1
                extrinsic phase (composition with extrinsic element(s)).
                This is a valid approximation for the case of dilute
                dopant/impurity concentrations. For high (non-dilute)
                concentrations of extrinsic species, use ``full_sub_approach = True``.
            codoping (bool):
                Whether to consider extrinsic competing phases containing
                multiple extrinsic species. Only relevant to high (non-dilute)
                co-doping concentrations. If set to True, then
                ``full_sub_approach`` is also set to ``True``.
                Default is ``False``.
            api_key (str):
                Materials Project (MP) API key, needed to access the MP database
                for competing phase generation. If not supplied, will attempt
                to read from environment variable ``PMG_MAPI_KEY`` (in
                ``~/.pmgrc.yaml``) - see the ``doped`` Installation docs page:
                https://doped.readthedocs.io/en/latest/Installation.html.
                MP API key is available at https://next-gen.materialsproject.org/api#api-key
            full_phase_diagram (bool):
                If ``True``, include all phases on the MP phase diagram (with
                energy above hull < ``e_above_hull`` eV/atom) for the chemical
                system of the input composition and extrinsic species (not
                recommended). If ``False``, only includes phases that would border
                the host material on the phase diagram (and thus set the chemical
                potential limits), if their relative energy was downshifted by
                ``e_above_hull`` eV/atom.
                Default is ``False``.
        """
        super().__init__(  # competing phases & entries of the OG system:
            composition=composition,
            e_above_hull=e_above_hull,
            api_key=api_key,
            full_phase_diagram=full_phase_diagram,
        )
        self.intrinsic_entries = copy.deepcopy(self.entries)
        self.entries = []
        self.intrinsic_species = [s.symbol for s in self.composition.reduced_composition.elements]
        self.MP_intrinsic_full_pd_entries = self.MP_full_pd_entries  # includes molecules-in-boxes
        self.codoping = codoping
        self.full_sub_approach = full_sub_approach

        if isinstance(extrinsic_species, str):
            extrinsic_species = [
                extrinsic_species,
            ]
        elif not isinstance(extrinsic_species, Iterable):
            raise TypeError(
                f"`extrinsic_species` must be a string (i.e. the extrinsic species "
                f"symbol, e.g. 'Mg') or an iterable object (list, set, tuple or dict; e.g. ['Mg', 'Na']), "
                f"got type {type(extrinsic_species)} instead!"
            )
        self.extrinsic_species = list(extrinsic_species)
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
                e_above_hull=self.e_above_hull,
                bulk_composition=self.composition.reduced_formula,  # for sorting
            )
            self.entries = self._generate_elemental_diatomic_phases(self.MP_full_pd_entries)

            if not full_phase_diagram:
                self.entries = prune_entries_to_border_candidates(
                    entries=self.entries,
                    bulk_computed_entry=self.MP_bulk_computed_entry,
                    e_above_hull=self.e_above_hull,
                )  # prune using phase diagram with all extrinsic species

        else:  # full_sub_approach (for now, to get self.entries) but not co-doping
            candidate_extrinsic_entries = []
            for sub_el in self.extrinsic_species:
                sub_el_MP_full_pd_entries = get_entries_in_chemsys(
                    [*self.intrinsic_species, sub_el],
                    api_key=self.api_key,
                    e_above_hull=self.e_above_hull,
                    bulk_composition=self.composition.reduced_formula,  # for sorting
                )
                sub_el_pd_entries = self._generate_elemental_diatomic_phases(sub_el_MP_full_pd_entries)
                self.MP_full_pd_entries.extend(
                    [entry for entry in sub_el_MP_full_pd_entries if entry not in self.MP_full_pd_entries]
                )

                if not full_phase_diagram:  # default, prune to only phases that would border the host
                    # material on the phase diagram, if their relative energy was downshifted by
                    # `e_above_hull`; prune using phase diagrams for one extrinsic species at a time here
                    sub_el_pd_entries = prune_entries_to_border_candidates(
                        entries=sub_el_pd_entries,
                        bulk_computed_entry=self.MP_bulk_computed_entry,
                        e_above_hull=self.e_above_hull,
                    )

                candidate_extrinsic_entries += [
                    entry for entry in sub_el_pd_entries if sub_el in entry.composition
                ]

            if self.full_sub_approach:
                self.entries = candidate_extrinsic_entries

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
                            extrinsic_bordering_phases  # TODO: Explicitly test this for all cases in tests
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
                        # warnings.warn(
                        #     f"Determined chemical potentials to be over-dependent on the extrinsic "
                        #     f"species {sub_el}, meaning we need to revert to `full_sub_approach = True` "
                        #     f"for this species."
                        # )  # Revert to this handling if we ever find a case of this actually happening
                        # self.entries += sub_el_entries
                        raise RuntimeError(
                            f"Determined chemical potentials to be over-dependent on the extrinsic "
                            f"species {sub_el} despite `full_sub_approach=False`, which shouldn't happen. "
                            f"Please report this to the developers on the GitHub issues page: "
                            f"https://github.com/SMTG-Bham/doped/issues"
                        )

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

        if not self.legacy_MP:  # need to pull ``SummaryDoc``s to get band_gap and magnetization info
            self.intrinsic_MP_docs = deepcopy(self.MP_docs)
            self.MP_docs = get_MP_summary_docs(
                entries=self.entries,  # sets "band_gap", "total_magnetization" and "database_IDs" fields
                api_key=self.api_key,
            )


def get_doped_chempots_from_entries(
    entries: Sequence[Union[ComputedEntry, ComputedStructureEntry, PDEntry]],
    composition: Union[str, Composition, ComputedEntry],
    sort_by: Optional[str] = None,
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
            the chemical potential limits for the host material (``composition``).
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
            in the calculated chemical potentials dictionary. Mainly intended for
            internal ``doped`` usage when the host material is calculated to be
            unstable with respect to the competing phases.

    Returns:
        dict:
            Dictionary of chemical potential limits in the ``doped`` format.
    """
    if isinstance(composition, (str, Composition)):
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


class CompetingPhasesAnalyzer:
    # TODO: Allow parsing using pymatgen ComputedEntries as well, to aid interoperability with
    #  high-throughput architectures like AiiDA or atomate2. See:
    #  https://github.com/SMTG-Bham/doped/commit/b4eb9a5083a0a2c9596be5ccc57d060e1fcec530
    # TODO: Parse into ``ComputedStructureEntries`` (similar to ``CompetingPhases`` handling) for easier
    #  maintenance, manipulation and interoperability, and update ``__repr__()`` to print entries list
    #  with folder and ``doped`` names (like in ``CompetingPhases``)
    def __init__(self, composition: Union[str, Composition]):
        """
        Class for post-processing competing phases calculations, to determine
        the corresponding chemical potentials for the host ``composition``.

        Args:
            composition (str, ``Composition``):
                Composition of the host material (e.g. ``'LiFePO4'``, or
                ``Composition('LiFePO4')``, or
                ``Composition({"Li":1, "Fe":1, "P":1, "O":4})``).

        Attributes:
            composition (str): The bulk (host) composition.
            elements (list):
                List of all elements in the chemical system (host + extrinsic),
                from all parsed calculations.
            extrinsic_elements (str):
                List of extrinsic elements in the chemical system (not present
                in ``composition``).
            data (list):
                List of dictionaries containing the parsed competing phases data.
            formation_energy_df (pandas.DataFrame):
                DataFrame containing the parsed competing phases data.
        """
        self.composition = Composition(composition)
        self.elements = [str(c) for c in self.composition.elements]
        self.extrinsic_elements: list[str] = []

    # TODO: `from_vaspruns` should just be the default initialisation of CompetingPhasesAnalyzer,
    #  which auto-parses vaspruns from the subdirectories (or optionally a list of vaspruns,
    #  or a csv path); see shelved changes for this
    def from_vaspruns(
        self,
        path: Union[PathLike, list[PathLike]] = "CompetingPhases",
        subfolder: Optional[PathLike] = "vasp_std",
        csv_path: Optional[PathLike] = None,
        verbose: bool = True,
        processes: Optional[int] = None,
    ):
        """
        Parses competing phase energies from ``vasprun.xml(.gz)`` outputs,
        computes the formation energies and generates the
        ``CompetingPhasesAnalyzer`` object.

        By default, tries multiprocessing to speed up parsing, which can be
        controlled with ``processes``. If parsing hangs, this may be due to
        memory issues, in which case you should reduce ``processes`` (e.g.
        4 or less).

        Args:
            path (PathLike or list):
                Either a path to the base folder in which you have your
                competing phase calculation outputs (e.g.
                ``formula_EaH_X/vasp_std/vasprun.xml(.gz)``, or
                ``formula_EaH_X/vasprun.xml(.gz)``), or a list of strings/Paths
                to ``vasprun.xml(.gz)`` files.
            subfolder (PathLike):
                The subfolder in which your vasprun.xml(.gz) output files
                are located (e.g. a file-structure like:
                ``formula_EaH_X/{subfolder}/vasprun.xml(.gz)``). Default
                is to search for ``vasp_std`` subfolders, or directly in the
                ``formula_EaH_X`` folder.
            csv_path (PathLike):
                If set will save the parsed data to a csv at this filepath.
                Further customisation of the output csv can be achieved with
                the CompetingPhasesAnalyzer.to_csv() method.
            verbose (bool):
                Whether to print out information about directories that were
                skipped (due to no ``vasprun.xml`` files being found).
                Default is ``True``.
            processes (int):
                Number of processes to use for multiprocessing for expedited
                parsing. If not set, defaults to one less than the number of
                CPUs available, or 1 if less than 10 phases to parse. Set to
                1 for no multiprocessing.

        Returns:
            None, sets ``self.data``, ``self.formation_energy_df`` and ``self.elemental_energies``
        """
        # TODO: Change this to just recursively search for vaspruns within the specified path (also
        #  currently doesn't seem to revert to searching for vaspruns in the base folder if no vasp_std
        #  subfolders are found) - see how this is done in DefectsParser in analysis.py
        # TODO: Add check for matching INCAR and POTCARs from these calcs - can use code/functions from
        #  analysis.py for this
        self.vasprun_paths = []
        skipped_folders = []
        self.parsed_folders = []

        if isinstance(path, list):  # if path is just a list of all competing phases
            for p in path:
                if "vasprun.xml" in str(p) and not str(p).startswith("."):
                    self.vasprun_paths.append(p)

                # try to find the file - will always pick the first match for vasprun.xml*
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

        self.vaspruns = []
        failed_parsing_dict: dict[str, list] = {}
        if processes is None:  # multiprocessing?
            if len(self.vasprun_paths) < 10:
                processes = 1
            else:  # only multiprocess as much as makes sense:
                processes = min(max(1, cpu_count() - 1), len(self.vasprun_paths) - 1)

        parsing_results = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnconvergedVASPWarning)  # checked and warned later
            if processes > 1:  # multiprocessing
                with Pool(processes=processes) as pool:  # result is parsed vasprun
                    for result in tqdm(
                        pool.imap_unordered(_parse_vasprun_and_catch_exception, self.vasprun_paths),
                        total=len(self.vasprun_paths),
                        desc="Parsing vaspruns...",
                    ):
                        parsing_results.append(result)
            else:
                for vasprun_path in tqdm(self.vasprun_paths, desc="Parsing vaspruns..."):
                    parsing_results.append(_parse_vasprun_and_catch_exception(vasprun_path))

        for result in parsing_results:
            if isinstance(result[0], Vasprun):  # successful parse; result is vr and parsed folder
                self.vaspruns.append(result[0])
                self.parsed_folders.append(result[1])
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
        electronic_unconverged_vaspruns = [
            vr.filename for vr in self.vaspruns if not vr.converged_electronic
        ]
        ionic_unconverged_vaspruns = [vr.filename for vr in self.vaspruns if not vr.converged_ionic]
        for unconverged_vaspruns, unconverged_type in zip(
            [electronic_unconverged_vaspruns, ionic_unconverged_vaspruns],
            ["Electronic", "Ionic"],
        ):
            if unconverged_vaspruns:
                warnings.warn(
                    f"{unconverged_type} convergence was not reached for:\n"
                    + "\n".join(unconverged_vaspruns)
                )

        if not self.vaspruns:
            raise FileNotFoundError(
                "No vasprun files have been parsed, suggesting issues with parsing! Please check that "
                "folders and input parameters are in the correct format (see docstrings/tutorials)."
            )

        data = []
        self.elemental_energies: dict[str, float] = {}

        for vr in self.vaspruns:
            comp = vr.final_structure.composition
            formulas_per_unit = comp.get_reduced_composition_and_factor()[1]
            energy_per_atom = vr.final_energy / len(vr.final_structure)

            kpoints = (
                "x".join(str(x) for x in vr.kpoints.kpts[0])
                if (vr.kpoints.kpts and len(vr.kpoints.kpts) == 1)
                else "N/A"
            )

            # check if elemental:
            if len(Composition(comp.reduced_formula).as_dict()) == 1:
                el = next(iter(vr.atomic_symbols))  # elemental, so first symbol is only (unique) element
                if el not in self.elemental_energies:
                    self.elemental_energies[el] = energy_per_atom
                    if el not in self.elements + self.extrinsic_elements:  # new (extrinsic) element
                        self.extrinsic_elements.append(el)

                elif energy_per_atom < self.elemental_energies[el]:
                    # only include lowest energy elemental polymorph
                    self.elemental_energies[el] = energy_per_atom

            d = {
                "Formula": comp.reduced_formula,
                "k-points": kpoints,
                "DFT Energy (eV/fu)": vr.final_energy / formulas_per_unit,
                "DFT Energy (eV/atom)": energy_per_atom,
                "DFT Energy (eV)": vr.final_energy,
            }
            data.append(d)

        # sort extrinsic elements and energies dict by periodic table positioning (deterministically),
        # and add to self.elements:
        self.extrinsic_elements = sorted(self.extrinsic_elements, key=_element_sort_func)
        self.elemental_energies = dict(
            sorted(self.elemental_energies.items(), key=lambda x: _element_sort_func(x[0]))
        )
        self.elements += self.extrinsic_elements

        formation_energy_df = _calculate_formation_energies(data, self.elemental_energies)
        self.data = formation_energy_df.to_dict(orient="records")
        self.formation_energy_df = pd.DataFrame(self._get_and_sort_formation_energy_data())  # sort data
        self.formation_energy_df.set_index("Formula")

        if csv_path is not None:
            self.to_csv(csv_path)
            print(f"Competing phase formation energies have been saved to {csv_path}")

    def _get_and_sort_formation_energy_data(self, sort_by_energy=False, prune_polymorphs=False):
        data = copy.deepcopy(self.data)

        if prune_polymorphs:  # only keep the lowest energy polymorphs
            formation_energy_df = _calculate_formation_energies(data, self.elemental_energies)
            indices = formation_energy_df.groupby("Formula")["DFT Energy (eV/atom)"].idxmin()
            pruned_df = formation_energy_df.loc[indices]
            data = pruned_df.to_dict(orient="records")

        if sort_by_energy:
            data = sorted(data, key=lambda x: x["Formation Energy (eV/fu)"], reverse=True)

        # moves the bulk composition to the top of the list
        _move_dict_to_start(data, "Formula", self.composition.reduced_formula)

        # for each dict in data list, sort the _keys_ as formula, formation_energy, energy_per_atom,
        # energy_per_fu, energy, kpoints, then element stoichiometries in order of appearance in
        # composition dict + extrinsic elements:
        copied_data = copy.deepcopy(data)
        formation_energy_data = [
            {
                **{
                    k: d.pop(k, None)
                    for k in [
                        "Formula",
                        "Formation Energy (eV/fu)",
                        "Formation Energy (eV/atom)",
                        "DFT Energy (eV/atom)",
                        "DFT Energy (eV/fu)",
                        "DFT Energy (eV)",
                        "k-points",
                    ]
                },
                **{  # num elts columns, sorted by order of occurrence in bulk composition:
                    str(elt): d.pop(str(elt), None)
                    for elt in sorted(
                        self.composition.elements,
                        key=lambda x: self.composition.reduced_formula.index(str(x)),
                    )
                    + self.extrinsic_elements
                },
            }
            for d in copied_data
        ]
        # if all values are None for a certain key, remove that key from all dicts in list:
        keys_to_remove = [
            k for k in formation_energy_data[0] if all(d[k] is None for d in formation_energy_data)
        ]
        return [{k: v for k, v in d.items() if k not in keys_to_remove} for d in formation_energy_data]

    def to_csv(self, csv_path: PathLike, sort_by_energy: bool = False, prune_polymorphs: bool = False):
        """
        Write parsed competing phases data to ``csv``.

        Can be re-loaded with ``CompetingPhasesAnalyzer.from_csv()``.

        Args:
            csv_path (Pathlike): Path to csv file to write to.
            sort_by_energy (bool):
                If True, sorts the csv by formation energy (highest to lowest).
                Default is False (sorting by formula).
            prune_polymorphs (bool):
                Whether to only write the lowest energy polymorphs for each composition.
                Doesn't affect chemical potential limits (only the ground-state
                polymorphs matter for this).
                Default is False.
        """
        formation_energy_data = self._get_and_sort_formation_energy_data(sort_by_energy, prune_polymorphs)
        pd.DataFrame(formation_energy_data).set_index("Formula").to_csv(csv_path)

    def from_csv(self, csv_path: PathLike):
        """
        Read in data from a previously parsed formation energies csv file.

        Args:
            csv_path (PathLike):
                Path to csv file. Must have columns 'Formula',
                and 'DFT Energy per Formula Unit (ev/fu)' or
                'DFT Energy per Atom (ev/atom)'.

        Returns:
            None, sets ``self.data`` and ``self.elemental_energies``.
        """
        formation_energy_df = pd.read_csv(csv_path)
        if "Formula" not in list(formation_energy_df.columns) or all(
            x not in list(formation_energy_df.columns)
            for x in [
                "DFT Energy (eV/fu)",
                "DFT Energy (eV/atom)",
            ]
        ):
            raise ValueError(
                "Supplied csv does not contain the minimal columns required ('Formula', and "
                "'DFT Energy (eV/fu)' or 'DFT Energy (eV/atom)')"
            )

        self.data = formation_energy_df.to_dict(orient="records")
        self.elemental_energies = {}
        for i in self.data:
            c = Composition(i["Formula"])
            if len(c.elements) == 1:
                el = str(next(iter(c.elements)))  # elemental, so first symbol is only (unique) element
                if "DFT Energy (eV/atom)" in list(formation_energy_df.columns):
                    el_energy_per_atom = i["DFT Energy (eV/atom)"]
                else:
                    el_energy_per_atom = i["DFT Energy (eV/fu)"] / c.num_atoms

                if el not in self.elemental_energies or el_energy_per_atom < self.elemental_energies[el]:
                    self.elemental_energies[el] = el_energy_per_atom
                    if el not in self.elements + self.extrinsic_elements:  # new (extrinsic) element
                        self.extrinsic_elements.append(el)

        if "Formation Energy (eV/fu)" not in list(formation_energy_df.columns):
            formation_energy_df = _calculate_formation_energies(self.data, self.elemental_energies)
            self.data = formation_energy_df.to_dict(orient="records")

        self.formation_energy_df = pd.DataFrame(self._get_and_sort_formation_energy_data())  # sort data
        self.formation_energy_df.set_index("Formula")

        # sort extrinsic elements and energies dict by periodic table positioning (deterministically),
        # and add to self.elements:
        self.extrinsic_elements = sorted(self.extrinsic_elements, key=_element_sort_func)
        self.elemental_energies = dict(
            sorted(self.elemental_energies.items(), key=lambda x: _element_sort_func(x[0]))
        )
        self.elements += self.extrinsic_elements

    def calculate_chempots(
        self,
        extrinsic_species: Optional[Union[str, Element, list[str], list[Element]]] = None,
        csv_path: Optional[PathLike] = None,
        sort_by: Optional[str] = None,
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
                of elements. If ``None`` (default), uses ``self.extrinsic_elements``.
            csv_path (PathLike):
                If set, will save the calculated chemical potential limits to ``csv_path``.
            sort_by (str):
                If set, will sort the chemical potential limits in the output
                ``DataFrame`` according to the chemical potential of the specified
                element (from element-rich to element-poor conditions).
            verbose (bool):
                If ``True`` (default), will print the parsed chemical potential limits.

        Returns:
            ``pandas`` ``DataFrame``, optionally saved to csv.
        """
        # TODO: Is outputting the chempot limits to `csv` useful? If so, should also be able to load from
        #  csv? Show this in tutorials, or at least add easy function
        if extrinsic_species is None:
            extrinsic_species = self.extrinsic_elements
        if not isinstance(extrinsic_species, list):
            extrinsic_species = [extrinsic_species]
        extrinsic_elements: list[Element] = [Element(e) for e in extrinsic_species]

        intrinsic_phase_diagram_entries = []
        extrinsic_formation_energies = []
        bulk_pde_list = []

        for d in self.data:
            pd_entry = PDEntry(d["Formula"], d["DFT Energy (eV/fu)"])
            if (np.isinf(d["Formation Energy (eV/fu)"]) or np.isnan(d["Formation Energy (eV/fu)"])) and (
                set(Composition(d["Formula"]).elements).issubset(self.composition.elements)
                or (
                    extrinsic_elements
                    and any(elt in Composition(d["Formula"]).elements for elt in extrinsic_elements)
                )
            ):
                warnings.warn(
                    f"Entry for {d['Formula']} has an infinite/NaN calculated formation energy, "
                    f"indicating an issue with parsing, and so will be skipped for calculating "
                    f"the chemical potential limits."
                )
                continue

            if set(Composition(d["Formula"]).elements).issubset(self.composition.elements):
                intrinsic_phase_diagram_entries.append(pd_entry)  # intrinsic phase
                if pd_entry.composition == self.composition:  # bulk phase
                    bulk_pde_list.append(pd_entry)

            elif extrinsic_elements and any(
                elt in Composition(d["Formula"]).elements for elt in extrinsic_elements
            ):
                # only take entries with the extrinsic species present, otherwise is additionally parsed
                # (but irrelevant) phases --- would need to be updated if adding codoping chemical
                # potentials _parsing_ functionality # TODO: ?
                if np.isinf(d["Formation Energy (eV/fu)"]) or np.isnan(d["Formation Energy (eV/fu)"]):
                    warnings.warn(
                        f"Entry for {d['Formula']} has an infinite/NaN calculated formation energy, "
                        f"indicating an issue with parsing, and so will be skipped for calculating "
                        f"the chemical potential limits"
                    )
                extrinsic_formation_energies.append(
                    {k: v for k, v in d.items() if k in ["Formula", "Formation Energy (eV/fu)"]}
                )
        for subdict in extrinsic_formation_energies:
            for el in self.elements:  # add element ratio stoichiometry columns for dataframe
                subdict[el] = Composition(subdict["Formula"]).as_dict().get(el, 0)
        extrinsic_formation_energy_df = pd.DataFrame(extrinsic_formation_energies)

        if not bulk_pde_list:
            intrinsic_phase_diagram_compositions = (
                {e.composition.reduced_formula for e in intrinsic_phase_diagram_entries}
                if intrinsic_phase_diagram_entries
                else None
            )
            raise ValueError(
                f"Could not find bulk phase for {self.composition.reduced_formula} in the supplied "
                f"data. Found intrinsic phase diagram entries for: {intrinsic_phase_diagram_compositions}"
            )
        # lowest energy bulk phase
        self.bulk_pde = sorted(bulk_pde_list, key=lambda x: x.energy_per_atom)[0]
        unstable_host = False

        self._intrinsic_phase_diagram = PhaseDiagram(
            intrinsic_phase_diagram_entries,
            list(map(Element, self.composition.elements)),  # preserve bulk comp element ordering
        )

        # check if it's stable and if not, warn user and downshift to get _least_ unstable point on convex
        # hull for the host material
        if self.bulk_pde not in self._intrinsic_phase_diagram.stable_entries:
            unstable_host = True
            eah = self._intrinsic_phase_diagram.get_e_above_hull(self.bulk_pde)
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
            # decrease bulk_pde energy per atom by ``e_above_hull`` + 0.1 meV/atom
            name = description = (
                "Manual energy adjustment to move the host composition to the calculated convex hull"
            )
            renormalised_bulk_pde = _renormalise_entry(
                self.bulk_pde, eah + 1e-4, name=name, description=description
            )
            self._intrinsic_phase_diagram = PhaseDiagram(
                [*intrinsic_phase_diagram_entries, renormalised_bulk_pde],
                list(map(Element, self.composition.elements)),  # preserve bulk comp element ordering
            )

        self._intrinsic_chempots = get_doped_chempots_from_entries(
            self._intrinsic_phase_diagram.entries,
            self.composition,
            sort_by=sort_by,
            single_chempot_limit=unstable_host,
        )

        # get chemical potentials as pandas dataframe
        chempots_df = pd.DataFrame.from_dict(
            {k: list(v.values()) for k, v in self._intrinsic_chempots["limits_wrt_el_refs"].items()},
            orient="index",
        )
        chempots_df.columns = [
            str(k) for k in next(iter(self._intrinsic_chempots["limits_wrt_el_refs"].values()))
        ]
        chempots_df.index.name = "Limit"

        missing_extrinsic = [
            elt for elt in extrinsic_elements if elt.symbol not in self.elemental_energies
        ]
        if not extrinsic_elements:  # intrinsic only
            self._chempots = self._intrinsic_chempots
        elif missing_extrinsic:
            raise ValueError(  # TODO: Test this
                f"Elemental reference phase for the specified extrinsic species "
                f"{[elt.symbol for elt in missing_extrinsic]} was not parsed, but is necessary for "
                f"chemical potential calculations. Please ensure that this phase is present in the "
                f"calculation directory and is being correctly parsed."
            )
        else:
            self._calculate_extrinsic_chempot_lims(  # updates self._chempots
                extrinsic_elements=extrinsic_elements,
                extrinsic_formation_energy_df=extrinsic_formation_energy_df,
                chempots_df=chempots_df,
            )
        # save and print
        if csv_path is not None:
            chempots_df.to_csv(csv_path)
            if verbose:
                print("Saved chemical potential limits to csv file: ", csv_path)

        if verbose:
            print("Calculated chemical potential limits (in eV wrt elemental reference phases): \n")
            print(chempots_df)

        return chempots_df

    # TODO: Add chempots df as a property? Then don't need to print here

    # TODO: This code (in all this module) should be rewritten to be more readable (re-used and
    #  uninformative variable names, missing informative comments, typing...)
    def _calculate_extrinsic_chempot_lims(
        self, extrinsic_elements, extrinsic_formation_energy_df, chempots_df
    ):
        # gets the df into a slightly more convenient dict
        cpd = chempots_df.to_dict(orient="records")
        extrinsic_chempots_dict: dict[str, list[float]] = {elt: [] for elt in extrinsic_elements}
        extrinsic_limiting_phases_dict: dict[str, list[str]] = {elt: [] for elt in extrinsic_elements}
        for extrinsic_elt in extrinsic_elements:
            for i, c in enumerate(cpd):
                name = f"mu_{extrinsic_elt.symbol}_{i}"
                extrinsic_formation_energy_df[name] = extrinsic_formation_energy_df[
                    "Formation Energy (eV/fu)"
                ]
                for k, v in c.items():
                    extrinsic_formation_energy_df[name] -= extrinsic_formation_energy_df[k] * v
                extrinsic_formation_energy_df[name] /= extrinsic_formation_energy_df[extrinsic_elt.symbol]
                # omit infinity values:
                non_infinite_formation_energy_df = extrinsic_formation_energy_df[
                    ~extrinsic_formation_energy_df[name].isin([-np.inf, np.inf, np.nan])
                ]
                extrinsic_chempots_dict[extrinsic_elt].append(non_infinite_formation_energy_df[name].min())
                extrinsic_limiting_phases_dict[extrinsic_elt].append(
                    non_infinite_formation_energy_df.loc[non_infinite_formation_energy_df[name].idxmin()][
                        "Formula"
                    ]
                )
            chempots_df[f"{extrinsic_elt.symbol}"] = extrinsic_chempots_dict[extrinsic_elt]

        for extrinsic_elt in extrinsic_elements:  # add limiting phases after, so at end of df
            chempots_df[f"{extrinsic_elt.symbol}-Limiting Phase"] = extrinsic_limiting_phases_dict[
                extrinsic_elt
            ]

        # 1. work out the formation energies of all dopant competing
        #    phases using the elemental energies
        # 2. for each of the chempots already calculated work out what
        #    the chemical potential of the dopant would be from
        #       mu_dopant = Hf(dopant competing phase) - sum(mu_elements)
        # 3. find the most negative mu_dopant which then becomes the new
        #    canonical chemical potential for that dopant species and the
        #    competing phase is the 'limiting phase' right
        # 4. update the chemical potential limits table to reflect this

        # reverse engineer chem lims for extrinsic
        chempot_lim_dict_list = chempots_df.copy().to_dict(orient="records")
        chempot_lims_w_extrinsic = {
            "elemental_refs": self.elemental_energies,
            "limits_wrt_el_refs": {},
            "limits": {},
        }

        for i, d in enumerate(chempot_lim_dict_list):
            key = (
                list(self._intrinsic_chempots["limits_wrt_el_refs"].keys())[i]
                + "-"
                + "-".join(d[col_name] for col_name in d if "Limiting Phase" in col_name)
            )
            new_vals = list(self._intrinsic_chempots["limits_wrt_el_refs"].values())[i]
            for extrinsic_elt in extrinsic_elements:
                new_vals[f"{extrinsic_elt.symbol}"] = d[f"{extrinsic_elt.symbol}"]
            chempot_lims_w_extrinsic["limits_wrt_el_refs"][key] = new_vals

        # relate the limits to the elemental
        # energies but in reverse this time
        for limit, chempot_dict in chempot_lims_w_extrinsic["limits_wrt_el_refs"].items():
            relative_chempot_dict = copy.deepcopy(chempot_dict)
            for e in relative_chempot_dict:
                relative_chempot_dict[e] += chempot_lims_w_extrinsic["elemental_refs"][e]
            chempot_lims_w_extrinsic["limits"].update({limit: relative_chempot_dict})

        self._chempots = chempot_lims_w_extrinsic

    @property
    def chempots(self) -> dict:
        """
        Returns the calculated chemical potential limits.

        If this is used with ``ExtrinsicCompetingPhases``
        before calling ``calculate_chempots`` with a specified
        ``extrinsic_species``, then the intrinsic chemical
        potential limits will be returned.
        """
        if not hasattr(self, "_chempots"):
            self.calculate_chempots(verbose=False)
        return self._chempots

    @property
    def intrinsic_chempots(self) -> dict:
        """
        Returns the calculated intrinsic chemical potential limits.
        """
        if not hasattr(self, "_intrinsic_chempots"):
            self.calculate_chempots(verbose=False)
        return self._intrinsic_chempots

    @property
    def intrinsic_phase_diagram(self) -> PhaseDiagram:
        """
        Returns the calculated intrinsic phase diagram.
        """
        if not hasattr(self, "_intrinsic_phase_diagram"):
            self.calculate_chempots(verbose=False)
        return self._intrinsic_phase_diagram

    def _cplap_input(self, dependent_variable: Optional[str] = None, filename: PathLike = "input.dat"):
        """
        Generates an ``input.dat`` file for the ``CPLAP`` ``FORTRAN`` code
        (legacy code for computing and analysing chemical potential limits, no
        longer recommended).

        Args:
            dependent_variable (str):
                Pick one of the variables as dependent, the first element in
                the composition is chosen if this isn't set.
            filename (PathLike):
                Filename, should end in ``.dat``.

        Returns:
            None, writes input.dat file.
        """
        if not hasattr(self, "chempots"):
            self.calculate_chempots(verbose=False)

        with open(filename, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            # get lowest energy bulk phase
            bulk_entries = [
                sub_dict
                for sub_dict in self.data
                if self.composition.reduced_composition
                == Composition(sub_dict["Formula"]).reduced_composition
            ]
            bulk_entry = min(bulk_entries, key=lambda x: x["Formation Energy (eV/fu)"])
            print(f"{len(self.composition.as_dict())}  # number of elements in bulk")
            for k, v in self.composition.as_dict().items():
                print(int(v), k, end=" ")
            print(
                f"{bulk_entry['Formation Energy (eV/fu)']}  # number of atoms, element, formation "
                f"energy (bulk)"
            )

            if dependent_variable is not None:
                print(f"{dependent_variable}  # dependent variable (element)")
            else:
                print(f"{self.elements[0]}  # dependent variable (element)")

            # get only the lowest energy entries of compositions in self.data which are on a
            # limit in self._intrinsic_chempots
            bordering_phases = {phase for limit in self._chempots["limits"] for phase in limit.split("-")}
            entries_for_cplap = [
                entry_dict
                for entry_dict in self.data
                if entry_dict["Formula"] in bordering_phases
                and Composition(entry_dict["Formula"]).reduced_composition
                != self.composition.reduced_composition
            ]
            # cull to only the lowest energy entries of each composition
            culled_cplap_entries: dict[str, dict] = {}
            for entry in entries_for_cplap:
                reduced_comp = Composition(entry["Formula"]).reduced_composition
                if (
                    reduced_comp not in culled_cplap_entries
                    or entry["Formation Energy (eV/fu)"]
                    < culled_cplap_entries[reduced_comp]["Formation Energy (eV/fu)"]
                ):
                    culled_cplap_entries[reduced_comp] = entry

            print(f"{len(culled_cplap_entries)}  # number of bordering phases")
            for i in culled_cplap_entries.values():
                print(f"{len(Composition(i['Formula']).as_dict())}  # number of elements in phase:")
                for k, v in Composition(i["Formula"]).as_dict().items():
                    print(int(v), k, end=" ")
                print(f"{i['Formation Energy (eV/fu)']}  # number of atoms, element, formation energy")

    def to_LaTeX_table(self, splits=1, sort_by_energy=False, prune_polymorphs=True):
        """
        A very simple function to print out the competing phase formation
        energies in a LaTeX table format, showing the formula, kpoints (if
        present in the parsed data) and formation energy.

        Needs the mhchem package to work and does `not` use the booktabs package
        - change hline to toprule, midrule and bottomrule if you want to use
        booktabs style.

        Args:
            splits (int):
                Number of splits for the table; either 1 (default) or 2 (with
                two large columns, each with the formula, kpoints (if present)
                and formation energy (sub-)columns).
            sort_by_energy (bool):
                If True, sorts the table by formation energy (highest to lowest).
                Default is False (sorting by formula).
            prune_polymorphs (bool):
                Whether to only print out the lowest energy polymorphs for each composition.
                Default is True.

        Returns:
            str: LaTeX table string
        """
        if splits not in [1, 2]:
            raise ValueError("`splits` must be either 1 or 2")
        # done in the pyscfermi report style
        formation_energy_data = self._get_and_sort_formation_energy_data(sort_by_energy, prune_polymorphs)

        kpoints_col = any("k-points" in item for item in formation_energy_data)

        string = "\\begin{table}[h]\n\\centering\n"
        string += (
            "\\caption{Formation energies per formula unit ($\\Delta E_f$) of \\ce{"
            + self.composition.reduced_formula
            + "} and all competing phases"
            + (", with k-meshes used in calculations." if kpoints_col else ".")
            + (" Only the lowest energy polymorphs are included}\n" if prune_polymorphs else "}\n")
        )
        string += "\\label{tab:competing_phase_formation_energies}\n"
        column_names_string = "Formula" + (" & k-mesh" if kpoints_col else "") + " & $\\Delta E_f$ (eV/fu)"

        if splits == 1:
            string += "\\begin{tabular}" + ("{ccc}" if kpoints_col else "{cc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " \\\\ \\hline \n"
            for i in formation_energy_data:
                kpoints = i.get("k-points", "0x0x0").split("x")
                fe = i["Formation Energy (eV/fu)"]
                string += (
                    "\\ce{"
                    + i["Formula"]
                    + "}"
                    + (f" & {kpoints[0]}$\\times${kpoints[1]}$\\times${kpoints[2]}" if kpoints_col else "")
                    + " & "
                    + f"{fe:.3f} \\\\ \n"
                )

        elif splits == 2:
            string += "\\begin{tabular}" + ("{ccc|ccc}" if kpoints_col else "{cc|cc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " & " + column_names_string + " \\\\ \\hline \n"

            mid = len(formation_energy_data) // 2
            first_half = formation_energy_data[:mid]
            last_half = formation_energy_data[mid:]

            for i, j in zip(first_half, last_half):
                kpoints1 = i.get("k-points", "0x0x0").split("x")
                fe1 = i["Formation Energy (eV/fu)"]
                kpoints2 = j.get("k-points", "0x0x0").split("x")
                fe2 = j["Formation Energy (eV/fu)"]
                string += (
                    "\\ce{"
                    + i["Formula"]
                    + "}"
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

        return string

    def plot_chempot_heatmap(
        self,
        dependent_element: Optional[Union[str, Element]] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        cbar_range: Optional[tuple[float, float]] = None,
        colormap: Optional[Union[str, colors.Colormap]] = None,
        padding: Optional[float] = None,
        title: Union[str, bool] = False,
        label_positions: Union[list[float], dict[str, float], bool] = True,
        filename: Optional[PathLike] = None,
        style_file: Optional[PathLike] = None,
    ) -> plt.Figure:
        """
        Plot a heatmap of the chemical potentials for a ternary system.

        In this plot, the ``dependent_element`` chemical potential is plotted
        as a heatmap over the stability region of the host composition, as a
        function of the other two elemental chemical potentials on the x and
        y axes.

        Note that due to an issue with ``matplotlib`` ``Stroke`` path effects,
        sometimes there can be odd holes in the whitespace around the chemical
        formula labels (see: https://github.com/matplotlib/matplotlib/issues/25669).
        This is only the case for ``png`` output, so saving to e.g. ``svg`` or
        ``pdf`` instead will avoid this issue.

        Args:
            dependent_element (str or Element):
                The element for which the chemical potential is plotted as a
                heatmap. If None (default), the last element in the bulk
                composition formula is used (which corresponds to the most
                electronegative element present).
            xlim (tuple):
                The x-axis limits for the plot. If None (default), the limits
                are set to the minimum and maximum values of the x-axis data,
                with padding equal to ``padding`` (default is 10% of the range).
            ylim (tuple):
                The y-axis limits for the plot. If None (default), the limits
                are set to the minimum and maximum values of the y-axis data,
                with padding equal to ``padding`` (default is 10% of the range).
            cbar_range (tuple):
                The range for the colourbar. If None (default), the range is
                set to the minimum and maximum values of the data.
            colormap (str, matplotlib.colors.Colormap):
                Colormap to use for the heatmap, either as a string (which can be
                a colormap name from https://www.fabiocrameri.ch/colourmaps or
                https://matplotlib.org/stable/users/explain/colors/colormaps), or
                a ``Colormap`` / ``ListedColormap`` object. If ``None`` (default),
                uses ``batlow`` from https://www.fabiocrameri.ch/colourmaps.

                Append "S" to the colormap name if using a sequential colormap
                from https://www.fabiocrameri.ch/colourmaps.
            padding (float):
                The padding to add to the x and y axis limits. If None (default),
                the padding is set to 10% of the range.
            title (str or bool):
                The title for the plot. If ``False`` (default), no title is added.
                If ``True``, the title is set to the bulk composition formula, or
                if ``str``, the title is set to the provided string.
            label_positions (list, dict or bool):
                The positions for the chemical formula line labels. If ``True``
                (default), the labels are placed using a custom ``doped`` algorithm
                which attempts to find the best possible positions (minimising
                overlap). If ``False``, no labels are added.
                Alternatively a dictionary can be provided, where the keys are
                the chemical formulae and the values are the x positions at which
                to place the line labels. If a list of floats, the labels are placed
                at the provided x positions.
            filename (PathLike):
                The filename to save the plot to. If None (default), the plot is not
                saved.
            style_file (PathLike):
                Path to a mplstyle file to use for the plot. If ``None`` (default),
                uses the default doped style (from ``doped/utils/doped.mplstyle``).

        Returns:
            plt.Figure: The ``matplotlib`` ``Figure`` object.
        """
        # TODO: Only works for ternary systems! For 2D, warn and return line plot?
        # For 4D+, could set constraint for fixed μ of other element, or fixed bordering phase?
        # -> Add to Future ToDo
        # Could look at Sungyhun's `cplapy` for doing 4D chempot plots?
        # TODO: Draft! Need to test for multiple systems
        # TODO: Plot extrinsic too?
        # TODO: Can use `yoffsets` parameter to shift the labels for vertical lines, to allow more
        #  control; implement this (removes need for np.unique() call)), units = plot y units
        # TODO: Code in this function (particularly label position handling and intersections) should be
        #  able to be made more succinct, and also modularise a bit?
        # TODO: Use ``ChemicalPotentialGrid`` from ``dopey_fermi`` PR when fixed and merged to develop (
        #  will also make handling >ternary systems easier?)
        # TODO: Option to only show all calculated competing phases?

        # Note that we could also add option to instead plot competing phases lines coloured,
        # with a legend added giving the composition of each competing phase line (as in the SI of
        # 10.1021/acs.jpcc.3c05204; Cs2SnTiI6 notebooks), but this isn't as nice/clear, and the same effect
        # can be achieved by the user by saving to PDF without labels, and manually colouring and adding
        # a legend in a vector graphics editor (e.g. Inkscape, Affinity Designer, Adobe Illustrator, etc.).
        from shakenbreak.plotting import _install_custom_font

        _install_custom_font()
        cpd = ChemicalPotentialDiagram(list(self.intrinsic_phase_diagram.entries))

        # check dimensionality:
        if len(cpd.elements) == 2:  # switch to line plot
            warnings.warn(
                "Chemical potential heatmap (i.e. 2D) plotting is not possible for a binary "
                "system, switching to a chemical potential line plot."
            )
            # TODO
        elif len(cpd.elements) != 3:
            raise ValueError(
                f"Chemical potential heatmap (i.e. 2D) plotting is only possible for ternary "
                f"systems, but this is a {len(cpd.elements)}-D system!"
            )
            # TODO: Allow fixed chempot value for other elements to reduce to 2/3D

        host_domains = cpd.domains[self.composition.reduced_formula]

        if dependent_element is None:  # set to last element in bulk comp, usually the anion as desired
            dependent_element = self.composition.elements[-1]
        elif isinstance(dependent_element, str):
            dependent_element = Element(dependent_element)
        assert isinstance(dependent_element, Element)  # typing

        dependent_el_idx = cpd.elements.index(dependent_element)
        independent_el_indices = [i for i in range(len(cpd.elements)) if i != dependent_el_idx]
        dependent_el_pts = np.array(host_domains[:, dependent_el_idx])
        independent_el_pts = np.array(host_domains[:, independent_el_indices])
        hull = ConvexHull(independent_el_pts)  # convex hull of points to get bounding polygon

        # create a dense grid that covers the entire range of the vertices:
        x_min, y_min = independent_el_pts.min(axis=0)
        x_max, y_max = independent_el_pts.max(axis=0)
        grid_x, grid_y = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]  # type: ignore
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        # Delaunay triangulation to get points inside the stability hull:
        delaunay = Delaunay(hull.points[hull.vertices])
        inside_hull = delaunay.find_simplex(grid_points) >= 0
        points_inside = grid_points[inside_hull]

        # interpolate the values to get the dependent chempot here:
        values_inside = griddata(independent_el_pts, dependent_el_pts, points_inside, method="linear")

        style_file = style_file or f"{os.path.dirname(__file__)}/utils/doped.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        fig, ax = plt.subplots()
        mesh_x, x_indices = np.unique(points_inside[:, 0], return_inverse=True)
        mesh_y, y_indices = np.unique(points_inside[:, 1], return_inverse=True)
        mesh_z = np.full((len(mesh_y), len(mesh_x)), np.nan)  # Create the mesh grid, init with NaNs
        mesh_z[y_indices, x_indices] = values_inside  # populate the grid

        vmin = cbar_range[0] if cbar_range else None
        vmax = cbar_range[1] if cbar_range else None
        if vmax is None and np.isclose(np.nanmax(mesh_z), 0, atol=3e-2):
            vmax = 0  # extend to 0, as sometimes cutoff at -0.01 eV etc

        cmap = get_colormap(colormap, default="batlow")  # get colormap choice
        dep_mu = ax.pcolormesh(
            mesh_x, mesh_y, mesh_z, rasterized=True, cmap=cmap, shading="auto", vmax=vmax, vmin=vmin
        )

        cbar = fig.colorbar(dep_mu)

        x_range = abs(x_max - x_min)
        y_range = abs(y_max - y_min)

        if xlim is None:
            x_padding = padding or x_range * 0.1
            xlim = (float(x_min - x_padding), float(x_max + x_padding))

        if ylim is None:
            y_padding = padding or y_range * 0.1
            ylim = (float(y_min - y_padding), float(y_max + y_padding))

        ax.set_xlim(*xlim), ax.set_ylim(*ylim)
        cbar.set_label(rf"$\Delta\mu$ ({dependent_element.symbol}) (eV)")
        ax.set_xlabel(rf"$\Delta\mu$ ({cpd.elements[independent_el_indices[0]].symbol}) (eV)")
        ax.set_ylabel(rf"$\Delta\mu$ ({cpd.elements[independent_el_indices[1]].symbol}) (eV)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if title:
            if not isinstance(title, str):
                title = latexify(f"{self.composition.reduced_formula}")
            ax.set_title(title)

        # plot formation energy lines:
        lines = []
        labels = {}  # {formula: line function}
        intersections = []
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        for formula, pts in cpd.domains.items():
            if formula == self.composition.reduced_formula:
                continue

            intersection = None  # catch cases where lines are not within plot boundaries
            x = np.linspace(-50, 50, 1000)
            # get domain points which match those in host_domains:
            domain_pts = [
                chempot_coords
                for chempot_coords in pts
                if any(np.allclose(chempot_coords, coords) for coords in host_domains)
            ]
            if len(domain_pts) < 2:
                continue  # not a stable bordering phase

            f = interp1d(
                np.array(domain_pts)[:, independent_el_indices[0]],
                np.array(domain_pts)[:, independent_el_indices[1]],
                kind="linear",
                assume_sorted=False,
                fill_value="extrapolate",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "divide by zero")
                vertical_line = (np.abs(f(x)) == np.inf).any()
            if vertical_line:  # handle any vertical lines
                line = ax.axvline(
                    domain_pts[0][independent_el_indices[0]], label=latexify(formula), color="k"
                )
                x_val = domain_pts[0][independent_el_indices[0]]
                y_min, y_max = ax.get_ylim()
                intersection = ((x_val, y_min), (x_val, y_max))
            else:
                (line,) = ax.plot(x, f(x), label=latexify(formula), color="k")

                # Find intersections with the bounding box:
                x_intersections = []
                y_intersections = []

                # Check intersections with vertical bounds (x_min and x_max)
                y_x_min = f(x_min)
                y_x_max = f(x_max)
                if y_min <= y_x_min <= y_max:
                    x_intersections.append((x_min, float(y_x_min)))
                if y_min <= y_x_max <= y_max:
                    x_intersections.append((x_max, float(y_x_max)))

                # Check intersections with horizontal bounds (y_min and y_max)
                if not np.isclose(float(y_x_min), float(y_x_max)):  # not a horizontal line
                    x_y_min = interp1d(
                        f(x), x, assume_sorted=False, kind="linear", fill_value="extrapolate"
                    )(y_min)
                    x_y_max = interp1d(
                        f(x), x, assume_sorted=False, kind="linear", fill_value="extrapolate"
                    )(y_max)
                    if x_min <= x_y_min <= x_max:
                        y_intersections.append((float(x_y_min), y_min))
                    if x_min <= x_y_max <= x_max:
                        y_intersections.append((float(x_y_max), y_max))

                intersection = np.unique(np.round((x_intersections + y_intersections), 4), axis=0)
                # in case intersects at x/y corner (which would give a duplicate)

            if intersection is not None and np.size(intersection) > 0:
                intersections.append(intersection)
                lines.append(line)
                labels[formula] = f  # labels is dict of formula: line function

        # pre-set x_points:
        if label_positions:
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

            elif isinstance(label_positions, dict):  # match formula (key) to line:
                label_positions = {k: labels[k] for k in label_positions if k in labels}

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The value at position")
                labelLines(lines, xvals=label_positions, align=False, color="black")

        # make sure all labels are well enclosed within the plot:
        latexified_labels = {latexify(k): v for k, v in labels.items()}
        # Get all the text artists (labels) in the current axes:
        label_text_artists = [
            artist
            for artist in ax.get_children()
            if isinstance(artist, plt.Text) and artist.get_text() in latexified_labels
        ]
        for text in label_text_artists:
            bbox = text.get_window_extent().transformed(plt.gca().transData.inverted())
            # if bbox bounds outside of plot, move text to inside plot:
            new_position = text.get_position()
            delta_x = delta_y = 0
            f = latexified_labels[text.get_text()]

            if bbox.xmin < xlim[0] or bbox.xmax > xlim[1] or bbox.ymin < ylim[0] or bbox.ymax > ylim[1]:
                if bbox.xmin < xlim[0]:
                    delta_x = (xlim[0] - bbox.xmin) + x_padding * 0.25
                elif bbox.xmax > xlim[1]:
                    delta_x = (xlim[1] - bbox.xmax) - x_padding * 0.25
                if delta_x != 0:
                    new_position = (new_position[0] + delta_x, new_position[1] + f(delta_x) - f(0))

                if bbox.ymin < ylim[0] or bbox.ymax > ylim[1]:
                    x = np.linspace(-50, 50, 1000)
                    f_inv = interp1d(f(x), x, assume_sorted=False, kind="linear", fill_value="extrapolate")

                if bbox.ymin < ylim[0]:
                    delta_y = (ylim[0] - bbox.ymin) + y_padding * 0.25
                if bbox.ymax > ylim[1]:
                    delta_y = (ylim[1] - bbox.ymax) - y_padding * 0.25
                if delta_y != 0:
                    new_position = (new_position[0] + f_inv(delta_y) - f_inv(0), new_position[1] + delta_y)

            text.set_position(new_position)

        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=600)

        return fig

    def __repr__(self):
        """
        Returns a string representation of the ``CompetingPhasesAnalyzer``
        object.
        """
        formula = self.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        # TODO: When entries parsing/handling implemented:
        # joined_entry_list = "\n".join([entry.data["doped_name"] for entry in self.entries])
        parsed_folder_list = "\n".join([str(folder) for folder in self.parsed_folders])
        return (
            f"doped CompetingPhasesAnalyzer for bulk composition {formula} with "
            f"{len(self.parsed_folders)} folders parsed:\n{parsed_folder_list}\n\n"
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )


def _parse_vasprun_and_catch_exception(vasprun_path: PathLike) -> tuple[Union[str, Vasprun], PathLike]:
    """
    Parse a VASP vasprun.xml file, catching any exceptions and returning the
    error message and the path to the vasprun.xml file if an exception is
    raised.
    """
    try:
        vasprun = get_vasprun(vasprun_path)
        folder = vasprun_path.rstrip(".gz").rstrip("vasprun.xml")
        return vasprun, folder
    except Exception as e:
        return str(e), vasprun_path


def _possible_label_positions_from_bbox_intersections(
    intersections: Union[list[float], np.ndarray[float]], positions_per_line=3
) -> np.ndarray[float]:
    """
    From a list or array of ``intersections``, which contains the intersections
    of lines with a plot bounding box (limits) and thus has shape ``(N_lines,
    2, 2)`` (because 2 intersections with 2 (x,y) coordinates per line),
    returns ``positions_per_line`` uniformly-spaced possible x,y coordinates
    for labels of those lines.

    e.g. if ``positions_per_line = 3`` (default), then returns the x,y coordinates
    for the positions which are 1/4, 2/4 and 3/4 along the line between the two
    bbox intersections.

    Args:
        intersections (list or np.ndarray):
            A list or array of intersections of lines with the plot bounding box.
            Should have shape ``(N_lines, 2, 2)``.
        positions_per_line (int):
            The number of possible label positions per line to return. Default is
            3, which returns positions at 1/4, 2/4 and 3/4 along the line between
            the two bbox intersections.

    Returns:
        np.ndarray:
            The possible label positions, with shape ``(N_lines, positions_per_line, 2)``.
    """
    poss_label_positions = np.zeros((len(intersections), positions_per_line, 2))
    for label_idx, points in enumerate(intersections):  # get possible label positions
        for line_pos_idx in range(positions_per_line):
            if (
                points[1][0] == points[0][0]
            ):  # vertical line, will only allow midpoint with current labellines
                # see https://github.com/cphyc/matplotlib-label-lines/pull/136
                poss_label_positions[label_idx, line_pos_idx, :] = (
                    points[0][0],
                    np.array(points)[:, 1].mean(),
                )
                continue

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
) -> Union[np.ndarray, tuple[np.ndarray, float]]:
    """
    From an array of possible label positions, find the best possible
    combination of label positions which maximises the distance between labels
    (i.e. minimises overlap).

    Args:
        poss_label_positions (np.ndarray):
            The possible label positions, with shape ``(N_lines, positions_per_line, 2)``.
        x_range (float):
            The range of the x-axis to use for normalisation. Default is 1.
        y_range (float):
            The range of the y-axis to use for normalisation. Default is 1.
        return_best_norm_dist (bool):
            Whether to return the best normalised minimum distance between labels.
            Default is False.

    Returns:
        np.ndarray:
            The best possible label positions, with shape ``(N_lines, 2)``.
        float:
            The best normalised minimum distance between labels,
            if ``return_best_norm_dist`` is True.
    """
    # Get all possible combinations of indices, for the first two dimensions (N_labels,
    # N_possibilities_per_label):
    N_labels, N_possibilities_per_label, N_xy = poss_label_positions.shape
    combinations = list(itertools.product(range(N_possibilities_per_label), repeat=N_labels))

    # Prepare an empty array to store the results
    result = np.zeros((len(combinations), N_labels, N_xy))  # N_xy should be 2

    # Fill the result array with the corresponding coordinates
    for i, combo in enumerate(combinations):
        result[i] = poss_label_positions[np.arange(N_labels), combo]

    #  result.shape should be (N_possibilities_per_label**N_labels, N_labels, N_xy = 2)
    all_combos = np.unique(result, axis=0)  # get unique combos (accounts for vertical lines which
    # currently only allow midpoint to be used)
    all_combos[:, :, 0] /= x_range
    all_combos[:, :, 1] /= y_range
    dists = np.linalg.norm(all_combos[:, :, np.newaxis] - all_combos[:, np.newaxis, :], axis=-1)
    # get upper diagonal of distances (removes self-distances = 0 and duplicates ((i,j) = (j,i))
    mask = np.triu(np.ones((N_labels, N_labels)), k=1).astype(bool)
    unique_dists = dists[:, mask]
    dists_list = [sorted(sublist) for sublist in unique_dists.tolist()]
    max_idx = dists_list.index(sorted(dists_list, reverse=True)[0])
    best_combo = all_combos[max_idx]
    best_combo[:, 0] *= x_range  # reverse normalisation

    if return_best_norm_dist:
        return best_combo[:, 0], dists_list[max_idx][0]

    return best_combo[:, 0]


def get_X_rich_limit(X: str, chempots: dict):
    """
    Determine the chemical potential limit of the input chempots dict which
    corresponds to the most X-rich conditions.

    Args:
        X (str): Elemental species (e.g. "Te")
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
        X (str): Elemental species (e.g. "Te")
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


def _move_dict_to_start(data, key, value):
    for index, item in enumerate(data):
        if key in item and item[key] == value:
            data.insert(0, data.pop(index))
            return


def combine_extrinsic(first, second, extrinsic_species):
    # TODO: Can we just integrate this to `CompetingPhaseAnalyzer`, so you just pass in a list of
    # extrinsic species and it does the right thing?
    """
    Combines chemical limits for different extrinsic species using chemical
    limits json file from ChemicalPotentialAnalysis.

    Usage explained in the example jupyter notebook
    Args:
        first (dict): First chemical potential dictionary, it can contain extrinsic species other
        than the set extrinsic species
        second (dict): Second chemical potential dictionary, it must contain the extrinsic species
        extrinsic_species (str): Extrinsic species in the second dictionary
    Returns:
        dict.
    """
    keys = ["elemental_refs", "limits", "limits_wrt_el_refs"]
    if any(key not in first for key in keys):
        raise KeyError(
            "the first dictionary doesn't contain the correct keys - it should include "
            "elemental_refs, limits and limits_wrt_el_refs"
        )

    if any(key not in second for key in keys):
        raise KeyError(
            "the second dictionary doesn't contain the correct keys - it should include "
            "elemental_refs, limits and limits_wrt_el_refs"
        )

    if extrinsic_species not in second["elemental_refs"]:
        raise ValueError("extrinsic species is not present in the second dictionary")

    cpa1 = copy.deepcopy(first)
    cpa2 = copy.deepcopy(second)
    new_limits = {}
    for (k1, v1), (k2, v2) in zip(list(cpa1["limits"].items()), list(cpa2["limits"].items())):
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
