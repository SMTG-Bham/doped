"""
Code for analysing the thermodynamics of defect formation in solids, including
calculation of formation energies as functions of Fermi level and chemical
potentials, charge transition levels, defect/carrier concentrations etc.
"""

import contextlib
import importlib.util
import os
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import deepcopy
from functools import partial, reduce
from itertools import chain, product
from operator import methodcaller
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.figure import Figure
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.core.composition import Composition
from pymatgen.electronic_structure.dos import Dos, FermiDos, Spin, f0
from pymatgen.io.vasp import Vasprun
from pymatgen.util.typing import PathLike
from scipy.optimize import brentq
from scipy.spatial import HalfspaceIntersection
from tqdm import tqdm

from doped import _doped_obj_properties_methods
from doped.chemical_potentials import ChemicalPotentialGrid, get_X_poor_limit, get_X_rich_limit
from doped.core import (
    DefectEntry,
    _get_dft_chempots,
    _no_chempots_warning,
    _orientational_degeneracy_warning,
)
from doped.generation import sort_defect_entries
from doped.utils.parsing import (
    _compare_incar_tags,
    _compare_kpoints,
    _compare_potcar_symbols,
    _get_bulk_supercell,
    _get_defect_supercell_site,
    get_nelect_from_vasprun,
    get_vasprun,
)
from doped.utils.plotting import _rename_key_and_dicts, formation_energy_plot
from doped.utils.symmetry import cluster_coords, get_all_equiv_sites, get_primitive_structure, get_sga

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from py_sc_fermi.defect_species import DefectSpecies
        from py_sc_fermi.defect_system import DefectSystem
        from py_sc_fermi.dos import DOS


def bold_print(string: str) -> None:
    """
    Does what it says on the tin.

    Prints the input string in bold.
    """
    print("\033[1m" + string + "\033[0m")


def _raise_limit_with_user_chempots_error(no_chempots=True):
    problem = (
        (
            "the supplied chempots are not in the doped format (i.e. with `limits` in the chempots dict), "
            "and instead correspond to just a single chemical potential limit"
        )
        if no_chempots
        else "no `chempots` have been supplied"
    )
    raise ValueError(
        f"You have specified a chemical potential limit, but {problem}, so `limit` cannot be used here!"
    )


def _parse_limit(chempots: dict, limit: str | None = None):
    if limit is not None:
        if limit in chempots["limits"]:
            return limit  # direct match, just return limit name
        if "limits" not in chempots or "User Chemical Potentials" in chempots["limits"]:
            # user specified chempots
            _raise_limit_with_user_chempots_error(no_chempots=True)
        if "No User Chemical Potentials" in chempots["limits"]:
            _raise_limit_with_user_chempots_error(no_chempots=False)
        if "rich" in limit:
            limit = get_X_rich_limit(limit.split("-")[0], chempots)
        elif "poor" in limit:
            limit = get_X_poor_limit(limit.split("-")[0], chempots)

    return limit


def get_rich_poor_limit_dict(chempots: dict) -> dict:
    """
    Get a dictionary of {"X-rich": limit, "X-poor": limit...} for each element
    X in the chempots phase diagram.

    Args:
        chempots (dict): The chempots dict, in the doped format.
    """
    if (
        "limits" not in chempots
        or "User Chemical Potentials" in chempots["limits"]
        or "No User Chemical Potentials" in chempots["limits"]
    ):
        raise ValueError(
            "The supplied chempots are not in the doped format (i.e. with `limits` in the "
            "chempots dict), and so the X-rich/poor limits cannot be determined!"
        )

    comps = {comp for key in chempots["limits"] for comp in key.split("-")}
    elts = {element.symbol for comp in comps for element in Composition(comp).elements}
    limit_dict = {f"{elt}-rich": get_X_rich_limit(elt, chempots) for elt in elts}
    limit_dict.update({f"{elt}-poor": get_X_poor_limit(elt, chempots) for elt in elts})
    return limit_dict


def _get_limit_name_from_dict(limit, limit_rich_poor_dict, bracket=False):
    if limit_rich_poor_dict and limit in limit_rich_poor_dict.values():
        # get first key with matching value:
        x_rich_poor = list(limit_rich_poor_dict.keys())[list(limit_rich_poor_dict.values()).index(limit)]
        return f"{x_rich_poor} ({limit})" if bracket else x_rich_poor
    return limit


def _update_old_chempots_dict(chempots: dict | None = None) -> dict | None:
    """
    Update a chempots dict in the old ``doped`` format (i.e. with ``facets``
    rather than ``limits``) to that of the new format.

    Also replaces any usages of ``"elt_refs"`` with ``"el_refs"``.
    """
    if chempots is not None and any("facets" in key or "elt_refs" in key for key in chempots):
        chempots = deepcopy(chempots)  # don't modify original dict, only deepcopy if needed
        for key in list(chempots.keys()):
            chempots[key.replace("elt_refs", "el_refs").replace("facets", "limits")] = chempots.pop(key)

    return chempots


def _parse_chempots(
    chempots: dict | None = None, el_refs: dict | None = None, update_el_refs: bool = False
) -> tuple[dict | None, dict | None]:
    """
    Parse the chemical potentials input, formatting them in the ``doped``
    format for use in analysis functions.

    Can be either ``doped`` format or user-specified format.

    Returns parsed ``chempots`` and ``el_refs``
    """
    for i in [chempots, el_refs]:
        if i is not None and not isinstance(i, dict):
            raise ValueError(
                f"Invalid chempots/el_refs format:\nchempots: {type(chempots)}\nel_refs: {type(el_refs)}\n"
                "Must be a dict (e.g. from `CompetingPhasesAnalyzer.chempots`) or `None`!"
            )

    chempots = _update_old_chempots_dict(chempots)  # this deepcopies, making sure we don't overwrite

    if chempots is None:
        if el_refs is not None:
            chempots = {
                "limits": {"User Chemical Potentials": el_refs},
                "elemental_refs": el_refs,
                "limits_wrt_el_refs": {"User Chemical Potentials": {el: 0 for el in el_refs}},
            }

        return chempots, el_refs

    if "limits_wrt_el_refs" in chempots:  # doped format
        if not update_el_refs:
            el_refs = chempots.get("elemental_refs", el_refs)

        if el_refs is not None:  # update el_refs in chempots dict
            chempots["elemental_refs"] = el_refs
            chempots["limits"] = {
                limit: {
                    el: relative_chempot + el_refs[el]
                    for el, relative_chempot in chempots["limits_wrt_el_refs"][limit].items()
                }
                for limit in chempots["limits_wrt_el_refs"]
            }

        return chempots, chempots.get("elemental_refs")

    if el_refs is None:
        chempots = {
            "limits": {"User Chemical Potentials": chempots},
            "elemental_refs": {el: 0 for el in chempots},
            "limits_wrt_el_refs": {"User Chemical Potentials": chempots},
        }

    else:  # relative chempots and el_refs given
        relative_chempots = chempots
        chempots = {"limits_wrt_el_refs": {"User Chemical Potentials": relative_chempots}}
        chempots["elemental_refs"] = el_refs
        chempots["limits"] = {
            "User Chemical Potentials": {
                el: relative_chempot + el_refs[el] for el, relative_chempot in relative_chempots.items()
            }
        }

    return chempots, chempots.get("elemental_refs")


def raw_energy_from_chempots(composition: str | dict | Composition, chempots: dict) -> float:
    """
    Given an input composition (as a ``str``, ``dict`` or ``pymatgen``
    ``Composition`` object) and chemical potentials dictionary, get the
    corresponding raw energy of the composition (i.e. taking the energies given
    in the ``'limits'`` subdicts of ``chempots``, in the ``doped`` chemical
    potentials dictionary format).

    Args:
        composition (Union[str, dict, Composition]):
            Composition to get the raw energy of.
        chempots (dict):
            Chemical potentials dictionary.

    Returns:
        Raw energy of the composition.
    """
    if not isinstance(composition, Composition):
        composition = Composition(composition)

    if "limits" not in chempots:
        chempots, _el_refs = _parse_chempots(chempots)  # type: ignore

    raw_energies_dict = dict(next(iter(chempots["limits"].values())))

    if any(el.symbol not in raw_energies_dict for el in composition.elements):
        warnings.warn(
            f"The chemical potentials dictionary (with elements {list(raw_energies_dict.keys())} does not "
            f"contain all the elements in the host composition "
            f"({[el.symbol for el in composition.elements]})!"
        )

    # note this can also be achieved with: (implements exact same code)
    # from pymatgen.core.composition import ChemicalPotential
    # raw_energy = ChemicalPotential(raw_energies_dict).get_energy(composition)

    return sum(raw_energies_dict.get(el.symbol, 0) * stoich for el, stoich in composition.items())


def group_defects_by_type_and_distance(
    defect_entries: list[DefectEntry], dist_tol: float = 1.5, symprec: float = 0.1
) -> dict[str, dict[int, set[DefectEntry]]]:
    """
    Given an input list of ``DefectEntry`` objects, returns a dictionary of
    format ``{simple defect name: {cluster index: {DefectEntry, ...}}``, with
    defect types as keys, and values being sub-dictionaries of defect entries
    clustered according to the given ``dist_tol`` distance tolerance (between
    symmetry-equivalent sites in the bulk supercell). ``simple defect name`` is
    the nominal defect type (e.g. ``Te_i`` for ``Te_i_Td_Cd2.83_+2``) and
    ``{DefectEntry, ...}`` is a set of ``DefectEntry`` objects which have been
    grouped in the same cluster.

    This is used to group together different defect entries (different charge
    states, and/or ground and metastable states (different spin or geometries))
    which correspond to the same defect type (e.g. interstitials at a given
    site) and occupy similar sites in the host lattice, which is then used in
    plotting, transition level analysis and defect concentration calculations;
    e.g. in the frozen defect approximation, the total concentration of a given
    defect type group is calculated at the annealing temperature, and then the
    equilibrium relative population of the constituent entries is recalculated
    at the quenched temperature.

    Note: The ``get_min_dist_between_equiv_sites`` function in
    ``doped.utils.symmetry`` can be useful for analysing the inter-defect
    distances and resulting clustering behaviour.

    Args:
        defect_entries (list[DefectEntry]):
            A list of ``DefectEntry`` objects to group together based on type
            and distance between symmetry-equivalent sites.
        dist_tol (float):
            Distance threshold (in Å) for clustering equivalent defect sites.
            (Default: 1.5)
        symprec (float):
            Symmetry precision for finding equivalent sites in the bulk
            supercell, for site clustering. Default is 0.1 (matching ``doped``
            default for point symmetry determination for relaxed defect
            supercells).

    Returns:
        dict:
            Dictionary of
            ``{simple defect name: {cluster index: {DefectEntry, ...}}}``.
    """
    # Note: This algorithm works well for the vast majority of cases, however because it involves
    # clustering, the results can be a little sensitive to which / how many defects are parsed together
    # (due to daisy-chaining effects). The user can always adjust `dist_tol` as desired.
    defect_type_cluster_dict: dict[str, dict[int, set[DefectEntry]]] = {
        entry.defect.name: defaultdict(set) for entry in defect_entries
    }
    for defect_name in list(defect_type_cluster_dict.keys()):
        defect_type_cluster_dict[defect_name] = group_defects_by_distance(
            [entry for entry in defect_entries if entry.defect.name == defect_name],
            dist_tol=dist_tol,
            symprec=symprec,
        )

    return defect_type_cluster_dict


def group_defects_by_distance(
    entry_list: list[DefectEntry], dist_tol: float = 1.5, symprec: float = 0.1
) -> dict[int, set[DefectEntry]]:
    r"""
    Given an input list of ``DefectEntry`` objects, returns a dictionary of
    defect entries clustered according to the given ``dist_tol`` distance
    tolerance (between symmetry-equivalent sites in the bulk supercell), in the
    format: ``{cluster index: {DefectEntry, ...}}``.

    This is used to group together different defect entries (different charge
    states, and/or ground and metastable states (different spin or geometries))
    which occupy similar sites in the host lattice, which is then used in
    plotting, transition level analysis and defect concentration calculations;
    e.g. in the frozen defect approximation, the total concentration of a given
    defect type group is calculated at the annealing temperature, and then the
    equilibrium relative population of the constituent entries is recalculated
    at the quenched temperature. When ``site_competition = True`` (default) in
    defect concentration calculations, the grouping returned by this function
    is used to determine which defects occupy the same sites (and hence compete
    for site occupancy). Note that while large ``dist_tol``\s are often
    preferable for plotting (to condense the defect formation energy lines),
    this is often not ideal for determining site competition in concentration
    analyses as it can lead to unrealistically-large clusters.

    Note: The ``get_min_dist_between_equiv_sites`` function in
    ``doped.utils.symmetry`` can be useful for analysing the inter-defect
    distances and resulting clustering behaviour.

    Args:
        entry_list ([DefectEntry, ...]):
            A list of ``DefectEntry`` objects to group together.
        dist_tol (float):
            Distance threshold (in Å) for clustering equivalent defect sites.
            (Default: 1.5)
        symprec (float):
            Symmetry precision for finding equivalent sites in the bulk
            supercell, for site clustering. Default is 0.1 (matching ``doped``
            default for point symmetry determination for relaxed defect
            supercells).

    Returns:
        dict: Dictionary of ``{cluster index: {DefectEntry, ...}}``.
    """
    bulk_supercell = _get_bulk_supercell(entry_list[0])
    bulk_lattice = bulk_supercell.lattice
    bulk_supercell_sga = get_sga(bulk_supercell, symprec=symprec)
    symm_bulk_struct = bulk_supercell_sga.get_symmetrized_structure()
    bulk_symm_ops = bulk_supercell_sga.get_symmetry_operations()

    equiv_sites_entries_dict: dict[tuple[tuple[float, float, float], ...], list[DefectEntry]] = (
        defaultdict(list)
    )  # {(equiv defect sites): entry list}}

    # Clustering Workflow:
    # 1. Generate ``equiv_sites_entries_dict``: {(equiv defect sites): entry list}, initially joining
    #    together any entries which are within min(0.05, dist_tol) Å of each other (i.e. (near-)exact
    #    matches).
    # 2. Cluster equivalent defect sites using ``cluster_coords`` with ``dist_tol``.
    # 3. Take unique clusters (of equivalent sites) and start with the largest (i.e. accounting for the
    #    fact that the defect entries corresponding to some clusters may be subsets of others,
    #    if e.g. they have a lower symmetry / higher multiplicity etc., and so we favour larger rather
    #    than smaller clusters, or equivalently less rather than more clusters), iterate through and add
    #    a defect entry cluster of all entries corresponding to that equiv site cluster which are not
    #    already members of an earlier defect entry cluster.

    for entry in entry_list:
        entry_bulk_supercell = _get_bulk_supercell(entry)
        if entry_bulk_supercell.lattice != bulk_lattice:
            # recalculate bulk_symm_ops if bulk supercell differs
            bulk_supercell_sga = get_sga(entry_bulk_supercell, symprec=symprec)
            symm_bulk_struct = bulk_supercell_sga.get_symmetrized_structure()
            bulk_symm_ops = bulk_supercell_sga.get_symmetry_operations()

        # need to use relaxed defect site if bulk_site not in calculation_metadata:
        bulk_site = entry.calculation_metadata.get("bulk_site") or _get_defect_supercell_site(entry)

        # get min distances to each equiv_site_tuple for previously checked defect entries:
        min_dist_list = [  # min dist for all equiv site tuples, in case multiple less than dist_tol
            bulk_site.lattice.get_all_distances(bulk_site.frac_coords, list(equiv_site_tuple)).min()
            for equiv_site_tuple in equiv_sites_entries_dict
        ]

        if min_dist_list and min(min_dist_list) < min(
            0.05, dist_tol
        ):  # near-exact match, add to corresponding entry list
            idxmin = np.argmin(min_dist_list)  # entry list
            equiv_sites_entries_dict[list(equiv_sites_entries_dict.keys())[idxmin]].append(entry)

        else:  # no exact match found, add new entry
            try:
                equiv_site_tuple = tuple(  # tuple because lists aren't hashable (can't be dict keys)
                    tuple(site.frac_coords) for site in symm_bulk_struct.find_equivalent_sites(bulk_site)
                )
            except ValueError:  # likely interstitials, need to add equiv sites to tuple
                equiv_site_tuple = tuple(  # tuple because lists aren't hashable (can't be dict keys)
                    tuple(frac_coords)
                    for frac_coords in get_all_equiv_sites(
                        bulk_site.frac_coords,
                        symm_bulk_struct,
                        bulk_symm_ops,
                        symprec=symprec,
                        just_frac_coords=True,
                    )
                )

            equiv_sites_entries_dict[equiv_site_tuple].append(entry)

    all_frac_coords = list(chain.from_iterable(equiv_sites_entries_dict.keys()))

    # cn is an array of cluster numbers, of length ``len(all_frac_coords)``, so we take the set of
    # cluster numbers ``n`` to get unique cluster numbers, sort by cluster size to favour larger
    # clusters (see workflow comment above), using ``np.where(cn == n)[0]`` to get the indices of ``cn`` /
    # ``all_frac_coords`` which are in cluster ``n``:
    cn = cluster_coords(all_frac_coords, bulk_supercell, dist_tol=dist_tol)
    unique_clusters = sorted(set(cn), key=lambda n: len(np.where(cn == n)[0]), reverse=True)

    clustered_defect_entries: dict[int, set[DefectEntry]] = {}  # {cluster index: {DefectEntry, ...}}

    def map_coords_to_keys(equiv_sites_entries_dict):
        coord_to_key_map = {}
        for key in equiv_sites_entries_dict:
            for coord in key:
                coord_to_key_map[coord] = key
        return coord_to_key_map

    # Pre-compute the map for quick look-up
    coord_to_key_map = map_coords_to_keys(equiv_sites_entries_dict)

    for n in unique_clusters:
        defect_entries = chain.from_iterable(  # get the corresponding defect entries:
            equiv_sites_entries_dict.get(  # get corresponding key and thus entries
                coord_to_key_map[all_frac_coords[i]], []
            )
            for i in np.where(cn == n)[0]
        )
        if new_entries_to_cluster := {  # reduce to unique entries which are not in any previous cluster
            entry
            for entry in defect_entries
            if not any(entry in entry_set for entry_set in clustered_defect_entries.values())
        }:
            clustered_defect_entries[n] = new_entries_to_cluster

    return clustered_defect_entries


def group_defects_by_name(entry_list: list[DefectEntry]) -> dict[str, list[DefectEntry]]:
    """
    Given an input list of ``DefectEntry`` objects, returns a dictionary of
    ``{defect name without charge: [DefectEntry]}``, where the values are lists
    of ``DefectEntry`` objects with the same defect name (excluding charge
    state).

    The ``DefectEntry.name`` attributes are used to get the defect names. These
    should be in the format:
    "{defect_name}_{optional_site_info}_{charge_state}". If the
    ``DefectEntry.name`` attribute is not defined or does not end with the
    charge state, then the entry will be renamed with the doped default name
    for the `unrelaxed` defect (i.e. using the point symmetry of the defect
    site in the bulk cell).

    For example, ``v_Cd_C3v_+1``, ``v_Cd_Td_+1`` and ``v_Cd_C3v_+2`` will be
    grouped as
    ``{"v_Cd_C3v": [v_Cd_C3v_+1, v_Cd_C3v_+2], "v_Cd_Td": [v_Cd_Td_+1]}``.

    Args:
        entry_list ([DefectEntry]):
            A list of ``DefectEntry`` objects to group together by defect name
            (without charge).

    Returns:
        dict: Dictionary of ``{defect name without charge: [DefectEntry]}``.
    """
    from doped.analysis import check_and_set_defect_entry_name

    grouped_entries: dict[str, list[DefectEntry]] = {}  # dict for groups of entries with the same prefix

    for _i, entry in enumerate(entry_list):
        # check defect entry name and (re)define if necessary
        check_and_set_defect_entry_name(entry, entry.name)
        entry_name_wout_charge = entry.name.rsplit("_", 1)[0]

        # If the prefix is not yet in the dictionary, initialize it with empty lists
        if entry_name_wout_charge not in grouped_entries:
            grouped_entries[entry_name_wout_charge] = []

        # Append the current entry to the appropriate group
        grouped_entries[entry_name_wout_charge].append(entry)

    return grouped_entries


class DefectThermodynamics(MSONable):
    """
    Class for analysing the calculated thermodynamics of defects in solids.
    Similar to a ``pymatgen`` ``PhaseDiagram`` object, allowing the analysis of
    formation energies as functions of Fermi level and chemical potentials
    (i.e. defect formation energy / transition level diagrams), charge
    transition levels, defect/carrier concentrations as functions of
    temperature, chemical potential etc.

    This class can be used to return:
        a) defect formation energy diagrams (plots),
        b) stability of charge states for a given defect,
        c) tables of all formation energies / symmetries & degeneracies,
        d) tables of equilibrium and constrained defect/carrier concentrations,
        e) charge transition levels, including metastable states,
        f) doping analysis,
        g) ...

    Note that ``DefectThermodynamics`` objects can be used to initialise the
    ``FermiSolver`` class in this module, which implements a number of
    convenience methods for thermodynamic analyses; such as scanning over
    temperatures, chemical potentials, effective dopant concentrations etc,
    minimising or maximising a target property (e.g. defect/carrier
    concentration), and also allowing greater control over constraints and
    approximations in defect concentration calculations; such as specifying
    (known) fixed concentrations of a defect/dopant, specifying defects to
    exclude from high-temperature concentration fixing in the frozen defect
    approximation (e.g. highly-mobile defects), and/or fixing defect charge
    states upon quenching.
    """

    def __init__(
        self,
        defect_entries: dict[str, DefectEntry] | list[DefectEntry],
        chempots: dict | None = None,
        el_refs: dict | None = None,
        vbm: float | None = None,
        band_gap: float | None = None,
        dist_tol: float = 1.5,
        check_compatibility: bool = True,
        bulk_dos: FermiDos | None = None,
        skip_dos_check: bool = False,
    ):
        r"""
        Create a ``DefectThermodynamics`` object, which can be used to analyse
        the calculated thermodynamics of defects in solids (formation energies,
        transition levels, concentrations etc.).

        Usually initialised using
        ``DefectsParser.get_defect_thermodynamics()``, but can also be
        initialised with a list or dict of ``DefectEntry`` objects (e.g. from
        ``DefectsParser.defect_dict``).

        Note that the ``DefectEntry.name`` attributes are used to label the
        defects in plots.

        Args:
            defect_entries (dict[str, DefectEntry] or list[DefectEntry]):
                A dict or list of ``DefectEntry`` objects. If a
                ``DefectEntry.name`` attribute is not defined or does not end
                with the charge state (as ``..._{charge state}``), then the
                entry will be renamed with the ``doped`` default name.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) which allows easy analysis over a range of
                chemical potentials -- where limit(s) (chemical potential
                limit(s)) to analyse/plot can later be chosen using the
                ``limits`` argument.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases in order to show the formal (relative)
                chemical potentials above the formation energy plot, in which
                case it is the formal chemical potentials (i.e. relative to
                the elemental references) that should be given here, otherwise
                the absolute (DFT) chemical potentials should be given.

                If ``None`` (default), sets all chemical potentials to zero.
                Chemical potentials can also be supplied later in each analysis
                function, or set using ``DefectThermodynamics.chempots = ...``
                (with the same input options).
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided in format generated by
                ``doped`` (see tutorials).

                If ``None`` (default), sets all elemental reference energies to
                zero. Reference energies can also be supplied later in each
                analysis function, or set using
                ``DefectThermodynamics.el_refs = ...`` (with the same input
                options).
            vbm (float):
                VBM eigenvalue to use as Fermi level reference point for
                analysis. If ``None`` (default), will use ``"vbm"`` from the
                ``calculation_metadata`` dict attributes of the parsed
                ``DefectEntry`` objects, which by default is taken from the
                bulk supercell VBM (unless ``bulk_band_gap_vr`` is set during
                defect parsing). Note that ``vbm`` should only affect the
                reference for the Fermi level values output by ``doped`` (as
                this VBM eigenvalue is used as the zero reference), thus
                affecting the position of the band edges in the defect
                formation energy plots and doping window / dopability limit
                functions, and the reference of the reported Fermi levels.
            band_gap (float):
                Band gap of the host, to use for analysis. If ``None``
                (default), will use "band_gap" from the
                ``calculation_metadata`` dict attributes of the ``DefectEntry``
                objects in ``defect_entries``.
            dist_tol (float):
                Threshold for the closest distance (in Å) between equivalent
                defect sites, for different species of the same defect type,
                to be grouped together (for plotting, transition level analysis
                and defect concentration calculations). In most cases, if the
                minimum distance between equivalent defect sites is less than
                ``dist_tol``, then they will be grouped together, otherwise
                treated as separate defects. See ``plot()`` and
                ``get_fermi_level_and_concentrations()`` for more information.
                (Default: 1.5)
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each
                defect entry (i.e. that all reference bulk energies are the
                same). Default is ``True``.
            bulk_dos (FermiDos or Vasprun or PathLike):
                ``pymatgen`` ``FermiDos`` for the bulk electronic density of
                states (DOS), for calculating Fermi level positions and
                defect/carrier concentrations. Alternatively, can be a
                ``pymatgen`` ``Vasprun`` object or path to the
                ``vasprun.xml(.gz)`` output of a bulk DOS calculation in VASP.
                Can also provide later in ``get_equilibrium_fermi_level()``,
                ``get_fermi_level_and_concentrations`` etc., or set using
                ``DefectThermodynamics.bulk_dos = ...`` (with the same input
                options).

                Usually this is a static calculation with the `primitive` cell
                of the bulk material, with relatively dense `k`-point sampling
                (especially for materials with disperse band edges) to ensure
                an accurately-converged DOS and thus Fermi level. Using large
                ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing)
                are recommended for best convergence (wrt `k`-point sampling)
                in VASP. Consistent functional settings should be used for the
                bulk DOS and defect supercell calculations. See
                https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations
                (Default: None)
            skip_dos_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                the defect entries VBM by >0.05 eV. Should only be used when
                the reason for this difference is known/acceptable. Default is
                ``False`` (don't skip check).

        Key Attributes:
            defect_entries (dict[str, DefectEntry]):
                Dict of ``DefectEntry`` objects included in the
                ``DefectThermodynamics`` set, with their names as keys.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and hence concentrations etc.), in
                the ``doped`` format.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials.
            vbm (float):
                VBM energy to use as Fermi level reference point for analysis.
            band_gap (float):
                Band gap of the host, to use for analysis.
            dist_tol (float):
                Threshold for the closest distance (in Å) between equivalent
                defect sites, for different species of the same defect type,
                to be grouped together (for plotting, transition level analysis
                and defect concentration calculations).
            transition_levels (dict):
                Dictionary of charge transition levels for each defect entry.
                (e.g. ``{defect_name: {charge: transition_level}}``).
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each
                defect entry (i.e. that all reference bulk energies are the
                same).
            bulk_formula (str):
                The reduced formula of the bulk structure (e.g. "CdTe").
            bulk_dos (FermiDos):
                ``pymatgen`` ``FermiDos`` for the bulk electronic density of
                states (DOS), used for calculating Fermi level positions and
                defect/carrier concentrations.
            skip_dos_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                the defect entries VBM by >0.05 eV. Should only be used when
                the reason for this difference is known/acceptable.
        """
        if not defect_entries:
            raise ValueError(
                "No defects found in `defect_entries`. Please check the supplied dictionary is in the "
                "correct format (i.e. {'defect_name': defect_entry}), or as a list: [defect_entry]."
            )
        if isinstance(defect_entries, list):
            defect_entries = {entry.name: entry for entry in defect_entries}

        self._defect_entries = defect_entries
        self._chempots, self._el_refs = _parse_chempots(chempots, el_refs, update_el_refs=True)
        self._dist_tol = dist_tol
        self.check_compatibility = check_compatibility
        self.skip_dos_check = skip_dos_check

        # get and check VBM/bandgap values:
        def _raise_VBM_band_gap_value_error(vals, type="VBM"):
            raise ValueError(
                f"{type} values for defects in `defect_dict` do not match within 0.05 eV of each other, "
                f"and so are incompatible for thermodynamic analysis with DefectThermodynamics. The "
                f"{type} values in the dictionary are: {vals}. You should recheck the correct/same bulk "
                f"files were used when parsing. If this is acceptable, you can instead manually specify "
                f"{type} in DefectThermodynamics initialisation."
            )

        self.vbm = vbm
        self.band_gap = band_gap
        if self.vbm is None or self.band_gap is None:
            vbm_vals = [
                defect_entry.calculation_metadata.get("vbm")
                for defect_entry in self.defect_entries.values()
            ]
            vbm_vals = [vbm for vbm in vbm_vals if vbm is not None]
            band_gap_vals = [
                defect_entry.calculation_metadata.get(
                    "band_gap", defect_entry.calculation_metadata.get("gap")
                )
                for defect_entry in self.defect_entries.values()
            ]
            band_gap_vals = [band_gap for band_gap in band_gap_vals if band_gap is not None]

            # get the max difference in VBM & band_gap vals:
            if vbm_vals and max(vbm_vals) - min(vbm_vals) > 0.05 and self.vbm is None:
                _raise_VBM_band_gap_value_error(vbm_vals, type="VBM")
            elif vbm_vals and self.vbm is None:
                self.vbm = vbm_vals[0]

            if band_gap_vals and max(band_gap_vals) - min(band_gap_vals) > 0.05 and self.band_gap is None:
                _raise_VBM_band_gap_value_error(band_gap_vals, type="band_gap")
            elif band_gap_vals and self.band_gap is None:
                self.band_gap = band_gap_vals[0]

        for i, name in [(self.vbm, "VBM eigenvalue"), (self.band_gap, "band gap value")]:
            if i is None:
                raise ValueError(
                    f"No {name} was supplied or able to be parsed from the defect entries "
                    f"(calculation_metadata attributes). Please specify the {name} in the function input."
                )

        self.bulk_dos = bulk_dos  # use setter method, needs to be after setting VBM

        # order entries for deterministic behaviour (particularly for plotting)
        self._sort_parse_and_check_entries()

        bulk_entry = next(iter(self.defect_entries.values())).bulk_entry
        if bulk_entry is not None:
            self.bulk_formula = bulk_entry.structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
        else:
            self.bulk_formula = None

    def _sort_parse_and_check_entries(self):
        """
        Sort the defect entries, parse the transition levels, and check the
        compatibility of the bulk entries (if ``self.check_compatibility`` is
        ``True``).
        """
        defect_entries_dict: dict[str, DefectEntry] = {}
        for entry in self.defect_entries.values():
            # rename defect entry names in dict if necessary ("_a", "_b"...)
            entry_name, [
                defect_entries_dict,
            ] = _rename_key_and_dicts(
                entry.name,
                [
                    defect_entries_dict,
                ],
            )
            defect_entries_dict[entry_name] = entry

        sorted_defect_entries_dict = sort_defect_entries(defect_entries_dict)
        self._defect_entries = sorted_defect_entries_dict
        self._parse_transition_levels()  # cluster defects and determine transition levels
        if self.check_compatibility:
            self._check_bulk_compatibility()
            self._check_bulk_chempots_compatibility(self._chempots)

    def as_dict(self):
        """
        Returns:
            JSON-serializable dict representation of ``DefectThermodynamics``.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "defect_entries": self.defect_entries,
            "chempots": self.chempots,
            "el_refs": self.el_refs,
            "vbm": self.vbm,
            "band_gap": self.band_gap,
            "dist_tol": self.dist_tol,
            "check_compatibility": self.check_compatibility,
            "bulk_formula": self.bulk_formula,
            "bulk_dos": self.bulk_dos,
            "skip_dos_check": self.skip_dos_check,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstitute a ``DefectThermodynamics`` object from a dict
        representation created using ``as_dict()``.

        Args:
            d (dict): dict representation of ``DefectThermodynamics``.

        Returns:
            ``DefectThermodynamics`` object
        """
        warnings.filterwarnings(
            "ignore", "Use of properties is"
        )  # `message` only needs to match start of message

        def _get_defect_entry(entry_dict):
            if isinstance(entry_dict, DefectEntry):
                return entry_dict
            return DefectEntry.from_dict(entry_dict)

        if isinstance(d.get("defect_entries"), list):
            d["defect_entries"] = [_get_defect_entry(entry_dict) for entry_dict in d.get("defect_entries")]
        else:
            d["defect_entries"] = {
                name: _get_defect_entry(entry_dict) for name, entry_dict in d.get("defect_entries").items()
            }

        return cls(
            defect_entries=d.get("defect_entries"),
            chempots=d.get("chempots"),
            el_refs=d.get("el_refs"),
            vbm=d.get("vbm"),
            band_gap=d.get("band_gap"),
            dist_tol=d.get("dist_tol", 1.5),
            check_compatibility=d.get("check_compatibility", True),
            bulk_dos=(
                FermiDos.from_dict(d["bulk_dos"])
                if isinstance(d.get("bulk_dos"), dict)
                else d.get("bulk_dos")
            ),
            skip_dos_check=d.get("skip_dos_check", False),
        )

    def to_json(self, filename: PathLike | None = None):
        """
        Save the ``DefectThermodynamics`` object as a json file, which can be
        reloaded with the ``DefectThermodynamics.from_json()`` class method.

        Note that file extensions with ".gz" will be automatically compressed
        (recommended to save space)!

        Args:
            filename (PathLike):
                Filename to save json file as. If ``None``, the filename will
                be set as ``{Chemical Formula}_defect_thermodynamics.json.gz``
                where ``{Chemical Formula}`` is the chemical formula of the
                host material.
        """
        if filename is None:
            if self.bulk_formula is not None:
                filename = f"{self.bulk_formula}_defect_thermodynamics.json.gz"
            else:
                filename = "defect_thermodynamics.json.gz"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: PathLike):
        """
        Load a ``DefectThermodynamics`` object from a json(.gz) file.

        Note that ``.json.gz`` files can be loaded directly.

        Args:
            filename (PathLike):
                Filename of json file to load ``DefectThermodynamics``
                object from.

        Returns:
            ``DefectThermodynamics`` object
        """
        return loadfn(filename)

    def _get_chempots(
        self, chempots: dict | None = None, el_refs: dict | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        Parse chemical potentials, either using input values (after formatting
        them in the ``doped`` format) or using the class attributes if set.
        """
        if isinstance(chempots, dict) and "elemental_refs" in chempots and el_refs is None:
            # doped chempot dict input, use its elemental refs
            chempots, el_refs = _parse_chempots(chempots, chempots["elemental_refs"])

        else:  # use stored or provided el_refs
            chempots, el_refs = _parse_chempots(
                chempots or self.chempots, el_refs or self.el_refs, update_el_refs=True
            )
        if self.check_compatibility:
            self._check_bulk_chempots_compatibility(chempots)

        return chempots, el_refs

    def _parse_transition_levels(self, symprec: float = 0.1):
        r"""
        Parses the charge transition levels for defect entries in the
        ``DefectThermodynamics`` object, and stores information about the
        stable charge states, transition levels etc.

        Defect entries of the same type (e.g. ``Te_i``, ``v_Cd``) are grouped
        together (for plotting and transition level analysis) based on the
        minimum distance between (equivalent) defect sites, controlled by
        ``dist_tol`` (1.5 Å by default), to distinguish between different
        inequivalent sites.

        Code for parsing the transition levels was originally templated from
        the pyCDT (pymatgen<=2022.7.25) thermodynamics code (deleted in later
        versions).

        This function uses ``scipy``\'s ``HalfspaceIntersection`` to construct
        the polygons corresponding to defect stability as a function of the
        Fermi-level. The Halfspace Intersection constructs N-dimensional
        hyperplanes, in this case N=2, based on the equation of defect
        formation energy with considering chemical potentials:

            E_form = E_0^{Corrected} + Q_{defect}*(E_{VBM} + E_{Fermi}).

        Extra hyperplanes are constructed to bound this space so that the
        algorithm can actually find enclosed region. This code was modeled
        after the Halfspace Intersection code for the Pourbaix Diagram.

        Note: The ``get_min_dist_between_equiv_sites`` function in
        ``doped.utils.symmetry`` can be useful for analysing the inter-defect
        distances and resulting clustering behaviour.

        Args:
            symprec (float):
                Symmetry precision for finding equivalent sites in the bulk
                supercell, for site clustering. Default is 0.1 (matching
                ``doped`` default for point symmetry determination for relaxed
                defect supercells).
        """
        # determine defect charge transition levels:
        with warnings.catch_warnings():  # ignore formation energies chempots warning when just parsing TLs
            warnings.filterwarnings("ignore", message="No chemical potentials")
            midgap_formation_energies = [  # without chemical potentials
                entry.formation_energy(
                    fermi_level=0.5 * (self.band_gap if self.band_gap is not None else 0),
                    vbm=entry.calculation_metadata.get("vbm", self.vbm),
                )
                for entry in self.defect_entries.values()
            ]
        # note that in general, we have chosen to favour ``entry.calculation_metadata.get("vbm")`` over
        # ``self.vbm`` for all calculations of formation energies / transition levels / concentrations.
        # These values should be the same, and we have warnings if they differ by too much,
        # but in general the ``DefectEntry`` value should be the most reliable, as this is tied to its
        # raw supercell energy difference from the chosen bulk cell for that ``DefectEntry``.
        # but we do use ``self.band_gap`` preferably as this only affects the plot ranges (rather than
        # formation energies)

        # set range to {min E_form - 30, max E_form +30} eV for y (formation energy), and
        # {VBM - 1, CBM + 1} eV for x (fermi level)
        min_y_lim = min(midgap_formation_energies) - 30
        max_y_lim = max(midgap_formation_energies) + 30
        limits = [[-1, self.band_gap + 1], [min_y_lim, max_y_lim]]  # type: ignore

        stable_entries: dict = {}
        defect_charge_map: dict = {}
        transition_level_map: dict = {}
        all_entries: dict = {}  # similar format to stable_entries, but with all (incl unstable) entries

        try:
            self._clustered_defect_entries = group_defects_by_distance(
                list(self.defect_entries.values()), dist_tol=self.dist_tol, symprec=symprec
            )  # {cluster index: {DefectEntry, ...}}
            self._clustered_defect_entries_by_type = group_defects_by_type_and_distance(
                list(self.defect_entries.values()), dist_tol=self.dist_tol, symprec=symprec
            )  # {simple defect name: {cluster index: {DefectEntry, ...}}}
            grouped_entries_list = list(
                chain(*map(methodcaller("values"), self._clustered_defect_entries_by_type.values()))
            )  # [[DefectEntry, ...], ...]

        except Exception as e:
            grouped_entries = group_defects_by_name(list(self.defect_entries.values()))
            grouped_entries_list = list(grouped_entries.values())
            warnings.warn(
                f"Grouping (inequivalent) defects by distance failed with error: {e!r}"
                f"\nGrouping by defect names (`DefectEntry.name`) instead."
            )  # possibly different bulks (though this should be caught/warned about earlier), or not
            # parsed with recent doped versions etc

        for grouped_defect_entries in grouped_entries_list:
            sorted_defect_entries = sorted(
                grouped_defect_entries, key=lambda x: (-x.charge_state, x.get_ediff())
            )  # sort by charge (most positive first, following left-to-right order in formation energy
            # diagrams), and then energy (for those of the same charge) for ordering in output plots/dfs

            # prepping coefficient matrix for half-space intersection
            # [-Q, 1, -1*(E_form+Q*VBM)] -> -Q*E_fermi+E+-1*(E_form+Q*VBM) <= 0 where E_fermi and E are
            # the variables in the hyperplanes
            hyperplanes = np.array(
                [
                    [
                        -1.0 * entry.charge_state,
                        1,
                        -1.0
                        * (
                            entry.get_ediff()
                            + entry.charge_state * entry.calculation_metadata.get("vbm", self.vbm)
                        ),  # type: ignore
                    ]
                    for entry in sorted_defect_entries
                ]
            )

            border_hyperplanes = [
                [-1, 0, limits[0][0]],
                [1, 0, -1 * limits[0][1]],
                [0, -1, limits[1][0]],
                [0, 1, -1 * limits[1][1]],
            ]
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
            interior_point = [self.band_gap / 2, min(midgap_formation_energies) - 1.0]  # type: ignore
            hs_ints = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

            # Group the intersections and corresponding facets
            ints_and_facets_zip = zip(hs_ints.intersections, hs_ints.dual_facets, strict=False)
            # Only include the facets corresponding to entries, not the boundaries
            total_entries = len(sorted_defect_entries)
            ints_and_facets_filter = filter(
                lambda int_and_facet: all(np.array(int_and_facet[1]) < total_entries),
                ints_and_facets_zip,
            )
            # sort based on transition level
            ints_and_facets_list = sorted(
                ints_and_facets_filter, key=lambda int_and_facet: int_and_facet[0][0]
            )

            # take simplest (shortest) possible defect name, with lowest energy, as the name for that group
            possible_defect_names_and_energies = [
                (defect_entry.name.rsplit("_", 1)[0], defect_entry.get_ediff())
                for defect_entry in sorted_defect_entries
            ]  # names without charge
            defect_name_wout_charge = min(
                possible_defect_names_and_energies, key=lambda x: (len(x[0]), x[1])
            )[0]

            defect_name_wout_charge, output_dicts = _rename_key_and_dicts(
                defect_name_wout_charge,
                [transition_level_map, all_entries, stable_entries, defect_charge_map],
            )
            transition_level_map, all_entries, stable_entries, defect_charge_map = output_dicts

            if len(ints_and_facets_list) > 0:  # unpack into lists
                _, facets = zip(*ints_and_facets_list, strict=False)
                transition_level_map[defect_name_wout_charge] = {  # map of transition level: charge states
                    intersection[0]: sorted(
                        [sorted_defect_entries[i].charge_state for i in facet], reverse=True
                    )
                    for intersection, facet in ints_and_facets_list
                }
                stable_entries[defect_name_wout_charge] = sorted(
                    {sorted_defect_entries[i] for dual in facets for i in dual},
                    key=lambda x: (-x.charge_state, x.get_ediff(), len(x.name)),
                )
                defect_charge_map[defect_name_wout_charge] = sorted(
                    [entry.charge_state for entry in sorted_defect_entries], reverse=True
                )

            elif len(sorted_defect_entries) == 1:
                transition_level_map[defect_name_wout_charge] = {}
                stable_entries[defect_name_wout_charge] = [sorted_defect_entries[0]]
                defect_charge_map[defect_name_wout_charge] = [sorted_defect_entries[0].charge_state]

            else:  # if ints_and_facets is empty, then there is likely only one defect...
                # confirm formation energies dominant for one defect over other identical defects
                name_set = [entry.name for entry in sorted_defect_entries]
                with warnings.catch_warnings():  # ignore chempots warning when just parsing TLs
                    warnings.filterwarnings("ignore", message="No chemical potentials")
                    vb_list = [
                        entry.formation_energy(
                            fermi_level=limits[0][0], vbm=entry.calculation_metadata.get("vbm", self.vbm)
                        )
                        for entry in sorted_defect_entries
                    ]
                    cb_list = [
                        entry.formation_energy(
                            fermi_level=limits[0][1], vbm=entry.calculation_metadata.get("vbm", self.vbm)
                        )
                        for entry in sorted_defect_entries
                    ]

                vbm_def_index = vb_list.index(min(vb_list))
                name_stable_below_vbm = name_set[vbm_def_index]
                cbm_def_index = cb_list.index(min(cb_list))
                name_stable_above_cbm = name_set[cbm_def_index]

                if name_stable_below_vbm != name_stable_above_cbm:
                    raise ValueError(
                        f"HalfSpace identified only one stable charge out of list: {name_set}\n"
                        f"But {name_stable_below_vbm} is stable below vbm and "
                        f"{name_stable_above_cbm} is stable above cbm.\nList of VBM formation "
                        f"energies: {vb_list}\nList of CBM formation energies: {cb_list}"
                    )

                transition_level_map[defect_name_wout_charge] = {}
                stable_entries[defect_name_wout_charge] = [sorted_defect_entries[vbm_def_index]]
                defect_charge_map[defect_name_wout_charge] = sorted(
                    [entry.charge_state for entry in sorted_defect_entries], reverse=True
                )

            all_entries[defect_name_wout_charge] = sorted_defect_entries

        self.transition_level_map = transition_level_map
        self.stable_entries = stable_entries
        self.all_entries = all_entries
        self.defect_charge_map = defect_charge_map

        # sort dictionaries deterministically:
        self.transition_level_map = dict(
            sorted(self.transition_level_map.items(), key=lambda item: self._map_sort_func(item[0]))
        )
        self.stable_entries = dict(
            sorted(self.stable_entries.items(), key=lambda item: self._map_sort_func(item[0]))
        )
        self.all_entries = dict(
            sorted(self.all_entries.items(), key=lambda item: self._map_sort_func(item[0]))
        )
        self.defect_charge_map = dict(
            sorted(self.defect_charge_map.items(), key=lambda item: self._map_sort_func(item[0]))
        )

        self.transition_levels = {
            defect_name: list(defect_tls.keys())
            for defect_name, defect_tls in transition_level_map.items()
        }
        self.stable_charges = {
            defect_name: [entry.charge_state for entry in entries]
            for defect_name, entries in stable_entries.items()
        }

    def _map_sort_func(self, name_wout_charge):
        """
        Convenience sorting function for dictionaries in and outputs from
        ``DefectThermodynamics``.
        """
        for i in range(name_wout_charge.count("_") + 1):  # number of underscores in name
            split_name = name_wout_charge.rsplit("_", i)[0]
            if indices := [  # find first occurrence of name_wout_charge in defect_entries
                i for i, name in enumerate(self._defect_entries.keys()) if name.startswith(split_name)
            ]:
                return min(indices), split_name

        return 100, split_name  # if name not in defect_entries, put at end

    def _check_bulk_compatibility(self):
        """
        Helper function to quickly check if all entries have compatible bulk
        calculation settings, by checking that the energy of
        ``defect_entry.bulk_entry`` is the same for all defect entries.

        By proxy checks that same bulk/defect calculation settings were used in
        all cases, from each bulk/defect combination already being checked when
        parsing. This is to catch any cases where defects may have been parsed
        separately and combined (rather than altogether with DefectsParser,
        which ensures the same bulk in each case), and where a different bulk
        reference calculation was (mistakenly) used.
        """
        bulk_energies = [entry.bulk_entry.energy for entry in self.defect_entries.values()]
        if max(bulk_energies) - min(bulk_energies) > 0.02:  # 0.02 eV tolerance
            warnings.warn(
                f"Note that not all defects in `defect_entries` have the same reference bulk energy (bulk "
                f"supercell calculation at `bulk_path` when parsing), with energies differing by >0.02 "
                f"eV. This can lead to inaccuracies in predicted formation energies! The bulk energies of "
                f"defect entries in `defect_entries` are:\n"
                f"{[(name, entry.bulk_entry.energy) for name, entry in self.defect_entries.items()]}\n"
                f"You can suppress this warning by setting `DefectThermodynamics.check_compatibility = "
                f"False`."
            )

    def _check_bulk_defects_compatibility(self):
        """
        Helper function to quickly check if all entries have compatible
        defect/bulk calculation settings.

        Currently not used, as the bulk/defect compatibility is checked when
        parsing, and the compatibility across bulk calculations is checked with
        ``_check_bulk_compatibility()``.
        """
        # check each defect entry against its own bulk, and also check each bulk against each other
        reference_defect_entry = next(iter(self.defect_entries.values()))
        reference_run_metadata = reference_defect_entry.calculation_metadata["run_metadata"]
        for defect_entry in self.defect_entries.values():
            with warnings.catch_warnings(record=True) as captured_warnings:
                run_metadata = defect_entry.calculation_metadata["run_metadata"]
                # compare defect and bulk:
                _compare_incar_tags(run_metadata["bulk_incar"], run_metadata["defect_incar"])
                _compare_potcar_symbols(
                    run_metadata["bulk_potcar_symbols"], run_metadata["defect_potcar_symbols"]
                )
                _compare_kpoints(
                    run_metadata["bulk_actual_kpoints"],
                    run_metadata["defect_actual_kpoints"],
                    run_metadata["bulk_kpoints"],
                    run_metadata["defect_kpoints"],
                )

                # compare bulk and reference bulk:
                _compare_incar_tags(
                    reference_run_metadata["bulk_incar"],
                    run_metadata["bulk_incar"],
                    defect_name=f"other bulk (for {reference_defect_entry.name})",
                )
                _compare_potcar_symbols(
                    reference_run_metadata["bulk_potcar_symbols"],
                    run_metadata["bulk_potcar_symbols"],
                    defect_name=f"other bulk (for {reference_defect_entry.name})",
                )
                _compare_kpoints(
                    reference_run_metadata["bulk_actual_kpoints"],
                    run_metadata["defect_actual_kpoints"],
                    reference_run_metadata["bulk_kpoints"],
                    run_metadata["defect_kpoints"],
                    defect_name=f"other bulk (for {reference_defect_entry.name})",
                )

            if captured_warnings:
                concatenated_warnings = "\n".join(str(warning.message) for warning in captured_warnings)
                warnings.warn(
                    f"Incompatible defect/bulk calculation settings detected for defect "
                    f"{defect_entry.name}: \n{concatenated_warnings}"
                )

    def _check_bulk_chempots_compatibility(self, chempots: dict | None = None):
        r"""
        Helper function to quickly check if the supplied chemical potentials
        dictionary matches the bulk supercell used for the defect calculations,
        by comparing the raw energies (from the bulk supercell calculation, and
        that corresponding to the chemical potentials supplied).

        Args:
            chempots (dict, optional):
                Dictionary of chemical potentials to check compatibility with
                the bulk supercell calculations (``DefectEntry.bulk_entry``\s),
                in the ``doped`` format.

                If ``None`` (default), will use ``self.chempots`` (= 0 for all
                chemical potentials by default). This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)), or alternatively a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.
        """
        if chempots is None and self.chempots is None:
            return

        bulk_entry = next(entry.bulk_entry for entry in self.defect_entries.values())
        bulk_supercell_energy_per_atom = bulk_entry.energy / bulk_entry.composition.num_atoms
        bulk_chempot_energy_per_atom = (
            raw_energy_from_chempots(bulk_entry.composition, chempots or self.chempots)
            / bulk_entry.composition.num_atoms
        )

        if abs(bulk_supercell_energy_per_atom - bulk_chempot_energy_per_atom) > 0.025:
            warnings.warn(  # 0.05 eV intrinsic defect formation energy error tolerance, taking per-atom
                # chempot error and multiplying by 2 to account for how this would affect antisite
                # formation energies (extreme case)
                f"Note that the raw (DFT) energy of the bulk supercell calculation ("
                f"{bulk_supercell_energy_per_atom:.2f} eV/atom) differs from that expected from the "
                f"supplied chemical potentials ({bulk_chempot_energy_per_atom:.2f} eV/atom) by >0.025 eV. "
                f"This will likely give inaccuracies of similar magnitude in the predicted formation "
                f"energies! \n"
                f"You can suppress this warning by setting `DefectThermodynamics.check_compatibility = "
                f"False`."
            )

    def add_entries(
        self,
        defect_entries: dict[str, DefectEntry] | list[DefectEntry],
        check_compatibility: bool = True,
    ):
        """
        Add additional defect entries to the ``DefectThermodynamics`` object.

        Args:
            defect_entries ({str: DefectEntry} or [DefectEntry]):
                A dict or list of ``DefectEntry`` objects, to add to the
                ``DefectThermodynamics.defect_entries`` dict. If a
                ``DefectEntry.name`` attribute is not defined or does not end
                with the charge state (as ``..._{charge state}``), then the
                entry will be renamed with the ``doped`` default name.
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each
                defect entry (i.e. that all reference bulk energies are the
                same). Default is ``True``.
        """
        self.check_compatibility = check_compatibility
        if not defect_entries:
            raise ValueError(
                "No defects found in `defect_entries`. Please check the supplied dictionary is in the "
                "correct format (i.e. {'defect_name': defect_entry}), or as a list: [defect_entry]."
            )
        if isinstance(defect_entries, list):  # append 'pre_formatting' so we don't overwrite any existing
            defect_entries = {f"{entry.name}_pre_formatting": entry for entry in defect_entries}

        self._defect_entries.update(defect_entries)  # add new entries and format names
        self._sort_parse_and_check_entries()

    @property
    def defect_entries(self):
        """
        Get the dict of parsed ``DefectEntry`` objects in the
        ``DefectThermodynamics`` analysis object.
        """
        return self._defect_entries

    @defect_entries.setter
    def defect_entries(self, input_defect_entries):
        r"""
        Set the dict of parsed ``DefectEntry``\s to include in the
        ``DefectThermodynamics`` object, and reparse the thermodynamic
        information (transition levels etc).
        """
        self._defect_entries = input_defect_entries
        self._sort_parse_and_check_entries()

    @property
    def chempots(self):
        r"""
        Get the chemical potentials dictionary used for calculating the defect
        formation energies.

        ``chempots`` is a dictionary of chemical potentials to use for
        calculating the defect formation energies, in the form of:
        ``{"limits": [{'limit': [chempot_dict]}]}`` (the format generated by
        ``doped``\'s chemical potential parsing functions (see tutorials))
        which allows easy analysis over a range of chemical potentials -- where
        limit(s) (chemical potential limit(s)) to analyse/plot can later be
        chosen using the ``limits`` argument.
        """
        return self._chempots

    @chempots.setter
    def chempots(self, input_chempots):
        r"""
        Set the chemical potentials dictionary (``chempots``), and reparse to
        have the required ``doped`` format.

        ``chempots`` is a dictionary of chemical potentials to use for
        calculating the defect formation energies, in the form of:
        ``{"limits": [{'limit': [chempot_dict]}]}`` (the format generated by
        ``doped``\'s chemical potential parsing functions (see tutorials))
        which allows easy analysis over a range of chemical potentials -- where
        limit(s) (chemical potential limit(s)) to analyse/plot can later be
        chosen using the ``limits`` argument.

        Alternatively this can be a dictionary of chemical potentials for a
        single limit, in the format:
        ``{element symbol: chemical potential}``. If manually specifying
        chemical potentials this way, you can set the ``el_refs`` option with
        the DFT reference energies of the elemental phases in order to show the
        formal (relative) chemical potentials above the formation energy plot,
        in which case it is the formal chemical potentials (i.e. relative to
        the elemental references) that should be given here, otherwise the
        absolute (DFT) chemical potentials should be given.

        If ``None`` (default), sets all formal chemical potentials (i.e.
        relative to the elemental reference energies in ``el_refs``) to zero.
        Chemical potentials can also be supplied later in each analysis
        function.
        (Default: None)
        """
        if isinstance(input_chempots, dict) and "elemental_refs" in input_chempots:
            # doped chempot dict input, use its el_refs
            self._chempots, self._el_refs = _parse_chempots(
                input_chempots, input_chempots.get("elemental_refs")
            )

        else:
            self._chempots, self._el_refs = _parse_chempots(
                input_chempots, self._el_refs, update_el_refs=False
            )
        if self.check_compatibility:
            self._check_bulk_chempots_compatibility(self._chempots)

    @property
    def el_refs(self):
        """
        Get the elemental reference energies for the chemical potentials.

        This is in the form of a dictionary of elemental reference energies for
        the chemical potentials, as: ``{element symbol: reference energy}``.
        """
        return self._el_refs

    @el_refs.setter
    def el_refs(self, input_el_refs):
        """
        Set the elemental reference energies for the chemical potentials
        (``el_refs``), and reparse to have the required ``doped`` format.

        This is in the form of a dictionary of elemental reference energies for
        the chemical potentials, in the format:
        ``{element symbol: reference energy}``, and is used to determine the
        formal chemical potentials, when ``chempots`` has been manually
        specified as ``{element symbol: chemical potential}``. Unnecessary if
        ``chempots`` is provided in format generated by ``doped`` (see
        tutorials).
        """
        self._chempots, self._el_refs = _parse_chempots(self._chempots, input_el_refs, update_el_refs=True)

    @property
    def bulk_dos(self):
        """
        Get the ``pymatgen``  ``FermiDos`` for the bulk electronic density of
        states (DOS), for calculating Fermi level positions and defect/carrier
        concentrations, if set.

        Otherwise, returns ``None``.
        """
        return self._bulk_dos

    @bulk_dos.setter
    def bulk_dos(self, input_bulk_dos: FermiDos | Vasprun | PathLike):
        r"""
        Set the ``pymatgen``  ``FermiDos`` for the bulk electronic density of
        states (DOS), for calculating Fermi level positions and defect/carrier
        concentrations.

        Should be a ``pymatgen`` ``FermiDos`` for the bulk electronic DOS, a
        ``pymatgen`` ``Vasprun`` object or path to the  ``vasprun.xml(.gz)``
        output of a bulk DOS calculation in VASP. Can also be provided later
        when using ``get_equilibrium_fermi_level()``,
        ``get_fermi_level_and_concentrations`` etc.

        Usually this is a static calculation with the `primitive` cell of the
        bulk material, with relatively dense `k`-point sampling (especially for
        materials with disperse band edges) to ensure an accurately-converged
        DOS and thus Fermi level. Using large ``NEDOS`` (>3000) and
        ``ISMEAR = -5`` (tetrahedron smearing) are recommended for best
        convergence (wrt `k`-point sampling) in VASP. Consistent functional
        settings should be used for the bulk DOS and defect supercell
        calculations. See
        https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations
        """
        self._bulk_dos = self._parse_fermi_dos(input_bulk_dos, skip_dos_check=self.skip_dos_check)

    @property
    def defect_names(self):
        """
        List of names of defects in the ``DefectThermodynamics`` set.
        """
        return list(self.defect_charge_map.keys())

    @property
    def all_stable_entries(self):
        """
        List all stable entries (defect + charge) in the
        ``DefectThermodynamics`` set.
        """
        return list(chain.from_iterable(self.stable_entries.values()))

    @property
    def all_unstable_entries(self):
        """
        List all unstable entries (defect + charge) in the
        ``DefectThermodynamics`` set.
        """
        return [e for e in self.defect_entries.values() if e not in self.all_stable_entries]

    @property
    def unstable_entries(self):
        """
        Dictionary of unstable entries in the ``DefectThermodynamics`` set, as
        ``{defect name without charge: [list of DefectEntry objects]}``.
        """
        return {
            k: [entry for entry in v if entry not in self.stable_entries[k]]
            for k, v in self.all_entries.items()
        }

    @property
    def dist_tol(self):
        r"""
        Get the distance tolerance (in Å) used for grouping defects together
        (for plotting, transition level analysis and defect concentration
        calculations) based on the distances between their symmetry-equivalent
        sites.

        For the most part, ``DefectEntry``\s of the same type and with
        distances between equivalent defect sites less than ``dist_tol`` (1.5 Å
        by default) are grouped together. If a ``DefectEntry``\'s site has a
        distance less than ``dist_tol`` to multiple sets of equivalent sites,
        then it should be matched to the one with the lowest minimum distance.

        This is used to group together different defect entries (different
        charge states, and/or ground and metastable states (different spin or
        geometries)) which correspond to the same defect type (e.g.
        interstitials at a given site), which is then used in plotting,
        transition level analysis and defect concentration calculations; e.g.
        in the frozen defect approximation, the total concentration of a given
        defect type group is calculated at the annealing temperature, and then
        the equilibrium relative population of the constituent entries is
        recalculated at the quenched temperature.
        """
        return self._dist_tol

    @dist_tol.setter
    def dist_tol(self, input_dist_tol: float):
        r"""
        Get the distance tolerance (in Å) used for grouping defects together
        (for plotting, transition level analysis and defect concentration
        calculations) based on the distances between their symmetry-equivalent
        sites, and reparse the thermodynamic information (transition levels
        etc) with this tolerance.

        For the most part, ``DefectEntry``\s of the same type and with
        distances between equivalent defect sites less than ``dist_tol`` (1.5 Å
        by default) are grouped together. If a ``DefectEntry``\'s site has a
        distance less than ``dist_tol`` to multiple sets of equivalent sites,
        then it should be matched to the one with the lowest minimum distance.

        This is used to group together different defect entries (different
        charge states, and/or ground and metastable states (different spin or
        geometries)) which correspond to the same defect type (e.g.
        interstitials at a given site), which is then used in plotting,
        transition level analysis and defect concentration calculations; e.g.
        in the frozen defect approximation, the total concentration of a given
        defect type group is calculated at the annealing temperature, and then
        the equilibrium relative population of the constituent entries is
        recalculated at the quenched temperature.
        """
        self._dist_tol = input_dist_tol
        self._parse_transition_levels()  # re-group and parse based on new dist_tol

    def get_formation_energies(
        self,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        fermi_level: float | None = None,
        skip_formatting: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        r"""
        Generates defect formation energy tables (DataFrames) for either a
        single chemical potential limit (i.e. phase diagram ``limit``) or each
        limit in the phase diagram (chempots dict), depending on input
        ``limit`` and ``chempots``.

        Table Key: (all energies in eV):

        - 'Defect':
            Defect name (without charge).
        - 'q':
            Defect charge state.
        - 'ΔEʳᵃʷ':
            Raw DFT energy difference between defect and host supercell
            ``(E_defect - E_host)``.
        - 'qE_VBM':
            Defect charge times the VBM eigenvalue (to reference the Fermi
            level to the VBM).
        - 'qE_F':
            Defect charge times the Fermi level (referenced to the VBM if
            qE_VBM is not 0 (if "vbm" in ``DefectEntry.calculation_metadata``).
        - 'Σμ_ref':
            Sum of reference energies of the elemental phases in the chemical
            potentials sum.
        - 'Σμ_formal':
            Sum of `formal` atomic chemical potential terms
            (Σμ_DFT = Σμ_ref + Σμ_formal).
        - 'E_corr':
            Finite-size supercell charge correction.
        - 'Eᶠᵒʳᵐ':
            Defect formation energy, with the specified chemical potentials and
            Fermi level. Equals the sum of all other terms.
        - 'Path':
            Path to the defect calculation folder.
        - 'Δ[E_corr]':
            Estimated error in the charge correction, from the variance of the
            potential in the sampling region.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies. If ``None`` (default), will use
                ``self.chempots``. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to tabulate formation
                energies. Can be either:

                - ``None``, in which case tables are generated for all limits
                  in ``chempots``.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is present/provided in format
                generated by ``doped`` (see tutorials).

                If ``None`` (default), sets all elemental reference energies to
                zero. Note that you can also set
                ``DefectThermodynamics.el_refs = ...`` (with the same input
                options) to set the default elemental reference energies for
                all calculations.
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM eigenvalue, which is taken from the
                ``calculation_metadata`` dict attributes of ``DefectEntry``\s
                in ``self.defect_entries`` if present, otherwise ``self.vbm``
                -- which corresponds to the VBM of the `bulk supercell`
                calculation by default, unless ``bulk_band_gap_vr`` is set
                during defect parsing).
                If ``None`` (default), set to the mid-gap Fermi level (E_g/2).
            skip_formatting (bool):
                Whether to skip formatting the defect charge states as
                strings (and keep as ``int``\s and ``float``\s instead).
                (default: False)

        Returns:
            ``pandas`` ``DataFrame`` or list of ``DataFrame``\s
        """
        fermi_level = self._get_and_set_fermi_level(fermi_level)
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None

        if chempots is None:
            _no_chempots_warning()  # warn only once
            chempots = {  # empty sub-dicts so they're iterable (for following code)
                "limits": {"No User Chemical Potentials": {}},
                "limits_wrt_el_refs": {"No User Chemical Potentials": {}},
            }

        limit = _parse_limit(chempots, limit)
        limits = [limit] if limit is not None else list(chempots["limits"].keys())

        list_of_dfs = []
        with warnings.catch_warnings():  # avoid double warning, already warned above
            warnings.filterwarnings("ignore", "Chemical potentials not present")
            for limit in limits:
                limits_wrt_el_refs = chempots.get("limits_wrt_el_refs") or chempots.get(
                    "limits_wrt_elt_refs"
                )
                if limits_wrt_el_refs is None:
                    raise ValueError("Supplied chempots are not in a recognised format (see docstring)!")
                relative_chempots = limits_wrt_el_refs[limit]
                if el_refs is None:
                    el_refs = (
                        {el: 0 for el in relative_chempots}
                        if chempots.get("elemental_refs") is None
                        else chempots["elemental_refs"]
                    )

                single_formation_energy_df = self._single_formation_energy_table(
                    relative_chempots, el_refs, fermi_level, skip_formatting
                )
                list_of_dfs.append(single_formation_energy_df)

        return list_of_dfs[0] if len(list_of_dfs) == 1 else list_of_dfs

    def _single_formation_energy_table(
        self,
        relative_chempots: dict,
        el_refs: dict,
        fermi_level: float = 0,
        skip_formatting: bool = False,
    ) -> pd.DataFrame:
        r"""
        Returns a defect formation energy table for a single chemical potential
        limit as a pandas ``DataFrame``.

        See ``get_formation_energies()`` for the table key.

        Args:
            relative_chempots (dict):
                Dictionary of formal (i.e. relative to elemental reference
                energies) chemical potentials in the form
                ``{element symbol: energy}``.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}``.
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM eigenvalue, which is taken from the
                ``calculation_metadata`` dict attributes of ``DefectEntry``\s
                in ``self.defect_entries`` if present, otherwise ``self.vbm``
                -- which corresponds to the VBM of the `bulk supercell`
                calculation by default, unless ``bulk_band_gap_vr`` is set
                during defect parsing). If ``None`` (default), set to the
                mid-gap Fermi level (E_g/2).
            skip_formatting (bool):
                Whether to skip formatting the defect charge states as
                strings (and keep as ``int``\s and ``float``\s instead).
                (default: False)

        Returns:
            ``pandas`` ``DataFrame`` sorted by formation energy
        """
        table = []

        for name, defect_entry in self.defect_entries.items():
            row = [
                name.rsplit("_", 1)[0],  # name without charge,
                (
                    defect_entry.charge_state
                    if skip_formatting
                    else f"{'+' if defect_entry.charge_state > 0 else ''}{defect_entry.charge_state}"
                ),
            ]
            row += [defect_entry.get_ediff() - sum(defect_entry.corrections.values())]
            row += [defect_entry.charge_state * defect_entry.calculation_metadata.get("vbm", self.vbm)]
            row += [defect_entry.charge_state * fermi_level]
            row += [defect_entry._get_chempot_term(el_refs) if any(el_refs.values()) else "N/A"]
            row += [defect_entry._get_chempot_term(relative_chempots)]
            row += [sum(defect_entry.corrections.values())]
            dft_chempots = {el: energy + el_refs[el] for el, energy in relative_chempots.items()}
            formation_energy = defect_entry.formation_energy(
                chempots=dft_chempots,
                fermi_level=fermi_level,
                vbm=defect_entry.calculation_metadata.get("vbm", self.vbm),
            )
            row += [formation_energy]
            row += [defect_entry.calculation_metadata.get("defect_path", "N/A")]
            row += [
                sum(
                    [
                        val
                        for key, val in defect_entry.corrections_metadata.items()
                        if "charge_correction_error" in key
                    ]
                )
            ]

            table.append(row)

        formation_energy_df = pd.DataFrame(
            table,
            columns=[
                "Defect",
                "q",
                "ΔEʳᵃʷ",
                "qE_VBM",
                "qE_F",
                "Σμ_ref",
                "Σμ_formal",
                "E_corr",
                "Eᶠᵒʳᵐ",
                "Path",
                "Δ[E_corr]",
            ],
        )

        # round all floats to 3dp:
        formation_energy_df = formation_energy_df.round(3)
        return formation_energy_df.set_index(["Defect", "q"])

    def get_formation_energy(
        self,
        defect_entry: str | DefectEntry,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        fermi_level: float | None = None,
    ) -> float:
        r"""
        Compute the formation energy for a ``DefectEntry`` at a given chemical
        potential limit and fermi_level. ``defect_entry`` can be a string of
        the defect name, of the ``DefectEntry`` object itself.

        Args:
            defect_entry (str or DefectEntry):
                Either a string of the defect entry name (in
                ``DefectThermodynamics.defect_entries``), or a ``DefectEntry``
                object. If the defect name is given without the charge state,
                then the formation energy of the lowest energy (stable) charge
                state at the chosen Fermi level will be given.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energy. If ``None`` (default), will use
                ``self.chempots``. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to
                calculate the formation energy. Can be either:

                - ``None``, default if ``chempots`` corresponds to a single
                  chemical potential limit -- otherwise will use the first
                  chemical potential limit in the ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM (using ``self.vbm``, which is the VBM of
                the `bulk supercell` calculation by default). If ``None``
                (default), set to the mid-gap Fermi level (E_g/2).

        Returns:
            Formation energy value (float)
        """
        fermi_level = self._get_and_set_fermi_level(fermi_level)

        if isinstance(defect_entry, DefectEntry):
            return defect_entry.formation_energy(
                chempots=chempots or self.chempots,
                limit=limit,
                el_refs=el_refs,
                vbm=defect_entry.calculation_metadata.get("vbm", self.vbm),
                fermi_level=fermi_level,
            )

        # otherwise is string:
        possible_defect_names = [
            defect_entry,
        ]
        with contextlib.suppress(ValueError):  # in case formatted/not charge state:
            possible_defect_names.append(
                f"{defect_entry.rsplit('_', 1)[0]}_{int(defect_entry.rsplit('_', 1)[1])}"
            )
            possible_defect_names.append(
                f"{defect_entry.rsplit('_', 1)[0]}_+{int(defect_entry.rsplit('_', 1)[1])}"
            )

        exact_match_defect_entries = [
            entry
            for entry in self.defect_entries.values()
            if any(entry.name == possible_defect_name for possible_defect_name in possible_defect_names)
        ]
        if len(exact_match_defect_entries) == 1:
            return exact_match_defect_entries[0].formation_energy(
                chempots=chempots or self.chempots,
                limit=limit,
                el_refs=el_refs,
                vbm=exact_match_defect_entries[0].calculation_metadata.get("vbm", self.vbm),
                fermi_level=fermi_level,
            )

        if matching_defect_entries := [
            entry
            for entry in self.defect_entries.values()
            if any(possible_defect_name in entry.name for possible_defect_name in possible_defect_names)
        ]:
            return min(
                entry.formation_energy(
                    chempots=chempots or self.chempots,
                    limit=limit,
                    el_refs=el_refs,
                    vbm=entry.calculation_metadata.get("vbm", self.vbm),
                    fermi_level=fermi_level,
                )
                for entry in matching_defect_entries
            )

        raise ValueError(
            f"No matching DefectEntry with {defect_entry} in name found in "
            f"DefectThermodynamics.defect_entries, which have names:\n{list(self.defect_entries.keys())}"
        )

    def get_dopability_limits(
        self, chempots: dict | None = None, limit: str | None = None, el_refs: dict | None = None
    ) -> pd.DataFrame:
        r"""
        Find the dopability limits of the defect system, searching over all
        limits (chemical potential limits) in ``chempots`` and returning the
        most p/n-type conditions, or for a given chemical potential limit (if
        ``limit`` is set or ``chempots`` corresponds to a single chemical
        potential limit; i.e. {element symbol: chemical potential}).

        The dopability limites are defined by the (first) Fermi level positions
        at which defect formation energies become negative as the Fermi level
        moves towards/beyond the band edges, thus determining the maximum
        possible Fermi level range upon doping for this chemical potential
        limit.

        Note that the Fermi level positions are given relative to ``self.vbm``,
        which is the VBM eigenvalue of the bulk supercell calculation by
        default, unless ``bulk_band_gap_vr`` is set during defect parsing.

        This is computed by obtaining the formation energy for every stable
        defect with non-zero charge, and then finding the highest Fermi level
        position at which a donor defect (positive charge) has zero formation
        energy (crosses the x-axis) -- giving the lower dopability limit, and
        the lowest Fermi level position at which an acceptor defect (negative
        charge) has zero formation energy -- giving the upper dopability limit.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus dopability limits). If
                ``None`` (default), will use ``self.chempots``. This can have
                the form of ``{"limits": [{'limit': [chempot_dict]}]}`` (the
                format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical
                potential limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to calculate formation
                energies (and thus dopability limits). Can be either:

                - ``None``, in which case we search over all limits in
                  ``chempots`` and return the most n/p-type conditions, unless
                  ``chempots`` is a single chemical potential limit.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)

        Returns:
            ``pandas`` ``DataFrame`` of dopability limits, with columns:
            "limit", "Compensating Defect", "Dopability Limit" for both
            p/n-type where 'Dopability limit' values are the corresponding
            Fermi level positions in eV, relative to the VBM (``self.vbm``).
        """
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots if chempots is None
        if chempots is None:
            raise ValueError(
                "No chemical potentials supplied or present in "
                "DefectThermodynamics.chempots, so dopability limits cannot be calculated."
            )

        limit = _parse_limit(chempots, limit)
        limits = [limit] if limit is not None else list(chempots["limits"].keys())

        donor_intercepts: list[tuple] = []
        acceptor_intercepts: list[tuple] = []

        for entry in self.all_stable_entries:
            if entry.charge_state > 0:  # donor
                # formation energy is y = mx + c where m = charge_state, c = vbm_formation_energy
                # so x-intercept is -c/m:
                donor_intercepts.extend(
                    (
                        limit,
                        entry.name,
                        -self.get_formation_energy(
                            entry,
                            chempots=chempots,
                            limit=limit,
                            el_refs=el_refs,
                            fermi_level=0,
                        )
                        / entry.charge_state,
                    )
                    for limit in limits
                )
            elif entry.charge_state < 0:  # acceptor
                acceptor_intercepts.extend(
                    (
                        limit,
                        entry.name,
                        -self.get_formation_energy(
                            entry,
                            chempots=chempots,
                            limit=limit,
                            el_refs=el_refs,
                            fermi_level=0,
                        )
                        / entry.charge_state,
                    )
                    for limit in limits
                )

        if not donor_intercepts:
            donor_intercepts = [("N/A", "N/A", -np.inf)]
        if not acceptor_intercepts:
            acceptor_intercepts = [("N/A", "N/A", np.inf)]

        donor_intercepts_df = pd.DataFrame(donor_intercepts, columns=["limit", "name", "intercept"])
        acceptor_intercepts_df = pd.DataFrame(acceptor_intercepts, columns=["limit", "name", "intercept"])

        # get the most p/n-type limit, by getting the limit with the minimum/maximum max/min-intercept,
        # where max/min-intercept is the max/min intercept for that limit (i.e. the compensating intercept)
        idx = (
            donor_intercepts_df.groupby("limit")["intercept"].transform("max")
            == donor_intercepts_df["intercept"]
        )
        limiting_donor_intercept_row = donor_intercepts_df.iloc[
            donor_intercepts_df[idx]["intercept"].idxmin()
        ]
        idx = (
            acceptor_intercepts_df.groupby("limit")["intercept"].transform("min")
            == acceptor_intercepts_df["intercept"]
        )
        limiting_acceptor_intercept_row = acceptor_intercepts_df.iloc[
            acceptor_intercepts_df[idx]["intercept"].idxmax()
        ]

        if limiting_donor_intercept_row["intercept"] > limiting_acceptor_intercept_row["intercept"]:
            warnings.warn(
                "Donor and acceptor doping limits intersect at negative defect formation energies "
                "(unphysical)!"
            )

        try:
            limit_dict = get_rich_poor_limit_dict(chempots)
        except ValueError:
            limit_dict = {}

        return pd.DataFrame(
            [
                [
                    _get_limit_name_from_dict(
                        limiting_donor_intercept_row["limit"], limit_dict, bracket=True
                    ),
                    limiting_donor_intercept_row["name"],
                    round(limiting_donor_intercept_row["intercept"], 3),
                ],
                [
                    _get_limit_name_from_dict(
                        limiting_acceptor_intercept_row["limit"], limit_dict, bracket=True
                    ),
                    limiting_acceptor_intercept_row["name"],
                    round(limiting_acceptor_intercept_row["intercept"], 3),
                ],
            ],
            columns=["limit", "Compensating Defect", "Dopability Limit (eV from VBM/CBM)"],
            index=["p-type", "n-type"],
        )

    def get_doping_windows(
        self, chempots: dict | None = None, limit: str | None = None, el_refs: dict | None = None
    ) -> pd.DataFrame:
        r"""
        Find the doping windows of the defect system, searching over all limits
        (chemical potential limits) in ``chempots`` and returning the most
        p/n-type conditions, or for a given chemical potential limit (if
        ``limit`` is set or ``chempots`` corresponds to a single chemical
        potential limit; i.e. {element symbol: chemical potential}).

        Doping window is defined by the formation energy of the lowest energy
        compensating defect species at the corresponding band edge (i.e. VBM
        for hole doping and CBM for electron doping), as these set the upper
        limit to the formation energy of dopants which could push the Fermi
        level close to the band edge without being negated by defect charge
        compensation.

        Note that the band edge positions are taken from ``self.vbm`` and
        ``self.band_gap``, which are parsed from the `bulk supercell
        calculation` by default, unless ``bulk_band_gap_vr`` is set during
        defect parsing.

        If a dopant has a higher formation energy than the doping window at the
        band edge, then its charge will be compensated by formation of the
        corresponding limiting defect species (rather than free carrier
        populations).

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus doping windows). If
                ``None`` (default), will use ``self.chempots``. This can have
                the form of ``{"limits": [{'limit': [chempot_dict]}]}`` (the
                format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical
                potential limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to calculate formation
                energies (and thus doping windows). Can be either:

                - ``None``, in which case we search over all limits in
                  ``chempots`` and return the most n/p-type conditions, unless
                  ``chempots`` is a single chemical potential limit.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)

        Returns:
            ``pandas`` ``DataFrame`` of doping windows, with columns:
            "limit", "Compensating Defect", "Doping Window" for both p/n-type
            where 'Doping Window' values are the corresponding doping windows
            in eV.
        """
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots if chempots is None
        if chempots is None:
            raise ValueError(
                "No chemical potentials supplied or present in "
                "DefectThermodynamics.chempots, so doping windows cannot be calculated."
            )

        limit = _parse_limit(chempots, limit)
        limits = [limit] if limit is not None else list(chempots["limits"].keys())

        vbm_donor_intercepts: list[tuple] = []
        cbm_acceptor_intercepts: list[tuple] = []

        for entry in self.all_stable_entries:
            if entry.charge_state > 0:  # donor
                vbm_donor_intercepts.extend(
                    (
                        limit,
                        entry.name,
                        self.get_formation_energy(
                            entry,
                            chempots=chempots,
                            limit=limit,
                            el_refs=el_refs,
                            fermi_level=0,
                        ),
                    )
                    for limit in limits
                )
            elif entry.charge_state < 0:  # acceptor
                cbm_acceptor_intercepts.extend(
                    (
                        limit,
                        entry.name,
                        self.get_formation_energy(
                            entry,
                            chempots=chempots,
                            limit=limit,
                            el_refs=el_refs,
                            fermi_level=self.band_gap,  # type: ignore[arg-type]
                        ),
                    )
                    for limit in limits
                )
        if not vbm_donor_intercepts:
            vbm_donor_intercepts = [("N/A", "N/A", np.inf)]
        if not cbm_acceptor_intercepts:
            cbm_acceptor_intercepts = [("N/A", "N/A", -np.inf)]

        vbm_donor_intercepts_df = pd.DataFrame(
            vbm_donor_intercepts, columns=["limit", "name", "intercept"]
        )
        cbm_acceptor_intercepts_df = pd.DataFrame(
            cbm_acceptor_intercepts, columns=["limit", "name", "intercept"]
        )

        try:
            limit_dict = get_rich_poor_limit_dict(chempots)
        except ValueError:
            limit_dict = {}

        # get the most p/n-type limit, by getting the limit with the maximum min-intercept, where
        # min-intercept is the min intercept for that limit (i.e. the compensating intercept)
        limiting_intercept_rows = []
        for intercepts_df in [vbm_donor_intercepts_df, cbm_acceptor_intercepts_df]:
            idx = (
                intercepts_df.groupby("limit")["intercept"].transform("min") == intercepts_df["intercept"]
            )
            limiting_intercept_row = intercepts_df.iloc[intercepts_df[idx]["intercept"].idxmax()]
            limiting_intercept_rows.append(
                [
                    _get_limit_name_from_dict(limiting_intercept_row["limit"], limit_dict, bracket=True),
                    limiting_intercept_row["name"],
                    round(limiting_intercept_row["intercept"], 3),
                ]
            )

        return pd.DataFrame(
            limiting_intercept_rows,
            columns=["limit", "Compensating Defect", "Doping Window (eV at VBM/CBM)"],
            index=["p-type", "n-type"],
        )

    def prune_to_stable_entries(
        self,
        unstable_entries: bool | str = "not shallow",
        shallow_charge_stability_tolerance: float | None = None,
        charge_stability_tolerance: float = 0,
        **kwargs,
    ) -> "DefectThermodynamics":
        """
        This function takes the defect entries in
        ``DefectThermodynamics.defect_entries``, prunes them to only those
        which pass a given stability criterion, and regenerates a new
        ``DefectThermodynamics`` object with these defect entries.

        This function can be used to prune out defect entries which are
        detected to be shallow (perturbed host, 'fake') states according to
        eigenvalue analysis (see
        https://doped.readthedocs.io/en/latest/Tips.html#eigenvalue-electronic-structure-analysis
        for more info), and/or entries which are only stable for certain
        Fermi levels outside or within a small window of the band edges.

        Doesn't modify the original ``DefectThermodynamics`` (``self``) object!

        This function is used internally in ``doped`` with the
        ``unstable_entries`` argument in ``DefectThermodynamics.plot()``,
        but can also be used to prune out shallow/unstable defect entries for
        other purposes (e.g. if one wants to exclude these entries in
        concentration calculations -- though usually these states are
        irrelevant in such calculations due to their low/near-negligible
        stabilities; particularly when reasonable supercell sizes, DFT
        functionals and structure searching are used).

        Args:
            unstable_entries (bool, str):
                Controls the pruning of unstable/shallow defect states; allowed
                values are ``True``, ``False`` or ``"not shallow"``. If
                ``"not shallow"`` (default), defect entries which are predicted
                to be shallow (perturbed host) states according to eigenvalue
                analysis and only stable for Fermi levels within a small window
                to a band edge (``shallow_stability_tol``) are omitted.
                If ``False``, `all` defects which are not stable for any Fermi
                level in the band gap (``charge_stability_tol``) are `also`
                omitted. ``shallow_stability_tol`` and ``charge_stability_tol``
                can be tuned with the ``shallow_charge_stability_tolerance``
                and ``charge_stability_tolerance`` keyword arguments
                respectively. If ``True``, defect entries are not pruned based
                on stability/shallow classification.
            shallow_charge_stability_tolerance (float):
                Tolerance for the Fermi level stability window for defects
                which have been classified as shallow states. If ``None``
                (default), will be set to the smaller of 0.05 eV or 10% of the
                band gap.
            charge_stability_tolerance (float):
                Tolerance for the Fermi level stability window for `all` defect
                charge states, if ``unstable_entries=False``. Default is 0 eV,
                meaning all charge states which are stable for any Fermi level
                in the band gap will be included, but can be set to a positive
                value (meaning only defect charge states which are stable at
                Fermi levels in the band gap `further` than this energy window
                from a band edge) or negative value (which is a less strict
                pruning, only excluding charge states which become stable at
                Fermi levels outside the band gap `further` than the absolute
                value of this energy window from a band edge).
            **kwargs:
                Additional keyword arguments to pass to the
                ``DefectThermodynamics()`` initialisation (via
                ``DefectThermodynamics.from_dict()``).

        Returns:
            New ``DefectThermodynamics`` object with pruned defect entries.
        """
        if unstable_entries is True:  # all
            return self

        # prune to chosen defects
        # determine tolerances:
        shallow_tol = (
            shallow_charge_stability_tolerance
            if shallow_charge_stability_tolerance is not None
            else min(0.05, self.band_gap * 0.1 if self.band_gap is not None else 0.05)
        )
        stability_tol = None if unstable_entries == "not shallow" else charge_stability_tolerance

        pruned_defect_entries = {}

        for name, defect_entry in self.defect_entries.items():
            fermi_stability_window = self._get_in_gap_fermi_level_stability_window(defect_entry)
            if stability_tol is not None and fermi_stability_window < stability_tol:
                continue  # skip

            if defect_entry.is_shallow and fermi_stability_window < shallow_tol:
                continue  # skip

            pruned_defect_entries[name] = defect_entry

        defect_thermo_dict = self.as_dict()
        defect_thermo_dict.update(
            {
                "defect_entries": pruned_defect_entries,
                "check_compatibility": False,
                "skip_dos_check": True,
            }
        )
        defect_thermo_dict.update(kwargs)
        return DefectThermodynamics.from_dict(defect_thermo_dict)

    # TODO: Add option to plot formation energies at the centroid of the chemical stability region? And
    #  make this the default if no chempots are specified? Or better default to plot both the most (
    #  most-electronegative-)anion-rich and the (most-electropositive-)cation-rich chempot limits?
    # TODO: Likewise, add example showing how to plot a metastable state (above the ground state)
    # TODO: Should have similar colours for similar defect types, an option to just show amalgamated
    #  lowest energy charge states for each _defect type_) -- equivalent to setting the dist_tol to
    #  infinity (but should be easier to just do here by taking the short defect name). NaP is an example
    #  for this -- should have a test built for however we want to handle cases like this. See Ke's example
    #  case too with different interstitial sites.
    #   Related: Currently updating `dist_tol` to change the number of defects being plotted,
    #   can also change the colours of the different defect lines (e.g. for CdTe_wout_meta increasing
    #   `dist_tol` to 2 to merge all Te interstitials, results in the colours of other defect lines (
    #   e.g. Cd_Te) changing at the same time -- ideally this wouldn't happen!

    def plot(
        self,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        all_entries: bool | str = False,
        unstable_entries: bool | str = "not shallow",
        chempot_table: bool | None = None,
        style_file: PathLike | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        fermi_level: float | None = None,
        include_site_info: bool = False,
        colormap: str | colors.Colormap | None = None,
        linestyles: str | list[str] = "-",
        auto_labels: bool = False,
        filename: PathLike | None = None,
        **kwargs,
    ) -> Figure | list[Figure]:
        r"""
        Produce a defect formation energy vs Fermi level plot (a.k.a. a defect
        formation energy / transition level diagram), returning the
        ``matplotlib`` ``Figure`` object to allow further plot customisation.

        Note that the band edge positions are taken from ``self.vbm`` and
        ``self.band_gap``, which are parsed from the `bulk supercell
        calculation` by default, unless ``bulk_band_gap_vr`` is set during
        defect parsing.

        Note that different defect entries (different charge states, and/or
        ground and metastable states (different spin or geometries); e.g.
        interstitials at a given site) are grouped together in distinct defect
        types according to ``self.dist_tol``, which is also used in transition
        level analysis and defect concentrations. This can be adjusted as shown
        in the plotting customisation tutorial:
        https://doped.readthedocs.io/en/latest/plotting_customisation_tutorial.html

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies. If ``None`` (default), will use
                ``self.chempots``. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases in order to show the formal (relative)
                chemical potentials above the formation energy plot, in which
                case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the
                absolute (DFT) chemical potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to plot formation
                energies. Can be either:

                - ``None``, in which case plots are generated for all limits in
                  ``chempots``.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            all_entries (bool, str):
                Whether to plot the formation energy lines of `all` defect
                entries, rather than the default of showing only the
                equilibrium states at each Fermi level position (traditional).
                If instead set to ``"faded"``, will plot the equilibrium states
                in bold, and all unstable states in faded grey
                (Default: False)
            unstable_entries (bool, str):
                Controls the plotting of unstable/shallow defect states;
                allowed values are ``True``, ``False`` or ``"not shallow"``.
                If ``"not shallow"`` (default), defect entries which are
                predicted to be shallow (perturbed host) states according to
                eigenvalue analysis and only stable for Fermi levels within a
                small window to a band edge (``shallow_stability_tol``) are
                omitted from plotting. If ``False``, `all` defects which are
                not stable for any Fermi level in the band gap are `also`
                omitted from plotting.
                ``shallow_stability_tol`` is set to the smaller of 0.05 eV or
                10% of the band gap by default, but can be set by a
                ``shallow_charge_stability_tolerance = X`` keyword argument. If
                ``unstable_entries=False``, the Fermi window stability
                tolerance for all defects (default = 0; meaning any in-gap
                stability) can be set by a ``charge_stability_tolerance = X``
                keyword argument (positive or negative).
                If ``True``, defect entries are not pruned based on stability /
                shallow classification.
                See ``prune_to_stable_entries`` for more info.
            chempot_table (Optional[bool]):
                Whether to include a table of the chemical potentials above the
                formation energy plot. If ``None`` (default), shown if multiple
                plots are generated (i.e. multiple chemical potential limits)
                else not shown.
            style_file (PathLike):
                Path to a ``mplstyle`` file to use for the plot. If ``None``
                (default), uses the default doped style (from
                ``doped/utils/doped.mplstyle``).
            xlim:
                Tuple (min,max) giving the range of the x-axis (Fermi level).
                May want to set manually when including transition level
                labels, to avoid crossing the axes.
                Default is to plot from -0.3 to +0.3 eV above the band gap.
            ylim:
                Tuple (min,max) giving the range for the y-axis (formation
                energy). May want to set manually when including transition
                level labels, to avoid crossing the axes. Default is from 0 to
                just above the maximum formation energy value in the band gap.
            fermi_level (float):
                If set, plots a dashed vertical line at this Fermi level value,
                typically used to indicate the equilibrium Fermi level position
                if known/calculated (e.g. with
                ``get_fermi_level_and_concentrations``). (Default: None)
            include_site_info (bool):
                Whether to include site info in defect names in the plot legend
                (e.g. ``$Cd_{i_{C3v}}^{0}$`` rather than ``$Cd_{i}^{0}$``).
                Default is ``False``, where site info is not included unless we
                have inequivalent sites for the same defect type. If, even with
                site info added, there are duplicate defect names, then
                "-a", "-b", "-c"... are appended to the names to differentiate.
            colormap (str, matplotlib.colors.Colormap):
                Colormap to use for the formation energy lines, either as a
                string (which can be a colormap name from
                https://matplotlib.org/stable/users/explain/colors/colormaps or
                from https://www.fabiocrameri.ch/colourmaps -- append 'S' if
                using a sequential colormap from the latter) or a ``Colormap``
                / ``ListedColormap`` object.
                If ``None`` (default), uses ``tab10`` with ``alpha=0.75`` (if
                10 or fewer lines to plot), ``tab20`` (if 20 or fewer lines) or
                ``batlow`` (if more than 20 lines; citation:
                https://zenodo.org/records/8409685).
            linestyles (list):
                Linestyles to use for the formation energy lines, either as a
                single linestyle (``str``) or list of linestyles
                (``list[str]``) in the order of appearance of lines in the plot
                legend. Default is ``"-"``; i.e. solid lines for all entries.
            auto_labels (bool):
                Whether to automatically label the transition levels with their
                charge states. If there are many transition levels, this can be
                quite ugly. (Default: False)
            filename (PathLike): Filename to save the plot to.
            (Default: None (not saved))
            **kwargs:
                Additional keyword arguments for advanced customisation, such
                as ``shallow_charge_stability_tolerance`` or
                ``charge_stability_tolerance`` for controlling stability window
                tolerances with the ``unstable_entries`` parameter (see
                argument description for more info).

        Returns:
            ``matplotlib`` ``Figure`` object, or list of ``Figure`` objects if
            multiple limits chosen.
        """
        from shakenbreak.plotting import _install_custom_font

        _install_custom_font()
        if all_entries not in [False, True, "faded"]:  # check input options
            raise ValueError(
                f"`all_entries` option must be either False, True, or 'faded', not {all_entries}"
            )

        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None
        if chempots is None:
            chempots = {
                "limits": {"No User Chemical Potentials": None}
            }  # empty chempots dict to allow plotting, user will be warned

        limit = _parse_limit(chempots, limit)
        limits = [limit] if limit is not None else list(chempots["limits"].keys())

        if (
            chempots
            and limit is None
            and el_refs is None
            and "limits" not in chempots
            and any(np.isclose(chempot, 0, atol=0.1) for chempot in chempots.values())
        ):
            # if any chempot is close to zero, this is likely a formal chemical potential and so inaccurate
            # here (trying to make this as idiotproof as possible to reduce unnecessary user queries...)
            warnings.warn(
                "At least one of your manually-specified chemical potentials is close to zero, "
                "which is likely a _formal_ chemical potential (i.e. relative to the elemental "
                "reference energies), but you have not specified the elemental reference "
                "energies with `el_refs`. This will give large errors in the absolute values "
                "of formation energies, but the transition level positions will be unaffected."
            )

        if unstable_entries not in [False, True, "not shallow"]:  # check unstable_entries input options
            raise ValueError(
                f"`unstable_entries` option must be either True, False, 'not shallow', "
                f"not {unstable_entries}. See DefectThermodynamics.plot docstring for more info."
            )

        # unstable_entries pruning:
        thermo_to_plot = self.prune_to_stable_entries(
            unstable_entries=unstable_entries, **kwargs
        )  # Note that this will need to be updated if we add other kwarg options to this function

        style_file = style_file or f"{os.path.dirname(__file__)}/utils/doped.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        with plt.style.context(style_file):
            figs = []
            for limit in limits:
                dft_chempots = chempots["limits"][limit]
                plot_title = limit if len(limits) > 1 else None
                plot_filename = (
                    f"{filename.rsplit('.', 1)[0]}_{limit}.{filename.rsplit('.', 1)[1]}"
                    if filename
                    else None
                )

                with warnings.catch_warnings():  # avoid double warning about no chempots supplied
                    warnings.filterwarnings("ignore", "No chemical potentials")
                    fig = formation_energy_plot(
                        thermo_to_plot,
                        dft_chempots=dft_chempots,
                        el_refs=el_refs,
                        chempot_table=chempot_table if chempot_table is not None else len(limits) > 1,
                        all_entries=all_entries,
                        xlim=xlim,
                        ylim=ylim,
                        fermi_level=fermi_level,
                        include_site_info=include_site_info,
                        title=plot_title,
                        colormap=colormap,
                        linestyles=linestyles,
                        auto_labels=auto_labels,
                        filename=plot_filename,
                    )
                figs.append(fig)

            return figs[0] if len(figs) == 1 else figs

    def get_transition_levels(
        self,
        all: bool = False,
        format_charges: bool = True,
    ) -> pd.DataFrame | None:
        """
        Return a ``DataFrame`` of the charge transition levels for the defects
        in the ``DefectThermodynamics`` object (stored in the
        ``transition_level_map`` attribute).

        Note that the transition level (and Fermi level) positions are given
        relative to ``self.vbm``, which is the VBM eigenvalue of the bulk
        supercell calculation by default, unless ``bulk_band_gap_vr`` is set
        during defect parsing.

        By default, only returns the thermodynamic ground-state transition
        levels (i.e. those visible on the defect formation energy diagram), not
        including metastable defect states -- which can be important for
        recombination, migration, degeneracy/concentrations etc., see e.g.
        https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E.
        e.g. negative-U defects will show the 2-electron transition level
        (N+1/N-1) rather than (N+1/N) and (N/N-1).
        If instead all single-electron transition levels are desired, set
        ``all = True``.

        Returns a ``DataFrame`` with columns:

        - "Defect": Defect name.
        - "Charges":
            Defect charge states which make up the transition level
            (as a string if ``format_charges=True``, otherwise as a list of
            integers).
        - "eV from VBM":
            Transition level position in eV from the VBM (``self.vbm``).
        - "In Band Gap?":
            Whether the transition level is within the host band gap.
        - "N(Metastable)":
            Number of metastable states involved in the transition level
            (0, 1 or 2). Only included if ``all = True``.

        Args:
              all (bool):
                    Whether to print all single-electron transition levels
                    (i.e. including metastable defect states), or just the
                    thermodynamic ground-state transition levels (default).
              format_charges (bool):
                    Whether to format the transition level charge states as
                    strings (e.g. ``"ε(+1/+2)"``) or keep in list format (e.g.
                    ``[1,2]``). Default is ``True``.
        """
        # create a dataframe from the transition level map, with defect name, transition level charges and
        # TL position in eV from the VBM:
        transition_level_map_list = []

        def _TL_naming_func(TL_charges, i_meta=False, j_meta=False):
            if not format_charges:
                return TL_charges
            i, j = TL_charges
            return (
                f"ε({'+' if i > 0 else ''}{i}{'*' if i_meta else ''}/"
                f"{'+' if j > 0 else ''}{j}{'*' if j_meta else ''})"
            )

        for defect_name, transition_level_dict in self.transition_level_map.items():
            if not transition_level_dict:
                transition_level_map_list.append(  # add defects with no TL to dataframe as "None"
                    {
                        "Defect": defect_name,
                        "Charges": "None",
                        "eV from VBM": np.inf,
                        "In Band Gap?": False,
                        "-q_i": np.inf,  # for sorting
                        "-q_j": np.inf,  # for sorting
                    }
                )
                if all:
                    transition_level_map_list[-1]["N(Metastable)"] = 0

            if not all:
                transition_level_map_list.extend(
                    {
                        "Defect": defect_name,
                        "Charges": _TL_naming_func(transition_level_charges),
                        "eV from VBM": round(TL, 3),
                        "In Band Gap?": (TL > 0) and (self.band_gap > TL),
                        "-q_i": -transition_level_charges[0],  # for sorting
                        "-q_j": -transition_level_charges[1],  # for sorting
                    }
                    for TL, transition_level_charges in transition_level_dict.items()
                )

        # now get metastable TLs
        if all:
            for defect_name_wout_charge, grouped_defect_entries in self.all_entries.items():
                sorted_defect_entries = sorted(
                    grouped_defect_entries, key=lambda x: x.charge_state
                )  # sort by charge
                for i, j in product(sorted_defect_entries, repeat=2):
                    if i.charge_state - j.charge_state == 1:
                        # take mean VBM, ofc should be the same, but allow for small differences
                        mean_VBM = np.mean([x.calculation_metadata.get("vbm", self.vbm) for x in [i, j]])
                        TL = j.get_ediff() - i.get_ediff() - mean_VBM
                        i_meta = not any(i == y for y in self.all_stable_entries)
                        j_meta = not any(j == y for y in self.all_stable_entries)
                        transition_level_map_list.append(
                            {
                                "Defect": defect_name_wout_charge,
                                "Charges": _TL_naming_func(
                                    [i.charge_state, j.charge_state], i_meta=i_meta, j_meta=j_meta
                                ),
                                "eV from VBM": round(TL, 3),
                                "In Band Gap?": (TL > 0) and (self.band_gap > TL),
                                "N(Metastable)": [i_meta, j_meta].count(True),
                                "-q_i": -i.charge_state,  # for sorting
                                "-q_j": -j.charge_state,  # for sorting
                            }
                        )

        if not transition_level_map_list:
            warnings.warn("No transition levels found for chosen parameters!")
            return None
        tl_df = pd.DataFrame(transition_level_map_list)

        if "N(Metastable)" not in tl_df.columns:  # add, because we use it for sorting in case all=True
            tl_df["N(Metastable)"] = 0

        # sort df by Defect appearance order in defect_entries, Defect, then by TL position:
        tl_df["Defect Appearance Order"] = tl_df["Defect"].map(self._map_sort_func)
        tl_df = tl_df.sort_values(
            by=["Defect Appearance Order", "Defect", "-q_i", "-q_j", "N(Metastable)", "eV from VBM"]
        )
        tl_df = tl_df.drop(columns=["Defect Appearance Order", "-q_i", "-q_j"])
        if not all:
            tl_df = tl_df.drop(columns="N(Metastable)")
        return tl_df.set_index(["Defect", "Charges"])

    def print_transition_levels(self, all: bool = False):
        """
        Iteratively prints the charge transition levels for the defects in the
        ``DefectThermodynamics`` object (stored in the ``transition_level_map``
        attribute).

        By default, only returns the thermodynamic ground-state transition
        levels (i.e. those visible on the defect formation energy diagram), not
        including metastable defect states -- which can be important for
        recombination, migration, degeneracy/concentrations etc., see e.g.
        https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E.
        e.g. negative-U defects will show the 2-electron transition level
        (N+1/N-1) rather than (N+1/N) and (N/N-1).
        If instead all single-electron transition levels are desired, set
        ``all = True``.

        Args:
              all (bool):
                    Whether to print all single-electron transition levels
                    (i.e. including metastable defect states), or just the
                    thermodynamic ground-state transition levels (default).
        """
        if not all:
            for defect_name, tl_info in self.transition_level_map.items():
                bold_print(f"Defect: {defect_name}")
                for tl_efermi, chargeset in tl_info.items():
                    print(
                        f"Transition level ε({max(chargeset):{'+' if max(chargeset) else ''}}/"
                        f"{min(chargeset):{'+' if min(chargeset) else ''}}) at {tl_efermi:.3f} eV above "
                        f"the VBM"
                    )
                print("")  # add space

        else:
            all_TLs_df = self.get_transition_levels(all=True)
            if all_TLs_df is None:
                return
            for defect_name, tl_df in all_TLs_df.groupby("Defect", sort=False):
                bold_print(f"Defect: {defect_name}")
                for index, row in tl_df.iterrows():
                    if index[1] != "None":  # charges
                        print(f"Transition level {index[1]} at {row['eV from VBM']:.3f} eV above the VBM")
                print("")  # add space

    def get_symmetries_and_degeneracies(
        self,
        skip_formatting: bool = False,
        symprec: float | None = None,
    ) -> pd.DataFrame:
        r"""
        Generates a table of the bulk-site & relaxed defect point group
        symmetries, spin/orientational/total degeneracies and (bulk-)site
        multiplicities for each defect in the ``DefectThermodynamics`` object.

        Table Key:

        - 'Defect': Defect name (without charge)
        - 'q': Defect charge state.
        - 'Site_Symm': Point group symmetry of the defect site in the bulk cell.
        - 'Defect_Symm': Point group symmetry of the relaxed defect.
        - 'g_Orient': Orientational degeneracy of the defect.
        - 'g_Spin': Spin degeneracy of the defect.
        - 'g_Total': Total degeneracy of the defect.
        - 'Mult': Multiplicity of the defect site in the bulk cell, per primitive unit cell.

        For interstitials, the bulk site symmetry corresponds to the point
        symmetry of the interstitial site with `no relaxation of the host
        structure`, while for vacancies/substitutions it is simply the symmetry
        of their corresponding bulk site. This corresponds to the point
        symmetry of ``DefectEntry.defect``, or
        ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``.

        Point group symmetries are taken from the calculation_metadata
        ("relaxed point symmetry" and "bulk site symmetry") if present (should
        be, if parsed with doped and defect supercell doesn't break host
        periodicity), otherwise are attempted to be recalculated.

        Note: ``doped`` tries to use the ``defect_entry.defect_supercell`` to
        determine the `relaxed` site symmetry. However, it should be noted that
        this is not guaranteed to work in all cases; namely for non-diagonal
        supercell expansions, or sometimes for non-scalar supercell expansion
        matrices (e.g. a 2x1x2 expansion)(particularly with high-symmetry
        materials) which can mess up the periodicity of the cell. doped tries
        to automatically check if this is the case, and will warn you if so.

        This can also be checked by using this function on your doped
        `generated` defects:

        .. code-block:: python

            from doped.generation import get_defect_name_from_entry
            for defect_name, defect_entry in defect_gen.items():
                print(defect_name,
                      get_defect_name_from_entry(defect_entry, relaxed=False),
                      get_defect_name_from_entry(defect_entry), "\n")

        And if the point symmetries match in each case, then doped should be
        able to correctly determine the final relaxed defect symmetry (and
        orientational degeneracy) -- otherwise periodicity-breaking prevents
        this.

        If periodicity-breaking prevents auto-symmetry determination, you can
        manually determine the relaxed defect and bulk-site point symmetries,
        and/or orientational degeneracy, from visualising the structures (e.g.
        using VESTA)(can use ``get_orientational_degeneracy`` to obtain the
        corresponding orientational degeneracy factor for given defect/bulk
        site point symmetries) and setting the corresponding values in the
        ``'relaxed point symmetry'`` / ``'bulk site symmetry'`` entries in
        ``DefectEntry.calculation_metadata`` and/or
        ``DefectEntry.degeneracy_factors['orientational degeneracy']``
        attributes.
        Note that the bulk-site point symmetry corresponds to that of
        ``DefectEntry.defect``, or equivalently
        ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``,
        which for vacancies/substitutions is the symmetry of the corresponding
        bulk site, while for interstitials it is the point symmetry of the
        `final relaxed` interstitial site when placed in the (unrelaxed) bulk
        structure. The degeneracy factor is used in the calculation of
        defect/carrier concentrations and Fermi level behaviour (see e.g.
        https://doi.org/10.1039/D2FD00043A,
        https://doi.org/10.1039/D3CS00432E).

        Args:
            skip_formatting (bool):
                Whether to skip formatting the defect charge states as
                strings (and keep as ``int``\s and ``float``\s instead).
                (default: False)
            symprec (float):
                Symmetry tolerance for ``spglib`` to use when determining
                relaxed defect point symmetries and thus orientational
                degeneracies. Default is ``0.1`` which matches that used by
                the ``Materials Project`` and is larger than the ``pymatgen``
                default of ``0.01`` (which is used by ``doped`` for
                unrelaxed/bulk structures) to account for residual structural
                noise in relaxed defect supercells.
                You may want to adjust for your system (e.g. if there are
                very slight octahedral distortions etc.). If ``symprec`` is
                set, then the point symmetries and corresponding orientational
                degeneracy will be re-parsed/computed even if already present
                in the ``DefectEntry`` object ``calculation_metadata``.

        Returns:
            ``pandas`` ``DataFrame``
        """
        table_list = []

        for name, defect_entry in self.defect_entries.items():
            defect_entry._parse_and_set_degeneracies(symprec=symprec)
            try:
                multiplicity_per_unit_cell = defect_entry.defect.multiplicity * (
                    len(get_primitive_structure(defect_entry.defect.structure))  # spglib primitive
                    / len(defect_entry.defect.structure)
                )  # ensure multiplicity corresponds to unit cell (which it should by default anyway,
                # now that parsed ``Defect``s are defined in the primitive unit cell)

            except Exception:
                multiplicity_per_unit_cell = "N/A"

            total_degeneracy = (
                reduce(lambda x, y: x * y, defect_entry.degeneracy_factors.values())
                if defect_entry.degeneracy_factors
                else "N/A"
            )

            table_list.append(
                {
                    "Defect": name.rsplit("_", 1)[0],  # name without charge
                    "q": defect_entry.charge_state,
                    "Site_Symm": defect_entry.calculation_metadata.get("bulk site symmetry", "N/A"),
                    "Defect_Symm": defect_entry.calculation_metadata.get("relaxed point symmetry", "N/A"),
                    "g_Orient": defect_entry.degeneracy_factors.get("orientational degeneracy", "N/A"),
                    "g_Spin": defect_entry.degeneracy_factors.get("spin degeneracy", "N/A"),
                    "g_Total": total_degeneracy,
                    "Mult": multiplicity_per_unit_cell,
                }
            )

        if any(
            defect_entry.calculation_metadata.get("periodicity_breaking_supercell", False)
            for defect_entry in self.defect_entries.values()
        ):
            warnings.warn(_orientational_degeneracy_warning)

        symmetry_df = pd.DataFrame(table_list)

        if not skip_formatting:
            symmetry_df["q"] = symmetry_df["q"].apply(lambda x: f"{'+' if x > 0 else ''}{x}")

        return symmetry_df.set_index(["Defect", "q"])

    def _get_and_set_fermi_level(self, fermi_level: float | None = None) -> float:
        """
        Handle the input Fermi level choice.

        If Fermi level not set, defaults to mid-gap Fermi level (E_g/2) and
        prints an info message to the user.
        """
        if fermi_level is None:
            fermi_level = 0.5 * self.band_gap  # type: ignore
            print(
                f"Fermi level was not set, so using mid-gap Fermi level (E_g/2 = {fermi_level:.2f} eV "
                f"relative to the VBM)."
            )
        return fermi_level

    def _sanitise_chempots_for_concentrations(self, chempots, el_refs, limit):
        limit = _check_chempots_and_limit_settings(chempots, limit)  # only warn once
        if chempots is None:
            all_comps = [
                entry.sc_entry.composition if entry.sc_entry else entry.bulk_entry.composition
                for entry in self.defect_entries.values()
            ]
            empty_el_dict = {el: 0 for el in {el.symbol for comp in all_comps for el in comp}}
            chempots = {
                "limits": {"No User Chemical Potentials": empty_el_dict},
                "limits_wrt_el_refs": {"No User Chemical Potentials": empty_el_dict},
                "elemental_refs": el_refs or empty_el_dict,
            }
            limit = "No User Chemical Potentials"

        return chempots, limit

    def get_equilibrium_concentrations(
        self,
        temperature: float = 300,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        fermi_level: float | None = None,
        per_charge: bool = True,
        per_site: bool = False,
        skip_formatting: bool = False,
        site_competition: bool | str = True,
        lean: bool = False,
    ) -> pd.DataFrame:
        r"""
        Compute the `equilibrium` concentrations (in cm^-3) for all
        ``DefectEntry``\s in the ``DefectThermodynamics`` object, at a given
        chemical potential limit, Fermi level and temperature, assuming the
        dilute limit approximation.

        Note that these are the `equilibrium` defect concentrations!
        ``DefectThermodynamics.get_fermi_level_and_concentrations()`` can
        instead be used to calculate the Fermi level and defect concentrations
        for a material grown/annealed at higher temperatures and then cooled
        (quenched) to room/operating temperature (where defect concentrations
        are assumed to remain fixed) -- this is known as the frozen defect
        approach and is typically the most valid approximation (see its
        docstring for more information).

        The degeneracy/multiplicity factor "g" is an important parameter in the
        defect concentration equation, affecting the final concentration by up
        to 2 orders of magnitude. This factor is taken from the product of the
        ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes. See discussion in:
        https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E.

        Note that the ``FermiSolver`` class implements a number of convenience
        methods for thermodynamic analyses; such as scanning over temperatures,
        chemical potentials, effective dopant concentrations etc., minimising
        or maximising a target property (e.g. defect/carrier concentration),
        and also allowing greater control over constraints and approximations
        in defect concentration calculations (i.e. fixed/variable defect(s) and
        charge states), which may be useful. See the ``FermiSolver`` tutorial:
        https://doped.readthedocs.io/en/latest/fermisolver_tutorial.html

        Args:
            temperature (float):
                Temperature in Kelvin at which to calculate the equilibrium concentrations.
                Default is 300 K.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus concentrations). If
                ``None`` (default), will use ``self.chempots`` (= 0 for all
                chemical potentials by default).. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to calculate formation
                energies (and thus concentrations). Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical
                  potential limit -- otherwise will use the first chemical
                  potential limit in the ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM (using ``self.vbm``, which is the VBM of
                the `bulk supercell` calculation by default). If ``None``
                (default), set to the mid-gap Fermi level (E_g/2).
            per_charge (bool):
                Whether to break down the defect concentrations into individual
                defect charge states (e.g. ``v_Cd_0``, ``v_Cd_-1``, ``v_Cd_-2``
                instead of ``v_Cd``). Default is ``True``.
            per_site (bool):
                Whether to also return the concentrations as per-site
                concentrations in percent (i.e. concentration divided by
                ``DefectEntry.bulk_site_concentration``). (default: False)
            skip_formatting (bool):
                Whether to skip formatting the defect charge states and
                concentrations as strings (and keep as ``int``\s and
                ``float``\s instead). (default: False)
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                If ``False``, uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            lean (bool):
                Whether to return a leaner ``DataFrame`` with `only` the defect
                name, charge state, and concentration in cm^-3 (assumes
                ``skip_formatting=True`` and ``per_site=False``). Mostly
                intended for internal ``doped`` usage, to reduce compute times
                when calculating defect concentrations repeatedly.
                (default: False)

        Returns:
            ``pandas`` ``DataFrame`` of defect concentrations (and formation
            energies) for each ``DefectEntry`` in the ``DefectThermodynamics``
            object.
        """
        fermi_level = self._get_and_set_fermi_level(fermi_level)
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None
        skip_formatting = skip_formatting or lean
        per_site = per_site and not lean

        energy_concentration_list = []

        chempots, limit = self._sanitise_chempots_for_concentrations(
            chempots, el_refs, limit
        )  # warns about chempots/limit choices if necessary

        # Note: DataFrame initialisation from the list of dicts here actually ends up contributing a
        # non-negligible compute cost (~10%), which could be made faster by using a dict of lists/arrays
        # which is possible, but would make the code much less readable (e.g. for implementing site
        # competition rescaling). Could be revisited if needed
        for defect_name_wout_charge, defect_entry_list in self.all_entries.items():
            for defect_entry in defect_entry_list:
                formation_energy = defect_entry.formation_energy(
                    chempots=chempots,
                    limit=limit,
                    el_refs=el_refs,
                    fermi_level=fermi_level,
                    vbm=defect_entry.calculation_metadata.get("vbm", self.vbm),
                )
                raw_concentration = defect_entry.equilibrium_concentration(
                    chempots=chempots,
                    limit=limit,
                    el_refs=el_refs,
                    fermi_level=fermi_level,
                    vbm=defect_entry.calculation_metadata.get("vbm", self.vbm),
                    temperature=temperature,
                    per_site=False,  # only concentration in cm^-3 here
                    formation_energy=formation_energy,  # reduce compute times
                    site_competition=False,  # only rescale after, if site_competition = True
                )
                per_site_concentration = (
                    raw_concentration / defect_entry.bulk_site_concentration
                    if (site_competition or per_site)
                    else None
                )  # only calculate if needed

                # this could be refactored if DefectEntry finding became a bottleneck (currently not)
                cluster_number = next(
                    k for k, v in self._clustered_defect_entries.items() if defect_entry in v
                )

                charge = (
                    defect_entry.charge_state
                    if skip_formatting
                    else f"{'+' if defect_entry.charge_state > 0 else ''}{defect_entry.charge_state}"
                )
                energy_concentration_list.append(
                    {
                        "Defect": defect_name_wout_charge,
                        "Charge": charge,
                        "Concentration (cm^-3)": raw_concentration,
                        "Formation Energy (eV)": round(formation_energy, 3),
                        "Concentration (per site)": per_site_concentration,
                        "Lattice Site Index": cluster_number,
                    }
                )

        if site_competition:
            with np.errstate(divide="ignore", invalid="ignore"):
                for cluster_number in self._clustered_defect_entries:
                    matching_concentration_dicts = [
                        concentration_dict
                        for concentration_dict in energy_concentration_list
                        if concentration_dict["Lattice Site Index"] == cluster_number
                    ]
                    summed_per_site_concentration = sum(
                        concentration_dict["Concentration (per site)"]
                        for concentration_dict in matching_concentration_dicts
                    )
                    for concentration_dict in matching_concentration_dicts:
                        concentration_dict["Concentration (per site)"] /= 1 + summed_per_site_concentration
                        concentration_dict["Concentration (cm^-3)"] /= 1 + summed_per_site_concentration

        for concentration_dict in energy_concentration_list:
            if not per_site:
                concentration_dict.pop("Concentration (per site)")

            if lean:  # pop formation energy (& per-site conc, site idx) for pd.DataFrame init speed
                concentration_dict.pop("Formation Energy (eV)")

            if lean or not isinstance(site_competition, str):
                concentration_dict.pop("Lattice Site Index")

        conc_df = pd.DataFrame(energy_concentration_list)
        # Note that in concentration / FermiSolver functions, we avoid altering the output ordering and
        # try to just use the DefectThermodynamics entry ordering (which is already controlled) as is

        conc_columns = [col for col in conc_df.columns if "Concentration" in col]
        if per_charge:
            if lean:
                return conc_df  # Defect/Charge not set as index w/lean=True & per_charge=False, for speed

            conc_col = next(iter(conc_columns))  # either conc col can be used, this is cm^-3 as it's 1st
            conc_df["Charge State Population"] = conc_df[conc_col] / conc_df.groupby("Defect")[
                conc_col
            ].transform("sum")
            conc_df["Charge State Population"] = conc_df["Charge State Population"].apply(
                lambda x: f"{x:.2%}"
            )
            conc_df = conc_df.set_index(["Defect", "Charge"])

        else:  # group by defect and sum concentrations:
            conc_df = _group_defect_charge_state_concentrations(conc_df)

        for conc_col in conc_columns:
            conc_df[conc_col] = conc_df[conc_col].apply(
                lambda x, conc_col=conc_col: _format_concentration(
                    x,
                    per_site=("per site" in conc_col),
                    skip_formatting=skip_formatting,
                )
            )

        return conc_df

    def _parse_fermi_dos(
        self, bulk_dos: PathLike | Vasprun | FermiDos | None = None, skip_dos_check: bool = False
    ) -> FermiDos | None:
        if bulk_dos is None:
            return None

        fdos = None

        if isinstance(bulk_dos, FermiDos):
            fdos = bulk_dos
            # most similar settings to Vasprun.eigenvalue_band_properties:
            fdos_vbm = fdos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1]  # tol 1e-4 is lowest possible, as VASP
            fdos_band_gap = fdos.get_gap(tol=1e-4, abs_tol=True)  # rounds the DOS outputs to 4 dp

        if isinstance(bulk_dos, PathLike):
            bulk_dos = get_vasprun(bulk_dos, parse_dos=True)  # converted to fdos in next block

        if isinstance(bulk_dos, Vasprun):  # either supplied Vasprun or parsed from string there
            fdos_band_gap, _cbm, fdos_vbm, _ = bulk_dos.eigenvalue_band_properties
            fdos = get_fermi_dos(bulk_dos)

        if (
            fdos
            and not skip_dos_check
            and (abs(fdos_vbm - self.vbm) > 0.05 or abs(fdos_band_gap - self.band_gap) > 0.05)
        ):
            mismatching = "band gap" if abs(fdos_band_gap - self.band_gap) > 0.05 else "VBM eigenvalue"
            warnings.warn(
                f"The {mismatching} of the bulk DOS calculation ({fdos_vbm:.2f} eV, band gap = "
                f"{fdos_band_gap:.2f} eV) differs by >0.05 eV from `DefectThermodynamics.vbm/gap` "
                f"({self.vbm:.2f} eV, band gap = {self.band_gap:.2f} eV; which are taken from the bulk "
                f"supercell calculation by default, unless `bulk_band_gap_vr` is set during defect "
                f"parsing). This can cause inaccuracies in thermodynamics & concentration analyses; see "
                f"https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations "
                f"for advice.\n"
                f"Note that the Fermi level will be always referenced to `DefectThermodynamics.vbm`!"
            )
        return fdos

    def get_equilibrium_fermi_level(
        self,
        bulk_dos: FermiDos | Vasprun | PathLike | None = None,
        temperature: float = 300,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        return_concs: bool = False,
        skip_dos_check: bool = False,
        effective_dopant_concentration: float | None = None,
        site_competition: bool = True,
    ) -> float | tuple[float, float, float]:
        r"""
        Calculate the self-consistent Fermi level, at a given chemical
        potential limit and temperature, assuming `equilibrium` defect
        concentrations (i.e. under annealing) and the dilute limit
        approximation, by self-consistently solving for the Fermi level which
        yields charge neutrality.

        Note that the returned Fermi level is given relative to ``self.vbm``,
        which is the VBM eigenvalue of the bulk supercell calculation by
        default, unless ``bulk_band_gap_vr`` is set during defect parsing.

        Note that this assumes `equilibrium` defect concentrations!
        ``DefectThermodynamics.get_fermi_level_and_concentrations()`` can
        instead be used to calculate the Fermi level and defect concentrations
        for a material grown/annealed at higher temperatures and then cooled
        (quenched) to room/operating temperature (where defect concentrations
        are assumed to remain fixed) -- this is known as the frozen defect
        approach and is typically the most valid approximation (see its
        docstring for more information).

        Note that the bulk DOS calculation should be well-converged with
        respect to `k`-points for accurate Fermi level predictions!

        The degeneracy/multiplicity factor "g" is an important parameter in the
        defect concentration equation, affecting the final concentration by up
        to 2 orders of magnitude. This factor is taken from the product of the
        ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes. See discussion in:
        https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E.

        Note that the ``FermiSolver`` class implements a number of convenience
        methods for thermodynamic analyses; such as scanning over temperatures,
        chemical potentials, effective dopant concentrations etc, minimising or
        maximising a target property (e.g. defect/carrier concentration), and
        also allowing greater control over constraints and approximations in
        defect concentration calculations (i.e. fixed/variable defect(s) and
        charge states), which may be useful. See the ``FermiSolver`` tutorial:
        https://doped.readthedocs.io/en/latest/fermisolver_tutorial.html

        Args:
            bulk_dos (FermiDos or Vasprun or PathLike):
                ``pymatgen`` ``FermiDos`` for the bulk electronic density of
                states (DOS), for calculating carrier concentrations.
                Alternatively, can be a ``pymatgen`` ``Vasprun`` object or path
                to the ``vasprun.xml(.gz)`` output of a bulk DOS calculation in
                VASP -- however this will be much slower when looping over many
                conditions as it will re-parse the DOS each time! (So
                preferably set ``DefectThermodynamics.bulk_docs = ...`` or upon
                initialisation: ``DefectThermodynamics(..., bulk_dos=...)``.

                Usually this is a static calculation with the `primitive` cell
                of the bulk material, with relatively dense `k`-point sampling
                (especially for materials with disperse band edges) to ensure
                an accurately-converged DOS and thus Fermi level. Using large
                ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing)
                are recommended for best convergence (wrt `k`-point sampling)
                in VASP. Consistent functional settings should be used for the
                bulk DOS and defect supercell calculations. See
                https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations

                ``bulk_dos`` can also be left as ``None`` (default), if it has
                previously been provided and parsed, and thus is set as the
                ``self.bulk_dos`` attribute. If provided, will overwrite the
                ``self.bulk_dos`` attribute!
            temperature (float):
                Temperature in Kelvin at which to calculate the equilibrium
                Fermi level. Default is 300 K.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus concentrations and Fermi
                level). If ``None`` (default), will use ``self.chempots``
                (= 0 for all chemical potentials by default). This can have the
                form of ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to determine the
                equilibrium Fermi level. Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical
                  potential limit -- otherwise will use the first chemical
                  potential limit in the ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            return_concs (bool):
                Whether to return the corresponding electron and hole
                concentrations (in cm^-3) as well as the Fermi level.
                Default is ``False``.
            skip_dos_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                ``self.vbm`` by >0.05 eV. Should only be used when the reason
                for this difference is known/acceptable. (default: False)
            effective_dopant_concentration (float):
                Fixed concentration (in cm^-3) of an arbitrary dopant/impurity
                in the material to include in the charge neutrality condition,
                in order to analyse the Fermi level / doping response under
                hypothetical doping conditions. If a negative value is given,
                the dopant is assumed to be an acceptor dopant (i.e. negative
                defect charge state), while a positive value corresponds to
                donor doping. For dopants of charge ``q``, the input value
                should be ``q * 'Dopant Concentration'``.
                (Default: None; no extrinsic dopant)
            site_competition (bool):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                If ``False``, uses the standard dilute limit approximation.

                Note that while large ``dist_tol``\s are often preferable for
                plotting (to condense the defect formation energy lines), this
                is often not ideal for determining site competition in
                concentration analyses as it can lead to unrealistically-large
                clusters.

        Returns:
            Self consistent Fermi level (in eV from the VBM (``self.vbm``)),
            and the corresponding electron and hole concentrations (in cm^-3)
            if ``return_concs=True``.
        """
        if bulk_dos is not None:
            self._bulk_dos = self._parse_fermi_dos(bulk_dos, skip_dos_check=skip_dos_check)

        if self.bulk_dos is None:  # none provided, and none previously set
            raise ValueError(
                "No bulk DOS calculation (`bulk_dos`) provided or previously parsed to "
                "`DefectThermodynamics.bulk_dos`, which is required for calculating carrier "
                "concentrations and solving for Fermi level position."
            )

        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None
        # handle chempot warning(s) once here, as otherwise brentq calls this function many times:
        chempots, limit = self._sanitise_chempots_for_concentrations(
            chempots, el_refs, limit
        )  # warns about chempots/limit choices if necessary

        def _get_total_q(fermi_level):
            conc_df = self.get_equilibrium_concentrations(
                chempots=chempots,
                limit=limit,
                el_refs=el_refs,
                temperature=temperature,
                fermi_level=fermi_level,
                site_competition=site_competition,
                lean=True,
            )
            # add effective dopant concentration if supplied:
            conc_df = _add_effective_dopant_concentration(conc_df, effective_dopant_concentration)
            # Defect/Charge not set as index w/lean=True & per_charge=False, for speed
            qd_tot = (conc_df["Charge"] * conc_df["Concentration (cm^-3)"]).sum()
            qd_tot += get_doping(
                fermi_dos=self.bulk_dos, fermi_level=fermi_level + self.vbm, temperature=temperature
            )
            return qd_tot

        eq_fermi_level: float = brentq(_get_total_q, -1.0, self.band_gap + 1.0)  # type: ignore
        if return_concs:
            e_conc, h_conc = get_e_h_concs(
                self.bulk_dos, eq_fermi_level + self.vbm, temperature  # type: ignore
            )
            return eq_fermi_level, e_conc, h_conc

        return eq_fermi_level

    def get_fermi_level_and_concentrations(
        self,
        bulk_dos: FermiDos | Vasprun | PathLike | None = None,
        annealing_temperature: float = 1000,
        quenched_temperature: float = 300,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        delta_gap: float | Callable = 0.0,
        per_charge: bool = True,
        per_site: bool = False,
        skip_formatting: bool = False,
        return_annealing_values: bool = False,
        effective_dopant_concentration: float | None = None,
        site_competition: bool | str = True,
        **kwargs,
    ) -> (
        tuple[float, float, float, pd.DataFrame]
        | tuple[float, float, float, pd.DataFrame, float, float, float, pd.DataFrame]
    ):
        r"""
        Calculate the self-consistent Fermi level and corresponding
        carrier/defect calculations, for a given chemical potential limit,
        annealing temperature and quenched/operating temperature, using the
        frozen defect and dilute limit approximations under the constraint of
        charge neutrality.

        This function works by calculating the self-consistent Fermi level and
        total concentration of each defect type at the annealing temperature,
        then fixing the total concentrations to these values and re-calculating
        the self-consistent (constrained equilibrium) Fermi level and relative
        charge state concentrations under this constraint at the quenched /
        operating temperature.

        According to the 'frozen defect' approximation, we typically expect
        defect concentrations to reach equilibrium during annealing/crystal
        growth (at elevated temperatures), but `not` upon quenching (i.e. at
        room/operating temperature) where we expect kinetic inhibition of
        defect annhiliation and hence non-equilibrium defect concentrations /
        Fermi level. Typically, this is approximated by computing the
        equilibrium Fermi level and defect concentrations at the annealing
        temperature, and then assuming the total concentration of each defect
        is fixed to this value, but that the relative populations of defect
        charge states (and the Fermi level) can re-equilibrate at the lower
        (room) temperature. See https://doi.org/10.1039/D3CS00432E (brief
        discussion), https://doi.org/10.1016/j.cpc.2019.06.017 (detailed) and
        ``doped``/``py-sc-fermi`` tutorials for more information. In certain
        cases (such as Li-ion battery materials or extremely slow charge
        capture/emission), these approximations may have to be adjusted such
        that some defects/charge states are considered fixed and some are
        allowed to re-equilibrate (e.g. highly mobile Li vacancies /
        interstitials). The ``FermiSolver`` class can be used in these cases
        for more fine-grained control over constraints and approximations in
        defect concentration calculations, as demonstrated in the tutorial:
        https://doped.readthedocs.io/en/latest/fermisolver_tutorial.html

        Note that different defect entries (different charge states, and/or
        ground and metastable states (different spin or geometries); e.g.
        interstitials at a given site) are grouped together in distinct defect
        types according to ``self.dist_tol`` , which is also used in plotting
        and transition level analysis. In the frozen defect approximation here,
        the total concentration of a given defect type group is calculated at
        the annealing temperature, and then the equilibrium relative population
        of the constituent entries is recalculated at the quenched temperature.

        Note that the bulk DOS calculation should be well-converged with
        respect to `k`-points for accurate Fermi level predictions!

        The degeneracy/multiplicity factor "g" is an important parameter in the
        defect concentration equation, affecting the final concentration by up
        to 2 orders of magnitude. This factor is taken from the product of the
        ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes. See discussion in:
        https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E.

        Note that, in addition to finer-grained control over constraints and
        approximations as mentioned above, the ``FermiSolver`` class implements
        a number of convenience methods for thermodynamic analyses; such as
        scanning over temperatures, chemical potentials, effective dopant
        concentrations etc., minimising or maximising a target property (e.g.
        defect/carrier concentration), which may be useful -- see tutorial.

        Args:
            bulk_dos (FermiDos or Vasprun or PathLike):
                ``pymatgen`` ``FermiDos`` for the bulk electronic density of
                states (DOS), for calculating carrier concentrations.
                Alternatively, can be a ``pymatgen`` ``Vasprun`` object or path
                to the ``vasprun.xml(.gz)`` output of a bulk DOS calculation in
                VASP -- however this will be much slower when looping over many
                conditions as it will re-parse the DOS each time! (So
                preferably set ``DefectThermodynamics.bulk_docs = ...`` or upon
                initialisation: ``DefectThermodynamics(..., bulk_dos=...)``.

                Usually this is a static calculation with the `primitive` cell
                of the bulk material, with relatively dense `k`-point sampling
                (especially for materials with disperse band edges) to ensure
                an accurately-converged DOS and thus Fermi level. Using large
                ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing)
                are recommended for best convergence (wrt `k`-point sampling)
                in VASP. Consistent functional settings should be used for the
                bulk DOS and defect supercell calculations. See
                https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations

                ``bulk_dos`` can also be left as ``None`` (default), if it has
                previously been provided and parsed, and thus is set as the
                ``self.bulk_dos`` attribute. If provided, will overwrite the
                ``self.bulk_dos`` attribute!
            annealing_temperature (float):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is 1000 K.
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Default is 300 K.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus concentrations and Fermi
                level). If ``None`` (default), will use ``self.chempots``. This
                can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.
            limit (str):
                The chemical potential limit for which to calculate formation
                energies (and thus concentrations and Fermi level). Can be:

                - ``None``, if ``chempots`` corresponds to a single chemical
                  potential limit -- otherwise will use the first chemical
                  potential limit in the ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` (with
                the same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of the ``FermiDos`` object
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``fermi_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            per_charge (bool):
                Whether to break down the defect concentrations into individual
                defect charge states (e.g. ``v_Cd_0``, ``v_Cd_-1``, ``v_Cd_-2``
                instead of ``v_Cd``). Default is ``True``.
            per_site (bool):
                Whether to also return the concentrations as per-site
                concentrations in percent (i.e. concentration divided by
                ``DefectEntry.bulk_site_concentration``). (default: False)
            skip_formatting (bool):
                Whether to skip formatting the defect charge states and
                concentrations as strings (and keep as ``int``\s and
                ``float``\s instead). (default: False)
            return_annealing_values (bool):
                If True, also returns the Fermi level, electron and hole
                concentrations and defect concentrations at the annealing
                temperature. (default: False)
            effective_dopant_concentration (float):
                Fixed concentration (in cm^-3) of an arbitrary dopant/impurity
                in the material to include in the charge neutrality condition,
                in order to analyse the Fermi level / doping response under
                hypothetical doping conditions. If a negative value is given,
                the dopant is assumed to be an acceptor dopant (i.e. negative
                defect charge state), while a positive value corresponds to
                donor doping. For dopants of charge ``q``, the input value
                should be ``q * 'Dopant Concentration'``.
                (Default: None; no extrinsic dopant)
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                If ``False``, uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0) or ``_parse_fermi_dos``
                (``skip_dos_check``; to skip the warning about the DOS VBM
                differing from ``self.vbm`` by >0.05 eV; default is False).

        Returns:
            Predicted quenched Fermi level (in eV from the VBM (``self.vbm``)),
            the corresponding electron and hole concentrations (in cm^-3) and a
            dataframe of the quenched defect concentrations (in cm^-3);
            ``(fermi_level, e_conc, h_conc, conc_df)``.
            If ``return_annealing_values=True``, `also` returns the annealing
            Fermi level, electron and hole concentrations and a dataframe of
            the annealing defect concentrations (in cm^-3);
            ``(...conc_df, A_fermi_level, A_e_conc, A_h_conc, A_conc_df)``
            where ``A_...`` is the property at the annealing temperature.
        """
        if kwargs and any(i not in ["verbose", "tol", "skip_dos_check"] for i in kwargs):
            raise ValueError(f"Invalid keyword arguments: {', '.join(kwargs.keys())}")

        if bulk_dos is not None:
            self._bulk_dos = self._parse_fermi_dos(
                bulk_dos, skip_dos_check=kwargs.get("skip_dos_check", False)
            )

        if self.bulk_dos is None:  # none provided, and none previously set
            raise ValueError(
                "No bulk DOS calculation (`bulk_dos`) provided or previously parsed to "
                "`DefectThermodynamics.bulk_dos`, which is required for calculating carrier "
                "concentrations and solving for Fermi level position."
            )
        orig_fermi_dos = deepcopy(self.bulk_dos)  # can get modified during annealing loops

        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None
        # handle chempot warning(s) once here, as otherwise brentq calls this function many times:
        chempots, limit = self._sanitise_chempots_for_concentrations(
            chempots, el_refs, limit
        )  # warns about chempots/limit choices if necessary

        annealing_dos = (
            self.bulk_dos
            if delta_gap == 0
            else scissor_dos(
                delta_gap if not callable(delta_gap) else delta_gap(annealing_temperature),
                self.bulk_dos,
                verbose=kwargs.get("verbose", False),
                tol=kwargs.get("tol", 1e-8),
            )
        )

        annealing_fermi_level = self.get_equilibrium_fermi_level(
            annealing_dos,
            chempots=chempots,
            limit=limit,
            el_refs=el_refs,
            temperature=annealing_temperature,
            return_concs=False,
            skip_dos_check=True,  # already warned if necessary
            effective_dopant_concentration=effective_dopant_concentration,
            site_competition=bool(site_competition),
        )
        assert not isinstance(annealing_fermi_level, tuple)  # float w/ return_concs=False, for typing
        self._bulk_dos = orig_fermi_dos  # reset to original DOS for quenched calculations

        annealing_defect_concentrations = self.get_equilibrium_concentrations(
            chempots=chempots,
            limit=limit,
            el_refs=el_refs,
            fermi_level=annealing_fermi_level,  # type: ignore
            temperature=annealing_temperature,
            per_charge=False,  # give total concentrations for each defect
            site_competition=site_competition,
            lean=True,
        )
        annealing_defect_concentrations = _add_effective_dopant_concentration(
            annealing_defect_concentrations, effective_dopant_concentration
        )  # add effective dopant concentration if supplied
        total_concentrations = dict(  # {Defect: Total Concentration (cm^-3)}
            zip(
                annealing_defect_concentrations.index,  # index is Defect name, when per_charge=False
                annealing_defect_concentrations["Concentration (cm^-3)"],
                strict=False,
            )
        )

        get_constrained_concentrations = partial(
            self._get_constrained_concentrations,
            total_concentrations=total_concentrations,
            temperature=quenched_temperature,
            chempots=chempots,
            limit=limit,
            el_refs=el_refs,
            per_charge=per_charge,
            per_site=per_site,
            skip_formatting=skip_formatting,
            effective_dopant_concentration=effective_dopant_concentration,
            site_competition=site_competition,
            lean=False,
        )

        def _get_constrained_total_q(fermi_level):
            conc_df = get_constrained_concentrations(
                fermi_level, per_charge=True, per_site=False, skip_formatting=True, lean=True
            )
            # Defect/Charge not set as index w/lean=True (default), for speed
            qd_tot = (conc_df["Charge"] * conc_df["Concentration (cm^-3)"]).sum()
            qd_tot += get_doping(  # use orig fermi dos for quenched temperature
                fermi_dos=orig_fermi_dos,
                fermi_level=fermi_level + self.vbm,
                temperature=quenched_temperature,
            )
            return qd_tot

        eq_fermi_level: float = brentq(_get_constrained_total_q, -1.0, self.band_gap + 1.0)  # type: ignore
        e_conc, h_conc = get_e_h_concs(
            orig_fermi_dos, eq_fermi_level + self.vbm, quenched_temperature  # type: ignore
        )
        conc_df = get_constrained_concentrations(eq_fermi_level)  # not lean for output

        if not return_annealing_values:
            return (eq_fermi_level, e_conc, h_conc, conc_df)

        annealing_e_conc, annealing_h_conc = get_e_h_concs(
            annealing_dos, annealing_fermi_level + self.vbm, annealing_temperature  # type: ignore
        )
        annealing_defect_concentrations = get_constrained_concentrations(
            annealing_fermi_level, temperature=annealing_temperature
        )

        return (
            eq_fermi_level,
            e_conc,
            h_conc,
            conc_df,
            annealing_fermi_level,
            annealing_e_conc,
            annealing_h_conc,
            annealing_defect_concentrations,
        )

    def _get_constrained_concentrations(
        self,
        fermi_level: float,
        total_concentrations: dict[str, float],
        temperature: float = 300,
        chempots: dict | None = None,
        limit: str | None = None,
        el_refs: dict | None = None,
        per_charge: bool = True,
        per_site: bool = False,
        skip_formatting: bool = True,
        effective_dopant_concentration: float | None = None,
        site_competition: bool | str = True,
        lean: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience method to calculate defect populations under constrained
        equilibrium, where total defect concentrations are fixed (according to
        ``total_concentrations``, which should be provided in the format:
        ``{defect name: concentration in cm^-3}``) and their relative charge
        state populations are re-calculated at the quenched temperature
        (``temperature``).

        See ``DefectThermodynamics.get_fermi_level_and_concentrations()`` for
        details.
        """
        conc_df = self.get_equilibrium_concentrations(
            chempots=chempots,
            limit=limit,
            el_refs=el_refs,
            temperature=temperature,
            fermi_level=fermi_level,
            per_charge=per_charge,
            per_site=per_site,
            skip_formatting=True,
            site_competition=site_competition,
            lean=lean,
        )
        defects = conc_df["Defect"] if lean else conc_df.index.get_level_values("Defect")
        unconstrained_total_concentrations = conc_df.groupby("Defect")["Concentration (cm^-3)"].transform(
            "sum"
        )
        # set total concentration to match annealing concentrations but with same relative concentrations
        conc_columns = [col for col in conc_df.columns if "Concentration" in col]
        conc_df["Total Concentration (cm^-3)"] = defects.map(total_concentrations)
        for conc_col in conc_columns:  # doesn't include Total Concentration
            conc_df[conc_col] *= (
                conc_df["Total Concentration (cm^-3)"] / unconstrained_total_concentrations
            )

        if not per_charge:
            conc_df = _group_defect_charge_state_concentrations(conc_df)
            # drop "Total Concentration" as it's a duplicate of "Concentration" ``per_charge=False``:
            conc_df = conc_df.drop(columns=["Total Concentration (cm^-3)"])

        conc_df = _add_effective_dopant_concentration(conc_df, effective_dopant_concentration)

        if not skip_formatting:
            conc_columns = [col for col in conc_df.columns if "Concentration" in col]
            for conc_col in conc_columns:  # Now includes Total Concentration (if present)
                conc_df[conc_col] = conc_df[conc_col].apply(
                    lambda x, conc_col=conc_col: _format_concentration(
                        x,
                        per_site=("per site" in conc_col),
                        skip_formatting=skip_formatting,
                    )
                )

            if per_charge:  # format charge states
                conc_df.index = conc_df.index.set_levels(
                    conc_df.index.levels[1].map(
                        lambda q: f"{'+' if q > 0 else ''}{int(q) if np.isclose(q, int(q)) else q}"
                    ),
                    level=1,
                )

        return conc_df

    def _get_in_gap_fermi_level_stability_window(self, defect_entry: str | DefectEntry) -> float:
        """
        Convenience method to calculate the maximum difference between a Fermi
        level at which ``defect_entry`` is the ground-state charge state, and
        the band edges.

        i.e. taken from the minimum of (CBM - lowest TL) and (highest TL - VBM)
        where TL is any transition level involving ``defect_entry``.

        Args:
            defect_entry (str or DefectEntry):
                Either a string of the defect entry name (in
                ``DefectThermodynamics.defect_entries``), or a
                ``DefectEntry`` object.

        Returns:
            float:
                Maximum difference between Fermi levels at which
                ``defect_entry`` is the ground-state charge state, and the band
                edges.
        """
        if not isinstance(defect_entry, DefectEntry):
            defect_entry = self.defect_entries[defect_entry]
        assert isinstance(defect_entry, DefectEntry)

        stable_entries_name_dict = {
            name: [ent.name for ent in entry_list] for name, entry_list in self.stable_entries.items()
        }
        if not any(defect_entry.name in name_list for name_list in stable_entries_name_dict.values()):
            return -np.inf

        grouped_defect_name_wout_charge = next(
            name
            for name in stable_entries_name_dict
            if defect_entry.name in stable_entries_name_dict[name]
        )

        # get highest and lowest TL (defining stability window):
        lowest = np.inf
        highest = -np.inf
        for tl, charge_list in self.transition_level_map[grouped_defect_name_wout_charge].items():
            if charge_list[-1] == defect_entry.charge_state:
                lowest = tl
            if charge_list[0] == defect_entry.charge_state:
                highest = tl

        if lowest == np.inf and highest == -np.inf:
            # no TLs and already checked stable -> only stable charge state
            return np.inf

        stability_windows = np.array([self.band_gap - lowest, highest])
        return min(stability_windows[np.isfinite(stability_windows)])

    def __repr__(self):
        """
        Returns a string representation of the ``DefectThermodynamics`` object.
        """
        formula = _get_bulk_supercell(
            next(iter(self.defect_entries.values()))
        ).composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectThermodynamics for bulk composition {formula} with {len(self.defect_entries)} "
            f"defect entries (in self.defect_entries). Available attributes:\n"
            f"{properties}\n\nAvailable methods:\n{methods}"
        )

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the ``defect_entries``
        dictionary attribute, if the attribute doesn't exist in
        ``DefectThermodynamics``.
        """
        try:
            super().__getattribute__(attr)
        except AttributeError as exc:
            if attr == "_defect_entries":
                raise exc
            return getattr(self._defect_entries, attr)

    def __getitem__(self, key):
        """
        Makes ``DefectThermodynamics`` object subscriptable, so that it can be
        indexed like a dictionary, using the ``defect_entries`` dictionary
        attribute.
        """
        return self.defect_entries[key]

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the ``defect_entries``
        dictionary, using ``add_entries`` (to check compatibility).
        """
        self.add_entries(
            [
                value,
            ]
        )

    def __delitem__(self, key):
        """
        Deletes the specified defect entry from the ``defect_entries``
        dictionary.
        """
        del self.defect_entries[key]

    def __contains__(self, key):
        """
        Returns ``True`` if the ``defect_entries`` dictionary contains the
        specified defect name.
        """
        return key in self.defect_entries

    def __len__(self):
        """
        Returns the number of entries in the ``defect_entries`` dictionary.
        """
        return len(self.defect_entries)

    def __iter__(self):
        """
        Returns an iterator over the ``defect_entries`` dictionary.
        """
        return iter(self.defect_entries)


DefectThermodynamics.get_quenched_fermi_level_and_concentrations = (
    DefectThermodynamics.get_fermi_level_and_concentrations
)  # for backwards compatibility, to be removed in next major release


def _check_chempots_and_limit_settings(chempots: dict | None = None, limit: str | None = None):
    """
    Convenience function to check the input values of ``chempots`` and
    ``limit`` for ``doped`` thermodynamic analysis functions, and warn users
    (once) if no chemical potentials or limits are specified.

    Returns the default chemical potential ``limit`` to be used, if
    ``chempots`` is ``None``.
    """
    if chempots is None:
        _no_chempots_warning()
    else:
        limit = _parse_limit(chempots, limit)
        if limit is None:
            limit = next(iter(chempots["limits"].keys()))
            if len(chempots["limits"]) > 1:  # more than 1 limit, so warn
                warnings.warn(
                    f"No chemical potential limit specified! Using {limit} for computing the formation "
                    f"energy"
                )

    return limit


def _add_effective_dopant_concentration(
    conc_df: pd.DataFrame,
    effective_dopant_concentration: float | None = None,
):
    """
    Add the effective dopant concentration to the concentration ``DataFrame``.

    Args:
        conc_df (pd.DataFrame):
            ``DataFrame`` of defect concentrations.
        effective_dopant_concentration (float):
            The effective dopant concentration to add to the ``DataFrame``.
            For dopants of charge ``q``, the input value should be
            ``q * 'Dopant Concentration'``.
            (Default: None; no extrinsic dopant)

    Returns:
        pd.DataFrame:
            ``DataFrame`` of defect concentrations with the effective
            dopant concentration
    """
    if effective_dopant_concentration is None:
        return conc_df

    # if ``lean`` was ``True``, then Defect/Charge not set as indices:
    lean = all(i in conc_df.columns for i in ["Defect", "Charge"])
    eff_dopant_df = pd.DataFrame(
        {
            "Defect": "Dopant",
            "Charge": int(np.sign(effective_dopant_concentration)),
            "Formation Energy (eV)": np.nan,
            "Concentration (cm^-3)": np.abs(effective_dopant_concentration),
            "Charge State Population": "100.0%",
        },
        index=[0],
    )
    if not lean:
        eff_dopant_df = eff_dopant_df.set_index(conc_df.index.names)

    for col in conc_df.columns:
        if col not in eff_dopant_df.columns:
            # if string, set to "N/A", otherwise e.g. concentration per site, if per_site=True
            eff_dopant_df[col] = "N/A" if isinstance(conc_df[col].iloc[0], str) else np.nan

    columns_to_drop = [col for col in eff_dopant_df.columns if col not in conc_df.columns]
    eff_dopant_df = eff_dopant_df.drop(columns=columns_to_drop)
    eff_dopant_df = eff_dopant_df[conc_df.columns]  # ensure it matches the original order
    return pd.concat([conc_df, eff_dopant_df], ignore_index=lean)


def _group_defect_charge_state_concentrations(
    conc_df: pd.DataFrame,
):
    summed_df = conc_df.groupby("Defect").sum(numeric_only=True)  # auto-reordered by groupby sum
    defects = (
        conc_df["Defect"] if "Defect" in conc_df.columns else conc_df.index.get_level_values("Defect")
    )
    summed_df = summed_df.loc[defects.unique()]  # retain ordering
    if "Lattice Site Index" in conc_df.columns:  # don't sum lattice site indices
        summed_df["Lattice Site Index"] = conc_df.groupby("Defect")["Lattice Site Index"].first()

    return summed_df.drop(  # Defect set as index now, from groupby()
        columns=[
            i
            for i in [
                "Charge",
                "Formation Energy (eV)",
                "Charge State Population",
            ]
            if i in summed_df.columns
        ]
    )


def _format_concentration(raw_concentration: float, per_site: bool = False, skip_formatting: bool = False):
    """
    Format concentration values for ``DataFrame`` outputs.
    """
    if skip_formatting:
        return raw_concentration
    if per_site:
        return _format_per_site_concentration(raw_concentration)

    return f"{raw_concentration:.3e}"


def _format_per_site_concentration(raw_concentration: float):
    """
    Format per-site concentrations for ``DataFrame`` outputs.
    """
    if isinstance(raw_concentration, str):
        return raw_concentration
    if np.isnan(raw_concentration):
        return "N/A"
    if raw_concentration > 1e-5:
        return f"{raw_concentration:.3%}"
    return f"{raw_concentration * 100:.3e} %"


def get_fermi_dos(dos_vr: PathLike | Vasprun):
    """
    Create a ``FermiDos`` object from the provided ``dos_vr``, which can be
    either a path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` ``Vasprun``
    object (parsed with ``parse_dos = True``).

    Args:
        dos_vr (Union[PathLike, Vasprun]):
            Path to a ``vasprun.xml(.gz)`` file, or a ``Vasprun`` object.

    Returns:
        FermiDos: The ``FermiDos`` object.
    """
    if not isinstance(dos_vr, Vasprun):
        dos_vr = get_vasprun(dos_vr, parse_dos=True)

    return FermiDos(dos_vr.complete_dos, nelecs=get_nelect_from_vasprun(dos_vr))


def get_e_h_concs(
    fermi_dos: FermiDos, fermi_level: float, temperature: float = 300
) -> tuple[float, float]:
    """
    Get the corresponding electron and hole concentrations (in cm^-3) for a
    given Fermi level (in eV) and temperature (in K), for a ``FermiDos``
    object.

    Note that the Fermi level here is NOT referenced to the VBM! So the Fermi
    level should be the corresponding eigenvalue within the calculation (or in
    other words, the Fermi level relative to the VBM plus the VBM eigenvalue).

    Args:
        fermi_dos (FermiDos):
            ``pymatgen`` ``FermiDos`` for the bulk electronic density of states
            (DOS), for calculating carrier concentrations.

            Usually this is a static calculation with the `primitive` cell of
            the bulk material, with relatively dense `k`-point sampling
            (especially for materials with disperse band edges) to ensure an
            accurately-converged DOS and thus Fermi level. Using large
            ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing) are
            recommended for best convergence (wrt `k`-point sampling) in VASP.
            Consistent functional settings should be used for the bulk DOS and
            defect supercell calculations. See:
            https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations
        fermi_level (float):
            Value corresponding to the electron chemical potential, **not**
            referenced to the VBM! (i.e. same eigenvalue reference as the raw
            calculation)
        temperature (float):
            Temperature in Kelvin at which to calculate the equilibrium
            concentrations. Default is 300 K.

    Returns:
        tuple[float, float]: The electron and hole concentrations in cm^-3.
    """
    with np.errstate(over="ignore"):  # ignore overflow warnings from f0, can remove in
        # future versions following SK's fix in https://github.com/materialsproject/pymatgen/pull/3879
        # code for obtaining the electron and hole concentrations here is taken from
        # FermiDos.get_doping(), and updated by SK to be independent of estimated VBM/CBM positions (using
        # correct DOS integral) and better handle exponential overflows (by editing `f0` in `pymatgen`)
        idx_mid_gap = int(fermi_dos.idx_vbm + (fermi_dos.idx_cbm - fermi_dos.idx_vbm) / 2)
        e_conc: float = np.sum(
            fermi_dos.tdos[max(idx_mid_gap, fermi_dos.idx_vbm + 1) :]
            * f0(
                fermi_dos.energies[max(idx_mid_gap, fermi_dos.idx_vbm + 1) :],
                fermi_level,  # type: ignore
                temperature,
            )
            * fermi_dos.de[max(idx_mid_gap, fermi_dos.idx_vbm + 1) :],
            axis=0,
        ) / (fermi_dos.volume * fermi_dos.A_to_cm**3)
        h_conc: float = np.sum(
            fermi_dos.tdos[: min(idx_mid_gap, fermi_dos.idx_cbm - 1) + 1]
            * f0(
                -fermi_dos.energies[: min(idx_mid_gap, fermi_dos.idx_cbm - 1) + 1],
                -fermi_level,  # type: ignore
                temperature,
            )
            * fermi_dos.de[: min(idx_mid_gap, fermi_dos.idx_cbm - 1) + 1],
            axis=0,
        ) / (fermi_dos.volume * fermi_dos.A_to_cm**3)

    return e_conc, h_conc


def get_doping(fermi_dos: FermiDos, fermi_level: float, temperature: float = 300) -> float:
    """
    Get the doping concentration (majority - minority carrier concentration) in
    cm^-3 for a given Fermi level (in eV) and temperature (in K), for a
    ``FermiDos`` object.

    Note that the Fermi level here is NOT referenced to the VBM! So the Fermi
    level should be the corresponding eigenvalue within the calculation (or in
    other words, the Fermi level relative to the VBM plus the VBM eigenvalue).

    Refactored from ``FermiDos.get_doping()`` to be more accurate/robust
    (independent of estimated VBM/CBM positions, avoiding overflow warnings).

    Args:
        fermi_dos (FermiDos):
            ``pymatgen`` ``FermiDos`` for the bulk electronic density of states
            (DOS), for calculating carrier concentrations.

            Usually this is a static calculation with the `primitive` cell of
            the bulk material, with relatively dense `k`-point sampling
            (especially for materials with disperse band edges) to ensure an
            accurately-converged DOS and thus Fermi level. Using large
            ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing) are
            recommended for best convergence (wrt `k`-point sampling) in VASP.
            Consistent functional settings should be used for the bulk DOS and
            defect supercell calculations. See:
            https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations
        fermi_level (float):
            Value corresponding to the electron chemical potential, **not**
            referenced to the VBM! (i.e. same eigenvalue reference as the raw
            calculation)
        temperature (float):
            Temperature in Kelvin at which to calculate the equilibrium
            concentrations. Default is 300 K.

    Returns:
        float: The doping concentration in cm^-3
    """
    # can replace this function with the ``FermiDos.get_doping()`` method in future versions following SK's
    # fix in https://github.com/materialsproject/pymatgen/pull/3879, whenever pymatgen>2024.6.10 becomes
    # a ``doped`` requirement (same for overflow catches in ``get_e_h_concs`` etc.)
    e_conc, h_conc = get_e_h_concs(fermi_dos, fermi_level, temperature)
    return h_conc - e_conc


def scissor_dos(delta_gap: float, dos: Dos | FermiDos, tol: float = 1e-8, verbose: bool = True):
    """
    Given an input ``Dos``/``FermiDos`` object, rigidly shifts the valence and
    conduction bands of the DOS object to give a band gap that is now
    increased/decreased by ``delta_gap`` eV, where this rigid scissor shift is
    applied symmetrically around the original gap (i.e. the VBM is downshifted
    by ``delta_gap/2`` and the CBM is upshifted by ``delta_gap/2``).

    Note this assumes a non-spin-polarised (i.e. non-magnetic) density of
    states!

    Args:
        delta_gap (float):
            The amount by which to increase/decrease the band gap (in eV).
        dos (Dos/FermiDos):
            The input DOS object to scissor.
        tol (float):
            The tolerance to use for determining the VBM and CBM (used in
            ``Dos.get_gap(tol=tol)``). Default: 1e-8.
        verbose (bool):
            Whether to print information about the original and new band gaps.

    Returns:
        FermiDos: The scissored DOS object.
    """
    dos = deepcopy(dos)  # don't overwrite object
    # shift just CBM upwards first, then shift all rigidly down by Eg/2 (simpler code with this approach)
    cbm_index = np.where(
        (dos.densities[Spin.up] > tol) & (dos.energies - dos.efermi > dos.get_gap(tol=tol) / 2)
    )[0][0]
    cbm_energy = dos.energies[cbm_index]
    # get closest index with energy near cbm_energy + delta_gap:
    new_cbm_index = np.argmin(np.abs(dos.energies - (cbm_energy + delta_gap)))
    new_cbm_energy = dos.energies[new_cbm_index]
    vbm_index = np.where(
        (dos.densities[Spin.up] > tol) & (dos.energies - dos.efermi < dos.get_gap(tol=tol) / 2)
    )[0][-1]
    vbm_energy = dos.energies[vbm_index]

    if not np.isclose(cbm_energy - vbm_energy, dos.get_gap(tol=tol), atol=1e-1) and np.isclose(
        new_cbm_energy - cbm_energy, delta_gap, atol=1e-2
    ):
        warnings.warn(
            "The new band gap does not appear to be equal to the original band gap plus the scissor "
            "shift, suggesting an error in `scissor_dos`, beware!\n"
            f"Got original gap (from manual indexing): {cbm_energy - vbm_energy}, {dos.get_gap(tol=tol)} "
            f"from dos.get_gap(tol=tol) and new gap: {dos.get_gap(tol=tol) + delta_gap}"
        )

    # Determine the number of values in energies/densities to remove/add to avoid duplication
    values_to_remove_or_add = int(new_cbm_index - cbm_index)
    scissored_dos_dict = dos.as_dict()

    # Shift the CBM and remove/add values:
    if values_to_remove_or_add < 0:  # remove values
        scissored_dos_dict["energies"] = np.concatenate(
            (dos.energies[: cbm_index + values_to_remove_or_add], dos.energies[cbm_index:] + delta_gap)
        )
        scissored_dos_dict["densities"][Spin.up] = np.concatenate(
            (
                dos.densities[Spin.up][: cbm_index + values_to_remove_or_add],
                dos.densities[Spin.up][cbm_index:],
            )
        )
        # Assuming non-spin-polarised bulk here:
        scissored_dos_dict["densities"][Spin.down] = scissored_dos_dict["densities"][Spin.up]
    elif values_to_remove_or_add > 0:
        # add more zero DOS values:
        scissored_dos_dict["energies"] = np.concatenate(
            (
                dos.energies[:cbm_index],
                dos.energies[cbm_index:new_cbm_index],
                dos.energies[cbm_index:] + delta_gap,
            )
        )
        scissored_dos_dict["densities"][Spin.up] = np.concatenate(
            (
                dos.densities[Spin.up][:cbm_index],
                np.zeros(values_to_remove_or_add),
                dos.densities[Spin.up][cbm_index:],
            )
        )
        scissored_dos_dict["densities"][Spin.down] = scissored_dos_dict["densities"][Spin.up]

    # now shift all energies rigidly, so we've shifted symmetrically around the original gap (eigenvalues)
    # ensure 'energies' is array (should be if function used correctly, not if band gap change is zero...):
    scissored_dos_dict["energies"] = np.array(scissored_dos_dict["energies"])
    scissored_dos_dict["energies"] -= np.float64(delta_gap / 2)
    scissored_dos_dict["efermi"] -= np.float64(delta_gap / 2)

    if verbose:
        print(f"Orig gap: {dos.get_gap(tol=tol):.4f}, new gap:{dos.get_gap(tol=tol) + delta_gap:.4f}")
    scissored_dos_dict["structure"] = dos.structure.as_dict()
    if isinstance(dos, FermiDos):
        return FermiDos.from_dict(scissored_dos_dict)
    return Dos.from_dict(scissored_dos_dict)


class FermiSolver(MSONable):
    def __init__(
        self,
        defect_thermodynamics: DefectThermodynamics,
        bulk_dos: Union[FermiDos, Vasprun, "PathLike"] | None = None,
        chempots: dict | None = None,
        el_refs: dict | None = None,
        backend: str = "doped",
        skip_dos_check: bool = False,
    ):
        r"""
        Class to calculate the Fermi level, defect and carrier concentrations
        under various conditions, using the input ``DefectThermodynamics``
        object.

        This class implements a number of convenience methods for thermodynamic
        analyses; such as scanning over temperatures, chemical potentials,
        effective dopant concentrations etc, minimising or maximising a target
        property (e.g. defect/carrier concentration), and also allowing greater
        control over constraints and approximations in defect concentration
        calculations; such as specifying (known) fixed concentrations of a
        defect/dopant, specifying defects to exclude from high-temperature
        concentration fixing in the frozen defect approximation (e.g. highly
        mobile defects), and/or fixing defect charge states upon quenching.

        This constructor initializes a ``FermiSolver`` object, setting up the
        necessary attributes, which includes loading the bulk density of states
        (DOS) data from either the input ``DefectThermodynamics`` or
        ``bulk_dos``.

        If using the ``py-sc-fermi`` backend (currently required for the
        ``fixed_defects``, ``free_defects`` and ``fix_charge_states`` options
        in the scanning functions), please cite the code paper:
        Squires et al., JOSS 2023; https://doi.org/10.21105/joss.04962.

        Args:
            defect_thermodynamics (DefectThermodynamics):
                A ``DefectThermodynamics`` object, providing access to defect
                formation energies and other related thermodynamic properties.
            bulk_dos (FermiDos or Vasprun or PathLike):
                Either a path to the ``vasprun.xml(.gz)`` output of a bulk DOS
                calculation in VASP, a ``pymatgen`` ``Vasprun`` object or a
                ``pymatgen`` ``FermiDos`` for the bulk electronic DOS, for
                calculating carrier concentrations.
                If not provided, uses ``DefectThermodynamics.bulk_dos`` if
                present.

                Usually this is a static calculation with the `primitive` cell
                of the bulk material, with relatively dense `k`-point sampling
                (especially for materials with disperse band edges) to ensure
                an accurately-converged DOS and thus Fermi level. Using large
                ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing)
                are recommended for best convergence (wrt `k`-point sampling)
                in VASP. Consistent functional settings should be used for the
                bulk DOS and defect supercell calculations. See:
                https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations

                Note that the ``DefectThermodynamics.bulk_dos`` will be set to
                match this input, if provided.
            chempots (Optional[dict]):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus concentrations and Fermi
                level), under different chemical environments.

                If ``None`` (default), will use ``DefectThermodynamics.chempots``
                (= 0 for all chemical potentials by default, if not set).

                This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                Chemical potentials can also be supplied later in each analysis
                function, or set using ``DefectThermodynamics.chempots = ...``
                or ``FermiSolver.defect_thermodynamics.chempots = ...`` (with
                the same input options) to set the default chemical potentials
                for all calculations.

                If provided here, then ``defect_thermodynamics.chempots`` is
                set to this input.
            el_refs (Optional[dict]):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in
                ``DefectThermodynamics`` in the format generated by ``doped``
                (see tutorials), or if ``DefectThermodynamics.el_refs`` has
                been set.

                If provided here, then ``defect_thermodynamics.el_refs`` is set
                to this input.

                Elemental reference energies can also be supplied later in each
                analysis function, or set using
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` (with the same input
                options). Default is ``None``.
            backend (Optional[str]):
                The code backend to use for the thermodynamic calculations,
                which can be either ``"doped"`` or ``"py-sc-fermi"``.
                ``"py-sc-fermi"`` allows the use of ``fixed_defects`` and
                ``free_defects`` for advanced constrained defect equilibria
                (e.g. mobile defects, see advanced thermodynamics tutorial),
                while ``"doped"`` is usually much quicker.
                Default is ``doped``, but will attempt to switch to
                ``py-sc-fermi`` if required (and installed).
            skip_dos_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                ``DefectThermodynamics.vbm`` by >0.05 eV. Should only be used
                when the reason for this difference is known/acceptable.
                Default is ``False``.

        Key attributes:
            defect_thermodynamics (DefectThermodynamics):
                The ``DefectThermodynamics`` object used for the thermodynamic
                calculations.
            backend (str):
                The code backend used for the thermodynamic calculations
                (``"doped"`` or ``"py-sc-fermi"``).
            volume (float):
                Volume of the unit cell in the bulk DOS calculation (stored in
                ``self.defect_thermodynamics.bulk_dos``).
            skip_dos_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                the defect entries VBM by >0.05 eV. Should only be used when
                the reason for this difference is known/acceptable.
            multiplicity_scaling (int):
                Scaling factor to account for the difference in volume between
                the defect supercells and the bulk DOS calculation cell, when
                using the ``py-sc-fermi`` backend.
            py_sc_fermi_dos (DOS):
                A ``py-sc-fermi`` ``DOS`` object, generated from the input
                ``FermiDos`` object, for use with the ``py-sc-fermi`` backend.
        """
        self.defect_thermodynamics = defect_thermodynamics
        self.skip_dos_check = skip_dos_check
        if bulk_dos is not None:
            self.defect_thermodynamics._bulk_dos = self.defect_thermodynamics._parse_fermi_dos(
                bulk_dos, skip_dos_check=self.skip_dos_check
            )
        if self.defect_thermodynamics.bulk_dos is None:
            raise ValueError(
                "No bulk DOS calculation (`bulk_dos`) provided or previously parsed to "
                "`DefectThermodynamics.bulk_dos`, which is required for calculating carrier "
                "concentrations and solving for Fermi level position."
            )
        self.volume: float = self.defect_thermodynamics.bulk_dos.volume

        if "fermi" in backend.lower():
            if bool(importlib.util.find_spec("py_sc_fermi")):
                self.backend = "py-sc-fermi"
            else:  # py-sc-fermi explicitly chosen but not installed
                raise ImportError(
                    "py-sc-fermi is not installed, so only the doped FermiSolver backend is available."
                )
        elif backend.lower() == "doped":
            self.backend = "doped"
        else:
            raise ValueError(f"Unrecognised `backend`: {backend}")

        self.multiplicity_scaling = 1
        self.py_sc_fermi_dos = None
        self._DefectSystem = self._DefectSpecies = self._DefectChargeState = self._DOS = None

        if self.backend == "py-sc-fermi":
            self._activate_py_sc_fermi_backend()

        # Parse chemical potentials, either using input values (after formatting them in the doped format)
        # or using the class attributes if set:
        self.defect_thermodynamics._chempots, self.defect_thermodynamics._el_refs = _parse_chempots(
            chempots or self.defect_thermodynamics.chempots, el_refs or self.defect_thermodynamics.el_refs
        )

        if self.defect_thermodynamics.chempots is None:
            raise ValueError(
                "You must supply a chemical potentials dictionary or have them present in the "
                "DefectThermodynamics object."
            )

    def _activate_py_sc_fermi_backend(self, error_message: str | None = None):
        try:
            from py_sc_fermi.defect_charge_state import DefectChargeState
            from py_sc_fermi.defect_species import DefectSpecies
            from py_sc_fermi.defect_system import DefectSystem
            from py_sc_fermi.dos import DOS
        except ImportError as exc:  # py-sc-fermi activation attempted but not installed
            finishing_message = (
                ", but py-sc-fermi is not installed, so only the doped FermiSolver "
                "backend is available!"
            )
            message = (
                error_message + finishing_message
                if error_message
                else "The py-sc-fermi backend was attempted to be activated" + finishing_message
            )
            raise ImportError(message) from exc

        self._DefectSystem = DefectSystem
        self._DefectSpecies = DefectSpecies
        self._DefectChargeState = DefectChargeState
        self._DOS = DOS

        if isinstance(self.defect_thermodynamics.bulk_dos, FermiDos):
            self.py_sc_fermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(
                self.defect_thermodynamics.bulk_dos,
                vbm=self.defect_thermodynamics.vbm,
                bandgap=self.defect_thermodynamics.band_gap,
            )

        ms = (
            next(iter(self.defect_thermodynamics.defect_entries.values())).defect.structure.volume
            / self.volume
        )
        if not np.isclose(ms, round(ms), atol=3e-2):  # check multiplicity scaling is almost an integer
            warnings.warn(
                f"The detected volume ratio ({ms:.3f}) between the defect supercells and the DOS "
                f"calculation cell is non-integer, indicating that they correspond to (slightly) "
                f"different unit cell volumes. This can cause quantitative errors of a similar "
                f"relative magnitude in the predicted defect/carrier concentrations!"
            )
        self.multiplicity_scaling = round(ms)

    def _check_required_backend_and_error(self, required_backend: str):
        """
        Check if the attributes needed for the ``required_backend`` backend are
        set, and throw an error message if not.

        Args:
            required_backend (str):
                Backend choice ("doped" or "py-sc-fermi") required.
        """
        if required_backend.lower() == "py-sc-fermi" and self._DOS is None:
            raise RuntimeError(
                f"This function is currently only supported for the {required_backend} backend, "
                f"but you are using the {self.backend} backend!"
            )

    def _get_fermi_level_and_carriers(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: dict[str, float] | None = None,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        site_competition: bool = True,
    ) -> tuple[float, float, float]:
        r"""
        Calculate the equilibrium Fermi level and carrier concentrations under
        a given chemical potential regime and temperature.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the
                equilibrium Fermi level position. Here, this should be a
                dictionary of chemical potentials for a single limit
                (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in
                ``self.defect_thermodynamics.el_refs`` then it is the formal
                chemical potentials (i.e. relative to the elemental reference
                energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if
                ``chempots`` was provided to ``self.defect_thermodynamics`` in
                the format generated by ``doped``). (Default: None)
            temperature (float):
                The temperature at which to solve for the Fermi level and
                carrier concentrations, in Kelvin. Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            site_competition (bool):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                If ``False``, uses the standard dilute limit approximation.

                Note that while large ``dist_tol``\s are often preferable for
                plotting (to condense the defect formation energy lines), this
                is often not ideal for determining site competition in
                concentration analyses as it can lead to unrealistically-large
                clusters.

        Returns:
            tuple[float, float, float]: A tuple containing:
                - The Fermi level (float) in eV.
                - The electron concentration (float) in cm^-3.
                - The hole concentration (float) in cm^-3.
        """
        self._check_required_backend_and_error("doped")
        fermi_level, electrons, holes = self.defect_thermodynamics.get_equilibrium_fermi_level(  # type: ignore
            chempots=single_chempot_dict,
            el_refs=el_refs,
            temperature=temperature,
            return_concs=True,
            effective_dopant_concentration=effective_dopant_concentration,
            site_competition=site_competition,
        )  # use already-set bulk dos
        return fermi_level, electrons, holes

    def _get_and_check_thermo_chempots(
        self, chempots: dict | None = None, el_refs: dict | None = None
    ) -> tuple[dict, dict]:
        """
        Convenience method to get the ``chempots`` and ``el_refs`` from
        ``self.defect_thermodynamics.chempots`` and check that ``chempots`` is
        not ``None``.
        """
        chempots, el_refs = self.defect_thermodynamics._get_chempots(
            chempots, el_refs
        )  # returns self.defect_thermodynamics.chempots if chempots is None
        if chempots is None:
            raise ValueError(  # this function is only called if no user chempots were supplied
                "No chemical potentials supplied or present in self.defect_thermodynamics.chempots!"
            )
        assert isinstance(el_refs, dict)  # both returned as dictionaries now
        return chempots, el_refs

    def _get_single_chempot_dict(
        self, limit: str | None = None, chempots: dict | None = None, el_refs: dict | None = None
    ) -> tuple[dict[str, float], Any]:
        """
        Get the chemical potentials for a single limit (``limit``) from the
        ``chempots`` (or ``self.defect_thermodynamics.chempots``) dictionary,
        `with respect to the elemental references` (i.e. from
        ``"limits_wrt_el_refs"``).

        Returns a `single` chemical potential dictionary for the specified
        limit.
        """
        chempots, el_refs = self._get_and_check_thermo_chempots(chempots, el_refs)
        limit = _parse_limit(chempots, limit)

        if (
            limit not in chempots["limits_wrt_el_refs"]
            and "User Chemical Potentials" not in chempots["limits_wrt_el_refs"]
        ):
            raise ValueError(
                f"Limit '{limit}' not found in the chemical potentials dictionary! You must specify "
                f"an appropriate limit or provide an appropriate chempots dictionary."
            )

        return chempots["limits_wrt_el_refs"][limit or "User Chemical Potentials"], el_refs

    def equilibrium_solve(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: dict[str, float] | None = None,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        append_chempots: bool = True,
        fixed_defects: dict[str, float] | None = None,
        site_competition: bool | str = True,
    ) -> pd.DataFrame:
        r"""
        Calculate the Fermi level and defect/carrier concentrations under
        `thermodynamic equilibrium`, given a set of chemical potentials and a
        temperature.

        Typically not intended for direct usage, as the same functionality is
        provided by ``DefectThermodynamics.get_equilibrium_concentrations`` /
        ``DefectThermodynamics.get_equilibrium_fermi_level()``.
        Rather this is used internally to provide a unified interface for both
        backends, within the ``scan_...`` functions.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the
                equilibrium Fermi level position and defect/carrier
                concentrations. Here, this should be a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in
                ``self.defect_thermodynamics.el_refs`` then it is the `formal`
                chemical potentials (i.e. relative to the elemental reference
                energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if
                ``chempots`` was provided to ``self.defect_thermodynamics`` in
                the format generated by ``doped``). (Default: None)
            temperature (float):
                The temperature at which to solve for defect concentrations,
                in Kelvin. Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            append_chempots (bool):
                Whether to append the chemical potentials (and effective dopant
                concentration, if provided) to the output ``DataFrame``.
                Default is ``True``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state.
                Concentrations should be given in cm^-3. This can be used to
                simulate the effect of a fixed impurity concentration.
                Defaults to ``None``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                If ``False``, uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the calculated defect and carrier
                concentrations, along with the self-consistent Fermi level.
                The columns include:

                    - "Defect":
                        The defect type.
                    - "Concentration (cm^-3)":
                        The concentration of the defect in cm^-3.
                    - "Temperature (K)":
                        The temperature at which the calculation was performed,
                        in Kelvin.
                    - "Fermi Level (eV wrt VBM)":
                        The self-consistent Fermi level in eV, relative to the
                        VBM of the bulk DOS.
                    - "Electrons (cm^-3)":
                        The electron concentration in cm^-3.
                    - "Holes (cm^-3)":
                        The hole concentration in cm^-3.
                    - "μ_X (eV)":
                        Chemical potentials in eV, if ``append_chempots`` is
                        ``True``.
                    - "Dopant (cm^-3)":
                        The effective arbitrary dopant concentration in cm^-3,
                        if set.
        """
        py_sc_fermi_required = fixed_defects is not None
        if py_sc_fermi_required and self._DOS is None:
            self._activate_py_sc_fermi_backend(
                error_message="The `fixed_defects` option is currently only supported for the py-sc-fermi "
                "backend"
            )

        if self.backend == "doped" and not py_sc_fermi_required:
            fermi_level, electrons, holes = self._get_fermi_level_and_carriers(
                single_chempot_dict=single_chempot_dict,
                el_refs=el_refs,
                temperature=temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                site_competition=bool(site_competition),
            )
            concentrations = self.defect_thermodynamics.get_equilibrium_concentrations(
                chempots=single_chempot_dict,
                el_refs=el_refs,
                fermi_level=fermi_level,
                temperature=temperature,
                per_charge=False,
                per_site=False,
                skip_formatting=True,  # keep concentration values as floats
                site_competition=site_competition,
            )
            # order in both cases is Defect, Concentration, Temperature, Fermi Level, e, h, Chempots
            new_columns = {
                "Temperature (K)": temperature,
                "Fermi Level (eV wrt VBM)": fermi_level,
                "Electrons (cm^-3)": electrons,
                "Holes (cm^-3)": holes,
            }

            for column, value in new_columns.items():
                concentrations[column] = value
            excluded_columns = ["Defect", "Charge", "Charge State Population"]
            for column in concentrations.columns.difference(excluded_columns):
                concentrations[column] = concentrations[column].astype(float)

        else:  # py-sc-fermi backend:
            defect_system = self._generate_defect_system(
                single_chempot_dict=single_chempot_dict,
                el_refs=el_refs,
                temperature=temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                fixed_defects=fixed_defects,
            )

            with np.errstate(all="ignore"):
                conc_dict = defect_system.concentration_dict()

            data = [
                {
                    "Defect": k,
                    "Concentration (cm^-3)": v,
                    "Temperature (K)": defect_system.temperature,
                    "Fermi Level (eV wrt VBM)": conc_dict["Fermi Energy"],
                    "Electrons (cm^-3)": conc_dict["n0"],
                    "Holes (cm^-3)": conc_dict["p0"],
                }
                for k, v in conc_dict.items()
                if k not in ["Fermi Energy", "n0", "p0", "Dopant"]
            ]

            concentrations = pd.DataFrame(data)
            concentrations = concentrations.set_index("Defect", drop=True)

        if append_chempots:
            for key, value in single_chempot_dict.items():
                concentrations[f"μ_{key} (eV)"] = value
        if effective_dopant_concentration is not None:
            concentrations["Dopant (cm^-3)"] = effective_dopant_concentration

        return concentrations

    def pseudo_equilibrium_solve(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: dict[str, float] | None = None,
        annealing_temperature: float = 1000,
        quenched_temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        append_chempots: bool = True,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Calculate the self-consistent Fermi level and corresponding
        carrier/defect calculations under pseudo-equilibrium conditions given a
        set of elemental chemical potentials, an annealing temperature, and a
        quenched temperature.

        Typically not intended for direct usage, as the same functionality is
        provided by
        ``DefectThermodynamics.get_fermi_level_and_concentrations()``. Rather
        this is used internally to provide a unified interface for both
        backends, within the ``scan_...`` functions.

        'Pseudo-equilibrium' here refers to the use of frozen defect and dilute
        limit approximations under the constraint of charge neutrality (i.e.
        constrained equilibrium). According to the 'frozen defect'
        approximation, we typically expect defect concentrations to reach
        equilibrium during annealing/crystal growth (at elevated temperatures),
        but `not` upon quenching (i.e. at room/operating temperature) where we
        expect kinetic inhibition of defect annhiliation and hence non-
        equilibrium defect concentrations / Fermi level. Typically, this is
        approximated by computing the equilibrium Fermi level and defect
        concentrations at the annealing temperature, and then assuming the
        total concentration of each defect is fixed to this value, but that the
        relative populations of defect charge states (and the Fermi level) can
        re-equilibrate at the lower (room) temperature. See discussion in
        https://doi.org/10.1039/D3CS00432E (brief),
        https://doi.org/10.1016/j.cpc.2019.06.017 (detailed) and ``doped`` /
        ``py-sc-fermi`` tutorials for more information. In certain cases (such
        as Li-ion battery materials or extremely slow charge capture/emission),
        these approximations may have to be adjusted such that some defects /
        charge states are considered fixed and some are allowed to
        re-equilibrate (e.g. highly mobile Li vacancies/interstitials).
        Modelling these specific cases can be achieved using the
        ``free_defects`` and/or ``fix_charge_states`` options, as demonstrated
        in: https://doped.readthedocs.io/en/latest/fermisolver_tutorial.html.

        This function works by calculating the self-consistent Fermi level and
        total concentration of each defect at the annealing temperature, then
        fixing the total concentrations to these values and re-calculating the
        self-consistent (constrained equilibrium) Fermi level and relative
        charge state concentrations under this constraint at the quenched /
        operating temperature. If using the ``"py-sc-fermi"`` backend, then you
        can optionally fix the concentrations of individual defect
        `charge states` (rather than fixing total defect concentrations) using
        ``fix_charge_states=True``, and/or optionally specify ``free_defects``
        to exclude from high-temperature concentration fixing (e.g. highly
        mobile defects).

        Note that the bulk DOS calculation should be well-converged with
        respect to k-points for accurate Fermi level predictions!

        The degeneracy/multiplicity factor "g" is an important parameter in the
        defect concentration equation (and thus Fermi level calculation),
        affecting the final concentration by up to 2 orders of magnitude. This
        factor is taken from the product of the
        ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes which are automatically
        determined during ``doped`` defect calculation parsing. Discussion in:
        https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the
                pseudo-equilibrium Fermi level position and defect/carrier
                concentrations. Here, this should be a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in
                ``self.defect_thermodynamics.el_refs`` then it is the `formal`
                chemical potentials (i.e. relative to the elemental reference
                energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if
                ``chempots`` was provided to ``self.defect_thermodynamics`` in
                the format generated by ``doped``). (Default: None)
            annealing_temperature (float):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Defaults to 1000 K.
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched
                temperature, in the format: ``{defect_name: concentration}``,
                where ``defect_name`` is the name of a defect entry without
                (e.g. ``"v_O"``) or with (e.g. ``"v_O_+2"``) the charge state;
                which will then fix either the total concentration of that
                defect or only the concentration for the specified charge
                state. Concentrations should be given in cm^-3. This can be
                used to fix the concentrations of specific defects regardless
                of the chemical potentials, or anneal-quench procedure (e.g. to
                simulate the effect of a fixed impurity concentration).
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names).
                Defaults to ``None``.
            append_chempots (bool):
                Whether to append the chemical potentials (and effective dopant
                concentration, if provided) to the output ``DataFrame``.
                Default is ``True``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the defect and carrier
                concentrations under pseudo-equilibrium conditions, along with
                the self-consistent Fermi level.
                The columns include:

                    - "Defect":
                        The defect type.
                    - "Concentration (cm^-3)":
                        The concentration of the defect in cm^-3.
                    - "Annealing Temperature (K)":
                        The annealing temperature in Kelvin.
                    - "Quenched Temperature (K)":
                        The quenched temperature in Kelvin.
                    - "Fermi Level (eV wrt VBM)":
                        The self-consistent Fermi level in eV, relative to the
                        VBM of the bulk DOS.
                    - "Electrons (cm^-3)":
                        The electron concentration in cm^-3.
                    - "Holes (cm^-3)":
                        The hole concentration in cm^-3.
                    - "μ_X (eV)":
                        Chemical potentials in eV, if ``append_chempots``
                        is ``True``.
                    - "Dopant (cm^-3)":
                        The effective arbitrary dopant concentration in cm^-3,
                        if set.

                Additional columns may include concentrations for specific
                defects and other relevant data.
        """
        # TODO: Allow matching of substring (e.g. "O_i" matching "O_i_C2" and "O_i_Ci") for fixed_defects
        #  and update docstrings ("...where the _total_ concentration of all defect entries whose names
        #  begin with ``defect_name`` will...") & tests accordingly, see _get_min_max_target_values
        # TODO: Related: Should allow just specifying an element for ``fixed_defects``, to allow the
        #  user to specify the known concentration of a dopant (or over/under-stoichiometry of an
        #  element? (but with unknown relative populations of different possible defects) -- not possible
        #  with current py-sc-fermi implementation, see code in _get_min_max_target_values()
        # TODO: In future the ``fixed_defects``, ``free_defects`` and ``fix_charge_states`` options may
        #  be added to the ``doped`` backend (in theory very simple to add)
        py_sc_fermi_required = fix_charge_states or free_defects or fixed_defects is not None
        if py_sc_fermi_required and self._DOS is None:
            self._activate_py_sc_fermi_backend(
                error_message="The `fix_charge_states`, `free_defects` and `fixed_defects` options are "
                "currently only supported for the py-sc-fermi backend"
            )

        # TODO: Add per-charge, per-site options like in ``doped`` ``DefectThermodynamics``
        if self.backend == "doped" and not py_sc_fermi_required:
            (
                fermi_level,
                electrons,
                holes,
                concentrations,
            ) = self.defect_thermodynamics.get_fermi_level_and_concentrations(  # type: ignore
                chempots=single_chempot_dict,
                el_refs=el_refs,
                annealing_temperature=annealing_temperature,
                quenched_temperature=quenched_temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                per_charge=False,
                skip_formatting=True,  # keep concentration values as floats
                site_competition=site_competition,
                delta_gap=delta_gap,
                **kwargs,
            )  # use already-set bulk dos

            # order in both cases is Defect, Concentration, Temperature, Fermi Level, e, h, Chempots
            new_columns = {
                "Annealing Temperature (K)": annealing_temperature,
                "Quenched Temperature (K)": quenched_temperature,
                "Fermi Level (eV wrt VBM)": fermi_level,
                "Electrons (cm^-3)": electrons,
                "Holes (cm^-3)": holes,
            }

            for column, value in new_columns.items():
                concentrations[column] = value

            # drop Dopant row, included as column instead
            concentrations = concentrations.drop("Dopant", errors="ignore")

        else:  # py-sc-fermi
            defect_system = self._generate_annealed_defect_system(
                annealing_temperature=annealing_temperature,
                single_chempot_dict=single_chempot_dict,
                el_refs=el_refs,
                quenched_temperature=quenched_temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                delta_gap=delta_gap,
                fixed_defects=fixed_defects,
                free_defects=free_defects,
                fix_charge_states=fix_charge_states,
                **kwargs,
            )

            with np.errstate(all="ignore"):
                conc_dict = defect_system.concentration_dict()

            # order is Defect, Concentration, Temperature, Fermi Level, e, h, Chempots
            data = [
                {
                    "Defect": k,
                    "Concentration (cm^-3)": v,
                    "Annealing Temperature (K)": annealing_temperature,
                    "Quenched Temperature (K)": quenched_temperature,
                    "Fermi Level (eV wrt VBM)": conc_dict["Fermi Energy"],
                    "Electrons (cm^-3)": conc_dict["n0"],
                    "Holes (cm^-3)": conc_dict["p0"],
                }
                for k, v in conc_dict.items()
                if k not in ["Fermi Energy", "n0", "p0", "Dopant"]
            ]
            concentrations = pd.DataFrame(data)
            concentrations = concentrations.set_index("Defect", drop=True)

        if append_chempots:
            for key, value in single_chempot_dict.items():
                concentrations[f"μ_{key} (eV)"] = value
        if effective_dopant_concentration is not None:
            concentrations["Dopant (cm^-3)"] = effective_dopant_concentration

        return concentrations

    def _check_temperature_settings(
        self,
        annealing_temperature: float | list[float] | None = None,
        temperature: float | list[float] = 300,
        quenched_temperature: float | list[float] = 300,
        range=False,
    ):
        """
        Helper function to check the user input temperature settings, and warn
        or raise errors as appropriate.
        """
        message = (
            "Both ``annealing_temperature`` and ``temperature`` were set, but they are "
            "mutually-exclusive options, with ``annealing_temperature`` employing the 'frozen defect "
            "approximation' (typically desired) and ``temperature`` assuming total equilibrium "
            "(see docstrings/tutorials)!"
        )
        if range:
            message = message.replace("`` ", "_range`` ")
        if annealing_temperature is not None and temperature != 300:  # both set by user
            raise ValueError(message)
        if annealing_temperature is None and quenched_temperature != 300:  # quenched set but no annealing
            raise ValueError(
                "Quenched temperature was set but no annealing temperature was given! Required for the "
                "'frozen defect approximation', see docstrings/tutorials."
            )

    def _solve(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: dict[str, float] | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        append_chempots: bool = True,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **user_kwargs,
    ) -> pd.DataFrame:
        PseudoSolveFunc: TypeAlias = Callable[..., pd.DataFrame]  # 'type aliases' for function signatures
        EquilibriumSolveFunc: TypeAlias = Callable[..., pd.DataFrame]
        solve_func: PseudoSolveFunc | EquilibriumSolveFunc = (
            self.pseudo_equilibrium_solve if annealing_temperature is not None else self.equilibrium_solve
        )
        kwargs = {
            "single_chempot_dict": single_chempot_dict,
            "el_refs": el_refs,
            "effective_dopant_concentration": effective_dopant_concentration,
            "fixed_defects": fixed_defects,
            "append_chempots": append_chempots,
            "site_competition": site_competition,
            **user_kwargs,
        }
        if annealing_temperature is not None:  # pseudo_equilibrium_solve
            kwargs.update(
                {
                    "annealing_temperature": annealing_temperature,
                    "quenched_temperature": quenched_temperature,
                    "delta_gap": delta_gap,
                    "free_defects": free_defects,  # type: ignore
                    "fix_charge_states": fix_charge_states,
                }
            )
        else:  # equilibrium_solve
            kwargs["temperature"] = temperature
            _ = kwargs.pop("verbose", None)  # not relevant for equilibrium solve

        return solve_func(**kwargs)

    # TODO: Should have a general ``scan()`` function, which takes in all variables, determines which
    #  are ranges/iterables to scan over (i.e. multi-dimension), and then calls the appropriate
    #  ``scan_...``  function(s) to scan over
    #  each N-dimensions, concatenates and returns the result
    # TODO: Related, can implement joblib Parallel / multiprocessing processing with smart batch size
    #  choices to make this much faster as well, if wanted. Smartest way would be to refactor all the
    #  internal pd.concat loops over self._solve to use a separate calc function, which internally
    #  determines parallelisation speedup and batch size based on grid size, and acts accordingly
    def scan_temperature(
        self,
        annealing_temperature_range: float | list[float] | None = None,
        quenched_temperature_range: float | list[float] = 300,
        temperature_range: float | list[float] = 300,
        chempots: dict[str, float] | None = None,
        limit: str | None = None,
        el_refs: dict[str, float] | None = None,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Scan over a range of temperatures and solve for the defect
        concentrations, carrier concentrations, and Fermi level at each
        temperature / annealing & quenched temperature pair.

        If ``annealing_temperature_range`` (and ``quenched_temperature_range``;
        just 300 K by default) are specified, then the frozen defect
        approximation is employed, whereby total defect concentrations are
        calculated at the elevated annealing temperature, then fixed at these
        values (unless ``free_defects`` or ``fix_charge_states`` are specified)
        and the Fermi level and relative charge state populations are
        recalculated at the quenched temperature. Otherwise, if only
        ``temperature_range`` is specified, then the Fermi level and
        defect/carrier concentrations are calculated assuming thermodynamic
        equilibrium at each temperature.

        See ``(pseudo_)equilibrium_solve`` docstrings for more details.

        Args:
            annealing_temperature_range (Optional[Union[float, list[float]]]):
                Temperature range in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is ``None`` (uses ``temperature_range`` under
                thermodynamic equilibrium).
            quenched_temperature_range (Union[float, list[float]]):
                Temperature, or range of temperatures, in Kelvin at which to
                calculate the self-consistent (constrained equilibrium) Fermi
                level and carrier concentrations, given the fixed total
                concentrations, which should correspond to operating
                temperature of the material (typically room temperature).
                Default is just 300 K.
            temperature_range (Union[float, list[float]]):
                Temperature range to solve over, under thermodynamic
                equilibrium (if ``annealing_temperature_range`` is not
                specified). Defaults to just 300 K.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus concentrations and Fermi
                level). If ``None`` (default), will use
                ``self.defect_thermodynamics.chempots`` (= 0 for all
                chemical potentials by default).
                This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` or
                ``FermiSolver.defect_thermodynamics.chempots = ...`` (with the
                same input options) to set the default chemical potentials for
                all calculations.
            limit (str):
                The chemical potential limit for which to calculate the Fermi
                level positions and defect/carrier concentrations. Can be:

                - ``None``, if ``chempots`` corresponds to a single chemical
                  potential limit -- otherwise will use the first chemical
                  potential limit in the ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in ``(self.defect_thermodynamics.)chempots["limits"]``.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` or
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` (with the
                same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to fix the concentrations of
                specific defects regardless of the chemical potentials, or
                anneal-quench procedure (e.g. to simulate the effect of a fixed
                impurity concentration). Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names).
                Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                DataFrame containing defect and carrier concentrations.
        """
        self._check_temperature_settings(
            annealing_temperature_range, temperature_range, quenched_temperature_range, range=True
        )
        # Ensure temperature ranges are lists:
        temperature_list = _ensure_list(temperature_range)
        annealing_temperature_list = _ensure_list(annealing_temperature_range)  # returns None if None
        quenched_temperature_list = _ensure_list(quenched_temperature_range)

        single_chempot_dict, el_refs = self._get_single_chempot_dict(limit, chempots, el_refs)

        temp_args = product(  # type: ignore
            *(
                i if isinstance(i, Iterable) else [i]
                for i in [annealing_temperature_list, quenched_temperature_list, temperature_list]
            )
        )
        return pd.concat(
            [
                self._solve(
                    single_chempot_dict=single_chempot_dict,
                    el_refs=el_refs,
                    annealing_temperature=annealing_temperature,
                    quenched_temperature=quenched_temperature,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    delta_gap=delta_gap,
                    fixed_defects=fixed_defects,
                    free_defects=free_defects,
                    fix_charge_states=fix_charge_states,
                    site_competition=site_competition,
                    **kwargs,
                )
                for annealing_temperature, quenched_temperature, temperature in tqdm(list(temp_args))
            ]
        )

    def scan_dopant_concentration(
        self,
        effective_dopant_concentration_range: float | list[float],
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        chempots: dict[str, float] | None = None,
        limit: str | None = None,
        el_refs: dict[str, float] | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Calculate the defect concentrations under a range of effective
        (hypothetical) dopant concentrations.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        See ``(pseudo_)equilibrium_solve`` docstrings for more details.

        Args:
            effective_dopant_concentration_range (Union[float, list[float]]):
                The range of effective dopant concentrations to solve over.
                This can be a single value or a list of values representing
                different concentrations. These are taken as fixed
                concentrations (in cm^-3) of an arbitrary dopant or impurity
                in the material, included in the charge neutrality condition
                to analyse the Fermi level and doping response under
                hypothetical doping conditions.
                Positive values correspond to donor doping, while negative
                values correspond to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is ``None`` (uses ``temperature`` under thermodynamic
                equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Defaults to 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies (and thus concentrations and Fermi
                level). If ``None`` (default), will use
                ``self.defect_thermodynamics.chempots`` (= 0 for all
                chemical potentials by default).
                This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can
                set the ``el_refs`` option with the DFT reference energies of
                the elemental phases, in which case it is the formal chemical
                potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical
                potentials should be given.

                One can also set ``DefectThermodynamics.chempots = ...`` or
                ``FermiSolver.defect_thermodynamics.chempots = ...`` (with the
                same input options) to set the default chemical potentials for
                all calculations.
            limit (str):
                The chemical potential limit for which to calculate the Fermi
                level positions and defect/carrier concentrations. Can be:

                - ``None``, if ``chempots`` corresponds to a single chemical
                  potential limit -- otherwise will use the first chemical
                  potential limit in the ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where ``X`` is an element in the
                  system, in which case the most X-rich/poor limit will be used
                  (e.g. "Li-rich").
                - A key in ``(self.defect_thermodynamics.)chempots["limits"]``.

                The latter two options can only be used if ``chempots`` is in
                the ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (see tutorials).

                One can also set ``DefectThermodynamics.el_refs = ...`` or
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` (with the
                same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to fix the concentrations of
                specific defects regardless of the chemical potentials, or
                anneal-quench procedure (e.g. to simulate the effect of a fixed
                impurity concentration). Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names).
                Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the defect and carrier
                concentrations for each effective dopant concentration. Each
                row represents the concentrations for a different dopant
                concentration.
        """
        self._check_temperature_settings(annealing_temperature, temperature, quenched_temperature)
        effective_dopant_concentration_list = _ensure_list(effective_dopant_concentration_range)

        single_chempot_dict, el_refs = self._get_single_chempot_dict(limit, chempots, el_refs)

        return pd.concat(
            [
                self._solve(
                    single_chempot_dict=single_chempot_dict,
                    el_refs=el_refs,
                    annealing_temperature=annealing_temperature,
                    quenched_temperature=quenched_temperature,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    delta_gap=delta_gap,
                    fixed_defects=fixed_defects,
                    free_defects=free_defects,
                    fix_charge_states=fix_charge_states,
                    site_competition=site_competition,
                    **kwargs,
                )
                for effective_dopant_concentration in tqdm(effective_dopant_concentration_list)
            ]
        )

    def interpolate_chempots(
        self,
        n_points: int = 10,
        chempots: list[dict] | dict | None = None,
        limits: list[str] | None = None,
        el_refs: dict[str, float] | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Interpolate between two sets of chemical potentials and solve for the
        defect concentrations and Fermi level at each interpolated point.

        Chemical potentials can be interpolated between two sets of chemical
        potential dictionaries/values, or between two specified limits.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        See ``(pseudo_)equilibrium_solve`` docstrings for more details.

        Args:
            n_points (int):
                The number of points to generate between chemical potential
                end points. Defaults to 10.
            chempots (Optional[list[dict]]):
                The chemical potentials to interpolate between. This can be
                either a list containing two dictionaries, each representing
                a set of chemical potentials for a single limit (in the format:
                ``{element symbol: chemical potential}``) to interpolate
                between, or can be a single chemical potentials dictionary in
                the ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``) -- in which
                case ``limits`` must be specified to pick the end-points to
                interpolate between.

                If ``None`` (default), will use
                ``self.defect_thermodynamics.chempots``. Note that you can also
                set ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input
                options) to set the default chemical potentials for all
                calculations.

                If manually specifying chemical potentials with a list of two
                dictionaries, you can also set the ``el_refs`` option with the
                DFT reference energies of the elemental phases if desired, in
                which case it is the formal chemical potentials (i.e. relative
                to the elemental references) that should be given here,
                otherwise the absolute (DFT) chemical potentials should be
                given.
            limits (Optional[list[str]]):
                The chemical potential limits to interpolate between, as a list
                containing two strings. Each string should be in the format
                ``"X-rich"/"X-poor"``, where X is an element in the system, or
                a key in ``(self.defect_thermodynamics.)chempots["limits"]``.

                If not provided, ``chempots`` must be specified as a list of
                two single chemical potential dictionaries for single limits,
                or must be a binary system with only 2 limits in ``chempots``.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``[{element symbol: chemical potential}, ...]``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``).

                One can also set ``DefectThermodynamics.el_refs = ...`` or
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` (with the
                same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is ``None`` (uses ``temperature`` under thermodynamic
                equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Defaults to 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to fix the concentrations of
                specific defects regardless of the chemical potentials, or
                anneal-quench procedure (e.g. to simulate the effect of a fixed
                impurity concentration). Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names). Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the defect and carrier
                concentrations for each interpolated set of chemical
                potentials. Each row represents the concentrations for a
                different interpolated point.
        """
        self._check_temperature_settings(annealing_temperature, temperature, quenched_temperature)

        if isinstance(chempots, list):  # should be two single chempot dictionaries
            if len(chempots) != 2:
                raise ValueError(
                    f"If `chempots` is a list, it must contain two dictionaries representing the starting "
                    f"and ending chemical potentials. The provided list has {len(chempots)} entries!"
                )
            single_chempot_dict_1, single_chempot_dict_2 = chempots

        else:  # should be a dictionary in the ``doped`` format or ``None``:
            chempots, el_refs = self._get_and_check_thermo_chempots(chempots, el_refs)

            if limits is None or len(limits) != 2:
                if len(chempots["limits"]) == 2:
                    limits = list(chempots["limits"].keys())
                else:
                    raise ValueError(
                        f"If `chempots` is not provided as a list, then `limits` must be a list "
                        f"containing two strings representing the chemical potential limits to "
                        f"interpolate between. The provided `limits` is: {limits}."
                    )

            single_chempot_dict_1, el_refs = self._get_single_chempot_dict(limits[0], chempots, el_refs)
            single_chempot_dict_2, el_refs = self._get_single_chempot_dict(limits[1], chempots, el_refs)

        interpolated_chempots = get_interpolated_chempots(
            single_chempot_dict_1, single_chempot_dict_2, n_points
        )

        return self.scan_chempots(
            interpolated_chempots,
            el_refs=el_refs,
            annealing_temperature=annealing_temperature,
            quenched_temperature=quenched_temperature,
            temperature=temperature,
            effective_dopant_concentration=effective_dopant_concentration,
            delta_gap=delta_gap,
            fixed_defects=fixed_defects,
            free_defects=free_defects,
            fix_charge_states=fix_charge_states,
            site_competition=site_competition,
            **kwargs,
        )

    def scan_chempots(
        self,
        chempots: list[dict[str, float]] | dict[str, dict] | None = None,
        limits: list[str] | None = None,
        el_refs: dict[str, float] | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Scan over a range of chemical potentials and solve for the defect
        concentrations and Fermi level at each set of chemical potentials.

        Note that this function only solves for the Fermi level and
        defect/carrier concentrations `at the given chemical potentials` (and
        not at any points between them), whereas
        ``scan_chemical_potential_grid``, ``interpolate_chempots`` and
        ``optimise`` scan over the grid/points between chemical potential
        limits, which may be desired.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        See ``(pseudo_)equilibrium_solve`` docstrings for more details.

        Args:
            chempots (Optional[Union[list[dict], dict]]):
                The chemical potentials to scan over. This can be either a list
                containing dictionaries of a set of chemical potentials for a
                `single` limit (in the format:
                ``{element symbol: chemical potential}``), or can be a single
                chemical potentials dictionary in the ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``) -- in which
                case ``limits`` can be specified to pick the chemical potential
                limits to scan over (otherwise scans over all limits in the
                ``chempots`` dictionary).

                By default, will use ``self.defect_thermodynamics.chempots``
                (if ``None``). Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input
                options) to set the default chemical potentials for all
                calculations.

                If manually specifying chemical potentials with a list of
                dictionaries, you can also set the ``el_refs`` option with the
                DFT reference energies of the elemental phases if desired, in
                which case it is the formal chemical potentials (i.e. relative
                to the elemental references) that should be given here,
                otherwise the absolute (DFT) chemical potentials should be
                given.
            limits (Optional[list[str]]):
                The chemical potential limits to scan over, as a list of
                strings, if ``chempots`` was provided / is present in the
                ``doped`` format. Each string should be in the format
                ``"X-rich"/"X-poor"``, where X is an element in the system, or
                a key in ``(self.defect_thermodynamics.)chempots["limits"]``.

                If ``None`` (default) and ``chempots`` is in the ``doped``
                format (rather than a list of single chemical potential
                limits), scans over all limits in the ``chempots`` dictionary.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``[{element symbol: chemical potential}, ...]``).
                Unnecessary if ``chempots`` is provided/present in format
                generated by ``doped`` (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``).

                One can also set ``DefectThermodynamics.el_refs = ...`` or
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` (with the
                same input options) to set the default elemental reference
                energies for all calculations.
                (Default: None)
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is ``None`` (uses ``temperature`` under thermodynamic
                equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Defaults to 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to simulate the effect of a fixed
                impurity concentration. Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names). Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing defect and carrier concentrations
                for each set of chemical potentials. Each row corresponds to a
                different set of chemical potentials.
        """
        chempots = chempots if chempots is not None else self.defect_thermodynamics.chempots
        self._check_temperature_settings(annealing_temperature, temperature, quenched_temperature)

        if isinstance(chempots, dict):  # should be a dictionary in the ``doped`` format or ``None``:
            chempots, el_refs = self._get_and_check_thermo_chempots(chempots, el_refs)
            chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)

            if limits is None:
                limits = list(chempots["limits_wrt_el_refs"].keys())
            elif not isinstance(limits, list):
                raise ValueError(
                    "`limits` must be either a list of limits (as strings) or `None` for `scan_chempots`!"
                )

            chempots = [self._get_single_chempot_dict(limit, chempots, el_refs)[0] for limit in limits]

        return pd.concat(
            [
                self._solve(
                    single_chempot_dict=single_chempot_dict,
                    el_refs=el_refs,
                    annealing_temperature=annealing_temperature,
                    quenched_temperature=quenched_temperature,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    delta_gap=delta_gap,
                    fixed_defects=fixed_defects,
                    free_defects=free_defects,
                    fix_charge_states=fix_charge_states,
                    site_competition=site_competition,
                    **kwargs,
                )
                for single_chempot_dict in tqdm(chempots)
            ]
        )

    def scan_chemical_potential_grid(
        self,
        chempots: dict | None = None,
        n_points: int = 10,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Given a ``doped``-formatted chemical potential dictionary, generate a
        ``ChemicalPotentialGrid`` object and calculate the Fermi level
        positions and defect/carrier concentrations at the grid points.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        See ``(pseudo_)equilibrium_solve`` docstrings for more details.

        Args:
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the
                ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``) -- the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials).

                By default, will use ``self.defect_thermodynamics.chempots``
                (if ``None``). Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input
                options) to set the default chemical potentials for all
                calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update
                the elemental reference energies for any reason.
            n_points (int):
                The number of points to generate along each axis of the grid.
                The actual number of grid points may be less, as points outside
                the convex hull are excluded. The higher the dimension of your
                chemical system (i.e. number of elements), the more sensitive
                the runtime will be to the choice of ``n_points``.
                Default is 10.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is ``None`` (uses ``temperature`` under thermodynamic
                equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Defaults to 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to simulate the effect of a fixed
                impurity concentration. Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names). Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the Fermi level solutions at the
                grid points, based on the provided chemical potentials and
                conditions.
        """
        self._check_temperature_settings(annealing_temperature, temperature, quenched_temperature)
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)
        grid = ChemicalPotentialGrid(chempots).get_grid(n_points)
        chempot_dict_list = [
            {k.split("_")[1].split()[0]: v for k, v in chempot_series.to_dict().items()}
            for _idx, chempot_series in grid.iterrows()
        ]
        return self.scan_chempots(
            chempots=chempot_dict_list,
            el_refs=el_refs,
            annealing_temperature=annealing_temperature,
            quenched_temperature=quenched_temperature,
            temperature=temperature,
            effective_dopant_concentration=effective_dopant_concentration,
            delta_gap=delta_gap,
            fixed_defects=fixed_defects,
            free_defects=free_defects,
            fix_charge_states=fix_charge_states,
            site_competition=site_competition,
            **kwargs,
        )

    def _parse_and_check_grid_like_chempots(self, chempots: dict | None = None) -> tuple[dict, dict]:
        r"""
        Parse a dictionary of chemical potentials for the chemical potential
        scanning functions, checking that it is in the correct format.

        Args:
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the
                ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``) -- the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials).

                By default, will use ``self.defect_thermodynamics.chempots``
                (if ``None``).

        Returns:
            dict:
                The chemical potentials in the correct format, along with the
                elemental reference energies.
        """
        chempots, el_refs = self._get_and_check_thermo_chempots(chempots)
        if len(chempots["limits"]) == 1:
            raise ValueError(
                "Only one chemical potential limit is present in "
                "`chempots`/`self.defect_thermodynamics.chempots`, which makes no sense for a chemical "
                "potential grid scan (with `scan_chemical_potential_grid`/`optimise`/`scan_chempots`)!"
            )

        return chempots, el_refs

    def optimise(
        self,
        target: str,
        min_or_max: str = "max",
        chempots: dict | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        tolerance: float = 0.01,
        n_points: int = 10,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Search for the chemical potentials that minimise or maximise a target
        variable, such as electron concentration, within a specified tolerance.

        See ``target`` argument description below for valid choices. This
        function iterates over a grid of chemical potentials and "zooms in" on
        the chemical potential that either minimises or maximises the target
        variable. The process continues until the `relative` change in the
        target variable is less than the specified tolerance.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        See ``(pseudo_)equilibrium_solve`` docstrings for more details.

        Args:
            target (str):
                The target variable to minimise or maximise, e.g., "Electrons
                (cm^-3)", "Te_i", "Fermi Level (eV wrt VBM)" etc. Valid
                ``target`` values are column names (or substrings), such as
                "Electrons (cm^-3)", "Holes (cm^-3)",
                "Fermi Level (eV wrt VBM)", "μ_X (eV)", etc., or defect names
                (without charge states), such as "v_O", "Te_i", etc.
                If a full defect name is given (e.g. ``Te_i_Td_Te2.83``) then
                the concentration of that defect will be used as the target
                variable. If a defect name substring is given instead (e.g.
                ``Te_i``), then the target variable will be the summed
                concentration of all defects with that substring in their name
                (e.g. ``Te_i_Td_Te2.83``, ``Te_i_C3v`` etc).
            min_or_max (str):
                Specify whether to "minimise" ("min") or "maximise" ("max";
                default) the target variable.
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the
                ``doped`` format (i.e.
                ``{"limits": [{'limit': [chempot_dict]}], ...}``) -- the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials).

                By default, will use ``self.defect_thermodynamics.chempots``
                (if ``None``). Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input
                options) to set the default chemical potentials for all
                calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update
                the elemental reference energies for any reason.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing /
                synthesis of the material (at which we assume equilibrium
                defect concentrations) within the frozen defect approach.
                Default is ``None`` (uses ``temperature`` under thermodynamic
                equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier
                concentrations, given the fixed total concentrations, which
                should correspond to operating temperature of the material
                (typically room temperature). Defaults to 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            tolerance (float):
                The convergence criterion for the target variable. The search
                stops when the relative change in the target value is less than
                this value. Defaults to ``0.01``.
            n_points (int):
                The number of points to generate along each axis of the
                chemical potentials grid for each iteration of the search. The
                higher the dimension of your chemical system (i.e. number of
                elements), the more sensitive the runtime will be to the choice
                of ``n_points``. Defaults to ``10``.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to simulate the effect of a fixed
                impurity concentration. Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names). Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``) upon quenching.
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            site_competition (bool | str):
                If ``True`` (default), uses the updated Fermi-Dirac-like
                formula for defect concentrations, which accounts for defect
                site competition at high concentrations (see Kasamatsu et al.
                (10.1016/j.ssi.2010.11.022) appendix for derivation -- updated
                here to additionally account for configurational degeneracies
                ``g`` (see https://doi.org/10.1039/D3CS00432E)), which gives
                the following defect concentration equation:
                ``N_X = N*[g*exp(-E/kT) / (1 + sum(g_i*exp(-E_i/kT)))]`` where
                ``i`` runs over all defects which occupy the same site.
                Note that this option is only supported for the ``doped``
                backend. If ``False`` (or using the ``py-sc-fermi`` backend),
                uses the standard dilute limit approximation.

                Alternatively ``site_competition`` can be set to a string
                (``"verbose"``), which will give the same behaviour as ``True``
                as well as including the lattice site indices (used to
                determine which defects will compete for the same sites) in the
                output ``DataFrame``. Note that while large ``dist_tol``\s are
                often preferable for plotting (to condense the defect formation
                energy lines), this is often not ideal for determining site
                competition in concentration analyses as it can lead to
                unrealistically-large clusters.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the results of the minimisation or
                maximisation process, including the optimal chemical potentials
                and the corresponding values of the target variable.

        Raises:
            ValueError:
                If neither ``chempots`` nor ``self.chempots`` is provided, or
                if ``min_or_max`` is not ``"minimise"/"min"`` or
                ``"maximise"/"max"``.
        """
        self._check_temperature_settings(annealing_temperature, temperature, quenched_temperature)
        # Determine the dimensionality of the chemical potential space, and call appropriate method
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)
        # TODO: Add option of just specifying an element, to min/max its summed defect concentrations
        # TODO: When per-charge option added, test setting target to a defect species (with charge)

        optimise_func = self._optimise_line if len(el_refs) == 2 else self._optimise_grid
        return optimise_func(
            target=target,
            min_or_max=min_or_max,
            chempots=chempots,
            annealing_temperature=annealing_temperature,
            quenched_temperature=quenched_temperature,
            temperature=temperature,
            tolerance=tolerance,
            n_points=n_points,
            effective_dopant_concentration=effective_dopant_concentration,
            delta_gap=delta_gap,
            fixed_defects=fixed_defects,
            free_defects=free_defects,
            fix_charge_states=fix_charge_states,
            site_competition=site_competition,
            **kwargs,
        )

    def _optimise_line(
        self,
        target: str,
        min_or_max: str = "max",
        chempots: dict | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        tolerance: float = 0.01,
        n_points: int = 10,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        ``optimise`` function for 1D chemical potential spaces (i.e. binary
        systems).

        See the main ``optimise`` docstring for more details.
        """
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)

        # Assuming 1D space, focus on one label and get the Rich/Poor limits:
        unformatted_chempots_labels = list(el_refs.keys())
        rich = self._get_single_chempot_dict(f"{unformatted_chempots_labels[0]}-rich")
        poor = self._get_single_chempot_dict(f"{unformatted_chempots_labels[0]}-poor")
        chempots_dict_list = get_interpolated_chempots(rich[0], poor[0], n_points)
        delta_gap_verbose = kwargs.pop("verbose", False)  # don't interfere with other verbose option here

        previous_value = None
        while True:  # Calculate results based on the given temperature conditions
            target_df, current_value, target_chempot, converged = self._scan_chempots_and_compare(
                target=target,
                min_or_max=min_or_max,
                previous_value=previous_value,
                tolerance=tolerance,
                chempots=chempots_dict_list,
                el_refs=el_refs,
                annealing_temperature=annealing_temperature,
                quenched_temperature=quenched_temperature,
                temperature=temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                delta_gap=delta_gap,
                fixed_defects=fixed_defects,
                free_defects=free_defects,
                fix_charge_states=fix_charge_states,
                verbose=previous_value is None,  # first iteration, print info on target cols/rows
                site_competition=site_competition,
                delta_gap_verbose=delta_gap_verbose,
                **kwargs,
            )
            if converged:
                break

            previous_value = current_value  # otherwise update

            # get midpoints of chempots_dict_list and target_chempot, and use these:
            midpoint_chempots = [
                {
                    k.split("_")[1].split()[0]: (chempot_dict[k.split("_")[1].split()[0]] + v) / 2
                    for k, v in target_chempot.iloc[0].items()
                }
                for chempot_dict in [chempots_dict_list[0], chempots_dict_list[-1]]
            ]
            # Note that this is a 'safe' option for zooming in the search grid. If it was a linear
            # function, then we could just take the closest vertices around ``target_chempot`` and use
            # these, but we know that chempots & temperature & other constraints -> defect concentrations
            # can be highly non-linear (e.g. CdTe concentrations in SK thesis, 10.1016/j.joule.2024.05.004,
            # 10.1002/smll.202102429, 10.1021/acsenergylett.4c02722), so best to use this safe (but slower)
            # approach to ensure we don't miss the true minimum/maximum. Same in both min_max functions.
            chempots_dict_list = get_interpolated_chempots(
                chempot_start=midpoint_chempots[0],
                chempot_end=midpoint_chempots[1],
                n_points=n_points,
            )

        return target_df

    def _scan_chempots_and_compare(  # noqa: D417
        self,
        target: str,
        min_or_max: str,
        previous_value: float | None = None,
        tolerance: float = 0.01,
        chempots: list[dict[str, float]] | dict[str, dict] | None = None,
        el_refs: dict[str, float] | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        verbose: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ):
        """
        Convenience method for use in the ``_optimise_...`` methods, which
        scans over a set of chemical potentials and compares the target value
        to a previous value, returning the new target dataframe, value and
        corresponding chemical potentials.

        Args:
            previous_value (float):
                The previous value of the target variable.
            verbose (bool):
                Whether to print information on identified target
                rows/columns.
            *args:
                All other arguments are the same as for the ``optimise``
                method, see its docstring for more details.

        Returns:
            target_df (pd.DataFrame):
                A ``DataFrame`` containing the current results of the
                optimisation, including the optimal chemical potentials and
                corresponding values of the target variable.
            current_value (float):
                The current (updated) value of the target variable.
            target_chempot (pd.DataFrame):
                The chemical potentials corresponding to the current value.
            converged (bool):
                Whether the search has converged to within ``tolerance``.
        """
        if "delta_gap_verbose" in kwargs:
            kwargs["verbose"] = kwargs.pop("delta_gap_verbose")
        results_df = self.scan_chempots(
            chempots=chempots,
            el_refs=el_refs,
            annealing_temperature=annealing_temperature,
            quenched_temperature=quenched_temperature,
            temperature=temperature,
            effective_dopant_concentration=effective_dopant_concentration,
            delta_gap=delta_gap,
            fixed_defects=fixed_defects,
            free_defects=free_defects,
            fix_charge_states=fix_charge_states,
            site_competition=site_competition,
            **kwargs,
        )

        target_df, current_value, target_chempot = _get_min_max_target_values(
            results_df, target, min_or_max, verbose=verbose
        )
        converged = (  # Check if the change in the target value is less than the tolerance
            previous_value is not None
            and (
                current_value == previous_value
                or abs((current_value - previous_value) / (previous_value or current_value)) < tolerance
            )
        )  # divide by (previous_value or current_value) to avoid division by zero

        return target_df, current_value, target_chempot, converged

    def _optimise_grid(
        self,
        target: str,
        min_or_max: str = "max",
        chempots: dict | None = None,
        annealing_temperature: float | None = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        tolerance: float = 0.01,
        n_points: int = 10,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        site_competition: bool | str = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        ``optimise`` function for >=2D chemical potential spaces (i.e.
        elementary/non-binary systems).

        See the main ``optimise`` docstring for more details.
        """
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)
        chempots_grid = ChemicalPotentialGrid(chempots)
        delta_gap_verbose = kwargs.pop("verbose", False)  # don't interfere with other verbose option here

        previous_value = None
        while True:
            chempots_dict_list = [
                {k.split("_")[1].split()[0]: v for k, v in chempot_series.to_dict().items()}
                for _idx, chempot_series in chempots_grid.get_grid(n_points).iterrows()
            ]
            target_df, current_value, target_chempot, converged = self._scan_chempots_and_compare(
                target=target,
                min_or_max=min_or_max,
                previous_value=previous_value,
                tolerance=tolerance,
                chempots=chempots_dict_list,
                el_refs=el_refs,
                annealing_temperature=annealing_temperature,
                quenched_temperature=quenched_temperature,
                temperature=temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                delta_gap=delta_gap,
                fixed_defects=fixed_defects,
                free_defects=free_defects,
                fix_charge_states=fix_charge_states,
                verbose=previous_value is None,  # first iteration, print info on target cols/rows
                site_competition=site_competition,
                delta_gap_verbose=delta_gap_verbose,
                **kwargs,
            )
            if converged:
                break

            previous_value = current_value  # otherwise update

            new_vertices_df = (
                chempots_grid.vertices + target_chempot.iloc[0]
            ) / 2  # 1 row - target_chempot
            # Note that this is a 'safe' option for zooming in the search grid. If it was a linear
            # function, then we could just take the closest vertices around ``target_chempot`` and use
            # these, but we know that chempots & temperature & other constraints -> defect concentrations
            # can be highly non-linear (e.g. CdTe concentrations in SK thesis, 10.1016/j.joule.2024.05.004,
            # 10.1002/smll.202102429, 10.1021/acsenergylett.4c02722), so best to use this safe (but slower)
            # approach to ensure we don't miss the true minimum/maximum. Same in both min_max functions.

            # Generate a new grid around target_chempot which doesn't go outside the starting grid bounds:
            chempots_grid = ChemicalPotentialGrid(new_vertices_df.to_dict("index"))

        return target_df

    def _generate_dopant_for_py_sc_fermi(self, effective_dopant_concentration: float) -> "DefectSpecies":
        """
        Generate a dopant defect charge state object, for ``py-sc-fermi``
        functions.

        This method creates a defect charge state object representing an
        arbitrary dopant or impurity in the material, used to include in the
        charge neutrality condition and analyse the Fermi level/doping response
        under hypothetical doping conditions.

        Args:
            effective_dopant_concentration (float):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.

        Returns:
            DefectSpecies:
                An instance of the ``DefectSpecies`` class, representing the
                generated dopant with the specified charge state and
                concentration.

        Raises:
            ValueError:
                If ``effective_dopant_concentration`` is zero or if there is an
                issue with generating the dopant.
        """
        self._check_required_backend_and_error("py-sc-fermi")
        assert self._DefectChargeState
        assert self._DefectSpecies
        dopant = self._DefectChargeState(
            charge=np.sign(effective_dopant_concentration),
            fixed_concentration=abs(effective_dopant_concentration) / 1e24 * self.volume,
            degeneracy=1,
        )
        return self._DefectSpecies(
            nsites=1, charge_states={np.sign(effective_dopant_concentration): dopant}, name="Dopant"
        )

    def _generate_defect_system(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: dict[str, float] | None = None,
        temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        fixed_defects: dict[str, float] | None = None,
    ) -> "DefectSystem":
        """
        Generates a ``py-sc-fermi`` ``DefectSystem`` object from
        ``self.defect_thermodynamics`` and a set of chemical potentials.

        This method constructs a ``DefectSystem`` object, which encompasses all
        relevant defect species and their properties under the given
        conditions, including temperature, chemical potentials, and an optional
        dopant concentration.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the
                equilibrium Fermi level position and defect/carrier
                concentrations. Here, this should be a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in
                ``self.defect_thermodynamics.el_refs`` then it is the `formal`
                chemical potentials (i.e. relative to the elemental reference
                energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if
                ``chempots`` was provided to ``self.defect_thermodynamics`` in
                the format generated by ``doped``).
                (Default: None)
            temperature (float):
                The temperature in Kelvin at which to perform the calculations.
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to simulate the effect of a fixed
                impurity concentration. Defaults to ``None``.

        Returns:
            DefectSystem:
                An initialized ``DefectSystem`` object, containing the defect
                species with their charge states, formation energies, and
                degeneracies, as well as the density of states (DOS), volume,
                and temperature of the system.
        """
        self._check_required_backend_and_error("py-sc-fermi")
        assert self._DefectSpecies
        assert self._DefectSystem
        single_chempot_dict, el_refs = self.defect_thermodynamics._get_chempots(
            single_chempot_dict, el_refs
        )  # returns self.defect_thermodynamics.chempots/self.defect_thermodynamics.el_refs if None
        dft_chempots = _get_dft_chempots(single_chempot_dict, el_refs)

        defect_species = []  # dicts of: {"charge_states": {...}, "nsites": X, "name": label}
        for label, entry_list in self.defect_thermodynamics.all_entries.items():
            defect_species_dict = {
                "charge_states": {},
                "nsites": entry_list[0].defect.multiplicity / self.multiplicity_scaling,
                "name": label,
            }
            for entry in entry_list:
                formation_energy = self.defect_thermodynamics.get_formation_energy(
                    entry, chempots=dft_chempots, fermi_level=0
                )
                degeneracy_factor = (
                    np.prod(list(entry.degeneracy_factors.values())) if entry.degeneracy_factors else 1
                )
                # py-sc-fermi assumes the same multiplicity (nsites) for all defect species / charge
                # states of a given grouped defect, but this is not necessarily the case for
                # interstitials (e.g. Te_i_Td_Te2.83_a), so we account for this in the degeneracy
                # factors here:
                degeneracy_factor *= entry.defect.multiplicity / entry_list[0].defect.multiplicity
                defect_species_dict["charge_states"][entry.charge_state] = {
                    "charge": entry.charge_state,
                    "energy": formation_energy,
                    "degeneracy": degeneracy_factor,
                }
            defect_species.append(defect_species_dict)

        all_defect_species = [self._DefectSpecies.from_dict(subdict) for subdict in defect_species]
        if effective_dopant_concentration is not None:
            all_defect_species.append(
                self._generate_dopant_for_py_sc_fermi(effective_dopant_concentration)
            )

        defect_system = self._DefectSystem(
            defect_species=all_defect_species,
            dos=self.py_sc_fermi_dos,
            volume=self.volume,
            temperature=temperature,
            convergence_tolerance=1e-20,
        )
        self._fix_defect_concentrations(defect_system, fixed_defects)

        return defect_system

    def _fix_defect_concentrations(
        self,
        defect_system: "DefectSystem",
        fixed_defects: dict[str, float] | None = None,
        fixed_concs: dict[str, float] | None = None,
    ) -> None:
        """
        Utility method to fix the concentrations of defects specified by
        ``fix_defects`` in the ``py-sc-fermi`` ``defect_system``.

        This method applies the concentration constraints to the
        ``defect_system.defect_species`` in place.

        Args:
            defect_system (DefectSystem):
                ``py-sc-fermi`` ``DefectSystem`` for which to fix the
                concentrations of defects (in ``defect_system.defect_species``)
                according to the ``fixed_defects`` input.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix regardless of
                chemical potentials / temperature / Fermi level, in the format:
                ``{defect_name: concentration}``, where ``defect_name`` is the
                name of a defect entry without (e.g. ``"v_O"``) or with (e.g.
                ``"v_O_+2"``) the charge state; which will then fix either the
                total concentration of that defect or only the concentration
                for the specified charge state. Concentrations should be given
                in cm^-3. This can be used to simulate the effect of a fixed
                impurity concentration. Defaults to ``None``.
            fixed_concs (dict):
                Dictionary of total concentrations of defects which, if
                provided, will be compared to input concentration constraints
                (``fixed_defects``) and a warning will be thrown if
                ``fixed_concs[defect_name_without_charge]`` is larger than
                ``fixed_defects[defect_name_with_charge]``.
                Default is ``None``.
        """
        if fixed_defects is None:
            return

        for k, v in fixed_defects.items():
            if k.split("_")[-1].strip("+-").isdigit():
                defect_name_wout_charge, q_str = k.rsplit("_", 1)
                q = int(q_str)
                defect_system.defect_species_by_name(defect_name_wout_charge).charge_states[
                    q
                ].fix_concentration(v / 1e24 * self.volume)

                if fixed_concs and v > fixed_concs[defect_name_wout_charge] * 1.001:  # small noise tol
                    warnings.warn(
                        f"Fixed concentration of {k} ({v}) is higher than the total concentration of "
                        f"({fixed_concs[defect_name_wout_charge]}) at the annealing temperature. "
                        f"Adjusting the total concentration of {defect_name_wout_charge} to {v}. Check "
                        f"that this is the behaviour you expect."
                    )
                    defect_system.defect_species_by_name(defect_name_wout_charge).fix_concentration(
                        v / 1e24 * self.volume
                    )
            else:
                defect_system.defect_species_by_name(k).fix_concentration(v / 1e24 * self.volume)

    def _generate_annealed_defect_system(
        self,
        annealing_temperature: float,
        single_chempot_dict: dict[str, float],
        el_refs: dict[str, float] | None = None,
        quenched_temperature: float = 300,
        effective_dopant_concentration: float | None = None,
        delta_gap: float | Callable = 0.0,
        fixed_defects: dict[str, float] | None = None,
        free_defects: list[str] | None = None,
        fix_charge_states: bool = False,
        **kwargs,
    ) -> "DefectSystem":
        r"""
        Generate a ``py-sc-fermi`` ``DefectSystem`` object that has defect
        concentrations fixed to the values determined at a high temperature
        (``annealing_temperature``), and then set to a lower temperature
        (``quenched_temperature``).

        This method creates a defect system where defect concentrations are
        initially calculated at an annealing temperature and then "frozen" as
        the system is cooled to a lower quenched temperature. It can optionally
        fix the concentrations of individual defect charge states or allow
        charge states to vary while keeping total defect concentrations fixed
        (default).

        See ``pseudo_equilibrium_solve`` docstring for more details.

        Args:
            annealing_temperature (float):
                The higher temperature (in Kelvin) at which the system is
                annealed to set initial defect concentrations.
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the
                equilibrium Fermi level position and defect/carrier
                concentrations. Here, this should be a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in
                ``self.defect_thermodynamics.el_refs`` then it is the `formal`
                chemical potentials (i.e. relative to the elemental reference
                energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if
                ``chempots`` was provided to ``self.defect_thermodynamics`` in
                the format generated by ``doped``).
                (Default: None)
            quenched_temperature (float):
                The lower temperature (in Kelvin) to which the system is
                quenched. Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyse the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping. For dopants of charge
                ``q``, the input should be ``q * 'Dopant Concentration'``.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            delta_gap (float | Callable):
                Change in band gap (in eV) of the host material at the
                annealing temperature (e.g. due to thermal renormalisation),
                relative to the original band gap of ``FermiSolver.bulk_dos``
                (assumed to correspond to the quenched temperature). If set,
                applies a scissor correction to ``bulk_dos`` which
                re-normalises the band gap symmetrically about the VBM and CBM
                (i.e. assuming equal up/downshifts of the band-edges around
                their original eigenvalues) while the defect levels remain
                fixed. Can be a value (in eV), or a function with annealing
                temperature as input; e.g. ``lambda T: -1e-6*500**2``.
                Default is 0 (no gap shifting).
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched
                temperature regardless of chemical potentials / temperature /
                Fermi level, in the format: ``{defect_name: concentration}``,
                where ``defect_name`` is the name of a defect entry without
                (e.g. ``"v_O"``) or with (e.g. ``"v_O_+2"``) the charge state;
                which will then fix either the total concentration of that
                defect or only the concentration for the specified charge
                state. Concentrations should be given in cm^-3. This can be
                used to simulate the effect of a fixed impurity concentration.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects (without charge states) to be excluded from
                high-temperature concentration fixing. Useful for highly mobile
                defects that are not expected to be "frozen-in" upon quenching.
                Any defects whose names begin with a string in this list will
                be excluded from high-temperature concentration fixing (e.g.
                ``"v_"`` will match all vacancy defects with
                ``doped``\-formatted names). Defaults to ``None``.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge
                states (``True``) or allow charge states to vary while keeping
                total defect concentrations fixed (``False``).
                Not expected to be physically sensible in most cases.
                Defaults to ``False``.
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if
                ``delta_gap`` is not 0).

        Returns:
            DefectSystem:
                A low-temperature defect system (``quenched_temperature``)
                with defect concentrations fixed to high-temperature
                (``annealing_temperature``) values.
        """
        self._check_required_backend_and_error("py-sc-fermi")
        free_defects = free_defects or []

        orig_py_sc_fermi_dos = self.py_sc_fermi_dos
        if delta_gap != 0.0:
            delta_gap = delta_gap if not callable(delta_gap) else delta_gap(annealing_temperature)
            assert self.defect_thermodynamics.vbm is not None
            assert self.defect_thermodynamics.band_gap is not None
            self.py_sc_fermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(
                scissor_dos(
                    delta_gap,
                    self.defect_thermodynamics.bulk_dos,
                    verbose=kwargs.get("verbose", False),
                    tol=kwargs.get("tol", 1e-8),
                ),
                vbm=self.defect_thermodynamics.vbm + delta_gap / 2,
                bandgap=self.defect_thermodynamics.band_gap + delta_gap,
            )

        defect_system = self._generate_defect_system(
            single_chempot_dict=single_chempot_dict,  # chempots handled in _generate_defect_system()
            el_refs=el_refs,
            temperature=annealing_temperature,
            effective_dopant_concentration=effective_dopant_concentration,
        )  # generated with delta_gap DOS
        initial_conc_dict = defect_system.concentration_dict()  # concentrations at initial temperature

        # Exclude the free_defects, carrier concentrations and Fermi level from fixing
        all_free_defects = ["Dopant", "Fermi Energy", "n0", "p0", *free_defects]

        # Get the fixed concentrations of non-exceptional (not-free) defects
        decomposed_conc_dict = defect_system.concentration_dict(decomposed=True)
        additional_data = {}
        for k, v in decomposed_conc_dict.items():
            if not any(k.startswith(i) for i in all_free_defects):
                for k1, v1 in v.items():
                    additional_data[k + "_" + str(k1)] = v1
        initial_conc_dict.update(additional_data)

        fixed_concs = {
            k: v
            for k, v in initial_conc_dict.items()
            if not any(k.startswith(i) for i in all_free_defects)
        }

        # Apply the fixed concentrations
        for defect_species in defect_system.defect_species:
            if fix_charge_states:
                for k, v in defect_species.charge_states.items():
                    key = f"{defect_species.name}_{int(k)}"
                    if key in list(fixed_concs.keys()):
                        v.fix_concentration(fixed_concs[key] / 1e24 * defect_system.volume)

            elif defect_species.name in fixed_concs and defect_species.name:
                defect_species.fix_concentration(
                    fixed_concs[defect_species.name] / 1e24 * defect_system.volume
                )

        self._fix_defect_concentrations(defect_system, fixed_defects, fixed_concs)
        defect_system.temperature = quenched_temperature
        if delta_gap != 0.0:
            self.py_sc_fermi_dos = orig_py_sc_fermi_dos
            defect_system.dos = orig_py_sc_fermi_dos  # set to original DOS
        return defect_system


def get_interpolated_chempots(
    chempot_start: dict,
    chempot_end: dict,
    n_points: int,
) -> list:
    """
    Generate a list of interpolated chemical potentials between two points.

    Here, these should be dictionaries of chemical potentials for `single`
    limits, in the format: ``{element symbol: chemical potential}``.

    Args:
        chempot_start (dict):
            A dictionary representing the starting chemical potentials.
        chempot_end (dict):
            A dictionary representing the ending chemical potentials.
        n_points (int):
            The number of interpolated points to generate, `including`
            the start and end points.

    Returns:
        list:
            A list of dictionaries, where each dictionary contains a
            `single` set of interpolated chemical potentials. The length of
            the list corresponds to `n_points`, and each dictionary
            corresponds to an interpolated state between the starting and
            ending chemical potentials.
    """
    return [
        {
            key: chempot_start[key] + (chempot_end[key] - chempot_start[key]) * i / (n_points - 1)
            for key in chempot_start
        }
        for i in range(n_points)
    ]


def _get_label_and_charge(name: str) -> tuple[str, int]:
    """
    Extracts the label and charge from a defect name string.

    Args:
        name (str): Name of the defect.

    Returns:
        tuple: A tuple containing the label and charge.
    """
    last_underscore = name.rfind("_")
    label = name[:last_underscore] if last_underscore != -1 else name
    charge_str = name[last_underscore + 1 :] if last_underscore != -1 else None

    charge = 0  # Initialize charge with a default value
    if charge_str is not None:
        with contextlib.suppress(ValueError):
            charge = int(charge_str)

    return label, charge


def _get_py_sc_fermi_dos_from_fermi_dos(
    fermi_dos: FermiDos,
    vbm: float | None = None,
    nelect: int | None = None,
    bandgap: float | None = None,
) -> "DOS":
    """
    Given an input ``pymatgen`` ``FermiDos`` object, return a corresponding
    ``py-sc-fermi`` ``DOS`` object (which can then be used with the ``py-sc-
    fermi`` ``FermiSolver`` backend).

    Args:
        fermi_dos (FermiDos):
            ``pymatgen`` ``FermiDos`` object to convert to ``py-sc-fermi``
            ``DOS``.
        vbm (float):
            The valence band maximum (VBM) eigenvalue in eV. If not provided,
            the VBM will be taken from the FermiDos object. When this function
            is used internally in ``doped``, the ``DefectThermodynamics.vbm``
            attribute is used.
        nelect (int):
            The total number of electrons in the system. If not provided, the
            number of electrons will be taken from the ``FermiDos`` object
            (which usually takes this value from the ``vasprun.xml(.gz)`` when
            parsing).
        bandgap (float):
            Band gap of the system in eV. If not provided, the band gap will be
            taken from the ``FermiDos`` object. When this function is used
            internally in ``doped``, the ``DefectThermodynamics.band_gap``
            attribute is used.

    Returns:
        DOS: A ``py-sc-fermi`` ``DOS`` object.
    """
    try:
        from py_sc_fermi.dos import DOS
    except ImportError as exc:
        raise ImportError("py-sc-fermi must be installed to use this function!") from exc

    densities = fermi_dos.densities
    if vbm is None:  # tol 1e-4 is lowest possible, as VASP rounds to 4 dp:
        vbm = fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1]

    edos = fermi_dos.energies - vbm
    if len(densities) == 2:
        dos = np.array([densities[Spin.up], densities[Spin.down]])
        spin_pol = True
    else:
        dos = np.array(densities[Spin.up])
        spin_pol = False

    if nelect is None:
        # this requires the input dos to be a FermiDos. NELECT could be calculated alternatively
        # by integrating the tdos of a ``pymatgen`` ``Dos`` object, but this isn't expected to be
        # a common use case and using parsed NELECT from vasprun.xml(.gz) is more reliable
        nelect = fermi_dos.nelecs
    if bandgap is None:
        bandgap = fermi_dos.get_gap(tol=1e-4, abs_tol=True)

    return DOS(dos=dos, edos=edos, nelect=nelect, bandgap=bandgap, spin_polarised=spin_pol)


def _get_min_max_target_values(
    results_df: pd.DataFrame, target: str, min_or_max: str, verbose: bool = False
) -> tuple:
    """
    Convenience function to get the minimum or maximum value(s) of a ``target``
    column or row in a ``results_df`` DataFrame, and the corresponding chemical
    potentials.

    Mainly intended for internal ``doped`` usage in the ``FermiSolver``
    ``optimise`` method.

    Args:
        results_df (pd.DataFrame):
            ``DataFrame`` of defect concentrations, as output by the
            ``FermiSolver`` ``scan_chempots`` /
            ``scan_chemical_potential_grid`` methods (which corresponds to the
            ``(pseudo_)equilibrium_solve`` ``DataFrame`` outputs, appended
            together for multiple chemical potentials).
        target (str):
            The target variable to minimise or maximise, e.g., "Electrons
            (cm^-3)", "Te_i", "Fermi Level (eV wrt VBM)" etc. Valid ``target``
            values are column names (or substrings), such as
            "Electrons (cm^-3)", "Holes (cm^-3)", "Fermi Level (eV wrt VBM)",
            "μ_X (eV)", etc., or defect names (without charge states), such as
            "v_O", "Te_i", etc.
            If a full defect name is given (e.g. ``Te_i_Td_Te2.83``) then the
            concentration of that defect will be used as the target variable.
            If a defect name substring is given instead (e.g. ``Te_i``), then
            the target variable will be the summed concentration of all defects
            with that substring in their name (e.g. ``Te_i_Td_Te2.83``,
            ``Te_i_C3v`` etc).
        min_or_max (str):
            Whether to find the minimum or maximum value(s) of the target.
            Should be either "min" or "max".
        verbose (bool):
            Whether to print information on identified target rows/columns.

    Returns:
        tuple:
            A tuple containing the results ``DataFrame`` at the chemical
            potentials which minimise/maximise the target property
            (``target_df``), the minimised/maximised value of the target
            property, and the corresponding chemical potentials -- in the given
            chemical potential range.
    """

    def min_or_max_func(x):
        return x.min() if "min" in min_or_max else x.max()

    chempots_labels = [col for col in results_df.columns if col.startswith("μ_")]

    # determine target; can be column, defect name, element (TODO), starting string of column name,
    # starting string of defect name, column name subset or defect name subset, w/that preferential order:

    target_names = (
        [col for col in results_df.columns if col == target]
        or [defect_name for defect_name in results_df.index if defect_name == target]
        or [col for col in results_df.columns if col.lower().startswith(target.lower())]
        or [
            defect_name
            for defect_name in results_df.index
            if defect_name.lower().startswith(target.lower())
        ]
        or [col for col in results_df.columns if target in col]
        or [defect_name for defect_name in (results_df.index) if target in defect_name]
    )
    target_names = sorted(set(target_names), key=target_names.index)  # preserve order

    if not target_names:
        raise ValueError(
            f"Target '{target}' not found in results DataFrame! Must be a column or defect "
            f"name/substring! See docstring for more info."
        )

    column = next(iter(target_names)) in results_df.columns

    if verbose:
        print(
            f"Searching for chemical potentials which {min_or_max}imise the target "
            f"{'column' if column else 'defect(s)'}: {target_names}..."
        )

    if column:
        target_name = next(iter(target_names))
        if len(target_names) > 1:  # can only match one column
            warnings.warn(
                f"Multiple columns with the name '{target}' found in the results DataFrame! "
                f"Choosing the first match '{target_name}' as the target."
            )
        current_value = min_or_max_func(results_df[target_name])
        target_df = results_df[results_df[target_name] == current_value]
        target_chempot = target_df[chempots_labels]

    else:
        filtered_df = results_df[results_df.index.isin(target_names)]  # filter df for the chosen defect(s)
        # group by chemical potentials, to sum values at the same chempots (e.g. for different defects):
        summed_df = filtered_df.groupby(chempots_labels).sum()
        # TODO: When adding element option, will need to subtract for vacancies...
        current_value = min_or_max_func(summed_df["Concentration (cm^-3)"])  # find the extremum row
        # get chempots which min/maximise the target:
        target_chempot = summed_df[summed_df["Concentration (cm^-3)"] == current_value].index.to_frame()
        # get all DataFrame rows which have the chempots matching the extremum row:
        target_df = results_df[results_df[chempots_labels].eq(target_chempot.iloc[0]).all(axis=1)]

    target_chempot = target_chempot.drop_duplicates(ignore_index=True)
    return target_df, current_value, target_chempot


def _ensure_list(
    var: float | range | list | np.ndarray | None = None,
) -> list[float | int] | np.ndarray[float | int] | None:
    if isinstance(var, range):
        return list(var)
    return [var] if isinstance(var, int | float) else var
