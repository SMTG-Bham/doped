"""
Code for analysing the thermodynamics of defect formation in solids, including
calculation of formation energies as functions of Fermi level and chemical
potentials, charge transition levels, defect/carrier concentrations etc.
"""

import contextlib
import os
import warnings
from copy import deepcopy
from functools import reduce
from itertools import chain, product
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.figure import Figure
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.core.composition import Composition
from pymatgen.electronic_structure.dos import Dos, FermiDos, Spin, f0
from pymatgen.io.vasp.outputs import Vasprun
from scipy.optimize import brentq
from scipy.spatial import HalfspaceIntersection

from doped import _doped_obj_properties_methods
from doped.chemical_potentials import get_X_poor_limit, get_X_rich_limit
from doped.core import DefectEntry, _no_chempots_warning, _orientational_degeneracy_warning
from doped.generation import _sort_defect_entries
from doped.utils.parsing import (
    _compare_incar_tags,
    _compare_kpoints,
    _compare_potcar_symbols,
    _get_bulk_supercell,
    _get_defect_supercell_site,
    get_nelect_from_vasprun,
    get_vasprun,
)
from doped.utils.plotting import _rename_key_and_dicts, _TLD_plot
from doped.utils.symmetry import _get_all_equiv_sites, _get_sga


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


def _parse_limit(chempots: dict, limit: Optional[str] = None):
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


def _update_old_chempots_dict(chempots: dict) -> dict:
    """
    Update a chempots dict in the old ``doped`` format (i.e. with ``facets``
    rather than ``limits``) to that of the new format.
    """
    if "facets" in chempots:
        chempots["limits"] = chempots.pop("facets")

    if "facets_wrt_el_refs" in chempots:
        chempots["limits_wrt_el_refs"] = chempots.pop("facets_wrt_el_refs")

    return chempots


def _parse_chempots(chempots: Optional[dict] = None, el_refs: Optional[dict] = None):
    """
    Parse the chemical potentials input, formatting them in the ``doped``
    format for use in analysis functions.

    Can be either ``doped`` format or user-specified format.

    Returns parsed ``chempots`` and ``el_refs``
    """
    if chempots is not None and "limits_wrt_elt_refs" in chempots:
        chempots["limits_wrt_el_refs"] = chempots.pop("limits_wrt_elt_refs")

    if chempots is not None and "facets_wrt_elt_refs" in chempots:
        chempots["facets_wrt_el_refs"] = chempots.pop("facets_wrt_elt_refs")

    if chempots is not None and ("facets" in chempots or "facets_wrt_el_refs" in chempots):
        chempots = _update_old_chempots_dict(chempots)

    if chempots is None:
        if el_refs is not None:
            chempots = {
                "limits": {"User Chemical Potentials": el_refs},
                "elemental_refs": el_refs,
                "limits_wrt_el_refs": {"User Chemical Potentials": {el: 0 for el in el_refs}},
            }

        return chempots, el_refs

    if "limits_wrt_el_refs" in chempots:  # doped format
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


def group_defects_by_distance(
    entry_list: list[DefectEntry], dist_tol: float = 1.5
) -> dict[str, dict[tuple, list[DefectEntry]]]:
    """
    Given an input list of DefectEntry objects, returns a dictionary of {simple
    defect name: {(equivalent defect sites): [DefectEntry]}, where 'simple
    defect name' is the nominal defect type (e.g. ``Te_i`` for
    ``Te_i_Td_Cd2.83_+2``), ``(equivalent defect sites)`` is a tuple of the
    equivalent defect sites (in the bulk supercell), and ``[DefectEntry]`` is a
    list of DefectEntry objects with the same simple defect name and a closest
    distance between symmetry-equivalent sites less than ``dist_tol``
    (1.5 Å by default).

    If a DefectEntry's site has a closest distance less than ``dist_tol`` to
    multiple sets of equivalent sites, then it is matched to the one with
    the lowest minimum distance.

    Args:
        entry_list ([DefectEntry]):
            A list of DefectEntry objects to group together.
        dist_tol (float):
            Threshold for the closest distance (in Å) between equivalent
            defect sites, for different species of the same defect type,
            to be grouped together (for plotting and transition level
            analysis). If the minimum distance between equivalent defect
            sites is less than ``dist_tol``, then they will be grouped
            together, otherwise treated as separate defects.
            (Default: 1.5)

    Returns:
        dict: {simple defect name: {(equivalent defect sites): [DefectEntry]}
    """
    # TODO: This algorithm works well for the vast majority of cases, however it can be sensitive to how
    #  many defects are parsed at once. For instance, in the full parsed CdTe defect dicts in test data,
    #  when parsing with metastable states, all Te_i are combined as each entry is within `dist_tol =
    #  1.5` of another interstitial, but without metastable states (`wout_meta`), Te_i_+2 is excluded (
    #  because it's not within `dist_tol` of any other _stable_ Te_i). Ideally our clustering algorithm
    #  would be independent of this... but challenging to setup without complex clustering approaches (
    #  for now this works very well as is, and this is a rare case and usually not a problem anyway as
    #  dist_tol can just be adjusted as needed)
    # initial group by Defect.name (same nominal defect), then distance to equiv sites
    # first make dictionary of nominal defect name: list of entries with that name
    defect_name_dict = {}
    for entry in entry_list:
        if entry.defect.name not in defect_name_dict:
            defect_name_dict[entry.defect.name] = [entry]
        else:
            defect_name_dict[entry.defect.name].append(entry)

    defect_site_dict: dict[str, dict[tuple, list[DefectEntry]]] = (
        {}
    )  # {defect name: {(equiv defect sites): entry list}}
    bulk_supercell = _get_bulk_supercell(entry_list[0])
    bulk_lattice = bulk_supercell.lattice
    bulk_supercell_sga = _get_sga(bulk_supercell)
    symm_bulk_struct = bulk_supercell_sga.get_symmetrized_structure()
    bulk_symm_ops = bulk_supercell_sga.get_symmetry_operations()

    for name, entry_list in defect_name_dict.items():
        defect_site_dict[name] = {}
        sorted_entry_list = sorted(
            entry_list, key=lambda x: abs(x.charge_state)
        )  # sort by charge, starting with closest to zero, for deterministic behaviour
        for entry in sorted_entry_list:
            entry_bulk_supercell = _get_bulk_supercell(entry)
            if entry_bulk_supercell.lattice != bulk_lattice:
                # recalculate bulk_symm_ops if bulk supercell differs
                bulk_supercell_sga = _get_sga(entry_bulk_supercell)
                symm_bulk_struct = bulk_supercell_sga.get_symmetrized_structure()
                bulk_symm_ops = bulk_supercell_sga.get_symmetry_operations()

            bulk_site = entry.calculation_metadata.get("bulk_site") or _get_defect_supercell_site(entry)
            # need to use relaxed defect site if bulk_site not in calculation_metadata

            min_dist_list = [
                min(  # get min dist for all equiv site tuples, in case multiple less than dist_tol
                    bulk_site.distance_and_image(site)[0] for site in equiv_site_tuple
                )
                for equiv_site_tuple in defect_site_dict[name]
            ]
            if min_dist_list and min(min_dist_list) < dist_tol:  # less than dist_tol, add to corresponding
                idxmin = np.argmin(min_dist_list)  # entry list
                if min_dist_list[idxmin] > 0.05:  # likely interstitials, need to add equiv sites to tuple
                    # pop old tuple, add new tuple with new equiv sites, and add entry to new tuple
                    orig_tuple = list(defect_site_dict[name].keys())[idxmin]
                    defect_entry_list = defect_site_dict[name].pop(orig_tuple)
                    equiv_site_tuple = (
                        tuple(  # tuple because lists aren't hashable (can't be dict keys)
                            _get_all_equiv_sites(
                                bulk_site.frac_coords,
                                symm_bulk_struct,
                                bulk_symm_ops,
                            )
                        )
                        + orig_tuple
                    )
                    defect_entry_list.extend([entry])
                    defect_site_dict[name][equiv_site_tuple] = defect_entry_list

                else:  # less than dist_tol, add to corresponding entry list
                    defect_site_dict[name][list(defect_site_dict[name].keys())[idxmin]].append(entry)

            else:  # no match found, add new entry
                try:
                    equiv_site_tuple = tuple(  # tuple because lists aren't hashable (can't be dict keys)
                        symm_bulk_struct.find_equivalent_sites(bulk_site)
                    )
                except ValueError:  # likely interstitials, need to add equiv sites to tuple
                    equiv_site_tuple = tuple(  # tuple because lists aren't hashable (can't be dict keys)
                        _get_all_equiv_sites(
                            bulk_site.frac_coords,
                            symm_bulk_struct,
                            bulk_symm_ops,
                        )
                    )

                defect_site_dict[name][equiv_site_tuple] = [entry]

    return defect_site_dict


def group_defects_by_name(entry_list: list[DefectEntry]) -> dict[str, list[DefectEntry]]:
    """
    Given an input list of DefectEntry objects, returns a dictionary of
    ``{defect name without charge: [DefectEntry]}``, where the values are lists
    of DefectEntry objects with the same defect name (excluding charge state).

    The ``DefectEntry.name`` attributes are used to get the defect names.
    These should be in the format:
    "{defect_name}_{optional_site_info}_{charge_state}".
    If the ``DefectEntry.name`` attribute is not defined or does not end with the
    charge state, then the entry will be renamed with the doped default name
    for the `unrelaxed` defect (i.e. using the point symmetry of the defect
    site in the bulk cell).

    For example, ``v_Cd_C3v_+1``, ``v_Cd_Td_+1`` and ``v_Cd_C3v_+2`` will be grouped
    as {``v_Cd_C3v``: [``v_Cd_C3v_+1``, ``v_Cd_C3v_+2``], ``v_Cd_Td``: [``v_Cd_Td_+1``]}.

    Args:
        entry_list ([DefectEntry]):
            A list of DefectEntry objects to group together by defect name
            (without charge).

    Returns:
        dict: Dictionary of {defect name without charge: [DefectEntry]}.
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
    Similar to a pymatgen PhaseDiagram object, allowing the analysis of
    formation energies as functions of Fermi level and chemical potentials
    (i.e. defect formation energy / transition level diagrams), charge
    transition levels, defect/carrier concentrations etc.

    This class is able to get:
        a) stability of charge states for a given defect,
        b) list of all formation energies,
        c) transition levels in the gap,
        d) used as input to doped plotting/analysis functions
    """

    def __init__(
        self,
        defect_entries: Union[list[DefectEntry], dict[str, DefectEntry]],
        chempots: Optional[dict] = None,
        el_refs: Optional[dict] = None,
        vbm: Optional[float] = None,
        band_gap: Optional[float] = None,
        dist_tol: float = 1.5,
        check_compatibility: bool = True,
    ):
        r"""
        Create a DefectThermodynamics object, which can be used to analyse the
        calculated thermodynamics of defects in solids (formation energies,
        transition levels, concentrations etc).

        Usually initialised using DefectsParser.get_defect_thermodynamics(), but
        can also be initialised with a list or dict of DefectEntry objects (e.g.
        from DefectsParser.defect_dict).

        Note that the ``DefectEntry.name`` attributes are used to label the defects in
        plots.

        Args:
            defect_entries ([DefectEntry] or {str: DefectEntry}):
                A list or dict of DefectEntry objects. Note that ``DefectEntry.name``
                attributes are used for grouping and plotting purposes! These should
                be in the format "{defect_name}_{optional_site_info}_{charge_state}".
                If the ``DefectEntry.name`` attribute is not defined or does not end with
                the charge state, then the entry will be renamed with the doped
                default name.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format generated by
                ``doped``\'s chemical potential parsing functions (see tutorials)) which
                allows easy analysis over a range of chemical potentials - where limit(s)
                (chemical potential limit(s)) to analyse/plot can later be chosen using
                the ``limits`` argument.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases
                in order to show the formal (relative) chemical potentials above the
                formation energy plot, in which case it is the formal chemical potentials
                (i.e. relative to the elemental references) that should be given here,
                otherwise the absolute (DFT) chemical potentials should be given.

                If None (default), sets all chemical potentials to zero. Chemical
                potentials can also be supplied later in each analysis function.
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided in format generated by ``doped`` (see tutorials).
                (Default: None)
            vbm (float):
                VBM eigenvalue in the bulk supercell, to use as Fermi level reference
                point for analysis. If None (default), will use "vbm" from the
                calculation_metadata dict attributes of the DefectEntry objects in
                ``defect_entries``.
            band_gap (float):
                Band gap of the host, to use for analysis.
                If None (default), will use "gap" from the calculation_metadata
                dict attributes of the DefectEntry objects in ``defect_entries``.
            dist_tol (float):
                Threshold for the closest distance (in Å) between equivalent
                defect sites, for different species of the same defect type,
                to be grouped together (for plotting and transition level
                analysis). If the minimum distance between equivalent defect
                sites is less than ``dist_tol``, then they will be grouped
                together, otherwise treated as separate defects.
                (Default: 1.5)
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each defect
                entry (i.e. that all reference bulk energies are the same).
                (Default: True)

        Key Attributes:
            defect_entries (list):
                List of DefectEntry objects included in the DefectThermodynamics set.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and hence concentrations etc), in the ``doped``
                format.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials.
            vbm (float):
                VBM eigenvalue in the bulk supercell, to use as Fermi level reference.
            band_gap (float):
                Band gap of the host, to use for analysis.
            dist_tol (float):
                Threshold for the closest distance (in Å) between equivalent
                defect sites, for different species of the same defect type,
                to be grouped together (for plotting and transition level
                analysis).
            transition_levels (dict):
                Dictionary of charge transition levels for each defect entry.
                (e.g. ``{defect_name: {charge: transition_level}}``).
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each defect
                entry (i.e. that all reference bulk energies are the same).
            bulk_formula (str):
                The reduced formula of the bulk structure (e.g. "CdTe").
        """
        if isinstance(defect_entries, dict):
            if not defect_entries:
                raise ValueError(
                    "No defects found in `defect_entries`. Please check the supplied dictionary is in the "
                    "correct format (i.e. {'defect_name': defect_entry}), or as a list: [defect_entry]."
                )
            defect_entries = list(defect_entries.values())

        self._defect_entries = defect_entries
        self._chempots, self._el_refs = _parse_chempots(chempots, el_refs)
        self._dist_tol = dist_tol
        self.check_compatibility = check_compatibility

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
            vbm_vals = []
            band_gap_vals = []
            for defect_entry in self.defect_entries:
                if "vbm" in defect_entry.calculation_metadata:
                    vbm_vals.append(defect_entry.calculation_metadata["vbm"])
                if "gap" in defect_entry.calculation_metadata:
                    band_gap_vals.append(defect_entry.calculation_metadata["gap"])

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

        # order entries for deterministic behaviour (particularly for plotting)
        self._sort_parse_and_check_entries(check_compatibility=check_compatibility)

        bulk_entry = self.defect_entries[0].bulk_entry
        if bulk_entry is not None:
            self.bulk_formula = bulk_entry.structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
        else:
            self.bulk_formula = None

    def _sort_parse_and_check_entries(self, check_compatibility: bool = True):
        """
        Sort the defect entries, parse the transition levels, and check the
        compatibility of the bulk entries (if check_compatibility is True).
        """
        defect_entries_dict: dict[str, DefectEntry] = {}
        for entry in self.defect_entries:  # rename defect entry names in dict if necessary ("_a", "_b"...)
            entry_name, [
                defect_entries_dict,
            ] = _rename_key_and_dicts(
                entry.name,
                [
                    defect_entries_dict,
                ],
            )
            defect_entries_dict[entry_name] = entry

        sorted_defect_entries_dict = _sort_defect_entries(defect_entries_dict)
        self._defect_entries = list(sorted_defect_entries_dict.values())
        with warnings.catch_warnings():  # ignore formation energies chempots warning when just parsing TLs
            warnings.filterwarnings("ignore", message="No chemical potentials")
            self._parse_transition_levels()
        if check_compatibility:
            self._check_bulk_compatibility()

    def as_dict(self):
        """
        Returns:
            JSON-serializable dict representation of DefectThermodynamics.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "defect_entries": [entry.as_dict() for entry in self.defect_entries],
            "chempots": self.chempots,
            "el_refs": self.el_refs,
            "vbm": self.vbm,
            "band_gap": self.band_gap,
            "dist_tol": self.dist_tol,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstitute a DefectThermodynamics object from a dict representation
        created using as_dict().

        Args:
            d (dict): dict representation of DefectThermodynamics.

        Returns:
            DefectThermodynamics object
        """
        warnings.filterwarnings(
            "ignore", "Use of properties is"
        )  # `message` only needs to match start of message
        defect_entries = [DefectEntry.from_dict(entry_dict) for entry_dict in d.get("defect_entries")]

        return cls(
            defect_entries,
            chempots=d.get("chempots"),
            el_refs=d.get("el_refs"),
            vbm=d.get("vbm"),
            band_gap=d.get("band_gap"),
            dist_tol=d.get("dist_tol"),
        )

    def to_json(self, filename: Optional[str] = None):
        """
        Save the ``DefectThermodynamics`` object as a json file, which can be
        reloaded with the ``DefectThermodynamics.from_json()`` class method.

        Args:
            filename (str): Filename to save json file as. If None, the filename will be
                set as "{Chemical Formula}_defect_thermodynamics.json" where
                {Chemical Formula} is the chemical formula of the host material.
        """
        if filename is None:
            if self.bulk_formula is not None:
                filename = f"{self.bulk_formula}_defect_thermodynamics.json"
            else:
                filename = "defect_thermodynamics.json"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: str):
        """
        Load a DefectThermodynamics object from a json file.

        Args:
            filename (str): Filename of json file to load DefectThermodynamics
            object from.

        Returns:
            DefectThermodynamics object
        """
        return loadfn(filename)

    def _get_chempots(self, chempots: Optional[dict] = None, el_refs: Optional[dict] = None):
        """
        Parse chemical potentials, either using input values (after formatting
        them in the doped format) or using the class attributes if set.
        """
        return _parse_chempots(chempots or self.chempots, el_refs or self.el_refs)

    def _parse_transition_levels(self):
        r"""
        Parses the charge transition levels for defect entries in the
        DefectThermodynamics object, and stores information about the stable
        charge states, transition levels etc.

        Defect entries of the same type (e.g. ``Te_i``, ``v_Cd``) are grouped together
        (for plotting and transition level analysis) based on the minimum
        distance between (equivalent) defect sites, to distinguish between
        different inequivalent sites. ``DefectEntry``\s of the same type and with
        a closest distance between equivalent defect sites less than ``dist_tol``
        (1.5 Å by default) are grouped together. If a DefectEntry's site has a
        closest distance less than ``dist_tol`` to multiple sets of equivalent
        sites, then it is matched to the one with the lowest minimum distance.

        Code for parsing the transition levels was originally templated from
        the pyCDT (pymatgen<=2022.7.25) thermodynamics code (deleted in later
        versions).

        This function uses scipy's HalfspaceIntersection
        to construct the polygons corresponding to defect stability as
        a function of the Fermi-level. The Halfspace Intersection
        constructs N-dimensional hyperplanes, in this case N=2, based
        on the equation of defect formation energy with considering chemical
        potentials:
            E_form = E_0^{Corrected} + Q_{defect}*(E_{VBM} + E_{Fermi}).

        Extra hyperplanes are constructed to bound this space so that
        the algorithm can actually find enclosed region.

        This code was modeled after the Halfspace Intersection code for
        the Pourbaix Diagram.
        """
        # determine defect charge transition levels:
        midgap_formation_energies = [  # without chemical potentials
            entry.formation_energy(
                fermi_level=0.5 * self.band_gap, vbm=entry.calculation_metadata.get("vbm", self.vbm)
            )
            for entry in self.defect_entries
        ]
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
            defect_site_dict = group_defects_by_distance(self.defect_entries, dist_tol=self.dist_tol)
            grouped_entries_list = [
                entry_list for sub_dict in defect_site_dict.values() for entry_list in sub_dict.values()
            ]
        except Exception as e:
            grouped_entries = group_defects_by_name(self.defect_entries)
            grouped_entries_list = list(grouped_entries.values())
            warnings.warn(
                f"Grouping (inequivalent) defects by distance failed with error: {e!r}"
                f"\nGrouping by defect names instead."
            )  # possibly different bulks (though this should be caught/warned about earlier), or not
            # parsed with recent doped versions etc

        for grouped_defect_entries in grouped_entries_list:
            sorted_defect_entries = sorted(
                grouped_defect_entries, key=lambda x: (abs(x.charge_state), x.get_ediff())
            )  # sort by charge, starting with closest to zero, and then formation energy for
            # deterministic behaviour

            # prepping coefficient matrix for half-space intersection
            # [-Q, 1, -1*(E_form+Q*VBM)] -> -Q*E_fermi+E+-1*(E_form+Q*VBM) <= 0 where E_fermi and E are
            # the variables in the hyperplanes
            hyperplanes = np.array(
                [
                    [
                        -1.0 * entry.charge_state,
                        1,
                        -1.0 * (entry.get_ediff() + entry.charge_state * self.vbm),  # type: ignore
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
            ints_and_facets_zip = zip(hs_ints.intersections, hs_ints.dual_facets)
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
                _, facets = zip(*ints_and_facets_list)
                transition_level_map[defect_name_wout_charge] = {  # map of transition level: charge states
                    intersection[0]: sorted(
                        [sorted_defect_entries[i].charge_state for i in facet], reverse=True
                    )
                    for intersection, facet in ints_and_facets_list
                }
                stable_entries[defect_name_wout_charge] = [
                    sorted_defect_entries[i] for dual in facets for i in dual
                ]
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
        self._name_wout_charge_appearance_order = {
            entry.name.rsplit("_", 1)[0]: self._defect_entries.index(entry)
            for entry in self._defect_entries  # already sorted according to _sort_defect_entries()
        }

        def _map_sorting_func(name_wout_charge):
            for i in range(name_wout_charge.count("_") + 1):  # number of underscores in name
                appearance_order = self._name_wout_charge_appearance_order.get(
                    name_wout_charge.rsplit("_", i)[0]
                )
                if appearance_order is not None:
                    return (appearance_order, name_wout_charge)

            return (100, name_wout_charge)

        self.transition_level_map = dict(
            sorted(self.transition_level_map.items(), key=lambda item: _map_sorting_func(item[0]))
        )
        self.stable_entries = dict(
            sorted(self.stable_entries.items(), key=lambda item: _map_sorting_func(item[0]))
        )
        self.all_entries = dict(
            sorted(self.all_entries.items(), key=lambda item: _map_sorting_func(item[0]))
        )
        self.defect_charge_map = dict(
            sorted(self.defect_charge_map.items(), key=lambda item: _map_sorting_func(item[0]))
        )

        self.transition_levels = {
            defect_name: list(defect_tls.keys())
            for defect_name, defect_tls in transition_level_map.items()
        }
        self.stable_charges = {
            defect_name: [entry.charge_state for entry in entries]
            for defect_name, entries in stable_entries.items()
        }

    def _check_bulk_compatibility(self):
        """
        Helper function to quickly check if all entries have compatible bulk
        calculation settings, by checking that the energy of
        defect_entry.bulk_entry is the same for all defect entries.

        By proxy checks that same bulk/defect calculation settings were used in
        all cases, from each bulk/defect combination already being checked when
        parsing. This is to catch any cases where defects may have been parsed
        separately and combined (rather than altogether with DefectsParser,
        which ensures the same bulk in each case), and where a different bulk
        reference calculation was (mistakenly) used.
        """
        bulk_energies = [entry.bulk_entry.energy for entry in self.defect_entries]
        if max(bulk_energies) - min(bulk_energies) > 0.02:  # 0.02 eV tolerance
            warnings.warn(
                f"Note that not all defects in `defect_entries` have the same reference bulk energy (bulk "
                f"supercell calculation at `bulk_path` when parsing), with energies differing by >0.02 "
                f"eV. This can lead to inaccuracies in predicted formation energies! The bulk energies of "
                f"defect entries in `defect_entries` are:\n"
                f"{[(entry.name, entry.bulk_entry.energy) for entry in self.defect_entries]}\n"
                f"You can suppress this warning by setting `check_compatibility=False` in "
                "`DefectThermodynamics` initialisation."
            )

    def _check_bulk_defects_compatibility(self):
        """
        Helper function to quickly check if all entries have compatible
        defect/bulk calculation settings.

        Currently not used, as the bulk/defect compatibility is checked when
        parsing, and the compatibility across bulk calculations is checked with
        _check_bulk_compatibility().
        """
        # check each defect entry against its own bulk, and also check each bulk against each other
        reference_defect_entry = self.defect_entries[0]
        reference_run_metadata = reference_defect_entry.calculation_metadata["run_metadata"]
        for defect_entry in self.defect_entries:
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

    def add_entries(
        self,
        defect_entries: Union[list[DefectEntry], dict[str, DefectEntry]],
        check_compatibility: bool = True,
    ):
        """
        Add additional defect entries to the DefectThermodynamics object.

        Args:
            defect_entries ([DefectEntry] or {str: DefectEntry}):
                A list or dict of DefectEntry objects, to add to the
                DefectThermodynamics.defect_entries list. Note that ``DefectEntry.name``
                attributes are used for grouping and plotting purposes! These should
                be in the format "{defect_name}_{optional_site_info}_{charge_state}".
                If the ``DefectEntry.name`` attribute is not defined or does not end with
                the charge state, then the entry will be renamed with the doped
                default name.
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each defect
                entry (i.e. that all reference bulk energies are the same).
                (Default: True)
        """
        if isinstance(defect_entries, dict):
            defect_entries = list(defect_entries.values())

        if not defect_entries:
            raise ValueError(
                "No defects found in `defect_entries`. Please check the supplied dictionary is in the "
                "correct format (i.e. {'defect_name': defect_entry}), or as a list: [defect_entry]."
            )

        self._defect_entries += defect_entries
        self._sort_parse_and_check_entries(check_compatibility=check_compatibility)

    @property
    def defect_entries(self):
        """
        Get the list of parsed DefectEntry objects in the DefectThermodynamics
        analysis object.
        """
        return self._defect_entries

    @defect_entries.setter
    def defect_entries(self, input_defect_entries):
        r"""
        Set the list of parsed ``DefectEntry``\s to include in the
        DefectThermodynamics object, and reparse the thermodynamic information
        (transition levels etc).
        """
        self._defect_entries = input_defect_entries
        self._sort_parse_and_check_entries()

    @property
    def chempots(self):
        r"""
        Get the chemical potentials dictionary used for calculating the defect
        formation energies.

        ``chempots`` is a dictionary of chemical potentials to use for calculating
        the defect formation energies, in the form of:
        ``{"limits": [{'limit': [chempot_dict]}]}`` (the format generated by
        ``doped``\'s chemical potential parsing functions (see tutorials)) which
        allows easy analysis over a range of chemical potentials - where limit(s)
        (chemical potential limit(s)) to analyse/plot can later be chosen using
        the ``limits`` argument.
        """
        return self._chempots

    @chempots.setter
    def chempots(self, input_chempots):
        r"""
        Set the chemical potentials dictionary (``chempots``), and reparse to
        have the required ``doped`` format.

        ``chempots`` is a dictionary of chemical potentials to use for calculating
        the defect formation energies, in the form of:
        ``{"limits": [{'limit': [chempot_dict]}]}`` (the format generated by
        ``doped``\'s chemical potential parsing functions (see tutorials)) which
        allows easy analysis over a range of chemical potentials - where limit(s)
        (chemical potential limit(s)) to analyse/plot can later be chosen using
        the ``limits`` argument.

        Alternatively this can be a dictionary of chemical potentials for a
        single limit (limit), in the format: ``{element symbol: chemical potential}``.
        If manually specifying chemical potentials this way, you can set the
        ``el_refs`` option with the DFT reference energies of the elemental phases
        in order to show the formal (relative) chemical potentials above the
        formation energy plot, in which case it is the formal chemical potentials
        (i.e. relative to the elemental references) that should be given here,
        otherwise the absolute (DFT) chemical potentials should be given.

        If None (default), sets all formal chemical potentials (i.e. relative
        to the elemental reference energies in ``el_refs``) to zero. Chemical
        potentials can also be supplied later in each analysis function.
        (Default: None)
        """
        self._chempots, self._el_refs = _parse_chempots(input_chempots, self._el_refs)

    @property
    def el_refs(self):
        """
        Get the elemental reference energies for the chemical potentials.

        This is in the form of a dictionary of elemental reference energies for
        the chemical potentials, in the format: ``{element symbol: reference
        energy}``.
        """
        return self._el_refs

    @el_refs.setter
    def el_refs(self, input_el_refs):
        """
        Set the elemental reference energies for the chemical potentials
        (``el_refs``), and reparse to have the required ``doped`` format.

        This is in the form of a dictionary of elemental reference energies for the
        chemical potentials, in the format: ``{element symbol: reference energy}``,
        and is used to determine the formal chemical potentials, when ``chempots``
        has been manually specified as ``{element symbol: chemical potential}``.
        Unnecessary if ``chempots`` is provided in format generated by ``doped``
        (see tutorials).
        """
        self._chempots, self._el_refs = _parse_chempots(self._chempots, input_el_refs)

    @property
    def defect_names(self):
        """
        List of names of defects in the DefectThermodynamics set.
        """
        return list(self.defect_charge_map.keys())

    @property
    def all_stable_entries(self):
        """
        List all stable entries (defect + charge) in the DefectThermodynamics
        set.
        """
        return list(chain.from_iterable(self.stable_entries.values()))

    @property
    def all_unstable_entries(self):
        """
        List all unstable entries (defect + charge) in the DefectThermodynamics
        set.
        """
        all_stable_entries = self.all_stable_entries
        return [e for e in self.defect_entries if e not in all_stable_entries]

    @property
    def dist_tol(self):
        r"""
        Get the distance tolerance (in Å) used for grouping (equivalent)
        defects together (for plotting and transition level analysis).

        ``DefectEntry``\s of the same type and with a closest distance between
        equivalent defect sites less than ``dist_tol`` (1.5 Å by default) are
        grouped together. If a DefectEntry's site has a closest distance less
        than ``dist_tol`` to multiple sets of equivalent sites, then it is
        matched to the one with the lowest minimum distance.
        """
        return self._dist_tol

    @dist_tol.setter
    def dist_tol(self, input_dist_tol: float):
        r"""
        Set the distance tolerance (in Å) used for grouping (equivalent)
        defects together (for plotting and transition level analysis), and
        reparse the thermodynamic information (transition levels etc) with this
        tolerance.

        ``DefectEntry``\s of the same type and with a closest distance between
        equivalent defect sites less than ``dist_tol`` (1.5 Å by default) are
        grouped together. If a DefectEntry's site has a closest distance less
        than ``dist_tol`` to multiple sets of equivalent sites, then it is
        matched to the one with the lowest minimum distance.
        """
        self._dist_tol = input_dist_tol
        with warnings.catch_warnings():  # ignore formation energies chempots warning when just parsing TLs
            warnings.filterwarnings("ignore", message="No chemical potentials")
            self._parse_transition_levels()

    def _get_and_set_fermi_level(self, fermi_level: Optional[float] = None) -> float:
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

    def get_equilibrium_concentrations(
        self,
        chempots: Optional[dict] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict] = None,
        fermi_level: Optional[float] = None,
        temperature: float = 300,
        per_charge: bool = True,
        per_site: bool = False,
        skip_formatting: bool = False,
    ) -> pd.DataFrame:
        r"""
        Compute the `equilibrium` concentrations (in cm^-3) for all
        ``DefectEntry``\s in the ``DefectThermodynamics`` object, at a given
        chemical potential limit, fermi_level and temperature, assuming the
        dilute limit approximation.

        Note that these are the `equilibrium` defect concentrations!
        DefectThermodynamics.get_quenched_fermi_level_and_concentrations() can
        instead be used to calculate the Fermi level and defect concentrations
        for a material grown/annealed at higher temperatures and then cooled
        (quenched) to room/operating temperature (where defect concentrations
        are assumed to remain fixed) - this is known as the frozen defect
        approach and is typically the most valid approximation (see its
        docstring for more information).

        The degeneracy/multiplicity factor "g" is an important parameter in the defect
        concentration equation (see discussion in https://doi.org/10.1039/D2FD00043A
        and https://doi.org/10.1039/D3CS00432E), affecting the final concentration by
        up to ~2 orders of magnitude. This factor is taken from the product of the
        ``defect_entry.defect.multiplicity`` and ``defect_entry.degeneracy_factors``
        attributes.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus concentrations). If ``None`` (default),
                will use ``self.chempots`` (= 0 for all chemical potentials by default).
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (``limit``), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            limit (str):
                The chemical potential limit for which to
                obtain the equilibrium concentrations. Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical potential
                  limit - otherwise will use the first chemical potential limit in the
                  ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential, referenced
                to the VBM. If None (default), set to the mid-gap Fermi level (E_g/2).
            temperature (float):
                Temperature in Kelvin at which to calculate the equilibrium concentrations.
                Default is 300 K.
            per_charge (bool):
                Whether to break down the defect concentrations into individual defect charge
                states (e.g. ``v_Cd_0``, ``v_Cd_-1``, ``v_Cd_-2`` instead of ``v_Cd``).
                (default: True)
            per_site (bool):
                Whether to return the concentrations as percent concentrations per site,
                rather than the default of per cm^3. (default: False)
            skip_formatting (bool):
                Whether to skip formatting the defect charge states and concentrations as
                strings (and keep as ints and floats instead). (default: False)

        Returns:
            ``pandas`` ``DataFrame`` of defect concentrations (and formation energies)
            for each ``DefectEntry`` in the ``DefectThermodynamics`` object.
        """
        fermi_level = self._get_and_set_fermi_level(fermi_level)
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None

        energy_concentration_list = []

        for defect_entry in self.defect_entries:
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
                vbm=self.vbm,
                temperature=temperature,
                per_site=per_site,
            )
            if skip_formatting:
                concentration = raw_concentration

            elif per_site:
                concentration = (
                    f"{raw_concentration:.3%}"  # greater than 0.001%
                    if raw_concentration > 1e-5
                    else f"{raw_concentration * 100:.3e} %"
                )
            else:
                concentration = f"{raw_concentration:.3e}"
            energy_concentration_list.append(
                {
                    "Defect": defect_entry.name.rsplit("_", 1)[0],  # name without charge
                    "Raw Charge": defect_entry.charge_state,  # for sorting
                    "Charge": (
                        defect_entry.charge_state
                        if skip_formatting
                        else f"{'+' if defect_entry.charge_state > 0 else ''}{defect_entry.charge_state}"
                    ),
                    "Formation Energy (eV)": round(formation_energy, 3),
                    "Raw Concentration": raw_concentration,
                    "Concentration (per site)" if per_site else "Concentration (cm^-3)": concentration,
                }
            )

        conc_df = pd.DataFrame(energy_concentration_list)

        if per_charge:
            conc_df["Charge State Population"] = conc_df["Raw Concentration"] / conc_df.groupby("Defect")[
                "Raw Concentration"
            ].transform("sum")
            conc_df["Charge State Population"] = conc_df["Charge State Population"].apply(
                lambda x: f"{x:.1%}"
            )
            conc_df = conc_df.drop(columns=["Raw Charge", "Raw Concentration"])
            return conc_df.reset_index(drop=True)

        # group by defect and sum concentrations:
        summed_df = conc_df.groupby("Defect").sum(numeric_only=True)
        summed_df[next(k for k in conc_df.columns if k.startswith("Concentration"))] = (
            summed_df["Raw Concentration"]
            if skip_formatting
            else summed_df["Raw Concentration"].apply(lambda x: f"{x:.3e}")
        )
        return summed_df.drop(
            columns=[
                i
                for i in ["Charge", "Formation Energy (eV)", "Raw Charge", "Raw Concentration"]
                if i in summed_df.columns
            ]
        )

    def _parse_fermi_dos(self, bulk_dos_vr: Union[str, Vasprun, FermiDos]):
        if isinstance(bulk_dos_vr, FermiDos):
            return bulk_dos_vr

        if isinstance(bulk_dos_vr, str):
            bulk_dos_vr = get_vasprun(bulk_dos_vr, parse_dos=True)

        fermi_dos_band_gap, _cbm, fermi_dos_vbm, _ = bulk_dos_vr.eigenvalue_band_properties
        if abs(fermi_dos_vbm - self.vbm) > 0.1:
            warnings.warn(
                f"The VBM eigenvalue of the bulk DOS calculation ({fermi_dos_vbm:.2f} eV, with band "
                f"gap of {fermi_dos_band_gap:.2f} eV) differs from that of the bulk supercell "
                f"calculations ({self.vbm:.2f} eV, with band gap of {self.band_gap:.2f} eV) by more "
                f"than 0.1 eV. If this is only due to slight differences in kpoint sampling for the "
                f"bulk DOS vs defect supercell calculations, and consistent functional settings "
                f"(LHFCALC, AEXX etc) were used, then the eigenvalue references should be consistent "
                f"and this warning can be ignored. If not, then this could lead to inaccuracies in "
                f"the predicted Fermi level. Note that the Fermi level will be referenced to the VBM "
                f"of the bulk supercell (i.e. DefectThermodynamics.vbm)"
            )
        return FermiDos(bulk_dos_vr.complete_dos, nelecs=get_nelect_from_vasprun(bulk_dos_vr))

    def get_equilibrium_fermi_level(
        self,
        bulk_dos_vr: Union[str, Vasprun, FermiDos],
        chempots: Optional[dict] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict] = None,
        temperature: float = 300,
        return_concs: bool = False,
    ) -> Union[float, tuple[float, float, float]]:
        r"""
        Calculate the self-consistent Fermi level, at a given chemical
        potential limit and temperature, assuming `equilibrium` defect
        concentrations (i.e. under annealing) and the dilute limit
        approximation, by self-consistently solving for the Fermi level which
        yields charge neutrality.

        Note that this assumes `equilibrium` defect concentrations!
        DefectThermodynamics.get_quenched_fermi_level_and_concentrations() can
        instead be used to calculate the Fermi level and defect concentrations
        for a material grown/annealed at higher temperatures and then cooled
        (quenched) to room/operating temperature (where defect concentrations
        are assumed to remain fixed) - this is known as the frozen defect
        approach and is typically the most valid approximation (see its
        docstring for more information).

        Note that the bulk DOS calculation should be well-converged with respect to
        k-points for accurate Fermi level predictions!
        Note: For dense k-point DOS calculations, loading the vasprun.xml(.gz) file
        can be the most time-consuming part of this function, so if you are running
        this function multiple times, it is faster to initialise the FermiDos object
        yourself (and set ``nelecs`` appropriately - important!), and set ``bulk_dos_vr``
        to this object instead.

        The degeneracy/multiplicity factor "g" is an important parameter in the defect
        concentration equation and thus Fermi level calculation (see discussion in
        https://doi.org/10.1039/D2FD00043A and https://doi.org/10.1039/D3CS00432E),
        affecting the final concentration by up to ~2 orders of magnitude. This factor
        is taken from the product of the ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes.

        Args:
            bulk_dos_vr (str or Vasprun or FermiDos):
                Path to the vasprun.xml(.gz) output of a bulk electronic density of states
                (DOS) calculation, or the corresponding pymatgen Vasprun object. Usually
                this is a static calculation with the `primitive` cell of the bulk, with
                relatively dense k-point sampling (especially for materials with disperse
                band edges) to ensure an accurately-converged DOS and thus Fermi level.
                ISMEAR = -5 (tetrahedron smearing) is usually recommended for best
                convergence. Consistent functional settings should be used for the bulk
                DOS and defect supercell calculations.
                Alternatively, a pymatgen FermiDos object can be supplied directly (e.g.
                in case you are using a DFT code other than VASP).
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus concentrations and Fermi level).
                If ``None`` (default), will use ``self.chempots`` (= 0 for all chemical
                potentials by default).
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            limit (str):
                The chemical potential limit for which to
                determine the equilibrium Fermi level. Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical potential
                  limit - otherwise will use the first chemical potential limit in the
                  ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)
            temperature (float):
                Temperature in Kelvin at which to calculate the equilibrium Fermi level.
                Default is 300 K.
            return_concs (bool):
                Whether to return the corresponding electron and hole concentrations
                (in cm^-3) as well as the Fermi level. (default: False)

        Returns:
            Self consistent Fermi level (in eV from the VBM), and the corresponding
            electron and hole concentrations (in cm^-3) if return_concs=True.
        """
        fermi_dos = self._parse_fermi_dos(bulk_dos_vr)
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None

        def _get_total_q(fermi_level):
            conc_df = self.get_equilibrium_concentrations(
                chempots=chempots,
                limit=limit,
                el_refs=el_refs,
                temperature=temperature,
                fermi_level=fermi_level,
                skip_formatting=True,
            )
            qd_tot = (conc_df["Charge"] * conc_df["Concentration (cm^-3)"]).sum()
            qd_tot += fermi_dos.get_doping(fermi_level=fermi_level + self.vbm, temperature=temperature)
            return qd_tot

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow")  # ignore overflow warnings

            eq_fermi_level: float = brentq(_get_total_q, -1.0, self.band_gap + 1.0)  # type: ignore
            if return_concs:
                e_conc, h_conc = get_e_h_concs(
                    fermi_dos, eq_fermi_level + self.vbm, temperature  # type: ignore
                )
                return eq_fermi_level, e_conc, h_conc

        return eq_fermi_level

    def get_quenched_fermi_level_and_concentrations(
        self,
        bulk_dos_vr: Union[str, Vasprun, FermiDos],
        chempots: Optional[dict] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict] = None,
        annealing_temperature: float = 1000,
        quenched_temperature: float = 300,
        delta_gap: float = 0,
        **kwargs,
    ) -> tuple[float, float, float, pd.DataFrame]:
        r"""
        Calculate the self-consistent Fermi level and corresponding
        carrier/defect calculations, for a given chemical potential limit,
        annealing temperature and quenched/operating temperature, using the
        frozen defect and dilute limit approximations under the constraint of
        charge neutrality.

        According to the 'frozen defect' approximation, we typically expect defect
        concentrations to reach equilibrium during annealing/crystal growth
        (at elevated temperatures), but `not` upon quenching (i.e. at
        room/operating temperature) where we expect kinetic inhibition of defect
        annhiliation and hence non-equilibrium defect concentrations / Fermi level.
        Typically this is approximated by computing the equilibrium Fermi level and
        defect concentrations at the annealing temperature, and then assuming the
        total concentration of each defect is fixed to this value, but that the
        relative populations of defect charge states (and the Fermi level) can
        re-equilibrate at the lower (room) temperature. See discussion in
        https://doi.org/10.1039/D3CS00432E (brief),
        https://doi.org/10.1016/j.cpc.2019.06.017 (detailed) and
        ``doped``/``py-sc-fermi`` tutorials for more information.
        In certain cases (such as Li-ion battery materials or extremely slow charge
        capture/emission), these approximations may have to be adjusted such that some
        defects/charge states are considered fixed and some are allowed to
        re-equilibrate (e.g. highly mobile Li vacancies/interstitials). Modelling
        these specific cases is demonstrated in:
        py-sc-fermi.readthedocs.io/en/latest/tutorial.html#3.-Applying-concentration-constraints

        This function works by calculating the self-consistent Fermi level and total
        concentration of each defect at the annealing temperature, then fixing the
        total concentrations to these values and re-calculating the self-consistent
        (constrained equilibrium) Fermi level and relative charge state concentrations
        under this constraint at the quenched/operating temperature.

        Note that the bulk DOS calculation should be well-converged with respect to
        k-points for accurate Fermi level predictions!
        Note: For dense k-point DOS calculations, loading the vasprun.xml(.gz) file
        can be the most time-consuming part of this function, so if you are running
        this function multiple times, it is faster to initialise the FermiDos object
        yourself (and set ``nelecs`` appropriately - important!), and set ``bulk_dos_vr``
        to this object instead.

        The degeneracy/multiplicity factor "g" is an important parameter in the defect
        concentration equation and thus Fermi level calculation (see discussion in
        https://doi.org/10.1039/D2FD00043A and https://doi.org/10.1039/D3CS00432E),
        affecting the final concentration by up to 2 orders of magnitude. This factor
        is taken from the product of the ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes.

        If you use this code in your work, please also cite:
        Squires et al., (2023). Journal of Open Source Software, 8(82), 4962
        https://doi.org/10.21105/joss.04962

        Args:
            bulk_dos_vr (str or Vasprun or FermiDos):
                Path to the vasprun.xml(.gz) output of a bulk electronic density of states
                (DOS) calculation, or the corresponding pymatgen Vasprun object. Usually
                this is a static calculation with the `primitive` cell of the bulk, with
                relatively dense k-point sampling (especially for materials with disperse
                band edges) to ensure an accurately-converged DOS and thus Fermi level.
                ISMEAR = -5 (tetrahedron smearing) is usually recommended for best
                convergence. Consistent functional settings should be used for the bulk
                DOS and defect supercell calculations.
                Alternatively, a pymatgen FermiDos object can be supplied directly (e.g.
                in case you are using a DFT code other than VASP).
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus concentrations and Fermi level).
                If ``None`` (default), will use ``self.chempots`` (= 0 for all chemical
                potentials by default).
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            limit (str):
                The chemical potential limit for which to
                determine the Fermi level and concentrations. Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical potential
                  limit - otherwise will use the first chemical potential limit in the
                  ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)
            annealing_temperature (float):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to the
                highest temperature during annealing/synthesis of the material (at
                which we assume equilibrium defect concentrations) within the frozen
                defect approach. Default is 1000 K.
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level, given the fixed total
                concentrations, which should correspond to operating temperature
                of the material (typically room temperature). Default is 300 K.
            delta_gap (float):
                Change in band gap (in eV) of the host material at the annealing
                temperature (e.g. due to thermal renormalisation), relative to the
                original band gap of the FermiDos object. If set, applies a scissor
                correction to ``fermi_dos`` which renormalises the band gap symmetrically
                about the VBM and CBM (i.e. assuming equal up/downshifts of the band-edges
                around their original eigenvalues) while the defect levels remain fixed.
                (Default: 0)
            **kwargs:
                Additional keyword arguments to pass to ``scissor_dos`` (if ``delta_gap``
                is not 0).

        Returns:
            Predicted quenched Fermi level (in eV from the VBM), the corresponding
            electron and hole concentrations (in cm^-3) and a dataframe of the
            quenched defect concentrations (in cm^-3).
            (fermi_level, e_conc, h_conc, conc_df)
        """
        # TODO: Update docstrings after `py-sc-fermi` interface written, to point toward it for more
        #  advanced analysis
        fermi_dos = self._parse_fermi_dos(bulk_dos_vr)
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None
        annealing_dos = (
            fermi_dos
            if delta_gap == 0
            else scissor_dos(
                delta_gap,
                fermi_dos,
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
        )
        annealing_defect_concentrations = self.get_equilibrium_concentrations(
            chempots=chempots,
            limit=limit,
            el_refs=el_refs,
            fermi_level=annealing_fermi_level,  # type: ignore
            temperature=annealing_temperature,
            per_charge=False,  # give total concentrations for each defect
            skip_formatting=True,
        )
        annealing_defect_concentrations = annealing_defect_concentrations.rename(
            columns={"Concentration (cm^-3)": "Total Concentration (cm^-3)"}
        )

        def _get_constrained_total_q(fermi_level, return_conc_df=False):
            conc_df = self.get_equilibrium_concentrations(
                chempots=chempots,
                limit=limit,
                el_refs=el_refs,
                temperature=quenched_temperature,
                fermi_level=fermi_level,
                skip_formatting=True,
            )

            conc_df = conc_df.merge(annealing_defect_concentrations, on="Defect")
            conc_df["Concentration (cm^-3)"] = (  # set total concentration to match annealing conc
                conc_df["Concentration (cm^-3)"]  # but with same relative concentrations
                / conc_df.groupby("Defect")["Concentration (cm^-3)"].transform("sum")
            ) * conc_df["Total Concentration (cm^-3)"]

            if return_conc_df:
                return conc_df
            qd_tot = (conc_df["Charge"] * conc_df["Concentration (cm^-3)"]).sum()
            qd_tot += fermi_dos.get_doping(
                fermi_level=fermi_level + self.vbm, temperature=quenched_temperature
            )
            return qd_tot

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow")  # ignore overflow warnings

            eq_fermi_level: float = brentq(
                _get_constrained_total_q, -1.0, self.band_gap + 1.0  # type: ignore
            )

        e_conc, h_conc = get_e_h_concs(
            fermi_dos, eq_fermi_level + self.vbm, quenched_temperature  # type: ignore
        )
        return (
            eq_fermi_level,
            e_conc,
            h_conc,
            _get_constrained_total_q(eq_fermi_level, return_conc_df=True),
        )

    def get_formation_energy(
        self,
        defect_entry: Union[str, DefectEntry],
        chempots: Optional[dict] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict] = None,
        fermi_level: Optional[float] = None,
    ) -> float:
        r"""
        Compute the formation energy for a ``DefectEntry`` at a given chemical
        potential limit and fermi_level. ``defect_entry`` can be a string of
        the defect name, of the ``DefectEntry`` object itself.

        Args:
            defect_entry (str or DefectEntry):
                Either a string of the defect entry name (in
                ``DefectThermodynamics.defect_entries``), or a ``DefectEntry`` object.
                If the defect name is given without the charge state, then the
                formation energy of the lowest energy (stable) charge state
                at the chosen Fermi level will be given.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energy. If None (default), will use ``self.chempots``.
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.

                If None (default), sets all chemical potentials to zero.
                (Default: None)
            limit (str):
                The chemical potential limit for which to
                calculate the formation energy. Can be either:

                - None (default), if ``chempots`` corresponds to a single chemical
                  potential limit - otherwise will use the first chemical potential
                  limit in the ``chempots`` dict.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM. The VBM value is taken from
                ``DefectEntry.calculation_metadata`` if present, otherwise
                from self.vbm.
                If None (default), set to the mid-gap Fermi level (E_g/2).

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
            for entry in self.defect_entries
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
            for entry in self.defect_entries
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
            f"DefectThermodynamics.defect_entries, which have "
            f"names:\n{[entry.name for entry in self.defect_entries]}"
        )

    def get_dopability_limits(
        self, chempots: Optional[dict] = None, limit: Optional[str] = None, el_refs: Optional[dict] = None
    ) -> pd.DataFrame:
        r"""
        Find the dopability limits of the defect system, searching over all
        limits (chemical potential limits) in ``chempots`` and returning the
        most p/n-type conditions, or for a given chemical potential limit (if
        ``limit`` is set or ``chempots`` corresponds to a single chemical
        potential limit; i.e. {element symbol: chemical potential}).

        The dopability limites are defined by the (first) Fermi level positions at
        which defect formation energies become negative as the Fermi level moves
        towards/beyond the band edges, thus determining the maximum possible Fermi
        level range upon doping for this chemical potential limit.

        This is computed by obtaining the formation energy for every stable defect
        with non-zero charge, and then finding the highest Fermi level position at
        which a donor defect (positive charge) has zero formation energy (crosses
        the x-axis) - giving the lower dopability limit, and the lowest Fermi level
        position at which an acceptor defect (negative charge) has zero formation
        energy - giving the upper dopability limit.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus dopability limits).
                If ``None`` (default), will use ``self.chempots``.
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            limit (str):
                The chemical potential limit for which to
                calculate formation energies (and thus dopability limits). Can be either:

                - ``None``, in which case we search over all limits (chemical potential
                  limits) in ``chempots`` and return the most n/p-type conditions,
                  unless ``chempots`` corresponds to a single chemical potential limit.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)

        Returns:
            ``pandas`` ``DataFrame`` of dopability limits, with columns:
            "limit", "Compensating Defect", "Dopability Limit" for both p/n-type
            where 'Dopability limit' are the corresponding Fermi level positions in
            eV, relative to the VBM.
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
        self, chempots: Optional[dict] = None, limit: Optional[str] = None, el_refs: Optional[dict] = None
    ) -> pd.DataFrame:
        r"""
        Find the doping windows of the defect system, searching over all limits
        (chemical potential limits) in ``chempots`` and returning the most
        p/n-type conditions, or for a given chemical potential limit (if
        ``limit`` is set or ``chempots`` corresponds to a single chemical
        potential limit; i.e. {element symbol: chemical potential}).

        Doping window is defined by the formation energy of the lowest energy
        compensating defect species at the corresponding band edge (i.e. VBM for
        hole doping and CBM for electron doping), as these set the upper limit to
        the formation energy of dopants which could push the Fermi level close to
        the band edge without being negated by defect charge compensation.

        If a dopant has a higher formation energy than the doping window at the
        band edge, then its charge will be compensated by formation of the
        corresponding limiting defect species (rather than free carrier populations).

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus doping windows).
                If ``None`` (default), will use ``self.chempots``.
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            limit (str):
                The chemical potential limit for which to
                calculate formation energies (and thus doping windows). Can be either:

                - ``None``, in which case we search over all limits (chemical potential
                  limits) in ``chempots`` and return the most n/p-type conditions,
                  unless ``chempots`` corresponds to a single chemical potential limit.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)

        Returns:
            ``pandas`` ``DataFrame`` of doping windows, with columns:
            "limit", "Compensating Defect", "Doping Window" for both p/n-type
            where 'Doping Window' are the corresponding doping windows in eV.
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

    # TODO: Add option to only plot defect states that are stable at some point in the bandgap
    # TODO: Add option to plot formation energies at the centroid of the chemical stability region? And
    #  make this the default if no chempots are specified? Or better default to plot both the most (
    #  most-electronegative-)anion-rich and the (most-electropositive-)cation-rich chempot limits?
    # TODO: Likewise, add example showing how to plot a metastable state (above the ground state)
    # TODO: Should have similar colours for similar defect types, an option to just show amalgamated
    #  lowest energy charge states for each _defect type_) - equivalent to setting the dist_tol to
    #  infinity (but should be easier to just do here by taking the short defect name). NaP is an example
    #  for this - should have a test built for however we want to handle cases like this. See Ke's example
    #  case too with different interstitial sites.
    #   Related: Currently updating `dist_tol` to change the number of defects being plotted,
    #   can also change the colours of the different defect lines (e.g. for CdTe_wout_meta increasing
    #   `dist_tol` to 2 to merge all Te interstitials, results in the colours of other defect lines (
    #   e.g. Cd_Te) changing at the same time - ideally this wouldn't happen!
    #  TODO: optionally retain/remove unstable (in the gap) charge states (rather than current
    #  default range of (VBM - 1eV, CBM + 1eV))... depends on if shallow defect tagging with pydefect is
    #  implemented or not really, what would be best to do by default

    def plot(
        self,
        chempots: Optional[dict] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict] = None,
        chempot_table: bool = True,
        all_entries: Union[bool, str] = False,
        style_file: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        fermi_level: Optional[float] = None,
        colormap: Optional[Union[str, colors.Colormap]] = None,
        auto_labels: bool = False,
        filename: Optional[str] = None,
    ) -> Union[Figure, list[Figure]]:
        r"""
        Produce a defect formation energy vs Fermi level plot (a.k.a. a defect
        formation energy / transition level diagram). Returns the Matplotlib
        Figure object to allow further plot customisation.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. If None (default), will use ``self.chempots``.
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases
                in order to show the formal (relative) chemical potentials above the
                formation energy plot, in which case it is the formal chemical potentials
                (i.e. relative to the elemental references) that should be given here,
                otherwise the absolute (DFT) chemical potentials should be given.

                If None (default), sets all chemical potentials to zero.
                (Default: None)
            limit (str):
                The chemical potential limit for which to
                plot formation energies. Can be either:

                - None, in which case plots are generated for all limits in ``chempots``.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)
            chempot_table (bool):
                Whether to print the chemical potential table above the plot.
                (Default: True)
            all_entries (bool, str):
                Whether to plot the formation energy lines of `all` defect entries,
                rather than the default of showing only the equilibrium states at each
                Fermi level position (traditional). If instead set to "faded", will plot
                the equilibrium states in bold, and all unstable states in faded grey
                (Default: False)
            style_file (str):
                Path to a mplstyle file to use for the plot. If None (default), uses
                the default doped style (from doped/utils/doped.mplstyle).
            xlim:
                Tuple (min,max) giving the range of the x-axis (Fermi level). May want
                to set manually when including transition level labels, to avoid crossing
                the axes. Default is to plot from -0.3 to +0.3 eV above the band gap.
            ylim:
                Tuple (min,max) giving the range for the y-axis (formation energy). May
                want to set manually when including transition level labels, to avoid
                crossing the axes. Default is from 0 to just above the maximum formation
                energy value in the band gap.
            fermi_level (float):
                If set, plots a dashed vertical line at this Fermi level value, typically
                used to indicate the equilibrium Fermi level position (e.g. calculated
                with py-sc-fermi). (Default: None)
            colormap (str, matplotlib.colors.Colormap):
                Colormap to use for the formation energy lines, either as a string (i.e.
                name from https://matplotlib.org/stable/users/explain/colors/colormaps.html)
                or a Colormap / ListedColormap object. If None (default), uses `Dark2` (if
                8 or less lines) or `tab20` (if more than 8 lines being plotted).
            auto_labels (bool):
                Whether to automatically label the transition levels with their charge
                states. If there are many transition levels, this can be quite ugly.
                (Default: False)
            filename (str): Filename to save the plot to. (Default: None (not saved))

        Returns:
            Matplotlib Figure object, or list of Figure objects if multiple limits
            chosen.
        """
        from shakenbreak.plotting import _install_custom_font

        _install_custom_font()
        # check input options:
        if all_entries not in [False, True, "faded"]:
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

        style_file = style_file or f"{os.path.dirname(__file__)}/utils/doped.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        with plt.style.context(style_file):
            figs = []
            for limit in limits:
                dft_chempots = chempots["limits"][limit]
                plot_title = limit if "User" not in limit else None
                plot_filename = (
                    f"{filename.rsplit('.', 1)[0]}_{limit}.{filename.rsplit('.', 1)[1]}"
                    if filename
                    else None
                )

                with warnings.catch_warnings():  # avoid double warning about no chempots supplied
                    warnings.filterwarnings("ignore", "No chemical potentials")
                    fig = _TLD_plot(
                        self,
                        dft_chempots=dft_chempots,
                        el_refs=el_refs,
                        chempot_table=chempot_table,
                        all_entries=all_entries,
                        xlim=xlim,
                        ylim=ylim,
                        fermi_level=fermi_level,
                        title=plot_title,
                        colormap=colormap,
                        auto_labels=auto_labels,
                        filename=plot_filename,
                    )
                figs.append(fig)

            return figs[0] if len(figs) == 1 else figs

    def get_transition_levels(
        self,
        all: bool = False,
        format_charges: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of the charge transition levels for the defects in
        the DefectThermodynamics object (stored in the transition_level_map
        attribute).

        By default, only returns the thermodynamic ground-state transition
        levels (i.e. those visible on the defect formation energy diagram),
        not including metastable defect states (which can be important for
        recombination, migration, degeneracy/concentrations etc, see e.g.
        https://doi.org/10.1039/D2FD00043A & https://doi.org/10.1039/D3CS00432E).
        e.g. negative-U defects will show the 2-electron transition level
        (N+1/N-1) rather than (N+1/N) and (N/N-1).
        If instead all single-electron transition levels are desired, set
        ``all = True``.

        Returns a DataFrame with columns:

        - "Defect": Defect name
        - "Charges": Defect charge states which make up the transition level
            (as a string if ``format_charges=True``, otherwise as a list of integers)
        - "eV from VBM": Transition level position in eV from the VBM
        - "In Band Gap?": Whether the transition level is within the host band gap
        - "N(Metastable)": Number of metastable states involved in the transition level
            (0, 1 or 2). Only included if all = True.

        Args:
              all (bool):
                    Whether to print all single-electron transition levels (i.e.
                    including metastable defect states), or just the thermodynamic
                    ground-state transition levels (default).
              format_charges (bool):
                    Whether to format the transition level charge states as strings
                    (e.g. "ε(+1/+2)") or keep in list format (e.g. [1,2]).
                    (Default: True)
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
                        TL = j.get_ediff() - i.get_ediff() - self.vbm
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
                            }
                        )

        tl_df = pd.DataFrame(transition_level_map_list)
        # sort df by Defect appearance order in defect_entries, Defect, then by TL position:
        tl_df["Defect Appearance Order"] = tl_df["Defect"].map(self._name_wout_charge_appearance_order)
        tl_df = tl_df.sort_values(by=["Defect Appearance Order", "Defect", "eV from VBM"])
        tl_df = tl_df.drop(columns="Defect Appearance Order")
        return tl_df.reset_index(drop=True)

    def print_transition_levels(self, all: bool = False):
        """
        Iteratively prints the charge transition levels for the defects in the
        DefectThermodynamics object (stored in the transition_level_map
        attribute).

        By default, only returns the thermodynamic ground-state transition
        levels (i.e. those visible on the defect formation energy diagram),
        not including metastable defect states (which can be important for
        recombination, migration, degeneracy/concentrations etc, see e.g.
        https://doi.org/10.1039/D2FD00043A & https://doi.org/10.1039/D3CS00432E).
        e.g. negative-U defects will show the 2-electron transition level
        (N+1/N-1) rather than (N+1/N) and (N/N-1).
        If instead all single-electron transition levels are desired, set
        ``all = True``.

        Args:
              all (bool):
                    Whether to print all single-electron transition levels (i.e.
                    including metastable defect states), or just the thermodynamic
                    ground-state transition levels (default).
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
            for defect_name, tl_df in all_TLs_df.groupby("Defect", sort=False):
                bold_print(f"Defect: {defect_name}")
                for _, row in tl_df.iterrows():
                    if row["Charges"] != "None":
                        print(
                            f"Transition level {row['Charges']} at {row['eV from VBM']:.3f} eV above the "
                            f"VBM"
                        )
                print("")  # add space

    def get_formation_energies(
        self,
        chempots: Optional[dict] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict] = None,
        fermi_level: Optional[float] = None,
        skip_formatting: bool = False,
    ) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        r"""
        Generates defect formation energy tables (DataFrames) for either a
        single chemical potential limit (i.e. phase diagram ``limit``) or each
        limit in the phase diagram (chempots dict), depending on input
        ``limit`` and ``chempots``.

        Table Key: (all energies in eV):

        - 'Defect':
            Defect name (without charge)
        - 'q':
            Defect charge state.
        - 'ΔEʳᵃʷ':
            Raw DFT energy difference between defect and host supercell (E_defect - E_host).
        - 'qE_VBM':
            Defect charge times the VBM eigenvalue (to reference the Fermi level to the VBM)
        - 'qE_F':
            Defect charge times the Fermi level (referenced to the VBM if qE_VBM is not 0
            (if "vbm" in ``DefectEntry.calculation_metadata``)
        - 'Σμ_ref':
            Sum of reference energies of the elemental phases in the chemical potentials sum.
        - 'Σμ_formal':
            Sum of `formal` atomic chemical potential terms (Σμ_DFT = Σμ_ref + Σμ_formal).
        - 'E_corr':
            Finite-size supercell charge correction.
        - 'ΔEᶠᵒʳᵐ':
            Defect formation energy, with the specified chemical potentials and Fermi level.
            Equals the sum of all other terms.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. If None (default), will use ``self.chempots``.
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}]}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.

                If None (default), sets all chemical potentials to zero.
                (Default: None)
            limit (str):
                The chemical potential limit for which to
                tabulate formation energies. Can be either:

                - None, in which case tables are generated for all limits in ``chempots``.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.)chempots["limits"]`` dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM. The VBM value is taken from the
                calculation_metadata dict attributes of ``DefectEntry``\s in
                self.defect_entries if present, otherwise self.vbm.
                If None (default), set to the mid-gap Fermi level (E_g/2).
            skip_formatting (bool):
                Whether to skip formatting the defect charge states as
                strings (and keep as ints and floats instead).
                (default: False)

        Returns:
            ``pandas`` ``DataFrame`` or list of ``DataFrame``\s
        """
        fermi_level = self._get_and_set_fermi_level(fermi_level)
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None

        if chempots is None:
            _no_chempots_warning()
            chempots = {
                "limits": {"No User Chemical Potentials": {}},  # empty dict so is iterable (for
                # following code)
                "limits_wrt_el_refs": {"No User Chemical Potentials": {}},  # empty dict so is iterable
            }

        limit = _parse_limit(chempots, limit)
        limits = [limit] if limit is not None else list(chempots["limits"].keys())

        list_of_dfs = []
        for limit in limits:
            limits_wrt_el_refs = chempots.get("limits_wrt_el_refs") or chempots.get("limits_wrt_elt_refs")
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
        """
        Returns a defect formation energy table for a single chemical potential
        limit as a pandas ``DataFrame``.

        Table Key: (all energies in eV):

        - 'Defect':
            Defect name (without charge)
        - 'q':
            Defect charge state.
        - 'ΔEʳᵃʷ':
            Raw DFT energy difference between defect and host supercell (E_defect - E_host).
        - 'qE_VBM':
            Defect charge times the VBM eigenvalue (to reference the Fermi level to the VBM)
        - 'qE_F':
            Defect charge times the Fermi level (referenced to the VBM if qE_VBM is not 0
            (if "vbm" in ``DefectEntry.calculation_metadata``)
        - 'Σμ_ref':
            Sum of reference energies of the elemental phases in the chemical potentials sum.
        - 'Σμ_formal':
            Sum of `formal` atomic chemical potential terms (Σμ_DFT = Σμ_ref + Σμ_formal).
        - 'E_corr':
            Finite-size supercell charge correction.
        - 'ΔEᶠᵒʳᵐ':
            Defect formation energy, with the specified chemical potentials and Fermi level.
            Equals the sum of all other terms.


        Args:
            relative_chempots (dict):
                Dictionary of formal (i.e. relative to elemental reference energies)
                chemical potentials in the form ``{element symbol: energy}``.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format: ``{element symbol: reference energy}``
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential, referenced
                to the VBM (where ``"vbm"`` in ``defect_entry.calculation_metadata`` is
                used, or if not present, ``self.vbm``).
            skip_formatting (bool):
                Whether to skip formatting the defect charge states as
                strings (and keep as ints and floats instead).
                (default: False)

        Returns:
            ``pandas`` ``DataFrame`` sorted by formation energy
        """
        table = []

        defect_entries = self.defect_entries.copy()
        for defect_entry in defect_entries:
            row = [
                defect_entry.name.rsplit("_", 1)[0],  # name without charge,
                (
                    defect_entry.charge_state
                    if skip_formatting
                    else f"{'+' if defect_entry.charge_state > 0 else ''}{defect_entry.charge_state}"
                ),
            ]
            row += [defect_entry.get_ediff() - sum(defect_entry.corrections.values())]
            if "vbm" in defect_entry.calculation_metadata:
                row += [defect_entry.charge_state * defect_entry.calculation_metadata["vbm"]]
            else:
                row += [defect_entry.charge_state * self.vbm]  # type: ignore[operator]
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
                "ΔEᶠᵒʳᵐ",
                "Path",
            ],
        )

        # round all floats to 3dp:
        return formation_energy_df.round(3)

    def get_symmetries_and_degeneracies(
        self,
        skip_formatting: bool = False,
        symprec: Optional[float] = None,
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

        For interstitials, the bulk site symmetry corresponds to the
        point symmetry of the interstitial site with `no relaxation
        of the host structure`, while for vacancies/substitutions it is
        simply the symmetry of their corresponding bulk site.
        This corresponds to the point symmetry of ``DefectEntry.defect``,
        or ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``.

        Point group symmetries are taken from the calculation_metadata
        ("relaxed point symmetry" and "bulk site symmetry") if
        present (should be, if parsed with doped and defect supercell
        doesn't break host periodicity), otherwise are attempted to be
        recalculated.

        Note: doped tries to use the defect_entry.defect_supercell to determine
        the `relaxed` site symmetry. However, it should be noted that this is not
        guaranteed to work in all cases; namely for non-diagonal supercell
        expansions, or sometimes for non-scalar supercell expansion matrices
        (e.g. a 2x1x2 expansion)(particularly with high-symmetry materials)
        which can mess up the periodicity of the cell. doped tries to automatically
        check if this is the case, and will warn you if so.

        This can also be checked by using this function on your doped `generated` defects:

        .. code-block:: python

            from doped.generation import get_defect_name_from_entry
            for defect_name, defect_entry in defect_gen.items():
                print(defect_name, get_defect_name_from_entry(defect_entry, relaxed=False),
                      get_defect_name_from_entry(defect_entry), "\n")

        And if the point symmetries match in each case, then doped should be able to
        correctly determine the final relaxed defect symmetry (and orientational degeneracy)
        - otherwise periodicity-breaking prevents this.

        If periodicity-breaking prevents auto-symmetry determination, you can manually
        determine the relaxed defect and bulk-site point symmetries, and/or orientational
        degeneracy, from visualising the structures (e.g. using VESTA)(can use
        ``get_orientational_degeneracy`` to obtain the corresponding orientational
        degeneracy factor for given defect/bulk-site point symmetries) and setting the
        corresponding values in the
        ``calculation_metadata['relaxed point symmetry']/['bulk site symmetry']`` and/or
        ``degeneracy_factors['orientational degeneracy']`` attributes.
        Note that the bulk-site point symmetry corresponds to that of ``DefectEntry.defect``,
        or equivalently ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``,
        which for vacancies/substitutions is the symmetry of the corresponding bulk site,
        while for interstitials it is the point symmetry of the `final relaxed` interstitial
        site when placed in the (unrelaxed) bulk structure.
        The degeneracy factor is used in the calculation of defect/carrier concentrations
        and Fermi level behaviour (see e.g. https://doi.org/10.1039/D2FD00043A &
        https://doi.org/10.1039/D3CS00432E).

        Args:
            skip_formatting (bool):
                Whether to skip formatting the defect charge states as
                strings (and keep as ints and floats instead).
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

        for defect_entry in self.defect_entries:
            defect_entry._parse_and_set_degeneracies(symprec=symprec)
            try:
                multiplicity_per_unit_cell = defect_entry.defect.multiplicity * (
                    len(defect_entry.defect.structure.get_primitive_structure())
                    / len(defect_entry.defect.structure)
                )

            except Exception:
                multiplicity_per_unit_cell = "N/A"

            total_degeneracy = (
                reduce(lambda x, y: x * y, defect_entry.degeneracy_factors.values())
                if defect_entry.degeneracy_factors
                else "N/A"
            )

            table_list.append(
                {
                    "Defect": defect_entry.name.rsplit("_", 1)[0],  # name without charge
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
            for defect_entry in self.defect_entries
        ):
            warnings.warn(_orientational_degeneracy_warning)

        symmetry_df = pd.DataFrame(table_list)

        if not skip_formatting:
            symmetry_df["q"] = symmetry_df["q"].apply(lambda x: f"{'+' if x > 0 else ''}{x}")

        return symmetry_df.reset_index(drop=True)

    def __repr__(self):
        """
        Returns a string representation of the ``DefectThermodynamics`` object.
        """
        formula = _get_bulk_supercell(self.defect_entries[0]).composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectThermodynamics for bulk composition {formula} with {len(self.defect_entries)} "
            f"defect entries (in self.defect_entries). Available attributes:\n"
            f"{properties}\n\nAvailable methods:\n{methods}"
        )


def get_e_h_concs(fermi_dos: FermiDos, fermi_level: float, temperature: float) -> tuple[float, float]:
    """
    Get the corresponding electron and hole concentrations (in cm^-3) for a
    given Fermi level (in eV) and temperature (in K), for a ``FermiDos``
    object.

    Note that the Fermi level here is NOT referenced to the VBM! So the Fermi
    level should be the corresponding eigenvalue within the calculation (or in
    other words, the Fermi level relative to the VBM plus the VBM eigenvalue).
    """
    # code for obtaining the electron and hole concentrations here is taken from
    # FermiDos.get_doping():
    e_conc: float = np.sum(
        fermi_dos.tdos[fermi_dos.idx_cbm :]
        * f0(
            fermi_dos.energies[fermi_dos.idx_cbm :],
            fermi_level,  # type: ignore
            temperature,
        )
        * fermi_dos.de[fermi_dos.idx_cbm :],
        axis=0,
    ) / (fermi_dos.volume * fermi_dos.A_to_cm**3)
    h_conc: float = np.sum(
        fermi_dos.tdos[: fermi_dos.idx_vbm + 1]
        * f0(
            -fermi_dos.energies[: fermi_dos.idx_vbm + 1],
            -fermi_level,  # type: ignore
            temperature,
        )
        * fermi_dos.de[: fermi_dos.idx_vbm + 1],
        axis=0,
    ) / (fermi_dos.volume * fermi_dos.A_to_cm**3)

    return e_conc, h_conc


def scissor_dos(delta_gap: float, dos: Union[Dos, FermiDos], tol: float = 1e-8, verbose: bool = True):
    """
    Given an input Dos/FermiDos object, rigidly shifts the valence and
    conduction bands of the DOS object to give a band gap that is now
    increased/decreased by ``delta_gap`` eV, where this rigid scissor shift is
    applied symmetrically around the original gap (i.e. the VBM is downshifted
    by ``delta_gap/2`` and the CBM is upshifted by ``delta_gap/2``).

    Note this assumes a non-spin-polarised (i.e. non-magnetic) density
    of states!

    Args:
        delta_gap (float):
            The amount by which to increase/decrease the band gap (in eV).
        dos (Dos/FermiDos):
            The input DOS object to scissor.
        tol (float):
            The tolerance to use for determining the VBM and CBM (used in
            Dos.get_gep(tol=tol)). Default: 1e-8.
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
        print(f"Orig gap: {dos.get_gap(tol=tol)}, new gap:{dos.get_gap(tol=tol) + delta_gap}")
    scissored_dos_dict["structure"] = dos.structure.as_dict()
    if isinstance(dos, FermiDos):
        return FermiDos.from_dict(scissored_dos_dict)
    return Dos.from_dict(scissored_dos_dict)
