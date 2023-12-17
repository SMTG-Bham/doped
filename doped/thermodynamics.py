"""
Code for analysing the thermodynamics of defect formation in solids, including
calculation of formation energies as functions of Fermi level and chemical
potentials, charge transition levels, defect/carrier concentrations etc.

This code for calculating defect formation energies was originally templated
from the pyCDT (pymatgen<=2022.7.25) DefectThermodynamics code (deleted in
later versions), before heavy modification to work with doped DefectEntry
objects and add additional functionality.
"""
import os
import warnings
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from scipy.spatial import HalfspaceIntersection

from doped.chemical_potentials import get_X_poor_facet, get_X_rich_facet
from doped.core import DefectEntry
from doped.generation import _sort_defect_entries
from doped.utils.plotting import _TLD_plot

# TODO: Previous `pymatgen` issues, fixed?
#   - Currently the `PointDefectComparator` object from `pymatgen.analysis.defects.thermodynamics` is
#     used to group defect charge states for the transition level plot / transition level map outputs.
#     For interstitials, if the closest Voronoi site from the relaxed structure thus differs significantly
#     between charge states, this will give separate lines for each charge state. This is kind of ok,
#     because they _are_ actually different defect sites, but should have intelligent defaults for dealing
#     with this (see `TODO` in `thermo_from_defect_dict` in `analysis.py`; at least similar colours for
#     similar defect types, an option to just show amalgamated lowest energy charge states for each
#     _defect type_). NaP is an example for this - should have a test built for however we want to handle
#     cases like this. See Ke's example case too with different interstitial sites.
#   - GitHub issue related to `DefectThermodynamics`: https://github.com/SMTG-Bham/doped/issues/3 -> Think
#     about how we want to refactor the `DefectThermodynamics` object!
#   - Note that if you edit the entries in a DefectThermodynamics after creating it, you need to
#     `thermo.find_stable_charges()` to update the transition level map etc.


def bold_print(string: str) -> None:
    """
    Does what it says on the tin.

    Prints the input string in bold.
    """
    print("\033[1m" + string + "\033[0m")


def _raise_facet_with_user_chempots_error():
    raise ValueError(
        "You have specified an X-rich/poor facet, but the supplied chempots are not in the doped format "
        "(i.e. with `facets` in the chempots dict), and instead correspond to just a single phase diagram "
        "facet / chemical potential limit, so `facet` cannot be used here!"
    )


def _parse_facet(chempots: Dict, facet: Optional[str] = None):
    if facet is not None:
        if "rich" in facet:
            if "facets" not in chempots:  # user specified chempots
                _raise_facet_with_user_chempots_error()
            facet = get_X_rich_facet(facet.split("-")[0], chempots)
        elif "poor" in facet:
            if "facets" not in chempots:  # user specified chempots
                _raise_facet_with_user_chempots_error()
            facet = get_X_poor_facet(facet.split("-")[0], chempots)

    return facet


def _parse_chempots(chempots: Optional[Dict] = None, el_refs: Optional[Dict] = None):
    """
    Parse the chemical potentials input, formatting them in the doped format
    for use in analysis functions.

    Can be either doped format or user-specified format.

    Returns parsed chempots and el_refs
    """
    if chempots is None:
        return chempots, el_refs

    if "facets" in chempots:  # doped format, use as is and get el_refs
        return chempots, chempots.get("elemental_refs")

    # otherwise user-specified format, convert to doped format
    # TODO: Add catch later, that if no chempot is set for an element, warn user and set to 0
    chempots = {"facets": {"User Chemical Potentials": chempots}}
    if el_refs is not None:
        chempots["elemental_refs"] = el_refs
        chempots["facets_wrt_el_refs"] = {
            "User Chemical Potentials": {el: chempot - el_refs[el] for el, chempot in chempots.items()}
        }

    return chempots, el_refs


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
        defect_entries: List[DefectEntry],
        chempots: Optional[Dict] = None,
        el_refs: Optional[Dict] = None,
        vbm: Optional[float] = None,
        band_gap: Optional[float] = None,
    ):
        """
        Create a DefectThermodynamics object, which can be used to analyse the
        calculated thermodynamics of defects in solids.

        Direct initiation with DefectThermodynamics() is typically not recommended.
        Rather DefectsParser.get_defect_thermodynamics() or
        DefectThermodynamics.from_defect_dict() as shown in the doped parsing tutorials.


        Args:
            defect_entries ([DefectEntry]):
                A list of DefectEntry objects. Note that `DefectEntry.name` attributes
                 are used for grouping and plotting purposes! These should end be in
                 the format "{defect_name}_{optional_site_info}_{charge_state}". If
                 the DefectEntry.name attribute is not defined or does not end with
                 the charge state, then the entry will be renamed with the doped
                 default name.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. This can have the form of
                {"facets": [{'facet': [chempot_dict]}]} (the format generated by
                doped's chemical potential parsing functions (see tutorials)) which
                allows easy analysis over a range of chemical potentials - where facet(s)
                (chemical potential limit(s)) to analyse/plot can later be chosen using
                the `facets` argument.
                Alternatively this can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!) for a single facet (limit),
                in the format: {element symbol: chemical potential} - if manually
                specifying chemical potentials this way, you can set the `el_refs` option
                with the DFT reference energies of the elemental phases in order to show
                the formal (relative) chemical potentials above the formation energy
                plot.
                If None (default), sets all chemical potentials to zero. Chemical
                potentials can also be supplied later in each analysis function.
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                {element symbol: reference energy} (to determine the formal chemical
                potentials, when chempots has been manually specified as
                {element symbol: chemical potential}). Unnecessary if chempots is
                provided in format generated by doped (see tutorials).
                (Default: None)
            vbm (float):
                VBM eigenvalue in the bulk supercell, to use as Fermi level reference
                point for analysis. If None (default), will use "vbm" from the
                calculation_metadata dict attributes of the DefectEntry objects in
                `defect_entries`.
            band_gap (float):
                Band gap of the host, to use for analysis.
                If None (default), will use "gap" from the calculation_metadata
                dict attributes of the DefectEntry objects in `defect_entries`.
        """
        self.defect_entries = defect_entries
        self.chempots, self.el_refs = _parse_chempots(chempots, el_refs)

        # get and check VBM/bandgap values:
        def _raise_VBM_bandgap_value_error(vals, type="VBM"):
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
            bandgap_vals = []
            for defect_entry in self.defect_entries:
                if "vbm" in defect_entry.calculation_metadata:
                    vbm_vals.append(defect_entry.calculation_metadata["vbm"])
                if "gap" in defect_entry.calculation_metadata:
                    bandgap_vals.append(defect_entry.calculation_metadata["gap"])

            # get the max difference in VBM & bandgap vals:
            if max(vbm_vals) - min(vbm_vals) > 0.05 and self.vbm is None:
                _raise_VBM_bandgap_value_error(vbm_vals, type="VBM")
            elif self.vbm is None:
                self.vbm = vbm_vals[0]

            if max(bandgap_vals) - min(bandgap_vals) > 0.05 and self.band_gap is None:
                _raise_VBM_bandgap_value_error(bandgap_vals, type="bandgap")
            elif self.band_gap is None:
                self.band_gap = bandgap_vals[0]

        for i, name in [(self.vbm, "VBM eigenvalue"), (self.band_gap, "band gap value")]:
            if i is None:
                raise ValueError(
                    f"No {name} was supplied or able to be parsed from the defect entries "
                    f"(calculation_metadata attributes). Please specify the {name} in the function input."
                )

        # order entries for deterministic behaviour (particularly for plotting)
        defect_entries_dict = {entry.name: entry for entry in self.defect_entries}
        sorted_defect_entries_dict = _sort_defect_entries(defect_entries_dict)
        self.defect_entries = list(sorted_defect_entries_dict.values())
        self._parse_transition_levels()

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
            chempots=d.get(
                "chempots"
            ),  # TODO: Check if this works in each case, may need to be refactored
            el_refs=d.get("el_refs"),
            vbm=d.get("vbm"),
            band_gap=d.get("band_gap"),
        )

    def to_json(self, filename: Optional[str] = None):
        """
        Save the DefectThermodynamics object as a json file, which can be
        reloaded with the DefectThermodynamics.from_json() class method.

        Args:
            filename (str): Filename to save json file as. If None, the filename will be
                set as "{Chemical Formula}_defect_thermodynamics.json" where
                {Chemical Formula} is the chemical formula of the host material.
        """
        if filename is None:
            bulk_entry = self.defect_entries[0].bulk_entry
            if bulk_entry is not None:
                formula = bulk_entry.structure.composition.get_reduced_formula_and_factor(
                    iupac_ordering=True
                )[0]
                filename = f"{formula}_defect_thermodynamics.json"
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

    @classmethod
    def from_defect_dict(
        cls,
        defect_dict: Dict,
        chempots: Optional[Dict] = None,
        el_refs: Optional[Dict] = None,
        vbm: Optional[float] = None,
        band_gap: Optional[float] = None,
    ):
        """
        Create a DefectThermodynamics object from a dictionary of parsed defect
        calculations in the format: {"defect_name": defect_entry}), likely
        created using DefectParser.from_paths().
        Can then be used to analyse and plot the defect thermodynamics (formation
        energies, transition levels, concentrations etc).

        Note that the DefectEntry.name attributes (rather than the defect_name
        key in the defect_dict) are used to label the defects in plots.

        Args:
            defect_dict(dict):
                Dictionary of parsed defect calculations in the format:
                {"defect_name": defect_entry}), likely created using
                DefectParser.from_paths().
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. This can have the form of
                {"facets": [{'facet': [chempot_dict]}]} (the format generated by
                doped's chemical potential parsing functions (see tutorials)) which
                allows easy analysis over a range of chemical potentials - where facet(s)
                (chemical potential limit(s)) to analyse/plot can later be chosen using
                the `facets` argument.
                Alternatively this can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!) for a single facet (limit),
                in the format: {element symbol: chemical potential} - if manually
                specifying chemical potentials this way, you can set the `el_refs` option
                with the DFT reference energies of the elemental phases in order to show
                the formal (relative) chemical potentials above the formation energy
                plot.
                If None (default), sets all chemical potentials to zero. Chemical
                potentials can also be supplied later in each analysis function.
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                {element symbol: reference energy} (to determine the formal chemical
                potentials, when chempots has been manually specified as
                {element symbol: chemical potential}). Unnecessary if chempots is
                provided in format generated by doped (see tutorials).
                (Default: None)
            vbm (float):
                VBM energy to use as Fermi level reference point for analysis.
                If None (default), will use "vbm" from the calculation_metadata
                dict attributes of the parsed DefectEntry objects.
            band_gap (float):
                Band gap of the host, to use for analysis.
                If None (default), will use "gap" from the calculation_metadata
                dict attributes of the parsed DefectEntry objects.

        Returns:
            doped DefectThermodynamics object (DefectThermodynamics)
        """
        # TODO: Can we make the thermo generation more efficient? What's the bottleneck in it's
        #  initialisation? `pymatgen` site-matching that can be avoided?
        # TODO: (1) refactor the site-matching to just use the already-parsed site positions, and then
        #  merge interstitials according to this algorithm:
        # 1. For each interstitial defect type, count the number of parsed calculations per charge
        # state, and take the charge state with the most calculations present as our starting point (
        # if multiple charge states have the same number of calculations, take the closest to neutral).
        # 2. For each interstitial in a different charge state, determine which of the starting
        # points has their (already-parsed) Voronoi site closest to its (already-parsed) Voronoi
        # site, making sure to account for symmetry equivalency (using just Voronoi sites + bulk
        # structure will be easiest), and merge with this.
        # Also add option to just amalgamate and show only the lowest energy states.
        #  (2) optionally retain/remove unstable (in the gap) charge states (rather than current
        #  default range of (VBM - 1eV, CBM + 1eV))...
        # When doing this, add DOS object attribute, to then use with Alex's doped - py-sc-fermi code.
        # TODO: Should loop over input defect entries and check that the same bulk (energy and
        #  calculation_metadata) was used in each case (by proxy checks that same bulk/defect
        #  incar/potcar/kpoints settings were used in all cases, from each bulk/defect combination being
        #  checked when parsing) - if defects have been parsed separately and combined, rather than
        #  altogether with DefectsParser (which ensures the same bulk in each case)
        # TODO: Add warning if, when creating thermo, only one charge state for a defect is input (i.e. the
        #  other charge states haven't been included), in case this isn't noticed by the user. Print a
        #  list of all parsed charge states as a check if so

        if not defect_dict:
            raise ValueError(
                "No defects found in `defect_dict`. Please check the supplied dictionary is in the "
                "correct format (i.e. {'defect_name': defect_entry})."
            )
        if not isinstance(defect_dict, dict):
            raise TypeError(
                f"Expected `defect_dict` to be a dictionary, but got {type(defect_dict)} instead."
            )

        return cls(
            list(defect_dict.values()), chempots=chempots, el_refs=el_refs, vbm=vbm, band_gap=band_gap
        )

    def _get_chempots(self, chempots: Optional[Dict] = None, el_refs: Optional[Dict] = None):
        """
        Parse chemical potentials, either using input values (after formatting
        them in the doped format) or using the class attributes if set.
        """
        if chempots is not None:
            return _parse_chempots(chempots, el_refs)

        return self.chempots, self.el_refs

    def _parse_transition_levels(self):
        """
        Parses the charge transition levels for defect entries in the
        DefectThermodynamics object, and stores information about the stable
        charge states, transition levels etc. Defect entries are grouped
        together based on their DefectEntry.name attributes. These should be in
        the format "{defect_name}_{optional_site_info}_{charge_state}". If the
        DefectEntry.name attribute is not defined or does not end with the
        charge state, then the entry will be renamed with the doped default
        name.

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
        # Old pymatgen defect-matching code: # TODO: Reconsider this approach. For now, we group based
        #  on defect entry names (which themselves should contain the information on inequivalent (
        #  initial) defect sites). Could match based on the entry.defect objects as was done before,
        #  if we had a reliable way of parsing these (but in a far more efficient way than before,
        #  just checking that the structure and site are the same rather than the old,
        #  slow PointDefectComparator; Bonan did this via hashing to avoid the old approach (see
        #  archived branch, but I think with updated comparisons this is unnecessary).
        # TODO: Should have an adjustable site-displacement tolerance for matching and grouping entries?
        # (i.e. distance between (equivalent) defect sites?)
        #  Along with option to just group all grouped_defect_entries of the same type and only show the
        #  lowest energy state (equivalent to setting this displacement tolerance to infinity).
        # def similar_defects(entry_list):
        #     """
        #     Used for grouping similar grouped_defect_entries of different charges
        #     Can distinguish identical grouped_defect_entries even if they are not
        #     in same position.
        #     """
        #     pdc = PointDefectComparator(
        #         check_charge=False, check_primitive_cell=True, check_lattice_scale=False
        #     )
        #     grp_def_sets = []
        #     grp_def_indices = []
        #     for ent_ind, ent in enumerate(entry_list):
        #         # TODO: more pythonic way of grouping entry sets with PointDefectComparator.
        #         # this is currently most time intensive part of DefectThermodynamics
        #         matched_ind = None
        #         for grp_ind, defgrp in enumerate(grp_def_sets):
        #             if pdc.are_equal(ent.defect, defgrp[0].defect):
        #                 matched_ind = grp_ind
        #                 break
        #         if matched_ind is not None:
        #             grp_def_sets[matched_ind].append(copy.deepcopy(ent))
        #             grp_def_indices[matched_ind].append(ent_ind)
        #         else:
        #             grp_def_sets.append([copy.deepcopy(ent)])
        #             grp_def_indices.append([ent_ind])
        #
        #     return zip(grp_def_sets, grp_def_indices)

        def similar_defects(entry_list):
            """
            Group grouped_defect_entries based on their DefectEntry.name attributes. Defect
            entries are grouped together based on their DefectEntry.name attributes.
            These should be in the format:
            "{defect_name}_{optional_site_info}_{charge_state}". If the
            DefectEntry.name attribute is not defined or does not end with the
            charge state, then the entry will be renamed with the doped default name.

            For example, 'defect_A_1' and 'defect_A_2' will be grouped together.
            """
            from doped.analysis import check_and_set_defect_entry_name

            grouped_entries = {}  # Dictionary to hold groups of entries with the same prefix

            for i, entry in enumerate(entry_list):
                # check defect entry name and (re)define if necessary
                check_and_set_defect_entry_name(entry, entry.name)
                entry_name_wout_charge = entry.name.rsplit("_", 1)[0]

                # If the prefix is not yet in the dictionary, initialize it with empty lists
                if entry_name_wout_charge not in grouped_entries:
                    grouped_entries[entry_name_wout_charge] = {"entries": [], "indices": []}

                # Append the current entry and its index to the appropriate group
                grouped_entries[entry_name_wout_charge]["entries"].append(entry)
                grouped_entries[entry_name_wout_charge]["indices"].append(i)

            # Convert the dictionary to the desired output format
            return [(group["entries"], group["indices"]) for group in grouped_entries.values()]

        # determine defect charge transition levels:
        midgap_formation_energies = [  # without chemical potentials
            self.get_formation_energy(entry, fermi_level=0.5 * self.band_gap)
            for entry in self.defect_entries
        ]
        # set range to {min E_form - 30, max E_form +30} eV for y (formation energy), and
        # {VBM - 1, CBM + 1} eV for x (fermi level)
        min_y_lim = min(midgap_formation_energies) - 30
        max_y_lim = max(midgap_formation_energies) + 30
        limits = [[-1, self.band_gap + 1], [min_y_lim, max_y_lim]]

        stable_entries = {}
        defect_charge_map = {}
        transition_level_map = {}

        # Grouping by defect types
        for grouped_defect_entries, _index_list in similar_defects(self.defect_entries):
            # prepping coefficient matrix for half-space intersection
            # [-Q, 1, -1*(E_form+Q*VBM)] -> -Q*E_fermi+E+-1*(E_form+Q*VBM) <= 0 where E_fermi and E are
            # the variables in the hyperplanes
            hyperplanes = np.array(
                [
                    [
                        -1.0 * entry.charge_state,
                        1,
                        -1.0 * (entry.get_ediff() + entry.charge_state * self.vbm),
                    ]
                    for entry in grouped_defect_entries
                ]
            )

            border_hyperplanes = [
                [-1, 0, limits[0][0]],
                [1, 0, -1 * limits[0][1]],
                [0, -1, limits[1][0]],
                [0, 1, -1 * limits[1][1]],
            ]
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
            interior_point = [self.band_gap / 2, min(midgap_formation_energies) - 1.0]
            hs_ints = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

            # Group the intersections and corresponding facets
            ints_and_facets = zip(hs_ints.intersections, hs_ints.dual_facets)
            # Only include the facets corresponding to entries, not the boundaries
            total_entries = len(grouped_defect_entries)
            ints_and_facets = filter(
                lambda int_and_facet: all(np.array(int_and_facet[1]) < total_entries),
                ints_and_facets,
            )
            # sort based on transition level
            ints_and_facets = sorted(ints_and_facets, key=lambda int_and_facet: int_and_facet[0][0])

            defect_name_wout_charge = grouped_defect_entries[0].name.rsplit("_", 1)[
                0
            ]  # name without charge
            if any(
                defect_name_wout_charge in i for i in transition_level_map
            ):  # defects with same name, rename to prevent overwriting:
                # append "a,b,c.." for different defect species with the same name
                i = 3

                if (
                    defect_name_wout_charge in transition_level_map
                ):  # first repeat, direct match, rename previous entry
                    for output_dict in [transition_level_map, stable_entries, defect_charge_map]:
                        val = output_dict.pop(defect_name_wout_charge)
                        output_dict[f"{defect_name_wout_charge}_{chr(96 + 1)}"] = val  # a

                    defect_name_wout_charge = f"{defect_name_wout_charge}_{chr(96 + 2)}"  # b

                else:
                    defect_name_wout_charge = f"{defect_name_wout_charge}_{chr(96 + i)}"  # c

                while defect_name_wout_charge in transition_level_map:
                    i += 1
                    defect_name_wout_charge = f"{defect_name_wout_charge}_{chr(96 + i)}"  # d, e, f etc

            if len(ints_and_facets) > 0:  # unpack into lists
                _, facets = zip(*ints_and_facets)
                transition_level_map[defect_name_wout_charge] = {  # map of transition level: charge states
                    intersection[0]: [grouped_defect_entries[i].charge_state for i in facet]
                    for intersection, facet in ints_and_facets
                }
                stable_entries[defect_name_wout_charge] = [
                    grouped_defect_entries[i] for dual in facets for i in dual
                ]
                defect_charge_map[defect_name_wout_charge] = [
                    entry.charge_state for entry in grouped_defect_entries
                ]

            else:
                # if ints_and_facets is empty, then there is likely only one defect...
                if len(grouped_defect_entries) != 1:
                    # confirm formation energies dominant for one defect over other identical defects
                    name_set = [entry.name for entry in grouped_defect_entries]
                    vb_list = [
                        self.get_formation_energy(entry, fermi_level=limits[0][0])
                        for entry in grouped_defect_entries
                    ]
                    cb_list = [
                        self.get_formation_energy(entry, fermi_level=limits[0][1])
                        for entry in grouped_defect_entries
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
                    stable_entries[defect_name_wout_charge] = [grouped_defect_entries[vbm_def_index]]
                    defect_charge_map[defect_name_wout_charge] = [
                        entry.charge_state for entry in grouped_defect_entries
                    ]
                else:
                    transition_level_map[defect_name_wout_charge] = {}
                    stable_entries[defect_name_wout_charge] = [grouped_defect_entries[0]]
                    defect_charge_map[defect_name_wout_charge] = [grouped_defect_entries[0].charge_state]

        self.transition_level_map = transition_level_map
        self.transition_levels = {
            defect_name: list(defect_tls.keys())
            for defect_name, defect_tls in transition_level_map.items()
        }
        self.stable_entries = stable_entries
        self.defect_charge_map = defect_charge_map
        self.stable_charges = {
            defect_name: [entry.charge_state for entry in entries]
            for defect_name, entries in stable_entries.items()
        }

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

    # TODO: Deal with these commented out methods:
    # Doesn't work as .defect_concentration() no longer a DefectEntry method, but can be done with
    # pmg-analysis-defects FormationEnergyDiagram (or py-sc-fermi ofc)
    # def defect_concentrations(self, chemical_potentials, temperature=300, fermi_level=0.0):
    #     """
    #     Give list of all concentrations at specified efermi in the DefectThermodynamics
    #     args:
    #         chemical_potentials = {Element: number} is dict of chemical potentials to provide formation
    #             energies for temperature = temperature to produce concentrations from
    #         fermi_level: (float) is fermi level relative to valence band maximum
    #             Default efermi = 0 = VBM energy
    #     returns:
    #         list of dictionaries of defect concentrations.
    #     """
    #     return [
    #         {
    #             "conc": dfct.defect_concentration(
    #                 chemical_potentials=chemical_potentials,
    #                 temperature=temperature,
    #                 fermi_level=fermi_level,
    #             ),
    #             "name": dfct.name,
    #             "charge": dfct.charge_state,
    #         }
    #         for dfct in self.all_stable_entries
    #     ]

    # Doesn't work as .defect_concentration() no longer a DefectEntry method (required in this code),
    # but can be done with pmg-analysis-defects MultiFormationEnergyDiagram (or py-sc-fermi ofc)
    # def solve_for_fermi_energy(self, temperature, chemical_potentials, bulk_dos):
    #     """
    #     Solve for the Fermi energy self-consistently as a function of T
    #     Observations are Defect concentrations, electron and hole conc
    #     Args:
    #         temperature: Temperature to equilibrate fermi energies for
    #         chemical_potentials: dict of chemical potentials to use for calculation fermi level
    #         bulk_dos: bulk system dos (pymatgen Dos object).
    #
    #     Returns:
    #         Fermi energy dictated by charge neutrality.
    #     """
    #     fdos = FermiDos(bulk_dos, bandgap=self.band_gap)
    #     _, fdos_vbm = fdos.get_cbm_vbm()
    #
    #     def _get_total_q(ef):
    #         qd_tot = sum(
    #             d["charge"] * d["conc"]
    #             for d in self.defect_concentrations(
    #                 chemical_potentials=chemical_potentials,
    #                 temperature=temperature,
    #                 fermi_level=ef,
    #             )
    #         )
    #         qd_tot += fdos.get_doping(fermi_level=ef + fdos_vbm, temperature=temperature)
    #         return qd_tot
    #
    #     return bisect(_get_total_q, -1.0, self.band_gap + 1.0)

    # Doesn't work as .defect_concentration() no longer a DefectEntry method (required in this code),
    # and can;t be done with pmg-analysis-defects, but can with py-sc-fermi ofc.
    # TODO: Worth seeing if this code works properly (agrees with py-sc-fermi), in which case could be
    #  useful to have as an option for quick checking?
    # def solve_for_non_equilibrium_fermi_energy(
    #     self, temperature, quench_temperature, chemical_potentials, bulk_dos
    # ):
    #     """
    #     Solve for the Fermi energy after quenching in the defect concentrations
    #     at a higher temperature (the quench temperature), as outlined in P.
    #     Canepa et al (2017) Chemistry of Materials (doi:
    #     10.1021/acs.chemmater.7b02909).
    #
    #     Args:
    #         temperature: Temperature to equilibrate fermi energy at after quenching in defects
    #         quench_temperature: Temperature to equilibrate defect concentrations at (higher temperature)
    #         chemical_potentials: dict of chemical potentials to use for calculation fermi level
    #         bulk_dos: bulk system dos (pymatgen Dos object)
    #
    #     Returns:
    #         Fermi energy dictated by charge neutrality with respect to frozen in defect concentrations
    #     """
    #     high_temp_fermi_level = self.solve_for_fermi_energy(
    #         quench_temperature, chemical_potentials, bulk_dos
    #     )
    #     fixed_defect_charge = sum(
    #         d["charge"] * d["conc"]
    #         for d in self.defect_concentrations(
    #             chemical_potentials=chemical_potentials,
    #             temperature=quench_temperature,
    #             fermi_level=high_temp_fermi_level,
    #         )
    #     )
    #
    #     fdos = FermiDos(bulk_dos, bandgap=self.band_gap)
    #     _, fdos_vbm = fdos.get_cbm_vbm()
    #
    #     def _get_total_q(ef):
    #         qd_tot = fixed_defect_charge
    #         qd_tot += fdos.get_doping(fermi_level=ef + fdos_vbm, temperature=temperature)
    #         return qd_tot
    #
    #     return bisect(_get_total_q, -1.0, self.band_gap + 1.0)

    def get_formation_energy(
        self,
        defect_entry: Union[str, DefectEntry],
        chempots: Optional[dict] = None,
        facet: Optional[str] = None,
        fermi_level: float = 0,
    ):
        """
        Compute the formation energy for a DefectEntry at a given chemical
        potential limit and fermi_level. defect_entry can be a string of the
        defect name, of the DefectEntry object itself.

        Args:
            defect_entry (str or DefectEntry):
                Either a string of the defect entry name (in
                DefectThermodynamics.defect_entries), or a DefectEntry object.
                If the defect name is given without the charge state, then the
                formation energy of the lowest energy (stable) charge state
                at the chosen Fermi level will be given.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energy. If None (default), will use self.chempots.
                This can have the doped form of: {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using `facet`.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                (Default: None)
            facet (str):
                The phase diagram facet (chemical potential limit) to use for
                calculating the formation energy. Can be:
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                - None (default), if `chempots` corresponds to a single chemical
                  potential limit - otherwise will use the first chemical potential
                  limit in the doped chempots dict.
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM. The VBM value is taken from
                DefectEntry.calculation_metadata if present, otherwise
                from self.vbm.

        Returns:
            Formation energy value (float)
        """
        if isinstance(defect_entry, DefectEntry):
            return defect_entry.formation_energy(
                chempots=chempots, facet=facet, vbm=self.vbm, fermi_level=fermi_level
            )

        exact_match_defect_entries = [entry for entry in self.defect_entries if entry.name == defect_entry]
        if len(exact_match_defect_entries) == 1:
            return exact_match_defect_entries[0].formation_energy(
                chempots=chempots, facet=facet, vbm=self.vbm, fermi_level=fermi_level
            )

        if matching_defect_entries := [
            entry for entry in self.defect_entries if defect_entry in entry.name
        ]:
            return min(
                entry.formation_energy(
                    chempots=chempots,
                    facet=facet,
                    vbm=self.vbm,
                    fermi_level=fermi_level,
                )
                for entry in matching_defect_entries
            )

        raise ValueError(
            f"No matching DefectEntry with {defect_entry} in name found in "
            f"DefectThermodynamics.defect_entries, which have "
            f"names:\n{[entry.name for entry in self.defect_entries]}"
        )

    def get_dopability_limits(self, chempots: Optional[Dict] = None, facet: Optional[str] = None):
        """
        Find the dopability limits of the defect system, searching over all
        facets (chemical potential limits) in `chempots` and returning the most
        p/n-type conditions, or for a given chemical potential limit (if
        `facet` is set or `chempots` corresponds to a single chemical potential
        limit; i.e. {element symbol: chemical potential}).

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
                formation energies. If None (default), will use self.chempots.
                This can have the form of {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using `facet`.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                (Default: None)
            facet (str):
                The phase diagram facet (chemical potential limit) to use for
                calculating formation energies (and thus dopability limits). Can be:
                - None, in which case we search over all facets (chemical potential
                  limits) in `chempots` and return the most n/p-type conditions,
                  unless `chempots` corresponds to a single chemical potential limit.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                (Default: None)

        Returns:
            pandas DataFrame of dopability limits, with columns:
            "Facet", "Compensating Defect", "Dopability Limit" for both p/n-type
            where 'Dopability limit' are the corresponding Fermi level positions in
            eV, relative to the VBM.
        """
        chempots, _el_refs = self._get_chempots(chempots)  # returns self.chempots if chempots is None
        if chempots is None:
            raise ValueError(
                "No chemical potentials supplied or present in "
                "DefectThermodynamics.chempots, so dopability limits cannot be calculated."
            )

        facet = _parse_facet(chempots, facet)
        facets = [facet] if facet is not None else list(chempots["facets"].keys())

        donor_intercepts: List[Tuple] = []
        acceptor_intercepts: List[Tuple] = []

        for entry in self.all_stable_entries:
            if entry.charge_state > 0:  # donor
                # formation energy is y = mx + c where m = charge_state, c = vbm_formation_energy
                # so x-intercept is -c/m:
                donor_intercepts.extend(
                    (
                        facet,
                        entry.name,
                        -self.get_formation_energy(
                            entry, chempots=chempots["facets"][facet], fermi_level=0
                        )
                        / entry.charge_state,
                    )
                    for facet in facets
                )
            elif entry.charge_state < 0:  # acceptor
                acceptor_intercepts.extend(
                    (
                        facet,
                        entry.name,
                        -self.get_formation_energy(
                            entry, chempots=chempots["facets"][facet], fermi_level=0
                        )
                        / entry.charge_state,
                    )
                    for facet in facets
                )

        if not donor_intercepts:
            if not acceptor_intercepts:
                raise ValueError("No stable charged defects found in the system!")
            donor_intercepts = [("N/A", "N/A", -np.inf)]
        if not acceptor_intercepts:
            acceptor_intercepts = [("N/A", "N/A", np.inf)]

        donor_intercepts_df = pd.DataFrame(donor_intercepts, columns=["facet", "name", "intercept"])
        acceptor_intercepts_df = pd.DataFrame(acceptor_intercepts, columns=["facet", "name", "intercept"])

        # get the most p/n-type limit, by getting the facet with the minimum/maximum max/min-intercept,
        # where max/min-intercept is the max/min intercept for that facet (i.e. the compensating intercept)
        idx = (
            donor_intercepts_df.groupby("facet")["intercept"].transform(max)
            == donor_intercepts_df["intercept"]
        )
        limiting_donor_intercept_row = donor_intercepts_df.iloc[
            donor_intercepts_df[idx]["intercept"].idxmin()
        ]
        idx = (
            acceptor_intercepts_df.groupby("facet")["intercept"].transform(min)
            == acceptor_intercepts_df["intercept"]
        )
        limiting_acceptor_intercept_row = acceptor_intercepts_df.iloc[
            acceptor_intercepts_df[idx]["intercept"].idxmax()
        ]

        # TODO: Test this function, and warnings etc
        if limiting_donor_intercept_row["intercept"] > limiting_acceptor_intercept_row["intercept"]:
            warnings.warn(
                "Donor and acceptor doping limits intersect at negative defect formation energies "
                "(unphysical)!"
            )

        return pd.DataFrame(
            [
                [
                    limiting_donor_intercept_row["facet"],
                    limiting_donor_intercept_row["name"],
                    round(limiting_donor_intercept_row["intercept"], 3),
                ],
                [
                    limiting_acceptor_intercept_row["facet"],
                    limiting_acceptor_intercept_row["name"],
                    round(limiting_acceptor_intercept_row["intercept"], 3),
                ],
            ],
            columns=["Facet", "Compensating Defect", "Dopability Limit"],
            index=["p-type", "n-type"],
        )

    def get_doping_windows(self, chempots: Optional[Dict] = None, facet: Optional[str] = None):
        """
        Find the doping windows of the defect system, searching over all facets
        (chemical potential limits) in `chempots` and returning the most
        p/n-type conditions, or for a given chemical potential limit (if
        `facet` is set or `chempots` corresponds to a single chemical potential
        limit; i.e. {element symbol: chemical potential}).

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
                formation energies. If None (default), will use self.chempots.
                This can have the form of {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using `facet`.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                (Default: None)
            facet (str):
                The phase diagram facet (chemical potential limit) to use for
                calculating formation energies (and thus doping windows). Can be:
                - None, in which case we search over all facets (chemical potential
                  limits) in `chempots` and return the most n/p-type conditions,
                  unless `chempots` corresponds to a single chemical potential limit.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                (Default: None)

        Returns:
            pandas DataFrame of doping windows, with columns:
            "Facet", "Compensating Defect", "Doping Window" for both p/n-type
            where 'Doping Window' are the corresponding doping windows in eV.
        """
        chempots, _el_refs = self._get_chempots(chempots)  # returns self.chempots if chempots is None
        if chempots is None:
            raise ValueError(
                "No chemical potentials supplied or present in "
                "DefectThermodynamics.chempots, so doping windows cannot be calculated."
            )

        facet = _parse_facet(chempots, facet)
        facets = [facet] if facet is not None else list(chempots["facets"].keys())

        vbm_donor_intercepts: List[Tuple] = []
        cbm_acceptor_intercepts: List[Tuple] = []

        for entry in self.all_stable_entries:
            if entry.charge_state > 0:  # donor
                vbm_donor_intercepts.extend(
                    (
                        facet,
                        entry.name,
                        self.get_formation_energy(
                            entry,
                            chempots=chempots["facets"][facet],
                            fermi_level=0,
                        ),
                    )
                    for facet in facets
                )
            else:  # acceptor
                cbm_acceptor_intercepts.extend(
                    (
                        facet,
                        entry.name,
                        self.get_formation_energy(
                            entry,
                            chempots=chempots["facets"][facet],
                            fermi_level=self.band_gap,  # type: ignore[arg-type]
                        ),
                    )
                    for facet in facets
                )
        if not vbm_donor_intercepts:
            if cbm_acceptor_intercepts:
                vbm_donor_intercepts = [("N/A", "N/A", np.inf)]
            else:
                raise ValueError("No stable charged defects found in the system!")
        if not cbm_acceptor_intercepts:
            cbm_acceptor_intercepts = [("N/A", "N/A", -np.inf)]

        vbm_donor_intercepts_df = pd.DataFrame(
            vbm_donor_intercepts, columns=["facet", "name", "intercept"]
        )
        cbm_acceptor_intercepts_df = pd.DataFrame(
            cbm_acceptor_intercepts, columns=["facet", "name", "intercept"]
        )

        # get the most p/n-type limit, by getting the facet with the maximum min-intercept, where
        # min-intercept is the min intercept for that facet (i.e. the compensating intercept)
        limiting_intercept_rows = []
        for intercepts_df in [vbm_donor_intercepts_df, cbm_acceptor_intercepts_df]:
            idx = intercepts_df.groupby("facet")["intercept"].transform(min) == intercepts_df["intercept"]
            limiting_intercept_row = intercepts_df.iloc[intercepts_df[idx]["intercept"].idxmax()]
            limiting_intercept_rows.append(
                [
                    limiting_intercept_row["facet"],
                    limiting_intercept_row["name"],
                    round(limiting_intercept_row["intercept"], 3),
                ]
            )

        return pd.DataFrame(
            limiting_intercept_rows,
            columns=["Facet", "Compensating Defect", "Doping Window"],
            index=["p-type", "n-type"],
        )

    # TODO: Make a specific tutorial in docs for editing return Matplotlib figures, or with rcParams,
    #  or with a stylesheet
    # TODO: Add option to only plot defect states that are stable at some point in the bandgap
    # TODO: Add option to plot formation energies at the centroid of the chemical stability region? And
    #  make this the default if no chempots are specified? Or better default to plot both the most (
    #  most-electronegative-)anion-rich and the (most-electropositive-)cation-rich chempot limits?
    def plot(
        self,
        chempots: Optional[Dict] = None,
        facet: Optional[str] = None,
        el_refs: Optional[Dict] = None,
        chempot_table: bool = True,
        all_entries: Union[bool, str] = False,
        style_file: Optional[str] = None,
        xlim: Optional[Tuple] = None,
        ylim: Optional[Tuple] = None,
        fermi_level: Optional[float] = None,
        colormap: Union[str, colors.Colormap] = "Dark2",
        auto_labels: bool = False,
        filename: Optional[str] = None,
    ):
        """
        Produce a defect formation energy vs Fermi level plot (a.k.a. a defect
        formation energy / transition level diagram). Returns the Matplotlib
        Figure object to allow further plot customisation.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. If None (default), will use self.chempots.
                This can have the form of {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using `facet`.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                (Default: None)
            facet (str):
                The phase diagram facet (chemical potential limit) to for which to
                plot formation energies. Can be:
                - None, in which case plots are generated for all facets in chempots.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                {element symbol: reference energy} (to determine the formal chemical
                potentials, when chempots has been manually specified as
                {element symbol: chemical potential}). Unnecessary if chempots is
                provided/present in format generated by doped (see tutorials).
                (Default: None)
            chempot_table (bool):
                Whether to print the chemical potential table above the plot.
                (Default: True)
            all_entries (bool, str):
                Whether to plot the formation energy lines of _all_ defect entries,
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
                or a Colormap / ListedColormap object. (default: "Dark2")
            auto_labels (bool):
                Whether to automatically label the transition levels with their charge
                states. If there are many transition levels, this can be quite ugly.
                (Default: False)
            filename (str): Filename to save the plot to. (Default: None (not saved))

        Returns:
            Matplotlib Figure object, or list of Figure objects if multiple facets
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
                "facets": {"No User Chemical Potentials": None}
            }  # empty chempots dict to allow plotting, user will be warned

        facet = _parse_facet(chempots, facet)
        facets = [facet] if facet is not None else list(chempots["facets"].keys())

        if (
            chempots
            and facet is None
            and el_refs is None
            and "facets" not in chempots
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
            for facet in facets:
                dft_chempots = chempots["facets"][facet]
                plot_title = facet if "User" not in facet else None
                plot_filename = (
                    f"{filename.rsplit('.', 1)[0]}_{facet}.{filename.rsplit('.', 1)[1]}"
                    if filename
                    else None
                )

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

    def print_transition_levels(self):
        """
        Iteratively prints the charge transition levels for the
        DefectThermodynamics object (stored in the transition_level_map
        attribute).
        """
        for defect_name, tl_info in self.transition_level_map.items():
            bold_print(f"Defect: {defect_name}")
            for tl_efermi, chargeset in tl_info.items():
                print(
                    f"Transition Level ({max(chargeset):{'+' if max(chargeset) else ''}}/"
                    f"{min(chargeset):{'+' if min(chargeset) else ''}}) at {tl_efermi:.3f}"
                    f" eV above the VBM"
                )
            print("")  # add space

    def formation_energy_table(
        self,
        chempots: Optional[dict] = None,
        el_refs: Optional[dict] = None,
        facet: Optional[str] = None,
        fermi_level: float = 0,
    ):
        """
        Generates defect formation energy tables (DataFrames) for either a
        single chemical potential limit (i.e. phase diagram `facet`) or each
        facet in the phase diagram (chempots dict), depending on input `facet`
        and `chempots`.

        Table Key: (all energies in eV)
        'Defect' -> Defect name
        'q' -> Defect charge state.
        'ΔEʳᵃʷ' -> Raw DFT energy difference between defect and host supercell (E_defect - E_host).
        'qE_VBM' -> Defect charge times the VBM eigenvalue (to reference the Fermi level to the VBM)
        'qE_F' -> Defect charge times the Fermi level (referenced to the VBM if qE_VBM is not 0
                  (if "vbm" in DefectEntry.calculation_metadata)
        'Σμ_ref' -> Sum of reference energies of the elemental phases in the chemical potentials sum.
        'Σμ_formal' -> Sum of _formal_ atomic chemical potential terms (Σμ_DFT = Σμ_ref + Σμ_formal).
        'E_corr' -> Finite-size supercell charge correction.
        'ΔEᶠᵒʳᵐ' -> Defect formation energy, with the specified chemical potentials and Fermi level.
                    Equals the sum of all other terms.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. If None (default), will use self.chempots.
                This can have the form of {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using `facet`.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                (Default: None)
            facet (str):
                The phase diagram facet (chemical potential limit) to for which to
                tabulate formation energies. Can be:
                - None, in which case tables are generated for all facets in chempots.
                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                {element symbol: reference energy} (to determine the formal chemical
                potentials, when chempots has been manually specified as
                {element symbol: chemical potential}). Unnecessary if chempots is
                provided/present in format generated by doped (see tutorials).
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM. The VBM value is taken from the
                calculation_metadata dict attributes of `DefectEntry`s in
                self.defect_entries if present, otherwise self.vbm.
                Default = 0 (i.e. at the VBM)

        Returns:
            pandas DataFrame or list of DataFrames
        """
        chempots, el_refs = self._get_chempots(
            chempots, el_refs
        )  # returns self.chempots/self.el_refs if chempots is None

        if chempots is None:
            chempots = {
                "facets": {"No User Chemical Potentials": {}},
                "facets_wrt_el_refs": {"No User Chemical Potentials": {}},
            }

            # TODO: Warn user if no chempots? (Inaccurate energies, like with plot function?)

        facet = _parse_facet(chempots, facet)
        facets = [facet] if facet is not None else list(chempots["facets"].keys())

        list_of_dfs = []
        for facet in facets:
            dft_chempots = chempots["facets_wrt_el_refs"][facet]
            if el_refs is None:
                el_refs = (
                    {el: 0 for el in dft_chempots}
                    if chempots.get("elemental_refs") is None
                    else chempots["elemental_refs"]
                )

            single_formation_energy_df = self._single_formation_energy_table(
                dft_chempots, el_refs, fermi_level
            )
            list_of_dfs.append(single_formation_energy_df)

        return list_of_dfs[0] if len(list_of_dfs) == 1 else list_of_dfs

    def _single_formation_energy_table(
        self,
        chempots: dict,
        el_refs: dict,
        fermi_level: float = 0,
    ):
        """
        Prints a defect formation energy table for a single chemical potential
        limit (i.e. phase diagram facet), and returns the results as a pandas
        DataFrame.

        Table Key: (all energies in eV)
        'Defect' -> Defect name
        'q' -> Defect charge state.
        'ΔEʳᵃʷ' -> Raw DFT energy difference between defect and host supercell (E_defect - E_host).
        'qE_VBM' -> Defect charge times the VBM eigenvalue (to reference the Fermi level to the VBM)
        'qE_F' -> Defect charge times the Fermi level (referenced to the VBM if qE_VBM is not 0
                  (if "vbm" in DefectEntry.calculation_metadata)
        'Σμ_ref' -> Sum of reference energies of the elemental phases in the chemical potentials sum.
        'Σμ_formal' -> Sum of _formal_ atomic chemical potential terms (Σμ_DFT = Σμ_ref + Σμ_formal).
        'E_corr' -> Finite-size supercell charge correction.
        'ΔEᶠᵒʳᵐ' -> Defect formation energy, with the specified chemical potentials and Fermi level.
                    Equals the sum of all other terms.

        Args:
            chempots (dict):
                Dictionary of chosen absolute/DFT chemical potentials: {El: Energy}.
                If not specified, chemical potentials are not included in the
                formation energy calculation (all set to zero energy).
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                {element symbol: reference energy} (to determine the formal chemical
                potentials, when chempots has been manually specified as
                {element symbol: chemical potential}). Unnecessary if chempots is
                provided in format generated by doped (see tutorials).
                (Default: None)
            fermi_level (float):
                Value corresponding to the electron chemical potential, referenced
                to the VBM (where "vbm" in defect_entry.calculation_metadata is
                used, or if not present, self.vbm).

        Returns:
            pandas DataFrame sorted by formation energy
        """
        table = []

        defect_entries = self.defect_entries.copy()
        # sort by defect name, then charge state (most positive to most negative), then energy:
        defect_entries = sorted(
            defect_entries, key=lambda entry: (entry.defect.name, -entry.charge_state, entry.get_ediff())
        )
        for defect_entry in defect_entries:
            row = [
                defect_entry.name,
                defect_entry.charge_state,
            ]
            row += [defect_entry.get_ediff() - sum(defect_entry.corrections.values())]
            if "vbm" in defect_entry.calculation_metadata:
                row += [defect_entry.charge_state * defect_entry.calculation_metadata["vbm"]]
            else:
                row += [defect_entry.charge_state * self.vbm]  # type: ignore[operator]
            row += [defect_entry.charge_state * fermi_level]
            # TODO: Test this behaviour:
            row += [self._get_chempot_term(defect_entry, el_refs) if any(el_refs.values()) else "N/A"]
            row += [self._get_chempot_term(defect_entry, chempots)]
            row += [sum(defect_entry.corrections.values())]
            dft_chempots = {el: energy + el_refs[el] for el, energy in chempots.items()}
            formation_energy = self.get_formation_energy(
                defect_entry, chempots=dft_chempots, fermi_level=fermi_level
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

    def __repr__(self):
        """
        Returns a string representation of the DefectThermodynamics object.
        """
        formula = self.defect_entries[0].bulk_entry.structure.composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )[0]
        attrs = {k for k in vars(self) if not k.startswith("_")}
        methods = {k for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")}
        return (
            f"doped DefectThermodynamics for bulk composition {formula} with {len(self.defect_entries)} "
            f"defect entries (in self.defect_entries). Available attributes:\n{attrs}\n\nAvailable "
            f"methods:\n{methods}"
        )
