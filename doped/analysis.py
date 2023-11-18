"""
Code to analyse VASP defect calculations.

These functions are built from a combination of useful modules from pymatgen,
alongside substantial modification, in the efforts of making an efficient,
user-friendly package for managing and analysing defect calculations, with
publication-quality outputs.
"""
import contextlib
import io
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.sites import PeriodicSite
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Vasprun
from tqdm import tqdm

from doped import _ignore_pmg_warnings
from doped.core import DefectEntry
from doped.generation import get_defect_name_from_entry, name_defect_entries
from doped.plotting import _format_defect_name
from doped.utils.legacy_pmg.thermodynamics import DefectPhaseDiagram
from doped.utils.parsing import (
    _compare_incar_tags,
    _compare_kpoints,
    _compare_potcar_symbols,
    _get_output_files_and_check_if_multiple,
    get_defect_site_idxs_and_unrelaxed_structure,
    get_defect_type_and_composition_diff,
    get_locpot,
    get_outcar,
    get_vasprun,
    reorder_s1_like_s2,
)
from doped.utils.wyckoff import _frac_coords_sort_func
from doped.vasp import DefectDictSet


def _custom_formatwarning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    """
    Reformat warnings to just print the warning message.
    """
    return f"{message}\n"


warnings.formatwarning = _custom_formatwarning

_ANGSTROM = "\u212B"  # unicode symbol for angstrom to print in strings
_ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings

_aniso_dielectric_but_outcar_problem_warning = (
    "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to compute the "
    "_anisotropic_ Kumagai eFNV charge correction) "
)
_aniso_dielectric_but_using_locpot_warning = (
    "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) charge "
    "correction developed for _isotropic_ materials will be applied here, which corresponds to using the "
    "effective isotropic average of the supplied anisotropic dielectric. This could lead to significant "
    "errors for very anisotropic systems and/or relatively small supercells!"
)


def bold_print(string: str) -> None:
    """
    Does what it says on the tin.

    Prints the input string in bold.
    """
    print("\033[1m" + string + "\033[0m")


def _convert_dielectric_to_tensor(dielectric):
    # check if dielectric in required 3x3 matrix format
    if not isinstance(dielectric, (float, int)):
        dielectric = np.array(dielectric)
        if dielectric.shape == (3,):
            dielectric = np.diag(dielectric)
        elif dielectric.shape != (3, 3):
            raise ValueError(
                f"Dielectric constant must be a float/int or a 3x1 matrix or 3x3 matrix, "
                f"got type {type(dielectric)} and shape {dielectric.shape}"
            )

    else:
        dielectric = np.eye(3) * dielectric

    return dielectric


def check_and_set_defect_entry_name(defect_entry: DefectEntry, possible_defect_name: str) -> None:
    """
    Check that `possible_defect_name` is a recognised format by doped (i.e. in
    the format "{defect_name}_{optional_site_info}_{charge_state}").

    If the DefectEntry.name attribute is not defined or does not end with the
    charge state, then the entry will be renamed with the doped default name.
    """
    formatted_defect_name = None
    charge_state = defect_entry.charge_state
    # check if defect folder name ends with charge state:
    defect_name_w_charge_state = (
        possible_defect_name
        if (possible_defect_name.endswith((f"_{charge_state}", f"_{charge_state:+}")))
        else f"{possible_defect_name}_{'+' if charge_state > 0 else ''}{charge_state}"
    )

    with contextlib.suppress(Exception):  # check if defect name is recognised
        formatted_defect_name = _format_defect_name(
            defect_name_w_charge_state, include_site_info_in_name=True
        )  # tries without site_info if with site_info fails

    if formatted_defect_name is not None:
        defect_entry.name = defect_name_w_charge_state
    else:
        defect_entry.name = (
            f"{get_defect_name_from_entry(defect_entry, unrelaxed=True)}_"
            f"{'+' if charge_state > 0 else ''}{charge_state}"
        )  # otherwise use default doped name  # TODO: Test!
        # Note this can determine the wrong point group symmetry if a non-diagonal supercell expansion
        # was used


# TODO: Automatically pull the magnetisation from the VASP calc to determine the spin multiplicity
#  (for later integration with `py-sc-fermi`).
# TODO: Can we add functions to auto-determine the orientational degeneracy? Any decent tools for this atm?
# Note that new pymatgen Freysoldt correction requires input dielectric to be an array (list not allowed)
# Neither new nor old pymatgen FNV correction can do anisotropic dielectrics (while new sxdefectalign can)


def defect_from_structures(bulk_supercell, defect_supercell, return_all_info=False):
    """
    Auto-determines the defect type and defect site from the supplied bulk and
    defect structures, and returns a corresponding `Defect` object.

    If `return_all_info` is set to true, then also returns:
    - _relaxed_ defect site in the defect supercell
    - the defect site in the bulk supercell
    - defect site index in the defect supercell
    - bulk site index (index of defect site in bulk supercell)
    - guessed initial defect structure (before relaxation)
    - 'unrelaxed defect structure' (also before relaxation, but with interstitials at their
      final _relaxed_ positions, and all bulk atoms at their unrelaxed positions).

    Args:
        bulk_supercell (Structure):
            Bulk supercell structure.
        defect_supercell (Structure):
            Defect structure to use for identifying the defect site and type.
        return_all_info (bool):
            If True, returns additional python objects related to the
            site-matching, listed above. (Default = False)

    Returns:
        defect (Defect):
            doped Defect object.
        If `return_all_info` is True, then also:
        defect_site (Site):
            pymatgen Site object of the _relaxed_ defect site in the defect supercell.
        defect_site_in_bulk (Site):
            pymatgen Site object of the defect site in the bulk supercell
            (i.e. unrelaxed vacancy/substitution site, or final _relaxed_ interstitial
            site for interstitials).
        defect_site_index (int):
            index of defect site in defect supercell (None for vacancies)
        bulk_site_index (int):
            index of defect site in bulk supercell (None for interstitials)
        guessed_initial_defect_structure (Structure):
            pymatgen Structure object of the guessed initial defect structure.
        unrelaxed_defect_structure (Structure):
            pymatgen Structure object of the unrelaxed defect structure.
    """
    try:
        def_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, defect_supercell)
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_supercell)} in bulk vs. {len(defect_supercell)} in defect?"
        ) from exc

    # Try automatic defect site detection - this gives us the "unrelaxed" defect structure
    try:
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_supercell, defect_supercell, def_type, comp_diff
        )

    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not identify {def_type} defect site in defect structure. Try supplying the initial "
            f"defect structure to DefectParser.from_paths()."
        ) from exc

    if def_type == "vacancy":
        defect_site_in_bulk = defect_site = bulk_supercell[bulk_site_idx]
    elif def_type == "substitution":
        defect_site = defect_supercell[defect_site_idx]
        site_in_bulk = bulk_supercell[bulk_site_idx]  # this is with orig (substituted) element
        defect_site_in_bulk = PeriodicSite(
            defect_site.species, site_in_bulk.frac_coords, site_in_bulk.lattice
        )
    else:
        defect_site_in_bulk = defect_site = defect_supercell[defect_site_idx]

    if unrelaxed_defect_structure:
        if def_type == "interstitial":
            # get closest Voronoi site in bulk supercell to final interstitial site as this is likely
            # the _initial_ interstitial site
            try:
                struc_and_node_dict = loadfn("./bulk_voronoi_nodes.json")
                if not StructureMatcher(
                    stol=0.05,
                    primitive_cell=False,
                    scale=False,
                    attempt_supercell=False,
                    allow_subset=False,
                ).fit(struc_and_node_dict["bulk_supercell"], bulk_supercell):
                    warnings.warn(
                        "Previous bulk_voronoi_nodes.json detected, but does not match current bulk "
                        "supercell. Recalculating Voronoi nodes."
                    )
                    raise FileNotFoundError

                voronoi_frac_coords = struc_and_node_dict["Voronoi nodes"]

            except FileNotFoundError:  # first time parsing
                from shakenbreak.input import _get_voronoi_nodes

                voronoi_frac_coords = [site.frac_coords for site in _get_voronoi_nodes(bulk_supercell)]
                struc_and_node_dict = {
                    "bulk_supercell": bulk_supercell,
                    "Voronoi nodes": voronoi_frac_coords,
                }
                dumpfn(struc_and_node_dict, "./bulk_voronoi_nodes.json")  # for efficient
                # parsing of multiple defects at once
                print(
                    "Saving parsed Voronoi sites (for interstitial site-matching) to "
                    "bulk_voronoi_sites.json to speed up future parsing."
                )

            closest_node_frac_coords = min(
                voronoi_frac_coords,
                key=lambda node: defect_site.distance_and_image_from_frac_coords(node)[0],
            )
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()
            int_site = guessed_initial_defect_structure[defect_site_idx]
            guessed_initial_defect_structure.remove_sites([defect_site_idx])
            guessed_initial_defect_structure.insert(
                defect_site_idx,  # Place defect at same position as in DFT calculation
                int_site.species_string,
                closest_node_frac_coords,
                coords_are_cartesian=False,
                validate_proximity=True,
            )
            guessed_initial_defect_structure[defect_site_idx]

        else:
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()

        # ensure unrelaxed_defect_structure ordered to match defect_structure:
        unrelaxed_defect_structure = reorder_s1_like_s2(unrelaxed_defect_structure, defect_supercell)

    else:
        warnings.warn(
            "Cannot determine the unrelaxed `initial_defect_structure`. Please ensure the "
            "`initial_defect_structure` is indeed unrelaxed."
        )

    for_monty_defect = {  # initialise doped Defect object, needs to use defect site in bulk (which for
        # substitutions differs from defect_site)
        "@module": "doped.core",
        "@class": def_type.capitalize(),
        "structure": bulk_supercell,
        "site": defect_site_in_bulk,
    }  # note that we now define the Defect in the bulk supercell, rather than the primitive structure
    # as done during generation. Future work could try mapping the relaxed defect site back to the
    # primitive cell, however interstitials will be very tricky for this...
    defect = MontyDecoder().process_decoded(for_monty_defect)

    if not return_all_info:
        return defect

    return (
        defect,
        defect_site,
        defect_site_in_bulk,
        defect_site_idx,
        bulk_site_idx,
        guessed_initial_defect_structure,
        unrelaxed_defect_structure,
    )


def defect_name_from_structures(bulk_structure, defect_structure):
    # TODO: Test this using DefectsGenerator outputs
    """
    Get the doped/SnB defect name using the bulk and defect structures.

    Args:
        bulk_structure (Structure):
            Bulk (pristine) structure.
        defect_structure (Structure):
            Defect structure.

    Returns:
        str: Defect name.
    """
    from doped.generation import get_defect_name_from_defect

    defect = defect_from_structures(bulk_structure, defect_structure)

    # note that if the symm_op approach fails for any reason here, the defect-supercell expansion
    # approach will only be valid if the defect structure is a diagonal expansion of the primitive...

    return get_defect_name_from_defect(defect)


def _determine_defect_charge_from_vasprun(defect_vr: Vasprun, charge_state: Optional[int]):
    try:
        defect_nelect = defect_vr.incar.get("NELECT", None)
        if defect_nelect is None:
            auto_charge = 0  # neutral defect if NELECT not specified
        else:
            potcar_symbols = [titel.split()[1] for titel in defect_vr.potcar_symbols]
            potcar_settings = {symbol.split("_")[0]: symbol for symbol in potcar_symbols}
            with warnings.catch_warnings():  # ignore POTCAR warnings if not available
                warnings.simplefilter("ignore", UserWarning)
                neutral_defect_dict_set = DefectDictSet(
                    defect_vr.structures[-1],
                    charge_state=0,
                    user_potcar_settings=potcar_settings,
                )
            try:
                auto_charge = -1 * (defect_nelect - neutral_defect_dict_set.nelect)

            except Exception as e:
                auto_charge = None
                if charge_state is None:
                    raise RuntimeError(
                        "Defect charge cannot be automatically determined as POTCARs have not been setup "
                        "with pymatgen (see Step 2 at https://github.com/SMTG-Bham/doped#installation). "
                        "Please specify defect charge manually using the `charge_state` argument, "
                        "or set up POTCARs with pymatgen."
                    ) from e

            if auto_charge is not None and abs(auto_charge) >= 10:  # crazy charge state predicted
                raise RuntimeError(
                    f"Auto-determined defect charge q={int(auto_charge):+} is unreasonably large. "
                    f"Please specify defect charge manually using the `charge` argument."
                )

        if (
            charge_state is not None
            and auto_charge is not None
            and int(charge_state) != int(auto_charge)
            and abs(auto_charge) < 5
        ):
            warnings.warn(
                f"Auto-determined defect charge q={int(auto_charge):+} does not match specified charge "
                f"q={int(charge_state):+}. Will continue with specified charge_state, but beware!"
            )
        elif charge_state is None and auto_charge is not None:
            charge_state = auto_charge

    except Exception as e:
        if charge_state is None:
            raise e

    if charge_state is None:
        raise RuntimeError(
            "Defect charge could not be automatically determined from the defect calculation outputs. "
            "Please manually specify defect charge using the `charge_state` argument."
        )

    return charge_state


def defect_entry_from_paths(
    defect_path: str,
    bulk_path: str,
    dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
    charge_state: Optional[int] = None,
    initial_defect_structure_path: Optional[str] = None,
    skip_corrections: bool = False,
    error_tolerance: float = 0.05,
    bulk_bandgap_path: Optional[str] = None,
    **kwargs,
):
    """
    Parse the defect calculation outputs in `defect_path` and return the parsed
    `DefectEntry` object. By default, the `DefectEntry.name` attribute (later
    used to label the defects in plots) is set to the defect_path folder name
    (if it is a recognised defect name), else it is set to the default doped
    name for that defect.

    Note that the bulk and defect supercells should have the same definitions/basis
    sets (for site-matching and finite-size charge corrections to work appropriately).

    Args:
        defect_path (str):
            Path to defect supercell folder (containing at least vasprun.xml(.gz)).
        bulk_path (str):
            Path to bulk supercell folder (containing at least vasprun.xml(.gz)).
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Ionic + static contributions to the dielectric constant. If not provided,
            charge corrections cannot be computed and so `skip_corrections` will be
            set to true.
        charge_state (int):
            Charge state of defect. If not provided, will be automatically determined
            from the defect calculation outputs (requires `POTCAR`s to be set up
            with `pymatgen`).
        initial_defect_structure_path (str):
            Path to the initial/unrelaxed defect structure. Only recommended for use
            if structure matching with the relaxed defect structure(s) fails (rare).
            Default is None.
        skip_corrections (bool):
            Whether to skip the calculation and application of finite-size charge
            corrections to the defect energy (not recommended in most cases).
            Default = False.
        error_tolerance (float):
            If the estimated error in the defect charge correction is greater
            than this value (in eV), then a warning is raised. (default: 0.05 eV)
        bulk_bandgap_path (str):
            Path to bulk OUTCAR file for determining the band gap. If the VBM/CBM
            occur at reciprocal space points not included in the bulk supercell
            calculation, you should use this tag to point to a bulk bandstructure
            calculation instead. Alternatively, you can edit/add the "gap" and "vbm"
            entries in self.defect_entry.calculation_metadata to match the correct
            (eigen)values.
            If None, will use DefectEntry.calculation_metadata["bulk_path"].
        **kwargs:
            Keyword arguments to pass to `DefectParser()` methods
            (`load_FNV_data()`, `load_eFNV_data()`, `load_bulk_gap_data()`)
            such as `bulk_locpot_dict`, `bulk_site_potentials` etc.

    Return:
        Parsed `DefectEntry` object.
    """
    dp = DefectParser.from_paths(
        defect_path,
        bulk_path,
        dielectric=dielectric,
        charge_state=charge_state,
        initial_defect_structure_path=initial_defect_structure_path,
        skip_corrections=skip_corrections,
        error_tolerance=error_tolerance,
        bulk_bandgap_path=bulk_bandgap_path,
        **kwargs,
    )
    return dp.defect_entry


def dpd_from_defect_dict(defect_dict: dict) -> DefectPhaseDiagram:
    """
    Generates a DefectPhaseDiagram object from a dictionary of parsed defect
    calculations in the format: {"defect_name": defect_entry}), likely created
    using defect_entry_from_paths() (or DefectParser), which can then be used to
    analyse and plot the defect thermodynamics (formation energies, transition
    levels, concentrations etc).
    Note that the DefectEntry.name attributes (rather than the defect_name key
    in the defect_dict) are used to label the defects in plots.

    Args:
        defect_dict (dict):
            Dictionary of parsed defect calculations in the format:
            {"defect_name": defect_entry}), likely created using defect_entry_from_paths()
            (or DefectParser). Must have 'vbm' and 'gap' in
            defect_entry.calculation_metadata for at least one defect (from
            DefectParser.load_bulk_gap_data())

    Returns:
        doped DefectPhaseDiagram object (DefectPhaseDiagram)
    """
    # TODO: Can we make the dpd generation more efficient? What's the bottleneck in it's
    #  initialisation? `pymatgen` site-matching that can be avoided?
    # TODO: Write our own DefectPhaseDiagram class, to (1) refactor the site-matching to just
    #  use the already-parsed site positions, and then merge interstitials according to this
    #  algorithm:
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
    # With this, `dpd_from_defect_dict()` should then be a classmethod
    # TODO: Should loop over input defect entries and check that the same bulk (energy and
    #  calculation_metadata) was used in each case (by proxy checks that same bulk/defect
    #  incar/potcar/kpoints settings were used in all cases, from each bulk/defect combination being
    #  checked when parsing) - if defects have been parsed separately and combined, rather than
    #  altogether with DefectsParser (which ensures the same bulk in each case)
    # TODO: Add warning if, when creating dpd, only one charge state for a defect is input (i.e. the
    #  other charge states haven't been included), in case this isn't noticed by the user. Print a list of
    #  all parsed charge states as a check if so

    if not defect_dict:
        raise ValueError(
            "No defects found in `defect_dict`. Please check the supplied dictionary is in the "
            "correct format (i.e. {'defect_name': defect_entry})."
        )
    if not isinstance(defect_dict, dict):
        raise TypeError(f"Expected `defect_dict` to be a dictionary, but got {type(defect_dict)} instead.")

    vbm_vals = []
    bandgap_vals = []
    for defect_entry in defect_dict.values():
        if "vbm" in defect_entry.calculation_metadata:
            vbm_vals.append(defect_entry.calculation_metadata["vbm"])
        if "gap" in defect_entry.calculation_metadata:
            bandgap_vals.append(defect_entry.calculation_metadata["gap"])

    def _raise_VBM_bandgap_value_error(vals, type="VBM"):
        raise ValueError(
            f"{type} values for defects in `defect_dict` do not match within 0.05 eV of each other, "
            f"and so are incompatible for thermodynamic analysis with DefectPhaseDiagram. The {type} "
            f"values in the dictionary are: {vals}. You should recheck the correct/same bulk files were "
            f"used when parsing."
        )

    # get the max difference in VBM & bandgap vals:
    if max(vbm_vals) - min(vbm_vals) > 0.05:  # Check if all defects give same vbm
        _raise_VBM_bandgap_value_error(vbm_vals, type="VBM")

    if max(bandgap_vals) - min(bandgap_vals) > 0.05:  # Check if all defects give same bandgap
        _raise_VBM_bandgap_value_error(bandgap_vals, type="bandgap")

    return DefectPhaseDiagram(list(defect_dict.values()), vbm_vals[0], bandgap_vals[0])


def dpd_transition_levels(defect_phase_diagram: DefectPhaseDiagram):
    """
    Iteratively prints the charge transition levels for the input
    DefectPhaseDiagram object (via the from a
    defect_phase_diagram.transition_level_map attribute).

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
            DefectPhaseDiagram object (likely created from
            analysis.dpd_from_defect_dict)

    Returns:
        None
    """
    for def_type, tl_info in defect_phase_diagram.transition_level_map.items():
        bold_print(f"\nDefect: {def_type.split('@')[0]}")
        for tl_efermi, chargeset in tl_info.items():
            print(
                f"Transition Level ({max(chargeset):{'+' if max(chargeset) else ''}}/"
                f"{min(chargeset):{'+' if min(chargeset) else ''}}) at {tl_efermi:.3f}"
                f" eV above the VBM"
            )


def formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempots: Optional[Dict] = None,
    el_refs: Optional[Dict] = None,
    facets: Optional[List] = None,
    fermi_level: float = 0,
):
    """
    Generates defect formation energy tables (DataFrames) for either a
    single chemical potential limit (i.e. phase diagram facet) or each
    facet in the phase diagram (chempots dict), depending on the chempots
    input supplied. This can either be a dictionary of chosen absolute/DFT
    chemical potentials: {Element: Energy} (giving a single formation
    energy table) or a dictionary including the key-value pair: {"facets":
    [{'facet': [chempot_dict]}]}, following the doped format. In the
    latter case, a subset of facet(s) / chemical potential limit(s)
    can be chosen with the facets argument, or if not specified, will
    print formation energy tables for each facet in the phase diagram.

    Returns the results as a pandas DataFrame or list of DataFrames.

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
        defect_phase_diagram (DefectPhaseDiagram):
            DefectPhaseDiagram for which to plot defect formation energies
            (typically created from analysis.dpd_from_defect_dict).
        chempots (dict):
            Dictionary of chemical potentials to use for calculating the defect
            formation energies. This can have the form of
            {"facets": [{'facet': [chempot_dict]}]} (the format generated by
            doped's chemical potential parsing functions (see tutorials)) and
            facet(s) (chemical potential limit(s)) to tabulate can be chosen using
            `facets`, or a dictionary of **DFT**/absolute chemical potentials
            (not formal chemical potentials!), in the format:
            {element symbol: chemical potential} - if manually specifying
            chemical potentials this way, you can set the el_refs option with
            the DFT reference energies of the elemental phases in order to show
            the formal (relative) chemical potentials as well.
            (Default: None)
        facets (list, str):
            A string or list of facet(s) (chemical potential limit(s)) for which
            to tabulate the defect formation energies, corresponding to 'facet' in
            {"facets": [{'facet': [chempot_dict]}]} (the format generated by
            doped's chemical potential parsing functions (see tutorials)). If
            not specified, will tabulate for each facet in `chempots`. (Default: None)
        el_refs (dict):
            Dictionary of elemental reference energies for the chemical potentials
            in the format:
            {element symbol: reference energy} (to determine the formal chemical
            potentials, when chempots has been manually specified as
            {element symbol: chemical potential}). Unnecessary if chempots is
            provided in format generated by doped (see tutorials).
            (Default: None)
        fermi_level (float):
            Value corresponding to the electron chemical potential. If "vbm" is
            supplied in DefectEntry.calculation_metadata, then fermi_level is
            referenced to the VBM. If "vbm" is NOT supplied in calculation_metadata,
            then fermi_level is referenced to the calculation's absolute DFT
            potential (and should include the vbm value provided by a band structure
            calculation). Default = 0 (i.e. at the VBM)

    Returns:
        pandas DataFrame or list of DataFrames
    """
    if chempots is None:
        chempots = {}

    if "facets_wrt_el_refs" in chempots:
        list_of_dfs = []
        if facets is None:
            facets = chempots["facets"].keys()  # Phase diagram facets to use for chemical
            # potentials, to tabulate formation energies
        for facet in facets:
            single_formation_energy_df = _single_formation_energy_table(
                defect_phase_diagram,
                chempots=chempots["facets_wrt_el_refs"][facet],
                el_refs=chempots["elemental_refs"],
                fermi_level=fermi_level,
            )
            list_of_dfs.append(single_formation_energy_df)

        return list_of_dfs[0] if len(list_of_dfs) == 1 else list_of_dfs

    # else return {El: Energy} dict for chempot_limits, or if unspecified, all zero energy
    single_formation_energy_df = _single_formation_energy_table(
        defect_phase_diagram,
        chempots=chempots,
        el_refs={el: 0 for el in chempots} if el_refs is None else el_refs,
        fermi_level=fermi_level,
    )
    return single_formation_energy_df


def _single_formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempots: Dict,
    el_refs: Dict,
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
        defect_phase_diagram (DefectPhaseDiagram):
            DefectPhaseDiagram for which to plot defect formation energies
            (typically created from analysis.dpd_from_defect_dict).
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
            Value corresponding to the electron chemical potential. If "vbm" is
            supplied in DefectEntry.calculation_metadata, then fermi_level is
            referenced to the VBM. If "vbm" is NOT supplied in calculation_metadata,
            then fermi_level is referenced to the calculation's absolute DFT
            potential (and should include the vbm value provided by a band structure
            calculation). Default = 0 (i.e. at the VBM)

    Returns:
        pandas DataFrame sorted by formation energy
    """
    table = []

    defect_entries = defect_phase_diagram.entries
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
            row += [0]
        row += [defect_entry.charge_state * fermi_level]
        row += [defect_phase_diagram._get_chempot_term(defect_entry, el_refs)]
        row += [defect_phase_diagram._get_chempot_term(defect_entry, chempots)]
        row += [sum(defect_entry.corrections.values())]
        dft_chempots = {el: energy + el_refs[el] for el, energy in chempots.items()}
        formation_energy = defect_phase_diagram._formation_energy(
            defect_entry, chemical_potentials=dft_chempots, fermi_level=fermi_level
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


def _update_defect_entry_charge_corrections(defect_entry, charge_correction_type):
    meta = defect_entry.calculation_metadata[f"{charge_correction_type}_meta"]
    corr = (
        meta[f"{charge_correction_type}_electrostatic"]
        + meta[f"{charge_correction_type}_potential_alignment_correction"]
    )
    defect_entry.corrections.update({f"{charge_correction_type}_charge_correction": corr})


def _multiple_files_warning(file_type, directory, chosen_filepath, action, dir_type="bulk"):
    warnings.warn(
        f"Multiple `{file_type}` files found in {dir_type} directory: {directory}. Using "
        f"{chosen_filepath} to {action}"
    )


class DefectsParser:
    def __init__(
        self,
        output_path: str = ".",
        dielectric: Optional[Union[float, int, np.ndarray]] = None,
        subfolder: Optional[str] = None,
        bulk_path: Optional[str] = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        bulk_bandgap_path: Optional[str] = None,
        processes: Optional[int] = None,
        json_filename: Optional[str] = None,
    ):
        """
        A class for rapidly parsing multiple VASP defect supercell calculations
        for a given host (bulk) material.

        Loops over calculation directories in `output_path` (likely the same
        `output_path` used with `DefectsSet` for file generation in `doped.vasp`)
        and parses the defect calculations into a dictionary of:
        {defect_name: DefectEntry}, where the defect_name is set to the defect
        calculation folder name (_if it is a recognised defect name_), else it is
        set to the default `doped` name for that defect. By default, searches for
        folders in `output_path` with `subfolder` containing `vasprun.xml(.gz)`
        files, and tries to parse them as `DefectEntry`s.

        By default, tries to use multiprocessing to speed up defect parsing, which
        can be controlled with the `processes` parameter.

        Defect charge states are automatically determined from the defect calculation
        outputs if `POTCAR`s are set up with `pymatgen` (see docs Installation page),
        or if that fails, using the defect folder name (must end in "_+X" or "_-X"
        where +/-X is the defect charge state).

        Uses the (single) `DefectParser` class to parse the individual defect
        calculations. Note that the bulk and defect supercells should have the same
        definitions/basis sets (for site-matching and finite-size charge corrections
        to work appropriately).

        Args:
            output_path (str):
                Path to the output directory containing the defect calculation
                folders (likely the same `output_path` used with `DefectsSet` for
                file generation in `doped.vasp`). Default = current directory.
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Ionic + static contributions to the dielectric constant. If not provided,
                charge corrections cannot be computed and so `skip_corrections` will be
                set to true.
            subfolder (str):
                Name of subfolder(s) within each defect calculation folder (in the
                `output_path` directory) containing the VASP calculation files to
                parse (e.g. `vasp_ncl`, `vasp_std`, `vasp_gam` etc.). If not
                specified, `doped` checks first for `vasp_ncl`, `vasp_std`, `vasp_gam`
                subfolders with calculation outputs (`vasprun.xml(.gz)` files) and uses
                the highest level VASP type (ncl > std > gam) found as `subfolder`,
                otherwise uses the defect calculation folder itself with no subfolder
                (set `subfolder = "."` to enforce this).
            bulk_path (str):
                Path to bulk supercell reference calculation folder. If not specified,
                searches for folder with name "X_bulk" in the `output_path` directory
                (matching the default `doped` name for the bulk supercell reference folder).
            skip_corrections (bool):
                Whether to skip the calculation and application of finite-size charge
                corrections to the defect energies (not recommended in most cases).
                Default = False.
            error_tolerance (float):
                If the estimated error in any charge correction is greater than
                this value (in eV), then a warning is raised. (default: 0.05 eV)
            bulk_bandgap_path (str):
                Path to bulk OUTCAR file for determining the band gap. If the VBM/CBM
                occur at reciprocal space points not included in the bulk supercell
                calculation, you should use this tag to point to a bulk bandstructure
                calculation instead. Alternatively, you can edit/add the "gap" and "vbm"
                entries in DefectParser.defect_entry.calculation_metadata to match the
                correct (eigen)values.
                If None, will calculate "gap"/"vbm" using the outputs at:
                DefectParser.defect_entry.calculation_metadata["bulk_path"]
            processes (int):
                Number of processes to use for multiprocessing for expedited parsing.
                If not set, defaults to one less than the number of CPUs available.
            json_filename (str):
                Filename to save the parsed defect entries dict (`DefectsParser.defect_dict`)
                to, to avoid having to re-parse defects when later analysing further and
                aiding calculation provenance. Can be reloaded using the `loadfn` function
                from `monty.serialization` as shown in the docs, or
                `DefectPhaseDiagram.from_json()`. If None (default), set as
                "{Chemical Formula}_defect_dict.json" where {Chemical Formula} is the
                chemical formula of the host material. If False, no json file is saved.

        Attributes:
            defect_dict (dict):
                Dictionary of parsed defect calculations in the format:
                {"defect_name": DefectEntry}) where the defect_name is set to the
                defect calculation folder name (_if it is a recognised defect name_),
                else it is set to the default `doped` name for that defect.
        """
        # TODO: Need to add `DefectPhaseDiagram.from_json()` etc methods as mention in docstring here
        # TODO: Update tutorials to show DefectsParser and DefectParser (for finer control)
        self.output_path = output_path
        self.dielectric = dielectric
        self.skip_corrections = skip_corrections
        self.error_tolerance = error_tolerance
        self.bulk_path = bulk_path
        self.subfolder = subfolder
        self.bulk_bandgap_path = bulk_bandgap_path
        self.processes = processes

        possible_defect_folders = [
            dir
            for dir in os.listdir(self.output_path)
            if os.path.isdir(os.path.join(self.output_path, dir))
            and any(
                "vasprun.xml" in file
                for file_list in [tup[2] for tup in os.walk(os.path.join(self.output_path, dir))]
                for file in file_list
            )
        ]

        if self.subfolder is None:  # determine subfolder to use
            vasp_subfolders = [
                subdir
                for possible_defect_folder in possible_defect_folders
                for subdir in os.listdir(os.path.join(self.output_path, possible_defect_folder))
                if os.path.isdir(os.path.join(self.output_path, possible_defect_folder, subdir))
                and "vasp_" in subdir
            ]
            vasp_type_count_dict = {  # Count Dik
                i: len([subdir for subdir in vasp_subfolders if i in subdir])
                for i in ["vasp_ncl", "vasp_std", "vasp_gam"]
            }
            # take first entry with non-zero count, else use defect folder itself:
            self.subfolder = next((subdir for subdir, count in vasp_type_count_dict.items() if count), ".")

        possible_bulk_folders = [dir for dir in possible_defect_folders if "bulk" in dir]
        if self.bulk_path is None:  # determine bulk_path to use
            if len(possible_bulk_folders) == 1:
                self.bulk_path = os.path.join(self.output_path, possible_bulk_folders[0])
            elif len([dir for dir in possible_bulk_folders if dir.endswith("_bulk")]) == 1:
                self.bulk_path = os.path.join(
                    self.output_path, [dir for dir in possible_bulk_folders if dir.endswith("_bulk")][0]
                )
            else:
                raise ValueError(
                    f"Could not automatically determine bulk supercell calculation folder in "
                    f"{self.output_path}, found {len(possible_bulk_folders)} folders containing "
                    f"`vasprun.xml(.gz)` files (in subfolders) and 'bulk' in the folder name. Please "
                    f"specify `bulk_path` manually."
                )

        # add subfolder to bulk_path if present with vasprun.xml(.gz), otherwise use bulk_path as is:
        if os.path.isdir(os.path.join(self.bulk_path, self.subfolder)) and any(
            "vasprun.xml" in file for file in os.listdir(os.path.join(self.bulk_path, self.subfolder))
        ):
            self.bulk_path = os.path.join(self.bulk_path, self.subfolder)

        self.defect_dict = {}
        self.defect_folders = [
            dir
            for dir in possible_defect_folders
            if dir not in possible_bulk_folders
            and self.subfolder in os.listdir(os.path.join(self.output_path, dir))
        ]

        self.bulk_corrections_data = {  # so we only load and parse bulk data once
            "bulk_locpot_dict": None,
            "bulk_site_potentials": None,
        }

        if self.processes is None:  # multiprocessing?
            self.processes = min(max(1, cpu_count() - 1), len(self.defect_folders) - 1)  # only
            # multiprocess as much as makes sense, if only a handful of defect folders

        if self.processes <= 1:  # no multiprocessing
            with tqdm(self.defect_folders, desc="Parsing defect calculations") as pbar:
                for defect_folder in pbar:
                    pbar.set_description(f"Parsing {defect_folder}/{self.subfolder}".replace("/.", ""))
                    # set tqdm progress bar description to defect folder being parsed:
                    parsed_defect_entry = self._parse_defect(defect_folder)
                    if parsed_defect_entry is not None:
                        self.defect_dict[parsed_defect_entry.name] = parsed_defect_entry

            return

        # otherwise multiprocessing:
        # guess a charged defect in defect_folders, to try initially check if dielectric and corrections
        # correctly set, before multiprocessing with the same settings for all folders:
        charged_defect_folder = None
        for possible_charged_defect_folder in self.defect_folders:
            with contextlib.suppress(Exception):
                if abs(int(possible_charged_defect_folder[-1])) > 0:  # likely charged defect
                    charged_defect_folder = possible_charged_defect_folder

        parsing_warnings = []
        defect_renaming_list = []
        pbar = tqdm(total=len(self.defect_folders))
        try:

            def _update_defect_dict_and_return_warnings_from_parsing(result, pbar):
                pbar.update()
                if result[0] is not None:
                    defect_folder = result[0].calculation_metadata["defect_path"].split("/")[-2]
                    pbar.set_description(f"Parsing {defect_folder}/{self.subfolder}".replace("/.", ""))
                    if result[0].name not in list(self.defect_dict.keys()):
                        self.defect_dict[result[0].name] = result[0]
                    else:  # add both to defect renaming list, to be renamed later:
                        defect_renaming_list.append(self.defect_dict.pop(result[0].name))
                        defect_renaming_list.append(result[0])

                    if result[1]:
                        return (
                            f"Warning(s) encountered when parsing {result[0].name} at "
                            f"{result[0].calculation_metadata['defect_path']}:\n{result[1]}"
                        )

                if result[1]:  # should be failed parsing warning if result[0] is None:
                    return result[1]

                return ""

            if charged_defect_folder is not None:
                # will throw warnings if dielectric is None / charge corrections not possible,
                # and set self.skip_corrections appropriately
                pbar.set_description(  # set this first as desc is only set after parsing run in function
                    f"Parsing {charged_defect_folder}/{self.subfolder}".replace("/.", "")
                )
                parsing_warnings.append(
                    _update_defect_dict_and_return_warnings_from_parsing(
                        self._multiprocess_parse_defect(charged_defect_folder), pbar
                    )
                )

            folders_to_process = [
                folder for folder in self.defect_folders if folder != charged_defect_folder
            ]
            pbar.set_description("Setting up multiprocessing")
            if self.processes > 1:
                with Pool(processes=self.processes) as pool:  # result is parsed_defect_entry, warnings
                    results = pool.imap_unordered(self._multiprocess_parse_defect, folders_to_process)
                    for result in results:
                        parsing_warnings.append(
                            _update_defect_dict_and_return_warnings_from_parsing(result, pbar)
                        )

        except Exception as exc:
            pbar.close()
            raise exc

        finally:
            pbar.close()

        if defect_renaming_list:  # TODO: Add test for this
            with contextlib.suppress(AttributeError):  # sort by conventional cell fractional
                # coordinates if these are defined, to aid deterministic naming:
                defect_renaming_list.sort(key=lambda x: _frac_coords_sort_func(x.conv_cell_frac_coords))

            new_named_defect_entries_dict = name_defect_entries(defect_renaming_list)
            # set name attribute: (these are names without charges!)
            for defect_name_wout_charge, defect_entry in new_named_defect_entries_dict.items():
                defect_entry.name = (
                    f"{defect_name_wout_charge}_{'+' if defect_entry.charge_state > 0 else ''}"
                    f"{defect_entry.charge_state}"
                )

            # if any duplicate names, crash (and burn, b...)
            duplicate_names = [
                defect_entry.name
                for defect_entry in defect_renaming_list
                if defect_entry.name in list(self.defect_dict.values())
            ]
            if duplicate_names:
                raise ValueError(
                    f"Some defect entries have the same name, due to mixing of doped-named and unnamed "
                    f"defect folders. This would cause defect entries to be overwritten. Please check "
                    f"your defect folder names in `output_path`!\nDuplicate defect names:\n"
                    f"{duplicate_names}"
                )

        parsing_warnings = [warning for warning in parsing_warnings if warning]  # remove empty strings
        if parsing_warnings:
            warnings.warn("\n".join(parsing_warnings))

        # check if same type of charge correction was used in each case or not:
        if (
            len(
                {
                    k
                    for defect_entry in self.defect_dict.values()
                    for k in defect_entry.corrections
                    if k.endswith("_charge_correction")
                }
            )
            > 1
        ):
            warnings.warn(
                "Beware: The Freysoldt (FNV) charge correction scheme has been used for some defects, "
                "while the Kumagai (eFNV) scheme has been used for others. For _isotropic_ materials, "
                "this should be fine, and the results should be the same regardless (assuming a "
                "relatively well-converged supercell size), while for _anisotropic_ materials this could "
                "lead to some quantitative inaccuracies. You can use the `formation_energy_table(dpd)` "
                "function to print out the calculated charge corrections for all defects, "
                "and/or visualise the charge corrections using "
                "`defect_entry.get_freysoldt_correction`/`get_kumagai_correction` with `plot=True` to "
                "check."
            )  # either way have the error analysis for the charge corrections so in theory should be grand
        # note that we also check if multiple charge corrections have been applied to the same defect
        # within the charge correction functions (with self._check_if_multiple_finite_size_corrections())

        if json_filename is not False:  # save to json unless json_filename is False:
            if json_filename is None:
                formula = list(self.defect_dict.values())[
                    0
                ].defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
                json_filename = f"{formula}_defect_dict.json"

            dumpfn(self.defect_dict, os.path.join(self.output_path, json_filename))

        # TODO: Test attribute setting
        # TODO: Warning/error handling for failed parsing defect folders?

    def _multiprocess_parse_defect(self, defect_folder):
        """
        Process defect and catch warnings along the way, so we can print which
        warnings came from which defect.
        """
        str_io = io.StringIO()  # Redirect stderr to capture warnings
        with contextlib.redirect_stderr(str_io):  # capture warnings
            parsed_defect_entry = self._parse_defect(defect_folder)
        warnings_string = str_io.getvalue()

        return parsed_defect_entry, warnings_string

    def _parse_defect(self, defect_folder):
        try:
            dp = DefectParser.from_paths(
                defect_path=os.path.join(self.output_path, defect_folder, self.subfolder),
                bulk_path=self.bulk_path,
                dielectric=self.dielectric,
                skip_corrections=self.skip_corrections,
                error_tolerance=self.error_tolerance,
                bulk_bandgap_path=self.bulk_bandgap_path,
                **self.bulk_corrections_data,
            )

            if dp.skip_corrections and dp.defect_entry.charge_state != 0 and self.dielectric is None:
                self.skip_corrections = dp.skip_corrections  # set skip_corrections to True if
                # dielectric is None and there are charged defects present (shows dielectric warning once)

            if (
                dp.defect_entry.calculation_metadata.get("bulk_locpot_dict") is not None
                and self.bulk_corrections_data.get("bulk_locpot_dict") is None
            ):
                self.bulk_corrections_data["bulk_locpot_dict"] = dp.defect_entry.calculation_metadata[
                    "bulk_locpot_dict"
                ]

            if (
                dp.defect_entry.calculation_metadata.get("bulk_site_potentials") is not None
                and self.bulk_corrections_data.get("bulk_site_potentials") is None
            ):
                self.bulk_corrections_data["bulk_site_potentials"] = (
                    dp.defect_entry.calculation_metadata
                )["bulk_site_potentials"]

        except Exception as exc:
            warnings.warn(
                f"Parsing failed for "
                f"{defect_folder if self.subfolder == '.' else defect_folder + '/' + self.subfolder}, "
                f"got error: {exc}"
            )
            return None

        return dp.defect_entry


class DefectParser:
    def __init__(
        self,
        defect_entry: DefectEntry,
        defect_vr: Optional[Vasprun] = None,
        bulk_vr: Optional[Vasprun] = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        **kwargs,
    ):
        """
        Create a DefectParser object, which has methods for parsing the results
        of defect supercell calculations.

        Direct initiation with DefectParser() is typically not recommended. Rather
        DefectParser.from_paths() or defect_entry_from_paths() are preferred as
        shown in the doped parsing tutorials.

        Args:
            defect_entry (DefectEntry):
                doped DefectEntry
            defect_vr (Vasprun):
                pymatgen Vasprun object for the defect supercell calculation
            bulk_vr (Vasprun):
                pymatgen Vasprun object for the reference bulk supercell calculation
            skip_corrections (bool):
                Whether to skip calculation and application of finite-size charge
                corrections to the defect energy (not recommended in most cases).
                Default = False.
            error_tolerance (float):
                If the estimated error in the defect charge correction is greater
                than this value (in eV), then a warning is raised. (default: 0.05 eV)
            **kwargs:
                Keyword arguments to pass to `DefectParser()` methods
                (`load_FNV_data()`, `load_eFNV_data()`, `load_bulk_gap_data()`)
                such as `bulk_locpot_dict`, `bulk_site_potentials` etc. Mainly
                used by DefectsParser to expedite parsing by avoiding reloading
                bulk data for each defect.
        """
        self.defect_entry: DefectEntry = defect_entry
        self.defect_vr = defect_vr
        self.bulk_vr = bulk_vr
        self.skip_corrections = skip_corrections
        self.error_tolerance = error_tolerance
        self.kwargs = kwargs or {}

    @classmethod
    def from_paths(
        cls,
        defect_path,
        bulk_path,
        dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
        charge_state: Optional[int] = None,
        initial_defect_structure_path: Optional[str] = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        bulk_bandgap_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Parse the defect calculation outputs in `defect_path` and return the
        `DefectParser` object. By default, the `DefectParser.defect_entry.name`
        attribute (later used to label defects in plots) is set to the
        defect_path folder name (if it is a recognised defect name), else it is
        set to the default doped name for that defect.

        Note that the bulk and defect supercells should have the same definitions/basis
        sets (for site-matching and finite-size charge corrections to work appropriately).

        Args:
            defect_path (str):
                Path to defect supercell folder (containing at least vasprun.xml(.gz)).
            bulk_path (str):
                Path to bulk supercell folder (containing at least vasprun.xml(.gz)).
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Ionic + static contributions to the dielectric constant. If not provided,
                charge corrections cannot be computed and so `skip_corrections` will be
                set to true.
            charge_state (int):
                Charge state of defect. If not provided, will be automatically determined
                from the defect calculation outputs (requires `POTCAR`s to be set up
                with `pymatgen`), or if that fails, using the defect folder name (must
                end in "_+X" or "_-X" where +/-X is the defect charge state).
            initial_defect_structure_path (str):
                Path to the initial/unrelaxed defect structure. Only recommended for use
                if structure matching with the relaxed defect structure(s) fails (rare).
                Default is None.
            skip_corrections (bool):
                Whether to skip the calculation and application of finite-size charge
                corrections to the defect energy (not recommended in most cases).
                Default = False.
            error_tolerance (float):
                If the estimated error in the defect charge correction is greater
                than this value (in eV), then a warning is raised. (default: 0.05 eV)
            bulk_bandgap_path (str):
                Path to bulk OUTCAR file for determining the band gap. If the VBM/CBM
                occur at reciprocal space points not included in the bulk supercell
                calculation, you should use this tag to point to a bulk bandstructure
                calculation instead. Alternatively, you can edit/add the "gap" and "vbm"
                entries in DefectParser.defect_entry.calculation_metadata to match the
                correct (eigen)values.
                If None, will calculate "gap"/"vbm" using the outputs at:
                DefectParser.defect_entry.calculation_metadata["bulk_path"]
            **kwargs:
                Keyword arguments to pass to `DefectParser()` methods
                (`load_FNV_data()`, `load_eFNV_data()`, `load_bulk_gap_data()`)
                such as `bulk_locpot_dict`, `bulk_site_potentials` etc. Mainly
                used by DefectsParser to expedite parsing by avoiding reloading
                bulk data for each defect.

        Return:
            `DefectParser` object.
        """
        _ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings

        calculation_metadata = {
            "bulk_path": bulk_path,
            "defect_path": defect_path,
        }

        # add bulk simple properties
        bulk_vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", bulk_path)
        if multiple:
            _multiple_files_warning(
                "vasprun.xml",
                bulk_path,
                bulk_vr_path,
                "parse the calculation energy and metadata.",
                dir_type="bulk",
            )
        bulk_vr = get_vasprun(bulk_vr_path)
        bulk_supercell = bulk_vr.final_structure.copy()

        # add defect simple properties
        (
            defect_vr_path,
            multiple,
        ) = _get_output_files_and_check_if_multiple("vasprun.xml", defect_path)
        if multiple:
            _multiple_files_warning(
                "vasprun.xml",
                defect_path,
                defect_vr_path,
                "parse the calculation energy and metadata.",
                dir_type="defect",
            )
        defect_vr = get_vasprun(defect_vr_path)

        possible_defect_name = os.path.basename(defect_path)  # set equal to folder name
        if "vasp" in possible_defect_name:  # get parent directory name:
            possible_defect_name = os.path.basename(os.path.dirname(defect_path))

        try:
            parsed_charge_state: int = _determine_defect_charge_from_vasprun(defect_vr, charge_state)
        except RuntimeError as orig_exc:  # auto charge guessing failed and charge_state not provided,
            # try determine from folder name - must have "-" or "+" at end of name for this
            try:
                charge_state_suffix = possible_defect_name.rsplit("_", 1)[-1]
                if charge_state_suffix[0] not in ["-", "+"]:
                    raise ValueError(
                        f"Could not guess charge state from folder name: {possible_defect_name}"
                    )

                parsed_charge_state = int(charge_state_suffix)
                if abs(parsed_charge_state) >= 7:
                    raise ValueError(
                        f"Guessed charge state from folder name was {parsed_charge_state:+} which is "
                        f"almost certainly unphysical"
                    )
            except Exception as next_exc:
                raise orig_exc from next_exc

        if dielectric is None and not skip_corrections and parsed_charge_state != 0:
            warnings.warn(
                "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
                "corrections, but none was provided, so charge corrections will be skipped "
                "(`skip_corrections = True`). Formation energies and transition levels of charged "
                "defects will likely be very inaccurate without charge corrections!"
            )
            skip_corrections = True

        if dielectric is not None:
            dielectric = _convert_dielectric_to_tensor(dielectric)
            calculation_metadata["dielectric"] = dielectric

        # Add defect structure to calculation_metadata, so it can be pulled later on (e.g. for eFNV)
        defect_structure = defect_vr.final_structure.copy()
        calculation_metadata["defect_structure"] = defect_structure

        # check if the bulk and defect supercells are the same size:
        if not np.isclose(defect_structure.volume, bulk_supercell.volume, rtol=1e-2):
            warnings.warn(
                f"The defect and bulk supercells are not the same size, having volumes of "
                f"{defect_structure.volume:.1f} and {bulk_supercell.volume:.1f} Å^3 respectively. This "
                f"may cause errors in parsing and/or output energies. In most cases (unless looking at "
                f"extremely high doping concentrations) the same fixed supercell (ISIF = 2) should be "
                f"used for both the defect and bulk calculations! (i.e. assuming the dilute limit)"
            )

        # identify defect site, structural information, and create defect object:
        # Can specify initial defect structure (to help find the defect site if we have a very distorted
        # final structure), but regardless try using the final structure (from defect OUTCAR) first:
        try:
            (
                defect,
                defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
                defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
                # w/interstitials
                defect_site_index,
                bulk_site_index,
                guessed_initial_defect_structure,
                unrelaxed_defect_structure,
            ) = defect_from_structures(bulk_supercell, defect_structure.copy(), return_all_info=True)

        except RuntimeError:
            if initial_defect_structure_path:
                defect_structure_for_ID = Poscar.from_file(initial_defect_structure_path).structure.copy()
                (
                    defect,
                    defect_site_in_initial_struct,
                    defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
                    # w/interstitials
                    defect_site_index,  # in this initial_defect_structure
                    bulk_site_index,
                    guessed_initial_defect_structure,
                    unrelaxed_defect_structure,
                ) = defect_from_structures(bulk_supercell, defect_structure_for_ID, return_all_info=True)

                # then try get defect_site in final structure:
                # need to check that this is the correct defect site, and hasn't been reordered/changed
                # compared to the initial_defect_structure used here, check same element and distance
                # reasonable:
                defect_site = defect_site_in_initial_struct

                if defect.defect_type != core.DefectType.Vacancy:
                    final_defect_site = defect_structure[defect_site_index]
                    if (
                        defect_site_in_initial_struct.species.elements[0].symbol
                        == final_defect_site.species.elements[0].symbol
                    ) and final_defect_site.distance(defect_site_in_initial_struct) < 2:
                        defect_site = final_defect_site

                        if defect.defect_type == core.DefectType.Interstitial:
                            pass

            else:
                raise

        calculation_metadata["guessed_initial_defect_structure"] = guessed_initial_defect_structure
        calculation_metadata["unrelaxed_defect_structure"] = unrelaxed_defect_structure

        defect_entry = DefectEntry(
            # pmg attributes:
            defect=defect,  # this corresponds to _unrelaxed_ defect
            charge_state=parsed_charge_state,
            sc_entry=defect_vr.get_computed_entry(),
            sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
            bulk_entry=bulk_vr.get_computed_entry(),
            # doped attributes:
            defect_supercell_site=defect_site,  # _relaxed_ defect site
            defect_supercell=defect_vr.final_structure,
            bulk_supercell=bulk_vr.final_structure,
            calculation_metadata=calculation_metadata,
        )

        check_and_set_defect_entry_name(defect_entry, possible_defect_name)

        dp = cls(
            defect_entry,
            defect_vr=defect_vr,
            bulk_vr=bulk_vr,
            skip_corrections=skip_corrections,
            error_tolerance=error_tolerance,
            **kwargs,
        )

        dp.load_and_check_calculation_metadata()  # Load standard defect metadata
        dp.load_bulk_gap_data(bulk_bandgap_path=bulk_bandgap_path)  # Load band gap data

        if not skip_corrections and defect_entry.charge_state != 0:
            # no finite-size charge corrections by default for neutral defects
            skip_corrections = dp._check_and_load_appropriate_charge_correction()

        if not skip_corrections and defect_entry.charge_state != 0:
            dp.apply_corrections()

            # check that charge corrections are not negative
            summed_corrections = sum(
                val
                for key, val in dp.defect_entry.corrections.items()
                if any(i in key.lower() for i in ["freysoldt", "kumagai", "fnv", "charge"])
            )
            if summed_corrections < -0.08:
                # usually unphysical for _isotropic_ dielectrics (suggests over-delocalised charge,
                # affecting the potential alignment)
                # how anisotropic is the dielectric?
                how_aniso = np.diag(
                    (dielectric - np.mean(np.diag(dielectric))) / np.mean(np.diag(dielectric))
                )
                if np.allclose(how_aniso, 0, atol=0.05):
                    warnings.warn(
                        f"The calculated finite-size charge corrections for defect at {defect_path} and "
                        f"bulk at {bulk_path} sum to a _negative_ value of {summed_corrections:.3f}. For "
                        f"relatively isotropic dielectrics (as is the case here) this is usually "
                        f"unphyical, and can indicate 'false charge state' behaviour (with the supercell "
                        f"charge occupying the band edge states and not localised at the defect), "
                        f"affecting the potential alignment, or some error/mismatch in the defect and "
                        f"bulk calculations. If this defect species is not stable in the formation "
                        f"energy diagram then this warning can usually be ignored, but if it is, "
                        f"you should double-check your calculations and parsed results!"
                    )

        return dp

    def _check_and_load_appropriate_charge_correction(self):
        skip_corrections = False
        dielectric = self.defect_entry.calculation_metadata["dielectric"]
        bulk_path = self.defect_entry.calculation_metadata["bulk_path"]
        defect_path = self.defect_entry.calculation_metadata["defect_path"]

        # determine charge correction to use, based on what output files are available (`LOCPOT`s or
        # `OUTCAR`s), and whether the supplied dielectric is isotropic or not
        def _check_folder_for_file_match(folder, filename):
            return any(
                filename.lower() in folder_filename.lower() for folder_filename in os.listdir(folder)
            )

        def _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(
            aniso_dielectric,
        ):
            return 3 / sum(1 / diagonal_elt for diagonal_elt in np.diag(aniso_dielectric))

        # check if dielectric (3x3 matrix) has diagonal elements that differ by more than 20%
        isotropic_dielectric = all(np.isclose(i, dielectric[0, 0], rtol=0.2) for i in np.diag(dielectric))

        # regardless, try parsing OUTCAR files first (quickest, more robust for cases where defect
        # charge is localised somewhat off the (auto-determined) defect site (e.g. split-interstitials
        # etc) and also works regardless of isotropic/anisotropic)
        if _check_folder_for_file_match(defect_path, "OUTCAR") and _check_folder_for_file_match(
            bulk_path, "OUTCAR"
        ):
            try:
                self.load_eFNV_data()
            except Exception as kumagai_exc:
                if _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
                    bulk_path, "LOCPOT"
                ):
                    try:
                        if not isotropic_dielectric:
                            # convert anisotropic dielectric to harmonic mean of the diagonal:
                            # (this is a better approximation than the pymatgen default of the
                            # standard arithmetic mean of the diagonal)
                            self.defect_entry.calculation_metadata[
                                "dielectric"
                            ] = _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                        self.load_FNV_data()
                        if not isotropic_dielectric:
                            warnings.warn(
                                _aniso_dielectric_but_outcar_problem_warning
                                + f"in the defect (at {defect_path}) & bulk (at {bulk_path}) folders were "
                                f"unable to be parsed, giving the following error message:"
                                f"\n{kumagai_exc}\n" + _aniso_dielectric_but_using_locpot_warning
                            )
                    except Exception as freysoldt_exc:
                        warnings.warn(
                            f"Got this error message when attempting to parse defect & bulk `OUTCAR` "
                            f"files to compute the Kumagai (eFNV) charge correction:"
                            f"\n{kumagai_exc}\n"
                            f"Then got this error message when attempting to parse defect & bulk "
                            f"`LOCPOT` files to compute the Freysoldt (FNV) charge correction:"
                            f"\n{freysoldt_exc}\n"
                            f"-> Charge corrections will not be applied for this defect."
                        )
                        if not isotropic_dielectric:
                            # reset dielectric to original anisotropic value if FNV failed as well:
                            self.defect_entry.calculation_metadata["dielectric"] = dielectric
                        skip_corrections = True

                else:
                    warnings.warn(
                        f"`OUTCAR` files (needed to compute the Kumagai eFNV charge correction for "
                        f"_anisotropic_ and isotropic systems) in the defect (at {defect_path}) & bulk "
                        f"(at {bulk_path}) folders were unable to be parsed, giving the following error "
                        f"message:"
                        f"\n{kumagai_exc}\n"
                        f"-> Charge corrections will not be applied for this defect."
                    )
                    skip_corrections = True

        elif _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
            bulk_path, "LOCPOT"
        ):
            try:
                if not isotropic_dielectric:
                    # convert anisotropic dielectric to harmonic mean of the diagonal:
                    # (this is a better approximation than the pymatgen default of the
                    # standard arithmetic mean of the diagonal)
                    self.defect_entry.calculation_metadata[
                        "dielectric"
                    ] = _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                self.load_FNV_data()
                if not isotropic_dielectric:
                    warnings.warn(
                        _aniso_dielectric_but_outcar_problem_warning
                        + f"were not found in the defect (at {defect_path}) & bulk (at {bulk_path}) "
                        f"folders.\n" + _aniso_dielectric_but_using_locpot_warning
                    )
            except Exception as freysoldt_exc:
                warnings.warn(
                    f"Got this error message when attempting to parse defect & bulk `LOCPOT` files to "
                    f"compute the Freysoldt (FNV) charge correction:"
                    f"\n{freysoldt_exc}\n"
                    f"-> Charge corrections will not be applied for this defect."
                )
                if not isotropic_dielectric:
                    # reset dielectric to original anisotropic value if FNV failed as well:
                    self.defect_entry.calculation_metadata["dielectric"] = dielectric
                skip_corrections = True

        else:
            if int(self.defect_entry.charge_state) != 0:
                warnings.warn(
                    f"`LOCPOT` or `OUTCAR` files are not present in both the defect (at {defect_path}) "
                    f"and bulk (at {bulk_path}) folders. These are needed to perform the finite-size "
                    f"charge corrections. Charge corrections will not be applied for this defect."
                )
                skip_corrections = True

        return skip_corrections

    def load_FNV_data(self, bulk_locpot_dict=None):
        """
        Load metadata required for performing Freysoldt correction (i.e. LOCPOT
        planar-averaged potential dictionary).

        Requires "bulk_path" and "defect_path" to be present in
        DefectEntry.calculation_metadata, and VASP LOCPOT files to be
        present in these directories. Can read compressed "LOCPOT.gz"
        files. The bulk_locpot_dict can be supplied if already parsed,
        for expedited parsing of multiple defects.

        Saves the `bulk_locpot_dict` and `defect_locpot_dict` dictionaries
        (containing the planar-averaged electrostatic potentials along each
        axis direction) to the DefectEntry.calculation_metadata dict, for
        use with DefectEntry.get_freysoldt_correction().

        Args:
            bulk_locpot_dict (dict): Planar-averaged potential dictionary
                for bulk supercell, if already parsed. If None (default),
                will load from LOCPOT(.gz) file in
                defect_entry.calculation_metadata["bulk_path"]

        Returns:
            bulk_locpot_dict for reuse in parsing other defect entries
        """
        if not self.defect_entry.charge_state:
            # no charge correction if charge is zero
            return None

        bulk_locpot_dict = bulk_locpot_dict or self.kwargs.get("bulk_locpot_dict", None)
        if bulk_locpot_dict is None:
            bulk_locpot_path, multiple = _get_output_files_and_check_if_multiple(
                "LOCPOT", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                _multiple_files_warning(
                    "LOCPOT",
                    self.defect_entry.calculation_metadata["bulk_path"],
                    bulk_locpot_path,
                    "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
                    dir_type="bulk",
                )
            bulk_locpot = get_locpot(bulk_locpot_path)
            bulk_locpot_dict = {str(k): bulk_locpot.get_average_along_axis(k) for k in [0, 1, 2]}

        defect_locpot_path, multiple = _get_output_files_and_check_if_multiple(
            "LOCPOT", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            _multiple_files_warning(
                "LOCPOT",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_locpot_path,
                "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
                dir_type="defect",
            )
        defect_locpot = get_locpot(defect_locpot_path)
        defect_locpot_dict = {str(k): defect_locpot.get_average_along_axis(k) for k in [0, 1, 2]}

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_locpot_dict": bulk_locpot_dict,
                "defect_locpot_dict": defect_locpot_dict,
            }
        )

        return bulk_locpot_dict

    def load_eFNV_data(self, bulk_site_potentials=None):
        """
        Load metadata required for performing Kumagai correction (i.e. atomic
        site potentials from the OUTCAR files).

        Requires "bulk_path" and "defect_path" to be present in
        DefectEntry.calculation_metadata, and VASP OUTCAR files to be
        present in these directories. Can read compressed "OUTCAR.gz"
        files. The bulk_site_potentials can be supplied if already
        parsed, for expedited parsing of multiple defects.

        Saves the `bulk_site_potentials` and `defect_site_potentials`
        lists (containing the atomic site electrostatic potentials, from
        -1*np.array(Outcar.electrostatic_potential)) to
        DefectEntry.calculation_metadata, for use with
        DefectEntry.get_kumagai_correction().

        Args:
            bulk_site_potentials (dict): Atomic site potentials for the
                bulk supercell, if already parsed. If None (default), will
                load from OUTCAR(.gz) file in
                defect_entry.calculation_metadata["bulk_path"]

        Returns:
            bulk_site_potentials for reuse in parsing other defect entries
        """
        from doped.corrections import _raise_incomplete_outcar_error  # avoid circular import

        if not self.defect_entry.charge_state:
            # don't need to load outcars if charge is zero
            return None

        bulk_site_potentials = bulk_site_potentials or self.kwargs.get("bulk_site_potentials", None)

        if bulk_site_potentials is None:
            bulk_outcar_path, multiple = _get_output_files_and_check_if_multiple(
                "OUTCAR", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                _multiple_files_warning(
                    "OUTCAR",
                    self.defect_entry.calculation_metadata["bulk_path"],
                    bulk_outcar_path,
                    "parse core levels and compute the Kumagai (eFNV) image charge correction.",
                    dir_type="bulk",
                )
            bulk_outcar = get_outcar(bulk_outcar_path)

            if bulk_outcar.electrostatic_potential is None:
                _raise_incomplete_outcar_error(bulk_outcar_path, dir_type="bulk")

            bulk_site_potentials = -1 * np.array(bulk_outcar.electrostatic_potential)

        defect_outcar_path, multiple = _get_output_files_and_check_if_multiple(
            "OUTCAR", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            _multiple_files_warning(
                "OUTCAR",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_outcar_path,
                "parse core levels and compute the Kumagai (eFNV) image charge correction.",
                dir_type="defect",
            )
        defect_outcar = get_outcar(defect_outcar_path)

        if defect_outcar.electrostatic_potential is None:
            _raise_incomplete_outcar_error(defect_outcar_path, dir_type="defect")

        defect_site_potentials = -1 * np.array(defect_outcar.electrostatic_potential)

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_site_potentials": bulk_site_potentials,
                "defect_site_potentials": defect_site_potentials,
            }
        )

        return bulk_site_potentials

    def load_and_check_calculation_metadata(self):
        """
        Pull metadata about the defect supercell calculations from the outputs,
        and check if the defect and bulk supercell calculations settings are
        compatible.
        """
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                _multiple_files_warning(
                    "vasprun.xml",
                    self.defect_entry.calculation_metadata["bulk_path"],
                    bulk_vr_path,
                    "parse the calculation energy and metadata.",
                    dir_type="bulk",
                )
            self.bulk_vr = get_vasprun(bulk_vr_path)

        if not self.defect_vr:
            defect_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["defect_path"]
            )
            if multiple:
                _multiple_files_warning(
                    "vasprun.xml",
                    self.defect_entry.calculation_metadata["defect_path"],
                    defect_vr_path,
                    "parse the calculation energy and metadata.",
                    dir_type="defect",
                )
            self.defect_vr = get_vasprun(defect_vr_path)

        run_metadata = {
            # incars need to be as dict without module keys otherwise not JSONable:
            "defect_incar": {k: v for k, v in self.defect_vr.incar.as_dict().items() if "@" not in k},
            "bulk_incar": {k: v for k, v in self.bulk_vr.incar.as_dict().items() if "@" not in k},
            "defect_kpoints": self.defect_vr.kpoints,
            "bulk_kpoints": self.bulk_vr.kpoints,
            "defect_potcar_symbols": self.defect_vr.potcar_spec,
            "bulk_potcar_symbols": self.bulk_vr.potcar_spec,
        }

        _compare_incar_tags(run_metadata["bulk_incar"], run_metadata["defect_incar"])
        _compare_potcar_symbols(run_metadata["bulk_potcar_symbols"], run_metadata["defect_potcar_symbols"])
        _compare_kpoints(run_metadata["bulk_kpoints"], run_metadata["defect_kpoints"])

        self.defect_entry.calculation_metadata.update({"run_metadata": run_metadata.copy()})

        # standard defect run metadata
        self.defect_entry.calculation_metadata.update(
            {
                "final_defect_structure": self.defect_vr.final_structure,
            }
        )

        # grab defect energy and eigenvalue information for band filling and localization analysis
        eigenvalues = {
            spincls.value: eigdict.copy() for spincls, eigdict in self.defect_vr.eigenvalues.items()
        }
        kpoint_weights = self.defect_vr.actual_kpoints_weights[:]
        self.defect_entry.calculation_metadata.update(
            {"eigenvalues": eigenvalues, "kpoint_weights": kpoint_weights}
        )

    def load_bulk_gap_data(self, bulk_bandgap_path=None, use_MP=False, mpid=None, api_key=None):
        """
        Get bulk band gap data from bulk OUTCAR file, or OUTCAR located at
        `actual_bulk_path`.

        Alternatively, one can specify query the Materials Project (MP) database
        for the bulk gap data, using `use_MP = True`, in which case the MP entry
        with the lowest number ID and composition matching the bulk will be used,
        or the MP ID (mpid) of the bulk material to use can be specified. This is
        not recommended as it will correspond to a severely-underestimated GGA DFT
        bandgap!

        Args:
            bulk_bandgap_path (str):
                Path to bulk OUTCAR file for determining the band gap. If the VBM/CBM
                occur at reciprocal space points not included in the bulk supercell
                calculation, you should use this tag to point to a bulk bandstructure
                calculation instead. If None, will use
                self.defect_entry.calculation_metadata["bulk_path"].
            use_MP (bool):
                If True, will query the Materials Project database for the bulk gap data.
            mpid (str):
                If provided, will query the Materials Project database for the bulk gap
                data, using this Materials Project ID.
            api_key (str): Materials API key to access database.
        """
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in bulk directory: "
                    f"{self.defect_entry.calculation_metadata['bulk_path']}. Using {bulk_vr_path} to "
                    f"parse the calculation energy and metadata."
                )
            self.bulk_vr = get_vasprun(bulk_vr_path)

        bulk_sc_structure = self.bulk_vr.initial_structure

        vbm, cbm, bandgap = None, None, None
        gap_calculation_metadata = {}

        use_MP = use_MP or self.kwargs.get("use_MP", False)
        mpid = mpid or self.kwargs.get("mpid", None)
        api_key = api_key or self.kwargs.get("api_key", None)

        if use_MP and mpid is None:
            try:
                with MPRester(api_key=api_key) as mp:
                    tmp_mplist = mp.get_entries_in_chemsys(list(bulk_sc_structure.symbol_set))
                mplist = [
                    mp_ent.entry_id
                    for mp_ent in tmp_mplist
                    if mp_ent.composition.reduced_composition
                    == bulk_sc_structure.composition.reduced_composition
                ]
            except Exception as exc:
                raise ValueError(
                    f"Error with querying MPRester for"
                    f" {bulk_sc_structure.composition.reduced_formula}:"
                ) from exc

            mpid_fit_list = []
            for trial_mpid in mplist:
                with MPRester(api_key=api_key) as mp:
                    mpstruct = mp.get_structure_by_material_id(trial_mpid)
                if StructureMatcher(
                    primitive_cell=True,
                    scale=False,
                    attempt_supercell=True,
                    allow_subset=False,
                ).fit(bulk_sc_structure, mpstruct):
                    mpid_fit_list.append(trial_mpid)

            if len(mpid_fit_list) == 1:
                mpid = mpid_fit_list[0]
                print(f"Single mp-id found for bulk structure:{mpid}.")
            elif len(mpid_fit_list) > 1:
                num_mpid_list = [int(mp.split("" - "")[1]) for mp in mpid_fit_list]
                num_mpid_list.sort()
                mpid = f"mp-{num_mpid_list[0]!s}"
                print(
                    f"Multiple mp-ids found for bulk structure:{mpid_fit_list}. Will use lowest "
                    f"number mpid for bulk band structure = {mpid}."
                )
            else:
                print(
                    "Could not find bulk structure in MP database after tying the following "
                    f"list:\n{mplist}"
                )
                mpid = None

        if mpid is not None:
            print(f"Using user-provided mp-id for bulk structure: {mpid}.")
            with MPRester(api_key=api_key) as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if bs:
                cbm = bs.get_cbm()["energy"]
                vbm = bs.get_vbm()["energy"]
                bandgap = bs.get_band_gap()["energy"]
                gap_calculation_metadata["MP_gga_BScalc_data"] = bs.get_band_gap().copy()

        if vbm is None or bandgap is None or cbm is None or not bulk_bandgap_path:
            if mpid and bandgap is None:
                print(
                    f"WARNING: Mpid {mpid} was provided, but no bandstructure entry currently "
                    "exists for it. \nReverting to use of bulk supercell calculation for band "
                    "edge extrema."
                )

            gap_calculation_metadata["MP_gga_BScalc_data"] = None  # to signal no MP BS is used
            bandgap, cbm, vbm, _ = self.bulk_vr.eigenvalue_band_properties

        if bulk_bandgap_path:
            print(f"Using actual bulk path: {bulk_bandgap_path}")
            actual_bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", bulk_bandgap_path
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in specified directory: "
                    f"{bulk_bandgap_path}. Using {actual_bulk_vr_path} to  parse the calculation "
                    f"energy and metadata."
                )
            actual_bulk_vr = get_vasprun(actual_bulk_vr_path)
            bandgap, cbm, vbm, _ = actual_bulk_vr.eigenvalue_band_properties

        gap_calculation_metadata = {
            "mpid": mpid,
            "cbm": cbm,
            "vbm": vbm,
            "gap": bandgap,
        }
        self.defect_entry.calculation_metadata.update(gap_calculation_metadata)

    def apply_corrections(self):
        """
        Get defect corrections and warn if likely to be inappropriate.
        """
        if not self.defect_entry.charge_state:  # no charge correction if charge is zero
            return

        # try run Kumagai (eFNV) correction if required info available:
        if (
            self.defect_entry.calculation_metadata.get("bulk_site_potentials", None) is not None
            and self.defect_entry.calculation_metadata.get("defect_site_potentials", None) is not None
        ):
            self.defect_entry.get_kumagai_correction(verbose=False, error_tolerance=self.error_tolerance)

        elif self.defect_entry.calculation_metadata.get(
            "bulk_locpot_dict"
        ) and self.defect_entry.calculation_metadata.get("defect_locpot_dict"):
            self.defect_entry.get_freysoldt_correction(verbose=False, error_tolerance=self.error_tolerance)

        else:
            raise ValueError(
                "No charge correction performed! Missing required metadata in "
                "defect_entry.calculation_metadata ('bulk/defect_site_potentials' for Kumagai ("
                "eFNV) correction, or 'bulk/defect_locpot_dict' for Freysoldt (FNV) correction) - these "
                "are loaded with either the load_eFNV_data() or load_FNV_data() methods for "
                "DefectParser."
            )

        if (
            self.defect_entry.charge_state != 0
            and (not self.defect_entry.corrections or sum(self.defect_entry.corrections.values())) == 0
        ):
            warnings.warn(
                f"No charge correction computed for {self.defect_entry.name} with charge"
                f" {self.defect_entry.charge_state:+}, indicating problems with the required data for "
                f"the charge correction (i.e. dielectric constant, LOCPOT files for Freysoldt "
                f"correction, OUTCAR (with ICORELEVEL = 0) for Kumagai correction etc)."
            )
