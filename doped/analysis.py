# coding: utf-8

"""
Code to analyse VASP defect calculations.
These functions are built from a combination of useful modules from pymatgen, alongside
substantial modification, in the efforts of making an efficient, user-friendly package for
managing and analysing defect calculations, with publication-quality outputs
"""

import copy
import os
import warnings
from operator import itemgetter

import pandas as pd
import numpy as np
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram
from pymatgen.analysis.defects.utils import TopographyAnalyzer
from pymatgen.util.string import unicodeify
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.defects.core import DefectEntry
from monty.serialization import loadfn, dumpfn
from monty.json import MontyDecoder
from tabulate import tabulate

from doped.pycdt.utils.vasp import DefectRelaxSet
from doped.pycdt.utils import parse_calculations
from doped import _ignore_pmg_warnings

_ANGSTROM = "\u212B"  # unicode symbol for angstrom to print in strings
_ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings


def bold_print(string: str) -> None:
    """Does what it says on the tin. Prints the input string in bold."""
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


# TODO: Should add check the bulk and defect KPOINTS/INCAR/POTCAR/POSCAR (size) settings
#  are compatible, and throw warning if not.
# TODO: Add `check_defects_compatibility()` function that checks the bulk and defect
#  KPOINTS/INCAR/POTCAR/POSCAR (size) settings for all defects in the supplied defect_dict
#  are compatible, if not throw warnings and say what the differences are. Should recommend
#  using this in the example notebook if a user has parsed the defects individually (rather
#  than with the single looping function described below):
# TODO: Add a function that loops over all the defects in a directory (with `defect_dir = .`,
#  and `subfolder = vasp_ncl` options) and parses them all (use `SnB` defect-name-matching
#  function?), returning a dictionary of defect entries, with the defect name as the key. (i.e.
#  doing the loop in the example notebook). Show both this function and the individual function
#  calls in the example notebook. Benefit of this one is that we can then auto-run
#  `check_defects_compatibility()` at the end of parsing the full defects dict. – When doing
#  this, look at `PostProcess` code, and then delete it once all functionality is moved here.
def defect_entry_from_paths(
    defect_path,
    bulk_path,
    dielectric,
    charge=None,
    initial_defect_structure=None,
    skip_corrections=False,
    bulk_bandgap_path=None,
    **kwargs,
):
    """
    Parse the defect calculation outputs in `defect_path` and return the parsed `DefectEntry`
    object.

    Args:
    defect_path (str): path to defect folder of interest (with vasprun.xml(.gz))
    bulk_path (str): path to bulk folder of interest (with vasprun.xml(.gz))
    dielectric (float or int or 3x1 matrix or 3x3 matrix):
        ionic + static contributions to dielectric constant
    charge (int): charge of defect. If not provided, will be automatically determined
        from the defect calculation outputs (requires POTCARs to be set up with `pymatgen`).
    initial_defect_structure (str):  Path to the unrelaxed defect structure,
        if structure matching with the relaxed defect structure(s) fails.
    skip_corrections (bool): Whether to skip the calculation and application of finite-size
        charge corrections to the defect energy.
    bulk_bandgap_path (str): Path to bulk OUTCAR file for determining the band gap. If the
        VBM/CBM occur at reciprocal space points not included in the bulk supercell calculation,
        you should use this tag to point to a bulk bandstructure calculation instead. If None,
        will use self.defect_entry.parameters["bulk_path"].

    Return:
        Parsed `DefectEntry` object.
    """
    _ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings
    dielectric = _convert_dielectric_to_tensor(dielectric)

    parameters = {
        "bulk_path": bulk_path,
        "defect_path": defect_path,
        "dielectric": dielectric,
    }

    # add bulk simple properties
    bulk_vr_path, multiple = parse_calculations._get_output_files_and_check_if_multiple(
        "vasprun.xml", bulk_path
    )
    if multiple:
        warnings.warn(
            f"Multiple `vasprun.xml` files found in bulk directory: {bulk_path}. Using "
            f"{bulk_vr_path} to parse the calculation energy and metadata."
        )
    bulk_vr = parse_calculations.get_vasprun(bulk_vr_path)
    bulk_energy = bulk_vr.final_energy
    bulk_sc_structure = bulk_vr.initial_structure.copy()

    # add defect simple properties
    (
        defect_vr_path,
        multiple,
    ) = parse_calculations._get_output_files_and_check_if_multiple(
        "vasprun.xml", defect_path
    )
    if multiple:
        warnings.warn(
            f"Multiple `vasprun.xml` files found in defect directory: {defect_path}. Using"
            f" {defect_vr_path} to parse the calculation energy and metadata."
        )
    defect_vr = parse_calculations.get_vasprun(defect_vr_path)
    defect_energy = defect_vr.final_energy

    # get defect charge
    try:
        defect_nelect = defect_vr.incar.get("NELECT", None)
        if defect_nelect is None:
            auto_charge = 0  # neutral defect if NELECT not specified
        else:
            potcar_symbols = [titel.split()[1] for titel in defect_vr.potcar_symbols]
            potcar_settings = {
                symbol.split("_")[0]: symbol for symbol in potcar_symbols
            }
            neutral_defect_relax_set = DefectRelaxSet(
                defect_vr.structures[-1],
                charge=0,
                user_potcar_settings=potcar_settings,
            )
            try:
                auto_charge = -1 * (defect_nelect - neutral_defect_relax_set.nelect)

            except Exception as e:
                raise RuntimeError(
                    f"Defect charge cannot be automatically determined as POTCARs have not "
                    f"been setup with pymatgen (see Step 2 at "
                    f"https://github.com/SMTG-UCL/doped#installation). Please specify defect "
                    f"charge manually using the `charge` argument, or set up POTCARs with "
                    f"pymatgen. Got error: {e}"
                )

            if abs(auto_charge) >= 10:  # crazy charge state predicted
                raise RuntimeError(
                    f"Auto-determined defect charge q={int(auto_charge):+} is unreasonably large. "
                    f"Please specify defect charge manually using the `charge` argument."
                )

        if (
            charge is not None
            and int(charge) != int(auto_charge)
            and abs(auto_charge) < 5
        ):
            warnings.warn(
                f"Auto-determined defect charge q={int(auto_charge):+} does not match "
                f"specified charge q={int(charge):+}. Will continue with specified "
                f"charge, but beware!"
            )
        else:
            charge = auto_charge

    except Exception as e:
        if charge is None:
            raise e

    # Can specify initial defect structure (to help PyCDT find the defect site if
    # multiple relaxations were required, else use from defect relaxation OUTCAR:
    if initial_defect_structure:
        initial_defect_structure = Poscar.from_file(
            initial_defect_structure
        ).structure.copy()
    else:
        initial_defect_structure = defect_vr.initial_structure.copy()

    # Add initial defect structure to parameters, so it can be pulled later on
    # (eg. for Kumagai loader)
    parameters["initial_defect_structure"] = initial_defect_structure

    # identify defect site, structural information, and create defect object
    try:
        def_type, comp_diff = parse_calculations.get_defect_type_and_composition_diff(
            bulk_sc_structure, initial_defect_structure
        )
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_sc_structure)} in bulk vs. {len(initial_defect_structure)} in defect?"
        ) from exc

    bulk_site_idx = None
    defect_site_idx = None
    unrelaxed_defect_structure = None
    # Try automatic defect site detection - this gives us the "unrelaxed" defect structure
    try:
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = parse_calculations.get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
    except RuntimeError as exc:
        # if auto site-matching failed, try use transformation.json
        # The goal is to find the `defect_site_idx` or `defect_site_idx` based on the
        # tranformation.
        transformation_path = os.path.join(defect_path, "transformation.json")
        if not os.path.exists(transformation_path):  # try next folder up
            orig_transformation_path = transformation_path
            transformation_path = os.path.join(
                os.path.dirname(os.path.normpath(defect_path)),
                "transformation.json",
            )
            if os.path.exists(transformation_path):
                print(
                    f"No transformation file found at {orig_transformation_path}, but found "
                    f"one at {transformation_path}. Using this for defect parsing."
                )

        if os.path.exists(transformation_path):
            tf = loadfn(transformation_path)
            site = tf["defect_supercell_site"]
            if def_type == "vacancy":
                poss_deflist = sorted(
                    bulk_sc_structure.get_sites_in_sphere(
                        site.coords, 0.2, include_index=True
                    ),
                    key=lambda x: x[1],
                )
                searched = "bulk_supercell"
                if poss_deflist:
                    bulk_site_idx = poss_deflist[0][2]
            else:
                poss_deflist = sorted(
                    initial_defect_structure.get_sites_in_sphere(
                        site.coords, 2.5, include_index=True
                    ),
                    key=lambda x: x[1],
                )
                searched = "initial_defect_structure"
                if poss_deflist:
                    defect_site_idx = poss_deflist[0][2]
            if not poss_deflist:
                raise ValueError(
                    f"{transformation_path} specified defect site {site}, but could not find "
                    f"it in {searched}. Abandoning parsing."
                ) from exc
            if poss_deflist[0][1] > 1:
                site_matched_defect = poss_deflist[0]  # pymatgen Neighbor object
                offsite_warning = (
                    f"Site-matching has determined {site_matched_defect.species} at "
                    f"{site_matched_defect.coords} as the defect site, located "
                    f"{site_matched_defect.nn_distance:.2f} {_ANGSTROM} from "
                    f"its initial position. This may incur small errors in the charge correction."
                )
                warnings.warn(message=offsite_warning)

        else:
            raise RuntimeError(
                f"Could not identify {def_type} defect site in defect structure. "
                f"Try supplying the initial defect structure to "
                f"SingleDefectParser.from_paths(), or making sure the doped "
                f"transformation.json files are in the defect directory."
            ) from exc

    if def_type == "vacancy":
        defect_site = bulk_sc_structure[bulk_site_idx]
    else:
        if unrelaxed_defect_structure:
            defect_site = unrelaxed_defect_structure[defect_site_idx]
        else:
            defect_site = initial_defect_structure[defect_site_idx]

    if unrelaxed_defect_structure:
        if def_type == "interstitial":
            # get closest Voronoi site in bulk supercell to final interstitial site as this is
            # likely to be the initial interstitial site
            try:
                struc_and_node_dict = loadfn("./bulk_voronoi_nodes.json")
                if not StructureMatcher(
                    stol=0.05,
                    primitive_cell=False,
                    scale=False,
                    attempt_supercell=False,
                    allow_subset=False,
                ).fit(struc_and_node_dict["bulk_supercell"], bulk_sc_structure):
                    warnings.warn(
                        "Previous bulk_voronoi_nodes.json detected, but does not "
                        "match current bulk supercell. Recalculating Voronoi nodes."
                    )
                    raise FileNotFoundError

                voronoi_frac_coords = struc_and_node_dict["Voronoi nodes"]

            except FileNotFoundError:  # first time parsing
                topography = TopographyAnalyzer(
                    bulk_sc_structure,
                    bulk_sc_structure.symbol_set,
                    [],
                    check_volume=False,
                )
                topography.cluster_nodes()
                topography.remove_collisions()
                voronoi_frac_coords = [site.frac_coords for site in topography.vnodes]
                struc_and_node_dict = {
                    "bulk_supercell": bulk_sc_structure,
                    "Voronoi nodes": voronoi_frac_coords,
                }
                dumpfn(
                    struc_and_node_dict, "./bulk_voronoi_nodes.json"
                )  # for efficient
                # parsing of multiple defects at once
                print(
                    "Saving parsed Voronoi sites (for interstitial site-matching) to "
                    "bulk_voronoi_sites.json to speed up future parsing."
                )

            closest_node_frac_coords = min(
                voronoi_frac_coords,
                key=lambda node: defect_site.distance_and_image_from_frac_coords(node)[
                    0
                ],
            )
            int_site = unrelaxed_defect_structure[defect_site_idx]
            unrelaxed_defect_structure.remove_sites([defect_site_idx])
            unrelaxed_defect_structure.insert(
                defect_site_idx,  # Place defect at same position as in DFT calculation
                int_site.species_string,
                closest_node_frac_coords,
                coords_are_cartesian=False,
                validate_proximity=True,
            )
            defect_site = unrelaxed_defect_structure[defect_site_idx]

        # Use the unrelaxed_defect_structure to fix the initial defect structure
        initial_defect_structure = parse_calculations.reorder_unrelaxed_structure(
            unrelaxed_defect_structure, initial_defect_structure
        )
        parameters["initial_defect_structure"] = initial_defect_structure
        parameters["unrelaxed_defect_structure"] = unrelaxed_defect_structure
    else:
        warnings.warn(
            "Cannot determine the unrelaxed `initial_defect_structure`. Please ensure the "
            "`initial_defect_structure` is indeed unrelaxed."
        )

    for_monty_defect = {
        "@module": "pymatgen.analysis.defects.core",
        "@class": def_type.capitalize(),
        "charge": charge,
        "structure": bulk_sc_structure,
        "defect_site": defect_site,
    }
    defect = MontyDecoder().process_decoded(for_monty_defect)

    if unrelaxed_defect_structure:
        # only do StructureMatcher test if unrelaxed structure exists
        test_defect_structure = defect.generate_defect_structure()
        if not StructureMatcher(
            stol=0.25,
            primitive_cell=False,
            scale=False,
            attempt_supercell=False,
            allow_subset=False,
        ).fit(test_defect_structure, unrelaxed_defect_structure):
            # NOTE: this does not insure that cartesian coordinates or indexing are identical
            raise ValueError(
                "Error in defect object matching! Unrelaxed structure (1st below) "
                "does not match pymatgen defect.generate_defect_structure() "
                f"(2nd below):\n{unrelaxed_defect_structure}"
                f"\n{test_defect_structure}"
            )

    defect_entry = DefectEntry(
        defect, defect_energy - bulk_energy, corrections={}, parameters=parameters
    )

    sdp = parse_calculations.SingleDefectParser(
        defect_entry,
        defect_vr=defect_vr,
        bulk_vr=bulk_vr,
        **kwargs,  # in case user wants to specify `DefectCompatibility()`
    )

    sdp.get_stdrd_metadata()  # Load standard defect metadata
    sdp.get_bulk_gap_data(bulk_bandgap_path=bulk_bandgap_path)  # Load band gap data

    if not skip_corrections:
        # determine charge correction to use, based on what output files are available (`LOCPOT`s or
        # `OUTCAR`s), and whether the supplied dielectric is isotropic or not
        def _check_folder_for_file_match(folder, filename):
            return any(
                filename.lower() in folder_filename.lower()
                for folder_filename in os.listdir(folder)
            )

        def _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(
            aniso_dielectric,
        ):
            return 3 / sum(
                1 / diagonal_elt for diagonal_elt in np.diag(aniso_dielectric)
            )

        # check if dielectric (3x3 matrix) has diagonal elements that differ by more than 20%
        isotropic_dielectric = all(
            np.isclose(i, dielectric[0, 0], rtol=0.2) for i in np.diag(dielectric)
        )

        # regardless, try parsing OUTCAR files first (quickest)
        if _check_folder_for_file_match(
            defect_path, "OUTCAR"
        ) and _check_folder_for_file_match(bulk_path, "OUTCAR"):
            try:
                sdp.kumagai_loader()
            except Exception as kumagai_exc:
                if _check_folder_for_file_match(
                    defect_path, "LOCPOT"
                ) and _check_folder_for_file_match(bulk_path, "LOCPOT"):
                    try:
                        if not isotropic_dielectric:
                            # convert anisotropic dielectric to harmonic mean of the diagonal:
                            # (this is a better approximation than the pymatgen default of the
                            # standard arithmetic mean of the diagonal)
                            sdp.defect_entry.parameters[
                                "dielectric"
                            ] = _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(
                                dielectric
                            )
                        sdp.freysoldt_loader()
                        if not isotropic_dielectric:
                            warnings.warn(
                                f"An anisotropic dielectric constant was supplied, but `OUTCAR` "
                                f"files (needed to compute the _anisotropic_ Kumagai eFNV charge "
                                f"correction) in the defect (at {defect_path}) & bulk (at "
                                f"{bulk_path}) folders were unable to be parsed, giving the "
                                f"following error message:\n{kumagai_exc}\n"
                                f"`LOCPOT` files were found in both defect & bulk folders, "
                                f"and so the Freysoldt (FNV) charge correction developed for "
                                f"_isotropic_ materials will be applied here, which corresponds "
                                f"to using the effective isotropic average of the supplied "
                                f"anisotropic dielectric. This could lead to significant errors "
                                f"for very anisotropic systems and/or relatively small supercells!"
                            )
                    except Exception as freysoldt_exc:
                        warnings.warn(
                            f"Got this error message when attempting to parse defect & bulk "
                            f"`OUTCAR` files to compute the Kumagai (eFNV) charge correction:"
                            f"\n{kumagai_exc}\nThen got this error message when attempting to "
                            f"parse defect & bulk `LOCPOT` files to compute the Freysoldt (FNV) "
                            f"charge correction:\n{freysoldt_exc}\n-> Charge corrections will not "
                            f"be applied for this defect."
                        )
                        if not isotropic_dielectric:
                            # reset dielectric to original anisotropic value if FNV failed as well:
                            sdp.defect_entry.parameters["dielectric"] = dielectric
                        skip_corrections = True

                else:
                    warnings.warn(
                        f"An anisotropic dielectric constant was supplied, but `OUTCAR` "
                        f"files (needed to compute the _anisotropic_ Kumagai eFNV charge "
                        f"correction) in the defect (at {defect_path}) & bulk (at "
                        f"{bulk_path}) folders were unable to be parsed, giving the "
                        f"following error message:\n{kumagai_exc}\n-> Charge corrections will not "
                        f"be applied for this defect."
                    )
                    skip_corrections = True

        elif _check_folder_for_file_match(
            defect_path, "LOCPOT"
        ) and _check_folder_for_file_match(bulk_path, "LOCPOT"):
            try:
                if not isotropic_dielectric:
                    # convert anisotropic dielectric to harmonic mean of the diagonal:
                    # (this is a better approximation than the pymatgen default of the
                    # standard arithmetic mean of the diagonal)
                    sdp.defect_entry.parameters[
                        "dielectric"
                    ] = _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(
                        dielectric
                    )
                sdp.freysoldt_loader()
                if not isotropic_dielectric:
                    warnings.warn(
                        f"An anisotropic dielectric constant was supplied, but `OUTCAR` "
                        f"files (needed to compute the _anisotropic_ Kumagai eFNV charge "
                        f"correction) were not found in the defect (at {defect_path}) & bulk "
                        f"(at {bulk_path}) folders.\n"
                        f"`LOCPOT` files were found in both defect & bulk folders, "
                        f"and so the Freysoldt (FNV) charge correction developed for "
                        f"_isotropic_ materials will be applied here, which corresponds "
                        f"to using the effective isotropic average of the supplied "
                        f"anisotropic dielectric. This could lead to significant errors "
                        f"for very anisotropic systems and/or relatively small supercells!"
                    )
            except Exception as freysoldt_exc:
                warnings.warn(
                    f"Got this error message when attempting to parse defect & bulk "
                    f"`LOCPOT` files to compute the Freysoldt (FNV) charge correction:"
                    f"\n{freysoldt_exc}\n-> Charge corrections will not be applied for this "
                    f"defect."
                )
                if not isotropic_dielectric:
                    # reset dielectric to original anisotropic value if FNV failed as well:
                    sdp.defect_entry.parameters["dielectric"] = dielectric
                skip_corrections = True

        else:
            if int(sdp.defect_entry.charge) != 0:
                warnings.warn(
                    f"`LOCPOT` or `OUTCAR` files are not present in both the defect (at "
                    f"{defect_path}) and bulk (at {bulk_path}) folders. These are needed to "
                    f"perform the finite-size charge corrections. Charge corrections will not be "
                    f"applied for this defect."
                )
                skip_corrections = True

        if not skip_corrections:
            # Check compatibility of defect corrections with loaded metadata, and apply
            sdp.run_compatibility()

    return sdp.defect_entry


def dpd_from_defect_dict(parsed_defect_dict: dict) -> DefectPhaseDiagram:
    """Generates a DefectPhaseDiagram object from a dictionary of parsed defect calculations (
    format: {"defect_name": defect_entry}), likely created using SingleDefectParser from
    doped.pycdt.utils.parse_calculations), which can then be used to analyse and plot the defect
    thermodynamics (formation energies, transition levels, concentrations etc)

    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (format: {"defect_name": defect_entry}),
            likely created using SingleDefectParser from doped.pycdt.utils.parse_calculations).
            Must have 'vbm' and 'gap' in defect_entry.parameters for each defect (from
            SingleDefectParser.get_bulk_gap_data())

    Returns:
        pymatgen DefectPhaseDiagram object (DefectPhaseDiagram)
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
    vbm_vals = []
    bandgap_vals = []
    for defect in parsed_defect_dict.values():
        vbm_vals.append(defect.parameters["vbm"])
        bandgap_vals.append(defect.parameters["gap"])
    if len(set(vbm_vals)) > 1:  # Check if all defects give same vbm
        raise ValueError(
            f"VBM values don't match for defects in given defect dictionary, "
            f"the VBM values in the dictionary are: {vbm_vals}. "
            f"Are you sure the correct/same bulk files were used with "
            f"SingleDefectParser and/or get_bulk_gap_data()?"
        )
    if len(set(bandgap_vals)) > 1:  # Check if all defects give same bandgap
        raise ValueError(
            f"Bandgap values don't match for defects in given defect dictionary, "
            f"the bandgap values in the dictionary are: {bandgap_vals}. "
            f"Are you sure the correct/same bulk files were used with "
            f"SingleDefectParser and/or get_bulk_gap_data()?"
        )
    vbm = vbm_vals[0]
    bandgap = bandgap_vals[0]
    dpd = DefectPhaseDiagram(
        list(parsed_defect_dict.values()), vbm, bandgap, filter_compatible=False
    )

    return dpd


def dpd_transition_levels(defect_phase_diagram: DefectPhaseDiagram):
    """Iteratively prints the charge transition levels for the input DefectPhaseDiagram object
    (via the from a defect_phase_diagram.transition_level_map attribute)

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
    chempot_limits: dict = None,
    pd_facets: list = None,
    fermi_level: float = 0,
    hide_cols: list = None,
    show_key: bool = True,
):
    """
    Prints defect formation energy tables for either a single chemical potential limit (i.e. phase
    diagram facet) or each facet in the phase diagram (chempot_limits dict), depending on the
    chempot_limits input supplied. This can either be a dictionary of chosen absolute/DFT chemical
    potentials: {Elt: Energy} (giving a single formation energy table) or a dictionary including
    the key-value pair: {"facets": [{'facet': [chempot_dict]}]}, following the format generated
    by chempot_limits = cpa.read_phase_diagram_and_chempots() (see example notebooks). In the
    latter case, a subset of facet(s) / chemical potential limit(s) can be chosen with the
    pd_facets argument, or if not specified, will print formation energy tables for each facet in
    the phase diagram.
    Returns the results a pandas DataFrame or list of DataFrames.

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
             DefectPhaseDiagram object (likely created from
             analysis.dpd_from_defect_dict)
        chempot_limits (dict):
            This can either be a dictionary of chosen absolute/DFT chemical potentials: {Elt:
            Energy} (giving a single formation energy table) or a dictionary including the
            key-value pair: {"facets": [{'facet': [chempot_dict]}]}, following the format generated
            by chempot_limits = cpa.read_phase_diagram_and_chempots() (see example notebooks). If
            not specified, chemical potentials are not included in the formation energy calculation
            (all set to zero energy).
        pd_facets (list):
            A list facet(s) / chemical potential limit(s) for which to print the defect formation
            energy tables. If not specified, will print formation energy tables for each facet in
            the phase diagram. (default: None)
        fermi_level (float):
            Fermi level to use for computing the defect formation energies. (default: 0 (i.e.
            at the VBM))
        hide_cols: (list):
            List of columns to hide from the output. (default: None)
        show_key (bool):
            Whether or not to print the table key at the bottom of the output. (default: True)

    Returns:
        pandas DataFrame or list of DataFrames
    """
    if chempot_limits is None:
        chempot_limits = {}

    if "facets" in chempot_limits:
        list_of_dfs = []
        if not pd_facets:
            pd_facets = chempot_limits[
                "facets"
            ].keys()  # Phase diagram facets to use for chemical
            # potentials, to tabulate formation energies
        for facet in pd_facets:
            bold_print("Facet: " + unicodeify(facet))
            df = single_formation_energy_table(
                defect_phase_diagram,
                chempots=chempot_limits["facets"][facet],
                fermi_level=fermi_level,
                hide_cols=hide_cols,
                show_key=show_key,
            )
            list_of_dfs.append(df)
            print("\n")

        return list_of_dfs

    # else return {Elt: Energy} dict for chempot_limits, or if unspecified, all zero energy
    df = single_formation_energy_table(
        defect_phase_diagram,
        chempots=chempot_limits,
        fermi_level=fermi_level,
        hide_cols=hide_cols,
        show_key=show_key,
    )
    return df


def single_formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempots: dict = None,
    fermi_level: float = 0,
    hide_cols: list = None,
    show_key: bool = True,
):
    """
    Prints a defect formation energy table for a single chemical potential limit (i.e. phase diagram
    facet), and returns the results as a pandas DataFrame.

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
             DefectPhaseDiagram object (likely created from
             analysis.dpd_from_defect_dict)
        chempots (dict):
            Dictionary of chosen absolute/DFT chemical potentials: {Elt: Energy}. If not
            specified, chemical potentials are not included in the formation energy calculation
            (all set to zero energy).
        fermi_level (float):
            Fermi level to use for computing the defect formation energies. (default: 0 (i.e.
            at the VBM))
        hide_cols: (list):
            List of columns to hide from the output. (default: None)
        show_key (bool):
            Whether or not to print the table key at the bottom of the output. (default: True)

    Returns:
        pandas DataFrame sorted by formation energy
    """
    header = ["Defect", "Charge", "Defect Path"]
    table = []
    if hide_cols is None:
        hide_cols = []

    for defect_entry in defect_phase_diagram.entries:
        row = [
            defect_entry.name,
            defect_entry.charge,
            defect_entry.parameters["defect_path"],
        ]
        if "Uncorrected Energy" not in hide_cols:
            header += ["Uncorrected Energy"]
            row += [f"{defect_entry.uncorrected_energy:.2f} eV"]
        if "Corrected Energy" not in hide_cols:
            header += ["Corrected Energy"]
            row += [
                f"{defect_entry.energy:.2f} eV"
            ]  # With 0 chemical potentials, at the calculation
            # fermi level
        header += ["Formation Energy"]
        formation_energy = defect_entry.formation_energy(
            chemical_potentials=chempots, fermi_level=fermi_level
        )
        row += [f"{formation_energy:.2f} eV"]

        table.append(row)
    table = sorted(table, key=itemgetter(0, 1))
    print(
        tabulate(
            table,
            headers=header,
            tablefmt="fancy_grid",
            stralign="left",
            numalign="left",
        ),
        "\n",
    )

    if show_key:
        bold_print("Table Key:")
        print(
            """'Defect' -> Defect Type and Multiplicity
'Charge' -> Defect Charge State
'Uncorrected Energy' -> Defect Energy from calculation, without corrections
'Corrected Energy' -> Defect Energy from calculation (E_defect - E_host + corrections)
(chemical potentials set to 0 and the fermi level at average electrostatic potential in the
supercell)
'Formation Energy' -> Final Defect Formation Energy, with the specified chemical potentials (
chempot_limits)(default: all 0) and the chosen fermi_level (default: 0)(i.e. at the VBM)
        """
        )

    sorted_df = pd.DataFrame(
        table,
        columns=[
            "Defect",
            "Charge",
            "Defect Path",
            "Uncorrected Energy",
            "Corrected Energy",
            "Formation Energy",
        ],
    )
    sorted_df = sorted_df.sort_values("Formation Energy")
    return sorted_df


def lany_zunger_corrected_defect_dict_from_freysoldt(defect_dict: dict):
    """Convert input parsed defect dictionary (presumably created using SingleDefectParser
    from doped.pycdt.utils.parse_calculations) with Freysoldt charge corrections to
    the same parsed defect dictionary but with the Lany-Zunger charge correction (same potential
    alignment plus 0.65 * Makov-Payne image charge correction (same image charge correction as
    Freysoldt scheme)).
    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (presumably created using SingleDefectParser
            from doped.pycdt.utils.parse_calculations) (see example notebook)
            Must have 'freysoldt_meta' in defect.parameters for each charged defect (from
            SingleDefectParser.freysoldt_loader())
    Returns:
        Parsed defect dictionary with Lany-Zunger charge corrections.
    """
    from doped.corrections import (
        get_murphy_image_charge_correction,
    )  # avoid circular import

    random_defect_entry = list(defect_dict.values())[
        0
    ]  # Just need any DefectEntry from
    # defect_dict to get the lattice and dielectric matrix
    lattice = random_defect_entry.bulk_structure.lattice.matrix
    dielectric = random_defect_entry.parameters["dielectric"]
    lz_image_charge_corrections = get_murphy_image_charge_correction(
        lattice, dielectric
    )
    lz_corrected_defect_dict = copy.deepcopy(defect_dict)
    for defect_name, defect_entry in lz_corrected_defect_dict.items():
        if defect_entry.charge != 0:
            potalign = defect_entry.parameters["freysoldt_meta"][
                "freysoldt_potential_alignment_correction"
            ]
            mp_pc_corr = lz_image_charge_corrections[
                abs(defect_entry.charge)
            ]  # Makov-Payne PC correction
            defect_entry.parameters.update(
                {
                    "Lany-Zunger_Corrections": {
                        "(Freysoldt)_Potential_Alignment_Correction": potalign,
                        "Makov-Payne_Image_Charge_Correction": mp_pc_corr,
                        "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65 * mp_pc_corr,
                        "Total_Lany-Zunger_Correction": potalign + 0.65 * mp_pc_corr,
                    }
                }
            )
            defect_entry.corrections["charge_correction"] = defect_entry.parameters[
                "Lany-Zunger_Corrections"
            ]["Total_Lany-Zunger_Correction"]

        lz_corrected_defect_dict.update({defect_name: defect_entry})
    return lz_corrected_defect_dict


def lany_zunger_corrected_defect_dict_from_kumagai(defect_dict: dict):
    """Convert input parsed defect dictionary (presumably created using SingleDefectParser
    from doped.pycdt.utils.parse_calculations) with Kumagai charge corrections to
    the same parsed defect dictionary but with the 'Lany-Zunger' charge correction (same potential
    alignment plus 0.65 * image charge correction.
    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (presumably created using SingleDefectParser
            from doped.pycdt.utils.parse_calculations) (see example notebook)
            Must have 'kumagai_meta' in defect.parameters for each charged defect (from
            SingleDefectParser.kumagai_loader())
    Returns:
        Parsed defect dictionary with Lany-Zunger charge corrections.
    """
    from doped.corrections import (
        get_murphy_image_charge_correction,
    )  # avoid circular import

    random_defect_entry = list(defect_dict.values())[
        0
    ]  # Just need any DefectEntry from
    # defect_dict to get the lattice and dielectric matrix
    lattice = random_defect_entry.bulk_structure.lattice.matrix
    dielectric = random_defect_entry.parameters["dielectric"]
    lz_image_charge_corrections = get_murphy_image_charge_correction(
        lattice, dielectric
    )
    lz_corrected_defect_dict = copy.deepcopy(defect_dict)
    for defect_name, defect_entry in lz_corrected_defect_dict.items():
        if defect_entry.charge != 0:
            potalign = defect_entry.parameters["kumagai_meta"][
                "kumagai_potential_alignment_correction"
            ]
            makove_payne_pc_correction = lz_image_charge_corrections[
                abs(defect_entry.charge)
            ]
            defect_entry.parameters.update(
                {
                    "Lany-Zunger_Corrections": {
                        "(Kumagai)_Potential_Alignment_Correction": potalign,
                        "Makov-Payne_Image_Charge_Correction": makove_payne_pc_correction,
                        "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65
                        * makove_payne_pc_correction,
                        "Total_Lany-Zunger_Correction": potalign
                        + 0.65 * makove_payne_pc_correction,
                    }
                }
            )
            defect_entry.corrections["charge_correction"] = defect_entry.parameters[
                "Lany-Zunger_Corrections"
            ]["Total_Lany-Zunger_Correction"]

        lz_corrected_defect_dict.update({defect_name: defect_entry})
    return lz_corrected_defect_dict
