"""
Code to analyse VASP defect calculations.

These functions are built from a combination of useful modules from pymatgen,
alongside substantial modification, in the efforts of making an efficient,
user-friendly package for managing and analysing defect calculations, with
publication-quality outputs.
"""
import contextlib
import os
import warnings
from typing import Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.util.string import unicodeify
from shakenbreak.input import _get_voronoi_nodes
from tabulate import tabulate

from doped import _ignore_pmg_warnings
from doped.core import DefectEntry, Interstitial, Substitution, Vacancy
from doped.generation import get_defect_name_from_entry
from doped.plotting import _format_defect_name
from doped.utils.legacy_pmg.defect_compatibility import DefectCompatibility
from doped.utils.legacy_pmg.thermodynamics import DefectPhaseDiagram
from doped.utils.parsing import (
    _get_output_files_and_check_if_multiple,
    get_defect_site_idxs_and_unrelaxed_structure,
    get_defect_type_and_composition_diff,
    get_locpot,
    get_outcar,
    get_vasprun,
    reorder_s1_like_s2,
)
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
            f"{get_defect_name_from_entry(defect_entry)}_"
            f"{'+' if charge_state > 0 else ''}{charge_state}"
        )  # otherwise use default doped name  # TODO: Test!


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
#  `check_defects_compatibility()` at the end of parsing the full defects dict. - When doing
#  this, look at `PostProcess` code, and then delete it once all functionality is moved here.
# TODO: With function/class for parsing all defects, implement the doped naming scheme as our fallback
#  option when the folder names aren't recognised (rather than defaulting to pmg name as currently the
#  case)
# TODO: Automatically pull the magnetisation from the VASP calc to determine the spin multiplicity
#  (for later integration with `py-sc-fermi`).
# TODO: Can we add functions to auto-determine the orientational degeneracy? Any decent tools for this atm?


def defect_entry_from_paths(
    defect_path,
    bulk_path,
    dielectric,
    charge_state=None,
    initial_defect_structure=None,
    skip_corrections=False,
    bulk_bandgap_path=None,
    **kwargs,
):
    """
    Parse the defect calculation outputs in `defect_path` and return the parsed
    `DefectEntry` object. By default, the `DefectEntry.name` attribute (later
    used to label the defects in plots) is set to the defect_path folder name
    (if it is a recognised defect name), else it is set to the default doped
    name for that defect.

    Args:
        defect_path (str):
            Path to defect supercell folder (containing at least vasprun.xml(.gz)).
        bulk_path (str):
            Path to bulk supercell folder (containing at least vasprun.xml(.gz)).
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Ionic + static contributions to the dielectric constant.
        charge_state (int):
            Charge state of defect. If not provided, will be automatically determined
            from the defect calculation outputs (requires `POTCAR`s to be set up
            with `pymatgen`).
        initial_defect_structure (str):
            Path to the initial/unrelaxed defect structure. Only recommended for use
            if structure matching with the relaxed defect structure(s) fails (rare).
            Default is None.
        skip_corrections (bool):
            Whether to skip the calculation and application of finite-size charge
            corrections to the defect energy (not recommended in most cases).
            Default = False.
        bulk_bandgap_path (str):
            Path to bulk OUTCAR file for determining the band gap. If the VBM/CBM
            occur at reciprocal space points not included in the bulk supercell
            calculation, you should use this tag to point to a bulk bandstructure
            calculation instead.
            If None, will use self.defect_entry.calculation_metadata["bulk_path"].
        **kwargs: Additional keyword arguments to pass to `DefectParser()`.

    Return:
        Parsed `DefectEntry` object.
    """
    _ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings
    dielectric = _convert_dielectric_to_tensor(dielectric)

    calculation_metadata = {
        "bulk_path": bulk_path,
        "defect_path": defect_path,
        "dielectric": dielectric,
    }

    # add bulk simple properties
    bulk_vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", bulk_path)
    if multiple:
        warnings.warn(
            f"Multiple `vasprun.xml` files found in bulk directory: {bulk_path}. Using "
            f"{bulk_vr_path} to parse the calculation energy and metadata."
        )
    bulk_vr = get_vasprun(bulk_vr_path)
    bulk_supercell = bulk_vr.final_structure.copy()

    # add defect simple properties
    (
        defect_vr_path,
        multiple,
    ) = _get_output_files_and_check_if_multiple("vasprun.xml", defect_path)
    if multiple:
        warnings.warn(
            f"Multiple `vasprun.xml` files found in defect directory: {defect_path}. Using"
            f" {defect_vr_path} to parse the calculation energy and metadata."
        )
    defect_vr = get_vasprun(defect_vr_path)

    # get defect charge
    try:
        defect_nelect = defect_vr.incar.get("NELECT", None)
        if defect_nelect is None:
            auto_charge = 0  # neutral defect if NELECT not specified
        else:
            potcar_symbols = [titel.split()[1] for titel in defect_vr.potcar_symbols]
            potcar_settings = {symbol.split("_")[0]: symbol for symbol in potcar_symbols}
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
                        "with pymatgen (see Step 2 at https://github.com/SMTG-UCL/doped#installation). "
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
                f"Auto-determined defect charge q={int(auto_charge):+} does not match "
                f"specified charge q={int(charge_state):+}. Will continue with specified "
                f"charge_state, but beware!"
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

    # Add defect structure to calculation_metadata, so it can be pulled later on (eg. for Kumagai loader)
    defect_structure = defect_vr.final_structure.copy()
    calculation_metadata["defect_structure"] = defect_structure

    # identify defect site, structural information, and create defect object:
    # Can specify initial defect structure (to help find the defect site if
    # multiple relaxations were required, else use from defect relaxation OUTCAR):
    if initial_defect_structure:
        defect_structure_for_ID = Poscar.from_file(initial_defect_structure).structure.copy()
    else:
        defect_structure_for_ID = defect_structure.copy()
    try:
        def_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, defect_structure_for_ID)
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_supercell)} in bulk vs. {len(defect_structure_for_ID)} in defect?"
        ) from exc

    # Try automatic defect site detection - this gives us the "unrelaxed" defect structure
    try:
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_supercell, defect_structure_for_ID, def_type, comp_diff
        )

    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not identify {def_type} defect site in defect structure. Try supplying the initial "
            f"defect structure to DefectParser.from_paths()."
        ) from exc

    if def_type == "vacancy":
        defect_site = guessed_initial_defect_site = bulk_supercell[bulk_site_idx]
    else:
        defect_site = defect_structure_for_ID[defect_site_idx]
        guessed_initial_defect_site = unrelaxed_defect_structure[defect_site_idx]

    if unrelaxed_defect_structure:
        if def_type == "interstitial":
            # get closest Voronoi site in bulk supercell to final interstitial site as this is
            # likely to be the _initial_ interstitial site
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
                        "Previous bulk_voronoi_nodes.json detected, but does not "
                        "match current bulk supercell. Recalculating Voronoi nodes."
                    )
                    raise FileNotFoundError

                voronoi_frac_coords = struc_and_node_dict["Voronoi nodes"]

            except FileNotFoundError:  # first time parsing
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
            guessed_initial_defect_site = guessed_initial_defect_structure[defect_site_idx]

        else:
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()

        # ensure unrelaxed_defect_structure ordered to match defect_structure, for appropriate charge
        # correction mapping
        unrelaxed_defect_structure = reorder_s1_like_s2(
            unrelaxed_defect_structure, defect_structure_for_ID
        )
        calculation_metadata["guessed_initial_defect_structure"] = guessed_initial_defect_structure
        calculation_metadata["unrelaxed_defect_structure"] = unrelaxed_defect_structure
    else:
        warnings.warn(
            "Cannot determine the unrelaxed `initial_defect_structure`. Please ensure the "
            "`initial_defect_structure` is indeed unrelaxed."
        )

    for_monty_defect = {  # initialise doped Defect object
        "@module": "doped.core",
        "@class": def_type.capitalize(),
        "structure": bulk_supercell,
        "site": guessed_initial_defect_site,
    }  # note that we now define the Defect in the bulk supercell, rather than the primitive structure
    # as done during generation. Future work could try mapping the relaxed defect site back to the
    # primitive cell, however interstitials will be tricky for this...
    defect = MontyDecoder().process_decoded(for_monty_defect)

    if unrelaxed_defect_structure:
        # only do StructureMatcher test if unrelaxed structure exists
        test_defect_structure = defect.get_supercell_structure(
            min_atoms=len(bulk_supercell), max_atoms=len(bulk_supercell)
        )
        if not StructureMatcher(
            stol=0.05,
            comparator=ElementComparator(),
        ).fit(test_defect_structure, guessed_initial_defect_structure):
            warnings.warn(
                f"Possible error in defect object matching. Determined defect: {defect.name} for defect "
                f"at {defect_path} in bulk at {bulk_path} but unrelaxed structure (1st below) does not "
                f"match defect.get_supercell_structure() (2nd below):"
                f"\n{guessed_initial_defect_structure}\n{test_defect_structure}"
            )

    defect_entry = DefectEntry(
        # pmg attributes:
        defect=defect,
        charge_state=charge_state,
        sc_entry=defect_vr.get_computed_entry(),
        sc_defect_frac_coords=defect_site.frac_coords,
        bulk_entry=bulk_vr.get_computed_entry(),
        # doped attributes:
        defect_supercell_site=defect_site,
        defect_supercell=defect_vr.final_structure,
        bulk_supercell=bulk_vr.final_structure,
        calculation_metadata=calculation_metadata,
    )

    defect_name = os.path.basename(defect_path)  # set equal to folder name
    if "vasp" in defect_name:  # get parent directory name:
        defect_name = os.path.basename(os.path.dirname(defect_path))

    check_and_set_defect_entry_name(defect_entry, defect_name)

    return_dp = kwargs.pop("return DefectParser", False)  # internal use for tests/debugging
    dp = DefectParser(
        defect_entry,
        defect_vr=defect_vr,
        bulk_vr=bulk_vr,
        **kwargs,  # in case user wants to specify `DefectCompatibility()`
    )
    if return_dp:
        return dp

    dp.get_stdrd_metadata()  # Load standard defect metadata
    dp.get_bulk_gap_data(bulk_bandgap_path=bulk_bandgap_path)  # Load band gap data

    if not skip_corrections:
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

        # regardless, try parsing OUTCAR files first (quickest)
        if _check_folder_for_file_match(defect_path, "OUTCAR") and _check_folder_for_file_match(
            bulk_path, "OUTCAR"
        ):
            try:
                dp.kumagai_loader()
            except Exception as kumagai_exc:
                if _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
                    bulk_path, "LOCPOT"
                ):
                    try:
                        if not isotropic_dielectric:
                            # convert anisotropic dielectric to harmonic mean of the diagonal:
                            # (this is a better approximation than the pymatgen default of the
                            # standard arithmetic mean of the diagonal)
                            dp.defect_entry.calculation_metadata[
                                "dielectric"
                            ] = _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                        dp.freysoldt_loader()
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
                            dp.defect_entry.calculation_metadata["dielectric"] = dielectric
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

        elif _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
            bulk_path, "LOCPOT"
        ):
            try:
                if not isotropic_dielectric:
                    # convert anisotropic dielectric to harmonic mean of the diagonal:
                    # (this is a better approximation than the pymatgen default of the
                    # standard arithmetic mean of the diagonal)
                    dp.defect_entry.calculation_metadata[
                        "dielectric"
                    ] = _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                dp.freysoldt_loader()
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
                    dp.defect_entry.calculation_metadata["dielectric"] = dielectric
                skip_corrections = True

        else:
            if int(dp.defect_entry.charge_state) != 0:
                warnings.warn(
                    f"`LOCPOT` or `OUTCAR` files are not present in both the defect (at "
                    f"{defect_path}) and bulk (at {bulk_path}) folders. These are needed to "
                    f"perform the finite-size charge corrections. Charge corrections will not be "
                    f"applied for this defect."
                )
                skip_corrections = True

        if not skip_corrections:
            # Check compatibility of defect corrections with loaded metadata, and apply
            dp.run_compatibility()

            # check that charge corrections are not negative
            summed_corrections = sum(
                val
                for key, val in dp.defect_entry.corrections.items()
                if any(i in key.lower() for i in ["freysoldt", "kumagai", "fnv", "charge"])
            )
            if summed_corrections < 0:
                warnings.warn(
                    f"The calculated finite-size charge corrections for defect at {defect_path} and bulk "
                    f"at {bulk_path} sum to a negative value of {summed_corrections:.3f}. This is likely "
                    f"due to some error or mismatch in the defect and bulk calculations, as the defect "
                    f"charge correction energy should (almost always) be positive. Please double-check "
                    f"your calculations and parsed results!"
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
            DefectParser.get_bulk_gap_data())

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

    return DefectPhaseDiagram(
        list(defect_dict.values()), vbm_vals[0], bandgap_vals[0], filter_compatible=False
    )


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
    chempot_limits: Optional[Dict] = None,
    facets: Optional[List] = None,
    fermi_level: float = 0,
    hide_cols: Optional[List] = None,
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
    facets argument, or if not specified, will print formation energy tables for each facet in
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
        facets (list):
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
        if facets is None:
            facets = chempot_limits["facets"].keys()  # Phase diagram facets to use for chemical
            # potentials, to tabulate formation energies
        for facet in facets:
            bold_print("Facet: " + unicodeify(facet))
            single_formation_energy_df = single_formation_energy_table(
                defect_phase_diagram,
                chempot_limits=chempot_limits["facets"][facet],
                fermi_level=fermi_level,
                hide_cols=hide_cols,
                show_key=show_key,
            )
            list_of_dfs.append(single_formation_energy_df)
            print("\n")

        return list_of_dfs

    # else return {Elt: Energy} dict for chempot_limits, or if unspecified, all zero energy
    single_formation_energy_df = single_formation_energy_table(
        defect_phase_diagram,
        chempot_limits=chempot_limits,
        fermi_level=fermi_level,
        hide_cols=hide_cols,
        show_key=show_key,
    )
    return single_formation_energy_df


def single_formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempot_limits: Optional[Dict] = None,
    fermi_level: float = 0,
    hide_cols: Optional[List] = None,
    show_key: bool = True,
):
    """
    Prints a defect formation energy table for a single chemical potential
    limit (i.e. phase diagram facet), and returns the results as a pandas
    DataFrame.

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
             DefectPhaseDiagram object (likely created from
             analysis.dpd_from_defect_dict)
        chempot_limits (dict):
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
    header = ["Defect", "q", "Path"]
    table = []
    if hide_cols is None:
        hide_cols = []

    defect_entries = defect_phase_diagram.entries
    # sort by defect name, then charge state (most positive to most negative), then energy:
    defect_entries = sorted(
        defect_entries, key=lambda entry: (entry.defect.name, -entry.charge_state, entry.get_ediff())
    )
    for defect_entry in defect_entries:
        row = [
            defect_entry.name,
            defect_entry.charge_state,
            defect_entry.calculation_metadata.get("defect_path", "N/A"),
        ]
        if "ΔEʳᵃʷ" not in hide_cols:
            header += ["ΔEʳᵃʷ"]
            row += [
                f"{defect_entry.get_ediff() - sum(defect_entry.corrections.values()):.2f} eV"
            ]  # With 0 chemical potentials, at the calculation fermi level
        if "E_corr" not in hide_cols:
            header += ["E_corr"]
            row += [f"{sum(defect_entry.corrections.values()):.2f} eV"]
        if "Σμ" not in hide_cols:
            header += ["Σμ"]
            row += [f"{defect_phase_diagram._get_chempot_term(defect_entry, chempot_limits):.2f} eV"]
        header += ["ΔEᶠᵒʳᵐ"]
        formation_energy = defect_phase_diagram._formation_energy(
            defect_entry, chemical_potentials=chempot_limits, fermi_level=fermi_level
        )
        row += [f"{formation_energy:.2f} eV"]

        table.append(row)

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
            """'Defect' -> Defect type and multiplicity.
'q' -> Defect charge state.
'ΔEʳᵃʷ' -> Energy difference between defect and host supercell (E_defect - E_host).
(chemical potentials set to 0 and the fermi level at average electrostatic potential in the supercell).
'E_corr' -> Defect energy correction.
'Σμ' -> Sum of chemical potential terms in the formation energy equation.
'ΔEᶠᵒʳᵐ' -> Final defect formation energy, with the specified chemical potentials (
chempot_limits)(default: all 0) and the chosen fermi_level (default: 0)(i.e. at the VBM).
        """
        )

    return pd.DataFrame(
        table,
        columns=[
            "Defect",
            "q",
            "Path",
            "ΔEʳᵃʷ",
            "E_corr",
            "Σμ",
            "ΔEᶠᵒʳᵐ",
        ],
    )


# def lany_zunger_corrected_defect_dict_from_freysoldt(defect_dict: dict):
#     """
#     Convert input parsed defect dictionary (presumably created using
#     DefectParser) with Freysoldt charge corrections to the same parsed
#     defect dictionary but with the Lany-Zunger charge correction (same potential
#     alignment plus 0.65 * Makov-Payne image charge correction (same image charge correction as
#     Freysoldt scheme)).
#
#     Args:
#         defect_dict (dict):
#             Dictionary of parsed defect calculations (presumably created
#             using DefectParser (see tutorials)
#             Must have 'freysoldt_meta' in defect.calculation_metadata for each charged defect (from
#             DefectParser.freysoldt_loader())
#
#     Returns:
#         Parsed defect dictionary with Lany-Zunger charge corrections.
#     """
#     from doped.utils.corrections import get_murphy_image_charge_correction
#
#     random_defect_entry = list(defect_dict.values())[0]  # Just need any DefectEntry from
#     # defect_dict to get the lattice and dielectric matrix
#     lattice = random_defect_entry.bulk_structure.lattice.matrix
#     dielectric = random_defect_entry.calculation_metadata["dielectric"]
#     lz_image_charge_corrections = get_murphy_image_charge_correction(lattice, dielectric)
#     lz_corrected_defect_dict = copy.deepcopy(defect_dict)
#     for defect_name, defect_entry in lz_corrected_defect_dict.items():
#         if defect_entry.charge_state != 0:
#             potalign = defect_entry.calculation_metadata["freysoldt_meta"][
#                 "freysoldt_potential_alignment_correction"
#             ]
#             mp_pc_corr = lz_image_charge_corrections[abs(defect_entry.charge_state)]  # Makov-Payne PC
#             correction
#             defect_entry.calculation_metadata.update(
#                 {
#                     "Lany-Zunger_Corrections": {
#                         "(Freysoldt)_Potential_Alignment_Correction": potalign,
#                         "Makov-Payne_Image_Charge_Correction": mp_pc_corr,
#                         "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65 * mp_pc_corr,
#                         "Total_Lany-Zunger_Correction": potalign + 0.65 * mp_pc_corr,
#                     }
#                 }
#             )
#             defect_entry.corrections["charge_correction"] = defect_entry.calculation_metadata[
#                 "Lany-Zunger_Corrections"
#             ]["Total_Lany-Zunger_Correction"]
#
#         lz_corrected_defect_dict.update({defect_name: defect_entry})
#     return lz_corrected_defect_dict
#
#
# def lany_zunger_corrected_defect_dict_from_kumagai(defect_dict: dict):
#     """
#     Convert input parsed defect dictionary (presumably created using
#     DefectParser) with Kumagai charge corrections to the same parsed
#     defect dictionary but with the 'Lany-Zunger' charge correction
#     (same potential alignment plus 0.65 * image charge correction.
#
#     Args:
#         defect_dict (dict):
#             Dictionary of parsed defect calculations (presumably created
#             using DefectParser (see tutorials)
#             Must have 'kumagai_meta' in defect.calculation_metadata for
#             each charged defect (from DefectParser.kumagai_loader())
#
#     Returns:
#         Parsed defect dictionary with Lany-Zunger charge corrections.
#     """
#     from doped.utils.corrections import get_murphy_image_charge_correction
#
#     random_defect_entry = list(defect_dict.values())[0]  # Just need any DefectEntry from
#     # defect_dict to get the lattice and dielectric matrix
#     lattice = random_defect_entry.bulk_structure.lattice.matrix
#     dielectric = random_defect_entry.calculation_metadata["dielectric"]
#     lz_image_charge_corrections = get_murphy_image_charge_correction(lattice, dielectric)
#     lz_corrected_defect_dict = copy.deepcopy(defect_dict)
#     for defect_name, defect_entry in lz_corrected_defect_dict.items():
#         if defect_entry.charge_state != 0:
#             potalign = defect_entry.calculation_metadata["kumagai_meta"][
#             "kumagai_potential_alignment_correction"]
#             makove_payne_pc_correction = lz_image_charge_corrections[abs(defect_entry.charge_state)]
#             defect_entry.calculation_metadata.update(
#                 {
#                     "Lany-Zunger_Corrections": {
#                         "(Kumagai)_Potential_Alignment_Correction": potalign,
#                         "Makov-Payne_Image_Charge_Correction": makove_payne_pc_correction,
#                         "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65 * makove_payne_pc_correction,
#                         "Total_Lany-Zunger_Correction": potalign + 0.65 * makove_payne_pc_correction,
#                     }
#                 }
#             )
#             defect_entry.corrections["charge_correction"] = defect_entry.calculation_metadata[
#                 "Lany-Zunger_Corrections"
#             ]["Total_Lany-Zunger_Correction"]
#
#         lz_corrected_defect_dict.update({defect_name: defect_entry})
#     return lz_corrected_defect_dict


class DefectParser:
    _delocalization_warning_printed = False  # class variable
    # ensures the verbose delocalization analysis warning is only printed once. Needs to be done
    # this way because the current workflow is to create a `DefectParser` object for each
    # defect, and then warning originates from the `run_compatibility()` method of different
    # `DefectParser` instances, so warnings detects each instance as a different source and
    # prints the warning multiple times. When we move to a single function call for all defects
    # (as described above), this can be removed.

    def __init__(
        self,
        defect_entry,
        compatibility=None,
        defect_vr=None,
        bulk_vr=None,
    ):
        """
        Parse a single Defect object.

        Args:
            defect_entry (DefectEntry):
                doped DefectEntry
            compatibility (DefectCompatibility):
                Compatibility class instance for performing charge correction compatibility analysis on
                the defect entry.
            defect_vr (Vasprun): pymatgen Vasprun object for the defect supercell calculation
            bulk_vr (Vasprun): pymatgen Vasprun object for the reference bulk supercell calculation
        """
        self.defect_entry: DefectEntry = defect_entry
        self.compatibility = compatibility or DefectCompatibility(
            plnr_avg_var_tol=0.01,
            plnr_avg_minmax_tol=0.3,
            atomic_site_var_tol=0.025,
            atomic_site_minmax_tol=0.3,
            tot_relax_tol=5.0,
            defect_tot_relax_tol=5.0,
            use_bandfilling=False,  # don't include bandfilling by default
        )
        self.defect_vr = defect_vr
        self.bulk_vr = bulk_vr

    @classmethod
    def from_paths(
        cls,
        defect_path,
        bulk_path,
        dielectric,
        charge_state=None,
        initial_defect_structure=None,
        skip_corrections=False,
        bulk_bandgap_path=None,
        **kwargs,
    ):
        """
        Parse the defect calculation outputs in `defect_path` and return the
        parsed `DefectEntry` object.

        Args:
        defect_path (str): Path to defect folder of interest (with vasprun.xml(.gz))
        bulk_path (str): Path to bulk folder of interest (with vasprun.xml(.gz))
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Ionic + static contributions to dielectric constant.
        charge_state (int):
            Charge state of defect. If not provided, then automatically determined
            from the defect calculation outputs (requires POTCARs to be set up with
            `pymatgen`).
        initial_defect_structure (str):
            Path to the initial/unrelaxed defect structure. Only recommended for use
            if structure matching with the relaxed defect structure(s) fails.
        skip_corrections (bool):
            Whether to skip the calculation of finite-size charge corrections for the
            DefectEntry.
        bulk_bandgap_path (str):
            Path to bulk OUTCAR file for determining the band gap. If the VBM/CBM occur
            at reciprocal space points not included in the bulk supercell calculation,
            you should use this tag to point to a bulk bandstructure calculation instead.
            If None, will use self.defect_entry.calculation_metadata["bulk_path"].
        **kwargs: kwargs to pass to `defect_entry_from_paths()`

        Return:
            Parsed `DefectEntry` object.
        """
        kwargs["return DefectParser"] = True
        return defect_entry_from_paths(
            defect_path,
            bulk_path,
            dielectric,
            charge_state=charge_state,
            initial_defect_structure=initial_defect_structure,
            skip_corrections=skip_corrections,
            bulk_bandgap_path=bulk_bandgap_path,
            **kwargs,
        )

    def _multiple_files_warning(self, file_type, directory, chosen_filepath, action, dir_type="bulk"):
        warnings.warn(
            f"Multiple `{file_type}` files found in {dir_type} directory: {directory}. Using "
            f"{chosen_filepath} to {action}"
        )

    def freysoldt_loader(self, bulk_locpot=None):
        """
        Load metadata required for performing Freysoldt correction requires
        "bulk_path" and "defect_path" to be loaded to DefectEntry
        calculation_metadata dict. Can read gunzipped "LOCPOT.gz" files as
        well.

        Args:
            bulk_locpot (Locpot): Add bulk Locpot object for expedited parsing.
                If None, will load from file path variable bulk_path
        Return:
            bulk_locpot object for reuse by another defect entry (for expedited parsing)
        """
        if not self.defect_entry.charge_state:
            # don't need to load locpots if charge is zero
            return

        if not bulk_locpot:
            bulk_locpot_path, multiple = _get_output_files_and_check_if_multiple(
                "LOCPOT", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                self._multiple_files_warning(
                    "LOCPOT",
                    self.defect_entry.calculation_metadata["bulk_path"],
                    bulk_locpot_path,
                    "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
                    dir_type="bulk",
                )
            bulk_locpot = get_locpot(bulk_locpot_path)

        defect_locpot_path, multiple = _get_output_files_and_check_if_multiple(
            "LOCPOT", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            self._multiple_files_warning(
                "LOCPOT",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_locpot_path,
                "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
                dir_type="defect",
            )
        defect_locpot = get_locpot(defect_locpot_path)

        axis_grid = [defect_locpot.get_axis_grid(i) for i in range(3)]
        bulk_planar_averages = [bulk_locpot.get_average_along_axis(i) for i in range(3)]
        defect_planar_averages = [defect_locpot.get_average_along_axis(i) for i in range(3)]

        self.defect_entry.calculation_metadata.update(
            {
                "axis_grid": axis_grid,
                "bulk_planar_averages": bulk_planar_averages,
                "defect_planar_averages": defect_planar_averages,
                "defect_frac_sc_coords": self.defect_entry.sc_defect_frac_coords,
            }
        )

    def kumagai_loader(self, bulk_outcar=None):
        """
        Load metadata required for performing Kumagai correction requires
        "bulk_path" and "defect_path" to be loaded to DefectEntry
        calculation_metadata dict.

        Args:
            bulk_outcar (Outcar): Add bulk Outcar object for expedited parsing.
                If None, will load from file path variable bulk_path
        Return:
            bulk_outcar object for reuse by another defect entry (for expedited parsing)
        """
        if not self.defect_entry.charge_state:
            # don't need to load outcars if charge is zero
            return

        if not bulk_outcar:
            bulk_outcar_path, multiple = _get_output_files_and_check_if_multiple(
                "OUTCAR", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                self._multiple_files_warning(
                    "OUTCAR",
                    self.defect_entry.calculation_metadata["bulk_path"],
                    bulk_outcar_path,
                    "parse core levels and compute the Kumagai (eFNV) image charge correction.",
                    dir_type="bulk",
                )
            bulk_outcar = get_outcar(bulk_outcar_path)
        else:
            bulk_outcar_path = bulk_outcar.filename

        defect_outcar_path, multiple = _get_output_files_and_check_if_multiple(
            "OUTCAR", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            self._multiple_files_warning(
                "OUTCAR",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_outcar_path,
                "parse core levels and compute the Kumagai (eFNV) image charge correction.",
                dir_type="defect",
            )
        defect_outcar = get_outcar(defect_outcar_path)

        bulk_atomic_site_averages = bulk_outcar.electrostatic_potential
        defect_atomic_site_averages = defect_outcar.electrostatic_potential

        def _raise_incomplete_outcar_error(outcar_path, dir_type="bulk"):
            raise ValueError(
                f"Unable to parse atomic core potentials from {dir_type} `OUTCAR` at {outcar_path}. This "
                f"can happen if `ICORELEVEL` was not set to 0 (= default) in the `INCAR`, or if the "
                f"calculation was finished prematurely with a `STOPCAR`. The Kumagai charge correction "
                f"cannot be computed without this data!"
            )

        if not bulk_atomic_site_averages:
            _raise_incomplete_outcar_error(bulk_outcar_path, dir_type="bulk")

        if not defect_atomic_site_averages:
            _raise_incomplete_outcar_error(defect_outcar_path, dir_type="defect")

        bulk_structure = self.defect_entry.bulk_entry.structure
        bulksites = [site.frac_coords for site in bulk_structure]

        defect_structure = self.defect_entry.calculation_metadata["unrelaxed_defect_structure"]
        initsites = [site.frac_coords for site in defect_structure]

        distmatrix = bulk_structure.lattice.get_all_distances(
            bulksites, initsites  # TODO: Should be able to take this from the defect ID functions?
        )  # first index of this list is bulk index
        min_dist_with_index = [
            [
                min(distmatrix[bulk_index]),
                int(bulk_index),
                int(distmatrix[bulk_index].argmin()),
            ]
            for bulk_index in range(len(distmatrix))
        ]  # list of [min dist, bulk ind, defect ind]

        site_matching_indices = []
        if isinstance(self.defect_entry.defect, (Vacancy, Interstitial)):
            for mindist, bulk_index, defect_index in min_dist_with_index:
                if mindist < 0.5:
                    site_matching_indices.append([bulk_index, defect_index])

        elif isinstance(self.defect_entry.defect, Substitution):
            for mindist, bulk_index, defect_index in min_dist_with_index:
                species_match = bulk_structure[bulk_index].specie == defect_structure[defect_index].specie
                if mindist < 0.5 and species_match:
                    site_matching_indices.append([bulk_index, defect_index])

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_atomic_site_averages": bulk_atomic_site_averages,
                "defect_atomic_site_averages": defect_atomic_site_averages,
                "site_matching_indices": site_matching_indices,
            }
        )

    def get_stdrd_metadata(self):
        """
        Get metadata required for standard defect parsing.
        """
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                self._multiple_files_warning(
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
                self._multiple_files_warning(
                    "vasprun.xml",
                    self.defect_entry.calculation_metadata["defect_path"],
                    defect_vr_path,
                    "parse the calculation energy and metadata.",
                    dir_type="defect",
                )
            self.defect_vr = get_vasprun(defect_vr_path)

        run_metadata = {  # TODO: Add check that incars, kpoints and potcars are compatible here
            "defect_incar": self.defect_vr.incar,
            "bulk_incar": self.bulk_vr.incar,
            "defect_kpoints": self.defect_vr.kpoints,
            "bulk_kpoints": self.bulk_vr.kpoints,
            "defect_potcar_symbols": self.defect_vr.potcar_spec,
            "bulk_potcar_symbols": self.bulk_vr.potcar_spec,
        }

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

    def get_bulk_gap_data(self, bulk_bandgap_path=None, use_MP=False, mpid=None, api_key=None):
        """
        Get bulk gap data from bulk OUTCAR file, or OUTCAR located at
        `actual_bulk_path`.

        Alternatively, one can specify query the Materials Project (MP) database for the bulk gap
        data, using `use_MP = True`, in which case the MP entry with the lowest number ID and
        composition matching the bulk will be used, or  the MP ID (mpid) of the bulk material to
        use can be specified. This is not recommended as it will correspond to a
        severely-underestimated GGA DFT bandgap!

        Args:
            bulk_bandgap_path (str): Path to bulk OUTCAR file for determining the band gap. If
                the VBM/CBM occur at reciprocal space points not included in the bulk supercell
                calculation, you should use this tag to point to a bulk bandstructure calculation
                instead. If None, will use self.defect_entry.calculation_metadata["bulk_path"].
            use_MP (bool): If True, will query the Materials Project database for the bulk gap
                data.
            mpid (str): If provided, will query the Materials Project database for the bulk gap
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

    def run_compatibility(self):
        """
        Get defect corrections and warn if likely to be inappropriate.
        """
        # Set potalign so pymatgen can calculate bandfilling for 'neutral' defects
        # (possible for resonant dopants etc.)
        if (
            self.defect_entry.charge_state == 0
            and "potalign" not in self.defect_entry.calculation_metadata
        ):
            self.defect_entry.calculation_metadata["potalign"] = 0

        self.defect_entry = self.compatibility.process_entry(self.defect_entry)

        if "delocalization_meta" in self.defect_entry.calculation_metadata:
            delocalization_meta = self.defect_entry.calculation_metadata["delocalization_meta"]
            if (
                "plnr_avg" in delocalization_meta and not delocalization_meta["plnr_avg"]["is_compatible"]
            ) or (
                "atomic_site" in delocalization_meta
                and not delocalization_meta["atomic_site"]["is_compatible"]
            ):
                specific_delocalized_warning = (
                    f"Delocalization analysis has indicated that {self.defect_entry.name} with "
                    f"charge {self.defect_entry.charge_state:+} may not be compatible with the chosen "
                    f"charge correction."
                )
                warnings.warn(message=specific_delocalized_warning)
                if not self._delocalization_warning_printed:
                    general_delocalization_warning = """
Note: Defects throwing a "delocalization analysis" warning may require a larger supercell for
accurate total energies. Recommended to look at the correction plots (i.e. run
`get_correction_freysoldt(DefectEntry,...,plot=True)` from
`doped.corrections`) to visually determine if the charge
correction scheme is still appropriate (replace 'freysoldt' with 'kumagai' if using anisotropic
correction). You can also change the DefectCompatibility() tolerance settings via the
`compatibility` parameter in `DefectParser.from_paths()`."""
                    warnings.warn(message=general_delocalization_warning)  # should only print once
                    DefectParser._delocalization_warning_printed = True  # don't print again

        if "num_hole_vbm" in self.defect_entry.calculation_metadata and (
            (self.compatibility.free_chg_cutoff < self.defect_entry.calculation_metadata["num_hole_vbm"])
            or (
                self.compatibility.free_chg_cutoff < self.defect_entry.calculation_metadata["num_elec_cbm"]
            )
        ):
            num_holes = self.defect_entry.calculation_metadata["num_hole_vbm"]
            num_electrons = self.defect_entry.calculation_metadata["num_elec_cbm"]
            warnings.warn(
                f"Eigenvalue analysis has determined that `num_hole_vbm` (= {num_holes}) or "
                f"`num_elec_cbm` (= {num_electrons}) is significant (>2.1) for "
                f"{self.defect_entry.name} with charge {self.defect_entry.charge_state}:+, "
                f"indicating that there are many free charges in this defect supercell "
                f"calculation and so the defect charge correction is unlikely to be accurate."
            )

        if "freysoldt_meta" in self.defect_entry.calculation_metadata:
            _update_defect_entry_charge_corrections(self.defect_entry, "freysoldt")
        elif "kumagai_meta" in self.defect_entry.calculation_metadata:
            _update_defect_entry_charge_corrections(self.defect_entry, "kumagai")
        if (
            self.defect_entry.charge_state != 0
            and (not self.defect_entry.corrections or sum(self.defect_entry.corrections.values())) == 0
        ):
            warnings.warn(
                f"No charge correction computed for {self.defect_entry.name} with "
                f"charge {self.defect_entry.charge_state:+}, indicating problems with the "
                f"required data for the charge correction (i.e. dielectric constant, "
                f"LOCPOT files for Freysoldt correction, OUTCAR (with ICORELEVEL = 0) "
                f"for Kumagai correction etc)."
            )


def _update_defect_entry_charge_corrections(defect_entry, charge_correction_type):
    meta = defect_entry.calculation_metadata[f"{charge_correction_type}_meta"]
    corr = (
        meta[f"{charge_correction_type}_electrostatic"]
        + meta[f"{charge_correction_type}_potential_alignment_correction"]
    )
    defect_entry.corrections.update({f"{charge_correction_type}_charge_correction": corr})
