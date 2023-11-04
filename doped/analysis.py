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
from pymatgen.analysis.defects import core
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.sites import PeriodicSite
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar

from doped import _ignore_pmg_warnings
from doped.core import DefectEntry
from doped.generation import get_defect_name_from_entry
from doped.plotting import _format_defect_name
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
            f"{get_defect_name_from_entry(defect_entry, unrelaxed=True)}_"
            f"{'+' if charge_state > 0 else ''}{charge_state}"
        )  # otherwise use default doped name  # TODO: Test!
        # Note this can determine the wrong point group symmetry if a non-diagonal supercell expansion
        # was used


# TODO: Should add check the bulk and defect KPOINTS/INCAR/POTCAR/POSCAR (size) settings
#  are compatible, and throw warning if not.
# TODO: Add `check_defects_compatibility()` function that checks the bulk and defect
#  KPOINTS/INCAR/POTCAR/POSCAR (size) settings for all defects in the supplied defect_dict
#  are compatible, if not throw warnings and say what the differences are. Should recommend
#  using this in the example notebook if a user has parsed the defects individually (rather
#  than with the single looping function described below). Also check that the same type of
#  correction was used in each case (FNV vs eFNV). If isotropic, shouldn't really matter, but
#  worth warning user as best to be consistent, and could give some unexpected behaviour
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

    Note that the bulk and defect supercells should have the same definitions/basis
    sets (for site-matching and finite-size charge corrections to work appropriately).

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
            calculation instead. Alternatively, you can edit/add the "gap" and "vbm"
            entries in self.defect_entry.calculation_metadata to match the correct
            (eigen)values.
    #  and defect
            If None, will use self.defect_entry.calculation_metadata["bulk_path"].
        **kwargs:
            Additional keyword arguments to pass to `DefectParser()` (such as
             `error_tolerance`). See `DefectParser()` docstring for more details.

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

    # get defect charge
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

    # Add defect structure to calculation_metadata, so it can be pulled later on (eg. for Kumagai loader)
    defect_structure = defect_vr.final_structure.copy()
    calculation_metadata["defect_structure"] = defect_structure

    # identify defect site, structural information, and create defect object:
    # Can specify initial defect structure (to help find the defect site if we have a very very
    # distorted final structure), but regardless try using the final structure (from defect OUTCAR) first:
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
        if initial_defect_structure:
            defect_structure_for_ID = Poscar.from_file(initial_defect_structure).structure.copy()
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
        charge_state=charge_state,
        sc_entry=defect_vr.get_computed_entry(),
        sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
        bulk_entry=bulk_vr.get_computed_entry(),
        # doped attributes:
        defect_supercell_site=defect_site,  # _relaxed_ defect site
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
        **kwargs,
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

        # regardless, try parsing OUTCAR files first (quickest, more robust for cases where defect
        # charge is localised somewhat off the (auto-determined) defect site (e.g. split-interstitials
        # etc) and also works regardless of isotropic/anisotropic)
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
            dp.apply_corrections()

            # check that charge corrections are not negative
            summed_corrections = sum(
                val
                for key, val in dp.defect_entry.corrections.items()
                if any(i in key.lower() for i in ["freysoldt", "kumagai", "fnv", "charge"])
            )
            if summed_corrections < -0.05:
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
    elt_refs: Optional[Dict] = None,
    facets: Optional[List] = None,
    fermi_level: float = 0,
):
    """
    Generates defect formation energy tables (DataFrames) for either a
    single chemical potential limit (i.e. phase diagram facet) or each
    facet in the phase diagram (chempots dict), depending on the chempots
    input supplied. This can either be a dictionary of chosen absolute/DFT
    chemical potentials: {Elt: Energy} (giving a single formation energy
    table) or a dictionary including the key-value pair: {"facets":
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
            chemical potentials this way, you can set the elt_refs option with
            the DFT reference energies of the elemental phases in order to show
            the formal (relative) chemical potentials as well.
            (Default: None)
        facets (list, str):
            A string or list of facet(s) (chemical potential limit(s)) for which
            to tabulate the defect formation energies, corresponding to 'facet' in
            {"facets": [{'facet': [chempot_dict]}]} (the format generated by
            doped's chemical potential parsing functions (see tutorials)). If
            not specified, will tabulate for each facet in `chempots`. (Default: None)
        elt_refs (dict):
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

    if "facets_wrt_elt_refs" in chempots:
        list_of_dfs = []
        if facets is None:
            facets = chempots["facets"].keys()  # Phase diagram facets to use for chemical
            # potentials, to tabulate formation energies
        for facet in facets:
            single_formation_energy_df = _single_formation_energy_table(
                defect_phase_diagram,
                chempots=chempots["facets_wrt_elt_refs"][facet],
                elt_refs=chempots["elemental_refs"],
                fermi_level=fermi_level,
            )
            list_of_dfs.append(single_formation_energy_df)

        return list_of_dfs[0] if len(list_of_dfs) == 1 else list_of_dfs

    # else return {Elt: Energy} dict for chempot_limits, or if unspecified, all zero energy
    single_formation_energy_df = _single_formation_energy_table(
        defect_phase_diagram,
        chempots=chempots,
        elt_refs={elt: 0 for elt in chempots} if elt_refs is None else elt_refs,
        fermi_level=fermi_level,
    )
    return single_formation_energy_df


def _single_formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempots: Dict,
    elt_refs: Dict,
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
            Dictionary of chosen absolute/DFT chemical potentials: {Elt: Energy}.
            If not specified, chemical potentials are not included in the
            formation energy calculation (all set to zero energy).
        elt_refs (dict):
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
        row += [defect_phase_diagram._get_chempot_term(defect_entry, elt_refs)]
        row += [defect_phase_diagram._get_chempot_term(defect_entry, chempots)]
        row += [sum(defect_entry.corrections.values())]
        dft_chempots = {elt: energy + elt_refs[elt] for elt, energy in chempots.items()}
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


class DefectParser:
    # TODO: Load bulk locpot once when looping through defects for expedited FNV parsing
    # TODO: Test that charge correction is not attempted by default when charge state is zero
    # TODO: Add comment/note somewhere that the supercells should have equal definitions for both bulk
    #  and defect

    def __init__(
        self,
        defect_entry,
        defect_vr=None,
        bulk_vr=None,
        error_tolerance: float = 0.05,
    ):
        """
        Parse a single Defect object.

        Args:
            defect_entry (DefectEntry):
                doped DefectEntry
            defect_vr (Vasprun):
                pymatgen Vasprun object for the defect supercell calculation
            bulk_vr (Vasprun):
                pymatgen Vasprun object for the reference bulk supercell calculation
            error_tolerance (float):
                If the estimated error in the charge correction is greater than
                this value (in eV), then a warning is raised. (default: 0.05 eV)
        """
        self.defect_entry: DefectEntry = defect_entry
        self.defect_vr = defect_vr
        self.bulk_vr = bulk_vr
        self.error_tolerance = error_tolerance

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
        # TODO: Update this docstring? (Using `defect_entry_from_paths`) Depending on recommended parsing
        #  workflow
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

    def freysoldt_loader(self, bulk_locpot_dict=None):
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

        if not bulk_locpot_dict:
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

    def kumagai_loader(self, bulk_site_potentials=None):
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

        if not bulk_site_potentials:
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

        if bulk_outcar.electrostatic_potential is None:
            _raise_incomplete_outcar_error(bulk_outcar_path, dir_type="bulk")

        if defect_outcar.electrostatic_potential is None:
            _raise_incomplete_outcar_error(defect_outcar_path, dir_type="defect")

        bulk_site_potentials = -1 * np.array(bulk_outcar.electrostatic_potential)
        defect_site_potentials = -1 * np.array(defect_outcar.electrostatic_potential)

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_site_potentials": bulk_site_potentials,
                "defect_site_potentials": defect_site_potentials,
            }
        )

        return bulk_site_potentials

    def get_stdrd_metadata(self):
        """
        Get standard defect calculation metadata.
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

        run_metadata = {  # TODO: Add check that incars, kpoints and potcars are compatible here
            # incars need to be as dict without module keys otherwise not JSONable:
            "defect_incar": {k: v for k, v in self.defect_vr.incar.as_dict().items() if "@" not in k},
            "bulk_incar": {k: v for k, v in self.bulk_vr.incar.as_dict().items() if "@" not in k},
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
                "are loaded with either the kumagai_loader() or freysoldt_loader() methods for "
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
