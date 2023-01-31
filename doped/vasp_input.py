# coding: utf-8

"""
Code to generate VASP defect calculation input files.
"""

import functools
import os
from copy import deepcopy  # See https://stackoverflow.com/a/22341377/14020960 why
import warnings
from typing import TYPE_CHECKING
import numpy as np

from monty.io import zopen
from monty.serialization import dumpfn, loadfn
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.io.vasp.inputs import (
    incar_params,
    BadIncarWarning,
    Kpoints_supported_modes,
)
from pymatgen.io.vasp.sets import DictSet, BadInputSetWarning
from ase.dft.kpoints import monkhorst_pack

from doped.pycdt.utils.vasp import DefectRelaxSet, _check_psp_dir


if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "default_POTCARs.yaml"))


def scaled_ediff(natoms):  # 1e-5 for 50 atoms, up to max 1e-4
    ediff = float(f"{((natoms/50)*1e-5):.1g}")
    return ediff if ediff <= 1e-4 else 1e-4


def prepare_vasp_defect_inputs(defects: dict) -> dict:
    """
    Generates a dictionary of folders for VASP defect calculations
    Args:
        defects (dict):
            Dictionary of defect-object-dictionaries from PyCDT's
            ChargedDefectsStructures class (see example notebook)
    """
    defect_input_dict = {}
    comb_defs = functools.reduce(
        lambda x, y: x + y, [defects[key] for key in defects if key != "bulk"]
    )

    for defect in comb_defs:
        # noinspection DuplicatedCode
        for charge in defect["charges"]:
            supercell = defect["supercell"]
            dict_transf = {
                "defect_type": defect["name"],
                "defect_site": defect["unique_site"],
                "defect_supercell_site": defect["bulk_supercell_site"],
                "defect_multiplicity": defect["site_multiplicity"],
                "charge": charge,
                "supercell": supercell["size"],
            }
            if "substitution_specie" in defect:
                dict_transf["substitution_specie"] = defect["substitution_specie"]

            defect_relax_set = DefectRelaxSet(supercell["structure"], charge=charge)

            poscar = defect_relax_set.poscar
            struct = defect_relax_set.structure
            poscar.comment = (
                defect["name"]
                + str(dict_transf["defect_supercell_site"].frac_coords)
                + "_-dNELECT="  # change in NELECT from bulk supercell
                + str(charge)
            )
            folder_name = defect["name"] + f"_{charge}"
            print(folder_name)

            defect_input_dict[folder_name] = {
                "Defect Structure": struct,
                "POSCAR Comment": poscar.comment,
                "Transformation Dict": dict_transf,
            }
    return defect_input_dict


def prepare_vasp_defect_dict(
    defects: dict, write_files: bool = False, sub_folders: list = None
) -> dict:
    """
    Creates a transformation dictionary so we can tell PyCDT the
    initial defect site for post-processing analysis, in case it
    can't do it itself later on (common if multiple relaxations occur)
            Args:
                defects (dict):
                    Dictionary of defect-object-dictionaries from PyCDT's
                    ChargedDefectsStructures class (see example notebook)
                write_files (bool):
                    If True, write transformation.json files to
                    {defect_folder}/ or {defect_folder}/{*sub_folders}/
                    if sub_folders specified
                    (default: False)
                sub_folders (list):
                    List of sub-folders (in the defect folder) to write
                    the transformation.json file to
                    (default: None)
    """
    overall_dict = {}
    comb_defs = functools.reduce(
        lambda x, y: x + y, [defects[key] for key in defects if key != "bulk"]
    )

    for defect in comb_defs:
        # noinspection DuplicatedCode
        for charge in defect["charges"]:
            supercell = defect["supercell"]
            dict_transf = {
                "defect_type": defect["name"],
                "defect_site": defect["unique_site"],
                "defect_supercell_site": defect["bulk_supercell_site"],
                "defect_multiplicity": defect["site_multiplicity"],
                "charge": charge,
                "supercell": supercell["size"],
            }
            if "substitution_specie" in defect:
                dict_transf["substitution_specie"] = defect["substitution_specie"]
            folder_name = defect["name"] + f"_{charge}"
            overall_dict[folder_name] = dict_transf

    if write_files:
        if sub_folders:
            for key, val in overall_dict.items():
                for sub_folder in sub_folders:
                    if not os.path.exists(f"{key}/{sub_folder}/"):
                        os.makedirs(f"{key}/{sub_folder}/")
                    dumpfn(val, f"{key}/{sub_folder}/transformation.json")
        else:
            for key, val in overall_dict.items():
                if not os.path.exists(f"{key}/"):
                    os.makedirs(f"{key}/")
                dumpfn(val, f"{key}/transformation.json")
    return overall_dict


# noinspection DuplicatedCode
def vasp_gam_files(
    single_defect_dict: dict,
    input_dir: str = None,
    incar_settings: dict = None,
    potcar_settings: dict = None,
) -> None:
    """
    Generates input files for VASP Gamma-point-only rough relaxation
    (before more expensive vasp_std relaxation)
    Args:
        single_defect_dict (dict):
            Single defect-dictionary from prepare_vasp_defect_inputs()
            output dictionary of defect calculations (see example notebook)
        input_dir (str):
            Folder in which to create vasp_gam calculation inputs folder
            (Recommended to set as the key of the prepare_vasp_defect_inputs()
            output directory)
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).

            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """
    supercell = single_defect_dict["Defect Structure"]
    poscar_comment = (
        single_defect_dict["POSCAR Comment"]
        if "POSCAR Comment" in single_defect_dict
        else None
    )

    # Directory
    vaspgaminputdir = input_dir + "/vasp_gam/" if input_dir else "VASP_Files/vasp_gam/"
    if not os.path.exists(vaspgaminputdir):
        os.makedirs(vaspgaminputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types
    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))

    defect_relax_set = DefectRelaxSet(
        supercell,
        charge=single_defect_dict["Transformation " "Dict"]["charge"],
        user_potcar_settings=potcar_dict["POTCAR"],
        user_potcar_functional=potcar_dict["POTCAR_FUNCTIONAL"],
    )
    vaspgamposcar = defect_relax_set.poscar
    if poscar_comment:
        vaspgamposcar.comment = poscar_comment

    potcars = _check_psp_dir()
    if potcars:
        defect_relax_set.potcar.write_file(vaspgaminputdir + "POTCAR")
    else:  # make the folders without POTCARs
        warnings.warn(
            "POTCAR directory not set up with pymatgen, so only POSCAR files will be "
            "generated (POTCARs also needed to determine appropriate NELECT setting in "
            "INCAR files)"
        )
        vaspgamposcar.write_file(vaspgaminputdir + "POSCAR")
        return  # exit here

    relax_set_incar = defect_relax_set.incar
    try:
        # Only set if change in NELECT
        nelect = relax_set_incar.as_dict()["NELECT"]
    except KeyError:
        # Get NELECT if no change (-dNELECT = 0)
        nelect = defect_relax_set.nelect

    # Variable parameters first
    vaspgamincardict = {
        "# May need to change NELECT, IBRION, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, ISPIN, "
        "POTIM": "variable parameters",
        "NELECT": nelect,
        "IBRION": "2 # While often slower than '1' (RMM-DIIS), this is more stable and "
        "reliable, and vasp_gam relaxations are typically cheap enough to justify it",
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if strong spin polarisation or magnetic behaviour present",
        "ISPIN": "2 # Spin polarisation likely for defects",
        "NCORE": 12,
        "#KPAR": "# No KPAR, only one kpoint here",
        "AEXX": 0.25,
        "ENCUT": 400,
        "POTIM": 0.2,
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "LSUBROT": "False # Change to True if relaxation poorly convergent",
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "EDIFF": f"{scaled_ediff(supercell.num_sites)} # May need to reduce for tricky relaxations",
        "EDIFFG": -0.01,
        "HFSCREEN": 0.2,  # assuming HSE06
        "ISIF": 2,
        "ISYM": "0 # Symmetry breaking extremely likely for defects",
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 300,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }
    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspgamincardict.update(incar_settings)

    vaspgamkpts = Kpoints().from_dict(
        {"comment": "Kpoints from doped.vasp_gam_files", "generation_style": "Gamma"}
    )
    vaspgamincar = Incar.from_dict(vaspgamincardict)
    vaspgamincar.write_file(vaspgaminputdir + "INCAR")

    vaspgamposcar.write_file(vaspgaminputdir + "POSCAR")
    vaspgamkpts.write_file(vaspgaminputdir + "KPOINTS")


def vasp_std_files(
    single_defect_dict: dict,
    input_dir: str = None,
    incar_settings: dict = None,
    kpoints_settings: dict = None,
    potcar_settings: dict = None,
    unperturbed_poscar: bool = False,
) -> None:
    """
    Generates INCAR, POTCAR and KPOINTS for vasp_std expensive k-point mesh relaxation.
    For POSCAR, use on command-line (to continue on from vasp_gam run):
    'cp vasp_gam/CONTCAR vasp_std/POSCAR; cp vasp_gam/CHGCAR vasp_std/'
    Args:
        single_defect_dict (dict):
            Single defect-dictionary from prepare_vasp_defect_inputs()
            output dictionary of defect calculations (see example notebook)
        input_dir (str):
            Folder in which to create vasp_std calculation inputs folder
            (Recommended to set as the key of the prepare_vasp_defect_inputs()
            output directory)
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        kpoints_settings (dict):
            Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
            options would be "generation_style": "Monkhorst" (rather than "Gamma"),
            and/or "kpoints": [[3, 3, 1]] etc.
            Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
        unperturbed_poscar (bool):
            If True, write the unperturbed defect POSCAR to the vasp_std folder as well. Not
            recommended, as the recommended workflow is to initially perform vasp_gam
            ground-state structure searching using ShakeNBreak (see example notebook;
            https://shakenbreak.readthedocs.io), then continue the vasp_std relaxations from the
            'Groundstate' CONTCARs.
            (default: False)
    """
    supercell = single_defect_dict["Defect Structure"]
    poscar_comment = (
        single_defect_dict["POSCAR Comment"]
        if "POSCAR Comment" in single_defect_dict
        else None
    )

    # Directory
    vaspstdinputdir = input_dir + "/vasp_std/" if input_dir else "VASP_Files/vasp_std/"
    if not os.path.exists(vaspstdinputdir):
        os.makedirs(vaspstdinputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

    # POTCAR
    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))

    defect_relax_set = DefectRelaxSet(
        supercell,
        charge=single_defect_dict["Transformation Dict"]["charge"],
        user_potcar_settings=potcar_dict["POTCAR"],
        user_potcar_functional=potcar_dict["POTCAR_FUNCTIONAL"],
    )
    vaspstdposcar = defect_relax_set.poscar
    if poscar_comment:
        vaspstdposcar.comment = poscar_comment

    potcars = _check_psp_dir()
    if not potcars:
        if unperturbed_poscar:
            warnings.warn(
                "POTCAR directory not set up with pymatgen, so only POSCAR files will be "
                "generated (POTCARs also needed to determine appropriate NELECT setting "
                "in INCAR files)"
            )
            vaspstdposcar.write_file(vaspstdinputdir + "POSCAR")

        else:
            warnings.warn(
                "POTCAR directory not set up with pymatgen, so no input files will be "
                "generated (you should use vasp_input.vasp_gam_files() to create the "
                "initial relaxation files, then continue from this pre-converged "
                "structure with vasp_std)"
            )
        return  # exit here
    defect_relax_set.potcar.write_file(vaspstdinputdir + "POTCAR")

    if unperturbed_poscar:
        vaspstdposcar.write_file(vaspstdinputdir + "POSCAR")

    relax_set_incar = defect_relax_set.incar
    try:
        # Only set if change in NELECT
        nelect = relax_set_incar.as_dict()["NELECT"]
    except KeyError:
        # Get NELECT if no change (-dNELECT = 0)
        nelect = defect_relax_set.nelect

    # Variable parameters first
    vaspstdincardict = {
        "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, "
        + "ISPIN, POTIM": "variable parameters",
        "NELECT": nelect,
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if strong spin polarisation or magnetic behaviour present",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 400,
        "POTIM": 0.2,
        "ISPIN": "2 # Spin polarisation likely for defects",
        "LSUBROT": "False # Change to True if relaxation poorly convergent",
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "EDIFF": f"{scaled_ediff(supercell.num_sites)} # May need to reduce for tricky relaxations",
        "EDIFFG": -0.01,
        "HFSCREEN": 0.2,  # assuming HSE06
        "IBRION": "1 # May need to change to 2 for difficult/poorly-convergent relaxations",
        "ISIF": 2,
        "ISYM": "0 # Symmetry breaking extremely likely for defects",
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 200,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }
    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspstdincardict.update(incar_settings)

    vaspstdkpointsdict = {
        "comment": "Kpoints from doped.vasp_std_files",
        "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
        "kpoints": [[2, 2, 2]],
    }
    if kpoints_settings:
        vaspstdkpointsdict.update(kpoints_settings)
    vaspstdkpts = Kpoints.from_dict(vaspstdkpointsdict)
    vaspstdkpts.write_file(vaspstdinputdir + "KPOINTS")

    vaspstdincar = Incar.from_dict(vaspstdincardict)
    vaspstdincar.write_file(vaspstdinputdir + "INCAR")


def vasp_ncl_files(
    single_defect_dict: dict,
    input_dir: str = None,
    incar_settings: dict = None,
    kpoints_settings: dict = None,
    potcar_settings: dict = None,
    unperturbed_poscar: bool = False,
) -> None:
    """
    Generates INCAR, POTCAR and non-symmetrised KPOINTS for vasp_ncl single-shot SOC energy
    calculation on vasp_std-relaxed defect structure.
    For POSCAR, use on command-line (to continue on from vasp_std run):
    'cp vasp_std/CONTCAR vasp_ncl/POSCAR'
    Args:
        single_defect_dict (dict):
            Single defect-dictionary from prepare_vasp_defect_inputs()
            output dictionary of defect calculations (see example notebook)
        input_dir (str):
            Folder in which to create vasp_ncl calculation inputs folder
            (Recommended to set as the key of the prepare_vasp_defect_inputs()
            output directory)
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        kpoints_settings (dict):
            Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Most
            common option would be {"kpoints": [[3, 3, 1]]} etc.
            Will generate a non-symmetrised (i.e. explicitly listed) KPOINTS, as typically required
            for vasp_ncl calculations. Default is Gamma-centred 2 x 2 x 2 mesh.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
        unperturbed_poscar (bool):
            If True, write the unperturbed defect POSCAR to the vasp_ncl folder as well. Not
            recommended, as the recommended workflow is to initially perform vasp_gam
            ground-state structure searching using ShakeNBreak (see example notebook;
            https://shakenbreak.readthedocs.io), then continue the vasp_std relaxations from the
            'Groundstate' CONTCARs, before doing final vasp_ncl singleshot calculations if SOC is
            important.
            (default: False)
    """
    supercell = single_defect_dict["Defect Structure"]
    poscar_comment = (
        single_defect_dict["POSCAR Comment"]
        if "POSCAR Comment" in single_defect_dict
        else None
    )

    # Directory
    vaspnclinputdir = input_dir + "/vasp_ncl/" if input_dir else "VASP_Files/vasp_ncl/"
    if not os.path.exists(vaspnclinputdir):
        os.makedirs(vaspnclinputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    defect_relax_set = DefectRelaxSet(
        supercell,
        charge=single_defect_dict["Transformation Dict"]["charge"],
        user_potcar_settings=potcar_dict["POTCAR"],
        user_potcar_functional=potcar_dict["POTCAR_FUNCTIONAL"],
    )
    vaspnclposcar = defect_relax_set.poscar
    if poscar_comment:
        vaspnclposcar.comment = poscar_comment

    potcars = _check_psp_dir()
    if not potcars:
        if unperturbed_poscar:
            warnings.warn(
                "POTCAR directory not set up with pymatgen, so only POSCAR files will be "
                "generated (POTCARs also needed to determine appropriate NELECT setting "
                "in INCAR files)"
            )
            vaspnclposcar.write_file(vaspnclinputdir + "POSCAR")

        else:
            warnings.warn(
                "POTCAR directory not set up with pymatgen, so no input files will be "
                "generated (you should use vasp_input.vasp_gam_files() to create the "
                "initial relaxation files, then continue from this pre-converged "
                "structure with vasp_std and finally vasp_ncl if SOC important)"
            )
        return  # exit here

    defect_relax_set.potcar.write_file(vaspnclinputdir + "POTCAR")
    if unperturbed_poscar:
        vaspnclposcar.write_file(vaspnclinputdir + "POSCAR")

    relax_set_incar = defect_relax_set.incar
    try:
        # Only set if change in NELECT
        nelect = relax_set_incar.as_dict()["NELECT"]
    except KeyError:
        # Get NELECT if no change (-dNELECT = 0)
        nelect = defect_relax_set.nelect

    # Variable parameters first
    vaspnclincardict = {
        "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN": "variable parameters",
        "NELECT": nelect,
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if strong spin polarisation or magnetic behaviour present",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 400,
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "NSW": 0,
        "LSORBIT": True,
        "EDIFF": 1e-06,  # tight for final energy and converged DOS
        "EDIFFG": -0.01,
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "HFSCREEN": 0.2,  # assuming HSE06
        "IBRION": -1,
        "ISIF": 2,
        "ISYM": 0,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }
    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspnclincardict.update(incar_settings)

    k_grid = (
        kpoints_settings.pop("kpoints")[0]
        if (kpoints_settings and "kpoints" in kpoints_settings)
        else [2, 2, 2]
    )
    shift = (
        np.array(kpoints_settings.pop("usershift"))
        if (kpoints_settings and "usershift" in kpoints_settings)
        else np.array([0, 0, 0])
    )
    vasp_ncl_kpt_array = monkhorst_pack(k_grid)
    vasp_ncl_kpt_array += -vasp_ncl_kpt_array[0] + shift  # Gamma-centred by default
    vasp_ncl_kpts = Kpoints(
        comment="Non-symmetrised KPOINTS for vasp_ncl, from doped",
        num_kpts=len(vasp_ncl_kpt_array),
        style=Kpoints_supported_modes(5),  # Reciprocal
        kpts=vasp_ncl_kpt_array,
        kpts_weights=[
            1,
        ]
        * len(vasp_ncl_kpt_array),
    )
    if kpoints_settings:
        modified_kpts_dict = vasp_ncl_kpts.as_dict()
        modified_kpts_dict.update(kpoints_settings)
        vasp_ncl_kpts = Kpoints.from_dict(modified_kpts_dict)
    vasp_ncl_kpts.write_file(vaspnclinputdir + "KPOINTS")

    vaspnclincar = Incar.from_dict(vaspnclincardict)
    vaspnclincar.write_file(vaspnclinputdir + "INCAR")


def is_metal(element: "pymatgen.core.periodic_table.Element") -> bool:
    """
    Checks if the input element is metallic
    Args:
        element (Pymatgen Element object):
            Element to check metallicity
    """

    return (
        element.is_transition_metal
        or element.is_post_transition_metal
        or element.is_alkali
        or element.is_alkaline
        or element.is_rare_earth_metal
    )


# noinspection DuplicatedCode
def vasp_converge_files(
    structure: "pymatgen.core.Structure",
    input_dir: str = None,
    incar_settings: dict = None,
    potcar_settings: dict = None,
    config: str = None,
) -> None:
    """
    Generates input files for single-shot GGA convergence test calculations.
    Automatically sets ISMEAR (in INCAR) to 2 (if metallic) or 0 if not.
    Recommended to use with vaspup2.0
    Args:
        structure (Structure object):
            Structure to create input files for.
        input_dir (str):
            Folder in which to create 'input' folder with VASP input files.
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        config (str):
            CONFIG file string. If provided, will also write the CONFIG file (to automate
            convergence tests with vaspup2.0) to each 'input' directory.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """

    # Variable parameters first
    vaspconvergeincardict = {
        "# May need to change ISMEAR, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, "
        + "ISPIN": "variable parameters",
        "NUPDOWN": "0 # But could be >0 if magnetic behaviour present",
        "NCORE": 12,
        "#KPAR": 1,
        "ENCUT": 400,
        "ISMEAR": "0 # Non-metal, use Gaussian smearing",
        "ISPIN": "1 # Change to 2 if spin polarisation or magnetic behaviour present",
        "GGA": "PS",  # PBEsol
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "EDIFF": 1e-06,
        "EDIFFG": -0.01,
        "IBRION": -1,
        "ISIF": 3,
        "LASPH": True,
        "LORBIT": 11,
        "LREAL": False,
        "LWAVE": "False # Save filespace, shouldn't need WAVECAR from convergence tests",
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 0,
        "PREC": "Accurate",
        "SIGMA": 0.2,
    }
    if all(is_metal(element) for element in structure.composition.elements):
        vaspconvergeincardict[
            "ISMEAR"
        ] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if incar_settings:
        for (
            k
        ) in incar_settings.keys():  # check INCAR flags and warn if they don't exist (
            # typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspconvergeincardict.update(incar_settings)

    # Directory
    vaspconvergeinputdir = input_dir + "/input/" if input_dir else "VASP_Files/input/"
    if not os.path.exists(vaspconvergeinputdir):
        os.makedirs(vaspconvergeinputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

    # POTCAR
    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    vaspconvergeinput = DictSet(structure, config_dict=potcar_dict)
    vaspconvergeinput.potcar.write_file(vaspconvergeinputdir + "POTCAR")

    vaspconvergekpts = Kpoints().from_dict(
        {"comment": "Kpoints from vasp_gam_files", "generation_style": "Gamma"}
    )
    vaspconvergeincar = Incar.from_dict(vaspconvergeincardict)
    vaspconvergeincar.write_file(vaspconvergeinputdir + "INCAR")

    vaspconvergeposcar = Poscar(structure)
    vaspconvergeposcar.write_file(vaspconvergeinputdir + "POSCAR")

    vaspconvergekpts.write_file(vaspconvergeinputdir + "KPOINTS")
    # generate CONFIG file
    if config:
        with open(vaspconvergeinputdir + "CONFIG", "w+") as config_file:
            config_file.write(config)
        with open(vaspconvergeinputdir + "CONFIG", "a") as config_file:
            config_file.write(f"""\nname="{input_dir[13:]}" # input_dir""")


# Input files for vasp_std


def vasp_std_chempot(
    structure: "pymatgen.core.Structure",
    input_dir: str = None,
    incar_settings: dict = None,
    kpoints_settings: dict = None,
    potcar_settings: dict = None,
) -> None:
    """
    Generates POSCAR, INCAR, POTCAR and KPOINTS for vasp_std chemical potentials relaxation.:
    Args:
        structure (Structure object):
            Structure to create input files for.
        input_dir (str):
            Folder in which to create vasp_std calculation inputs folder
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        kpoints_settings (dict):
            Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
            options would be "generation_style": "Monkhorst" (rather than "Gamma"),
            and/or "kpoints": [[3, 3, 1]] etc.
            Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """
    # INCAR Parameters
    vaspstdincardict = {
        "# May need to change NCORE, KPAR, ENCUT"
        + "ISPIN, POTIM": "variable parameters",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 400,
        "POTIM": 0.2,
        "LSUBROT": "False # Change to True if relaxation poorly convergent",
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "EDIFF": 1e-06,  # May need to reduce for tricky relaxations",
        "EDIFFG": -0.01,
        "HFSCREEN": 0.2,  # assuming HSE06
        "IBRION": "1 # May need to change to 2 for difficult/poorly-convergent relaxations",
        "ISIF": 3,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 200,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }

    # Directory
    vaspstdinputdir = input_dir + "/vasp_std/" if input_dir else "VASP_Files/vasp_std/"
    if not os.path.exists(vaspstdinputdir):
        os.makedirs(vaspstdinputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

    # POTCAR
    potcar_dict = default_potcar_dict
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    vaspstdinput = DictSet(structure, config_dict=potcar_dict)
    vaspstdinput.potcar.write_file(vaspstdinputdir + "POTCAR")

    if all(is_metal(element) for element in structure.composition.elements):
        vaspstdincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if all(is_metal(element) for element in structure.composition.elements):
        vaspstdincardict["SIGMA"] = 0.02

    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspstdincardict.update(incar_settings)

    # POSCAR
    vaspstdposcar = Poscar(structure)
    vaspstdposcar.write_file(vaspstdinputdir + "POSCAR")

    # KPOINTS
    vaspstdkpointsdict = {
        "comment": "Kpoints from doped.vasp_std_files",
        "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
        "kpoints": [[2, 2, 2]],
    }
    if kpoints_settings:
        vaspstdkpointsdict.update(kpoints_settings)
    vaspstdkpts = Kpoints.from_dict(vaspstdkpointsdict)
    vaspstdkpts.write_file(vaspstdinputdir + "KPOINTS")

    # INCAR
    vaspstdincar = Incar.from_dict(vaspstdincardict)
    with zopen(vaspstdinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspstdincar.get_string())


# Input files for vasp_ncl


def vasp_ncl_chempot(
    structure: "pymatgen.core.Structure",
    input_dir: str = None,
    incar_settings: dict = None,
    kpoints_settings: dict = None,
    potcar_settings: dict = None,
) -> None:
    """
    Generates INCAR, POTCAR and KPOINTS for vasp_ncl chemical potentials relaxation.
    Take CONTCAR from vasp_std for POSCAR.:
    Args:
        structure (Structure object):
            Structure to create input files for.
        input_dir (str):
            Folder in which to create vasp_ncl calculation inputs folder
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        kpoints_settings (dict):
            Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
            options would be "generation_style": "Monkhorst" (rather than "Gamma"),
            and/or "kpoints": [[3, 3, 1]] etc.
            Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """
    # INCAR Parameters
    vaspnclincardict = {
        "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN": "variable parameters",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 400,
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "NSW": 0,
        "LSORBIT": True,
        "EDIFF": 1e-06,  # tight for final energy and converged DOS
        "EDIFFG": -0.01,
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "HFSCREEN": 0.2,
        "IBRION": -1,
        "ISYM": 0,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }

    # Directory
    vaspnclinputdir = input_dir + "/vasp_ncl/" if input_dir else "VASP_Files/vasp_ncl/"
    if not os.path.exists(vaspnclinputdir):
        os.makedirs(vaspnclinputdir)

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

    # POTCAR
    potcar_dict = default_potcar_dict
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    vaspnclinput = DictSet(structure, config_dict=potcar_dict)
    vaspnclinput.potcar.write_file(vaspnclinputdir + "POTCAR")

    if all(is_metal(element) for element in structure.composition.elements):
        vaspnclincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if all(is_metal(element) for element in structure.composition.elements):
        vaspnclincardict["SIGMA"] = 0.02

    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspnclincardict.update(incar_settings)

    # KPOINTS
    vaspnclkpointsdict = {
        "comment": "Kpoints from doped.vasp_ncl_files",
        "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
        "kpoints": [[2, 2, 2]],
    }
    if kpoints_settings:
        vaspnclkpointsdict.update(kpoints_settings)
    vaspnclkpts = Kpoints.from_dict(vaspnclkpointsdict)
    vaspnclkpts.write_file(vaspnclinputdir + "KPOINTS")

    # INCAR
    vaspnclincar = Incar.from_dict(vaspnclincardict)
    with zopen(vaspnclinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspnclincar.get_string())
