# coding: utf-8

"""
Code to generate VASP defect calculation input files.
"""

__author__ = "Seán Kavanagh"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Seán Kavanagh"
__email__ = "sean.kavanagh.19@ucl.ac.uk"
__date__ = "May 19, 2020"

import functools
import os
from typing import TYPE_CHECKING

from monty.io import zopen
from monty.serialization import dumpfn
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.io.vasp.sets import DictSet

from DefectsWithTheBoys.pycdt.utils.vasp import DefectRelaxSet


if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure


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

            # try: # Actually, we don't need this code block
            #     potcar = defect_relax_set.potcar
            # except AttributeError as error:
            #     print(
            #         f"""Error {error} occurred with arguments {error.args},
            #     meaning no POTCAR generated, have you set the PseudoPotential directory
            #     in your .pmgrc.yaml file? (See https://bitbucket.org/mbkumar/pycdt).
            #     """
            #     )
            #     break

            incar = defect_relax_set.incar
            poscar = defect_relax_set.poscar
            struct = defect_relax_set.structure
            poscar.comment = (
                defect["name"]
                + str(dict_transf["defect_supercell_site"].frac_coords)
                + "_KV=-dNELECT="
                + str(charge)
            )
            folder_name = defect["name"] + f"_{charge}"
            print(folder_name)
            try:
                # Only set if change in NELECT
                nelect = incar.as_dict()["NELECT"]
            except KeyError:
                # Get NELECT if no change (KV = -dNELECT = 0)
                nelect = defect_relax_set.nelect
            defect_input_dict[folder_name] = {
                "Defect Structure": struct,
                "NELECT": nelect,
                "POSCAR Comment": poscar.comment,
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
    single_defect_dict: dict, input_dir: str = None, incar_settings: dict = None
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
            Highly recommended to look at output INCARs or DefectsWithTheBoys.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
    """
    supercell = single_defect_dict["Defect Structure"]
    nelect = single_defect_dict["NELECT"]
    poscar_comment = (
        single_defect_dict["POSCAR Comment"] if single_defect_dict["POSCAR Comment"] else None
    )

    # Variable parameters first
    vaspgamincardict = {
        "# May need to change NELECT, IBRION, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, ISPIN, "
        "POTIM": "variable parameters",
        "NELECT": nelect,
        "IBRION": "2 # vasp_gam cheap enough, this is more reliable",
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} if ya think we got a "
        "bit of crazy ferromagnetic shit going down",
        "NCORE": 12,
        "#KPAR": "One pal, only one k-point yeh",
        "AEXX": 0.25,
        "ENCUT": 450,
        "POTIM": 0.2,
        "ISPIN": 2,
        "ICORELEVEL": 0,
        "LSUBROT": True,
        "ALGO": "All",
        "ADDGRID": True,
        "EDIFF": 1e-06,
        "EDIFFG": "-0.005 # Tight relax criteria for vasp_gam, it's cheap enough buddy",
        "HFSCREEN": 0.2,
        "ICHARG": 1,
        "ISIF": 2,
        "ISYM": 0,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": True,
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 300,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }
    if incar_settings:
        vaspgamincardict.update(incar_settings)

    # Directory
    vaspgaminputdir = input_dir + "/vasp_gam/" if input_dir else "VASP_Files/vasp_gam/"
    if not os.path.exists(vaspgaminputdir):
        os.makedirs(vaspgaminputdir)

    vasppotcardict = {
        "POTCAR_FUNCTIONAL": "PBE_54",  # May need to change this if you've got
        # different POTCARs
        "POTCAR": {
            "Ac": "Ac",
            "Ag": "Ag",
            "Al": "Al",
            "Ar": "Ar",
            "As": "As",
            "Au": "Au",
            "B": "B",
            "Ba": "Ba_sv",
            "Be": "Be_sv",
            "Bi": "Bi",
            "Br": "Br",
            "C": "C",
            "Ca": "Ca_sv",
            "Cd": "Cd",
            "Ce": "Ce",
            "Cl": "Cl",
            "Co": "Co",
            "Cr": "Cr_pv",
            "Cs": "Cs_sv",
            "Cu": "Cu_pv",
            "Dy": "Dy_3",
            "Er": "Er_3",
            "Eu": "Eu",
            "F": "F",
            "Fe": "Fe_pv",
            "Ga": "Ga_d",
            "Gd": "Gd",
            "Ge": "Ge_d",
            "H": "H",
            "He": "He",
            "Hf": "Hf_pv",
            "Hg": "Hg",
            "Ho": "Ho_3",
            "I": "I",
            "In": "In_d",
            "Ir": "Ir",
            "K": "K_sv",
            "Kr": "Kr",
            "La": "La",
            "Li": "Li_sv",
            "Lu": "Lu_3",
            "Mg": "Mg_pv",
            "Mn": "Mn_pv",
            "Mo": "Mo_pv",
            "N": "N",
            "Na": "Na_pv",
            "Nb": "Nb_pv",
            "Nd": "Nd_3",
            "Ne": "Ne",
            "Ni": "Ni_pv",
            "Np": "Np",
            "O": "O",
            "Os": "Os_pv",
            "P": "P",
            "Pa": "Pa",
            "Pb": "Pb_d",
            "Pd": "Pd",
            "Pm": "Pm_3",
            "Pr": "Pr_3",
            "Pt": "Pt",
            "Pu": "Pu",
            "Rb": "Rb_sv",
            "Re": "Re_pv",
            "Rh": "Rh_pv",
            "Ru": "Ru_pv",
            "S": "S",
            "Sb": "Sb",
            "Sc": "Sc_sv",
            "Se": "Se",
            "Si": "Si",
            "Sm": "Sm_3",
            "Sn": "Sn_d",
            "Sr": "Sr_sv",
            "Ta": "Ta_pv",
            "Tb": "Tb_3",
            "Tc": "Tc_pv",
            "Te": "Te",
            "Th": "Th",
            "Ti": "Ti_pv",
            "Tl": "Tl_d",
            "Tm": "Tm_3",
            "U": "U",
            "V": "V_pv",
            "W": "W_pv",
            "Xe": "Xe",
            "Y": "Y_sv",
            "Yb": "Yb_2",
            "Zn": "Zn",
            "Zr": "Zr_sv",
        },
    }
    vaspgamkpts = Kpoints().from_dict(
        {"comment": "Kpoints from DefectsWithTheBoys.vasp_gam_files", "generation_style": "Gamma"}
    )
    vaspgamincar = Incar.from_dict(vaspgamincardict)
    vaspgaminput = DictSet(supercell, config_dict=vasppotcardict)
    vaspgamposcar = vaspgaminput.poscar
    vaspgaminput.potcar.write_file(vaspgaminputdir + "POTCAR")
    if poscar_comment:
        vaspgamposcar.comment = poscar_comment
    vaspgamposcar.write_file(vaspgaminputdir + "POSCAR")
    with zopen(vaspgaminputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspgamincar.get_string())
    vaspgamkpts.write_file(vaspgaminputdir + "KPOINTS")


def vasp_std_files(
    single_defect_dict: dict, input_dir: str = None, incar_settings: dict = None
) -> None:
    """
    Generates INCAR and KPOINTS for vasp_std expensive k-point mesh relaxation.
    For POSCAR and POTCAR, use on command-line (to continue on from vasp_gam run):
    'cp vasp_gam/CONTCAR vasp_std/POSCAR; cp vasp_gam/{POTCAR,CHGCAR} vasp_std/'
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
            Highly recommended to look at output INCARs or DefectsWithTheBoys.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
    """
    nelect = single_defect_dict["NELECT"]

    # Variable parameters first
    vaspstdincardict = {
        "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, "
        + "ISPIN, POTIM": "variable parameters",
        "NELECT": nelect,
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if ya think we got a bit of crazy ferromagnetic shit going down",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 450,
        "POTIM": 0.2,
        "ISPIN": "2 # Check mag in OSZICAR though, if 0 don't bother with spin polarisation ("
        "change to 1",
        "LSUBROT": True,
        "ICORELEVEL": "0 # Get core potentials in OUTCAR for Kumagai corrections",
        "ALGO": "All",
        "ADDGRID": True,
        "EDIFF": 1e-05,
        "EDIFFG": -0.01,
        "HFSCREEN": 0.2,
        "IBRION": "1 # May need to change to 2 for difficult relaxations though",
        "ICHARG": 1,
        "ISIF": 2,
        "ISYM": 0,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": True,
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 200,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }
    if incar_settings:
        vaspstdincardict.update(incar_settings)

    # Directory
    vaspstdinputdir = input_dir + "/vasp_std/" if input_dir else "VASP_Files/vasp_std/"
    if not os.path.exists(vaspstdinputdir):
        os.makedirs(vaspstdinputdir)

    # Might need to alter KPOINTS!
    vaspstdkpts = "Kpoints from DefectsWithTheBoys.vasp_std_files\n0\nGamma\n2 2 2"

    vaspstdincar = Incar.from_dict(vaspstdincardict)

    with zopen(vaspstdinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspstdincar.get_string())
    with open(vaspstdinputdir + "KPOINTS", "wt") as kpoints_file:
        kpoints_file.write(vaspstdkpts)


def vasp_ncl_files(
    single_defect_dict: dict, input_dir: str = None, incar_settings: dict = None
) -> None:
    """
    Generates INCAR for vasp_ncl single-shot SOC energy calculation on
    vasp_std-relaxed defect structure.
    For POSCAR, POTCAR, KPOINTS, use on command-line (to continue on from vasp_std run):
    'cp vasp_std/CONTCAR vasp_ncl/POSCAR; cp vasp_std/{POTCAR,CHGCAR} vasp_ncl/'
    and 'cp vasp_std/IBZKPT vasp_ncl/KPOINTS' because you need to use non-symmetrised k-points
    (single-weighted) for accurate SOC calculations.
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
            Highly recommended to look at output INCARs or DefectsWithTheBoys.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
    """
    nelect = single_defect_dict["NELECT"]
    # Variable parameters first
    vaspnclincardict = {
        "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN": "variable parameters",
        "NELECT": nelect,
        "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
        + "if ya think we got a bit of crazy ferromagnetic shit going down",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 450,
        "ICORELEVEL": "0 # Get core potentials in OUTCAR for Kumagai corrections",
        "NSW": 0,
        "LSORBIT": True,
        "EDIFF": 1e-06,
        "EDIFFG": -0.01,
        "ALGO": "All",
        "ADDGRID": True,
        "HFSCREEN": 0.2,
        "IBRION": -1,
        "ICHARG": 1,
        "ISIF": 2,
        "ISYM": 0,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": True,
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }
    if incar_settings:
        vaspnclincardict.update(incar_settings)

    # Directory
    vaspnclinputdir = input_dir + "/vasp_ncl/" if input_dir else "VASP_Files/vasp_ncl/"
    if not os.path.exists(vaspnclinputdir):
        os.makedirs(vaspnclinputdir)

    vaspnclincar = Incar.from_dict(vaspnclincardict)

    with zopen(vaspnclinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspnclincar.get_string())


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
            Highly recommended to look at output INCARs or DefectsWithTheBoys.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
        config (str):
            CONFIG file string. If provided, will also write the CONFIG file (to automate
            convergence tests with vaspup2.0) to each 'input' directory.
            (default: None)
    """

    # Variable parameters first
    vaspconvergeincardict = {
        "# May need to change ISMEAR, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, "
        + "ISPIN": "variable parameters",
        "NUPDOWN": "0 # But could be >0 if ya think there's some magnetic shit going down",
        "NCORE": 12,
        "#KPAR": 1,
        "ENCUT": 450,
        "ISMEAR": "0 # Non-metal, use Gaussian smearing",
        "ISPIN": "1 # 2 if ya think there's some magnetic shit going down",
        "ICORELEVEL": 0,
        "GGA": "PS",
        "ALGO": "Normal",
        "ADDGRID": True,
        "EDIFF": 1e-07,
        "EDIFFG": -0.01,
        "IBRION": -1,
        "ICHARG": 1,
        "ISIF": 3,
        "LASPH": True,
        "LORBIT": 11,
        "LREAL": False,
        "LVHAR": True,
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 0,
        "PREC": "Accurate",
        "SIGMA": 0.2,
    }
    if all(is_metal(element) for element in structure.composition.elements):
        vaspconvergeincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if incar_settings:
        vaspconvergeincardict.update(incar_settings)

    # Directory
    vaspconvergeinputdir = input_dir + "/input/" if input_dir else "VASP_Files/input/"
    if not os.path.exists(vaspconvergeinputdir):
        os.makedirs(vaspconvergeinputdir)

    vasppotcardict = {
        "POTCAR_FUNCTIONAL": "PBE_54",  # May need to change this if you've got
        # different POTCARs
        "POTCAR": {
            "Ac": "Ac",
            "Ag": "Ag",
            "Al": "Al",
            "Ar": "Ar",
            "As": "As",
            "Au": "Au",
            "B": "B",
            "Ba": "Ba_sv",
            "Be": "Be_sv",
            "Bi": "Bi",
            "Br": "Br",
            "C": "C",
            "Ca": "Ca_sv",
            "Cd": "Cd",
            "Ce": "Ce",
            "Cl": "Cl",
            "Co": "Co",
            "Cr": "Cr_pv",
            "Cs": "Cs_sv",
            "Cu": "Cu_pv",
            "Dy": "Dy_3",
            "Er": "Er_3",
            "Eu": "Eu",
            "F": "F",
            "Fe": "Fe_pv",
            "Ga": "Ga_d",
            "Gd": "Gd",
            "Ge": "Ge_d",
            "H": "H",
            "He": "He",
            "Hf": "Hf_pv",
            "Hg": "Hg",
            "Ho": "Ho_3",
            "I": "I",
            "In": "In_d",
            "Ir": "Ir",
            "K": "K_sv",
            "Kr": "Kr",
            "La": "La",
            "Li": "Li_sv",
            "Lu": "Lu_3",
            "Mg": "Mg_pv",
            "Mn": "Mn_pv",
            "Mo": "Mo_pv",
            "N": "N",
            "Na": "Na_pv",
            "Nb": "Nb_pv",
            "Nd": "Nd_3",
            "Ne": "Ne",
            "Ni": "Ni_pv",
            "Np": "Np",
            "O": "O",
            "Os": "Os_pv",
            "P": "P",
            "Pa": "Pa",
            "Pb": "Pb_d",
            "Pd": "Pd",
            "Pm": "Pm_3",
            "Pr": "Pr_3",
            "Pt": "Pt",
            "Pu": "Pu",
            "Rb": "Rb_sv",
            "Re": "Re_pv",
            "Rh": "Rh_pv",
            "Ru": "Ru_pv",
            "S": "S",
            "Sb": "Sb",
            "Sc": "Sc_sv",
            "Se": "Se",
            "Si": "Si",
            "Sm": "Sm_3",
            "Sn": "Sn_d",
            "Sr": "Sr_sv",
            "Ta": "Ta_pv",
            "Tb": "Tb_3",
            "Tc": "Tc_pv",
            "Te": "Te",
            "Th": "Th",
            "Ti": "Ti_pv",
            "Tl": "Tl_d",
            "Tm": "Tm_3",
            "U": "U",
            "V": "V_pv",
            "W": "W_pv",
            "Xe": "Xe",
            "Y": "Y_sv",
            "Yb": "Yb_2",
            "Zn": "Zn",
            "Zr": "Zr_sv",
        },
    }
    vaspconvergekpts = Kpoints().from_dict(
        {"comment": "Kpoints from vasp_gam_files", "generation_style": "Gamma"}
    )
    vaspconvergeincar = Incar.from_dict(vaspconvergeincardict)
    vaspconvergeinput = DictSet(structure, config_dict=vasppotcardict)

    vaspconvergeinput.potcar.write_file(vaspconvergeinputdir + "POTCAR")
    vaspconvergeposcar = Poscar(structure)
    vaspconvergeposcar.write_file(vaspconvergeinputdir + "POSCAR")
    with zopen(vaspconvergeinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspconvergeincar.get_string())
    vaspconvergekpts.write_file(vaspconvergeinputdir + "KPOINTS")
    # generate CONFIG file
    if config:
        with open(vaspconvergeinputdir + "CONFIG", "w+") as config_file:
            config_file.write(config)
        with open(vaspconvergeinputdir + "CONFIG", "a") as config_file:
            config_file.write(f"""\nname="{input_dir[13:]}" # input_dir""")
