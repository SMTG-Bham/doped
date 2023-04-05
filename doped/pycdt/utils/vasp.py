#!/usr/bin/env python

from __future__ import print_function

__status__ = "Development"

import functools
import os
from copy import deepcopy

import numpy as np
from monty.json import MontyEncoder
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from pymatgen.io.vasp.inputs import Kpoints, Potcar, PotcarSingle
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, DictSet


def _check_psp_dir():  # Provided by Katarina Brlec, from github.com/SMTG-UCL/surfaxe
    """
    Helper function to check if potcars are set up correctly for use with
    pymatgen, to be compatible across pymatgen versions (breaking changes in v2022)
    """
    potcar = False
    try:
        import pymatgen.settings

        pmg_settings = pymatgen.settings.SETTINGS
        if "PMG_VASP_PSP_DIR" in pmg_settings:
            potcar = True
    except ModuleNotFoundError:
        try:
            import pymatgen

            pmg_settings = pymatgen.SETTINGS
            if "PMG_VASP_PSP_DIR" in pmg_settings:
                potcar = True
        except AttributeError:
            from pymatgen.core import SETTINGS

            pmg_settings = SETTINGS
            if "PMG_VASP_PSP_DIR" in pmg_settings:
                potcar = True
    return potcar


def _import_psp():
    """import pmg settings for PotcarSingleMod"""
    pmg_settings = None
    try:
        import pymatgen.settings

        pmg_settings = pymatgen.settings.SETTINGS
    except ModuleNotFoundError:
        try:
            import pymatgen

            pmg_settings = pymatgen.SETTINGS
        except AttributeError:
            from pymatgen.core import SETTINGS

            pmg_settings = SETTINGS

    if pmg_settings is None:
        raise ValueError("pymatgen settings not found?")
    else:
        return pmg_settings


class PotcarSingleMod(PotcarSingle):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    # @staticmethod
    # def from_file(filename):
    #    with smart_open.smart_open(filename, "rt") as f:
    #        return PotcarSingle(f.read())

    @staticmethod
    def from_symbol_and_functional(symbol, functional=None):
        settings = _import_psp()
        if functional is None:
            functional = settings.get("PMG_DEFAULT_FUNCTIONAL", "PBE_54")
        funcdir = PotcarSingle.functional_dir[functional]

        if not os.path.isdir(os.path.join(settings.get("PMG_VASP_PSP_DIR"), funcdir)):
            functional_dir = {
                "LDA_US": "pot",
                "PW91_US": "pot_GGA",
                "LDA": "potpaw",
                "PW91": "potpaw_GGA",
                "LDA_52": "potpaw_LDA.52",
                "LDA_54": "potpaw_LDA.54",
                "PBE": "potpaw_PBE",
                "PBE_52": "potpaw_PBE.52",
                "PBE_54": "potpaw_PBE.54",
            }
            funcdir = functional_dir[functional]

        d = settings.get("PMG_VASP_PSP_DIR")
        if d is None:
            raise ValueError(
                "No POTCAR directory found. Please set "
                "the VASP_PSP_DIR environment variable"
            )

        paths_to_try = [
            os.path.join(d, funcdir, "POTCAR.{}".format(symbol)),
            os.path.join(d, funcdir, symbol, "POTCAR.Z"),
            os.path.join(d, funcdir, symbol, "POTCAR"),
        ]
        for p in paths_to_try:
            p = os.path.expanduser(p)
            p = zpath(p)
            if os.path.exists(p):
                return PotcarSingleMod.from_file(p)
        raise IOError(
            "You do not have the right POTCAR with functional "
            + "{} and label {} in your VASP_PSP_DIR".format(functional, symbol)
        )


class PotcarMod(Potcar):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

    def set_symbols(self, symbols, functional=None, sym_potcar_map=None):
        """
        Initialize the POTCAR from a set of symbols. Currently, the POTCARs can
        be fetched from a location specified in .pmgrc.yaml. Use pmg config
        to add this setting.

        Args:
            symbols ([str]): A list of element symbols
            functional (str): The functional to use. If None, the setting
                PMG_DEFAULT_FUNCTIONAL in .pmgrc.yaml is used, or if this is
                not set, it will default to PBE_54.
            sym_potcar_map (dict): A map of symbol:raw POTCAR string. If
                sym_potcar_map is specified, POTCARs will be generated from
                the given map data rather than the config file location.
        """
        del self[:]
        if sym_potcar_map:
            for el in symbols:
                self.append(PotcarSingleMod(sym_potcar_map[el]))
        else:
            for el in symbols:
                p = PotcarSingleMod.from_symbol_and_functional(el, functional)
                self.append(p)


class DefectRelaxSet(DictSet):
    """
    pymatgen DictSet for VASP Defect Relaxation Calculations
    Additional Args:
        charge: Charge of the defect structure
    """

    def __init__(self, structure, **kwargs):
        charge = kwargs.pop("charge", 0)
        config_dict = kwargs.pop("config_dict", {})
        if config_dict:
            super(self.__class__, self).__init__(structure, config_dict=config_dict, **kwargs)
        else:
            mp_set = MPRelaxSet(structure, **kwargs)
            super(self.__class__, self).__init__(structure, config_dict=mp_set.CONFIG, **kwargs)

        self.charge = charge

    @property
    def incar(self):
        inc = super(self.__class__, self).incar
        try:
            inc["NELECT"] = self.nelect - self.charge
            if inc["NELECT"] % 2 != 0:  # odd number of electrons
                inc["NUPDOWN"] = 1
        except Exception:
            print("NELECT and NUPDOWN flags are not set due to non-availability of POTCARs")

        return inc

    @property
    def potcar(self):
        """
        Potcar object.
        """
        return PotcarMod(symbols=self.potcar_symbols, functional=self.potcar_functional)

    @property
    def all_input(self):
        """
        Returns all input files as a dict of {filename: vasp object}

        Returns:
            dict of {filename: object}, e.g., {'INCAR': Incar object, ...}
        """
        try:
            return super(DefectRelaxSet, self).all_input
        except:  # Expecting the error to be POTCAR related, its ignored
            kpoints = self.kpoints
            incar = self.incar
            if np.product(kpoints.kpts) < 4 and incar.get("ISMEAR", 0) == -5:
                incar["ISMEAR"] = 0

            return {"INCAR": incar, "KPOINTS": kpoints, "POSCAR": self.poscar}

    @property
    def structure(self):
        """
        :return: Structure
        """
        return self._structure


class DefectStaticSet(MPStaticSet):
    """
    Extension to MPStaticSet which modifies some parameters appropriate
    for bulk supercell calculation
    """

    def __init__(self, structure, **kwargs):
        super(self.__class__, self).__init__(structure, **kwargs)

    @property
    def potcar(self):
        """
        Potcar object.
        """
        return PotcarMod(symbols=self.potcar_symbols, functional=self.potcar_functional)

    @property
    def all_input(self):
        """
        Returns all input files as a dict of {filename: vasp object}

        Returns:
            dict of {filename: object}, e.g., {'INCAR': Incar object, ...}
        """
        try:
            return super(DefectStaticSet, self).all_input
        except:
            kpoints = self.kpoints
            incar = self.incar
            if np.product(kpoints.kpts) < 4 and incar.get("ISMEAR", 0) == -5:
                incar["ISMEAR"] = 0

            return {"INCAR": incar, "KPOINTS": kpoints, "POSCAR": self.poscar}


class DielectricSet(MPStaticSet):
    """
    Extension to MPStaticSet which modifies some parameters appropriate
    for bulk supercell calculation
    """

    def __init__(self, structure, **kwargs):
        super(self.__class__, self).__init__(structure, lepsilon=True, **kwargs)

    @property
    def potcar(self):
        """
        Potcar object.
        """
        return PotcarMod(symbols=self.potcar_symbols, functional=self.potcar_functional)

    @property
    def all_input(self):
        """
        Returns all input files as a dict of {filename: vasp object}

        Returns:
            dict of {filename: object}, e.g., {'INCAR': Incar object, ...}
        """
        try:
            return super(self.__class__, self).all_input
        except:
            kpoints = self.kpoints
            incar = self.incar
            if np.product(kpoints.kpts) < 4 and incar.get("ISMEAR", 0) == -5:
                incar["ISMEAR"] = 0

            return {"INCAR": incar, "KPOINTS": kpoints, "POSCAR": self.poscar}


def write_additional_files(path, trans_dict=None, incar={}, kpoints=None, hse=False):
    """
        NOTE from developers:
            This code is primarily used for dumping extra files
            for users desiring to do HSE
            will not be used in command line code or maintained (with unit tests etc.)
            going forward (as of 12/15/2017).
            However, we are keeping function here to allow for
            current users to make use of it...

    Write the additional files based on user settings
    """
    if kpoints:
        kpoints.write_file(os.path.join(path, "KPOINTS"))

    if trans_dict:
        dumpfn(trans_dict, os.path.join(path, "transformation.json"), cls=MontyEncoder)

    if hse:
        if not incar:
            raise ValueError("Incar settings need to passed to write incar")

        incar.update({"LWAVE": True})
        incar.write_file(os.path.join(path, "INCAR.gga"))
        incar.update(
            {
                "LHFCALC": True,
                "ALGO": "All",
                "HFSCREEN": 0.2,
                "PRECFOCK": "Fast",
                "NKRED": 2,
            }
        )
        incar.write_file(os.path.join(path, "INCAR.hse1"))
        del incar["PRECFOCK"]
        del incar["NKRED"]
        incar.write_file(os.path.join(path, "INCAR.hse2"))


def make_vasp_defect_files(defects, path_base, user_settings={}, hse=False):
    """
    Generates VASP files for defect computations
    Args:
        defect_structs:
            the defects data as a dictionnary. Ideally this is generated
            from core.defectsmaker.ChargedDefectsStructures.
        path_base:
            where we write the files
        user_settings:
            Settings in dict format to override the defaults used in
            generating vasp files. The format of the dictionary is
            {'defects:{'INCAR':{...},'KPOINTS':{...},
             'bulk':{'INCAR':{...},'KPOINTS':{...}}
        hse:
            hse run or not
    """
    bulk_sys = defects["bulk"]["supercell"]
    comb_defs = functools.reduce(
        lambda x, y: x + y, [defects[key] for key in defects if key != "bulk"]
    )

    # User setting dicts
    user_settings = deepcopy(user_settings)
    user_incar = user_settings.pop("INCAR", {})
    user_incar_blk_tmp = user_incar.pop("bulk", {})
    user_incar_blk_def = user_incar.pop("defects", {})
    user_incar.pop("dielectric", {})
    user_incar_blk = deepcopy(user_incar)
    user_incar_def = deepcopy(user_incar)
    user_incar_blk.update(user_incar_blk_tmp)
    user_incar_def.update(user_incar_blk_def)
    user_kpoints = user_settings.pop("KPOINTS", {})
    potcar_settings = user_settings.pop("POTCAR", {})
    potcar_functional = potcar_settings.pop("functional", "PBE_54")

    for defect in comb_defs:
        for charge in defect["charges"]:
            s = defect["supercell"]
            dict_transf = {
                "defect_type": defect["name"],
                "defect_site": defect["unique_site"],
                "defect_supercell_site": defect["bulk_supercell_site"],
                "defect_multiplicity": defect["site_multiplicity"],
                "charge": charge,
                "supercell": s["size"],
            }
            if "substitution_specie" in defect:
                dict_transf["substitution_specie"] = defect["substitution_specie"]

            defect_relax_set = DefectRelaxSet(
                s["structure"],
                user_incar_settings=user_incar_def,
                user_potcar_settings=potcar_settings,
                potcar_functional=potcar_functional,
                charge=charge,
            )

            path = os.path.join(path_base, defect["name"], "charge_" + str(charge))
            try:
                potcar = defect_relax_set.potcar
            except:
                potcar = None

            if potcar or not charge:
                defect_relax_set.write_input(path)
                incar = defect_relax_set.incar if hse else {}
                kpoints = Kpoints.from_dict(user_kpoints) if user_kpoints else None

                write_additional_files(
                    path, dict_transf, incar=incar, kpoints=kpoints, hse=hse
                )
            else:
                os.makedirs(path)
                with open(os.path.join(path, "readme.txt"), "w") as fp:
                    print(
                        "Vasp input files not generated for charged defects "
                        "due to unavailability of POTCAR. "
                        "If charged defects desired, please supply POTCAR file "
                        "path to .pmgrc.yaml file.",
                        end="",
                        file=fp,
                    )

    # Generate bulk supercell inputs
    s = bulk_sys
    dict_transf = {"defect_type": "bulk", "supercell": s["size"]}

    # potcar_functional = user_potcar.get('functional', 'PBE_54')
    blk_static_set = DefectStaticSet(
        s["structure"],
        user_incar_settings=user_incar_blk,
        user_potcar_settings=potcar_settings,
        potcar_functional=potcar_functional,
    )
    path = os.path.join(path_base, "bulk")
    blk_static_set.write_input(path)

    incar = blk_static_set.incar if hse else {}
    kpoints = Kpoints.from_dict(user_kpoints) if user_kpoints else None

    write_additional_files(path, dict_transf, incar=incar, kpoints=kpoints, hse=hse)


def make_vasp_defect_files_dos(
    defects, path_base, user_settings={}, hse=False, dos_limits=(-1, 7)
):
    """
        NOTE from developers:
            This code will not be used in command line code or
            maintained going forward (as of 12/15/2017).
            However, we are keeping function here to allow for
            current users to make use of it...

    Generates VASP files for defect computations which include dos
    generation. Useful when the user don't want to use MPWorks for
    dos calculations.
    Args:
        defects:
            the defects data as a dictionnary. Ideally this is generated
            from core.defectsmaker.ChargedDefectsStructures.
        path_base:
            where we write the files
        user_settings:
            Settings in dict format to override the defaults used in
            generating vasp files. The format of the dictionary is
            {'defects:{'INCAR':{...},'KPOINTS':{...},
             'bulk':{'INCAR':{...},'KPOINTS':{...}}
        hse:
            hse run or not
        dos_limits:
            Lower and upper limits for dos plot as a tuple. The default
            (-1,7) should work for most of the cases.
    """
    bulk_sys = defects["bulk"]["supercell"]
    comb_defs = functools.reduce(
        lambda x, y: x + y, [defects[key] for key in defects if key != "bulk"]
    )

    for defect in comb_defs:
        for charge in defect["charges"]:
            s = defect["supercell"]
            dict_transf = {
                "defect_type": defect["name"],
                "defect_site": defect["unique_site"],
                "defect_supercell_site": defect["bulk_supercell_site"],
                "charge": charge,
                "supercell": s["size"],
            }
            if "substitution_specie" in defect:
                dict_transf["substitution_specie"] = defect["substitution_specie"]

            mp_relax_set = MPRelaxSet(s["structure"])
            incar = mp_relax_set.incar

            incar.update(
                {
                    "IBRION": 2,
                    "ISIF": 2,
                    "ISPIN": 2,
                    "LWAVE": False,
                    "EDIFF": 1e-5,
                    "EDIFFG": -1e-2,
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "LVTOT": True,
                    "LVHAR": True,
                    "LORBIT": 14,
                    "ALGO": "Fast",
                    "ISYM": 0,
                }
            )
            if user_settings:
                if "INCAR" in user_settings.get("defects", {}):
                    incar.update(user_settings["defects"]["INCAR"])

            comp = s["structure"].composition
            sum_elec = 0
            elts = set()
            for p in mp_relax_set.potcar:
                if p.element not in elts:
                    sum_elec += comp.as_dict()[p.element] * p.nelectrons
                    elts.add(p.element)

            incar["NELECT"] = sum_elec - charge

            kpoint = mp_relax_set.kpoints.monkhorst_automatic()
            path = os.path.join(path_base, defect["name"], "charge_" + str(charge))
            try:
                os.makedirs(path)
            except:
                pass
            if hse:
                incar.update({"LWAVE": True})
                incar.write_file(os.path.join(path, "INCAR.relax.gga"))
                incar.update(
                    {
                        "LHFCALC": True,
                        "ALGO": "All",
                        "HFSCREEN": 0.208,
                        "PRECFOCK": "Fast",
                        "NKRED": 2,
                    }
                )
                incar.write_file(os.path.join(path, "INCAR.relax.hse1"))
                del incar["PRECFOCK"]
                del incar["NKRED"]
                incar.write_file(os.path.join(path, "INCAR.relax.hse2"))
            else:
                incar.write_file(os.path.join(path, "INCAR.relax"))
            kpoint.write_file(os.path.join(path, "KPOINTS"))

            mp_relax_set.poscar.write_file(os.path.join(path, "POSCAR.orig"))
            mp_relax_set.potcar.write_file(os.path.join(path, "POTCAR"))
            dumpfn(
                dict_transf, os.path.join(path, "transformation.json"), cls=MontyEncoder
            )

            # Write addition incar files for dos plots of defect levels
            del incar["NSW"]
            del incar["ISIF"]
            del incar["EDIFFG"]
            del incar["LVHAR"]
            del incar["LVTOT"]
            incar["IBRION"] = -1
            incar["ICHARG"] = 1
            incar["EDIFF"] = 1e-6
            # High acc not required for chgcar
            incar.update({"PRECFOCK": "Fast", "NKRED": 2})
            incar.write_file(os.path.join(path, "INCAR.static"))
            del incar["PRECFOCK"]
            del incar["NKRED"]
            incar["ICHARG"] = 11
            incar["NEDOS"] = int((dos_limits[1] - dos_limits[0]) / 0.006)
            incar["EMIN"] = dos_limits[0]
            incar["EMAX"] = dos_limits[1]
            incar.write_file(os.path.join(path, "INCAR.dos"))

    # Generate bulk supercell inputs
    s = bulk_sys
    dict_transf = {"defect_type": "bulk", "supercell": s["size"]}

    mp_relax_set = MPRelaxSet(s["structure"])
    incar = mp_relax_set.incar

    incar.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ISPIN": 2,
            "LWAVE": False,
            "EDIFF": 1e-5,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVTOT": True,
            "LVHAR": True,
            "ALGO": "Fast",
            "ISYM": 0,
        }
    )
    if user_settings:
        if "INCAR" in user_settings.get("bulk", {}):
            incar.update(user_settings["bulk"]["INCAR"])

    kpoint = mp_relax_set.kpoints.monkhorst_automatic()
    path = os.path.join(path_base, "bulk")
    try:
        os.makedirs(path)
    except:
        pass
    if hse:
        incar.update({"LWAVE": True})
        incar.write_file(os.path.join(path, "INCAR.gga"))
        incar.update(
            {
                "LHFCALC": True,
                "ALGO": "All",
                "HFSCREEN": 0.2,  # "AEXX": 0.45,
                "PRECFOCK": "Fast",
                "NKRED": 2,
            }
        )
        incar.write_file(os.path.join(path, "INCAR.hse1"))
        del incar["PRECFOCK"]
        del incar["NKRED"]
        incar.write_file(os.path.join(path, "INCAR.hse2"))
    else:
        incar.write_file(os.path.join(path, "INCAR"))
    kpoint.write_file(os.path.join(path, "KPOINTS"))
    mp_relax_set.poscar.write_file(os.path.join(path, "POSCAR"))
    mp_relax_set.potcar.write_file(os.path.join(path, "POTCAR"))
    dumpfn(dict_transf, os.path.join(path, "transformation.json"), cls=MontyEncoder)


def make_vasp_dielectric_files(struct, path=None, user_settings={}, hse=False):
    """
    Generates VASP files for dielectric constant computations
    Args:
        struct:
            unitcell in pymatgen structure format
        user_settings:
            Settings in dict format to override the defaults used in
            generating vasp files. The format of the dictionary is
            {'INCAR':{...}, 'KPOINTS':{...}}
        hse:
            hse run or not
    """

    # Generate vasp inputs for dielectric constant
    user_settings = deepcopy(user_settings)
    user_incar = user_settings.pop("INCAR", {})
    user_incar.pop("bulk", {})
    user_incar.pop("defects", {})
    user_incar_diel = user_incar.pop("dielectric", {})
    user_incar.update(user_incar_diel)
    user_kpoints = user_settings.pop("KPOINTS", {})
    grid_density = user_kpoints.get("grid_density", 1000)
    potcar_settings = user_settings.pop("POTCAR", {})
    potcar_functional = potcar_settings.pop("functional", "PBE_54")
    dielectric_set = DielectricSet(
        struct,
        user_incar_settings=user_incar,
        user_potcar_settings=potcar_settings,
        potcar_functional=potcar_functional,
    )

    if not path:
        path_base = struct.composition.reduced_formula
        path = os.path.join(path_base, "dielectric")
    dielectric_set.write_input(path)

    kpoints = Kpoints.automatic_density(struct, grid_density, force_gamma=True)
    incar = dielectric_set.incar if hse else {}
    write_additional_files(path, incar=incar, kpoints=kpoints, hse=hse)
