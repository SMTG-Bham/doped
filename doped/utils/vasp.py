import os
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints, Potcar, PotcarSingle, Poscar
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
    def __init__(self,
                 structure: Structure,
                 config_dict: Optional[Dict] = None,
                 charge: int = 0,
                 poscar_comment: Optional[str] = None,
                 **kwargs):
        """
        Extension to pymatgen DictSet object for VASP Defect Relaxation Calculations

        Args:
            structure (Structure): pymatgen Structure object of the defect supercell
            config_dict (dict): The config dictionary to use.
            charge (int): Charge of the defect structure
            poscar_comment (str): POSCAR file comment
            **kwargs: Additional kwargs to pass to DictSet
        """
        self.charge = charge
        self.poscar_comment = poscar_comment

        if config_dict is not None:
            self.CONFIG = config_dict  # Bug in pymatgen 2023.5.10 # TODO: PR pymatgen to fix this Yb issue with `DictSet`
            super(self.__class__, self).__init__(
                structure, config_dict=config_dict, **kwargs
            )
        else:
            MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
            default_potcar_dict = loadfn(
                os.path.join(MODULE_DIR, "../PotcarSet.yaml")
            )
            relax_set = loadfn(os.path.join(MODULE_DIR, "../HSE06_RelaxSet.yaml"))
            defect_set = loadfn(os.path.join(MODULE_DIR, "../DefectSet.yaml"))
            relax_set["INCAR"].update(defect_set["INCAR"])
            relax_set.update(default_potcar_dict)

            from doped.vasp_input import scaled_ediff

            relax_set["INCAR"]["EDIFF"] = scaled_ediff(len(structure))

            self.CONFIG = relax_set
            super(self.__class__, self).__init__(
                structure, config_dict=relax_set, **kwargs
            )

    @property
    def incar(self):
        inc = super(self.__class__, self).incar
        try:
            inc["NELECT"] = self.nelect - self.charge
            if inc["NELECT"] % 2 != 0:  # odd number of electrons
                inc["NUPDOWN"] = 1
            else:
                # when writing VASP just resets this to 0 anyway:
                inc["NUPDOWN"] = "0  # If defect has multiple spin-polarised states (e.g. bipolarons) could " \
                                 "also have triplet (NUPDOWN=2), but ΔE typically small."

        except Exception as e:
            warnings.warn(
                f"NELECT and NUPDOWN flags are not set due to non-availability of POTCARs; "
                f"got error {e}"
            )

        return inc

    @property
    def potcar(self):
        """
        Potcar object.
        """
        return PotcarMod(symbols=self.potcar_symbols, functional=self.potcar_functional)

    @property
    def poscar(self) -> Poscar:
        """
        Return Poscar object with comment
        """
        return Poscar(self.structure, comment=self.poscar_comment)

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
