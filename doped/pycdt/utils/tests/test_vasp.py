# coding: utf-8

from __future__ import division

import glob
import os
import unittest

from monty.json import MontyDecoder
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar

from doped.pycdt.utils.vasp import *

__status__ = "Development"

file_loc = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "test_files"))


class PotcarSingleModTest(unittest.TestCase):
    """
    This test is applicable for the specific case where POTCAR files are not
    organized according to The Materials Project directory layout, but
    in the default layout.
    """

    def setUp(self):
        pass

    @unittest.expectedFailure
    def test_from_symbol_and_functional(self):
        try:
            potcar = PotcarSingleMod.from_symbol_and_functional("Ni")
        except:
            potcar = None

        self.assertIsNotNone(potcar)


class PotcarModTest(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.expectedFailure
    def test_set_symbols(self):
        try:
            potcar = PotcarMod(symbols=["Ni", "O"])
        except:
            potcar = None

        self.assertIsNotNone(potcar)


class DefectRelaxTest(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_file(os.path.join(file_loc, "POSCAR_Cr2O3"))
        self.user_settings = loadfn(os.path.join(file_loc, "test_vasp_settings.yaml"))
        self.path = "Cr2O3"
        self.neutral_def_incar_min = {
            "LVHAR": True,
            "ISYM": 0,
            "ISMEAR": 0,
            "ISIF": 2,
            "ISPIN": 2,
        }
        self.def_keys = ["EDIFF", "EDIFFG", "IBRION"]

    def test_neutral_defect_incar(self):
        drs = DefectRelaxSet(self.structure)
        self.assertTrue(self.neutral_def_incar_min.items() <= drs.incar.items())
        self.assertTrue(set(self.def_keys).issubset(drs.incar))

    @unittest.expectedFailure
    def test_charged_defect_incar(self):
        drs = DefectRelaxSet(self.structure, charge=1)
        self.assertIn("NELECT", drs.incar)
        self.assertTrue(self.neutral_def_incar_min.items() <= drs.incar.items())
        self.assertTrue(set(self.def_keys).issubset(drs.incar))

    def test_user_settings_defect_incar(self):
        user_incar_settings = {"EDIFF": 1e-8, "EDIFFG": 0.1, "ENCUT": 720}
        drs = DefectRelaxSet(
            self.structure, charge=1, user_incar_settings=user_incar_settings
        )
        self.assertTrue(self.neutral_def_incar_min.items() <= drs.incar.items())
        self.assertTrue(set(self.def_keys).issubset(drs.incar))
        self.assertEqual(drs.incar["ENCUT"], 720)
        self.assertEqual(drs.incar["EDIFF"], 1e-8)
        self.assertEqual(drs.incar["EDIFFG"], 0.1)


class MakeVaspDefectFilesTest(unittest.TestCase):
    def setUp(self):
        self.defects = loadfn(os.path.join(file_loc, "Cr2O3_defects.json"))
        self.user_settings = loadfn(os.path.join(file_loc, "test_vasp_settings.yaml"))
        self.path = "Cr2O3"
        self.neutral_def_incar_min = {
            "LVHAR": True,
            "ISYM": 0,
            "ISMEAR": 0,
            "ISIF": 2,
            "ISPIN": 2,
        }
        self.def_keys = ["EDIFF", "EDIFFG", "IBRION"]
        self.bulk_incar = {
            "LVHAR": True,
            "ISYM": 0,
            "ISMEAR": 0,
            "IBRION": -1,
            "ISPIN": 2,
        }
        self.bulk_keys = ["EDIFF"]

    def test_neutral_defect_incar(self):
        with ScratchDir("."):
            make_vasp_defect_files(self.defects, self.path)
            cr_def_path = glob.glob(os.path.join(self.path, "vac*Cr"))[0]
            incar_loc = os.path.join(cr_def_path, "charge_0")
            incar = Incar.from_file(os.path.join(incar_loc, "INCAR"))
            self.assertTrue(self.neutral_def_incar_min.items() <= incar.items())
            self.assertTrue(set(self.def_keys).issubset(incar))

    def test_charged_defect_incar(self):
        with ScratchDir("."):
            make_vasp_defect_files(self.defects, self.path)
            cr_def_path = glob.glob(os.path.join(self.path, "vac*Cr"))[0]
            incar_loc = os.path.join(cr_def_path, "charge_-1")
            try:
                potcar = Potcar.from_file(os.path.join(incar_loc, "POTCAR"))
            except:
                potcar = None
            if potcar:
                incar = Incar.from_file(os.path.join(incar_loc, "INCAR"))
                self.assertIsNotNone(incar.pop("NELECT", None))
                self.assertTrue(self.neutral_def_incar_min.items() <= incar.items())
                self.assertTrue(set(self.def_keys).issubset(incar))

    def test_bulk_incar(self):
        with ScratchDir("."):
            make_vasp_defect_files(self.defects, self.path)
            incar_loc = os.path.join(self.path, "bulk")
            incar = Incar.from_file(os.path.join(incar_loc, "INCAR"))
            self.assertTrue(self.bulk_incar.items() <= incar.items())
            self.assertTrue(set(self.bulk_keys).issubset(incar))

    def test_kpoints(self):
        with ScratchDir("."):
            make_vasp_defect_files(self.defects, self.path)
            kpoints_loc = os.path.join(self.path, "bulk")
            kpoints = Kpoints.from_file(os.path.join(kpoints_loc, "KPOINTS"))

    def test_poscar(self):
        with ScratchDir("."):
            make_vasp_defect_files(self.defects, self.path)
            poscar_loc = os.path.join(self.path, "bulk")
            poscar = Poscar.from_file(os.path.join(poscar_loc, "POSCAR"))
            self.assertTrue(
                poscar.structure.matches(self.defects["bulk"]["supercell"]["structure"])
            )

    def test_user_settings_bulk(self):
        user_settings = loadfn(os.path.join(file_loc, "test_vasp_settings.yaml"))
        with ScratchDir("."):
            make_vasp_defect_files(self.defects, self.path, user_settings=user_settings)
            incar_loc = os.path.join(self.path, "bulk")
            incar = Incar.from_file(os.path.join(incar_loc, "INCAR"))
            self.assertTrue(self.bulk_incar.items() <= incar.items())
            self.assertTrue(set(self.bulk_keys).issubset(incar))
            self.assertEqual(incar["ENCUT"], 620)

    def test_hse_settings(self):
        pass


class MakeVaspDielectricFilesTest(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_file(os.path.join(file_loc, "POSCAR_Cr2O3"))
        self.user_settings = loadfn(os.path.join(file_loc, "test_vasp_settings.yaml"))
        self.path = "dielectric"
        self.dielectric_min = {"LPEAD": True, "LEPSILON": True, "IBRION": 8}
        self.keys = ["EDIFF"]

    def test_dielectric_files(self):
        with ScratchDir("."):
            make_vasp_dielectric_files(self.structure, self.path)
            incar = Incar.from_file(os.path.join(self.path, "INCAR"))
            self.assertTrue(self.dielectric_min.items() <= incar.items())
            self.assertTrue(set(self.keys).issubset(incar))

    def test_user_modified_dielectric_files(self):
        with ScratchDir("."):
            make_vasp_dielectric_files(
                self.structure, self.path, user_settings=self.user_settings
            )
            incar = Incar.from_file(os.path.join(self.path, "INCAR"))
            self.assertTrue(self.dielectric_min.items() <= incar.items())
            self.assertTrue(set(self.keys).issubset(incar))
            self.assertEqual(incar["EDIFF"], 5e-7)
            self.assertEqual(incar["ENCUT"], 620)


if __name__ == "__main__":
    unittest.main()
