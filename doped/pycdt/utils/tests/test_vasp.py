# coding: utf-8

from __future__ import division

import unittest
import os

from pymatgen.core.structure import Structure
from monty.serialization import loadfn

from doped.pycdt.utils.vasp import PotcarMod, PotcarSingleMod, DefectRelaxSet

__status__ = "Development"

file_loc = os.path.abspath(os.path.join(__file__, "..", "..", "..", "test_files"))


class DefectRelaxTest(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_file(os.path.join(file_loc, "POSCAR_Cr2O3"))
        self.user_settings = loadfn(os.path.join(file_loc, "test_vasp_settings.yaml"))
        self.path = "Cr2O3"
        self.neutral_def_incar_min = {
              "ICORELEVEL": "0  # Needed if using the Kumagai-Oba (eFNV) anisotropic charge "
                            "correction scheme".lower(),
              "ISIF": 2,  # Fixed supercell for defects
              "ISPIN": 2,  # Spin polarisation likely for defects
              "ISYM": "0  # Symmetry breaking extremely likely for defects".lower(),
              "LVHAR": True,
              "ISMEAR": 0,
        }
        self.def_keys = ["EDIFF", "EDIFFG", "IBRION"]

    def test_neutral_defect_incar(self):
        drs = DefectRelaxSet(self.structure)
        self.assertTrue(self.neutral_def_incar_min.items() <= drs.incar.items())
        self.assertTrue(set(self.def_keys).issubset(drs.incar))

    def test_charged_defect_incar(self):
        drs = DefectRelaxSet(self.structure, charge=1)
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


if __name__ == "__main__":
    unittest.main()
