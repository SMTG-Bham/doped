import os
import shutil
import unittest

from monty.serialization import dumpfn, loadfn
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar

from doped import vasp_input


class VaspInputTestCase(unittest.TestCase):
    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.CDTE_DATA_DIR = os.path.join(self.DATA_DIR, "CdTe")
        self.cdte_generated_defect_dict = loadfn(
            os.path.join(self.CDTE_DATA_DIR, "CdTe_generated_defect_dict.json")
        )

    def tearDown(self):
        for folder in os.listdir(
            self.CDTE_DATA_DIR
        ):  # remove all generated CdTe defect folders
            if os.path.isdir(folder):
                shutil.rmtree(folder)

    def check_generated_vasp_inputs(self, data_dir=None, generated_dir=".",
                                    vasp_type="vasp_gam", check_poscar=True, check_potcar=False,
                                    check_potcar_spec=False):
        if data_dir is None:
            data_dir = self.CDTE_DATA_DIR

        for folder in os.listdir(data_dir):
            if os.path.isdir(folder):
                self.assertTrue(os.path.exists(f"{generated_dir}/{folder}"))
                self.assertTrue(os.path.exists(f"{generated_dir}/{folder}/{vasp_type}"))
                # TODO: May need to do POTCAR spec test for GitHub if possible, otherwise remove

                # load the Incar, Poscar, Potcar and Kpoints and check it matches the previous:
                test_incar = Incar.from_file(
                    f"{self.CDTE_DATA_DIR}/{folder}/{vasp_type}/INCAR"
                )
                incar = Incar.from_file(f"{generated_dir}/{folder}/{vasp_type}/INCAR")
                self.assertEqual(test_incar, incar)

                if check_poscar:
                    test_poscar = Poscar.from_file(
                        f"{self.CDTE_DATA_DIR}/{folder}/vasp_gam/POSCAR"  # POSCAR always checked
                        # against vasp_gam unperturbed POSCAR
                    )
                    poscar = Poscar.from_file(f"{generated_dir}/{folder}/{vasp_type}/POSCAR")
                    self.assertEqual(test_poscar.structure, poscar.structure)

                if check_potcar:
                    test_potcar = Potcar.from_file(
                        f"{self.CDTE_DATA_DIR}/{folder}/{vasp_type}/POTCAR"
                    )
                    potcar = Potcar.from_file(f"{generated_dir}/{folder}/{vasp_type}/POTCAR")
                    self.assertDictEqual(test_potcar.as_dict(), potcar.as_dict())

                if check_potcar_spec:
                    with open(f"{generated_dir}/{folder}/{vasp_type}/POTCAR.spec", "r") as f:
                        contents = f.readlines()
                        self.assertIn(contents[0], ["Cd", "Cd\n"])
                        self.assertIn(contents[1], ["Te", "Te\n"])
                        if "Se" in folder:
                            self.assertIn(contents[2], ["Se", "Se\n"])

                test_kpoints = Kpoints.from_file(
                    f"{self.CDTE_DATA_DIR}/{folder}/{vasp_type}/KPOINTS"
                )
                kpoints = Kpoints.from_file(f"{generated_dir}/{folder}/{vasp_type}/KPOINTS")
                self.assertDictEqual(test_kpoints.as_dict(), kpoints.as_dict())


    def test_vasp_gam_files(self):
        defect_input_dict = vasp_input.prepare_vasp_defect_inputs(
            self.cdte_generated_defect_dict
        )
        for key, val in defect_input_dict.items():
            vasp_input.vasp_gam_files(
                val, input_dir=key, user_incar_settings={"ENCUT": 350}, potcar_spec=True
            )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        self.check_generated_vasp_inputs(check_potcar_spec=True)

    def test_vasp_std_files(self):
        defect_input_dict = vasp_input.prepare_vasp_defect_inputs(
            self.cdte_generated_defect_dict
        )
        for key, val in defect_input_dict.items():
            vasp_input.vasp_std_files(
                val,
                input_dir=key,
                user_incar_settings={
                    "ENCUT": 350,
                    "LREAL": "Auto",
                    "IBRION": 5,
                    "ADDGRID": True,
                },
            )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=False)

                # TODO: Need to pylint the files to check no glaring issues
                # TODO: Add note to example notebook that it should be easy to use with atomate
                #  etc because now returns the DefectRelaxSet objects

        for key, val in defect_input_dict.items():
            vasp_input.vasp_std_files(
                val,
                input_dir=key,
                user_incar_settings={
                    "ENCUT": 350,
                    "LREAL": "Auto",
                    "IBRION": 5,
                    "ADDGRID": True,
                },
                unperturbed_poscar=True,
            )

        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=True)

    def test_vasp_ncl_files(self):
        defect_input_dict = vasp_input.prepare_vasp_defect_inputs(
            self.cdte_generated_defect_dict
        )
        for key, val in defect_input_dict.items():
            vasp_input.vasp_ncl_files(
                val,
                input_dir=key,
                user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
            )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=False)

        for key, val in defect_input_dict.items():
            vasp_input.vasp_ncl_files(
                val,
                input_dir=key,
                user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
                unperturbed_poscar=True,
            )

        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=True)


if __name__ == "__main__":
    unittest.main()
