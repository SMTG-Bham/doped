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

    def check_generated_vasp_inputs(
        self,
        data_dir=None,
        generated_dir=".",
        vasp_type="vasp_gam",
        check_poscar=True,
        check_potcar_spec=False,
    ):
        if data_dir is None:
            data_dir = self.CDTE_DATA_DIR

        for folder in os.listdir(data_dir):
            if os.path.isdir(folder):
                print(f"{generated_dir}/{folder}")
                self.assertTrue(os.path.exists(f"{generated_dir}/{folder}"))
                self.assertTrue(os.path.exists(f"{generated_dir}/{folder}/{vasp_type}"))

                # load the Incar, Poscar and Kpoints and check it matches the previous:
                test_incar = Incar.from_file(
                    f"{self.CDTE_DATA_DIR}/{folder}/{vasp_type}/INCAR"
                )
                incar = Incar.from_file(f"{generated_dir}/{folder}/{vasp_type}/INCAR")
                # remove NUPDOWN and NELECT entries from test_incar:
                test_incar.pop("NUPDOWN", None)
                incar.pop("NUPDOWN", None)
                test_incar.pop("NELECT", None)
                incar.pop("NELECT", None)
                self.assertEqual(test_incar, incar)

                if check_poscar:
                    test_poscar = Poscar.from_file(
                        f"{self.CDTE_DATA_DIR}/{folder}/vasp_gam/POSCAR"  # POSCAR always checked
                        # against vasp_gam unperturbed POSCAR
                    )
                    poscar = Poscar.from_file(
                        f"{generated_dir}/{folder}/{vasp_type}/POSCAR"
                    )
                    self.assertEqual(test_poscar.structure, poscar.structure)

                if check_potcar_spec:
                    with open(
                        f"{generated_dir}/{folder}/{vasp_type}/POTCAR.spec", "r"
                    ) as f:
                        contents = f.readlines()
                        self.assertIn(contents[0], ["Cd", "Cd\n"])
                        self.assertIn(contents[1], ["Te", "Te\n"])
                        if "Se" in folder:
                            self.assertIn(contents[2], ["Se", "Se\n"])

                test_kpoints = Kpoints.from_file(
                    f"{self.CDTE_DATA_DIR}/{folder}/{vasp_type}/KPOINTS"
                )
                kpoints = Kpoints.from_file(
                    f"{generated_dir}/{folder}/{vasp_type}/KPOINTS"
                )
                self.assertDictEqual(test_kpoints.as_dict(), kpoints.as_dict())

    def test_vasp_gam_files(self):
        vasp_input.vasp_gam_files(
            self.cdte_generated_defect_dict,
            user_incar_settings={"ENCUT": 350},
            potcar_spec=True,
            user_potcar_functional=None,
        )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        self.check_generated_vasp_inputs(check_potcar_spec=True)

        # test custom POTCAR choice:
        vac_Te_dict = {
            defect_type: [
                defect for defect in defect_list if defect["name"] == "vac_2_Te"
            ]
            for defect_type, defect_list in self.cdte_generated_defect_dict.items()
            if defect_type == "vacancies"
        }

        vasp_input.vasp_gam_files(
            vac_Te_dict,
            user_potcar_settings={"Cd": "Cd_sv_GW", "Te": "Te_sv_GW"},
            user_potcar_functional=None,
            potcar_spec=True,
        )

        with open(f"vac_2_Te_0/vasp_gam/POTCAR.spec", "r") as f:
            contents = f.readlines()
            self.assertIn(contents[0], ["Cd_sv_GW", "Cd_sv_GW\n"])
            self.assertIn(contents[1], ["Te_sv_GW", "Te_sv_GW\n"])

    def test_vasp_std_files(self):
        vasp_input.vasp_std_files(
            self.cdte_generated_defect_dict,
            user_incar_settings={
                "ENCUT": 350,
                "LREAL": "Auto",
                "IBRION": 5,
                "ADDGRID": True,
            },
            user_potcar_functional=None,
        )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=False)

        # TODO: Need to pylint the files to check no glaring issues
        # TODO: Add note to example notebook that it should be easy to use with atomate
        #  etc because now returns the DefectRelaxSet objects

        vasp_input.vasp_std_files(
            self.cdte_generated_defect_dict,
            user_incar_settings={
                "ENCUT": 350,
                "LREAL": "Auto",
                "IBRION": 5,
                "ADDGRID": True,
            },
            unperturbed_poscar=True,
            user_potcar_functional=None,
        )
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=True)

    def test_vasp_ncl_files(self):
        vasp_input.vasp_ncl_files(
            self.cdte_generated_defect_dict,
            user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
            user_potcar_functional=None,
        )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=False)

        vasp_input.vasp_ncl_files(
            self.cdte_generated_defect_dict,
            user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
            unperturbed_poscar=True,
            user_potcar_functional=None,
        )
        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=True)


if __name__ == "__main__":
    unittest.main()
