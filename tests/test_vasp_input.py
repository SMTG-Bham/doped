import os
import shutil
import unittest

from monty.serialization import loadfn
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

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
        single_defect_dir=False,
    ):
        def _check_single_vasp_dir(
            data_dir=None,
            generated_dir=".",
            folder="",
            vasp_type="vasp_gam",
            check_poscar=True,
            check_potcar_spec=False,
        ):
            self.assertTrue(os.path.exists(f"{generated_dir}/{folder}"))
            self.assertTrue(os.path.exists(f"{generated_dir}/{folder}/{vasp_type}"))

            # load the Incar, Poscar and Kpoints and check it matches the previous:
            test_incar = Incar.from_file(f"{data_dir}/{folder}/{vasp_type}/INCAR")
            incar = Incar.from_file(f"{generated_dir}/{folder}/{vasp_type}/INCAR")
            # remove NUPDOWN and NELECT entries from test_incar:
            test_incar.pop("NUPDOWN", None)
            incar.pop("NUPDOWN", None)
            test_incar.pop("NELECT", None)
            incar.pop("NELECT", None)
            self.assertEqual(test_incar, incar)

            if check_poscar:
                test_poscar = Poscar.from_file(
                    f"{data_dir}/{folder}/vasp_gam/POSCAR"  # POSCAR always checked
                    # against vasp_gam unperturbed POSCAR
                )
                poscar = Poscar.from_file(
                    f"{generated_dir}/{folder}/{vasp_type}/POSCAR"
                )
                self.assertEqual(test_poscar.structure, poscar.structure)

            if check_potcar_spec:
                with open(
                    f"{generated_dir}/{folder}/{vasp_type}/POTCAR.spec", "r"
                ) as file:
                    contents = file.readlines()
                    self.assertIn(contents[0], ["Cd", "Cd\n"])
                    self.assertIn(contents[1], ["Te", "Te\n"])
                    if "Se" in folder:
                        self.assertIn(contents[2], ["Se", "Se\n"])

            test_kpoints = Kpoints.from_file(f"{data_dir}/{folder}/{vasp_type}/KPOINTS")
            kpoints = Kpoints.from_file(f"{generated_dir}/{folder}/{vasp_type}/KPOINTS")
            self.assertDictEqual(test_kpoints.as_dict(), kpoints.as_dict())

        if data_dir is None:
            data_dir = self.CDTE_DATA_DIR

        if single_defect_dir:
            _check_single_vasp_dir(
                data_dir=data_dir,
                generated_dir=generated_dir,
                folder="",
                vasp_type=vasp_type,
                check_poscar=check_poscar,
                check_potcar_spec=check_potcar_spec,
            )

        else:
            for folder in os.listdir(data_dir):
                if os.path.isdir(f"{data_dir}/{folder}"):
                    _check_single_vasp_dir(
                        data_dir=data_dir,
                        generated_dir=generated_dir,
                        folder=folder,
                        vasp_type=vasp_type,
                        check_poscar=check_poscar,
                        check_potcar_spec=check_potcar_spec,
                    )

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

        with open("vac_2_Te_0/vasp_gam/POTCAR.spec", "r") as file:
            contents = file.readlines()
            self.assertIn(contents[0], ["Cd_sv_GW", "Cd_sv_GW\n"])
            self.assertIn(contents[1], ["Te_sv_GW", "Te_sv_GW\n"])

        # test no files written with write_files=False
        self.tearDown()
        vasp_input.vasp_gam_files(
            self.cdte_generated_defect_dict,
            user_potcar_functional=None,
            write_files=False
        )
        for folder in os.listdir(self.CDTE_DATA_DIR):
            if os.path.isdir(f"{self.CDTE_DATA_DIR}/{folder}"):
                self.assertFalse(os.path.exists(f"./{folder}"))


    def test_vasp_gam_files_single_defect_dict(self):
        single_defect_dict = self.cdte_generated_defect_dict["vacancies"][0]  # V_Cd
        vasp_input.vasp_gam_files(
            single_defect_dict,
            user_incar_settings={"ENCUT": 350},
            potcar_spec=True,
            user_potcar_functional=None,
        )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        for charge in range(-2, 1):
            self.check_generated_vasp_inputs(
                generated_dir=f"vac_1_Cd_{charge}",
                data_dir=f"{self.CDTE_DATA_DIR}/vac_1_Cd_{charge}",
                check_potcar_spec=True,
                single_defect_dir=True,
            )

        # test custom POTCAR choice:
        vac_Te_dict = self.cdte_generated_defect_dict["vacancies"][1]  # V_Te

        vasp_input.vasp_gam_files(
            vac_Te_dict,
            user_potcar_settings={"Cd": "Cd_sv_GW", "Te": "Te_sv_GW"},
            user_potcar_functional=None,
            potcar_spec=True,
        )

        with open("vac_2_Te_0/vasp_gam/POTCAR.spec", "r") as file:
            contents = file.readlines()
            self.assertIn(contents[0], ["Cd_sv_GW", "Cd_sv_GW\n"])
            self.assertIn(contents[1], ["Te_sv_GW", "Te_sv_GW\n"])

        # test no files written with write_files=False
        self.tearDown()
        vasp_input.vasp_gam_files(
            vac_Te_dict,
            user_potcar_functional=None,
            write_files=False
        )
        self.assertFalse(os.path.exists(f"vac_2_Te_0"))

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

        # test no files written with write_files=False
        self.tearDown()
        vasp_input.vasp_std_files(
            self.cdte_generated_defect_dict,
            user_potcar_functional=None,
            write_files=False
        )
        for folder in os.listdir(self.CDTE_DATA_DIR):
            if os.path.isdir(f"{self.CDTE_DATA_DIR}/{folder}"):
                self.assertFalse(os.path.exists(f"./{folder}"))

    def test_vasp_std_files_single_defect_dict(self):
        # test interstitials this time:
        single_defect_dict = self.cdte_generated_defect_dict["interstitials"][
            0
        ]  # inter_1_Cd
        vasp_input.vasp_std_files(
            single_defect_dict,
            user_incar_settings={
                "ENCUT": 350,
                "LREAL": "Auto",
                "IBRION": 5,
                "ADDGRID": True,
            },
            user_potcar_functional=None,
        )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        for charge in range(0, 3):
            self.check_generated_vasp_inputs(
                generated_dir=f"inter_1_Cd_{charge}",
                data_dir=f"{self.CDTE_DATA_DIR}/inter_1_Cd_{charge}",
                vasp_type="vasp_std",
                check_poscar=False,
                single_defect_dir=True,
            )

        # test unperturbed POSCAR:
        vasp_input.vasp_std_files(
            single_defect_dict,
            user_incar_settings={
                "ENCUT": 350,
                "LREAL": "Auto",
                "IBRION": 5,
                "ADDGRID": True,
            },
            user_potcar_functional=None,
            unperturbed_poscar=True,
        )

        for charge in range(0, 3):
            self.check_generated_vasp_inputs(
                generated_dir=f"inter_1_Cd_{charge}",
                data_dir=f"{self.CDTE_DATA_DIR}/inter_1_Cd_{charge}",
                vasp_type="vasp_std",
                check_poscar=True,
                single_defect_dir=True,
            )

        # test no files written with write_files=False
        self.tearDown()
        vasp_input.vasp_gam_files(
            single_defect_dict,
            user_potcar_functional=None,
            write_files=False
        )
        self.assertFalse(os.path.exists(f"inter_1_Cd_0"))
        self.assertFalse(os.path.exists(f"inter_1_Cd_1"))
        self.assertFalse(os.path.exists(f"inter_1_Cd_2"))

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

        # test no files written with write_files=False
        self.tearDown()
        vasp_input.vasp_ncl_files(
            self.cdte_generated_defect_dict,
            user_potcar_functional=None,
            write_files=False
        )
        for folder in os.listdir(self.CDTE_DATA_DIR):
            if os.path.isdir(f"{self.CDTE_DATA_DIR}/{folder}"):
                self.assertFalse(os.path.exists(f"./{folder}"))

    def test_vasp_ncl_files_single_defect_dict(self):
        # test substitutions this time:
        single_defect_dict = self.cdte_generated_defect_dict["substitutions"][
            0
        ]  # sub_2_Se_on_Te
        vasp_input.vasp_ncl_files(
            single_defect_dict,
            user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
            user_potcar_functional=None,
        )

        # assert that the same folders in self.CDTE_DATA_DIR are present in the current directory
        for charge in range(-1, 6):
            self.check_generated_vasp_inputs(
                generated_dir=f"sub_2_Se_on_Te_{charge}",
                data_dir=f"{self.CDTE_DATA_DIR}/sub_2_Se_on_Te_{charge}",
                vasp_type="vasp_ncl",
                check_poscar=False,
                single_defect_dir=True,
            )

        # test unperturbed POSCAR:
        vasp_input.vasp_ncl_files(
            single_defect_dict,
            user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
            user_potcar_functional=None,
            unperturbed_poscar=True,
        )

        for charge in range(-1, 6):
            self.check_generated_vasp_inputs(
                generated_dir=f"sub_2_Se_on_Te_{charge}",
                data_dir=f"{self.CDTE_DATA_DIR}/sub_2_Se_on_Te_{charge}",
                vasp_type="vasp_ncl",
                check_poscar=True,
                single_defect_dir=True,
            )

        # test no files written with write_files=False
        self.tearDown()
        vasp_input.vasp_ncl_files(
            single_defect_dict,
            user_potcar_functional=None,
            write_files=False
        )
        for charge in range(-1, 6):
            self.assertFalse(os.path.exists(f"sub_2_Se_on_Te_{charge}"))


if __name__ == "__main__":
    unittest.main()
