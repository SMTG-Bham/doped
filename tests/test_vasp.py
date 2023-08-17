"""
Tests for the `doped.vasp` module.
"""
import os
import shutil
import unittest

from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

from doped.generation import DefectsGenerator
from doped.vasp import DefectDictSet, DefectRelaxSet


class DefectDictSetTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.prim_cdte = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        self.cdte_defect_gen = DefectsGenerator(self.prim_cdte)
        self.ytos_bulk_supercell = Structure.from_file(f"{self.example_dir}/YTOS/Bulk/POSCAR")
        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.prim_cu = Structure.from_file(f"{self.data_dir}/Cu_prim_POSCAR")

        self.neutral_def_incar_min = {
            "ICORELEVEL": "0  # Needed if using the Kumagai-Oba (eFNV) anisotropic charge "
            "correction scheme".lower(),
            "ISIF": 2,  # Fixed supercell for defects
            "ISPIN": 2,  # Spin polarisation likely for defects
            "ISYM": "0  # Symmetry breaking extremely likely for defects".lower(),
            "LVHAR": True,
            "ISMEAR": 0,
        }
        self.hse06_incar_min = {
            "LHFCALC": True,
            "PRECFOCK": "Fast",
            "GGA": "PE",  # underlying GGA functional for HSE/PBE0 is PBE
            "AEXX": 0.25,  # changed for HSE(a); HSE06 assumed by default
            "HFSCREEN": 0.208,  # # correct HSE screening parameter; changed for PBE0
        }
        self.def_keys = ["EDIFF", "EDIFFG", "IBRION"]

    def test_neutral_defect_incar(self):
        dds = DefectDictSet(self.prim_cdte)  # fine for a bulk primitive input as well
        assert self.neutral_def_incar_min.items() <= dds.incar.items()
        assert set(self.def_keys).issubset(dds.incar)
        assert self.hse06_incar_min.items() <= dds.incar.items()  # HSE06 by default
        assert dds.structure == self.prim_cdte
        assert dds.charge_state == 0
        print(dds.kpoints.as_dict())

        dds = DefectDictSet(self.cdte_defect_gen["Te_Cd_0"].defect_supercell)
        assert self.neutral_def_incar_min.items() <= dds.incar.items()
        assert set(self.def_keys).issubset(dds.incar)
        assert self.hse06_incar_min.items() <= dds.incar.items()  # HSE06 by default

    def test_charged_defect_incar(self):
        dds = DefectDictSet(self.prim_cdte, charge_state=-2)  # fine for a bulk primitive input as well
        assert self.neutral_def_incar_min.items() <= dds.incar.items()
        assert set(self.def_keys).issubset(dds.incar)

        dds = DefectDictSet(self.cdte_defect_gen["Te_Cd_-2"].defect_supercell)  # default, charge state
        # from
        # DefectEntry
        assert self.neutral_def_incar_min.items() <= dds.incar.items()
        assert set(self.def_keys).issubset(dds.incar)

        dds = DefectDictSet(self.cdte_defect_gen["Te_Cd_-2"].defect_supercell, charge_state=-2)
        assert self.neutral_def_incar_min.items() <= dds.incar.items()
        assert set(self.def_keys).issubset(dds.incar)

    def test_user_settings_defect_incar(self):
        user_incar_settings = {"EDIFF": 1e-8, "EDIFFG": 0.1, "ENCUT": 720}
        drs = DefectRelaxSet(self.structure, charge=1, user_incar_settings=user_incar_settings)
        assert self.neutral_def_incar_min.items() <= drs.incar.items()
        assert set(self.def_keys).issubset(drs.incar)
        assert drs.incar["ENCUT"] == 720
        assert drs.incar["EDIFF"] == 1e-08
        assert drs.incar["EDIFFG"] == 0.1


class VaspInputTestCase(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
        self.cdte_defects_generator = loadfn(
            os.path.join(self.cdte_data_dir, "CdTe_defects_generator.json")
        )

    def tearDown(self):
        for folder in os.listdir(self.cdte_data_dir):  # remove all generated CdTe defect folders
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
            print(f"{generated_dir}/{folder}")
            assert os.path.exists(f"{generated_dir}/{folder}")
            assert os.path.exists(f"{generated_dir}/{folder}/{vasp_type}")

            # load the Incar, Poscar and Kpoints and check it matches the previous:
            test_incar = Incar.from_file(f"{data_dir}/{folder}/{vasp_type}/INCAR")
            incar = Incar.from_file(f"{generated_dir}/{folder}/{vasp_type}/INCAR")
            # remove NUPDOWN and NELECT entries from test_incar:
            test_incar.pop("NUPDOWN", None)
            incar.pop("NUPDOWN", None)
            test_incar.pop("NELECT", None)
            incar.pop("NELECT", None)
            assert test_incar == incar

            if check_poscar:
                test_poscar = Poscar.from_file(
                    f"{data_dir}/{folder}/vasp_gam/POSCAR"  # POSCAR always checked
                    # against vasp_gam unperturbed POSCAR
                )
                poscar = Poscar.from_file(f"{generated_dir}/{folder}/{vasp_type}/POSCAR")
                assert test_poscar.structure == poscar.structure

            if check_potcar_spec:
                with open(f"{generated_dir}/{folder}/{vasp_type}/POTCAR.spec") as file:
                    contents = file.readlines()
                    assert contents[0] in ["Cd", "Cd\n"]
                    assert contents[1] in ["Te", "Te\n"]
                    if "Se" in folder:
                        assert contents[2] in ["Se", "Se\n"]

            test_kpoints = Kpoints.from_file(f"{data_dir}/{folder}/{vasp_type}/KPOINTS")
            kpoints = Kpoints.from_file(f"{generated_dir}/{folder}/{vasp_type}/KPOINTS")
            assert test_kpoints.as_dict() == kpoints.as_dict()

        if data_dir is None:
            data_dir = self.cdte_data_dir

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

    # def test_vasp_gam_files(self):
    #     vasp_gam_files(
    #         self.cdte_defects_generator,
    #         user_incar_settings={"ENCUT": 350},
    #         potcar_spec=True,
    #         user_potcar_functional=None,
    #     )
    #
    #     # assert that the same folders in self.cdte_data_dir are present in the current directory
    #     self.check_generated_vasp_inputs(check_potcar_spec=True)
    #
    #     # test custom POTCAR choice:
    #     # dict call acts on DefectsGenerator.defect_entries dict attribute
    #     vac_Te_dict = {
    #         defect_species: defect_entry
    #         for defect_species, defect_entry in self.cdte_defects_generator.items()
    #         if "v_Te" in defect_species
    #     }
    #
    #     vasp_gam_files(
    #         vac_Te_dict,
    #         user_potcar_settings={"Cd": "Cd_sv_GW", "Te": "Te_sv_GW"},
    #         user_potcar_functional=None,
    #         potcar_spec=True,
    #     )
    #
    #     with open("v_Te_s1_0/vasp_gam/POTCAR.spec") as file:
    #         contents = file.readlines()
    #         assert contents[0] in ["Cd_sv_GW", "Cd_sv_GW\n"]
    #         assert contents[1] in ["Te_sv_GW", "Te_sv_GW\n"]
    #
    #     # test no files written with write_files=False
    #     self.tearDown()
    #     vasp_gam_files(
    #         self.cdte_defects_generator,
    #         user_potcar_functional=None,
    #         write_files=False,
    #     )
    #     for folder in os.listdir(self.cdte_data_dir):
    #         if os.path.isdir(f"{self.cdte_data_dir}/{folder}"):
    #             assert not os.path.exists(f"./{folder}")
    #
    # def test_vasp_gam_files_single_defect_entry(self):
    #     # dict call acts on DefectsGenerator.defect_entries dict attribute:
    #     single_defect_entry = self.cdte_defects_generator["v_Cd_s0_0"]  # V_Cd
    #     vasp_gam_files(
    #         single_defect_entry,
    #         user_incar_settings={"ENCUT": 350},
    #         potcar_spec=True,
    #         user_potcar_functional=None,
    #     )
    #
    #     # assert that the same folders in self.cdte_data_dir are present in the current directory
    #     self.check_generated_vasp_inputs(
    #         generated_dir="v_Cd_s0_0",
    #         data_dir=f"{self.cdte_data_dir}/v_Cd_s0_0",
    #         check_potcar_spec=True,
    #         single_defect_dir=True,
    #     )
    #
    #     # assert only neutral directory written:
    #     assert not os.path.exists("v_Cd_s0_-1")
    #
    #     # test custom POTCAR choice:
    #     # dict call acts on DefectsGenerator.defect_entries dict attribute
    #     vac_Te_dict = {
    #         defect_species: defect_entry
    #         for defect_species, defect_entry in self.cdte_defects_generator.items()
    #         if "v_Te" in defect_species
    #     }
    #
    #     vasp_gam_files(
    #         vac_Te_dict,
    #         user_potcar_settings={"Cd": "Cd_sv_GW", "Te": "Te_sv_GW"},
    #         user_potcar_functional=None,
    #         potcar_spec=True,
    #     )
    #
    #     with open("v_Te_s1_0/vasp_gam/POTCAR.spec") as file:
    #         contents = file.readlines()
    #         assert contents[0] in ["Cd_sv_GW", "Cd_sv_GW\n"]
    #         assert contents[1] in ["Te_sv_GW", "Te_sv_GW\n"]
    #
    #     # test no files written with write_files=False
    #     self.tearDown()
    #     vasp_gam_files(vac_Te_dict, user_potcar_functional=None, write_files=False)
    #     assert not os.path.exists("v_Te_s1_0")
    #
    # def test_vasp_std_files(self):
    #     vasp_std_files(
    #         self.cdte_defects_generator,
    #         user_incar_settings={
    #             "ENCUT": 350,
    #             "LREAL": "Auto",
    #             "IBRION": 5,
    #             "ADDGRID": True,
    #         },
    #         user_potcar_functional=None,
    #     )
    #
    #     # assert that the same folders in self.cdte_data_dir are present in the current directory
    #     self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=False)
    #
    #     vasp_std_files(
    #         self.cdte_defects_generator,
    #         user_incar_settings={
    #             "ENCUT": 350,
    #             "LREAL": "Auto",
    #             "IBRION": 5,
    #             "ADDGRID": True,
    #         },
    #         unperturbed_poscar=True,
    #         user_potcar_functional=None,
    #     )
    #     self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=True)
    #
    #     # test no files written with write_files=False
    #     self.tearDown()
    #     vasp_std_files(
    #         self.cdte_defects_generator,
    #         user_potcar_functional=None,
    #         write_files=False,
    #     )
    #     for folder in os.listdir(self.cdte_data_dir):
    #         if os.path.isdir(f"{self.cdte_data_dir}/{folder}"):
    #             assert not os.path.exists(f"./{folder}")
    #
    # def test_vasp_std_files_single_defect_entry(self):
    #     # test interstitials this time:
    #     single_defect_entry = self.cdte_defects_generator["Cd_i_m1b_+2"]  # Cd_i
    #     # TODO: Fails because defect_entry.name is nuked when reloading from json, will fix!
    #     vasp_std_files(
    #         single_defect_entry,
    #         user_incar_settings={
    #             "ENCUT": 350,
    #             "LREAL": "Auto",
    #             "IBRION": 5,
    #             "ADDGRID": True,
    #         },
    #         user_potcar_functional=None,
    #     )
    #
    #     # assert that the same folders in self.cdte_data_dir are present in the current directory
    #     print(os.listdir())
    #     self.check_generated_vasp_inputs(
    #         generated_dir="Cd_i_m1b_+2",
    #         data_dir=f"{self.cdte_data_dir}/Cd_i_m1b_+2",
    #         vasp_type="vasp_std",
    #         check_poscar=False,
    #         single_defect_dir=True,
    #     )
    #
    #     # test unperturbed POSCAR:
    #     vasp_std_files(
    #         single_defect_entry,
    #         user_incar_settings={
    #             "ENCUT": 350,
    #             "LREAL": "Auto",
    #             "IBRION": 5,
    #             "ADDGRID": True,
    #         },
    #         user_potcar_functional=None,
    #         unperturbed_poscar=True,
    #     )
    #
    #     self.check_generated_vasp_inputs(
    #         generated_dir="Cd_i_m1b_+2",
    #         data_dir=f"{self.cdte_data_dir}/Cd_i_m1b_+2",
    #         vasp_type="vasp_std",
    #         check_poscar=False,
    #         single_defect_dir=True,
    #     )
    #
    #     # test no files written with write_files=False
    #     self.tearDown()
    #     vasp_gam_files(single_defect_entry, user_potcar_functional=None, write_files=False)
    #     assert not os.path.exists("Cd_i_m1b_+2")
    #
    # def test_vasp_ncl_files(self):
    #     vasp_ncl_files(
    #         self.cdte_defects_generator,
    #         user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
    #         user_potcar_functional=None,
    #     )
    #
    #     # assert that the same folders in self.cdte_data_dir are present in the current directory
    #     self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=False)
    #
    #     vasp_ncl_files(
    #         self.cdte_defects_generator,
    #         user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
    #         unperturbed_poscar=True,
    #         user_potcar_functional=None,
    #     )
    #     self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=True)
    #
    #     # test no files written with write_files=False
    #     self.tearDown()
    #     vasp_ncl_files(
    #         self.cdte_defects_generator,
    #         user_potcar_functional=None,
    #         write_files=False,
    #     )
    #     for folder in os.listdir(self.cdte_data_dir):
    #         if os.path.isdir(f"{self.cdte_data_dir}/{folder}"):
    #             assert not os.path.exists(f"./{folder}")
    #
    # def test_vasp_ncl_files_single_defect_dict(self):
    #     # test substitutions this time:
    #     single_defect_dict = self.cdte_generated_defect_dict["substitutions"][0]  # sub_2_Se_on_Te
    #     vasp_ncl_files(
    #         single_defect_dict,
    #         user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
    #         user_potcar_functional=None,
    #     )
    #
    #     # assert that the same folders in self.cdte_data_dir are present in the current directory
    #     for charge in range(-1, 6):
    #         self.check_generated_vasp_inputs(
    #             generated_dir=f"sub_2_Se_on_Te_{charge}",
    #             data_dir=f"{self.cdte_data_dir}/sub_2_Se_on_Te_{charge}",
    #             vasp_type="vasp_ncl",
    #             check_poscar=False,
    #             single_defect_dir=True,
    #         )
    #
    #     # test unperturbed POSCAR:
    #     vasp_ncl_files(
    #         single_defect_dict,
    #         user_incar_settings={"ENCUT": 750, "LREAL": True, "ADDGRID": False},
    #         user_potcar_functional=None,
    #         unperturbed_poscar=True,
    #     )
    #
    #     for charge in range(-1, 6):
    #         self.check_generated_vasp_inputs(
    #             generated_dir=f"sub_2_Se_on_Te_{charge}",
    #             data_dir=f"{self.cdte_data_dir}/sub_2_Se_on_Te_{charge}",
    #             vasp_type="vasp_ncl",
    #             check_poscar=True,
    #             single_defect_dir=True,
    #         )
    #
    #     # test no files written with write_files=False
    #     self.tearDown()
    #     vasp_ncl_files(single_defect_dict, user_potcar_functional=None, write_files=False)
    #     for charge in range(-1, 6):
    #         assert not os.path.exists(f"sub_2_Se_on_Te_{charge}")


if __name__ == "__main__":
    unittest.main()
