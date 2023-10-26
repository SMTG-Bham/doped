"""
Tests for the `doped.vasp` module.
"""
import filecmp
import os
import unittest

import numpy as np
from ase.build import bulk, make_supercell
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from test_generation import if_present_rm

from doped.generation import DefectsGenerator
from doped.vasp import (
    DefectDictSet,
    DefectsSet,
    _test_potcar_functional_choice,
    default_defect_relax_set,
    default_potcar_dict,
    scaled_ediff,
)

# TODO: Flesh out these tests. Try test most possible combos, warnings and errors too. Test DefectEntry
#  jsons etc. Test POTCAR warning when POTCARs not available


def _potcars_available() -> bool:
    """
    Check if the POTCARs are available for the tests (i.e. testing locally).

    If not (testing on GitHub Actions), POTCAR testing will be skipped.
    """
    try:
        _test_potcar_functional_choice("PBE")
        return True
    except ValueError:
        return False


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
        # AgCu:
        atoms = bulk("Cu")
        atoms = make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        atoms.set_chemical_symbols(["Cu", "Ag"] * 4)
        aaa = AseAtomsAdaptor()
        self.agcu = aaa.get_structure(atoms)

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
            "GGA": "Pe",  # gets changed from PE to Pe in DictSet initialisation
            "AEXX": 0.25,  # changed for HSE(a); HSE06 assumed by default
            "HFSCREEN": 0.208,  # # correct HSE screening parameter; changed for PBE0
        }
        self.doped_std_kpoint_comment = "KPOINTS from doped, with reciprocal_density = 100/Å⁻³"
        self.doped_gam_kpoint_comment = "Γ-only KPOINTS from doped"

    def defect_dict_set_defaults_check(self, struct, incar_check=True, **dds_kwargs):
        dds = DefectDictSet(
            struct,
            potcars=_potcars_available(),  # to allow testing on GH Actions
            **dds_kwargs,
        )  # fine for a bulk primitive input as well
        if incar_check:
            assert self.neutral_def_incar_min.items() <= dds.incar.items()
            assert self.hse06_incar_min.items() <= dds.incar.items()  # HSE06 by default
            assert dds.incar["EDIFF"] == scaled_ediff(len(struct))
            for k, v in default_defect_relax_set["INCAR"].items():
                if k in [
                    "EDIFF_PER_ATOM",
                    *list(self.neutral_def_incar_min.keys()),
                    *list(self.hse06_incar_min.keys()),
                ]:  # already tested
                    continue

                assert k in dds.incar
                if isinstance(v, str):  # DictSet converts all strings to capitalised lowercase
                    try:
                        val = float(v[:2])
                        assert val == dds.incar[k]
                    except ValueError:
                        assert v.lower().capitalize() == dds.incar[k]
                else:
                    assert v == dds.incar[k]

        if dds.potcars:
            for potcar_functional in [
                dds.potcar_functional,
                dds.potcar.functional,
                dds.potcar.as_dict()["functional"],
            ]:
                assert "PBE" in potcar_functional

            assert set(dds.potcar.as_dict()["symbols"]) == {
                default_potcar_dict["POTCAR"][el_symbol] for el_symbol in dds.structure.symbol_set
            }

        assert dds.structure == struct
        if "charge_state" not in dds_kwargs:
            assert dds.charge_state == 0
        else:
            assert dds.charge_state == dds_kwargs["charge_state"]
        assert dds.kpoints.comment == self.doped_std_kpoint_comment
        return dds

    def kpts_nelect_nupdown_check(self, dds, kpt, nelect, nupdown):
        assert dds.kpoints.kpts == [[kpt, kpt, kpt]]
        if dds.potcars:
            assert dds.incar["NELECT"] == nelect
            assert dds.incar["NUPDOWN"] == nupdown

    def test_neutral_defect_incar(self):
        dds = self.defect_dict_set_defaults_check(self.prim_cdte.copy())
        self.kpts_nelect_nupdown_check(dds, 7, 18, 0)  # reciprocal_density = 100/Å⁻³ for prim CdTe

        defect_entry = self.cdte_defect_gen["Te_Cd_0"]
        dds = self.defect_dict_set_defaults_check(defect_entry.defect_supercell)
        self.kpts_nelect_nupdown_check(dds, 2, 570, 0)  # reciprocal_density = 100/Å⁻³ for CdTe supercell

    def test_charged_defect_incar(self):
        dds = self.defect_dict_set_defaults_check(
            self.prim_cdte.copy(), charge_state=-2
        )  # also tests dds.charge_state
        self.kpts_nelect_nupdown_check(dds, 7, 20, 0)  # reciprocal_density = 100/Å⁻³ for prim CdTe

        defect_entry = self.cdte_defect_gen["Te_Cd_0"]
        dds = self.defect_dict_set_defaults_check(defect_entry.defect_supercell.copy(), charge_state=-2)
        self.kpts_nelect_nupdown_check(dds, 2, 572, 0)  # reciprocal_density = 100/Å⁻³ for CdTe supercell

        defect_entry = self.cdte_defect_gen["Te_Cd_-2"]
        dds = self.defect_dict_set_defaults_check(defect_entry.defect_supercell.copy(), charge_state=-2)
        self.kpts_nelect_nupdown_check(dds, 2, 572, 0)  # reciprocal_density = 100/Å⁻³ for CdTe supercell

    def test_user_settings_defect_incar(self):
        user_incar_settings = {"EDIFF": 1e-8, "EDIFFG": 0.1, "ENCUT": 720, "NCORE": 4, "KPAR": 7}
        dds = self.defect_dict_set_defaults_check(
            self.prim_cdte.copy(),
            incar_check=False,
            charge_state=1,
            user_incar_settings=user_incar_settings,
        )
        self.kpts_nelect_nupdown_check(dds, 7, 17, 1)  # reciprocal_density = 100/Å⁻³ for prim CdTe
        assert self.neutral_def_incar_min.items() <= dds.incar.items()
        assert self.hse06_incar_min.items() <= dds.incar.items()  # HSE06 by default
        for k, v in user_incar_settings.items():
            assert v == dds.incar[k]

        # non-HSE settings:
        gga_dds = self.defect_dict_set_defaults_check(
            self.prim_cdte.copy(),
            incar_check=False,
            charge_state=10,
            user_incar_settings={"LHFCALC": False},
        )
        self.kpts_nelect_nupdown_check(gga_dds, 7, 8, 0)  # reciprocal_density = 100/Å⁻³ for prim CdTe
        assert gga_dds.incar["LHFCALC"] is False
        for k in self.hse06_incar_min:
            if k not in ["LHFCALC", "GGA"]:
                assert k not in gga_dds.incar

        assert gga_dds.incar["GGA"] == "Ps"  # GGA functional set to Ps (PBEsol) by default

    def test_initialisation_for_all_structs(self):
        """
        Test the initialisation of DefectDictSet for a range of structure
        types.
        """
        for struct in [
            self.ytos_bulk_supercell,
            self.lmno_primitive,
            self.prim_cu,
            self.agcu,
        ]:
            self.defect_dict_set_defaults_check(struct)

            self.defect_dict_set_defaults_check(struct, charge_state=np.random.randint(-5, 5))

            DefectDictSet(
                struct,
                user_incar_settings={"ENCUT": 350},
                user_potcar_functional="PBE_52",
                user_potcar_settings={"Cu": "Cu_pv"},
                user_kpoints_settings={"reciprocal_density": 200},
                poscar_comment="Test pop",
            )


class DefectsSetTest(unittest.TestCase):
    def setUp(self):
        # get setup attributes from DefectDictSetTest:
        dds_test = DefectDictSetTest()
        dds_test.setUp()
        for attr in dir(dds_test):
            if not attr.startswith("_") and "setUp" not in attr and "tearDown" not in attr:
                setattr(self, attr, getattr(dds_test, attr))

        self.cdte_defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/cdte_defect_gen.json")
        self.cdte_custom_test_incar_settings = {"ENCUT": 350, "NCORE": 10, "LVHAR": False, "ALGO": "All"}

    def tearDown(self):
        for file in os.listdir():
            if file.endswith(".json"):
                if_present_rm(file)

        for folder in os.listdir():
            if any(os.path.exists(f"{folder}/vasp_{xxx}") for xxx in ["gam", "std", "ncl"]):
                # generated output files
                if_present_rm(folder)

    def check_generated_vasp_inputs(
        self,
        data_dir=None,
        generated_dir=".",
        vasp_type="vasp_gam",
        check_poscar=True,
        check_potcar_spec=False,
        single_defect_dir=False,
        bulk=True,
    ):
        def _check_single_vasp_dir(
            data_dir=None,
            generated_dir=".",
            folder="",
            vasp_type="vasp_gam",
            check_poscar=True,
            check_potcar_spec=False,
        ):
            # print(f"{generated_dir}/{folder}")  # to help debug if tests fail
            assert os.path.exists(f"{generated_dir}/{folder}")
            assert os.path.exists(f"{generated_dir}/{folder}/{vasp_type}")

            # load the Incar, Poscar and Kpoints and check it matches the previous:
            test_incar = Incar.from_file(f"{data_dir}/{folder}/{vasp_type}/INCAR")
            incar = Incar.from_file(f"{generated_dir}/{folder}/{vasp_type}/INCAR")
            # test NELECT and NUPDOWN if present in generated INCAR (i.e. if POTCARs available (testing
            # locally)), otherwise pop from test_incar:
            if not _potcars_available():  # to allow testing on GH Actions
                test_incar.pop("NELECT", None)
                test_incar.pop("NUPDOWN", None)

            assert test_incar == incar

            if check_poscar:
                test_poscar = Poscar.from_file(
                    f"{data_dir}/{folder}/vasp_gam/POSCAR"  # POSCAR always checked
                    # against vasp_gam unperturbed POSCAR
                )
                poscar = Poscar.from_file(f"{generated_dir}/{folder}/{vasp_type}/POSCAR")
                assert test_poscar.structure == poscar.structure

            if check_potcar_spec:
                with open(f"{generated_dir}/{folder}/{vasp_type}/POTCAR.spec", encoding="utf-8") as file:
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
                if os.path.isdir(f"{data_dir}/{folder}") and ("bulk" not in folder or bulk):
                    _check_single_vasp_dir(
                        data_dir=data_dir,
                        generated_dir=generated_dir,
                        folder=folder,
                        vasp_type=vasp_type,
                        check_poscar=check_poscar,
                        check_potcar_spec=check_potcar_spec,
                    )

    def test_cdte_files(self):
        cdte_se_defect_gen = DefectsGenerator(self.prim_cdte, extrinsic="Se")
        defects_set = DefectsSet(
            cdte_se_defect_gen,
            user_incar_settings=self.cdte_custom_test_incar_settings,
            user_potcar_functional=None,
        )
        defects_set.write_files(potcar_spec=True)
        # test no vasp_gam files written:
        for folder in os.listdir("."):
            assert not os.path.exists(f"{folder}/vasp_gam")

        # test no (unperturbed) POSCAR files written:
        for folder in os.listdir("."):
            if os.path.isdir(folder) and "bulk" not in folder:
                for subfolder in os.listdir(folder):
                    assert not os.path.exists(f"{folder}/{subfolder}/POSCAR")

        defects_set.write_files(potcar_spec=True, unperturbed_poscar=True, vasp_gam=True)

        bulk_supercell = Structure.from_file("CdTe_bulk/vasp_ncl/POSCAR")
        structure_matcher = StructureMatcher(
            comparator=ElementComparator(), primitive_cell=False
        )  # ignore oxidation states
        assert structure_matcher.fit(bulk_supercell, self.cdte_defect_gen.bulk_supercell)
        # check_generated_vasp_inputs also checks bulk folders

        assert os.path.exists("CdTe_defects_generator.json")
        cdte_se_defect_gen.to_json("test_CdTe_defects_generator.json")
        assert filecmp.cmp("CdTe_defects_generator.json", "test_CdTe_defects_generator.json")

        # assert that the same folders in self.cdte_data_dir are present in the current directory
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=False)  # tests vasp_gam
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=False, bulk=False)  # vasp_std

        # test vasp_nkred_std: same as vasp_std except for NKRED
        for folder in os.listdir("."):
            if os.path.isdir(f"{folder}/vasp_std"):
                assert filecmp.cmp(f"{folder}/vasp_nkred_std/KPOINTS", f"{folder}/vasp_std/KPOINTS")
                # assert filecmp.cmp(f"{folder}/vasp_nkred_std/POTCAR", f"{folder}/vasp_std/POTCAR")
                nkred_incar = Incar.from_file(f"{folder}/vasp_nkred_std/INCAR")
                std_incar = Incar.from_file(f"{folder}/vasp_std/INCAR")
                nkred_incar.pop("NKRED", None)
                assert nkred_incar == std_incar
        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=False, bulk=True)  # vasp_ncl

        # test unperturbed POSCARs and all bulk
        defects_set = DefectsSet(
            self.cdte_defect_gen,
            user_incar_settings=self.cdte_custom_test_incar_settings,
            user_potcar_functional=None,
        )
        defects_set.write_files(potcar_spec=True, unperturbed_poscar=True, bulk="all", vasp_gam=True)
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=True, bulk=True)  # vasp_std
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=True)  # tests vasp_gam
        self.check_generated_vasp_inputs(vasp_type="vasp_nkred_std", check_poscar=False, bulk=True)

        # test DefectDictSet objects:
        for _defect_species, defect_relax_set in defects_set.defect_sets.items():
            for defect_dict_set in [defect_relax_set.vasp_gam, defect_relax_set.bulk_vasp_gam]:
                assert defect_dict_set.kpoints.kpts == [[1, 1, 1]]
            for defect_dict_set in [
                defect_relax_set.vasp_std,
                defect_relax_set.bulk_vasp_std,
                defect_relax_set.vasp_nkred_std,
                defect_relax_set.bulk_vasp_nkred_std,
                defect_relax_set.vasp_ncl,
                defect_relax_set.bulk_vasp_ncl,
            ]:
                assert defect_dict_set.kpoints.kpts == [[2, 2, 2]]

            # TODO: Add more tests here once DefectRelaxSet tests written

        # test custom POTCAR and KPOINTS choices (INCAR already tested): also tests dictionary input to
        # DefectsSet
        self.tearDown()
        defects_set = DefectsSet(
            {k: v for k, v in self.cdte_defect_gen.items() if "v_Te" in k},
            user_potcar_settings={"Cd": "Cd_sv_GW", "Te": "Te_GW"},
            user_kpoints_settings={"reciprocal_density": 500},
            user_potcar_functional=None,
        )
        defects_set.write_files(potcar_spec=True, vasp_gam=True)  # include vasp_gam to compare POTCAR.spec
        for folder in os.listdir("."):
            if os.path.isdir(f"{folder}/vasp_gam"):
                with open(f"{folder}/vasp_gam/POTCAR.spec", encoding="utf-8") as file:
                    contents = file.readlines()
                    assert contents[0] in ["Cd_sv_GW", "Cd_sv_GW\n"]
                    assert contents[1] in ["Te_GW", "Te_GW\n"]

                for subfolder in ["vasp_std", "vasp_nkred_std", "vasp_ncl"]:
                    kpoints = Kpoints.from_file(f"{folder}/{subfolder}/KPOINTS")
                    assert kpoints.kpts == [[3, 3, 3]]

    def test_write_files_single_defect_entry(self):
        single_defect_entry = self.cdte_defect_gen["Cd_i_C3v_+2"]
        defects_set = DefectsSet(
            single_defect_entry,
            user_incar_settings=self.cdte_custom_test_incar_settings,
            user_potcar_functional=None,
        )
        defects_set.write_files(potcar_spec=True, vasp_gam=True, unperturbed_poscar=True)

        # assert that the same folders in self.cdte_data_dir are present in the current directory
        self.check_generated_vasp_inputs(  # tests vasp_gam
            generated_dir="Cd_i_C3v_+2",
            data_dir=f"{self.cdte_data_dir}/Cd_i_C3v_+2",
            check_potcar_spec=True,
            single_defect_dir=True,
        )
        self.check_generated_vasp_inputs(  # vasp_std
            generated_dir="Cd_i_C3v_+2",
            data_dir=f"{self.cdte_data_dir}/Cd_i_C3v_+2",
            vasp_type="vasp_std",
            check_poscar=True,
            single_defect_dir=True,
        )
        self.check_generated_vasp_inputs(  # vasp_ncl
            generated_dir="Cd_i_C3v_+2",
            data_dir=f"{self.cdte_data_dir}/Cd_i_C3v_+2",
            vasp_type="vasp_ncl",
            check_poscar=True,
            single_defect_dir=True,
        )

        # assert only +2 directory written:
        assert not os.path.exists("Cd_i_C3v_0")

    def test_write_files_defect_entry_list(self):
        defect_entry_list = [
            defect_entry
            for defect_species, defect_entry in self.cdte_defect_gen.items()
            if "Cd_i" in defect_species
        ]
        defects_set = DefectsSet(
            defect_entry_list,
            user_incar_settings=self.cdte_custom_test_incar_settings,
            user_potcar_functional=None,
        )
        defects_set.write_files(potcar_spec=True)

        for defect_entry in defect_entry_list:
            for vasp_type in ["vasp_nkred_std", "vasp_std", "vasp_ncl"]:  # no vasp_gam by default
                self.check_generated_vasp_inputs(
                    generated_dir=defect_entry.name,
                    data_dir=f"{self.cdte_data_dir}/{defect_entry.name}",
                    vasp_type=vasp_type,
                    single_defect_dir=True,
                    check_poscar=False,
                )

    def test_initialise_and_write_all_defect_gens(self):
        # test initialising DefectsSet with our generation-tests materials, and writing files to disk
        for defect_gen_name in [
            "ytos_defect_gen",
            "ytos_defect_gen_supercell",
            "lmno_defect_gen",
            "cu_defect_gen",
            "agcu_defect_gen",
            "cd_i_supercell_defect_gen",
        ]:
            print(f"Initialising and testing:{defect_gen_name}")
            defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json")
            defects_set = DefectsSet(
                defect_gen,
                user_potcar_functional=None,  # to allow testing on GH Actions
            )
            defects_set.write_files(potcar_spec=True)
            self.tearDown()  # delete generated folders each time


if __name__ == "__main__":
    unittest.main()
