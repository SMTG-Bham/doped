"""
Tests for the `doped.vasp` module.
"""
import contextlib
import filecmp
import locale
import os
import random
import unittest
import warnings
from threading import Thread

import numpy as np
from ase.build import bulk, make_supercell
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import BadIncarWarning, Incar, Kpoints, Poscar, Potcar
from test_generation import if_present_rm

from doped.generation import DefectsGenerator
from doped.vasp import (
    DefectDictSet,
    DefectRelaxSet,
    DefectsSet,
    DopedKpoints,
    _test_potcar_functional_choice,
    default_defect_relax_set,
    default_potcar_dict,
    scaled_ediff,
    singleshot_incar_settings,
)

# TODO: Flesh out these tests. Try test most possible combos, warnings and errors too. Test DefectEntry
#  jsons etc. See AgSbTe2 testing in `doped_generation` notebook -> add tests for all these combos
# TODO: All warnings and errors tested? (So far all DefectDictSet ones done)


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


def _check_potcar_dir_not_setup_warning_error(dds, message, unperturbed_poscar=True):
    if unperturbed_poscar and dds.charge_state != 0:
        ending_string = "so only '(unperturbed) `POSCAR` and `KPOINTS` files will be generated."

    elif not unperturbed_poscar and dds.charge_state != 0:  # only KPOINTS can be written so no good
        ending_string = "so no input files will be generated."

    else:
        ending_string = "so `POTCAR` files will not be generated."

    return all(x in str(message) for x in ["POTCAR directory not set up with pymatgen", ending_string])


def _check_no_potcar_available_warning_error(symbol, message):
    return all(
        x in str(message)
        for x in [
            f"No POTCAR for {symbol} with functional",
            "Please set the PMG_VASP_PSP_DIR",
        ]
    )


def _check_nelect_nupdown_error(message):
    return "NELECT (i.e. supercell charge) and NUPDOWN (i.e. spin state) INCAR flags cannot be set" in str(
        message
    )


def _check_nupdown_neutral_cell_warning(message):
    return all(
        x in str(message)
        for x in [
            "NUPDOWN (i.e. spin state) INCAR flag cannot be set",
            "As this is a neutral supercell, the INCAR file will be written",
        ]
    )


class DefectDictSetTest(unittest.TestCase):
    def setUp(self):
        # don't run heavy tests on GH Actions, these are run locally (too slow without multiprocessing etc)
        self.heavy_tests = bool(_potcars_available())
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.CdTe_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.prim_cdte = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        self.CdTe_defect_gen = DefectsGenerator(self.prim_cdte)
        self.ytos_bulk_supercell = Structure.from_file(f"{self.example_dir}/YTOS/Bulk/POSCAR")
        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.prim_cu = Structure.from_file(f"{self.data_dir}/Cu_prim_POSCAR")
        self.N_doped_diamond_supercell = Structure.from_file(f"{self.data_dir}/N_C_diamond_POSCAR")
        # AgCu:
        atoms = bulk("Cu")
        atoms = make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        atoms.set_chemical_symbols(["Cu", "Ag"] * 4)
        aaa = AseAtomsAdaptor()
        self.agcu = aaa.get_structure(atoms)
        self.sqs_agsbte2 = Structure.from_file(f"{self.data_dir}/AgSbTe2_SQS_POSCAR")

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

    def tearDown(self):
        for i in ["test_pop", "YTOS_test_dir"]:
            if_present_rm(i)

    def _general_defect_dict_set_check(self, dds, struct, incar_check=True, **dds_kwargs):
        if incar_check:
            self._check_dds_incar(dds, struct)
        if _potcars_available():
            for potcar_functional in [
                dds.potcar_functional,
                dds.potcar.functional,
                dds.potcar.as_dict()["functional"],
            ]:
                assert "PBE" in potcar_functional

            potcar_settings = default_potcar_dict["POTCAR"].copy()
            potcar_settings.update(dds.user_potcar_settings or {})
            assert set(dds.potcar.as_dict()["symbols"]) == {
                potcar_settings[el_symbol] for el_symbol in dds.structure.symbol_set
            }
        else:
            assert not dds.potcars
            with self.assertRaises(ValueError) as e:
                _test_pop = dds.potcar
            assert _check_no_potcar_available_warning_error(dds.potcar_symbols[0], e.exception)

            if dds.charge_state != 0:
                with self.assertRaises(ValueError) as e:
                    _test_pop = dds.incar
                assert _check_nelect_nupdown_error(e.exception)
            else:
                self._check_dds_incar_and_writing_warnings(dds)
        assert dds.structure == struct
        # test no unwanted structure reordering
        assert len(Poscar(dds.structure).site_symbols) == len(set(Poscar(dds.structure).site_symbols))

        if "charge_state" not in dds_kwargs:
            assert dds.charge_state == 0
        else:
            assert dds.charge_state == dds_kwargs["charge_state"]
        if isinstance(dds.user_kpoints_settings, dict) and dds.user_kpoints_settings.get(
            "reciprocal_density", False
        ):  # comment changed!
            assert dds.kpoints.comment == self.doped_std_kpoint_comment.replace(
                "100", str(dds.user_kpoints_settings.get("reciprocal_density"))
            )
        else:
            assert dds.kpoints.comment in [self.doped_std_kpoint_comment, self.doped_gam_kpoint_comment]

    def _check_potcar_nupdown_dds_warnings(self, w, dds):
        assert any(_check_potcar_dir_not_setup_warning_error(dds, warning.message) for warning in w)
        assert any(_check_nupdown_neutral_cell_warning(warning.message) for warning in w)
        assert any(
            _check_no_potcar_available_warning_error(dds.potcar_symbols[0], warning.message)
            for warning in w
        )

    def _check_dds_incar(self, dds, struct):
        expected_incar_settings = self.neutral_def_incar_min.copy()
        expected_incar_settings.update(self.hse06_incar_min)  # HSE06 by default
        expected_incar_settings.update(dds.user_incar_settings)
        if dds.incar.get("IBRION") == -1 or dds.incar.get("NSW") == 0:
            _isif = expected_incar_settings.pop("ISIF", None)  # ISIF = 2 not set w/static calculations now
        expected_incar_settings_w_none_vals = expected_incar_settings.copy()
        # remove any entries where value is None:
        expected_incar_settings = {k: v for k, v in expected_incar_settings.items() if v is not None}
        assert expected_incar_settings.items() <= dds.incar.items()
        if dds.incar.get("NSW", 0) > 0:
            assert dds.incar["EDIFF"] == scaled_ediff(len(struct))
        else:
            assert dds.incar["EDIFF"] == 1e-6  # hard set to 1e-6 for static calculations

        for k, v in default_defect_relax_set["INCAR"].items():
            if k in [
                "EDIFF_PER_ATOM",
                *list(expected_incar_settings_w_none_vals.keys()),  # to ensure we skip EDIFFG/POTIM
                # -> None in singleshot calcs
                "ISIF",
            ]:  # already tested
                continue

            assert k in dds.incar
            if isinstance(v, str):  # DictSet converts all strings to capitalised lowercase
                try:
                    val = float(v[:2])
                    if k in dds.user_incar_settings:  # has been overwritten
                        assert val != dds.incar[k]
                    else:
                        assert val == dds.incar[k]
                except ValueError:
                    if k in dds.user_incar_settings:
                        assert v.lower().capitalize() == dds.user_incar_settings[k]
                    else:
                        assert v.lower().capitalize() == dds.incar[k]
            elif k in dds.user_incar_settings:
                assert dds.incar[k] == dds.user_incar_settings[k]
            else:
                assert v == dds.incar[k]

    def _check_dds_incar_and_writing_warnings(self, dds):
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            _test_pop = dds.incar
        assert any(_check_nupdown_neutral_cell_warning(warning.message) for warning in w)

        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            dds.write_input("test_pop")

        self._check_potcar_nupdown_dds_warnings(w, dds)
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            dds.write_input("test_pop", unperturbed_poscar=False)

        self._check_potcar_nupdown_dds_warnings(w, dds)

    def _check_dds(self, dds, struct, **kwargs):
        # INCARs only generated for charged defects when POTCARs available:
        if _potcars_available():
            self._general_defect_dict_set_check(  # also tests dds.charge_state
                dds, struct, incar_check=kwargs.pop("incar_check", True), **kwargs
            )
        else:
            if kwargs.pop("incar_check", True) and dds.charge_state != 0:  # charged defect INCAR
                with self.assertRaises(ValueError) as e:
                    self._general_defect_dict_set_check(  # also tests dds.charge_state
                        dds, struct, incar_check=kwargs.pop("incar_check", True), **kwargs
                    )
                _check_nelect_nupdown_error(e.exception)
            self._general_defect_dict_set_check(  # also tests dds.charge_state
                dds, struct, incar_check=kwargs.pop("incar_check", False), **kwargs
            )

    def _generate_and_check_dds(self, struct, incar_check=True, **dds_kwargs):
        dds = DefectDictSet(struct, **dds_kwargs)  # fine for bulk prim input as well
        self._check_dds(dds, struct, incar_check=incar_check, **dds_kwargs)
        return dds

    def kpts_nelect_nupdown_check(self, dds, kpt, nelect, nupdown):
        if isinstance(kpt, int):
            assert dds.kpoints.kpts == [[kpt, kpt, kpt]]
        else:
            assert dds.kpoints.kpts == kpt
        if _potcars_available():
            assert dds.incar["NELECT"] == nelect
            assert dds.incar["NUPDOWN"] == nupdown
        else:
            assert not dds.potcars

    def test_neutral_defect_dict_set(self):
        dds = self._generate_and_check_dds(self.prim_cdte.copy())  # fine for bulk prim input as well
        # reciprocal_density = 100/Å⁻³ for prim CdTe:
        self.kpts_nelect_nupdown_check(dds, 7, 18, 0)
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)
        self._write_and_check_dds_files(dds, potcar_spec=True)

        defect_entry = self.CdTe_defect_gen["Te_Cd_0"]
        dds = self._generate_and_check_dds(defect_entry.defect_supercell)
        # reciprocal_density = 100/Å⁻³ for CdTe supercell:
        self.kpts_nelect_nupdown_check(dds, 2, 480, 0)
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, potcar_spec=True)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

    def test_charged_defect_incar(self):
        dds = self._generate_and_check_dds(self.prim_cdte.copy(), charge_state=1)  # fine w/bulk prim
        self.kpts_nelect_nupdown_check(dds, 7, 17, 1)  # 100/Å⁻³ for prim CdTe
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

        defect_entry = self.CdTe_defect_gen["Te_Cd_0"]
        dds = self._generate_and_check_dds(defect_entry.defect_supercell.copy(), charge_state=-2)
        self.kpts_nelect_nupdown_check(dds, 2, 482, 0)  # 100/Å⁻³ for CdTe supercell
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

        defect_entry = self.CdTe_defect_gen["Te_Cd_-2"]
        dds = self._generate_and_check_dds(defect_entry.defect_supercell.copy(), charge_state=-2)
        self.kpts_nelect_nupdown_check(dds, 2, 482, 0)  # 100/Å⁻³ for CdTe supercell
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

    def test_user_settings_defect_incar(self):
        user_incar_settings = {"EDIFF": 1e-8, "EDIFFG": 0.1, "ENCUT": 720, "NCORE": 4, "KPAR": 7}

        dds = self._generate_and_check_dds(
            self.prim_cdte.copy(),
            incar_check=False,
            charge_state=1,
            user_incar_settings=user_incar_settings,
        )
        self.kpts_nelect_nupdown_check(dds, 7, 17, 1)  # reciprocal_density = 100/Å⁻³ for prim CdTe
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

        if _potcars_available():
            assert self.neutral_def_incar_min.items() <= dds.incar.items()
            assert self.hse06_incar_min.items() <= dds.incar.items()  # HSE06 by default
            for k, v in user_incar_settings.items():
                assert v == dds.incar[k]

        # non-HSE settings:
        gga_dds = self._generate_and_check_dds(
            self.prim_cdte.copy(),
            incar_check=False,
            charge_state=10,
            user_incar_settings={"LHFCALC": False},
        )
        self.kpts_nelect_nupdown_check(gga_dds, 7, 8, 0)  # reciprocal_density = 100/Å⁻³ for prim CdTe
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

        if _potcars_available():
            assert gga_dds.incar["LHFCALC"] is False
            for k in self.hse06_incar_min:
                if k not in ["LHFCALC", "GGA"]:
                    assert k not in gga_dds.incar

            assert gga_dds.incar["GGA"] == "Ps"  # GGA functional set to Ps (PBEsol) by default

    def test_bad_incar_setting(self):
        with warnings.catch_warnings(record=True) as w:
            # warnings.resetwarnings()  # neither of these should've been called previously
            dds = DefectDictSet(
                self.prim_cdte.copy(),
                user_incar_settings={"Whoops": "lol", "KPAR": 7},
                user_kpoints_settings={"reciprocal_density": 1},  # gamma only babyyy
            )
            _incar_pop = dds.incar  # get KPAR warning

        assert any(
            "Cannot find Whoops from your user_incar_settings in the list of INCAR flags"
            in str(warning.message)
            for warning in w
        )
        assert any(warning.category == BadIncarWarning for warning in w)
        assert any(
            "KPOINTS are Γ-only (i.e. only one kpoint), so KPAR is being set to 1" in str(warning.message)
            for warning in w
        )
        assert "1  # only one k-point" in dds.incar["KPAR"]  # pmg makes it lowercase and can change
        # gamma symbol

        self.kpts_nelect_nupdown_check(dds, 1, 18, 0)  # reciprocal_density = 1/Å⁻³ for prim CdTe
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

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
            self.sqs_agsbte2,
            self.N_doped_diamond_supercell,  # has unordered site symbols (C N C), so good to test
            # ordered site symbols in written POSCARs
        ]:
            dds = self._generate_and_check_dds(struct)  # fine for a bulk primitive input as well
            self._write_and_check_dds_files(dds)
            self._write_and_check_dds_files(dds, unperturbed_poscar=False)
            self._write_and_check_dds_files(dds, potcar_spec=True)  # can test potcar_spec w/neutral

            # charged_dds:
            self._generate_and_check_dds(struct, charge_state=np.random.randint(-5, 5))
            self._write_and_check_dds_files(dds)
            self._write_and_check_dds_files(dds, unperturbed_poscar=False)

            DefectDictSet(
                struct,
                user_incar_settings={"ENCUT": 350},
                user_potcar_functional="PBE_52",
                user_potcar_settings={"Cu": "Cu_pv"},
                user_kpoints_settings={"reciprocal_density": 200},
                poscar_comment="Test pop",
            )
            self._write_and_check_dds_files(dds)
            self._write_and_check_dds_files(dds, unperturbed_poscar=False)
            self._write_and_check_dds_files(dds, potcar_spec=True)  # can test potcar_spec w/neutral

    def test_file_writing_with_without_POTCARs(self):
        """
        Test the behaviour of the `DefectDictSet` attributes and
        `.write_input()` method when `POTCAR`s are and are not available.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            dds = self._generate_and_check_dds(self.ytos_bulk_supercell.copy())  # fine for bulk prim
            self._write_and_check_dds_files(dds)
            self._write_and_check_dds_files(dds, potcar_spec=True)  # can only test potcar_spec w/neutral
            self._write_and_check_dds_files(dds, unperturbed_poscar=False)
        self.kpts_nelect_nupdown_check(dds, [[2, 2, 1]], 1584, 0)
        # reciprocal_density = 100/Å⁻³ for YTOS

        if not _potcars_available():
            for test_warning_message in [
                "NUPDOWN (i.e. spin state) INCAR flag cannot be set",
                "POTCAR directory not set up with pymatgen",
            ]:
                assert any(test_warning_message in str(warning.message) for warning in w)

        # check changing charge state
        dds = self._generate_and_check_dds(self.ytos_bulk_supercell.copy(), charge_state=1)
        self.kpts_nelect_nupdown_check(dds, [[2, 2, 1]], 1583, 1)
        # reciprocal_density = 100/Å⁻³ for YTOS
        self._write_and_check_dds_files(dds, output_path="YTOS_test_dir")
        self._write_and_check_dds_files(dds, unperturbed_poscar=False)

    def _write_and_check_dds_files(self, dds, **kwargs):
        output_path = kwargs.pop("output_path", "test_pop")
        delete_dir = kwargs.pop("delete_dir", True)  # delete directory after testing?

        if (
            not kwargs.get("unperturbed_poscar", True)
            and dds.charge_state != 0
            and not _potcars_available()
        ):
            # error with charged defect and unperturbed_poscar=False
            with self.assertRaises(ValueError) as e:
                dds.write_input(output_path, **kwargs)
            assert _check_potcar_dir_not_setup_warning_error(dds, e.exception, unperturbed_poscar=False)
            return

        dds.write_input(output_path, **kwargs)

        # print(output_path)  # to help debug if tests fail
        assert os.path.exists(output_path)

        if _potcars_available() or dds.charge_state == 0:  # INCARs should be written
            # load INCAR and check it matches dds.incar
            written_incar = Incar.from_file(f"{output_path}/INCAR")
            dds_incar_without_comments = dds.incar.copy()
            dds_incar_without_comments["ICORELEVEL"] = 0
            dds_incar_without_comments["ISYM"] = 0
            dds_incar_without_comments["ALGO"] = "Normal"
            if "KPAR" in dds_incar_without_comments and isinstance(
                dds_incar_without_comments["KPAR"], str
            ):
                dds_incar_without_comments["KPAR"] = int(dds_incar_without_comments["KPAR"][0])
            dds_incar_without_comments.pop([k for k in dds.incar if k.startswith("#")][0])
            assert written_incar == dds_incar_without_comments

            with open(f"{output_path}/INCAR") as f:
                incar_lines = f.readlines()
            for comment_string in [
                "# May want to change NCORE, KPAR, AEXX, ENCUT",
                "change to all if zhegv, fexcp/f or zbrent",
                "needed if using the kumagai-oba",
                "symmetry breaking extremely likely",
            ]:
                assert any(comment_string in line for line in incar_lines)

        else:
            assert not os.path.exists(f"{output_path}/INCAR")

        if _potcars_available() and not kwargs.get("potcar_spec", False):
            written_potcar = Potcar.from_file(f"{output_path}/POTCAR")
            # assert dicts equal, as Potcar __eq__ fails due to hashing I believe
            alt_dds_potcar_dict = dds.potcar.as_dict().copy()
            if "PBE" in alt_dds_potcar_dict["functional"]:
                alt_dds_potcar_dict["functional"] = "PBE"  # new pymatgen sets PBE to PBE_54
            assert written_potcar.as_dict() in [dds.potcar.as_dict(), alt_dds_potcar_dict]
            assert len(written_potcar.symbols) == len(set(written_potcar.symbols))  # no duplicates
        elif kwargs.get("potcar_spec", False):
            with open(f"{output_path}/POTCAR.spec", encoding="utf-8") as file:
                contents = file.readlines()
            for i, line in enumerate(contents):
                assert line in [f"{dds.potcar_symbols[i]}", f"{dds.potcar_symbols[i]}\n"]
        else:
            assert not os.path.exists(f"{output_path}/POTCAR")

        written_kpoints = Kpoints.from_file(f"{output_path}/KPOINTS")  # comment not parsed by pymatgen
        with open(f"{output_path}/KPOINTS") as f:
            comment = f.readlines()[0].replace("\n", "")
        if isinstance(dds.user_kpoints_settings, dict) and dds.user_kpoints_settings.get(
            "reciprocal_density", False
        ):  # comment changed!
            assert comment == self.doped_std_kpoint_comment.replace(
                "100", str(dds.user_kpoints_settings.get("reciprocal_density"))
            )
        elif np.prod(dds.kpoints.kpts[0]) == 1:
            assert comment == self.doped_gam_kpoint_comment
        else:
            assert comment == self.doped_std_kpoint_comment

        for k in written_kpoints.as_dict():
            if k not in ["comment", "usershift", "@module", "@class"]:
                # user shift can be tuple or list and causes failure, and we use DopedKpoints not Kpoints
                assert written_kpoints.as_dict()[k] == dds.kpoints.as_dict()[k]

        if kwargs.get("unperturbed_poscar", True):
            written_poscar = Poscar.from_file(f"{output_path}/POSCAR")
            # `get_defect_name_from_entry(relaxed=True)` causes `Defect` initialisation which adds oxi
            # states to `structure` (`defect_entry.bulk_supercell` in this case), which is used in our
            # YTOS defect gen tests (to test periodicity breaking warning), so remove these for comparison:
            written_poscar.structure.remove_oxidation_states()
            dds.structure.remove_oxidation_states()
            assert written_poscar.structure == dds.structure
            # no duplicates:
            assert len(written_poscar.site_symbols) == len(set(written_poscar.site_symbols))
        else:
            assert not os.path.exists(f"{output_path}/POSCAR")

        if delete_dir:
            if_present_rm(output_path)


class DefectRelaxSetTest(unittest.TestCase):
    def setUp(self):
        self.dds_test = DefectDictSetTest()
        self.dds_test.setUp()  # get attributes from DefectDictSetTest
        DefectDictSetTest.setUp(self)  # get attributes from DefectDictSetTest

        self.CdTe_defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/CdTe_defect_gen.json")
        self.CdTe_custom_test_incar_settings = {"ENCUT": 350, "NCORE": 10, "LCHARG": False}

    def tearDown(self):
        self.dds_test.tearDown()  # use tearDown from DefectDictSetTest
        if_present_rm("test_dir")
        if_present_rm("CdTe_bulk")

    def _general_defect_relax_set_check(self, defect_relax_set, **kwargs):
        dds_test_list = [
            (defect_relax_set.vasp_gam, "vasp_gam"),
            (defect_relax_set.vasp_std, "vasp_std"),
            (defect_relax_set.vasp_nkred_std, "vasp_nkred_std"),
            (defect_relax_set.vasp_ncl, "vasp_ncl"),
        ]
        dds_bulk_test_list = [
            (defect_relax_set.bulk_vasp_gam, "bulk_vasp_gam"),
            (defect_relax_set.bulk_vasp_std, "bulk_vasp_std"),
            (defect_relax_set.bulk_vasp_ncl, "bulk_vasp_ncl"),
        ]
        if _potcars_available():  # needed because bulk NKRED pulls NKRED values from defect nkred
            # std INCAR to be more computationally efficient
            dds_bulk_test_list.append((defect_relax_set.bulk_vasp_nkred_std, "bulk_vasp_nkred_std"))

        def _check_drs_dds_attribute_transfer(parent_drs, child_dds):
            child_incar_settings = child_dds.user_incar_settings.copy()
            if "KPAR" in child_incar_settings and "KPAR" not in parent_drs.user_incar_settings:
                assert child_incar_settings.pop("KPAR") == 2
                assert parent_drs.vasp_std
            if "LSORBIT" in child_incar_settings and "LSORBIT" not in parent_drs.user_incar_settings:
                assert child_incar_settings.pop("LSORBIT") is True
                for k, v in singleshot_incar_settings.items():
                    assert child_incar_settings.pop(k) == v
                assert parent_drs.soc
            if any("NKRED" in k for k in child_incar_settings) and all(
                "NKRED" not in k for k in parent_drs.user_incar_settings
            ):
                for k in list(child_incar_settings.keys()):
                    if "NKRED" in k:
                        assert child_incar_settings.pop(k) in [2, 3]
                assert parent_drs.vasp_nkred_std
            if "NSW" in child_incar_settings and "NSW" not in parent_drs.user_incar_settings:
                for k, v in singleshot_incar_settings.items():  # bulk singleshots
                    assert child_incar_settings.pop(k) == v

            assert parent_drs.user_incar_settings == child_incar_settings
            assert parent_drs.user_potcar_functional == child_dds.user_potcar_functional or str(
                parent_drs.user_potcar_functional[:3]
            ) == str(
                child_dds.user_potcar_functional[:3]
            )  # if PBE_52 set but not available, defaults to PBE
            assert parent_drs.user_potcar_settings == child_dds.user_potcar_settings
            if isinstance(child_dds.user_kpoints_settings, (DopedKpoints, Kpoints)):
                assert (
                    child_dds.user_kpoints_settings.as_dict()
                    == DopedKpoints()
                    .from_dict(
                        {
                            "comment": "Γ-only KPOINTS from doped",
                            "generation_style": "Gamma",
                        }
                    )
                    .as_dict()
                ) or (
                    child_dds.user_kpoints_settings.as_dict()
                    == Kpoints()
                    .from_dict(
                        {
                            "comment": "Γ-only KPOINTS from doped",
                            "generation_style": "Gamma",
                        }
                    )
                    .as_dict()
                )
            else:
                assert parent_drs.user_kpoints_settings == child_dds.user_kpoints_settings

        for defect_dict_set, type in dds_test_list:
            if defect_dict_set is not None:
                print(f"Testing {defect_relax_set.defect_entry.name}, {type}")
                self.dds_test._check_dds(
                    defect_dict_set,
                    defect_relax_set.defect_supercell,
                    charge_state=defect_relax_set.charge_state,
                    **kwargs,
                )
                self.dds_test._write_and_check_dds_files(defect_dict_set)
                self.dds_test._write_and_check_dds_files(
                    defect_dict_set, output_path=f"{defect_relax_set.defect_entry.name}"
                )
                self.dds_test._write_and_check_dds_files(defect_dict_set, unperturbed_poscar=False)
                if defect_relax_set.charge_state == 0:
                    self.dds_test._write_and_check_dds_files(defect_dict_set, potcar_spec=True)

                _check_drs_dds_attribute_transfer(defect_relax_set, defect_dict_set)

        for defect_dict_set, type in dds_bulk_test_list:
            if defect_dict_set is not None:
                print(f"Testing {defect_relax_set.defect_entry.name}, {type}")
                self.dds_test._check_dds(
                    defect_dict_set, defect_relax_set.bulk_supercell, charge_state=0, **kwargs
                )
                self.dds_test._write_and_check_dds_files(defect_dict_set)
                self.dds_test._write_and_check_dds_files(defect_dict_set, unperturbed_poscar=False)
                self.dds_test._write_and_check_dds_files(defect_dict_set, potcar_spec=True)

                _check_drs_dds_attribute_transfer(defect_relax_set, defect_dict_set)

    def test_initialisation_and_writing(self):
        """
        Test the initialisation of DefectRelaxSet for a range of
        `DefectEntry`s.
        """
        if not self.heavy_tests:
            return

        def _check_drs_defect_entry_attribute_transfer(parent_drs, input_defect_entry):
            assert parent_drs.defect_entry == input_defect_entry
            assert parent_drs.defect_supercell == input_defect_entry.defect_supercell
            assert parent_drs.charge_state == input_defect_entry.charge_state
            assert parent_drs.bulk_supercell == input_defect_entry.bulk_supercell

        # test initialising DefectRelaxSet with our generation-tests materials, and writing files to disk
        defect_gen_test_list = [
            (self.CdTe_defect_gen, "CdTe defect_gen"),
            (DefectsGenerator(self.sqs_agsbte2), "SQS AgSbTe2 defect_gen"),
        ]
        for defect_gen_name in [
            "ytos_defect_gen",
            "ytos_defect_gen_supercell",
            "lmno_defect_gen",
            "cu_defect_gen",
            "agcu_defect_gen",
            "cd_i_supercell_defect_gen",
            "N_diamond_defect_gen",  # input structure for this is unordered (but this checks
            # that POSCAR site symbols output should be ordered)
        ]:
            defect_gen_test_list.append(
                (DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json"), defect_gen_name)
            )

        for defect_gen, defect_gen_name in defect_gen_test_list:
            print(f"Initialising and testing: {defect_gen_name}")
            # randomly choose a defect entry from the defect_gen dict:
            defect_entry = random.choice(list(defect_gen.values()))

            print(f"Randomly testing {defect_entry.name}")
            drs = DefectRelaxSet(defect_entry)
            self._general_defect_relax_set_check(drs)

            _check_drs_defect_entry_attribute_transfer(drs, defect_entry)

            custom_drs = DefectRelaxSet(
                defect_entry,
                user_incar_settings={"ENCUT": 350},
                user_potcar_functional="PBE_52",
                user_potcar_settings={"Cu": "Cu_pv"},
                user_kpoints_settings={"reciprocal_density": 200},
                poscar_comment="Test pop",
            )
            self._general_defect_relax_set_check(custom_drs)
            _check_drs_defect_entry_attribute_transfer(custom_drs, defect_entry)

    def test_vasp_gam_folder_writing(self):
        """
        Test DefectRelaxSet write_all() methods; writing/not writing vasp_gam
        POSCARs.
        """
        for defect_gen_name in [
            "CdTe_defect_gen",
            "ytos_defect_gen",
            "lmno_defect_gen",
            "cu_defect_gen",
        ]:
            print(f"Testing: {defect_gen_name}")
            defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json")
            defect_entry = random.choice(list(defect_gen.values()))
            print(f"Testing DefectRelaxSet with: {defect_entry.name}")
            drs = DefectRelaxSet(defect_entry)
            drs.write_all("test_dir")

            assert os.path.exists("test_dir")
            assert not os.path.exists("test_dir/vasp_gam")
            assert not os.path.exists("test_dir/vasp_gam/POSCAR")
            assert not os.path.exists("test_dir/vasp_std/POSCAR")
            assert os.path.exists("test_dir/vasp_std/INCAR")
            if_present_rm("test_dir")

            drs = DefectRelaxSet(defect_entry)
            drs.write_all("test_dir", vasp_gam=True)

            assert os.path.exists("test_dir")
            assert os.path.exists("test_dir/vasp_gam")
            assert os.path.exists("test_dir/vasp_gam/POSCAR")
            assert not os.path.exists("test_dir/vasp_std/POSCAR")
            assert os.path.exists("test_dir/vasp_std/INCAR")
            if_present_rm("test_dir")

            drs = DefectRelaxSet(defect_entry)
            drs.write_all("test_dir", vasp_gam=True, unperturbed_poscar=True)

            assert os.path.exists("test_dir")
            assert os.path.exists("test_dir/vasp_gam")
            assert os.path.exists("test_dir/vasp_gam/POSCAR")
            assert os.path.exists("test_dir/vasp_std/POSCAR")
            assert os.path.exists("test_dir/vasp_std/INCAR")
            if_present_rm("test_dir")

    def test_poscar_comments(self):
        drs = DefectRelaxSet(self.CdTe_defect_gen["Cd_i_C3v_0"])
        drs.write_all("test_dir", unperturbed_poscar=True)
        poscar = Poscar.from_file("test_dir/vasp_std/POSCAR")
        assert poscar.comment == "Cd_i_C3v ~[0.5417,0.5417,0.5417] 0"

        poscar = Poscar.from_file("test_dir/vasp_nkred_std/POSCAR")
        assert poscar.comment == "Cd_i_C3v ~[0.5417,0.5417,0.5417] 0"

        poscar = Poscar.from_file("test_dir/vasp_ncl/POSCAR")
        assert poscar.comment == "Cd_i_C3v ~[0.5417,0.5417,0.5417] 0"

        assert not os.path.exists("test_dir/CdTe_bulk")
        drs.write_all("test_dir", bulk=True)
        assert not os.path.exists("test_dir/CdTe_bulk")

        poscar = Poscar.from_file("CdTe_bulk/vasp_ncl/POSCAR")
        assert poscar.comment == "Cd27 Te27 - Bulk"

        drs = DefectRelaxSet(self.CdTe_defect_gen["Cd_i_C3v_0"], poscar_comment="Test pop")
        drs.write_all("test_dir", unperturbed_poscar=True)
        poscar = Poscar.from_file("test_dir/vasp_std/POSCAR")
        assert poscar.comment == "Test pop"

        drs = DefectRelaxSet(self.CdTe_defect_gen["v_Cd_-2"])
        drs.write_all("test_dir", unperturbed_poscar=True)
        poscar = Poscar.from_file("test_dir/vasp_std/POSCAR")
        assert poscar.comment == "v_Cd ~[0.3333,0.3333,0.3333] -2"

        poscar = Poscar.from_file("test_dir/vasp_nkred_std/POSCAR")
        assert poscar.comment == "v_Cd ~[0.3333,0.3333,0.3333] -2"

        poscar = Poscar.from_file("test_dir/vasp_ncl/POSCAR")
        assert poscar.comment == "v_Cd ~[0.3333,0.3333,0.3333] -2"

        assert not os.path.exists("test_dir/CdTe_bulk")
        drs.write_all("test_dir", bulk=True)
        assert not os.path.exists("test_dir/CdTe_bulk")

        poscar = Poscar.from_file("CdTe_bulk/vasp_ncl/POSCAR")
        assert poscar.comment == "Cd27 Te27 - Bulk"

    def test_default_kpoints_soc_handling(self):
        """
        Check vasp_std created when necessary, and not when vasp_gam converged.
        """
        defect_gen_test_list = [
            (self.CdTe_defect_gen, "CdTe defect_gen"),
        ]
        for defect_gen_name in [
            "ytos_defect_gen",
            "lmno_defect_gen",
            "agcu_defect_gen",
            "cd_i_supercell_defect_gen",
        ]:
            defect_gen_test_list.append(
                (DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json"), defect_gen_name)
            )

        for defect_gen, defect_gen_name in defect_gen_test_list:
            print(f"Testing:{defect_gen_name}")
            # randomly choose 10 defect entries from the defect_gen dict:
            defect_entries = random.sample(list(defect_gen.values()), min(10, len(defect_gen)))

            for defect_entry in defect_entries:
                print(f"Randomly testing {defect_entry.name}")
                drs = DefectRelaxSet(defect_entry)
                if defect_gen_name in [
                    "CdTe defect_gen",
                    "ytos_defect_gen",
                    "agcu_defect_gen",
                    "cd_i_supercell_defect_gen",
                ]:
                    assert drs.vasp_std
                    assert drs.bulk_vasp_std
                    assert drs.vasp_nkred_std

                    if (
                        _potcars_available()
                    ):  # needed because bulk NKRED pulls NKRED values from defect nkred std INCAR to be
                        # more computationally efficient
                        assert drs.bulk_vasp_nkred_std

                    assert drs.vasp_ncl
                    assert drs.bulk_vasp_ncl

                else:  # no SOC for LMNO  # vasp_gam test
                    assert drs.vasp_std
                    assert drs.bulk_vasp_std
                    assert drs.vasp_nkred_std

                    if _potcars_available():
                        assert drs.bulk_vasp_nkred_std

                    assert not drs.vasp_ncl
                    assert not drs.bulk_vasp_ncl

        # Test manually turning off SOC and making vasp_gam converged:
        defect_entries = random.sample(list(self.CdTe_defect_gen.values()), 5)

        for defect_entry in defect_entries:
            print(f"Randomly testing {defect_entry.name}")
            drs = DefectRelaxSet(defect_entry, soc=False, user_kpoints_settings={"reciprocal_density": 50})
            assert not drs.vasp_std
            assert not drs.bulk_vasp_std
            assert not drs.vasp_nkred_std

            if (
                _potcars_available()
            ):  # needed because bulk NKRED pulls NKRED values from defect nkred std INCAR to be more
                # computationally efficient
                assert not drs.bulk_vasp_nkred_std

            assert not drs.vasp_ncl
            assert not drs.bulk_vasp_ncl

        # Test manually turning _on_ SOC and making vasp_gam _not_ converged:
        defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/lmno_defect_gen.json")
        defect_entries = random.sample(list(defect_gen.values()), min(5, len(defect_gen)))

        for defect_entry in defect_entries:
            print(f"Randomly testing {defect_entry.name}")
            drs = DefectRelaxSet(defect_entry, soc=True, user_kpoints_settings={"reciprocal_density": 200})
            assert drs.vasp_std
            assert drs.bulk_vasp_std
            assert drs.vasp_nkred_std

            if (
                _potcars_available()
            ):  # needed because bulk NKRED pulls NKRED values from defect nkred std INCAR to be more
                # computationally efficient
                assert drs.bulk_vasp_nkred_std

            assert drs.vasp_ncl
            assert drs.bulk_vasp_ncl


class DefectsSetTest(unittest.TestCase):
    def setUp(self):
        self.dds_test = DefectDictSetTest()
        self.dds_test.setUp()  # get attributes from DefectDictSetTest
        DefectDictSetTest.setUp(self)  # get attributes from DefectDictSetTest

        self.CdTe_defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/CdTe_defect_gen.json")

        # Note this is different to above: (for testing against pre-generated input files with these
        # settings):
        self.CdTe_custom_test_incar_settings = {"ENCUT": 350, "NCORE": 10, "LVHAR": False, "ALGO": "All"}

        # Get the current locale setting
        self.original_locale = locale.getlocale(locale.LC_CTYPE)  # should be UTF-8

    def tearDown(self):
        # reset locale:
        locale.setlocale(locale.LC_CTYPE, self.original_locale)  # should be UTF-8

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
        check_incar=None,
        single_defect_dir=False,
        bulk=True,
    ):
        if data_dir is None:
            data_dir = self.CdTe_data_dir

        if check_incar is None:
            check_incar = _potcars_available()

        if single_defect_dir:
            folders = [
                "",
            ]

        else:
            folders = [
                folder
                for folder in os.listdir(data_dir)
                if os.path.isdir(f"{data_dir}/{folder}") and ("bulk" not in folder.lower() or bulk)
            ]

        for folder in folders:
            print(f"{generated_dir}/{folder}")  # for debugging
            assert os.path.exists(f"{generated_dir}/{folder}")
            assert os.path.exists(f"{generated_dir}/{folder}/{vasp_type}")

            # load the Incar, Poscar and Kpoints and check it matches the previous:
            if check_incar:
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
                assert len(poscar.site_symbols) == len(set(poscar.site_symbols))

            if check_potcar_spec:
                with open(f"{generated_dir}/{folder}/{vasp_type}/POTCAR.spec", encoding="utf-8") as file:
                    contents = file.readlines()
                    assert contents[0] in ["Cd", "Cd\n"]
                    assert contents[1] in ["Te", "Te\n"]
                    if "Se" in folder:
                        assert contents[2] in ["Se", "Se\n"]

            with open(f"{generated_dir}/{folder}/{vasp_type}/KPOINTS") as f:
                print(f.read())  # for debugging
            with open(f"{data_dir}/{folder}/{vasp_type}/KPOINTS") as f:
                print(f.read())  # for debugging
            test_kpoints = Kpoints.from_file(f"{data_dir}/{folder}/{vasp_type}/KPOINTS")
            kpoints = Kpoints.from_file(f"{generated_dir}/{folder}/{vasp_type}/KPOINTS")
            assert test_kpoints.as_dict() == kpoints.as_dict()

    def _general_defects_set_check(self, defects_set, **kwargs):
        # TODO: Use function above here
        for entry_name, defect_relax_set in defects_set.defect_sets.items():
            print(f"Testing {entry_name} DefectRelaxSet")
            # check bulk vasp attributes same as DefectsSet:
            assert defect_relax_set.bulk_vasp_ncl.structure == defects_set.bulk_vasp_ncl.structure
            assert defect_relax_set.bulk_vasp_ncl.incar == defects_set.bulk_vasp_ncl.incar
            assert defect_relax_set.bulk_vasp_ncl.kpoints == defects_set.bulk_vasp_ncl.kpoints

            dds_test_list = [
                (defect_relax_set.vasp_gam, "vasp_gam"),
                (defect_relax_set.bulk_vasp_gam, "bulk_vasp_gam"),
                (defect_relax_set.vasp_std, "vasp_std"),
                (defect_relax_set.bulk_vasp_std, "bulk_vasp_std"),
                (defect_relax_set.vasp_nkred_std, "vasp_nkred_std"),
                (defect_relax_set.vasp_ncl, "vasp_ncl"),
                (defect_relax_set.bulk_vasp_ncl, "bulk_vasp_ncl"),
            ]
            if _potcars_available():  # needed because bulk NKRED pulls NKRED values from defect nkred
                # std INCAR to be more computationally efficient
                dds_test_list.append((defect_relax_set.bulk_vasp_nkred_std, "bulk_vasp_nkred_std"))

            for defect_dict_set, name in dds_test_list:
                print(f"Testing {name}")
                try:
                    self.dds_test._check_dds(
                        defect_dict_set,
                        defect_relax_set.defect_supercell,
                        charge_state=defect_relax_set.charge_state,
                        **kwargs,
                    )
                except AssertionError:  # try bulk structure
                    print("Defect DefectDictSet test failed, trialling bulk DefectDictSet")
                    self.dds_test._check_dds(
                        defect_dict_set, defect_relax_set.bulk_supercell, charge_state=0, **kwargs
                    )

    def test_CdTe_files(self):
        if not self.heavy_tests:
            return

        CdTe_se_defect_gen = DefectsGenerator(self.prim_cdte, extrinsic="Se")
        defects_set = DefectsSet(
            CdTe_se_defect_gen,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )
        self._general_defects_set_check(defects_set)

        if _potcars_available():
            defects_set.write_files(potcar_spec=True)  # unperturbed_poscar=False by default

            # test no (unperturbed) POSCAR files written:
            for folder in os.listdir("."):
                if os.path.isdir(folder) and "bulk" not in folder:
                    for subfolder in os.listdir(folder):
                        assert not os.path.exists(f"{folder}/{subfolder}/POSCAR")

        else:
            with self.assertRaises(ValueError):
                defects_set.write_files(
                    potcar_spec=True
                )  # INCAR ValueError for charged defects if POTCARs not
                # available and unperturbed_poscar=False
            defects_set.write_files(potcar_spec=True, unperturbed_poscar=True)

        # test no vasp_gam files written:
        for folder in os.listdir("."):
            assert not os.path.exists(f"{folder}/vasp_gam")

        defects_set.write_files(potcar_spec=True, unperturbed_poscar=True, vasp_gam=True)

        bulk_supercell = Structure.from_file("CdTe_bulk/vasp_ncl/POSCAR")
        structure_matcher = StructureMatcher(
            comparator=ElementComparator(), primitive_cell=False
        )  # ignore oxidation states
        assert structure_matcher.fit(bulk_supercell, self.CdTe_defect_gen.bulk_supercell)
        # check_generated_vasp_inputs also checks bulk folders

        assert os.path.exists("CdTe_defects_generator.json")
        CdTe_se_defect_gen.to_json("test_CdTe_defects_generator.json")
        assert filecmp.cmp("CdTe_defects_generator.json", "test_CdTe_defects_generator.json")

        # assert that the same folders in self.CdTe_data_dir are present in the current directory
        print("Checking vasp_gam files")
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=False)  # tests
        # vasp_gam
        print("Checking vasp_std files")
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=False, bulk=False)  # vasp_std

        # test vasp_nkred_std: same as vasp_std except for NKRED
        print("Checking NKRED vasp_std files")
        for folder in os.listdir("."):
            if os.path.isdir(f"{folder}/vasp_std"):
                assert filecmp.cmp(f"{folder}/vasp_nkred_std/KPOINTS", f"{folder}/vasp_std/KPOINTS")
                if _potcars_available():
                    assert filecmp.cmp(
                        f"{folder}/vasp_nkred_std/POTCAR.spec", f"{folder}/vasp_std/POTCAR.spec"
                    )
                    nkred_incar = Incar.from_file(f"{folder}/vasp_nkred_std/INCAR")
                    std_incar = Incar.from_file(f"{folder}/vasp_std/INCAR")
                    nkred_incar.pop("NKRED", None)
                    assert nkred_incar == std_incar

        print("Checking vasp_ncl files")
        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=False, bulk=True)  # vasp_ncl

        # test unperturbed POSCARs and all bulk
        print("Checking unperturbed POSCARs and all bulk")
        defects_set = DefectsSet(
            self.CdTe_defect_gen,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )
        defects_set.write_files(potcar_spec=True, unperturbed_poscar=True, bulk="all", vasp_gam=True)
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=True)  # tests vasp_gam
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=True, bulk=True)  # vasp_std
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

        # test custom POTCAR and KPOINTS choices (INCAR already tested): also tests dictionary input to
        # DefectsSet
        self.tearDown()
        defects_set = DefectsSet(
            {k: v for k, v in self.CdTe_defect_gen.items() if "v_Te" in k},
            user_potcar_settings={"Cd": "Cd_sv_GW", "Te": "Te_GW"},
            user_kpoints_settings={"reciprocal_density": 500},
        )

        if _potcars_available():
            defects_set.write_files(potcar_spec=True, vasp_gam=True)  # vasp_gam to test POTCAR.spec
        else:
            with self.assertRaises(ValueError):
                defects_set.write_files(potcar_spec=True, vasp_gam=True)  # INCAR ValueError for charged
                # defects if POTCARs not available and unperturbed_poscar=False
            defects_set.write_files(potcar_spec=True, vasp_gam=True, unperturbed_poscar=True)

        for folder in os.listdir("."):
            if os.path.isdir(f"{folder}/vasp_gam"):
                with open(f"{folder}/vasp_gam/POTCAR.spec", encoding="utf-8") as file:
                    contents = file.readlines()
                    assert contents[0] in ["Cd_sv_GW", "Cd_sv_GW\n"]
                    assert contents[1] in ["Te_GW", "Te_GW\n"]

                for subfolder in ["vasp_std", "vasp_nkred_std", "vasp_ncl"]:
                    kpoints = Kpoints.from_file(f"{folder}/{subfolder}/KPOINTS")
                    assert kpoints.kpts == [[4, 4, 4]]  # 4x4x4 with 54-atom 3x3x3 prim supercell,
                    # 3x3x3 with 64-atom 2x2x2 conv supercell

    def test_write_files_single_defect_entry(self):
        single_defect_entry = self.CdTe_defect_gen["Cd_i_C3v_+2"]
        defects_set = DefectsSet(
            single_defect_entry,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )
        defects_set.write_files(potcar_spec=True, vasp_gam=True, unperturbed_poscar=True)

        # assert that the same folders in self.CdTe_data_dir are present in the current directory
        self.check_generated_vasp_inputs(  # tests vasp_gam
            generated_dir="Cd_i_C3v_+2",
            data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
            check_potcar_spec=True,
            single_defect_dir=True,
        )
        self.check_generated_vasp_inputs(  # vasp_std
            generated_dir="Cd_i_C3v_+2",
            data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
            vasp_type="vasp_std",
            check_poscar=True,
            check_potcar_spec=True,
            single_defect_dir=True,
        )
        self.check_generated_vasp_inputs(  # vasp_ncl
            generated_dir="Cd_i_C3v_+2",
            data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
            vasp_type="vasp_ncl",
            check_poscar=True,
            check_potcar_spec=True,
            single_defect_dir=True,
        )

        # assert only +2 directory written:
        assert not os.path.exists("Cd_i_C3v_0")

        # check passing of kwargs on to DefectRelaxSet/DefectDictSet:
        defects_set = DefectsSet(single_defect_entry, poscar_comment="test pop")
        assert defects_set.defect_sets["Cd_i_C3v_+2"].poscar_comment == "test pop"
        assert defects_set.defect_sets["Cd_i_C3v_+2"].vasp_gam.poscar_comment == "test pop"
        assert defects_set.defect_sets["Cd_i_C3v_+2"].vasp_gam.poscar.comment == "test pop"

    def test_write_files_ASCII_encoding(self):
        """
        Test writing VASP input files for a system that's not on UTF-8
        encoding.

        Weirdly seems to be the case on some old HPCs/Windows systems.
        """
        with contextlib.suppress(locale.Error):  # not supported on GH Actions
            # Temporarily set the locale to ASCII/latin encoding (doesn't support emojis or "Γ"):
            locale.setlocale(locale.LC_CTYPE, "en_US.US-ASCII")

            single_defect_entry = self.CdTe_defect_gen["Cd_i_C3v_+2"]
            defects_set = DefectsSet(
                single_defect_entry,
                user_incar_settings=self.CdTe_custom_test_incar_settings,
            )
            defects_set.write_files(potcar_spec=True, vasp_gam=True, unperturbed_poscar=True)
            locale.setlocale(locale.LC_CTYPE, self.original_locale)  # should be UTF-8

            # assert that the same folders in self.CdTe_data_dir are present in the current directory
            self.check_generated_vasp_inputs(  # tests vasp_gam
                generated_dir="Cd_i_C3v_+2",
                data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
                check_potcar_spec=True,
                single_defect_dir=True,
            )
            self.check_generated_vasp_inputs(  # vasp_std
                generated_dir="Cd_i_C3v_+2",
                data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
                vasp_type="vasp_std",
                check_poscar=True,
                check_potcar_spec=True,
                single_defect_dir=True,
            )
            self.check_generated_vasp_inputs(  # vasp_ncl
                generated_dir="Cd_i_C3v_+2",
                data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
                vasp_type="vasp_ncl",
                check_poscar=True,
                check_potcar_spec=True,
                single_defect_dir=True,
            )

            # assert only +2 directory written:
            assert not os.path.exists("Cd_i_C3v_0")

    def test_write_files_defect_entry_list(self):
        defect_entry_list = [
            defect_entry
            for defect_species, defect_entry in self.CdTe_defect_gen.items()
            if "Cd_i" in defect_species
        ]
        defects_set = DefectsSet(
            defect_entry_list,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )

        if _potcars_available():
            defects_set.write_files(potcar_spec=True)  # unperturbed_poscar=False by default
        else:
            with self.assertRaises(ValueError):
                defects_set.write_files(potcar_spec=True)  # INCAR ValueError for charged defects if
                # POTCARs not available and unperturbed_poscar=False
            defects_set.write_files(potcar_spec=True, unperturbed_poscar=True)

        for defect_entry in defect_entry_list:
            for vasp_type in ["vasp_nkred_std", "vasp_std", "vasp_ncl"]:  # no vasp_gam by default
                self.check_generated_vasp_inputs(
                    generated_dir=defect_entry.name,
                    data_dir=f"{self.CdTe_data_dir}/{defect_entry.name}",
                    vasp_type=vasp_type,
                    single_defect_dir=True,
                    check_poscar=not _potcars_available(),
                    check_potcar_spec=True,
                )

    def test_initialise_and_write_all_defect_gens(self):
        """
        Test initialising DefectsSet with our generation-tests materials, and
        writing files to disk.
        """

        def initialise_and_write_files(defect_gen_json):
            print(f"Initialising and testing: {defect_gen_json}")
            defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_json}.json")
            defects_set = DefectsSet(defect_gen)
            if _potcars_available():
                defects_set.write_files()
            else:
                with self.assertRaises(ValueError):
                    defects_set.write_files()  # INCAR ValueError for charged defects if POTCARs not
                    # available
                defects_set.write_files(unperturbed_poscar=True)

            del defects_set  # delete python objects to ensure memory released
            del defect_gen  # delete python objects to ensure memory released

        defect_gens_to_test = [
            "ytos_defect_gen",
            "lmno_defect_gen",
            "cu_defect_gen",
        ]
        if self.heavy_tests:  # uses too much memory on GH Actions
            defect_gens_to_test.extend(
                [
                    "ytos_defect_gen_supercell",
                    "agcu_defect_gen",
                    "cd_i_supercell_defect_gen",
                ]
            )

        for defect_gen_name in defect_gens_to_test:
            test_thread = Thread(target=initialise_and_write_files, args=(defect_gen_name,))
            test_thread.start()
            test_thread.join(timeout=300)  # Timeout set to 5 minutes

            if test_thread.is_alive():
                print(f"Test for {defect_gen_name} timed out!")

            self.tearDown()  # delete generated folders each time
        self.tearDown()


if __name__ == "__main__":
    unittest.main()
