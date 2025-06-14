"""
Tests for the `doped.vasp` module.
"""

import contextlib
import filecmp
import gzip
import locale
import os
import random
import unittest
import warnings
from threading import Thread

import numpy as np
import pytest
from ase.build import bulk, make_supercell
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import ElementComparator
from pymatgen.io.vasp.inputs import BadIncarWarning, Incar, Kpoints, Poscar, Potcar
from test_generation import _compare_attributes, if_present_rm

from doped.generation import DefectsGenerator
from doped.utils.efficiency import Structure, StructureMatcher
from doped.vasp import (
    DefectDictSet,
    DefectRelaxSet,
    DefectsSet,
    DopedDictSet,
    DopedKpoints,
    _test_potcar_functional_choice,
    default_defect_relax_set,
    default_potcar_dict,
    singleshot_incar_settings,
)


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


def _check_potcar_dir_not_setup_warning_error(dds, message, poscar=True):
    if poscar and dds.charge_state != 0:
        ending_string = "so only '(unperturbed) `POSCAR` and `KPOINTS` files will be generated."

    elif not poscar and dds.charge_state != 0:  # only KPOINTS can be written so no good
        ending_string = "so no input files will be generated."

    else:
        ending_string = "so `POTCAR` files will not be generated."

    return all(x in str(message) for x in ["POTCAR directory not set up with pymatgen", ending_string])


def _check_no_potcar_available_warning_error(message):
    return "Set PMG_VASP_PSP_DIR=<directory-path> in .pmgrc.yaml (needed to find POTCARs)" in str(message)


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
        self.agcu = Structure.from_ase_atoms(atoms)
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

            potcar_settings = default_potcar_dict.copy()
            potcar_settings.update(dds.user_potcar_settings or {})
            assert set(dds.potcar.as_dict()["symbols"]) == {
                potcar_settings[el_symbol] for el_symbol in dds.structure.symbol_set
            }
        else:
            with pytest.raises(ValueError) as e:
                _test_pop = dds.potcar
            assert _check_no_potcar_available_warning_error(e.value)

            if dds.charge_state != 0:
                with pytest.raises(ValueError) as e:
                    _test_pop = dds.incar
                assert _check_nelect_nupdown_error(e.value)
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

        # test __repr__ info:
        assert all(
            i in dds.__repr__()
            for i in [
                f"doped DefectDictSet with supercell composition {struct.composition}. "
                "Available attributes",
                "nelect",
                "incar",
                "Available methods",
                "write_input",
            ]
        )

    def _check_potcar_nupdown_dds_warnings(self, w, dds):
        print("Testing:", [str(warning.message) for warning in w])  # for debugging
        assert any(_check_potcar_dir_not_setup_warning_error(dds, warning.message) for warning in w)
        assert any(_check_nupdown_neutral_cell_warning(warning.message) for warning in w)
        assert any(_check_no_potcar_available_warning_error(warning.message) for warning in w)

    def _check_dds_incar(self, dds, struct):
        # these are the base INCAR settings:
        expected_incar_settings = self.neutral_def_incar_min.copy()
        expected_incar_settings.update(self.hse06_incar_min)  # HSE06 by default
        expected_incar_settings.update(dds.user_incar_settings)
        expected_incar_settings = {k: v for k, v in expected_incar_settings.items() if v is not None}

        default_relax_settings = default_defect_relax_set["INCAR"].copy()
        default_relax_settings.update(dds.user_incar_settings)
        default_relax_settings = {k: v for k, v in default_relax_settings.items() if v is not None}
        print(dds.user_incar_settings, dds.incar, default_relax_settings)  # for debugging

        if dds.incar.get("NSW") == 0:
            assert "EDIFFG" not in dds.incar
            default_relax_settings.pop("EDIFFG", None)  # EDIFFG not set w/static calculations
            default_relax_settings.pop("IBRION", None)  # IBRION not set w/static calculations now
            default_relax_settings.pop("ISIF", None)  # ISIF = 2 not set w/static calculations now
            default_relax_settings.pop("NSW", None)
            assert "ISIF" not in dds.incar
            expected_incar_settings.pop("ISIF", None)  # ISIF = 2 not set w/static calculations now

        print("Comparing expected base INCAR settings with dds.incar")
        assert expected_incar_settings.items() <= dds.incar.items()

        if dds.incar.get("NSW", 0) > 0:
            assert dds.incar["EDIFF"] == 1e-5  # default EDIFF
        else:
            assert dds.incar["EDIFF"] == 1e-6  # hard set to 1e-6 for static calculations

        for k, v in default_relax_settings.items():
            if k == "GGA" and dds.incar.get("LHFCALC", False):
                assert dds.incar[k] in ["Pe", "PE"]  # auto set to PE if LHFCALC = True
                continue

            assert k in dds.incar
            print(k, v, dds.incar[k])  # for debugging
            if isinstance(v, str):  # DictSet converts all ``user_incar_settings`` strings to capitalised
                # lowercase, but not if set as config_dict["INCAR"]
                try:
                    val = float(v[:2])
                    if k in dds.user_incar_settings:  # has been overwritten
                        assert val != dds.incar[k]
                    else:
                        assert val == dds.incar[k]
                except (ValueError, AssertionError):
                    if k in dds.user_incar_settings:
                        assert dds.user_incar_settings[k] in [v, v.lower().capitalize(), bool(v[:5])]
                    else:
                        assert dds.incar[k] in [v, v.lower().capitalize(), bool(v[:5])]
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
            dds.write_input("test_pop", poscar=False)

        self._check_potcar_nupdown_dds_warnings(w, dds)

    def _check_dds(self, dds, struct, **kwargs):
        # INCARs only generated for charged defects when POTCARs available:
        if _potcars_available():
            self._general_defect_dict_set_check(  # also tests dds.charge_state
                dds, struct, incar_check=kwargs.pop("incar_check", True), **kwargs
            )
        else:
            if kwargs.pop("incar_check", True) and dds.charge_state != 0:  # charged defect INCAR
                with pytest.raises(ValueError) as e:
                    self._general_defect_dict_set_check(  # also tests dds.charge_state
                        dds, struct, incar_check=kwargs.pop("incar_check", True), **kwargs
                    )
                _check_nelect_nupdown_error(e.value)
            self._general_defect_dict_set_check(  # also tests dds.charge_state
                dds, struct, incar_check=kwargs.pop("incar_check", False), **kwargs
            )

    def _generate_and_check_dds(self, struct, incar_check=True, **dds_kwargs):
        with warnings.catch_warnings(record=True) as w:
            dds = DefectDictSet(struct, **dds_kwargs)  # fine for bulk prim input as well
        print([str(warning.message) for warning in w])  # for debugging
        self._check_dds(dds, struct, incar_check=incar_check, **dds_kwargs)
        return dds

    def kpts_nelect_nupdown_check(self, dds, kpt, nelect, nupdown):
        if isinstance(kpt, int):
            assert dds.kpoints.kpts == [(kpt, kpt, kpt)]
        else:
            assert dds.kpoints.kpts == kpt
        if _potcars_available():
            if dds.charge_state != 0:
                assert dds.incar["NELECT"] == nelect
            else:
                assert "NELECT" not in dds.incar
            assert dds.incar["NUPDOWN"] == nupdown

    def test_neutral_defect_dict_set(self):
        dds = self._generate_and_check_dds(self.prim_cdte.copy())  # fine for bulk prim input as well
        # reciprocal_density = 100/Å⁻³ for prim CdTe:
        self.kpts_nelect_nupdown_check(dds, 7, 18, 0)
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, poscar=False)
        self._write_and_check_dds_files(dds, potcar_spec=True)

        defect_entry = self.CdTe_defect_gen["Te_Cd_0"]
        dds = self._generate_and_check_dds(defect_entry.defect_supercell)
        # reciprocal_density = 100/Å⁻³ for CdTe supercell:
        self.kpts_nelect_nupdown_check(dds, 2, 480, 0)
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, potcar_spec=True)
        self._write_and_check_dds_files(dds, poscar=False)

    def test_charged_defect_incar(self):
        dds = self._generate_and_check_dds(self.prim_cdte.copy(), charge_state=1)  # fine w/bulk prim
        self.kpts_nelect_nupdown_check(dds, 7, 17, 1)  # 100/Å⁻³ for prim CdTe
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, poscar=False)

        defect_entry = self.CdTe_defect_gen["Te_Cd_0"]
        dds = self._generate_and_check_dds(defect_entry.defect_supercell.copy(), charge_state=-2)
        self.kpts_nelect_nupdown_check(dds, 2, 482, 0)  # 100/Å⁻³ for CdTe supercell
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, poscar=False)

        defect_entry = self.CdTe_defect_gen["Te_Cd_-2"]
        dds = self._generate_and_check_dds(defect_entry.defect_supercell.copy(), charge_state=-2)
        self.kpts_nelect_nupdown_check(dds, 2, 482, 0)  # 100/Å⁻³ for CdTe supercell
        self._write_and_check_dds_files(dds)
        self._write_and_check_dds_files(dds, poscar=False)

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
        self._write_and_check_dds_files(dds, poscar=False)

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
        self._write_and_check_dds_files(dds, poscar=False)

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

        print([str(warning.message) for warning in w])  # for debugging
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
        self._write_and_check_dds_files(dds, poscar=False)

    def test_ediff_per_atom_custom_setting(self):
        # first with DopedDictSet:
        with warnings.catch_warnings(record=True) as w:
            dds = DopedDictSet(
                self.prim_cdte.copy(),
                user_incar_settings={"EDIFF_PER_ATOM": 1e-2},
                user_kpoints_settings={"reciprocal_density": 123},
            )
        print([warning.message for warning in w])  # for debugging
        assert any(
            "EDIFF_PER_ATOM was set to 1.00e-02 eV/atom, which" in str(warning.message) for warning in w
        )
        assert any("This is a very large EDIFF for VASP" in str(warning.message) for warning in w)
        assert np.isclose(dds.incar["EDIFF"], 1e-2 * len(self.prim_cdte))

        # now with DefectDictSet:
        dds = DefectDictSet(
            self.prim_cdte.copy(),
            user_incar_settings={"EDIFF_PER_ATOM": 1e-2},
        )
        assert np.isclose(dds.incar["EDIFF"], 1e-2 * len(self.prim_cdte))

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
            self._write_and_check_dds_files(dds, poscar=False)
            self._write_and_check_dds_files(dds, potcar_spec=True)  # can test potcar_spec w/neutral

            # charged_dds:
            self._generate_and_check_dds(struct, charge_state=np.random.randint(-5, 5))
            self._write_and_check_dds_files(dds)
            self._write_and_check_dds_files(dds, poscar=False)

            DefectDictSet(
                struct,
                user_incar_settings={"ENCUT": 350},
                user_potcar_functional="PBE_52",
                user_potcar_settings={"Cu": "Cu_pv"},
                user_kpoints_settings={"reciprocal_density": 200},
                poscar_comment="Test pop",
            )
            self._write_and_check_dds_files(dds)
            self._write_and_check_dds_files(dds, poscar=False)
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
            self._write_and_check_dds_files(dds, poscar=False)
        self.kpts_nelect_nupdown_check(dds, [(2, 2, 1)], 1584, 0)
        # reciprocal_density = 100/Å⁻³ for YTOS

        if not _potcars_available():
            for test_warning_message in [
                "NUPDOWN (i.e. spin state) INCAR flag cannot be set",
                "POTCAR directory not set up with pymatgen",
            ]:
                assert any(test_warning_message in str(warning.message) for warning in w)

        # check changing charge state
        dds = self._generate_and_check_dds(self.ytos_bulk_supercell.copy(), charge_state=1)
        self.kpts_nelect_nupdown_check(dds, [(2, 2, 1)], 1583, 1)
        # reciprocal_density = 100/Å⁻³ for YTOS
        self._write_and_check_dds_files(dds, output_path="YTOS_test_dir")
        self._write_and_check_dds_files(dds, poscar=False)

    def _write_and_check_dds_files(self, dds, **kwargs):
        output_path = kwargs.pop("output_path", "test_pop")
        delete_dir = kwargs.pop("delete_dir", True)  # delete directory after testing?

        if not kwargs.get("poscar", True) and dds.charge_state != 0 and not _potcars_available():
            # error with charged defect and poscar=False
            with pytest.raises(ValueError) as e:
                dds.write_input(output_path, **kwargs)
            assert _check_potcar_dir_not_setup_warning_error(dds, e.value, poscar=False)
            return

        dds.write_input(output_path, **kwargs)

        print(output_path)  # to help debug if tests fail
        assert os.path.exists(output_path)

        if _potcars_available() or dds.charge_state == 0:  # INCARs should be written
            # load INCAR and check it matches dds.incar
            written_incar = Incar.from_file(f"{output_path}/INCAR")
            dds_incar_without_comments = dds.incar.copy()
            dds_incar_without_comments["ICORELEVEL"] = 0
            dds_incar_without_comments["ISYM"] = 0
            dds_incar_without_comments["ALGO"] = (
                "Normal"
                if "normal" in dds_incar_without_comments.get("ALGO", "normal").lower()
                else dds_incar_without_comments.get("ALGO", "normal")
            )
            if "KPAR" in dds_incar_without_comments and isinstance(
                dds_incar_without_comments["KPAR"], str
            ):
                dds_incar_without_comments["KPAR"] = int(dds_incar_without_comments["KPAR"][0])
            dds_incar_without_comments.pop(next(k for k in dds.incar if k.startswith("#")))
            assert written_incar == dds_incar_without_comments, "Written INCAR does not match dds.incar"

            with open(f"{output_path}/INCAR") as f:
                incar_lines = f.readlines()
            print(f"{output_path}/INCAR:", incar_lines)
            print("Testing comment strings")
            for comment_string in [
                "# MAY WANT TO CHANGE NCORE, KPAR, AEXX, ENCUT",
                "needed if using the kumagai-oba",
                "symmetry breaking extremely likely",
            ]:
                assert any(comment_string in line for line in incar_lines)

            print("Testing ALGO")
            if dds.incar.get("ALGO", "normal").lower() == "normal":  # ALGO = Normal default, has comment
                assert any(
                    "change to all if zhegv, fexcp/f or zbrent, or poor electronic convergence" in line
                    for line in incar_lines
                )

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

        if kwargs.get("poscar", True):
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

        self.prim_MgO = Structure.from_file(f"{self.example_dir}/MgO/Bulk_relax/CONTCAR")

    def tearDown(self):
        self.dds_test.tearDown()  # use tearDown from DefectDictSetTest
        if_present_rm("test_dir")
        if_present_rm("CdTe_bulk")

        for i in os.listdir():
            if os.path.isdir(i) and (
                "MgO" in i
                or any("vasp_" in j and os.path.isdir(os.path.join(i, j)) for j in os.listdir(i))
            ):
                if_present_rm(i)

        if_present_rm("MgO_defects_generator.json.gz")

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
                assert (
                    child_incar_settings.pop("KPAR") == 2 or int(child_incar_settings.pop("KPAR")[0]) == 2
                )
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

            print("Checking incar settings")
            assert parent_drs.user_incar_settings == child_incar_settings
            assert parent_drs.user_potcar_functional == child_dds.user_potcar_functional or str(
                parent_drs.user_potcar_functional[:3]
            ) == str(
                child_dds.user_potcar_functional[:3]
            )  # if PBE_52 set but not available, defaults to PBE
            assert parent_drs.user_potcar_settings == child_dds.user_potcar_settings
            if isinstance(child_dds.user_kpoints_settings, DopedKpoints | Kpoints):
                assert child_dds.user_kpoints_settings.as_dict() in [
                    DopedKpoints()
                    .from_dict(
                        {
                            "comment": "Γ-only KPOINTS from doped",
                            "generation_style": "Gamma",
                        }
                    )
                    .as_dict(),
                    Kpoints()
                    .from_dict(
                        {
                            "comment": "Γ-only KPOINTS from doped",
                            "generation_style": "Gamma",
                        }
                    )
                    .as_dict(),
                ]
            else:
                assert child_dds.user_kpoints_settings in [
                    parent_drs.user_kpoints_settings,
                    {"reciprocal_density": 100},  # default
                ]

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
                self.dds_test._write_and_check_dds_files(defect_dict_set, poscar=False)
                if defect_relax_set.charge_state == 0:
                    self.dds_test._write_and_check_dds_files(defect_dict_set, potcar_spec=True)

                print("Checking DRS/DDS attribute transfer")
                _check_drs_dds_attribute_transfer(defect_relax_set, defect_dict_set)

        for defect_dict_set, type in dds_bulk_test_list:
            if defect_dict_set is not None:
                print(f"Testing {defect_relax_set.defect_entry.name}, {type}")
                self.dds_test._check_dds(
                    defect_dict_set, defect_relax_set.bulk_supercell, charge_state=0, **kwargs
                )
                self.dds_test._write_and_check_dds_files(defect_dict_set)
                self.dds_test._write_and_check_dds_files(defect_dict_set, poscar=False)
                self.dds_test._write_and_check_dds_files(defect_dict_set, potcar_spec=True)

                _check_drs_dds_attribute_transfer(defect_relax_set, defect_dict_set)

        # test __repr__ info:
        assert all(
            i in defect_relax_set.__repr__()
            for i in [
                "doped DefectRelaxSet for bulk composition",
                f"and defect entry {defect_relax_set.defect_entry.name}. Available attributes:\n",
                "vasp_std",
                "vasp_gam",
                "Available methods:\n",
                "write_all",
            ]
        )

    def test_initialisation_and_writing(self):
        """
        Test the initialisation of DefectRelaxSet for a range of
        `DefectEntry`s.
        """
        if not self.heavy_tests:
            pytest.skip("Skipping heavy test on GH Actions")

        def _check_drs_defect_entry_attribute_transfer(parent_drs, input_defect_entry):
            assert parent_drs.defect_entry == input_defect_entry
            assert parent_drs.defect_supercell == input_defect_entry.defect_supercell
            assert parent_drs.charge_state == input_defect_entry.charge_state
            assert parent_drs.bulk_supercell == input_defect_entry.bulk_supercell

        # test initialising DefectRelaxSet with our generation-tests materials, and writing files to disk
        for defect_gen_name in [
            "CdTe defect_gen",
            "ytos_defect_gen",
            "ytos_defect_gen_supercell",
            "lmno_defect_gen",
            "cu_defect_gen",
            "agcu_defect_gen",
            "cd_i_supercell_defect_gen",
            "N_diamond_defect_gen",  # input structure for this is unordered (but this checks
            # that POSCAR site symbols output should be ordered)
            "SQS AgSbTe2 defect_gen",
        ]:
            print(f"Initialising and testing: {defect_gen_name}")
            if defect_gen_name == "CdTe defect_gen":
                defect_gen = self.CdTe_defect_gen
            elif defect_gen_name == "SQS AgSbTe2 defect_gen":
                defect_gen = DefectsGenerator(self.sqs_agsbte2, generate_supercell=False)
            else:
                defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json")

            # randomly choose a defect entry from the defect_gen dict:
            defect_entry = random.choice(list(defect_gen.values()))

            print(f"Randomly testing {defect_entry.name}")
            drs = DefectRelaxSet(defect_entry)
            self._general_defect_relax_set_check(drs)

            _check_drs_defect_entry_attribute_transfer(drs, defect_entry)

            print(f"Testing {defect_entry.name} with custom INCAR settings")
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
        if not _potcars_available():  # need to write files without error for charged defects
            pytest.skip()

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
            drs.write_all("test_dir", vasp_gam=True, poscar=True)

            assert os.path.exists("test_dir")
            assert os.path.exists("test_dir/vasp_gam")
            assert os.path.exists("test_dir/vasp_gam/POSCAR")
            assert os.path.exists("test_dir/vasp_std/POSCAR")
            assert os.path.exists("test_dir/vasp_std/INCAR")
            if_present_rm("test_dir")

            drs.write_gam("test_dir")
            assert os.path.exists("test_dir")
            assert os.path.exists("test_dir/vasp_gam")
            assert os.path.exists("test_dir/vasp_gam/POSCAR")
            if_present_rm("test_dir")

    def test_poscar_comments(self):
        drs = DefectRelaxSet(self.CdTe_defect_gen["Cd_i_C3v_0"])
        drs.write_all("test_dir", poscar=True)
        poscar = Poscar.from_file("test_dir/vasp_std/POSCAR")
        assert poscar.comment == "Cd_i_C3v ~[0.5417,0.5417,0.5417] 0"

        poscar = Poscar.from_file("test_dir/vasp_nkred_std/POSCAR")
        assert poscar.comment == "Cd_i_C3v ~[0.5417,0.5417,0.5417] 0"

        poscar = Poscar.from_file("test_dir/vasp_ncl/POSCAR")
        assert poscar.comment == "Cd_i_C3v ~[0.5417,0.5417,0.5417] 0"

        assert not os.path.exists("test_dir/CdTe_bulk")
        if _potcars_available():  # need to write files without error for charged defects
            drs.write_all("test_dir", bulk=True)
            assert not os.path.exists("test_dir/CdTe_bulk")

            poscar = Poscar.from_file("CdTe_bulk/vasp_ncl/POSCAR")
            assert poscar.comment == "Cd27 Te27 -- Bulk"

            drs = DefectRelaxSet(self.CdTe_defect_gen["Cd_i_C3v_0"], poscar_comment="Test pop")
            drs.write_all("test_dir", poscar=True)
            poscar = Poscar.from_file("test_dir/vasp_std/POSCAR")
            assert poscar.comment == "Test pop"

            drs = DefectRelaxSet(self.CdTe_defect_gen["v_Cd_-2"])
            drs.write_all("test_dir", poscar=True)
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
            assert poscar.comment == "Cd27 Te27 -- Bulk"

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

    def test_file_folder_overwriting(self):
        """
        Test that when writing to a folder with existing files, the files are
        not deleted if they are not being overwritten.

        i.e. test compatibility with `snb-groundstate -d vasp_nkred_std` being
        run prior to `write_files()` etc, as recommended in tutorials.
        """
        if not _potcars_available():  # need to write files without error for charged defects
            return

        defect_gen_test_list = [
            (
                DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json"),
                defect_gen_name,
            )
            for defect_gen_name in [
                "ytos_defect_gen",
                "ytos_defect_gen_supercell",
                "cu_defect_gen",
                "agcu_defect_gen",
            ]
        ]
        for defect_gen, defect_gen_name in defect_gen_test_list:
            print(f"Testing: {defect_gen_name}")
            # randomly choose a defect entry from the defect_gen dict:
            defect_entry = random.choice(list(defect_gen.values()))

            print(f"Randomly testing {defect_entry.name}")
            drs = DefectRelaxSet(defect_entry)
            self._general_defect_relax_set_check(drs)
            os.mkdir(defect_entry.name)
            os.mkdir(f"{defect_entry.name}/vasp_nkred_std")
            os.mkdir(f"{defect_entry.name}/vasp_std")
            with open(f"{defect_entry.name}/vasp_nkred_std/INCAR", "w") as file:
                file.write("Test pop amháin")
            with open(f"{defect_entry.name}/vasp_nkred_std/POSCAR", "w") as file:
                file.write("Test pop a dó")
            with open(f"{defect_entry.name}/vasp_std/POSCAR", "w") as file:
                file.write("Test pop a trí")  # tír gan teanga, tír gan anam
            drs.write_all()

            with open(f"{defect_entry.name}/vasp_nkred_std/INCAR") as file:
                assert "Test pop" not in file.read()
            with open(f"{defect_entry.name}/vasp_nkred_std/POSCAR") as file:
                assert "Test pop" in file.read()
            with open(f"{defect_entry.name}/vasp_std/POSCAR") as file:
                assert "Test pop" in file.read()

    def test_GGA_defect_sets(self):
        """
        Test initialising defects sets with `LHFCALC = False` (i.e. semi-local
        GGA DFT).

        So `vasp_nkred_std` is None (but no warnings unless `write_nkred_std`
        called etc.)
        """
        mgo_defect_gen = DefectsGenerator(self.prim_MgO, supercell_gen_kwargs={"force_cubic": True})

        from pymatgen.io.vasp.inputs import Kpoints

        with warnings.catch_warnings(record=True) as w:
            defect_set = DefectsSet(
                mgo_defect_gen,
                user_incar_settings={
                    "ENCUT": 450,
                    "GGA": "PS",  # Functional (PBEsol for this tutorial)
                    "LHFCALC": False,  # Disable Hybrid functional
                    "NCORE": 8,
                },
                user_kpoints_settings=Kpoints(kpts=[(2, 2, 2)]),
            )
            assert defect_set.defect_sets["Mg_O_+2"].vasp_nkred_std is None
            assert defect_set.defect_sets["Mg_O_+2"].bulk_vasp_nkred_std is None
            assert defect_set.defect_sets["Mg_O_+2"].vasp_ncl is None  # SOC = False by default for MgO
            assert defect_set.defect_sets["Mg_O_+2"].bulk_vasp_ncl is None

            if _potcars_available():  # need to write files without error for charged defects
                defect_set.write_files()

        print([str(warning.message) for warning in w])  # for debugging
        non_potcar_warnings = [warning for warning in w if "POTCAR" not in str(warning.message)]
        assert not non_potcar_warnings  # no warnings about LHFCALC / SOC

        # use neutral charge states here to avoid errors when POTCARs not available:
        defect_names = ["Mg_O_0", "Mg_O_+2"] if _potcars_available() else ["Mg_O_0"]
        for defect_name in defect_names:
            print(f"Testing: {defect_name}")
            with warnings.catch_warnings(record=True) as w:
                defect_set.defect_sets[defect_name].write_nkred_std()
            print([str(warning.message) for warning in w])  # for debugging
            assert (
                "`LHFCALC` is set to `False` in user_incar_settings, so `vasp_nkred_std` is None"
                in str(w[0].message)
            )

            with warnings.catch_warnings(record=True) as w:
                defect_set.defect_sets[defect_name].write_ncl()
            print([str(warning.message) for warning in w])  # for debugging
            assert (
                "DefectRelaxSet.soc is False, so `vasp_ncl` is None (i.e. no `vasp_ncl` input files "
                in str(w[0].message)
            )

        # test setting SOC = True:
        with warnings.catch_warnings(record=True) as w:
            defect_set = DefectsSet(
                mgo_defect_gen,
                user_incar_settings={
                    "ENCUT": 450,
                    "GGA": "PS",  # Functional (PBEsol for this tutorial)
                    "LHFCALC": False,  # Disable Hybrid functional
                    "NCORE": 8,
                },
                user_kpoints_settings=Kpoints(kpts=[(2, 2, 2)]),
                soc=True,
            )
            assert defect_set.defect_sets["Mg_O_+2"].vasp_nkred_std is None
            assert defect_set.defect_sets["Mg_O_+2"].bulk_vasp_nkred_std is None
            assert defect_set.defect_sets["Mg_O_+2"].vasp_ncl is not None  # SOC = True now
            assert defect_set.defect_sets["Mg_O_+2"].bulk_vasp_ncl is not None

            if _potcars_available():  # need to write files without error for charged defects
                defect_set.write_files()

        print([str(warning.message) for warning in w])  # for debugging
        non_potcar_warnings = [warning for warning in w if "POTCAR" not in str(warning.message)]
        assert not non_potcar_warnings  # no warnings about LHFCALC / SOC

        for defect_name in defect_names:
            print(f"Testing: {defect_name}")
            with warnings.catch_warnings(record=True) as w:
                defect_set.defect_sets[defect_name].write_ncl()
            print([str(warning.message) for warning in w])  # for debugging
            non_potcar_warnings = [warning for warning in w if "POTCAR" not in str(warning.message)]
            assert not non_potcar_warnings


class DefectsSetTest(unittest.TestCase):
    def setUp(self):
        self.dds_test = DefectDictSetTest()
        self.dds_test.setUp()  # get attributes from DefectDictSetTest
        DefectDictSetTest.setUp(self)  # get attributes from DefectDictSetTest

        self.drs_test = DefectRelaxSetTest()
        self.drs_test.setUp()

        self.CdTe_defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/CdTe_defect_gen.json")
        self.structure_matcher = StructureMatcher(
            comparator=ElementComparator(), primitive_cell=False
        )  # ignore oxidation states when comparing structures

        # Note this is different to above: (for testing against pre-generated input files with these
        # settings):
        self.CdTe_custom_test_incar_settings = {"ENCUT": 350, "NCORE": 10, "LVHAR": False, "ALGO": "All"}

    def tearDown(self):
        locale.setlocale(locale.LC_ALL, "")  # reset locale

        for file in os.listdir():
            if file.endswith(".json.gz"):
                if_present_rm(file)

        for folder in os.listdir():
            if any(os.path.exists(f"{folder}/vasp_{xxx}") for xxx in ["gam", "std", "ncl"]) or (
                os.path.isdir(folder) and "INCAR" in os.listdir(folder)
            ):
                # generated output files
                if_present_rm(folder)

        for i in [
            "test_pop",
            "AgSbTe2_test",
            "CdTe_defects_generator.json",
            "test_CdTe_defects_generator.json",
        ]:
            if_present_rm(i)

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
        ascii_encoding=False,
    ):
        if data_dir is None:
            data_dir = self.CdTe_data_dir

        if check_incar is None:
            check_incar = _potcars_available()

        folders = (
            [
                folder
                for folder in os.listdir(data_dir)
                if os.path.isdir(f"{data_dir}/{folder}") and ("bulk" not in folder.lower() or bulk)
            ]
            if not single_defect_dir
            else [""]
        )

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
            if ascii_encoding:
                test_kpoints.comment = test_kpoints.comment.replace("Å⁻³", "Angstrom^(-3)").replace(
                    "Γ", "Gamma"
                )

            assert test_kpoints.as_dict() == kpoints.as_dict()

    def _general_defects_set_check(self, defects_set, **kwargs):
        for entry_name, defect_relax_set in defects_set.defect_sets.items():
            print(f"Testing {entry_name} DefectRelaxSet")
            self.drs_test._general_defect_relax_set_check(defect_relax_set, **kwargs)
            # check bulk vasp attributes same as DefectsSet:
            assert defect_relax_set.bulk_vasp_ncl.structure == defects_set.bulk_vasp_ncl.structure
            assert defect_relax_set.bulk_vasp_ncl.incar == defects_set.bulk_vasp_ncl.incar
            assert defect_relax_set.bulk_vasp_ncl.kpoints == defects_set.bulk_vasp_ncl.kpoints

        assert all(
            i in defects_set.__repr__()
            for i in [
                "doped DefectsSet for bulk composition",
                f", with {len(defects_set.defect_entries)} defect entries in self.defect_entries. "
                "Available attributes:\n",
                "bulk_vasp_gam",
                "soc",
                "Available methods:\n",
                "write_files",
            ]
        )

    def test_CdTe_files(self):
        if not self.heavy_tests:
            pytest.skip("Skipping heavy test on GH Actions")

        CdTe_se_defect_gen = DefectsGenerator(self.prim_cdte, extrinsic="Se")
        defects_set = DefectsSet(
            CdTe_se_defect_gen,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )
        self._general_defects_set_check(defects_set)

        if _potcars_available():
            defects_set.write_files(potcar_spec=True)  # poscar=False by default

            # test no (unperturbed) POSCAR files written:
            for folder in os.listdir("."):
                if os.path.isdir(folder) and "bulk" not in folder:
                    for subfolder in os.listdir(folder):
                        if "vasp" in subfolder:
                            assert not os.path.exists(f"{folder}/{subfolder}/POSCAR")

        else:
            with pytest.raises(ValueError):
                defects_set.write_files(
                    potcar_spec=True
                )  # INCAR ValueError for charged defects if POTCARs not
                # available and poscar=False
            defects_set.write_files(potcar_spec=True, poscar=True)

        # test no vasp_gam files written:
        for folder in os.listdir("."):
            assert not os.path.exists(f"{folder}/vasp_gam")

        defects_set.write_files(potcar_spec=True, poscar=True, vasp_gam=True, rattle=False)

        bulk_supercell = Structure.from_file("CdTe_bulk/vasp_ncl/POSCAR")
        assert self.structure_matcher.fit(bulk_supercell, self.CdTe_defect_gen.bulk_supercell)
        # check_generated_vasp_inputs also checks bulk folders

        assert os.path.exists("CdTe_defects_generator.json.gz")
        CdTe_se_defect_gen.to_json("test_CdTe_defects_generator.json")
        with (
            gzip.open("CdTe_defects_generator.json.gz", "rt") as f,
            open("CdTe_defects_generator.json", "w") as f_out,
        ):
            f_out.write(f.read())
        assert filecmp.cmp("CdTe_defects_generator.json", "test_CdTe_defects_generator.json")

        # assert that the same folders in self.CdTe_data_dir are present in the current directory
        print("Checking vasp_gam files")
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=False)  # tests vasp_gam
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
        defects_set.write_files(potcar_spec=True, poscar=True, bulk="all", vasp_gam=True, rattle=False)
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=True)  # tests vasp_gam
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=True, bulk=True)  # vasp_std
        self.check_generated_vasp_inputs(vasp_type="vasp_nkred_std", check_poscar=False, bulk=True)

        # test DefectDictSet objects:
        for _defect_species, defect_relax_set in defects_set.defect_sets.items():
            for defect_dict_set in [defect_relax_set.vasp_gam, defect_relax_set.bulk_vasp_gam]:
                assert defect_dict_set.kpoints.kpts == [(1, 1, 1)]
            for defect_dict_set in [
                defect_relax_set.vasp_std,
                defect_relax_set.bulk_vasp_std,
                defect_relax_set.vasp_nkred_std,
                defect_relax_set.bulk_vasp_nkred_std,
                defect_relax_set.vasp_ncl,
                defect_relax_set.bulk_vasp_ncl,
            ]:
                assert defect_dict_set.kpoints.kpts == [(2, 2, 2)]

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
            with pytest.raises(ValueError):
                defects_set.write_files(potcar_spec=True, vasp_gam=True)  # INCAR ValueError for charged
                # defects if POTCARs not available and poscar=False
            defects_set.write_files(potcar_spec=True, vasp_gam=True, poscar=True)

        for folder in os.listdir("."):
            if os.path.isdir(f"{folder}/vasp_gam"):
                with open(f"{folder}/vasp_gam/POTCAR.spec", encoding="utf-8") as file:
                    contents = file.readlines()
                    assert contents[0] in ["Cd_sv_GW", "Cd_sv_GW\n"]
                    assert contents[1] in ["Te_GW", "Te_GW\n"]

                for subfolder in ["vasp_std", "vasp_nkred_std", "vasp_ncl"]:
                    kpoints = Kpoints.from_file(f"{folder}/{subfolder}/KPOINTS")
                    assert kpoints.kpts == [(4, 4, 4)]  # 4x4x4 with 54-atom 3x3x3 prim supercell,
                    # 3x3x3 with 64-atom 2x2x2 conv supercell

    def test_write_files_single_defect_entry(self):
        single_defect_entry = self.CdTe_defect_gen["Cd_i_C3v_+2"]
        defects_set = DefectsSet(
            single_defect_entry,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )
        defects_set.write_files(potcar_spec=True, vasp_gam=True, poscar=True, rattle=False)

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
            defects_set.write_files(potcar_spec=True, vasp_gam=True, poscar=True, rattle=False)
            locale.setlocale(locale.LC_ALL, "")  # resets locale

            # assert that the same folders in self.CdTe_data_dir are present in the current directory
            self.check_generated_vasp_inputs(  # tests vasp_gam
                generated_dir="Cd_i_C3v_+2",
                data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
                check_potcar_spec=True,
                single_defect_dir=True,
                ascii_encoding=True,
            )
            self.check_generated_vasp_inputs(  # vasp_std
                generated_dir="Cd_i_C3v_+2",
                data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
                vasp_type="vasp_std",
                check_poscar=True,
                check_potcar_spec=True,
                single_defect_dir=True,
                ascii_encoding=True,
            )
            self.check_generated_vasp_inputs(  # vasp_ncl
                generated_dir="Cd_i_C3v_+2",
                data_dir=f"{self.CdTe_data_dir}/Cd_i_C3v_+2",
                vasp_type="vasp_ncl",
                check_poscar=True,
                check_potcar_spec=True,
                single_defect_dir=True,
                ascii_encoding=True,
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
            defects_set.write_files(potcar_spec=True)  # poscar=False by default
        else:
            with pytest.raises(ValueError):
                defects_set.write_files(potcar_spec=True)  # INCAR ValueError for charged defects if
                # POTCARs not available and poscar=False
            defects_set.write_files(potcar_spec=True, poscar=True, rattle=False)

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
                with pytest.raises(ValueError):
                    defects_set.write_files()  # INCAR ValueError for charged defects if POTCARs not
                    # available
                defects_set.write_files(poscar=True)

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

    def test_SQS_AgSbTe2(self):
        """
        Test VASP IO functions with SQS AgSbTe2 supercell (tricky case for
        certain functions, like structure reordering etc).
        """
        sqs_defect_gen = DefectsGenerator(self.sqs_agsbte2, generate_supercell=False)
        defect_supercell = sqs_defect_gen["Ag_Sb_Cs_Te2.90_-1"].sc_entry.structure
        with warnings.catch_warnings(record=True) as w:
            dds = DefectDictSet(defect_supercell, charge_state=0)
        print([str(warning.message) for warning in w])  # for debugging
        non_potcar_warnings = [warning for warning in w if "POTCAR" not in str(warning.message)]
        assert not non_potcar_warnings  # no warnings with DDS generation
        if _potcars_available():
            neutral_nelect = dds.nelect

        def _check_agsbte2_vasp_folder(folder_name, structure, **kwargs):
            with open(f"{folder_name}/KPOINTS", encoding="utf-8") as f:
                kpoints_lines = f.readlines()
            if "gam" not in folder_name:
                assert kpoints_lines[0] == "KPOINTS from doped, with reciprocal_density = 100/Å⁻³\n"
                assert kpoints_lines[3] == "2 2 2\n"
            else:
                assert kpoints_lines[0] == "Γ-only KPOINTS from doped\n"
                assert kpoints_lines[3] == "1 1 1\n"
            assert kpoints_lines[1] == "0\n"
            assert kpoints_lines[2] == "Gamma\n"

            if _potcars_available():  # otherwise INCARs not written for charged defects
                with open(f"{folder_name}/INCAR", encoding="utf-8") as f:
                    assert "HFSCREEN = 0.208\n" in f.readlines()

            if kwargs.get("poscar", True):
                struct = Structure.from_file(f"{folder_name}/POSCAR")
                assert self.structure_matcher.fit(struct, structure)
            else:
                assert not os.path.exists(f"{folder_name}/POSCAR")

        def _write_and_test_agsbte2_dds(dds, structure, folder_name="test_pop", **kwargs):
            dds.write_input(folder_name, **kwargs)  # general DDS test already done for this system
            assert os.path.exists(folder_name)
            _check_agsbte2_vasp_folder(folder_name, structure, **kwargs)

        _write_and_test_agsbte2_dds(dds, defect_supercell)

        with warnings.catch_warnings(record=True) as w:
            dds = DefectDictSet(defect_supercell, charge_state=+2)
        print([str(warning.message) for warning in w])  # for debugging
        non_potcar_warnings = [warning for warning in w if "POTCAR" not in str(warning.message)]
        assert not non_potcar_warnings  # no warnings with DDS generation

        if _potcars_available():
            assert dds.nelect == neutral_nelect - 2
        _write_and_test_agsbte2_dds(dds, defect_supercell, "AgSbTe2_test")
        if _potcars_available():  # with poscar=False, only works for charged defects w/POTCARs
            _write_and_test_agsbte2_dds(dds, defect_supercell, "AgSbTe2_test_no_POSCAR", poscar=False)

        # test DefectRelaxSet behaviour:
        defect_entry = sqs_defect_gen["Ag_Sb_Cs_Te2.90_-2"]
        with warnings.catch_warnings(record=True) as w:
            drs = DefectRelaxSet(defect_entry)
        print([str(warning.message) for warning in w])  # for debugging
        non_potcar_warnings = [warning for warning in w if "POTCAR" not in str(warning.message)]
        assert not non_potcar_warnings  # no warnings with DRS generation

        if _potcars_available():
            drs.write_std()
            assert os.path.exists("Ag_Sb_Cs_Te2.90_-2/vasp_std")
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_std", defect_entry.defect_supercell, poscar=False
            )

        drs.write_std(poscar=True)
        _check_agsbte2_vasp_folder("Ag_Sb_Cs_Te2.90_-2/vasp_std", defect_entry.defect_supercell)

        def _check_reloaded_defect_entry(filename, ref_defect_entry):
            reloaded_defect_entry = loadfn(filename)
            assert reloaded_defect_entry.name == ref_defect_entry.name
            assert np.allclose(
                reloaded_defect_entry.sc_defect_frac_coords,
                ref_defect_entry.sc_defect_frac_coords,
            )
            _compare_attributes(reloaded_defect_entry, ref_defect_entry)

        _check_reloaded_defect_entry(
            "Ag_Sb_Cs_Te2.90_-2/vasp_std/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
        )

        if _potcars_available():
            drs.write_nkred_std()
            assert os.path.exists("Ag_Sb_Cs_Te2.90_-2/vasp_nkred_std")
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_nkred_std",
                defect_entry.defect_supercell,
                poscar=False,
            )

        drs.write_nkred_std(poscar=True)
        _check_agsbte2_vasp_folder("Ag_Sb_Cs_Te2.90_-2/vasp_nkred_std", defect_entry.defect_supercell)
        _check_reloaded_defect_entry(
            "Ag_Sb_Cs_Te2.90_-2/vasp_nkred_std/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
        )

        assert not any(i in os.listdir("Ag_Sb_Cs_Te2.90_-2") for i in ["vasp_gam", "vasp_ncl"])

        drs.write_gam()
        assert os.path.exists("Ag_Sb_Cs_Te2.90_-2/vasp_gam")
        _check_agsbte2_vasp_folder(
            "Ag_Sb_Cs_Te2.90_-2/vasp_gam", defect_entry.defect_supercell, poscar=True
        )  # poscar True by default when write_gam called directly
        _check_reloaded_defect_entry(
            "Ag_Sb_Cs_Te2.90_-2/vasp_gam/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
        )

        if _potcars_available():
            drs.write_ncl()
            assert os.path.exists("Ag_Sb_Cs_Te2.90_-2/vasp_ncl")
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_ncl", defect_entry.defect_supercell, poscar=False
            )
            _check_reloaded_defect_entry(
                "Ag_Sb_Cs_Te2.90_-2/vasp_ncl/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
            )
            assert "bulk" not in os.listdir()  # no bulk folders written yet

        if_present_rm("Ag_Sb_Cs_Te2.90_-2")
        if _potcars_available():
            drs.write_all()
            assert not os.path.exists("Ag_Sb_Cs_Te2.90_-2/vasp_gam")  # not written by default
            for i in ["vasp_nkred_std", "vasp_std", "vasp_ncl"]:
                assert os.path.exists(f"Ag_Sb_Cs_Te2.90_-2/{i}")
                _check_agsbte2_vasp_folder(
                    f"Ag_Sb_Cs_Te2.90_-2/{i}", defect_entry.defect_supercell, poscar=False
                )
                _check_reloaded_defect_entry(
                    f"Ag_Sb_Cs_Te2.90_-2/{i}/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
                )

        if _potcars_available():
            drs.write_all(vasp_gam=True)
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_gam", defect_entry.defect_supercell, poscar=True
            )

        drs.write_all(poscar=True)
        for i in ["vasp_nkred_std", "vasp_std", "vasp_ncl"]:
            _check_agsbte2_vasp_folder(
                f"Ag_Sb_Cs_Te2.90_-2/{i}", defect_entry.defect_supercell, poscar=True
            )
            _check_reloaded_defect_entry(
                f"Ag_Sb_Cs_Te2.90_-2/{i}/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
            )
        assert "bulk" not in os.listdir()  # no bulk folders written by default

        if _potcars_available():
            drs.write_all(bulk=True)
            assert "AgSbTe2_bulk" in os.listdir()
            for i in ["vasp_gam", "vasp_nkred_std", "vasp_std", "vasp_ncl"]:
                _check_agsbte2_vasp_folder(
                    f"Ag_Sb_Cs_Te2.90_-2/{i}", defect_entry.defect_supercell, poscar=True
                )
                _check_reloaded_defect_entry(
                    f"Ag_Sb_Cs_Te2.90_-2/{i}/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
                )
            _check_agsbte2_vasp_folder("AgSbTe2_bulk/vasp_ncl", defect_entry.bulk_supercell, poscar=True)
            assert all(i not in os.listdir("AgSbTe2_bulk") for i in ["vasp_gam", "vasp_std"])

            if_present_rm("Ag_Sb_Cs_Te2.90_-2")
            drs.write_all(bulk="all")
            assert not os.path.exists("AgSbTe2_bulk/vasp_gam")  # not written by default
            for i in ["vasp_nkred_std", "vasp_std", "vasp_ncl"]:
                _check_agsbte2_vasp_folder(
                    f"Ag_Sb_Cs_Te2.90_-2/{i}", defect_entry.defect_supercell, poscar=False
                )
                _check_reloaded_defect_entry(
                    f"Ag_Sb_Cs_Te2.90_-2/{i}/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
                )
                _check_agsbte2_vasp_folder(f"AgSbTe2_bulk/{i}", defect_entry.bulk_supercell, poscar=True)
            drs.write_all(bulk="all", vasp_gam=True)
            _check_agsbte2_vasp_folder("AgSbTe2_bulk/vasp_gam", defect_entry.bulk_supercell, poscar=True)

        drs.write_gam(bulk=True)
        _check_agsbte2_vasp_folder(
            "Ag_Sb_Cs_Te2.90_-2/vasp_gam", defect_entry.defect_supercell, poscar=True
        )
        _check_agsbte2_vasp_folder("AgSbTe2_bulk/vasp_gam", defect_entry.bulk_supercell, poscar=True)

        if _potcars_available():
            drs.write_std(bulk=True)
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_std", defect_entry.defect_supercell, poscar=False
            )
            _check_agsbte2_vasp_folder("AgSbTe2_bulk/vasp_std", defect_entry.bulk_supercell, poscar=True)
            drs.write_nkred_std(bulk=True)
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_nkred_std",
                defect_entry.defect_supercell,
                poscar=False,
            )
            _check_agsbte2_vasp_folder(
                "AgSbTe2_bulk/vasp_nkred_std", defect_entry.bulk_supercell, poscar=True
            )
            drs.write_ncl(bulk=True)
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_ncl", defect_entry.defect_supercell, poscar=False
            )
            _check_agsbte2_vasp_folder("AgSbTe2_bulk/vasp_ncl", defect_entry.bulk_supercell, poscar=True)

        drs.write_all("test_pop", poscar=True)
        for i in ["vasp_nkred_std", "vasp_std", "vasp_ncl"]:
            _check_agsbte2_vasp_folder(f"test_pop/{i}", defect_entry.defect_supercell, poscar=True)
            _check_reloaded_defect_entry(f"test_pop/{i}/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry)

        if_present_rm("Ag_Sb_Cs_Te2.90_-2")
        if_present_rm("AgSbTe2_bulk")

        # test behaviour with DefectsSet initialised from DefectEntry list
        ds = DefectsSet([sqs_defect_gen["Ag_Sb_Cs_Te2.90_-2"], sqs_defect_gen["Ag_Sb_Cs_Te2.90_0"]])
        if _potcars_available():
            ds.write_files()
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_0/vasp_ncl", defect_entry.defect_supercell, poscar=False
            )
            _check_agsbte2_vasp_folder(
                "Ag_Sb_Cs_Te2.90_-2/vasp_ncl", defect_entry.defect_supercell, poscar=False
            )
            _check_agsbte2_vasp_folder("AgSbTe2_bulk/vasp_ncl", defect_entry.bulk_supercell, poscar=True)
            assert not os.path.exists("AgSbTe2_bulk/vasp_std")  # only top one

        ds.write_files(vasp_gam=True, poscar=True, processes=2, bulk="all")
        for i in ["vasp_gam", "vasp_nkred_std", "vasp_std", "vasp_ncl"]:
            _check_agsbte2_vasp_folder(
                f"Ag_Sb_Cs_Te2.90_-2/{i}", defect_entry.defect_supercell, poscar=True
            )
            _check_agsbte2_vasp_folder(
                f"Ag_Sb_Cs_Te2.90_0/{i}", defect_entry.defect_supercell, poscar=True
            )
            _check_reloaded_defect_entry(
                f"Ag_Sb_Cs_Te2.90_-2/{i}/Ag_Sb_Cs_Te2.90_-2.json.gz", defect_entry
            )
            _check_agsbte2_vasp_folder(f"AgSbTe2_bulk/{i}", defect_entry.bulk_supercell, poscar=True)
            _check_agsbte2_vasp_folder(f"AgSbTe2_bulk/{i}", defect_entry.bulk_supercell, poscar=True)

        # test convenience methods for ``DefectsSet``, where dict methods are passed to
        # ``self.defect_sets``:
        assert len(ds) == len(ds.defect_sets)  # __len__ method
        assert len(ds) == 2  # 2 entries provided above
        assert hasattr(ds, "defect_sets")  # Test accessing attributes
        assert "Ag_Sb_Cs_Te2.90_-2" in list(ds.keys())  # __getattr__ method
        assert "Ag_Sb_Cs_Te2.90_-2" in ds  # __contains__ method
        assert sqs_defect_gen["Ag_Sb_Cs_Te2.90_-2"] == ds["Ag_Sb_Cs_Te2.90_-2"].defect_entry  # __getitem__
        assert isinstance(ds["Ag_Sb_Cs_Te2.90_-2"], DefectRelaxSet)  # __getitem__
        if _potcars_available():
            assert ds["Ag_Sb_Cs_Te2.90_-2"].vasp_nkred_std.incar["HFSCREEN"] == 0.208  # __getitem__

        new_drs = DefectRelaxSet(sqs_defect_gen["Ag_Sb_Cs_Te2.90_-1"])
        ds["Ag_Sb_Cs_Te2.90_-1"] = new_drs
        assert len(ds) == 3  # 3 entries now
        assert "Ag_Sb_Cs_Te2.90_-1" in ds

        del ds["Ag_Sb_Cs_Te2.90_0"]  # __delitem__ method
        assert "Ag_Sb_Cs_Te2.90_0" not in ds
        assert len(ds) == 2

        for key in ds:
            assert key in ["Ag_Sb_Cs_Te2.90_-2", "Ag_Sb_Cs_Te2.90_-1"]

    def test_rattled_CdTe_files(self):
        CdTe_se_defect_gen = DefectsGenerator(self.prim_cdte, extrinsic="Se")
        defects_set = DefectsSet(
            CdTe_se_defect_gen,
            user_incar_settings=self.CdTe_custom_test_incar_settings,
        )

        defects_set.write_files(potcar_spec=True, poscar=True)  # rattle = True by default
        # test no vasp_gam files written:
        for folder in os.listdir("."):
            assert not os.path.exists(f"{folder}/vasp_gam")

        defects_set.write_files(potcar_spec=True, poscar=True, vasp_gam=True)

        bulk_supercell = Structure.from_file("CdTe_bulk/vasp_ncl/POSCAR")
        assert self.structure_matcher.fit(bulk_supercell, self.CdTe_defect_gen.bulk_supercell)
        # check_generated_vasp_inputs also checks bulk folders
        assert os.path.exists("CdTe_defects_generator.json.gz")

        # assert that the same folders in self.CdTe_data_dir are present in the current directory
        print("Checking vasp_gam files")
        self.check_generated_vasp_inputs(check_potcar_spec=True, bulk=False, check_poscar=False)
        print("Checking vasp_std files")
        self.check_generated_vasp_inputs(vasp_type="vasp_std", check_poscar=False, bulk=False)  # vasp_std
        print("Checking vasp_ncl files")
        self.check_generated_vasp_inputs(vasp_type="vasp_ncl", check_poscar=False, bulk=True)  # vasp_ncl

        # test rattled POSCARs and bulk
        # rattle = True by default for DefectsSet
        def _check_rattled_and_bulk():
            print("Checking rattled POSCARs and bulk")
            # check bulk first:
            reference_bulk = Structure.from_file(f"{self.CdTe_data_dir}/CdTe_bulk/vasp_ncl/POSCAR")
            generated_bulk = Structure.from_file("CdTe_bulk/vasp_ncl/POSCAR")
            assert reference_bulk == generated_bulk
            for defect_entry_name in ["v_Cd_0", "Se_i_C3v_-1", "Cd_Te_+2"]:
                print(defect_entry_name)
                reference_struct = Structure.from_file(
                    f"{self.CdTe_data_dir}/{defect_entry_name}/vasp_gam/POSCAR"
                )
                generated_struct = Structure.from_file(f"{defect_entry_name}/vasp_std/POSCAR")
                assert not StructureMatcher(stol=0.1).fit(
                    reference_struct, generated_struct
                )  # now rattled
                print(StructureMatcher().get_rms_dist(reference_struct, generated_struct))  # for debugging
                assert (
                    StructureMatcher().get_rms_dist(reference_struct, generated_struct)[0] > 0.05
                )  # rattled

                for other_vasp_dir in ["vasp_gam", "vasp_nkred_std", "vasp_ncl"]:
                    print(f"Checking {other_vasp_dir}")
                    other_generated_struct = Structure.from_file(
                        f"{defect_entry_name}/{other_vasp_dir}/POSCAR"
                    )
                    assert other_generated_struct == generated_struct

                for other_dir in os.listdir():
                    if other_dir.startswith(defect_entry_name.rsplit("_", 1)[0]):
                        print(f"Checking {other_dir}/vasp_std")
                        other_generated_struct = Structure.from_file(f"{other_dir}/vasp_std/POSCAR")
                        assert other_generated_struct == generated_struct

        _check_rattled_and_bulk()

        # test DefectRelaxSet:
        # rattle = True by default for DefectRelaxSet
        for defect_species, defect_relax_set in defects_set.defect_sets.items():
            defect_relax_set.write_all(defect_species, poscar=True, vasp_gam=True, bulk=True)

        _check_rattled_and_bulk()

        # test DefectDictSet objects:
        for defect_species, defect_relax_set in defects_set.defect_sets.items():
            # rattle = False by default for DefectDictSet
            defect_relax_set.bulk_vasp_ncl.write_input("CdTe_bulk/vasp_ncl", poscar=True)

            for defect_dict_set, subfolder in [
                (defect_relax_set.vasp_gam, "vasp_gam"),
                (defect_relax_set.vasp_nkred_std, "vasp_nkred_std"),
                (defect_relax_set.vasp_std, "vasp_std"),
                (defect_relax_set.vasp_ncl, "vasp_ncl"),
            ]:
                defect_dict_set.write_input(f"{defect_species}/{subfolder}", poscar=True, rattle=True)

        _check_rattled_and_bulk()


# TODO: All warnings and errors tested? (So far all DefectDictSet ones done)

if __name__ == "__main__":
    unittest.main()
