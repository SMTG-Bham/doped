"""
Tests for doped.thermodynamics module.
"""

import os
import shutil
import sys
import unittest
import warnings
from copy import deepcopy
from functools import wraps
from io import StringIO

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from monty.serialization import dumpfn, loadfn

from doped.thermodynamics import DefectThermodynamics

# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")
mpl.use("Agg")  # don't show interactive plots if testing from CLI locally

# Define paths for baseline_dir and style as constants
BASELINE_DIR = f"{data_dir}/remote_baseline_plots"
STYLE = f"{module_path}/../doped/utils/doped.mplstyle"


def custom_mpl_image_compare(filename):
    """
    Set our default settings for MPL image compare.
    """

    def decorator(func):
        @wraps(func)
        @pytest.mark.mpl_image_compare(
            baseline_dir=BASELINE_DIR,
            filename=filename,
            style=STYLE,
            savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def if_present_rm(path):
    """
    Remove file or directory if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _run_func_and_capture_stdout_warnings(func, *args, **kwargs):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
    w = None
    try:
        with warnings.catch_warnings(record=True) as w:
            result = func(*args, **kwargs)
        output = sys.stdout.getvalue()  # Return a str containing the printed output
    finally:
        sys.stdout = original_stdout  # Reset standard output to its original value.

    print(f"Running {func.__name__} with args: {args} and kwargs: {kwargs}:")
    print(output)
    if w:
        print(f"Warnings:\n{[str(warning.message) for warning in w]}")
    print(f"Result: {result}\n")

    return result, output, w


class DefectThermodynamicsTestCase(unittest.TestCase):
    def setUp(self):
        self.CdTe_defect_thermo = deepcopy(self.orig_CdTe_defect_thermo)
        self.CdTe_defect_dict = deepcopy(self.orig_CdTe_defect_dict)
        self.YTOS_defect_thermo = deepcopy(self.orig_YTOS_defect_thermo)
        self.YTOS_defect_dict = deepcopy(self.orig_YTOS_defect_dict)
        self.Sb2Se3_defect_thermo = deepcopy(self.orig_Sb2Se3_defect_thermo)
        self.Sb2Se3_defect_dict = deepcopy(self.orig_Sb2Se3_defect_dict)
        self.Sb2Si2Te6_defect_thermo = deepcopy(self.orig_Sb2Si2Te6_defect_thermo)
        self.Sb2Si2Te6_defect_dict = deepcopy(self.orig_Sb2Si2Te6_defect_dict)
        self.V2O5_defect_thermo = deepcopy(self.orig_V2O5_defect_thermo)
        self.V2O5_defect_dict = deepcopy(self.orig_V2O5_defect_dict)

    @classmethod
    def setUpClass(cls):
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.EXAMPLE_DIR = os.path.join(cls.module_path, "../examples")
        cls.CdTe_EXAMPLE_DIR = os.path.join(cls.module_path, "../examples/CdTe")
        cls.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe
        cls.CdTe_chempots = loadfn(os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_chempots.json"))

        cls.YTOS_EXAMPLE_DIR = os.path.join(cls.module_path, "../examples/YTOS")
        cls.ytos_dielectric = [  # from legacy Materials Project
            [40.71948719643814, -9.282128210266565e-14, 1.26076160303219e-14],
            [-9.301652644020242e-14, 40.71948719776858, 4.149879443489052e-14],
            [5.311743673463141e-15, 2.041077680836527e-14, 25.237620491130023],
        ]

        cls.Sb2Se3_DATA_DIR = os.path.join(cls.module_path, "data/Sb2Se3")
        cls.Sb2Se3_dielectric = np.array([[85.64, 0, 0], [0.0, 128.18, 0], [0, 0, 15.00]])

        cls.Sb2Si2Te6_dielectric = [44.12, 44.12, 17.82]
        cls.Sb2Si2Te6_DATA_DIR = os.path.join(cls.EXAMPLE_DIR, "Sb2Si2Te6")

        cls.V2O5_DATA_DIR = os.path.join(cls.module_path, "data/V2O5")

        cls.orig_CdTe_defect_dict = loadfn(
            os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_example_defect_dict.json")
        )
        cls.orig_CdTe_defect_thermo = loadfn(
            os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json")
        )
        cls.orig_YTOS_defect_dict = loadfn(
            os.path.join(cls.YTOS_EXAMPLE_DIR, "YTOS_example_defect_dict.json")
        )
        cls.orig_YTOS_defect_thermo = loadfn(
            os.path.join(cls.YTOS_EXAMPLE_DIR, "YTOS_example_thermo.json")
        )
        cls.orig_Sb2Se3_defect_dict = loadfn(
            os.path.join(cls.Sb2Se3_DATA_DIR, "defect/Sb2Se3_O_example_defect_dict.json")
        )
        cls.orig_Sb2Se3_defect_thermo = loadfn(
            os.path.join(cls.Sb2Se3_DATA_DIR, "Sb2Se3_O_example_thermo.json")
        )
        cls.orig_Sb2Si2Te6_defect_dict = loadfn(
            os.path.join(cls.Sb2Si2Te6_DATA_DIR, "Sb2Si2Te6_example_defect_dict.json")
        )
        cls.orig_Sb2Si2Te6_defect_thermo = loadfn(
            os.path.join(cls.Sb2Si2Te6_DATA_DIR, "Sb2Si2Te6_example_thermo.json")
        )

        cls.orig_V2O5_defect_dict = loadfn(
            os.path.join(cls.V2O5_DATA_DIR, "V2O5_example_defect_dict.json")
        )
        cls.orig_V2O5_defect_thermo = loadfn(os.path.join(cls.V2O5_DATA_DIR, "V2O5_example_thermo.json"))
        cls.V2O5_chempots = loadfn(os.path.join(cls.V2O5_DATA_DIR, "chempots.json"))

    def _compare_defect_thermo_and_dict(self, defect_thermo, defect_dict):
        assert len(defect_thermo.defect_entries) == len(defect_dict)
        assert {entry.name for entry in defect_thermo.defect_entries} == set(defect_dict.keys())
        assert {round(entry.get_ediff(), 3) for entry in defect_thermo.defect_entries} == {
            round(entry.get_ediff(), 3) for entry in defect_dict.values()
        }
        assert {  # check coords are the same by getting their products
            round(np.product(entry.sc_defect_frac_coords), 3) for entry in defect_thermo.defect_entries
        } == {round(np.product(entry.sc_defect_frac_coords), 3) for entry in defect_dict.values()}

    def _compare_defect_thermos(self, defect_thermo1, defect_thermo2):
        assert len(defect_thermo1.defect_entries) == len(defect_thermo2.defect_entries)
        assert {entry.name for entry in defect_thermo1.defect_entries} == {
            entry.name for entry in defect_thermo2.defect_entries
        }
        assert {round(entry.get_ediff(), 3) for entry in defect_thermo1.defect_entries} == {
            round(entry.get_ediff(), 3) for entry in defect_thermo2.defect_entries
        }
        assert {  # check coords are the same by getting their products
            round(np.product(entry.sc_defect_frac_coords), 3) for entry in defect_thermo1.defect_entries
        } == {round(np.product(entry.sc_defect_frac_coords), 3) for entry in defect_thermo2.defect_entries}

    def _check_defect_thermo(
        self,
        defect_thermo,
        defect_dict=None,
        dist_tol=1.5,
        chempots=None,
        el_refs=None,
        check_compatibility=True,
    ):
        if defect_dict is not None:
            self._compare_defect_thermo_and_dict(defect_thermo, defect_dict)

        print(defect_thermo)
        assert defect_thermo.dist_tol == dist_tol
        assert defect_thermo.chempots == chempots
        assert defect_thermo.el_refs == el_refs

        # CdTe, YTOS, Sb2Se3, Sb2Si2Te6, V2O5 values:
        assert any(np.isclose(defect_thermo.vbm, i, atol=1e-2) for i in [1.65, 3.26, 4.19, 6.60, 0.90])
        assert any(
            np.isclose(defect_thermo.band_gap, i, atol=1e-2) for i in [1.5, 0.7, 1.47, 0.44, 2.22]
        )  # note YTOS is GGA calcs so band gap underestimated

        assert defect_thermo.check_compatibility == check_compatibility

        # check methods run ok without errors:
        defect_thermo.to_json("test_thermo.json")
        reloaded_thermo = DefectThermodynamics.from_json("test_thermo.json")
        self._compare_defect_thermos(defect_thermo, reloaded_thermo)
        if_present_rm("test_thermo.json")

        defect_thermo.to_json()  # test default naming
        # CdTe, YTOS, Sb2Se3, Sb2Si2Te6, V2O5 values:
        assert any(
            os.path.exists(f"{i}_defect_thermodynamics.json")
            for i in ["CdTe", "Y2Ti2S2O5", "Sb2Se3", "SiSbTe3", "V2O5"]
        )
        assert defect_thermo.bulk_formula in ["CdTe", "Y2Ti2S2O5", "Sb2Se3", "SiSbTe3", "V2O5"]
        for i in ["CdTe", "Y2Ti2S2O5", "Sb2Se3", "SiSbTe3", "V2O5"]:
            if_present_rm(f"{i}_defect_thermodynamics.json")

        thermo_dict = defect_thermo.as_dict()
        dumpfn(thermo_dict, "test_thermo.json")
        reloaded_thermo = loadfn("test_thermo.json")
        self._compare_defect_thermos(defect_thermo, reloaded_thermo)
        if_present_rm("test_thermo.json")

        reloaded_thermo = DefectThermodynamics.from_dict(thermo_dict)
        self._compare_defect_thermos(defect_thermo, reloaded_thermo)

        assert all(
            i in defect_thermo.__repr__()
            for i in [
                "doped DefectThermodynamics for bulk composition",
                "Available attributes",
                "dist_tol",
                "vbm",
                "Available methods",
                "get_equilibrium_concentrations",
                "get_symmetries_and_degeneracies",
            ]
        )
        assert isinstance(defect_thermo.defect_names, list)
        assert isinstance(defect_thermo.all_stable_entries, list)
        assert isinstance(defect_thermo.all_unstable_entries, list)

        assert isinstance(defect_thermo.get_equilibrium_concentrations(), pd.DataFrame)
        assert isinstance(defect_thermo.get_symmetries_and_degeneracies(), pd.DataFrame)
        assert isinstance(defect_thermo.get_formation_energies(), (pd.DataFrame, list))
        tl_df = defect_thermo.get_transition_levels()
        assert set(tl_df.columns) == {"Charges", "Defect", "In Band Gap?", "eV from VBM"}
        all_tl_df = defect_thermo.get_transition_levels(all=True)
        assert set(all_tl_df.columns) == {
            "Charges",
            "Defect",
            "In Band Gap?",
            "eV from VBM",
            "N(Metastable)",
        }
        defect_thermo.print_transition_levels()
        defect_thermo.print_transition_levels(all=True)
        assert isinstance(defect_thermo.plot(), (mpl.figure.Figure, list))

        with self.assertRaises(TypeError) as exc:
            defect_thermo.get_equilibrium_fermi_level()
        assert "missing 1 required positional argument: 'bulk_dos_vr'" in str(exc.exception)

        with self.assertRaises(TypeError) as exc:
            defect_thermo.get_quenched_fermi_level_and_concentrations()
        assert "missing 1 required positional argument: 'bulk_dos_vr'" in str(exc.exception)

        if defect_thermo.chempots is None:
            with self.assertRaises(ValueError) as exc:
                defect_thermo.get_doping_windows()
            assert "No chemical potentials supplied or present" in str(exc.exception)
            assert "so doping windows cannot be calculated." in str(exc.exception)

            with self.assertRaises(ValueError) as exc:
                defect_thermo.get_dopability_limits()
            assert "No chemical potentials supplied or present" in str(exc.exception)
            assert "so dopability limits cannot be calculated." in str(exc.exception)

        else:
            assert isinstance(defect_thermo.get_doping_windows(), pd.DataFrame)
            assert isinstance(defect_thermo.get_dopability_limits(), pd.DataFrame)
            if "V2O5" in defect_thermo.bulk_formula:
                for df in [defect_thermo.get_doping_windows(), defect_thermo.get_dopability_limits()]:
                    assert set(df.columns).issubset(
                        {
                            "Compensating Defect",
                            "Dopability Limit",
                            "Facet",
                            "Doping Window",
                        }
                    )
                    assert set(df.index) == {"n-type", "p-type"}
                    assert df.shape == (2, 3)
                    assert set(df.loc["n-type"]).issubset({"N/A", -np.inf, np.inf})
                    assert set(df.loc["p-type"]).issubset({"N/A", -np.inf, np.inf})

            print(defect_thermo.get_doping_windows())
            print(defect_thermo.get_dopability_limits())

        # test setting dist_tol:
        if defect_thermo.bulk_formula == "CdTe" and defect_thermo.dist_tol == 1.5:
            self._check_CdTe_example_dist_tol(defect_thermo, 3)
        self._set_and_check_dist_tol(1.0, defect_thermo, 4)
        self._set_and_check_dist_tol(0.5, defect_thermo, 5)

    def _check_CdTe_example_dist_tol(self, defect_thermo, num_grouped_defects):
        print(f"Testing CdTe updated dist_tol: {defect_thermo.dist_tol}")
        tl_df = defect_thermo.get_transition_levels()
        assert tl_df.shape == (num_grouped_defects, 4)
        # number of entries in Figure label (i.e. grouped defects)
        assert len(defect_thermo.plot().get_axes()[0].get_legend().get_texts()) == num_grouped_defects

    def _set_and_check_dist_tol(self, dist_tol, defect_thermo, num_grouped_defects=0):
        defect_thermo.dist_tol = dist_tol
        assert defect_thermo.dist_tol == dist_tol
        if defect_thermo.bulk_formula == "CdTe":
            self._check_CdTe_example_dist_tol(defect_thermo, num_grouped_defects)
        defect_thermo.print_transition_levels()
        defect_thermo.print_transition_levels(all=True)

    def test_initialisation_and_attributes(self):
        """
        Test initialisation and attributes of DefectsThermodynamics.
        """
        for defect_dict, name in [
            (self.CdTe_defect_dict, "CdTe_defect_dict"),
            (self.YTOS_defect_dict, "YTOS_defect_dict"),
            (self.Sb2Se3_defect_dict, "Sb2Se3_defect_dict"),
            (self.Sb2Si2Te6_defect_dict, "Sb2Si2Te6_defect_dict"),
            (self.V2O5_defect_dict, "V2O5_defect_dict"),
        ]:
            print(f"Checking {name}")
            defect_thermo = DefectThermodynamics(list(defect_dict.values()))
            self._check_defect_thermo(defect_thermo, defect_dict)  # default values

            if "V2O5" in name:
                defect_thermo = DefectThermodynamics(
                    list(defect_dict.values()),
                    chempots=self.V2O5_chempots,
                    el_refs=self.V2O5_chempots["elemental_refs"],
                )
                self._check_defect_thermo(
                    defect_thermo,
                    defect_dict,
                    chempots=self.V2O5_chempots,
                    el_refs=self.V2O5_chempots["elemental_refs"],
                )

                defect_thermo = DefectThermodynamics(
                    list(defect_dict.values()), chempots=self.V2O5_chempots
                )
                self._check_defect_thermo(
                    defect_thermo,
                    defect_dict,
                    chempots=self.V2O5_chempots,
                    el_refs=self.V2O5_chempots["elemental_refs"],
                )

    def test_DefectsParser_thermo_objs(self):
        """
        Test the `DefectThermodynamics` objects created from the
        `DefectsParser.get_defect_thermodynamics()` method.
        """
        for defect_thermo, name in [
            (self.CdTe_defect_thermo, "CdTe_defect_thermo"),
            (self.YTOS_defect_thermo, "YTOS_defect_thermo"),
            (self.Sb2Se3_defect_thermo, "Sb2Se3_defect_thermo"),
            (self.Sb2Si2Te6_defect_thermo, "Sb2Si2Te6_defect_thermo"),
            (self.V2O5_defect_thermo, "V2O5_defect_thermo"),
        ]:
            print(f"Checking {name}")
            if "V2O5" in name:
                self._check_defect_thermo(
                    defect_thermo,
                    chempots=self.V2O5_chempots,
                    el_refs=self.V2O5_chempots["elemental_refs"],
                )

            else:
                self._check_defect_thermo(defect_thermo)  # default values

    def test_transition_levels_CdTe(self):
        """
        Test outputs of transition level functions for CdTe.
        """
        assert self.CdTe_defect_thermo.transition_level_map == {
            "v_Cd": {0.46988348089141413: [0, -2]},
            "Te_Cd": {},
            "Int_Te_3": {0.03497090517885537: [2, 1]},
        }

        tl_info = ["Defect: v_Cd", "Defect: Te_Cd"]
        tl_info_not_all = ["Transition level ε(0/-2) at 0.470 eV above the VBM"]
        tl_info_all = [
            "Transition level ε(-1*/-2) at 0.397 eV above the VBM",
            "Transition level ε(0/-1*) at 0.543 eV above the VBM",
            "*",
        ]

        result, tl_output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.print_transition_levels
        )
        assert not result
        assert not w
        for i in (
            tl_info
            + tl_info_not_all
            + [
                "Defect: Int_Te_3\x1b",
                "Transition level ε(+2/+1) at 0.035 eV above the VBM",
            ]
        ):
            assert i in tl_output
        for i in [*tl_info_all, "Int_Te_3_a", "*", "Int_Te_3_b", "Int_Te_3_Unperturbed"]:
            assert i not in tl_output

        result, tl_output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.print_transition_levels, all=True
        )
        assert not result
        assert not w
        for i in (
            tl_info
            + tl_info_all
            + [
                "Defect: Int_Te_3\x1b",
                "Transition level ε(+2/+1) at 0.035 eV above the VBM",
                "Transition level ε(+2/+1*) at 0.090 eV above the VBM",
            ]
        ):
            assert i in tl_output

        for i in tl_info_not_all:
            assert i not in tl_output

        self.CdTe_defect_thermo.dist_tol = 1
        result, tl_output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.print_transition_levels
        )
        assert not result
        assert not w
        for i in (
            tl_info
            + tl_info_not_all
            + [
                "Defect: Int_Te_3_a",
                "Defect: Int_Te_3_b",
                "Transition level ε(+2/+1) at 0.090 eV above the VBM",
            ]
        ):
            assert i in tl_output
        for i in [
            *tl_info_all,
            "Defect: Int_Te_3\x1b",
            "Transition level ε(+2/+1) at 0.035 eV above the VBM",
        ]:
            assert i not in tl_output

        result, tl_output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.print_transition_levels, all=True
        )
        assert not result
        assert not w
        for i in (
            tl_info
            + tl_info_all
            + [
                "Defect: Int_Te_3_a",
                "Defect: Int_Te_3_b",
                "Transition level ε(+2/+1) at 0.090 eV above the VBM",
            ]
        ):
            assert i in tl_output

        for i in [
            *tl_info_not_all,
            "Defect: Int_Te_3\x1b",
            "Transition level ε(+2/+1) at 0.035 eV above the VBM",
        ]:
            assert i not in tl_output

        self.CdTe_defect_thermo.dist_tol = 0.5
        result, tl_output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.print_transition_levels
        )
        assert not result
        assert not w
        for i in (
            tl_info
            + tl_info_not_all
            + [
                "Defect: Int_Te_3_a",
                "Defect: Int_Te_3_b",
                "Defect: Int_Te_3_Unperturbed",
            ]
        ):
            assert i in tl_output
        for i in [
            *tl_info_all,
            "Defect: Int_Te_3\x1b",
            "Transition level ε(+2/+1) at 0.090 eV above the VBM",
            "Transition level ε(+2/+1) at 0.035 eV above the VBM",
        ]:
            assert i not in tl_output

        result, tl_output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.print_transition_levels, all=True
        )
        assert not result
        assert not w
        for i in (
            tl_info
            + tl_info_all
            + [
                "Defect: Int_Te_3_a",
                "Defect: Int_Te_3_b",
                "Defect: Int_Te_3_Unperturbed",
            ]
        ):
            assert i in tl_output

        for i in [
            *tl_info_not_all,
            "Defect: Int_Te_3\x1b",
            "Transition level ε(+2/+1) at 0.035 eV above the VBM",
            "Transition level ε(+2/+1) at 0.090 eV above the VBM",
        ]:
            assert i not in tl_output

    def test_get_transition_levels_CdTe(self):
        tl_df = self.CdTe_defect_thermo.get_transition_levels()
        assert tl_df.shape == (3, 4)
        assert list(tl_df.iloc[0]) == ["v_Cd", "ε(0/-2)", 0.47, True]
        assert list(tl_df.iloc[1]) == ["Te_Cd", "None", np.inf, False]
        assert list(tl_df.iloc[2]) == ["Int_Te_3", "ε(+2/+1)", 0.035, True]

        tl_df = self.CdTe_defect_thermo.get_transition_levels(all=True)
        assert tl_df.shape == (5, 5)
        assert list(tl_df.iloc[0]) == ["v_Cd", "ε(-1*/-2)", 0.397, True, 1]
        assert list(tl_df.iloc[1]) == ["v_Cd", "ε(0/-1*)", 0.543, True, 1]
        assert list(tl_df.iloc[2]) == ["Te_Cd", "None", np.inf, False, 0]
        assert list(tl_df.iloc[3]) == ["Int_Te_3", "ε(+2/+1)", 0.035, True, 0]
        assert list(tl_df.iloc[4]) == ["Int_Te_3", "ε(+2/+1*)", 0.09, True, 1]

        self.CdTe_defect_thermo.dist_tol = 1
        tl_df = self.CdTe_defect_thermo.get_transition_levels()
        assert tl_df.shape == (4, 4)
        assert list(tl_df.iloc[2]) == ["Int_Te_3_a", "None", np.inf, False]
        assert list(tl_df.iloc[3]) == ["Int_Te_3_b", "ε(+2/+1)", 0.09, True]

        tl_df = self.CdTe_defect_thermo.get_transition_levels(all=True)
        assert tl_df.shape == (5, 5)
        assert list(tl_df.iloc[3]) == ["Int_Te_3_a", "None", np.inf, False, 0]
        assert list(tl_df.iloc[4]) == ["Int_Te_3_b", "ε(+2/+1)", 0.09, True, 0]

        self.CdTe_defect_thermo.dist_tol = 0.5
        tl_df = self.CdTe_defect_thermo.get_transition_levels()
        assert tl_df.shape == (5, 4)
        assert list(tl_df.iloc[2]) == ["Int_Te_3_Unperturbed", "None", np.inf, False]
        assert list(tl_df.iloc[3]) == ["Int_Te_3_a", "None", np.inf, False]
        assert list(tl_df.iloc[4]) == ["Int_Te_3_b", "None", np.inf, False]

        tl_df = self.CdTe_defect_thermo.get_transition_levels(all=True)
        assert tl_df.shape == (6, 5)
        assert list(tl_df.iloc[3]) == ["Int_Te_3_Unperturbed", "None", np.inf, False, 0]
        assert list(tl_df.iloc[4]) == ["Int_Te_3_a", "None", np.inf, False, 0]
        assert list(tl_df.iloc[5]) == ["Int_Te_3_b", "None", np.inf, False, 0]

    def test_get_symmetries_degeneracies_CdTe(self):
        sym_degen_df = self.CdTe_defect_thermo.get_symmetries_and_degeneracies()
        print(sym_degen_df)
        assert sym_degen_df.shape == (7, 8)
        assert list(sym_degen_df.columns) == [
            "Defect",
            "q",
            "Site_Symm",
            "Defect_Symm",
            "g_Orient",
            "g_Spin",
            "g_Total",
            "Mult",
        ]
        # hardcoded tests to ensure ordering is consistent (by defect type according to
        # _sort_defect_entries, then by charge state from left (most positive) to right (most negative),
        # as would appear on a TL diagram)
        cdte_sym_degen_lists = [
            ["v_Cd", "0", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["v_Cd", "-1", "Td", "C3v", 4.0, 2, 8.0, 1.0],
            ["v_Cd", "-2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["Te_Cd", "+1", "Td", "C3v", 4.0, 2, 8.0, 1.0],
            ["Int_Te_3", "+2", "C1", "Cs", 0.5, 1, 0.5, 24.0],
            ["Int_Te_3", "+1", "Cs", "C2v", 0.5, 2, 1.0, 12.0],
            ["Int_Te_3_Unperturbed", "+1", "C1", "Cs", 0.5, 2, 1.0, 24.0],
        ]
        for i, row in enumerate(cdte_sym_degen_lists):
            print(i, row)
            assert list(sym_degen_df.iloc[i]) == row

        non_formatted_sym_degen_df = self.CdTe_defect_thermo.get_symmetries_and_degeneracies(
            skip_formatting=True
        )
        print(non_formatted_sym_degen_df)  # for debugging
        for i, row in enumerate(cdte_sym_degen_lists):
            row[1] = int(row[1])
            assert list(non_formatted_sym_degen_df.iloc[i]) == row

    def _check_form_en_df_total(self, form_en_df):
        """
        Check the sum of formation energy terms equals the total formation
        energy in the `get_formation_energies` DataFrame.
        """
        columns_to_sum = form_en_df.iloc[:, 2:8]
        # ignore strings if chempots is "N/A":
        numeric_columns = columns_to_sum.apply(pd.to_numeric, errors="coerce")
        print(f"Numeric columns: {numeric_columns}")  # for debugging

        # assert the sum of formation energy terms equals the total formation energy:
        for i, _row in enumerate(form_en_df.iterrows()):
            assert np.isclose(numeric_columns.iloc[i].sum(), form_en_df.iloc[i]["ΔEᶠᵒʳᵐ"], atol=1e-3)

    def test_formation_energies_CdTe(self):
        """
        Here we test the ``DefectThermodynamics.get_formation_energies``,
        ``DefectThermodynamics.get_formation_energy``, and
        ``DefectEntry.formation_energy`` methods.
        """
        form_en_df_cols = [
            "Defect",
            "q",
            "ΔEʳᵃʷ",
            "qE_VBM",
            "qE_F",
            "Σμ_ref",
            "Σμ_formal",
            "E_corr",
            "ΔEᶠᵒʳᵐ",
            "Path",
        ]

        form_en_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies
        )

        self._check_chempot_w_and_fermi_message(w, output)
        assert form_en_df.shape == (7, 10)
        assert list(form_en_df.columns) == form_en_df_cols
        # hardcoded tests to ensure ordering is consistent:
        cdte_form_en_lists = [
            [
                "v_Cd",
                "0",
                4.166,
                0.0,
                0.0,
                "N/A",
                0,
                0.0,
                4.166,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_0/vasp_ncl",
            ],
            [
                "v_Cd",
                "-1",
                6.13,
                -1.646,
                -0.749,
                "N/A",
                0,
                0.225,
                3.959,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-1/vasp_ncl",
            ],
            [
                "v_Cd",
                "-2",
                7.661,
                -3.293,
                -1.499,
                "N/A",
                0,
                0.738,
                3.607,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl",
            ],
            [
                "Te_Cd",
                "+1",
                -2.906,
                1.646,
                0.749,
                "N/A",
                0,
                0.238,
                -0.272,
                f"{self.CdTe_EXAMPLE_DIR}/Te_Cd_+1/vasp_ncl",
            ],
            [
                "Int_Te_3",
                "+2",
                -7.105,
                3.293,
                1.499,
                "N/A",
                0,
                0.904,
                -1.409,
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
            ],
            [
                "Int_Te_3",
                "+1",
                -4.819,
                1.646,
                0.749,
                "N/A",
                0,
                0.3,
                -2.123,
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_1/vasp_ncl",
            ],
            [
                "Int_Te_3_Unperturbed",
                "+1",
                -4.762,
                1.646,
                0.749,
                "N/A",
                0,
                0.297,
                -2.069,
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_Unperturbed_1/vasp_ncl",
            ],
        ]
        self._check_form_en_df_total(form_en_df)  # test sum of formation energy terms equals total

        def _check_formation_energy_methods(form_en_df_row, thermo_obj, fermi_level):
            defect_name_w_charge_state = f"{form_en_df_row[0]}_{int(form_en_df_row[1])}"
            print(defect_name_w_charge_state)  # for debugging
            defect_entry = [
                entry
                for entry in thermo_obj.defect_entries
                if entry.name.rsplit("_", 1)[0] == defect_name_w_charge_state.rsplit("_", 1)[0]
                and entry.charge_state == int(defect_name_w_charge_state.rsplit("_", 1)[1])
            ][0]

            for name, entry in [
                ("string", defect_name_w_charge_state),
                ("DefectEntry", defect_entry),
            ]:
                print(f"Testing formation energy methods with {name} input")  # for debugging
                if thermo_obj.chempots is None:
                    facets = [
                        None,
                    ]
                    with warnings.catch_warnings(record=True) as w:
                        _form_en = thermo_obj.get_formation_energy(entry, facet="test pop b...")
                    assert len(w) == 2
                    assert "No chemical potentials supplied" in str(w[0].message)
                    assert (
                        "You have specified a facet (chemical potential limit) but no chemical potentials "
                        "(`chempots`) were supplied, so `facet` will be ignored." in str(w[-1].message)
                    )

                elif len(thermo_obj.chempots["facets_wrt_el_refs"]) == 2:  # CdTe full chempots
                    with warnings.catch_warnings(record=True) as w:
                        assert (
                            not np.isclose(  # if chempots present, uses the first facet which is Cd-rich
                                thermo_obj.get_formation_energy(entry),
                                form_en_df_row[8],
                                atol=1e-3,
                            )
                        )
                    assert len(w) == 1
                    assert (
                        "No facet (chemical potential limit) specified! Using Cd-CdTe for computing "
                        "the formation energy" in str(w[0].message)
                    )
                    facets = ["CdTe-Te", "Te-rich"]
                else:
                    facets = list(thermo_obj.chempots["facets_wrt_el_refs"].keys())  # user supplied

                for facet in facets:
                    if np.isclose(fermi_level, 0.75):  # default CdTe:
                        assert np.isclose(  # test get_formation_energy method
                            thermo_obj.get_formation_energy(entry, facet=facet),
                            form_en_df_row[8],
                            atol=1e-3,
                        )
                    assert np.isclose(  # test get_formation_energy method
                        thermo_obj.get_formation_energy(entry, facet=facet, fermi_level=fermi_level),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    # Test usage of ``DefectThermodynamics.get_formation_energy()`` where charge state
                    # isn't specified:
                    lowest_e_form = thermo_obj.get_formation_energy(
                        form_en_df_row[0], facet=facet, fermi_level=fermi_level
                    )
                    assert np.isclose(  # test get_formation_energy method
                        lowest_e_form,
                        min(
                            [
                                thermo_obj.get_formation_energy(
                                    entry, facet=facet, fermi_level=fermi_level
                                )
                                for entry in thermo_obj.defect_entries
                                if form_en_df_row[0] in entry.name
                            ]
                        ),
                    )

                    assert np.isclose(  # test get_formation_energy() method
                        thermo_obj.get_formation_energy(
                            entry, fermi_level=fermi_level, facet=facet, chempots=thermo_obj.chempots
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    lowest_e_form = thermo_obj.get_formation_energy(
                        form_en_df_row[0],
                        facet=facet,
                        fermi_level=fermi_level,
                        chempots=thermo_obj.chempots,
                    )
                    assert np.isclose(  # test get_formation_energy method
                        lowest_e_form,
                        min(
                            [
                                thermo_obj.get_formation_energy(
                                    entry,
                                    facet=facet,
                                    fermi_level=fermi_level,
                                    chempots=thermo_obj.chempots,
                                )
                                for entry in thermo_obj.defect_entries
                                if form_en_df_row[0] in entry.name
                            ]
                        ),
                    )

                    # test DefectEntry.formation_energy() method:
                    assert np.isclose(
                        defect_entry.formation_energy(
                            fermi_level=fermi_level, facet=facet, chempots=thermo_obj.chempots
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    assert np.isclose(  # test DefectEntry.formation_energy() method
                        defect_entry.formation_energy(
                            fermi_level=fermi_level,
                            vbm=thermo_obj.vbm,
                            facet=facet,
                            chempots=thermo_obj.chempots,
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    assert np.isclose(  # test DefectEntry.formation_energy() method
                        defect_entry.formation_energy(
                            fermi_level=fermi_level + 0.1,
                            vbm=thermo_obj.vbm - 0.1,
                            facet=facet,
                            chempots=thermo_obj.chempots,
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    if thermo_obj.chempots and "rich" not in facet:  # needs to be 'CdTe-Te' etc for
                        # sub-selecting like this
                        assert np.isclose(  # test DefectEntry.formation_energy() method
                            defect_entry.formation_energy(
                                fermi_level=fermi_level,
                                chempots=thermo_obj.chempots["facets_wrt_el_refs"][facet],
                                el_refs=thermo_obj.chempots["elemental_refs"],
                            ),
                            form_en_df_row[8],
                            atol=1e-3,
                        )
                        assert np.isclose(  # test DefectThermodynamics.get_formation_energy() method
                            thermo_obj.get_formation_energy(
                                entry,
                                fermi_level=fermi_level,
                                chempots=thermo_obj.chempots["facets_wrt_el_refs"][facet],
                                el_refs=thermo_obj.chempots["elemental_refs"],
                            ),
                            form_en_df_row[8],
                            atol=1e-3,
                        )

        for i, row in enumerate(cdte_form_en_lists):
            assert list(form_en_df.iloc[i]) == row
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)  # default mid-gap value

        with self.assertRaises(ValueError) as exc:
            self.CdTe_defect_thermo.get_formation_energy("v_Cd_3")
        assert "No matching DefectEntry with v_Cd_3 in name found in " in str(exc.exception)
        assert "DefectThermodynamics.defect_entries, which have names:" in str(exc.exception)
        assert (
            "['v_Cd_0', 'v_Cd_-1', 'v_Cd_-2', 'Te_Cd_+1', 'Int_Te_3_2', 'Int_Te_3_1', "
            "'Int_Te_3_Unperturbed_1']" in str(exc.exception)
        )

        for i, defect_entry in enumerate(self.CdTe_defect_thermo.defect_entries):
            new_entry = deepcopy(defect_entry)
            _vbm = new_entry.calculation_metadata.pop("vbm")
            assert np.isclose(
                new_entry.formation_energy(fermi_level=0.7493, vbm=self.CdTe_defect_thermo.vbm),
                form_en_df.iloc[i][8],
                atol=1e-3,
            )
            with warnings.catch_warnings(record=True) as w:
                _form_en = new_entry.formation_energy(fermi_level=0.7493)
            print([str(i.message) for i in w])  # for debugging
            if new_entry.charge_state != 0:
                assert "No chemical potentials supplied" in str(w[0].message)
                assert (
                    "VBM eigenvalue was not set, and is not present in "
                    "DefectEntry.calculation_metadata. Formation energy will be inaccurate!"
                ) in str(w[-1].message)
            else:
                print(f"No VBM warnings for {new_entry.name}")  # v_Cd_0
                assert "No chemical potentials supplied" in str(w[0].message)
                assert not any("VBM eigenvalue was not set" in str(i.message) for i in w)

        non_formatted_form_en_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies, skip_formatting=True
        )

        self._check_chempot_w_and_fermi_message(w, output)
        for i, row in enumerate(cdte_form_en_lists):
            row[1] = int(row[1])
            assert list(non_formatted_form_en_df.iloc[i]) == row  # and all other terms the same
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)  # default mid-gap value

        # with chempots:
        list_of_dfs, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies, self.CdTe_chempots
        )
        assert isinstance(list_of_dfs, list)
        assert len(list_of_dfs) == 2
        assert not w
        assert "Fermi level was not set" in output

        te_rich_df = list_of_dfs[1]  # Te-rich
        for facet in ["Te-rich", "CdTe-Te"]:
            df, output, w = _run_func_and_capture_stdout_warnings(
                self.CdTe_defect_thermo.get_formation_energies, self.CdTe_chempots, facet=facet
            )
            self._check_no_w_fermi_message_and_new_matches_ref_df(output, w, df, te_rich_df)
            assert df.shape == (7, 10)
            assert list(df.columns) == form_en_df_cols

        list_of_dfs, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies,
            chempots=self.CdTe_chempots,
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        assert "Fermi level was not set" in output
        assert not w
        assert list_of_dfs[1].equals(te_rich_df)

        manual_te_rich_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies,
            chempots=self.CdTe_chempots["facets_wrt_el_refs"]["CdTe-Te"],
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        self._check_no_w_fermi_message_and_new_matches_ref_df(output, w, manual_te_rich_df, te_rich_df)
        manual_te_rich_df_w_fermi, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies,
            chempots=self.CdTe_chempots["facets_wrt_el_refs"]["CdTe-Te"],
            el_refs=self.CdTe_chempots["elemental_refs"],
            fermi_level=0.7493,  # default mid-gap value
        )
        assert "Fermi level was not set" not in output
        assert not w
        assert manual_te_rich_df_w_fermi.equals(te_rich_df)

        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        preset_te_rich_dfs, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies
        )
        assert "Fermi level was not set" in output
        assert not w
        assert preset_te_rich_dfs[1].equals(te_rich_df)

        cdte_te_rich_form_en_lists = [
            [
                "v_Cd",
                "0",
                4.166,
                0.0,
                0.0,
                -1.016,
                -1.251,
                0.0,
                1.899,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_0/vasp_ncl",
            ],
            [
                "v_Cd",
                "-1",
                6.13,
                -1.646,
                -0.749,
                -1.016,
                -1.251,
                0.225,
                1.692,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-1/vasp_ncl",
            ],
            [
                "v_Cd",
                "-2",
                7.661,
                -3.293,
                -1.499,
                -1.016,
                -1.251,
                0.738,
                1.34,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl",
            ],
            [
                "Te_Cd",
                "+1",
                -2.906,
                1.646,
                0.749,
                3.455,
                -1.251,
                0.238,
                1.932,
                f"{self.CdTe_EXAMPLE_DIR}/Te_Cd_+1/vasp_ncl",
            ],
            [
                "Int_Te_3",
                "+2",
                -7.105,
                3.293,
                1.499,
                4.471,
                0.0,
                0.904,
                3.062,
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
            ],
            [
                "Int_Te_3",
                "+1",
                -4.819,
                1.646,
                0.749,
                4.471,
                0.0,
                0.3,
                2.347,
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_1/vasp_ncl",
            ],
            [
                "Int_Te_3_Unperturbed",
                "+1",
                -4.762,
                1.646,
                0.749,
                4.471,
                0.0,
                0.297,
                2.402,
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_Unperturbed_1/vasp_ncl",
            ],
        ]

        self._check_form_en_df_total(te_rich_df)  # test sum of formation energy terms equals total

        for i, row in enumerate(cdte_te_rich_form_en_lists):
            assert list(te_rich_df.iloc[i]) == row
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)  # default mid-gap
            # value, chempots also now attached

        # test same non-formatted output with chempots:
        list_of_dfs, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies, self.CdTe_chempots, skip_formatting=True
        )
        assert not w
        assert "Fermi level was not set" in output
        non_formatted_te_rich_df = list_of_dfs[1]  # Te-rich
        for facet in ["Te-rich", "CdTe-Te"]:
            df, output, w = _run_func_and_capture_stdout_warnings(
                self.CdTe_defect_thermo.get_formation_energies,
                self.CdTe_chempots,
                facet=facet,
                skip_formatting=True,
            )
            self._check_no_w_fermi_message_and_new_matches_ref_df(output, w, df, non_formatted_te_rich_df)
        for i, row in enumerate(cdte_te_rich_form_en_lists):
            row[1] = int(row[1])
            assert list(non_formatted_te_rich_df.iloc[i]) == row
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)

        # hard test random case with random chempots, el_refs and fermi level
        manual_form_en_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies,
            chempots={"Te": 3, "Cd": -1},
            fermi_level=3,
            el_refs={"Cd": -12, "Te": -1},
        )
        assert "Fermi level was not set" not in output
        assert not w
        assert manual_form_en_df.shape == (7, 10)
        self._check_form_en_df_total(manual_form_en_df)  # test sum of formation energy terms equals total

        # first 4 columns the same, then 3 diff, one the same (E_corr), E_form diff, path the same:
        manual_thermo = deepcopy(self.CdTe_defect_thermo)
        manual_thermo.chempots = {"Te": 3, "Cd": -1}
        manual_thermo.el_refs = {"Cd": -12, "Te": -1}
        for i, row in manual_form_en_df.iterrows():
            assert list(row)[:4] == list(te_rich_df.iloc[i])[:4]
            assert list(row)[4:7] != list(te_rich_df.iloc[i])[4:7]
            assert list(row)[7] == list(te_rich_df.iloc[i])[7]
            assert list(row)[8] != list(te_rich_df.iloc[i])[8]
            assert list(row)[9] == list(te_rich_df.iloc[i])[9]
            _check_formation_energy_methods(row, manual_thermo, 3)

        assert list(manual_form_en_df.iloc[2]) == [
            "v_Cd",
            "-2",
            7.661,
            -3.293,
            -6,
            -12,
            -1,
            0.738,
            -13.895,
            f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl",
        ]
        assert list(manual_form_en_df.iloc[4]) == [
            "Int_Te_3",
            "+2",
            -7.105,
            3.293,
            6,
            1,
            -3,
            0.904,
            1.092,
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
        ]

    def _check_no_w_fermi_message_and_new_matches_ref_df(self, output, w, new_df, ref_df):
        assert "Fermi level was not set" in output
        assert not w
        assert new_df.equals(ref_df)

    def _check_chempot_w_and_fermi_message(self, w, output):
        assert len(w) == 1
        assert (
            "No chemical potentials supplied, so using 0 for all chemical potentials. Formation "
            "energies (and concentrations) will likely be highly inaccurate!"
        ) in str(w[0].message)

        assert (
            "Fermi level was not set, so using mid-gap Fermi level (E_g/2 = 0.75 eV relative to the "
            "VBM)."
        ) in output

    def _check_chempots_dict(self, chempots_dict):
        # in the chempots dict, for each subdict in chempots["facets"], it should match the sum of
        # the chempots["facets_wrt_el_refs"] and chempots["elemental_refs"]:
        for facet, subdict in chempots_dict["facets"].items():
            for el, mu in subdict.items():
                assert np.isclose(
                    mu,
                    chempots_dict["facets_wrt_el_refs"][facet][el] + chempots_dict["elemental_refs"][el],
                )

    def test_parse_chempots_CdTe(self):
        """
        Testing different combos of setting `chempots` and `el_refs`.
        """
        # Note that `_parse_chempots()` has also been indirectly tested via the plotting tests
        orig_cdte_defect_thermo = deepcopy(self.CdTe_defect_thermo)
        assert not self.CdTe_defect_thermo.chempots
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.chempots == self.CdTe_chempots
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.chempots == self.CdTe_chempots  # the same
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # the same

        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}  # Te-rich
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        semi_manual_chempots_dict = {
            "facets_wrt_el_refs": {"User Chemical Potentials": {"Cd": -1.25, "Te": 0}},
            "elemental_refs": {"Te": -4.47069234, "Cd": -1.01586484},
            "facets": {"User Chemical Potentials": {"Cd": -2.26586484, "Te": -4.47069234}},
        }
        assert self.CdTe_defect_thermo.chempots == semi_manual_chempots_dict
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.chempots == semi_manual_chempots_dict  # the same
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # the same

        self.CdTe_defect_thermo.chempots = None
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        manual_zeroed_rel_chempots_dict = deepcopy(semi_manual_chempots_dict)
        manual_zeroed_rel_chempots_dict["facets_wrt_el_refs"]["User Chemical Potentials"] = {
            "Cd": 0,
            "Te": 0,
        }
        manual_zeroed_rel_chempots_dict["facets"]["User Chemical Potentials"] = self.CdTe_chempots[
            "elemental_refs"
        ]
        assert self.CdTe_defect_thermo.chempots == manual_zeroed_rel_chempots_dict
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # unchanged
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # the same
        assert self.CdTe_defect_thermo.chempots == manual_zeroed_rel_chempots_dict  # the same

        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}  # Te-rich
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.chempots == semi_manual_chempots_dict

        defect_thermo = deepcopy(orig_cdte_defect_thermo)
        defect_thermo.chempots = {"Cd": -1.25, "Te": 0}  # Te-rich
        self._check_chempots_dict(defect_thermo.chempots)
        zero_el_refs_te_rich_chempots_dict = {
            "facets_wrt_el_refs": {"User Chemical Potentials": {"Cd": -1.25, "Te": 0}},
            "elemental_refs": {"Te": 0, "Cd": 0},
            "facets": {"User Chemical Potentials": {"Cd": -1.25, "Te": 0}},
        }
        assert defect_thermo.chempots == zero_el_refs_te_rich_chempots_dict
        assert defect_thermo.el_refs == {"Te": 0, "Cd": 0}
        defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == semi_manual_chempots_dict
        assert defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        defect_thermo = deepcopy(orig_cdte_defect_thermo)
        defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == manual_zeroed_rel_chempots_dict
        assert defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        defect_thermo.chempots = {"Cd": -1.25, "Te": 0}  # Te-rich
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == semi_manual_chempots_dict
        assert defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        # test including chempots/el_refs when initialised:
        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()),
            chempots={"Cd": -1.25, "Te": 0},
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == semi_manual_chempots_dict
        assert defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()), chempots=self.CdTe_chempots
        )
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == self.CdTe_chempots
        assert defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()),
            chempots={"Cd": -1.25, "Te": 0},
        )
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == zero_el_refs_te_rich_chempots_dict
        assert defect_thermo.el_refs == {"Te": 0, "Cd": 0}
        defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == semi_manual_chempots_dict

        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()),
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots == manual_zeroed_rel_chempots_dict
        assert defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

    def test_add_entries(self):
        partial_defect_thermo = DefectThermodynamics(list(self.CdTe_defect_dict.values())[:4])
        assert not partial_defect_thermo.get_formation_energies().equals(
            self.CdTe_defect_thermo.get_formation_energies()
        )
        assert not partial_defect_thermo.get_symmetries_and_degeneracies().equals(
            self.CdTe_defect_thermo.get_symmetries_and_degeneracies()
        )

        partial_defect_thermo.add_entries(list(self.CdTe_defect_dict.values())[4:])
        assert partial_defect_thermo.get_formation_energies().equals(
            self.CdTe_defect_thermo.get_formation_energies()
        )
        assert partial_defect_thermo.get_symmetries_and_degeneracies().equals(
            self.CdTe_defect_thermo.get_symmetries_and_degeneracies()
        )


class DefectThermodynamicsPlotsTestCase(unittest.TestCase):
    def setUp(self):
        DefectThermodynamicsTestCase.setUp(self)  # get attributes from DefectThermodynamicsTestCase

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(facet="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_specified_chempots(self):
        return self.CdTe_defect_thermo.plot(chempots=self.CdTe_chempots, facet="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots(self):
        return self.CdTe_defect_thermo.plot(
            chempots={"Cd": -1.25, "Te": 0}, el_refs=self.CdTe_chempots["elemental_refs"]
        )

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_at_init(self):
        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()),
            chempots={"Cd": -1.25, "Te": 0},
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        return defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1(self):
        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}
        return self.CdTe_defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other2(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot(chempots={"Cd": -1.25, "Te": 0})

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other3(self):
        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}
        return self.CdTe_defect_thermo.plot(el_refs=self.CdTe_chempots["elemental_refs"])

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_edited_el_refs(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot(facet="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_edited_el_refs_other(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(facet="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_Cd_rich.png")
    def test_default_CdTe_plot_Cd_rich(self):
        return self.CdTe_defect_thermo.plot(chempots=self.CdTe_chempots, facet="Cd-rich")


# TODO: Test check_compatibility
# TODO: Save all defects in CdTe thermo to JSON and test methods on it
# TODO: Test warnings for failed symmetry determination with periodicity-breaking Sb2Si2Te6 and ZnS (and
#  no warnings with CdTe, Sb2Se3, YTOS)
# TODO: Spot check one or two DefectEntry concentration methods
# TODO: Add V2O5 test plotting all lines
# TODO: Add GGA MgO tests as well
