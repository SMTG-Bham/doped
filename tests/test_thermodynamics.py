"""
Tests for doped.thermodynamics module.
"""

import os
import shutil
import sys
import unittest
import warnings
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


class DefectThermodynamicsTestCase(unittest.TestCase):
    def setUp(self):
        for thermo, name in [
            (self.CdTe_defect_thermo, "CdTe_defect_thermo"),
            (self.YTOS_defect_thermo, "YTOS_defect_thermo"),
            (self.Sb2Se3_defect_thermo, "Sb2Se3_defect_thermo"),
            (self.Sb2Si2Te6_defect_thermo, "Sb2Si2Te6_defect_thermo"),
            (self.V2O5_defect_thermo, "V2O5_defect_thermo"),
        ]:
            thermo.dist_tol = 1.5  # ensure dist_tol at default value
            if "V2O5" not in name:
                thermo.chempots = None  # reset to default value
                thermo.el_refs = None  # reset to default value

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

        cls.CdTe_defect_dict = loadfn(os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_example_defect_dict.json"))
        cls.CdTe_defect_thermo = loadfn(os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"))
        cls.YTOS_defect_dict = loadfn(os.path.join(cls.YTOS_EXAMPLE_DIR, "YTOS_example_defect_dict.json"))
        cls.YTOS_defect_thermo = loadfn(os.path.join(cls.YTOS_EXAMPLE_DIR, "YTOS_example_thermo.json"))
        cls.Sb2Se3_defect_dict = loadfn(
            os.path.join(cls.Sb2Se3_DATA_DIR, "defect/Sb2Se3_O_example_defect_dict.json")
        )
        cls.Sb2Se3_defect_thermo = loadfn(
            os.path.join(cls.Sb2Se3_DATA_DIR, "Sb2Se3_O_example_thermo.json")
        )
        cls.Sb2Si2Te6_defect_dict = loadfn(
            os.path.join(cls.Sb2Si2Te6_DATA_DIR, "Sb2Si2Te6_example_defect_dict.json")
        )
        cls.Sb2Si2Te6_defect_thermo = loadfn(
            os.path.join(cls.Sb2Si2Te6_DATA_DIR, "Sb2Si2Te6_example_thermo.json")
        )

        cls.V2O5_defect_dict = loadfn(os.path.join(cls.V2O5_DATA_DIR, "V2O5_example_defect_dict.json"))
        cls.V2O5_defect_thermo = loadfn(os.path.join(cls.V2O5_DATA_DIR, "V2O5_example_thermo.json"))
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

        def _capture_transition_level_print(defect_thermo, **kwargs):
            """
            Capture the printed output of the transition levels.
            """
            original_stdout = sys.stdout  # Save a reference to the original standard output
            sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
            w = None
            try:
                with warnings.catch_warnings(record=True) as w:
                    defect_thermo.print_transition_levels(**kwargs)
                output = sys.stdout.getvalue()  # Return a str containing the printed output
            finally:
                sys.stdout = original_stdout  # Reset standard output to its original value.

            if w:
                print([str(warning.message) for warning in w])  # for debugging
            assert not w
            return output

        tl_info = ["Defect: v_Cd", "Defect: Te_Cd"]
        tl_info_not_all = ["Transition level ε(0/-2) at 0.470 eV above the VBM"]
        tl_info_all = [
            "Transition level ε(-1*/-2) at 0.397 eV above the VBM",
            "Transition level ε(0/-1*) at 0.543 eV above the VBM",
            "*",
        ]

        tl_output = _capture_transition_level_print(self.CdTe_defect_thermo)
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

        tl_output = _capture_transition_level_print(self.CdTe_defect_thermo, all=True)
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
        tl_output = _capture_transition_level_print(self.CdTe_defect_thermo)
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

        tl_output = _capture_transition_level_print(self.CdTe_defect_thermo, all=True)
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
        tl_output = _capture_transition_level_print(self.CdTe_defect_thermo)
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

        tl_output = _capture_transition_level_print(self.CdTe_defect_thermo, all=True)
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
        for i, row in enumerate(cdte_sym_degen_lists):
            row[1] = int(row[1])
            assert list(non_formatted_sym_degen_df.iloc[i]) == row


class DefectThermodynamicsPlotsTestCase(DefectThermodynamicsTestCase):
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

    # def test_add_entries(self):


# TODO: Save all defects in CdTe thermo to JSON and test methods on it
# TODO: Test warnings for failed symmetry determination with periodicity-breaking Sb2Si2Te6 and ZnS (and
#  no warnings with CdTe, Sb2Se3, YTOS)
# TODO: Test DefectEntry formation energy and concentration methods
# TODO: Test add entries, check_compatibility
# TODO: Add V2O5 test plotting all lines
# TODO: Move over all symmetry/degeneracy thermo tests from test_analysis.py to here
# TODO: Add GGA MgO tests as well
