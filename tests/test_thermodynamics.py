"""
Tests for doped.thermodynamics module.
"""

import os
import shutil
import unittest

import matplotlib as mpl
import numpy as np
import pandas as pd
from monty.serialization import dumpfn, loadfn

from doped.thermodynamics import DefectThermodynamics

# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")
mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


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
    @classmethod
    def setUpClass(cls):
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.EXAMPLE_DIR = os.path.join(cls.module_path, "../examples")
        cls.CdTe_EXAMPLE_DIR = os.path.join(cls.module_path, "../examples/CdTe")
        cls.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

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
        assert isinstance(defect_thermo.get_formation_energies(), pd.DataFrame)
        assert isinstance(defect_thermo.get_transition_levels(), pd.DataFrame)
        defect_thermo.print_transition_levels()
        assert isinstance(defect_thermo.plot(), mpl.figure.Figure)

        with self.assertRaises(TypeError) as exc:
            defect_thermo.get_equilibrium_fermi_level()
        assert "missing 1 required positional argument: 'bulk_dos_vr'" in str(exc.exception)

        with self.assertRaises(TypeError) as exc:
            defect_thermo.get_quenched_fermi_level_and_concentrations()
        assert "missing 1 required positional argument: 'bulk_dos_vr'" in str(exc.exception)

        with self.assertRaises(ValueError) as exc:
            defect_thermo.get_doping_windows()
        assert "No chemical potentials supplied or present" in str(exc.exception)
        assert "so doping windows cannot be calculated." in str(exc.exception)

        with self.assertRaises(ValueError) as exc:
            defect_thermo.get_dopability_limits()
        assert "No chemical potentials supplied or present" in str(exc.exception)
        assert "so dopability limits cannot be calculated." in str(exc.exception)

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


# TODO: Test add entries, check_compatibility
# TODO: Add V2O5 test plotting all lines
# TODO: Move over all symmetry/degeneracy thermo tests from test_analysis.py to here
# TODO: Add GGA MgO tests as well
