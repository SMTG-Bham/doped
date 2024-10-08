"""
Tests for doped.thermodynamics module.
"""

import os
import random
import shutil
import sys
import unittest
import warnings
from copy import deepcopy
from functools import wraps
from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.core.composition import Composition
from pymatgen.electronic_structure.dos import FermiDos

from doped.generation import _sort_defect_entries
from doped.thermodynamics import DefectThermodynamics, get_fermi_dos, scissor_dos
from doped.utils.parsing import _get_defect_supercell_bulk_site_coords, get_vasprun
from doped.utils.symmetry import get_sga, point_symmetry_from_site, point_symmetry_from_structure

# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")

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


class DefectThermodynamicsSetupMixin(unittest.TestCase):
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
        self.MgO_defect_thermo = deepcopy(self.orig_MgO_defect_thermo)
        self.MgO_defect_dict = deepcopy(self.orig_MgO_defect_dict)
        self.Sb2O5_defect_thermo = deepcopy(self.orig_Sb2O5_defect_thermo)
        self.ZnS_defect_thermo = deepcopy(self.orig_ZnS_defect_thermo)

        self.Se_ext_no_pnict_thermo = deepcopy(self.orig_Se_ext_no_pnict_thermo)
        self.Se_pnict_thermo = deepcopy(self.orig_Se_pnict_thermo)
        self.cdte_chempot_warning_message = (
            "Note that the raw (DFT) energy of the bulk supercell calculation (-3.37 eV/atom) differs "
            "from that expected from the supplied chemical potentials (-3.50 eV/atom) by >0.025 eV. This "
            "will likely give inaccuracies of similar magnitude in the predicted formation energies! "
            "\nYou can suppress this warning by setting `DefectThermodynamics.check_compatibility = "
            "False`."
        )

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

        cls.MgO_EXAMPLE_DIR = os.path.join(cls.EXAMPLE_DIR, "MgO")

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

        cls.orig_MgO_defect_thermo = loadfn(os.path.join(cls.MgO_EXAMPLE_DIR, "MgO_thermo.json.gz"))
        cls.orig_MgO_defect_dict = loadfn(os.path.join(cls.MgO_EXAMPLE_DIR, "MgO_defect_dict.json.gz"))
        cls.MgO_chempots = loadfn(os.path.join(cls.EXAMPLE_DIR, "CompetingPhases/mgo_chempots.json"))

        cls.Sb2O5_chempots = loadfn(os.path.join(data_dir, "Sb2O5/Sb2O5_chempots.json"))
        cls.orig_Sb2O5_defect_thermo = loadfn(os.path.join(data_dir, "Sb2O5/Sb2O5_thermo.json.gz"))

        cls.orig_ZnS_defect_thermo = loadfn(os.path.join(data_dir, "ZnS/ZnS_thermo.json"))

        cls.orig_Se_ext_no_pnict_thermo = loadfn(os.path.join(data_dir, "Se_Ext_No_Pnict_Thermo.json.gz"))
        cls.orig_Se_pnict_thermo = loadfn(os.path.join(data_dir, "Se_Pnict_Thermo.json.gz"))

        # cls.CdTe_fermi_dos = get_fermi_dos(
        #     os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_prim_k181818_NKRED_2_vasprun.xml.gz")
        # )  # not used twice yet


class DefectThermodynamicsTestCase(DefectThermodynamicsSetupMixin):
    def _compare_defect_thermo_and_dict(self, defect_thermo, defect_dict):
        assert len(defect_thermo.defect_entries) == len(defect_dict)
        assert set(defect_thermo.defect_entries.keys()) == set(defect_dict.keys())
        assert {round(entry.get_ediff(), 3) for entry in defect_thermo.defect_entries.values()} == {
            round(entry.get_ediff(), 3) for entry in defect_dict.values()
        }
        assert {  # check coords are the same by getting their products
            round(np.prod(entry.sc_defect_frac_coords), 3)
            for entry in defect_thermo.defect_entries.values()
        } == {round(np.prod(entry.sc_defect_frac_coords), 3) for entry in defect_dict.values()}

    def _compare_defect_thermos(self, defect_thermo1, defect_thermo2):
        assert len(defect_thermo1.defect_entries) == len(defect_thermo2.defect_entries)
        assert {entry.name for entry in defect_thermo1.defect_entries.values()} == {
            entry.name for entry in defect_thermo2.defect_entries.values()
        }
        assert {round(entry.get_ediff(), 3) for entry in defect_thermo1.defect_entries.values()} == {
            round(entry.get_ediff(), 3) for entry in defect_thermo2.defect_entries.values()
        }
        assert {  # check coords are the same by getting their products
            round(np.prod(entry.sc_defect_frac_coords), 3)
            for entry in defect_thermo1.defect_entries.values()
        } == {
            round(np.prod(entry.sc_defect_frac_coords), 3)
            for entry in defect_thermo2.defect_entries.values()
        }

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
        assert set(defect_thermo.defect_entries.keys()) == {
            defect_entry.name for defect_entry in defect_thermo.defect_entries.values()
        }
        assert defect_thermo.dist_tol == dist_tol
        assert defect_thermo.chempots == chempots
        assert defect_thermo.el_refs == el_refs

        # CdTe, YTOS, Sb2Se3, Sb2Si2Te6, V2O5, MgO, Sb2O5, ZnS values:
        assert any(
            np.isclose(defect_thermo.vbm, i, atol=1e-2)
            for i in [1.65, 3.26, 4.19, 6.60, 0.90, 3.1293, 4.0002, 1.2707]
        )
        assert any(
            np.isclose(defect_thermo.band_gap, i, atol=1e-2)
            for i in [1.5, 0.7, 1.47, 0.44, 2.22, 4.7218, 3.1259, 3.3084]
        )  # note YTOS is GGA calcs so band gap underestimated

        assert defect_thermo.check_compatibility == check_compatibility

        # check methods run ok without errors:
        defect_thermo.to_json("test_thermo.json")
        reloaded_thermo = DefectThermodynamics.from_json("test_thermo.json")
        self._compare_defect_thermos(defect_thermo, reloaded_thermo)
        if_present_rm("test_thermo.json")

        defect_thermo.to_json()  # test default naming
        compositions = ["CdTe", "Y2Ti2S2O5", "Sb2Se3", "SiSbTe3", "V2O5", "MgO", "Sb2O5", "ZnS"]
        assert defect_thermo.bulk_formula in compositions
        assert any(os.path.exists(f"{i}_defect_thermodynamics.json.gz") for i in compositions)
        for i in compositions:
            if_present_rm(f"{i}_defect_thermodynamics.json.gz")

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

        df, output, symm_w = _run_func_and_capture_stdout_warnings(
            defect_thermo.get_symmetries_and_degeneracies
        )
        assert not output
        assert isinstance(df, pd.DataFrame)
        df, conc_output, conc_w = _run_func_and_capture_stdout_warnings(
            defect_thermo.get_equilibrium_concentrations
        )
        assert isinstance(df, pd.DataFrame)
        if chempots is not None:
            assert any(
                "No chemical potential limit specified! Using" in str(warn.message) for warn in conc_w
            )

        df_or_list, form_e_output, form_e_w = _run_func_and_capture_stdout_warnings(
            defect_thermo.get_formation_energies
        )
        assert isinstance(df_or_list, (pd.DataFrame, list))
        for output in [form_e_output, conc_output]:
            assert "Fermi level was not set, so using mid-gap Fermi level" in output
        for w in [conc_w, form_e_w]:
            chempots_warning = any(
                "No chemical potentials supplied, so using 0 for all chemical potentials"
                in str(warn.message)
                for warn in w
            )
            assert chempots_warning == (chempots is None)

        for w in [symm_w, conc_w]:  # the dub
            if defect_thermo.bulk_formula in ["SiSbTe3", "ZnS"]:  # periodicity-breaking -> warning:
                assert any(
                    "The defect supercell has been detected to possibly have" in str(warn.message)
                    for warn in w
                )
            else:
                assert len(w) in {0, 1}

        figure_or_list, output, w = _run_func_and_capture_stdout_warnings(defect_thermo.plot)
        assert isinstance(figure_or_list, (mpl.figure.Figure, list))
        assert not output
        if chempots is None:
            assert any(
                "You have not specified chemical potentials (`chempots`), so chemical potentials are set "
                "to zero for each species." in str(warn.message)
                for warn in w
            )

        with warnings.catch_warnings(record=True) as w:
            tl_df = defect_thermo.get_transition_levels()
            assert set(tl_df.columns) == {"In Band Gap?", "eV from VBM"}
            assert set(tl_df.index.names) == {"Charges", "Defect"}
            all_tl_df = defect_thermo.get_transition_levels(all=True)
            assert set(all_tl_df.columns) == {
                "In Band Gap?",
                "eV from VBM",
                "N(Metastable)",
            }
            assert set(all_tl_df.index.names) == {"Charges", "Defect"}
            defect_thermo.print_transition_levels()
            defect_thermo.print_transition_levels(all=True)
        print([str(warning.message) for warning in w])  # for debugging
        assert not w

        with pytest.raises(ValueError) as exc:
            defect_thermo.get_equilibrium_fermi_level()
        assert "No bulk DOS calculation (`bulk_dos`) provided" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            defect_thermo.get_quenched_fermi_level_and_concentrations()
        assert "No bulk DOS calculation (`bulk_dos`) provided" in str(exc.value)

        if defect_thermo.chempots is None:
            with pytest.raises(ValueError) as exc:
                defect_thermo.get_doping_windows()
            assert "No chemical potentials supplied or present" in str(exc.value)
            assert "so doping windows cannot be calculated." in str(exc.value)

            with pytest.raises(ValueError) as exc:
                defect_thermo.get_dopability_limits()
            assert "No chemical potentials supplied or present" in str(exc.value)
            assert "so dopability limits cannot be calculated." in str(exc.value)

        else:
            with warnings.catch_warnings(record=True) as w:
                assert isinstance(defect_thermo.get_doping_windows(), pd.DataFrame)
                assert isinstance(defect_thermo.get_dopability_limits(), pd.DataFrame)
            print([str(warning.message) for warning in w])  # for debugging
            assert not w
            if "V2O5" in defect_thermo.bulk_formula:
                for df in [defect_thermo.get_doping_windows(), defect_thermo.get_dopability_limits()]:
                    assert set(df.columns).issubset(
                        {
                            "Compensating Defect",
                            "Dopability Limit (eV from VBM/CBM)",
                            "limit",
                            "Doping Window (eV at VBM/CBM)",
                        }
                    )
                    assert set(df.index) == {"n-type", "p-type"}
                    assert df.shape == (2, 3)
                    assert set(df.loc["n-type"]).issubset({"N/A", -np.inf, np.inf})
                    assert set(df.loc["p-type"]).issubset({"N/A", -np.inf, np.inf})

            print(defect_thermo.get_doping_windows())
            print(defect_thermo.get_dopability_limits())

        # test setting dist_tol:
        if (
            defect_thermo.bulk_formula == "CdTe"
            and defect_thermo.dist_tol == 1.5
            and len(defect_thermo.defect_entries) < 20
        ):  # CdTe example defects
            self._check_CdTe_example_dist_tol(defect_thermo, 3)
        self._set_and_check_dist_tol(1.0, defect_thermo, 4)
        self._set_and_check_dist_tol(0.5, defect_thermo, 5)

        # test mismatching chempot warnings:
        print("Checking mismatching chempots")
        mismatch_chempots = {el: -3 for el in Composition(defect_thermo.bulk_formula).as_dict()}
        with warnings.catch_warnings(record=True) as w:
            defect_thermo.chempots = mismatch_chempots
        print([str(warning.message) for warning in w])  # for debugging
        assert any(
            "Note that the raw (DFT) energy of the bulk supercell calculation" in str(warning.message)
            for warning in w
        )

    def _check_CdTe_example_dist_tol(self, defect_thermo, num_grouped_defects):
        print(f"Testing CdTe updated dist_tol: {defect_thermo.dist_tol}")
        tl_df = defect_thermo.get_transition_levels()
        assert tl_df.shape == (num_grouped_defects, 2)
        # number of entries in Figure label (i.e. grouped defects)
        assert len(defect_thermo.plot().get_axes()[0].get_legend().get_texts()) == num_grouped_defects

    def _set_and_check_dist_tol(self, dist_tol, defect_thermo, num_grouped_defects=0):
        defect_thermo.dist_tol = dist_tol
        assert defect_thermo.dist_tol == dist_tol
        if defect_thermo.bulk_formula == "CdTe" and len(defect_thermo.defect_entries) < 20:
            self._check_CdTe_example_dist_tol(defect_thermo, num_grouped_defects)
        defect_thermo.print_transition_levels()
        defect_thermo.print_transition_levels(all=True)

    def _clear_symmetry_degeneracy_info(self, defect_thermo):
        for defect_entry in defect_thermo.defect_entries.values():
            for key in list(defect_entry.calculation_metadata.keys()):
                if "symmetry" in key:
                    del defect_entry.calculation_metadata[key]
            for key in list(defect_entry.degeneracy_factors.keys()):
                del defect_entry.degeneracy_factors[key]

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
            (self.MgO_defect_dict, "MgO_defect_dict"),
        ]:
            print(f"Checking {name}")
            with warnings.catch_warnings(record=True) as w:
                defect_thermo = DefectThermodynamics(list(defect_dict.values()))  # test init with list
            print([str(warning.message) for warning in w])  # for debugging
            assert not w
            self._check_defect_thermo(defect_thermo, defect_dict)  # default values

            defect_thermo = DefectThermodynamics(defect_dict)  # test init with dict
            self._check_defect_thermo(defect_thermo, defect_dict)  # default values

            if "V2O5" in name:
                defect_thermo = DefectThermodynamics(
                    defect_dict,
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

        print("Checking CdTe with mismatching chempots")
        with warnings.catch_warnings(record=True) as w:
            defect_thermo = DefectThermodynamics(self.CdTe_defect_dict, chempots={"Cd": -1.0, "Te": -6})
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1  # only chempot incompatibility warning
        assert str(w[0].message) == self.cdte_chempot_warning_message

        assert defect_thermo.bulk_dos is None
        defect_thermo.bulk_dos = os.path.join(
            self.CdTe_EXAMPLE_DIR, "CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        )
        assert isinstance(defect_thermo.bulk_dos, FermiDos)
        assert np.isclose(defect_thermo.bulk_dos.get_cbm_vbm()[1], 1.65, atol=1e-2)

        defect_thermo.bulk_dos = defect_thermo.bulk_dos  # test setting with FermiDos input
        assert np.isclose(defect_thermo.bulk_dos.get_cbm_vbm()[1], 1.65, atol=1e-2)

        print("Checking CdTe with fermi dos")
        with warnings.catch_warnings(record=True) as w:
            defect_thermo = DefectThermodynamics(
                self.CdTe_defect_dict,
                bulk_dos=os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"),
            )
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 0  # no warning (e.g. VBM mismatch)
        assert isinstance(defect_thermo.bulk_dos, FermiDos)
        assert np.isclose(defect_thermo.bulk_dos.get_cbm_vbm()[1], 1.65, atol=1e-2)

    def test_DefectsParser_thermo_objs(self):
        """
        Test the `DefectThermodynamics` objects created from the
        ``DefectsParser.get_defect_thermodynamics()`` method.
        """
        for defect_thermo, name in [
            (self.CdTe_defect_thermo, "CdTe_defect_thermo"),
            (self.YTOS_defect_thermo, "YTOS_defect_thermo"),
            (self.Sb2Se3_defect_thermo, "Sb2Se3_defect_thermo"),
            (self.Sb2Si2Te6_defect_thermo, "Sb2Si2Te6_defect_thermo"),
            (self.V2O5_defect_thermo, "V2O5_defect_thermo"),
            (self.MgO_defect_thermo, "MgO_defect_thermo"),
            (self.Sb2O5_defect_thermo, "Sb2O5_defect_thermo"),
            (self.ZnS_defect_thermo, "ZnS_defect_thermo"),
        ]:
            print(f"Checking {name}")
            if "V2O5" in name:
                self._check_defect_thermo(
                    defect_thermo,
                    chempots=self.V2O5_chempots,
                    el_refs=self.V2O5_chempots["elemental_refs"],
                )
            elif "MgO" in name:
                self._check_defect_thermo(
                    defect_thermo,
                    chempots=self.MgO_chempots,
                    el_refs=self.MgO_chempots["elemental_refs"],
                )
            elif "Sb2O5" in name:
                self._check_defect_thermo(
                    defect_thermo,
                    chempots=self.Sb2O5_chempots,
                    el_refs=self.Sb2O5_chempots["elemental_refs"],
                )
            elif "ZnS" in name:
                self._check_defect_thermo(defect_thermo, dist_tol=2.5)
            else:
                self._check_defect_thermo(defect_thermo)  # default values

    def test_DefectsParser_thermo_objs_no_metadata(self):
        """
        Test the `DefectThermodynamics` objects created from the
        ``DefectsParser.get_defect_thermodynamics()`` method.
        """
        for defect_thermo, name in [
            (self.CdTe_defect_thermo, "CdTe_defect_thermo"),
            (self.YTOS_defect_thermo, "YTOS_defect_thermo"),
            (self.Sb2Se3_defect_thermo, "Sb2Se3_defect_thermo"),
            (self.Sb2Si2Te6_defect_thermo, "Sb2Si2Te6_defect_thermo"),
            (self.V2O5_defect_thermo, "V2O5_defect_thermo"),
            (self.MgO_defect_thermo, "MgO_defect_thermo"),
            (self.Sb2O5_defect_thermo, "Sb2O5_defect_thermo"),
            (self.ZnS_defect_thermo, "ZnS_defect_thermo"),
        ]:
            print(f"Checking {name}")
            # get set of random 7 entries from defect_entries_wout_metadata (7 because need full CdTe
            # example set for its semi-hard 😏 tests)
            defect_entries_wout_metadata = random.sample(
                list(defect_thermo.defect_entries.values()), min(7, len(defect_thermo.defect_entries))
            )
            for entry in defect_entries_wout_metadata:
                print(f"Setting metadata to empty for {entry.name}")
                entry.calculation_metadata = {}

            with pytest.raises(ValueError) as exc:
                DefectThermodynamics(defect_entries_wout_metadata)
            assert (
                "No VBM eigenvalue was supplied or able to be parsed from the defect entries ("
                "calculation_metadata attributes). Please specify the VBM eigenvalue in the function "
                "input."
            ) in str(exc.value)
            with pytest.raises(ValueError) as exc:
                DefectThermodynamics(defect_entries_wout_metadata, vbm=1.65)
            assert (
                "No band gap value was supplied or able to be parsed from the defect entries ("
                "calculation_metadata attributes). Please specify the band gap value in the function "
                "input."
            ) in str(exc.value)

            thermo_wout_metadata = DefectThermodynamics(
                defect_entries_wout_metadata, vbm=1.65, band_gap=1.499
            )
            self._check_defect_thermo(thermo_wout_metadata)  # default values

            defect_entries_wout_metadata_or_degeneracy = random.sample(
                list(defect_thermo.defect_entries.values()), min(7, len(defect_thermo.defect_entries))
            )  # get set of random 7 entries from defect_entries_wout_metadata (7 needed for CdTe tests)
            for entry in defect_entries_wout_metadata_or_degeneracy:
                print(f"Setting metadata and degeneracy factors to empty for {entry.name}")
                entry.calculation_metadata = {}
                entry.degeneracy_factors = {}

            thermo_wout_metadata_or_degeneracy = DefectThermodynamics(
                defect_entries_wout_metadata_or_degeneracy, vbm=1.65, band_gap=1.499
            )
            self._check_defect_thermo(thermo_wout_metadata_or_degeneracy)  # default values
            symm_df = thermo_wout_metadata_or_degeneracy.get_symmetries_and_degeneracies()
            assert symm_df["g_Spin"].apply(lambda x: isinstance(x, int)).all()

            for defect_entry in defect_entries_wout_metadata_or_degeneracy:
                assert defect_entry.degeneracy_factors["spin degeneracy"] in {1, 2}
                assert isinstance(
                    defect_entry.degeneracy_factors["orientational degeneracy"], (int, float)
                )

                defect_entry.degeneracy_factors = {}
                defect_entry.calculation_metadata = {}

                _conc = defect_entry.equilibrium_concentration()
                assert defect_entry.degeneracy_factors["spin degeneracy"] in {1, 2}
                assert isinstance(
                    defect_entry.degeneracy_factors["orientational degeneracy"], (int, float)
                )

    def test_transition_levels_CdTe(self):
        """
        Test outputs of transition level functions for CdTe.
        """
        assert self.CdTe_defect_thermo.transition_level_map == {
            "v_Cd": {0.47047144459596113: [0, -2]},
            "Te_Cd": {},
            "Int_Te_3": {0.03497090517885537: [2, 1]},
        }

        tl_info = ["Defect: v_Cd", "Defect: Te_Cd"]
        tl_info_not_all = ["Transition level ε(0/-2) at 0.470 eV above the VBM"]
        tl_info_all = [
            "Transition level ε(-1*/-2) at 0.399 eV above the VBM",
            "Transition level ε(0/-1*) at 0.542 eV above the VBM",
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
        assert tl_df.shape == (3, 2)
        assert list(tl_df.iloc[0]) == [0.47, True]
        assert list(tl_df.index.to_numpy()[0]) == ["v_Cd", "ε(0/-2)"]
        assert list(tl_df.iloc[1]) == [np.inf, False]
        assert list(tl_df.index.to_numpy()[1]) == ["Te_Cd", "None"]
        assert list(tl_df.iloc[2]) == [0.035, True]
        assert list(tl_df.index.to_numpy()[2]) == ["Int_Te_3", "ε(+2/+1)"]

        tl_df = self.CdTe_defect_thermo.get_transition_levels(all=True)
        assert tl_df.shape == (5, 3)
        assert list(tl_df.iloc[0]) == [0.542, True, 1]
        assert list(tl_df.index.to_numpy()[0]) == ["v_Cd", "ε(0/-1*)"]
        assert list(tl_df.iloc[1]) == [0.399, True, 1]
        assert list(tl_df.index.to_numpy()[1]) == ["v_Cd", "ε(-1*/-2)"]
        assert list(tl_df.iloc[2]) == [np.inf, False, 0]
        assert list(tl_df.index.to_numpy()[2]) == ["Te_Cd", "None"]
        assert list(tl_df.iloc[3]) == [0.035, True, 0]
        assert list(tl_df.index.to_numpy()[3]) == ["Int_Te_3", "ε(+2/+1)"]
        assert list(tl_df.iloc[4]) == [0.09, True, 1]
        assert list(tl_df.index.to_numpy()[4]) == ["Int_Te_3", "ε(+2/+1*)"]

        self.CdTe_defect_thermo.dist_tol = 1
        tl_df = self.CdTe_defect_thermo.get_transition_levels()
        assert tl_df.shape == (4, 2)
        assert list(tl_df.iloc[2]) == [np.inf, False]
        assert list(tl_df.index.to_numpy()[2]) == ["Int_Te_3_a", "None"]
        assert list(tl_df.iloc[3]) == [0.09, True]
        assert list(tl_df.index.to_numpy()[3]) == ["Int_Te_3_b", "ε(+2/+1)"]

        tl_df = self.CdTe_defect_thermo.get_transition_levels(all=True)
        assert tl_df.shape == (5, 3)
        assert list(tl_df.iloc[3]) == [np.inf, False, 0]
        assert list(tl_df.index.to_numpy()[3]) == ["Int_Te_3_a", "None"]
        assert list(tl_df.iloc[4]) == [0.09, True, 0]
        assert list(tl_df.index.to_numpy()[4]) == ["Int_Te_3_b", "ε(+2/+1)"]

        self.CdTe_defect_thermo.dist_tol = 0.5
        tl_df = self.CdTe_defect_thermo.get_transition_levels()
        assert tl_df.shape == (5, 2)
        assert list(tl_df.iloc[2]) == [np.inf, False]
        assert list(tl_df.index.to_numpy()[2]) == ["Int_Te_3_a", "None"]
        assert list(tl_df.iloc[3]) == [np.inf, False]
        assert list(tl_df.index.to_numpy()[3]) == ["Int_Te_3_b", "None"]
        assert list(tl_df.iloc[4]) == [np.inf, False]
        assert list(tl_df.index.to_numpy()[4]) == ["Int_Te_3_Unperturbed", "None"]

        tl_df = self.CdTe_defect_thermo.get_transition_levels(all=True)
        assert tl_df.shape == (6, 3)
        assert list(tl_df.iloc[3]) == [np.inf, False, 0]
        assert list(tl_df.index.to_numpy()[3]) == ["Int_Te_3_a", "None"]
        assert list(tl_df.iloc[4]) == [np.inf, False, 0]
        assert list(tl_df.index.to_numpy()[4]) == ["Int_Te_3_b", "None"]
        assert list(tl_df.iloc[5]) == [np.inf, False, 0]
        assert list(tl_df.index.to_numpy()[5]) == ["Int_Te_3_Unperturbed", "None"]

    def test_get_symmetries_degeneracies(self):
        """
        Test symmetry and degeneracy functions.

        Also tests that the symmetry and degeneracy functions still work if the
        symmetry and degeneracy metadata wasn't parsed earlier for any reason
        (e.g. transferring from old doped versions, from pymatgen-analysis-
        defects objects etc), with ``reset_thermo=True``.
        """
        for reset_thermo in [False, True]:
            for check_func, thermo in [
                (self._check_CdTe_symmetries_degeneracies, self.CdTe_defect_thermo),
                (
                    self._check_MgO_symmetries_degeneracies,
                    loadfn(f"{module_path}/../examples/MgO/MgO_thermo.json.gz"),
                ),
                (self._check_YTOS_symmetries_degeneracies, self.YTOS_defect_thermo),
                (self._check_Sb2O5_symmetries_degeneracies, self.Sb2O5_defect_thermo),
            ]:
                print(
                    f"Checking symmetries & degeneracies for {thermo.bulk_formula}, reset_thermo"
                    f"={reset_thermo}"
                )
                if reset_thermo:
                    self._clear_symmetry_degeneracy_info(thermo)
                check_func(thermo)

            self.setUp()  # reset thermo objects on 2nd run (with ``reset_thermo=True``)

    def _check_CdTe_symmetries_degeneracies(self, CdTe_defect_thermo: DefectThermodynamics):
        sym_degen_df = CdTe_defect_thermo.get_symmetries_and_degeneracies()
        print(sym_degen_df)
        assert sym_degen_df.shape == (7, 6)
        assert list(sym_degen_df.columns) == [
            "Site_Symm",
            "Defect_Symm",
            "g_Orient",
            "g_Spin",
            "g_Total",
            "Mult",
        ]
        assert list(sym_degen_df.index.names) == ["Defect", "q"]
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
            assert list(sym_degen_df.iloc[i]) == row[2:]
            assert list(sym_degen_df.index.to_numpy()[i]) == row[:2]

        non_formatted_sym_degen_df = self.CdTe_defect_thermo.get_symmetries_and_degeneracies(
            skip_formatting=True
        )
        print(non_formatted_sym_degen_df)  # for debugging
        for i, row in enumerate(cdte_sym_degen_lists):
            row[1] = int(row[1])  # type: ignore
            assert list(non_formatted_sym_degen_df.iloc[i]) == row[2:]
            assert list(non_formatted_sym_degen_df.index.to_numpy()[i]) == row[:2]

    def _check_MgO_symmetries_degeneracies(self, MgO_defect_thermo: DefectThermodynamics):
        sym_degen_df = MgO_defect_thermo.get_symmetries_and_degeneracies()
        # print(sym_degen_df)
        assert sym_degen_df.shape == (5, 6)
        assert list(sym_degen_df.columns) == [
            "Site_Symm",
            "Defect_Symm",
            "g_Orient",
            "g_Spin",
            "g_Total",
            "Mult",
        ]
        assert list(sym_degen_df.index.names) == ["Defect", "q"]
        # hardcoded tests to ensure ordering is consistent (by defect type according to
        # _sort_defect_entries, then by charge state from left (most positive) to right (most negative),
        # as would appear on a TL diagram)
        mgo_sym_degen_lists = [
            ["Mg_O", "+4", "Oh", "C2v", 12.0, 1, 12.0, 1.0],
            ["Mg_O", "+3", "Oh", "C3v", 8.0, 2, 16.0, 1.0],
            ["Mg_O", "+2", "Oh", "C3v", 8.0, 1, 8.0, 1.0],
            ["Mg_O", "+1", "Oh", "Cs", 24.0, 2, 48.0, 1.0],
            ["Mg_O", "0", "Oh", "Cs", 24.0, 1, 24.0, 1.0],
        ]
        for i, row in enumerate(mgo_sym_degen_lists):
            print(i, row)
            assert list(sym_degen_df.iloc[i]) == row[2:]
            assert list(sym_degen_df.index.to_numpy()[i]) == row[:2]

        non_formatted_sym_degen_df = MgO_defect_thermo.get_symmetries_and_degeneracies(
            skip_formatting=True
        )
        print(non_formatted_sym_degen_df)  # for debugging
        for i, row in enumerate(mgo_sym_degen_lists):
            row[1] = int(row[1])  # type: ignore
            assert list(non_formatted_sym_degen_df.iloc[i]) == row[2:]
            assert list(non_formatted_sym_degen_df.index.to_numpy()[i]) == row[:2]

    def _check_YTOS_symmetries_degeneracies(self, YTOS_defect_thermo: DefectThermodynamics):
        sym_degen_df = YTOS_defect_thermo.get_symmetries_and_degeneracies()
        # hardcoded tests to ensure symmetry determination working as expected:
        assert any(
            list(sym_degen_df.iloc[i]) == ["Cs", "C4v", 0.25, 1, 0.25, 4.0]
            for i in range(sym_degen_df.shape[0])
        )
        assert any(
            list(sym_degen_df.index.to_numpy()[i]) == ["Int_F", "-1"] for i in range(sym_degen_df.shape[0])
        )

        sym_degen_df = YTOS_defect_thermo.get_symmetries_and_degeneracies(symprec=0.1)
        assert any(
            list(sym_degen_df.iloc[i]) == ["Cs", "C4v", 0.25, 1, 0.25, 4.0]
            for i in range(sym_degen_df.shape[0])
        )

        sym_degen_df = YTOS_defect_thermo.get_symmetries_and_degeneracies(symprec=0.01)
        assert any(
            list(sym_degen_df.iloc[i]) == ["Cs", "Cs", 1.0, 1, 1.0, 4.0]
            for i in range(sym_degen_df.shape[0])
        )

    def _check_Sb2O5_symmetries_degeneracies(self, Sb2O5_defect_thermo: DefectThermodynamics):
        sym_degen_df = Sb2O5_defect_thermo.get_symmetries_and_degeneracies()
        # hardcoded tests to ensure symmetry determination working as expected:
        assert any(
            list(sym_degen_df.iloc[i]) == ["C1", "C1", 1.0, 2, 2.0, 4.0]
            for i in range(sym_degen_df.shape[0])
        )
        assert any(
            list(sym_degen_df.index.to_numpy()[i]) == ["inter_2_Sb", "0"]
            for i in range(sym_degen_df.shape[0])
        )

        sym_degen_df = Sb2O5_defect_thermo.get_symmetries_and_degeneracies(symprec=0.1)
        assert any(
            list(sym_degen_df.iloc[i]) == ["C1", "C1", 1.0, 2, 2.0, 4.0]
            for i in range(sym_degen_df.shape[0])
        )

        sym_degen_df = Sb2O5_defect_thermo.get_symmetries_and_degeneracies(symprec=0.2)
        assert any(
            list(sym_degen_df.iloc[i]) == ["C1", "Ci", 0.5, 2, 1.0, 4.0]
            for i in range(sym_degen_df.shape[0])
        )

    def _check_form_en_df(self, form_en_df, fermi_level=None, defect_thermo=None):
        """
        Check the sum of formation energy terms equals the total formation
        energy in the `get_formation_energies` DataFrame.
        """
        if defect_thermo is not None:  # check defect sorting
            sorted_defect_entries = _sort_defect_entries(
                {defect_entry.name: defect_entry for defect_entry in defect_thermo.defect_entries.values()}
            )
            assert list(form_en_df.index.get_level_values("Defect").values) == [
                defect_entry.name.rsplit("_", 1)[0] for defect_entry in sorted_defect_entries.values()
            ]

        columns_to_sum = form_en_df.iloc[:, :-3]  # last 3 cols are form E, path, corr error, so omit
        # ignore strings if chempots is "N/A":
        numeric_columns = columns_to_sum.apply(pd.to_numeric, errors="coerce")
        print(f"Numeric columns: {numeric_columns}")  # for debugging

        # assert the sum of formation energy terms equals the total formation energy:
        for i, _row in enumerate(form_en_df.iterrows()):
            assert np.isclose(numeric_columns.iloc[i].sum(), form_en_df.iloc[i]["Eᶠᵒʳᵐ"], atol=1e-3)

        if fermi_level is not None:  # check q*E_F term
            print(
                form_en_df.index.get_level_values("q").to_numpy().astype(int) * fermi_level,
                form_en_df.iloc[:, 2],
            )
            assert np.allclose(
                form_en_df.index.get_level_values("q").to_numpy().astype(int) * fermi_level,
                form_en_df.iloc[:, 2],
                atol=2e-3,
            )

    def test_formation_energies_CdTe(self):
        """
        Here we test the ``DefectThermodynamics.get_formation_energies``,
        ``DefectThermodynamics.get_formation_energy``, and
        ``DefectEntry.formation_energy`` methods.
        """
        form_en_df_cols = [
            "ΔEʳᵃʷ",
            "qE_VBM",
            "qE_F",
            "Σμ_ref",
            "Σμ_formal",
            "E_corr",
            "Eᶠᵒʳᵐ",
            "Path",
            "Δ[E_corr]",
        ]
        form_en_df_indices = ["Defect", "q"]

        form_en_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies
        )

        self._check_chempot_w_and_fermi_message(w, output)
        assert form_en_df.shape == (7, 9)
        assert list(form_en_df.columns) == form_en_df_cols
        assert list(form_en_df.index.names) == form_en_df_indices
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
                0.0,
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
                0.008,
            ],
            [
                "v_Cd",
                "-2",
                7.661,
                -3.293,
                -1.499,
                "N/A",
                0,
                0.739,
                3.608,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl",
                0.011,
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
                0.002,
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
                0.012,
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
                0.003,
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
                0.005,
            ],
        ]
        # test sum of formation energy terms equals total and other formation energy df properties:
        self._check_form_en_df(form_en_df, fermi_level=0.749, defect_thermo=self.CdTe_defect_thermo)

        def _check_formation_energy_methods(form_en_df_row, thermo_obj, fermi_level):
            defect_name_w_charge_state = f"{form_en_df_row[0]}_{int(form_en_df_row[1])}"
            print(defect_name_w_charge_state)  # for debugging
            defect_entry = next(
                entry
                for entry in thermo_obj.defect_entries.values()
                if entry.name.rsplit("_", 1)[0] == defect_name_w_charge_state.rsplit("_", 1)[0]
                and entry.charge_state == int(defect_name_w_charge_state.rsplit("_", 1)[1])
            )

            for name, entry in [
                ("string", defect_name_w_charge_state),
                ("DefectEntry", defect_entry),
            ]:
                print(f"Testing formation energy methods with {name} input")  # for debugging
                if thermo_obj.chempots is None:
                    limits = [
                        None,
                    ]
                    with warnings.catch_warnings(record=True) as w:
                        _form_en = thermo_obj.get_formation_energy(entry, limit="test pop b...")
                    print([str(warn.message) for warn in w])  # for debugging
                    assert len(w) == 2
                    assert "No chemical potentials supplied" in str(w[0].message)
                    assert (
                        "You have specified a chemical potential limit but no chemical potentials "
                        "(`chempots`) were supplied, so `limit` will be ignored." in str(w[-1].message)
                    )

                elif len(thermo_obj.chempots["limits_wrt_el_refs"]) == 2:  # CdTe full chempots
                    with warnings.catch_warnings(record=True) as w:
                        assert (
                            not np.isclose(  # if chempots present, uses the first limit which is Cd-rich
                                thermo_obj.get_formation_energy(entry),
                                form_en_df_row[8],
                                atol=1e-3,
                            )
                        )
                    assert len(w) == 1
                    assert (
                        "No chemical potential limit specified! Using Cd-CdTe for computing "
                        "the formation energy" in str(w[0].message)
                    )
                    limits = ["CdTe-Te", "Te-rich"]
                else:
                    limits = list(thermo_obj.chempots["limits_wrt_el_refs"].keys())  # user supplied

                for limit in limits:
                    if np.isclose(fermi_level, 0.75):  # default CdTe:
                        assert np.isclose(  # test get_formation_energy method
                            thermo_obj.get_formation_energy(entry, limit=limit),
                            form_en_df_row[8],
                            atol=1e-3,
                        )
                    assert np.isclose(  # test get_formation_energy method
                        thermo_obj.get_formation_energy(entry, limit=limit, fermi_level=fermi_level),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    # Test usage of ``DefectThermodynamics.get_formation_energy()`` where charge state
                    # isn't specified:
                    lowest_e_form = thermo_obj.get_formation_energy(
                        form_en_df_row[0], limit=limit, fermi_level=fermi_level
                    )
                    assert np.isclose(
                        lowest_e_form,
                        min(
                            thermo_obj.get_formation_energy(entry, limit=limit, fermi_level=fermi_level)
                            for entry in thermo_obj.defect_entries.values()
                            if form_en_df_row[0] in entry.name
                        ),
                    )

                    assert np.isclose(  # test get_formation_energy() method
                        thermo_obj.get_formation_energy(
                            entry, fermi_level=fermi_level, limit=limit, chempots=thermo_obj.chempots
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    lowest_e_form = thermo_obj.get_formation_energy(
                        form_en_df_row[0],
                        limit=limit,
                        fermi_level=fermi_level,
                        chempots=thermo_obj.chempots,
                    )
                    assert np.isclose(  # test get_formation_energy method
                        lowest_e_form,
                        min(
                            [
                                thermo_obj.get_formation_energy(
                                    entry,
                                    limit=limit,
                                    fermi_level=fermi_level,
                                    chempots=thermo_obj.chempots,
                                )
                                for entry in thermo_obj.defect_entries.values()
                                if form_en_df_row[0] in entry.name
                            ]
                        ),
                    )

                    # test DefectEntry.formation_energy() method:
                    assert np.isclose(
                        defect_entry.formation_energy(
                            fermi_level=fermi_level, limit=limit, chempots=thermo_obj.chempots
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    assert np.isclose(  # test DefectEntry.formation_energy() method
                        defect_entry.formation_energy(
                            fermi_level=fermi_level,
                            vbm=thermo_obj.vbm,
                            limit=limit,
                            chempots=thermo_obj.chempots,
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    assert np.isclose(  # test DefectEntry.formation_energy() method
                        defect_entry.formation_energy(
                            fermi_level=fermi_level + 0.1,
                            vbm=thermo_obj.vbm - 0.1,
                            limit=limit,
                            chempots=thermo_obj.chempots,
                        ),
                        form_en_df_row[8],
                        atol=1e-3,
                    )
                    if thermo_obj.chempots and "rich" not in limit:  # needs to be 'CdTe-Te' etc for
                        # sub-selecting like this
                        assert np.isclose(  # test DefectEntry.formation_energy() method
                            defect_entry.formation_energy(
                                fermi_level=fermi_level,
                                chempots=thermo_obj.chempots["limits_wrt_el_refs"][limit],
                                el_refs=thermo_obj.chempots["elemental_refs"],
                            ),
                            form_en_df_row[8],
                            atol=1e-3,
                        )
                        assert np.isclose(  # test DefectThermodynamics.get_formation_energy() method
                            thermo_obj.get_formation_energy(
                                entry,
                                fermi_level=fermi_level,
                                chempots=thermo_obj.chempots["limits_wrt_el_refs"][limit],
                                el_refs=thermo_obj.chempots["elemental_refs"],
                            ),
                            form_en_df_row[8],
                            atol=1e-3,
                        )

        for i, row in enumerate(cdte_form_en_lists):
            assert list(form_en_df.iloc[i])[:-2] == row[2:-2]  # compare everything except path, ΔE_corr
            assert list(form_en_df.iloc[i])[-1] == row[-1]  # compare ΔE_corr
            assert list(form_en_df.index.to_numpy()[i]) == row[:2]
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)  # default mid-gap value

        with pytest.raises(ValueError) as exc:
            self.CdTe_defect_thermo.get_formation_energy("v_Cd_3")
        assert "No matching DefectEntry with v_Cd_3 in name found in " in str(exc.value)
        assert "DefectThermodynamics.defect_entries, which have names:" in str(exc.value)
        assert (
            "['v_Cd_0', 'v_Cd_-1', 'v_Cd_-2', 'Te_Cd_+1', 'Int_Te_3_2', 'Int_Te_3_1', "
            "'Int_Te_3_Unperturbed_1']" in str(exc.value)
        )

        for i, defect_entry in enumerate(self.CdTe_defect_thermo.defect_entries.values()):
            new_entry = deepcopy(defect_entry)
            _vbm = new_entry.calculation_metadata.pop("vbm")
            assert np.isclose(
                new_entry.formation_energy(fermi_level=0.7493, vbm=self.CdTe_defect_thermo.vbm),
                form_en_df.iloc[i][6],
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
                assert all("VBM eigenvalue was not set" not in str(i.message) for i in w)

        non_formatted_form_en_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies, skip_formatting=True
        )

        self._check_chempot_w_and_fermi_message(w, output)
        for i, row in enumerate(cdte_form_en_lists):
            row[1] = int(row[1])
            assert list(non_formatted_form_en_df.iloc[i])[:-2] == row[2:-2]  # and all other terms the same
            assert list(non_formatted_form_en_df.iloc[i])[-1] == row[-1]  # check ΔE_corr
            assert list(non_formatted_form_en_df.index.to_numpy()[i]) == row[:2]
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
        for limit in ["Te-rich", "CdTe-Te"]:
            df, output, w = _run_func_and_capture_stdout_warnings(
                self.CdTe_defect_thermo.get_formation_energies, self.CdTe_chempots, limit=limit
            )
            self._check_no_w_fermi_message_and_new_matches_ref_df(output, w, df, te_rich_df)
            assert df.shape == (7, 9)
            assert list(df.columns) == form_en_df_cols
            assert list(df.index.names) == form_en_df_indices

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
            chempots=self.CdTe_chempots["limits_wrt_el_refs"]["CdTe-Te"],
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        self._check_no_w_fermi_message_and_new_matches_ref_df(output, w, manual_te_rich_df, te_rich_df)
        manual_te_rich_df_w_fermi, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies,
            chempots=self.CdTe_chempots["limits_wrt_el_refs"]["CdTe-Te"],
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
                0.0,
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
                0.008,
            ],
            [
                "v_Cd",
                "-2",
                7.661,
                -3.293,
                -1.499,
                -1.016,
                -1.251,
                0.739,
                1.341,
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl",
                0.011,
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
                0.002,
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
                0.012,
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
                0.003,
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
                0.005,
            ],
        ]

        # test sum of formation energy terms equals total and other formation energy df properties:
        self._check_form_en_df(te_rich_df, fermi_level=0.7493, defect_thermo=self.CdTe_defect_thermo)

        for i, row in enumerate(cdte_te_rich_form_en_lists):
            assert list(te_rich_df.iloc[i])[:-2] == row[2:-2]  # compare everything except path, ΔE_corr
            assert list(te_rich_df.iloc[i])[-1] == row[-1]  # compare ΔE_corr
            assert list(te_rich_df.index.to_numpy()[i]) == row[:2]
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)  # default mid-gap
            # value, chempots also now attached

        # test same non-formatted output with chempots:
        list_of_dfs, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies, self.CdTe_chempots, skip_formatting=True
        )
        assert not w
        assert "Fermi level was not set" in output
        non_formatted_te_rich_df = list_of_dfs[1]  # Te-rich
        for limit in ["Te-rich", "CdTe-Te"]:
            df, output, w = _run_func_and_capture_stdout_warnings(
                self.CdTe_defect_thermo.get_formation_energies,
                self.CdTe_chempots,
                limit=limit,
                skip_formatting=True,
            )
            self._check_no_w_fermi_message_and_new_matches_ref_df(output, w, df, non_formatted_te_rich_df)
        for i, row in enumerate(cdte_te_rich_form_en_lists):
            row[1] = int(row[1])
            assert list(non_formatted_te_rich_df.iloc[i])[:-2] == row[2:-2]
            assert list(non_formatted_te_rich_df.iloc[i])[-1] == row[-1]  # check ΔE_corr
            assert list(non_formatted_te_rich_df.index.to_numpy()[i]) == row[:2]
            _check_formation_energy_methods(row, self.CdTe_defect_thermo, 0.7493)

        # hard test random case with random chempots, el_refs and fermi level
        manual_form_en_df, output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.get_formation_energies,
            chempots={"Te": 3, "Cd": -1},
            fermi_level=3,
            el_refs={"Cd": -12, "Te": -1},
        )
        assert "Fermi level was not set" not in output
        assert len(w) == 1  # only mis-matching chempot warning
        assert manual_form_en_df.shape == (7, 9)
        # test sum of formation energy terms equals total and other formation energy df properties:
        self._check_form_en_df(manual_form_en_df, fermi_level=3, defect_thermo=self.CdTe_defect_thermo)

        # first 4 columns the same, then 3 diff, one the same (E_corr), E_form diff, path the same:
        manual_thermo = deepcopy(self.CdTe_defect_thermo)
        manual_thermo.chempots = {"Te": 3, "Cd": -1}
        manual_thermo.el_refs = {"Cd": -12, "Te": -1}
        for i, row in manual_form_en_df.iterrows():
            assert list(row)[:2] == list(te_rich_df.loc[i])[:2]
            assert list(row)[2:5] != list(te_rich_df.loc[i])[2:5]
            assert list(row)[5] == list(te_rich_df.loc[i])[5]
            assert list(row)[6] != list(te_rich_df.loc[i])[6]
            assert list(row)[7] == list(te_rich_df.loc[i])[7]
            assert list(row)[8] == list(te_rich_df.loc[i])[8]
            _check_formation_energy_methods(list(i) + list(row), manual_thermo, 3)

        assert list(manual_form_en_df.iloc[2])[:-2] == [  # "v_Cd", "-2",
            7.661,
            -3.293,
            -6,
            -12,
            -1,
            0.739,
            -13.893,
        ]
        assert list(manual_form_en_df.index.to_numpy()[2]) == ["v_Cd", "-2"]
        assert list(manual_form_en_df.iloc[4])[:-2] == [  # "Int_Te_3", "+2",
            -7.105,
            3.293,
            6,
            1,
            -3,
            0.904,
            1.092,
        ]
        assert list(manual_form_en_df.index.to_numpy()[4]) == ["Int_Te_3", "+2"]

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
        # in the chempots dict, for each subdict in chempots["limits"], it should match the sum of
        # the chempots["limits_wrt_el_refs"] and chempots["elemental_refs"]:
        for limit, subdict in chempots_dict["limits"].items():
            for el, mu in subdict.items():
                assert np.isclose(
                    mu,
                    chempots_dict["limits_wrt_el_refs"][limit][el] + chempots_dict["elemental_refs"][el],
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
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.chempots == self.CdTe_chempots  # the same
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # the same

        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}  # Te-rich
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        semi_manual_chempots_dict = {
            "limits_wrt_el_refs": {"User Chemical Potentials": {"Cd": -1.25, "Te": 0}},
            "elemental_refs": {"Te": -4.47069234, "Cd": -1.01586484},
            "limits": {"User Chemical Potentials": {"Cd": -2.26586484, "Te": -4.47069234}},
        }
        assert self.CdTe_defect_thermo.chempots == semi_manual_chempots_dict
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]

        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        assert self.CdTe_defect_thermo.chempots == semi_manual_chempots_dict  # the same
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # the same
        self.CdTe_defect_thermo.chempots = None
        self._check_chempots_dict(self.CdTe_defect_thermo.chempots)
        manual_zeroed_rel_chempots_dict = deepcopy(semi_manual_chempots_dict)
        manual_zeroed_rel_chempots_dict["limits_wrt_el_refs"]["User Chemical Potentials"] = {
            "Cd": 0,
            "Te": 0,
        }
        manual_zeroed_rel_chempots_dict["limits"]["User Chemical Potentials"] = self.CdTe_chempots[
            "elemental_refs"
        ]
        assert self.CdTe_defect_thermo.chempots == manual_zeroed_rel_chempots_dict
        assert self.CdTe_defect_thermo.el_refs == self.CdTe_chempots["elemental_refs"]  # unchanged
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
            "limits_wrt_el_refs": {"User Chemical Potentials": {"Cd": -1.25, "Te": 0}},
            "elemental_refs": {"Te": 0, "Cd": 0},
            "limits": {"User Chemical Potentials": {"Cd": -1.25, "Te": 0}},
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

        # test updating chempots with a set of elements greater than original:
        manual_chempots_dict = deepcopy(manual_zeroed_rel_chempots_dict)
        manual_chempots_dict["limits_wrt_el_refs"]["User Chemical Potentials"]["H"] = -2
        manual_chempots_dict["elemental_refs"]["H"] = -5.4
        defect_thermo.chempots = manual_chempots_dict
        self._check_chempots_dict(defect_thermo.chempots)
        assert defect_thermo.chempots["elemental_refs"].get("H") == -5.4
        assert defect_thermo.chempots["limits"]["User Chemical Potentials"].get("H") == -7.4

        # test error catch when someone tries to input a DataFrame for the chempots...
        import pandas as pd

        chempots_df = pd.DataFrame(self.CdTe_chempots["limits"]).T  # swap the columns and rows
        with pytest.raises(ValueError) as exc:
            self.CdTe_defect_thermo.chempots = chempots_df
        for i in [
            "Invalid chempots/el_refs format:",
            "chempots: <class 'pandas.core.frame.DataFrame'>",
            "Must be a dict (e.g. from `CompetingPhasesAnalyzer.chempots`) or `None`!",
        ]:
            print(i)
            assert i in str(exc.value)

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

    def test_CdTe_all_intrinsic_defects(self):
        for i in [
            "CdTe_defect_dict_v2.3",
            "CdTe_LZ_defect_dict_v2.3_wout_meta",
            "CdTe_defect_dict_old_names",
        ]:
            cdte_defect_dict = loadfn(os.path.join(self.module_path, f"data/{i}.json.gz"))
            cdte_defect_thermo = DefectThermodynamics(cdte_defect_dict)
            self._check_defect_thermo(cdte_defect_thermo, cdte_defect_dict)

        sym_degen_df = cdte_defect_thermo.get_symmetries_and_degeneracies()
        print(sym_degen_df)
        assert sym_degen_df.shape == (50, 6)
        assert list(sym_degen_df.columns) == [
            "Site_Symm",
            "Defect_Symm",
            "g_Orient",
            "g_Spin",
            "g_Total",
            "Mult",
        ]
        assert list(sym_degen_df.index.names) == ["Defect", "q"]

        # hardcoded tests to ensure ordering is consistent (by defect type according to
        # _sort_defect_entries, then by charge state from left (most positive) to right (most negative),
        cdte_sym_degen_lists = [  # manually checked by SK (see thesis pg. 146/7)
            ["vac_1_Cd", "0", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["vac_1_Cd", "-1", "Td", "C3v", 4.0, 2, 8.0, 1.0],
            ["vac_1_Cd", "-2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["vac_1_Cd_1_not_in_gap", "+1", "Td", "Td", 1.0, 2, 2.0, 1.0],
            ["vac_1_Cd_C2v_Bipolaron_S0", "0", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["vac_1_Cd_C2v_Bipolaron_S1", "0", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["vac_1_Cd_Td", "0", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["vac_2_Te", "+2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["vac_2_Te", "+1", "Td", "Td", 1.0, 2, 2.0, 1.0],
            ["vac_2_Te", "0", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["vac_2_Te", "-1", "Td", "Cs", 12.0, 2, 24.0, 1.0],
            ["vac_2_Te", "-2", "Td", "Cs", 12.0, 1, 12.0, 1.0],
            ["vac_2_Te_C3v_low_energy_metastable", "0", "Td", "C3v", 4.0, 1, 4.0, 1.0],
            ["vac_2_Te_orig_metastable", "-1", "Td", "Cs", 12.0, 2, 24.0, 1.0],
            ["vac_2_Te_orig_non_JT_distorted", "0", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["vac_2_Te_shaken", "-2", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["vac_2_Te_unperturbed", "-2", "Td", "C2", 12.0, 1, 12.0, 1.0],
            ["as_1_Cd_on_Te", "+2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["as_1_Cd_on_Te", "+1", "Td", "C2v", 6.0, 2, 12.0, 1.0],
            ["as_1_Cd_on_Te", "0", "Td", "Cs", 12.0, 1, 12.0, 1.0],
            ["as_1_Cd_on_Te", "-1", "Td", "Td", 1.0, 2, 2.0, 1.0],
            ["as_1_Cd_on_Te", "-2", "Td", "C1", 24.0, 1, 24.0, 1.0],
            ["as_2_Cd_on_Te_metastable", "0", "Td", "C2v", 6.0, 1, 6.0, 1.0],
            ["as_2_Cd_on_Te_orig_C2v", "0", "Td", "D2d", 3.0, 1, 3.0, 1.0],
            ["as_1_Te_on_Cd", "+2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["as_1_Te_on_Cd", "+1", "Td", "C3v", 4.0, 2, 8.0, 1.0],
            ["as_1_Te_on_Cd", "0", "Td", "C3v", 4.0, 1, 4.0, 1.0],
            ["as_1_Te_on_Cd", "-1", "Td", "Cs", 12.0, 2, 24.0, 1.0],
            ["as_1_Te_on_Cd", "-2", "Td", "Cs", 12.0, 1, 12.0, 1.0],
            ["as_2_Te_on_Cd_C1_meta", "-2", "Td", "C1", 24.0, 1, 24.0, 1.0],
            ["as_2_Te_on_Cd_C2v_meta", "+1", "Td", "C2v", 6.0, 2, 12.0, 1.0],
            ["as_2_Te_on_Cd_C3v_metastable", "+1", "Td", "C3v", 4.0, 2, 8.0, 1.0],
            ["as_2_Te_on_Cd_Cs_meta", "-2", "Td", "Cs", 12.0, 1, 12.0, 1.0],
            ["as_2_Te_on_Cd_metastable1", "-1", "Td", "C1", 24.0, 2, 48.0, 1.0],
            ["as_2_Te_on_Cd_metastable2", "-1", "Td", "C1", 24.0, 2, 48.0, 1.0],
            ["Int_Cd_1", "+2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["Int_Cd_1", "+1", "Td", "Td", 1.0, 2, 2.0, 1.0],
            ["Int_Cd_1", "0", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["Int_Cd_3", "+2", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["Int_Cd_3", "+1", "Td", "Td", 1.0, 2, 2.0, 1.0],
            ["Int_Cd_3", "0", "Td", "Td", 1.0, 1, 1.0, 1.0],
            ["Int_Te_3", "+2", "C1", "Cs", 0.5, 1, 0.5, 24.0],
            ["Int_Te_3", "+1", "Cs", "C2v", 0.5, 2, 1.0, 12.0],
            ["Int_Te_3", "0", "C1", "C2", 0.5, 1, 0.5, 24.0],
            ["Int_Te_3", "-1", "C1", "C2", 0.5, 2, 1.0, 24.0],
            ["Int_Te_3", "-2", "C1", "C2", 0.5, 1, 0.5, 24.0],
            # with previous parsing (-> symprec = 0.2); gives C3v (also with symprec=0.15), but C3 with
            # symprec=0.1 (new default):
            ["Int_Te_3_C3v_meta", "+2", "C1", "C3v", 0.16666666666666666, 1, 0.16666666666666666, 24.0],
            ["Int_Te_3_orig_dimer_meta", "+2", "C1", "C2", 0.5, 1, 0.5, 24.0],
            ["Int_Te_3_unperturbed", "+1", "C1", "Cs", 0.5, 2, 1.0, 24.0],
            ["Int_Te_3_unperturbed", "0", "Cs", "C2v", 0.5, 1, 0.5, 12.0],
        ]
        for i, row in enumerate(cdte_sym_degen_lists):
            print(i, row)
            assert list(sym_degen_df.iloc[i]) == row[2:]
            assert list(sym_degen_df.index.to_numpy()[i]) == row[:2]

        # test with direct use of point_symmetry_from_structure function:
        sym_degen_dict = {
            f"{row[0]}_{int(row[1])}": {"bulk site symmetry": row[2], "relaxed point symmetry": row[3]}
            for row in cdte_sym_degen_lists
        }
        bulk_symm_ops = get_sga(
            next(iter(cdte_defect_thermo.defect_entries.values())).bulk_supercell
        ).get_symmetry_operations()
        skipped = 0
        for name, defect_entry in cdte_defect_thermo.defect_entries.items():
            print(f"Testing point_symmetry for {defect_entry.name}")
            if name == "Int_Te_3_C3v_meta_2":  # C3v with symprec = 0.2, 0.15, C3 with 0.1
                assert point_symmetry_from_structure(defect_entry.defect_supercell) == "C3"
                assert point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.15) == "C3v"
                assert (
                    point_symmetry_from_site(
                        defect_entry.sc_defect_frac_coords, defect_entry.defect_supercell, symprec=0.1
                    )
                    == "C3"
                )
                continue

            if name == "Int_Te_3_unperturbed_0":  # C2v with symprec = 0.2, Cs with 0.1
                assert point_symmetry_from_structure(defect_entry.defect_supercell) == "Cs"
                assert point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.2) == "C2v"
                assert (
                    point_symmetry_from_site(
                        defect_entry.sc_defect_frac_coords, defect_entry.defect_supercell, symprec=0.2
                    )
                    == "C2v"
                )
                continue

            if name not in sym_degen_dict:  # only v_Cd_1_not_in_gap_+1
                skipped += 1
                continue

            assert (
                point_symmetry_from_structure(defect_entry.defect_supercell)
                == sym_degen_dict[defect_entry.name]["relaxed point symmetry"]
            )
            assert (
                point_symmetry_from_structure(
                    defect_entry.defect_supercell, bulk_structure=defect_entry.bulk_supercell
                )
                == sym_degen_dict[defect_entry.name]["relaxed point symmetry"]
            )
            symm, matching = point_symmetry_from_structure(
                defect_entry.defect_supercell,
                bulk_structure=defect_entry.bulk_supercell,
                symm_ops=bulk_symm_ops,
                return_periodicity_breaking=True,
            )
            assert symm == sym_degen_dict[defect_entry.name]["relaxed point symmetry"]
            assert not matching

            assert (
                point_symmetry_from_structure(
                    defect_entry.defect_supercell,
                    bulk_structure=defect_entry.bulk_supercell,
                    relaxed=False,
                )
                == sym_degen_dict[defect_entry.name]["bulk site symmetry"]
            )
            assert (
                point_symmetry_from_site(
                    _get_defect_supercell_bulk_site_coords(defect_entry, relaxed=False),
                    defect_entry.bulk_supercell,
                )
                == sym_degen_dict[defect_entry.name]["bulk site symmetry"]
            )

            assert (
                point_symmetry_from_structure(
                    defect_entry.defect_supercell,
                    bulk_structure=defect_entry.bulk_supercell,
                    relaxed=False,
                    symm_ops=bulk_symm_ops,
                )
                == sym_degen_dict[defect_entry.name]["bulk site symmetry"]
            )
            assert (
                point_symmetry_from_site(
                    _get_defect_supercell_bulk_site_coords(defect_entry, relaxed=False),
                    defect_entry.bulk_supercell,
                    symm_ops=bulk_symm_ops,
                )
                == sym_degen_dict[defect_entry.name]["bulk site symmetry"]
            )
            assert (
                point_symmetry_from_site(
                    defect_entry.bulk_supercell.lattice.get_cartesian_coords(
                        _get_defect_supercell_bulk_site_coords(defect_entry, relaxed=False)
                    ),
                    defect_entry.bulk_supercell,
                    coords_are_cartesian=True,
                    symm_ops=bulk_symm_ops,
                )
                == sym_degen_dict[defect_entry.name]["bulk site symmetry"]
            )

            # test adjusting symprec:
            if name in [
                "vac_1_Cd_0",
                "as_1_Te_on_Cd_0",
                "Int_Te_3_unperturbed_0",
                "as_2_Te_on_Cd_C3v_metastable_1",
            ]:
                assert point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.01) == "Cs"
                if "vac" not in name:
                    assert (
                        point_symmetry_from_site(
                            defect_entry.sc_defect_frac_coords,
                            defect_entry.defect_supercell,
                            symprec=0.0085,
                        )  # default symprec for point_symmetry_from_site is 0.01
                        == "Cs"  # "as_1_Te_on_Cd_0" and "as_2_Te_on_Cd_C3v_metastable_1" change
                        # symmetries for symprec values just under 0.01 here
                    )
            elif name in [
                "vac_1_Cd_Td_0",
                "as_1_Cd_on_Te_2",
            ]:
                assert point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.01) == "D2d"
            elif name == "vac_2_Te_orig_non_JT_distorted_0":
                assert point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.01) == "C3v"
            elif name in [
                "as_1_Te_on_Cd_-1",
                "as_1_Cd_on_Te_1",
                "as_2_Cd_on_Te_metastable_0",
                "vac_2_Te_orig_metastable_-1",
                "vac_2_Te_shaken_-2",
                "Int_Te_3_C3v_meta_2",
                "Int_Te_3_unperturbed_1",
            ]:
                assert point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.01) == "C1"
            else:
                assert (
                    point_symmetry_from_structure(defect_entry.defect_supercell, symprec=0.01)
                    == sym_degen_dict[name]["relaxed point symmetry"]
                )
                if not any(i in name for i in ["vac", "Int_Te_3"]):
                    # for vacancies, the centre of mass may be different to the V site, and Int_Te_3 are
                    # split-interstitials which again has CoM away from interstitial site
                    assert (
                        point_symmetry_from_site(
                            defect_entry.sc_defect_frac_coords,
                            defect_entry.defect_supercell,
                            # default symprec for point_symmetry_from_site is 0.01
                        )
                        == sym_degen_dict[name]["relaxed point symmetry"]
                    )

        assert skipped == 1  # only v_Cd_1_not_in_gap_+1, because different format ("_+1" vs "_1")

        cdte_defect_dict = loadfn(
            os.path.join(self.module_path, "data/CdTe_defect_dict_old_names.json.gz")
        )
        cdte_defect_thermo = DefectThermodynamics(cdte_defect_dict)
        cdte_defect_thermo.chempots = self.CdTe_chempots
        self._check_defect_thermo(
            cdte_defect_thermo,
            cdte_defect_dict,
            chempots=self.CdTe_chempots,
            el_refs=self.CdTe_chempots["elemental_refs"],
        )

        sym_degen_df = cdte_defect_thermo.get_symmetries_and_degeneracies()
        for i, row in enumerate(cdte_sym_degen_lists):
            print(i, row)
            assert list(sym_degen_df.iloc[i]) == row[2:]
            assert list(sym_degen_df.index.to_numpy()[i]) == row[:2]

        # delete symmetry info to force re-parsing, to test symmetry/degeneracy functions
        self._clear_symmetry_degeneracy_info(cdte_defect_thermo)
        sym_degen_df = cdte_defect_thermo.get_symmetries_and_degeneracies()
        for i, row in enumerate(cdte_sym_degen_lists):
            print(i, row)
            assert list(sym_degen_df.index.to_numpy()[i]) == row[:2]
            if sym_degen_df.index.to_numpy()[i][0] == "Int_Te_3_C3v_meta":
                assert list(sym_degen_df.iloc[i])[1] == "C3"
            elif sym_degen_df.index.to_numpy()[i][0] == "Int_Te_3_unperturbed" and row[1] == "0":
                assert list(sym_degen_df.iloc[i])[1] == "Cs"
            else:
                assert list(sym_degen_df.iloc[i]) == row[2:]

    def test_formation_energy_mult_degen(self):
        cdte_defect_thermo = DefectThermodynamics.from_json(
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_thermo_wout_meta.json.gz")
        )
        # random defect_entry:
        for _ in range(10):
            random_defect_entry = random.choice(list(cdte_defect_thermo.defect_entries.values()))
            print(f"Randomly testing concentration method for {random_defect_entry.name}")

            for temperature in [300, 1000]:
                for fermi_level in [0.25, 0.9, 3]:
                    orig_conc = random_defect_entry.equilibrium_concentration(
                        chempots=self.CdTe_chempots,
                        limit="Cd-rich",
                        fermi_level=fermi_level,
                        temperature=temperature,
                    )
                    new_entry = deepcopy(random_defect_entry)
                    new_entry.defect.multiplicity *= 2

                    new_conc = new_entry.equilibrium_concentration(
                        chempots=self.CdTe_chempots,
                        limit="Cd-rich",
                        fermi_level=fermi_level,
                        temperature=temperature,
                    )
                    assert np.isclose(new_conc, orig_conc * 2)

                    new_entry.degeneracy_factors["spin degeneracy"] *= 0.5
                    new_conc = new_entry.equilibrium_concentration(
                        chempots=self.CdTe_chempots,
                        limit="Cd-rich",
                        fermi_level=fermi_level,
                        temperature=temperature,
                    )
                    assert np.isclose(new_conc, orig_conc)

                    new_entry.degeneracy_factors["orientational degeneracy"] *= 3
                    new_conc = new_entry.equilibrium_concentration(
                        chempots=self.CdTe_chempots,
                        limit="Cd-rich",
                        fermi_level=fermi_level,
                        temperature=temperature,
                    )
                    assert np.isclose(new_conc, orig_conc * 3)

                    new_entry.degeneracy_factors["fake degeneracy"] = 7
                    new_conc = new_entry.equilibrium_concentration(
                        chempots=self.CdTe_chempots,
                        limit="Cd-rich",
                        fermi_level=fermi_level,
                        temperature=temperature,
                    )
                    assert np.isclose(new_conc, orig_conc * 21)

                    # test per_site and bulk_site_concentration attributes:
                    new_conc = random_defect_entry.equilibrium_concentration(
                        chempots=self.CdTe_chempots,
                        limit="Cd-rich",
                        fermi_level=fermi_level,
                        temperature=temperature,
                        per_site=True,
                    )
                    assert np.isclose(orig_conc, new_conc * random_defect_entry.bulk_site_concentration)

    def test_Sb2O5_formation_energies(self):
        formation_energy_table_df = self.Sb2O5_defect_thermo.get_formation_energies(
            self.Sb2O5_chempots, limit="Sb2O5-SbO2", fermi_level=3
        )
        self._check_form_en_df(
            formation_energy_table_df, fermi_level=3, defect_thermo=self.Sb2O5_defect_thermo
        )

        formation_energy_table_df = self.Sb2O5_defect_thermo.get_formation_energies(
            # test default w/ E_F = mid-gap (1.563 eV for Sb2O5)
            self.Sb2O5_chempots,
            limit="Sb2O5-O2",
        )
        self._check_form_en_df(
            formation_energy_table_df, fermi_level=1.563, defect_thermo=self.Sb2O5_defect_thermo
        )

        formation_energy_table_df_manual_chempots = (
            self.Sb2O5_defect_thermo.get_formation_energies(  # test default with E_F = E_g/2
                chempots=self.Sb2O5_chempots["limits_wrt_el_refs"]["Sb2O5-O2"],
                el_refs=self.Sb2O5_chempots["elemental_refs"],
            )
        )
        self._check_form_en_df(
            formation_energy_table_df_manual_chempots,
            fermi_level=1.563,
            defect_thermo=self.Sb2O5_defect_thermo,
        )

        # check manual and auto chempots the same:
        assert formation_energy_table_df_manual_chempots.equals(formation_energy_table_df)

        # assert runs fine without chempots:
        self.Sb2O5_defect_thermo.chempots = None
        formation_energy_table_df = self.Sb2O5_defect_thermo.get_formation_energies(fermi_level=0)
        self._check_form_en_df(
            formation_energy_table_df, fermi_level=0, defect_thermo=self.Sb2O5_defect_thermo
        )

        # assert runs fine with only raw chempots:
        formation_energy_table_df = self.Sb2O5_defect_thermo.get_formation_energies(
            chempots=self.Sb2O5_chempots["limits_wrt_el_refs"]["Sb2O5-O2"],
        )
        self._check_form_en_df(
            formation_energy_table_df, fermi_level=1.563, defect_thermo=self.Sb2O5_defect_thermo
        )
        # check same formation energies as with manual chempots plus el_refs:
        assert formation_energy_table_df.iloc[:, 8].equals(
            formation_energy_table_df_manual_chempots.iloc[:, 8]
        )

        # check saving to csv and reloading all works fine:
        formation_energy_table_df.to_csv("test.csv", index=False)
        formation_energy_table_df_reloaded = pd.read_csv("test.csv")
        formation_energy_table_df_wout_charge_formatting = formation_energy_table_df_reloaded.copy()
        formation_energy_table_df_wout_charge_formatting.iloc[:, 1] = (
            formation_energy_table_df_wout_charge_formatting.iloc[:, 1].apply(int)
        )
        for i in range(len(formation_energy_table_df)):  # more robust comparison method than df.equals()
            df_row = formation_energy_table_df.iloc[i]
            reloaded_df_row = formation_energy_table_df_wout_charge_formatting.iloc[i]
            for j, val in enumerate(df_row):
                print(val, reloaded_df_row.iloc[j])
            assert val == reloaded_df_row.iloc[j]

        os.remove("test.csv")

    def test_incompatible_chempots_warning(self):
        """
        Test that we get the expected warnings when we provide incompatible
        chemical potentials for our DefectThermodynamics object.
        """
        slightly_off_chempots = {"Cd": -1.0, "Te": -6}
        for func in [
            self.CdTe_defect_thermo.get_equilibrium_concentrations,
            self.CdTe_defect_thermo.get_dopability_limits,
            self.CdTe_defect_thermo.get_doping_windows,
            self.CdTe_defect_thermo.get_formation_energies,
            self.CdTe_defect_thermo.plot,
        ]:
            _result, _tl_output, w = _run_func_and_capture_stdout_warnings(
                func, chempots=slightly_off_chempots
            )
            assert any(str(warning.message) == self.cdte_chempot_warning_message for warning in w)

        with warnings.catch_warnings(record=True) as w:
            self.CdTe_defect_thermo.chempots = slightly_off_chempots
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1  # only chempot incompatibility warning
        assert str(w[0].message) == self.cdte_chempot_warning_message


def belas_linear_fit(T):  #
    """
    Linear fit of CdTe gap dependence with temperature.
    """
    return 1.6395 - 0.000438 * T


class DefectThermodynamicsCdTePlotsTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.CdTe_EXAMPLE_DIR = os.path.join(cls.module_path, "../examples/CdTe")
        cls.CdTe_chempots = loadfn(os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_chempots.json"))
        cls.defect_dict = loadfn(
            os.path.join(cls.module_path, "data/CdTe_LZ_defect_dict_v2.3_wout_meta.json.gz")
        )
        cls.defect_thermo = DefectThermodynamics(cls.defect_dict)
        cls.defect_thermo.chempots = cls.CdTe_chempots

        cls.fermi_dos = get_fermi_dos(
            os.path.join(cls.CdTe_EXAMPLE_DIR, "CdTe_prim_k181818_NKRED_2_vasprun.xml.gz")
        )
        cls.anneal_temperatures = np.arange(200, 1401, 50)
        cls.reduced_anneal_temperatures = np.arange(200, 1401, 100)  # for quicker testing

        cls.annealing_dict = {}

        for anneal_temp in cls.anneal_temperatures:
            gap_shift = belas_linear_fit(anneal_temp) - 1.5
            scissored_dos = scissor_dos(gap_shift, cls.fermi_dos, verbose=True)

            (
                fermi_level,
                e_conc,
                h_conc,
                conc_df,
            ) = cls.defect_thermo.get_quenched_fermi_level_and_concentrations(
                # quenching to 300K (default)
                cls.fermi_dos,
                limit="Te-rich",
                annealing_temperature=anneal_temp,
                delta_gap=gap_shift,
            )
            (
                annealing_fermi_level,
                annealing_e_conc,
                annealing_h_conc,
            ) = cls.defect_thermo.get_equilibrium_fermi_level(
                scissored_dos,
                limit="Te-rich",
                temperature=anneal_temp,
                return_concs=True,
            )
            cls.annealing_dict[anneal_temp] = {
                "annealing_fermi_level": annealing_fermi_level,
                "annealing_e_conc": annealing_e_conc,
                "annealing_h_conc": annealing_h_conc,
                "fermi_level": fermi_level,
                "e_conc": e_conc,
                "h_conc": h_conc,
                "conc_df": conc_df,
            }

    def belas_linear_fit(self, T):  # linear fit of CdTe gap dependence with temperature
        return 1.6395 - 0.000438 * T

    @custom_mpl_image_compare(filename="CdTe_LZ_Te_rich_Fermi_levels.png")
    def test_calculated_fermi_levels(self):
        plt.style.use(STYLE)
        f, ax = plt.subplots()

        anneal_fermi_levels = np.array(
            [v["annealing_fermi_level"] for k, v in self.annealing_dict.items()]
        )
        quenched_fermi_levels = np.array([v["fermi_level"] for k, v in self.annealing_dict.items()])
        assert np.isclose(np.mean(self.anneal_temperatures[12:16]), 875)
        # remember this is LZ thermo, not FNV thermo shown in thermodynamics tutorial
        assert np.isclose(np.mean(quenched_fermi_levels[12:16]), 0.318674, atol=1e-3)

        ax.plot(
            self.anneal_temperatures,
            anneal_fermi_levels,
            marker="o",
            label="$E_F$ during annealing (@ $T_{anneal}$)",
            color="k",
            alpha=0.25,
        )
        ax.plot(
            self.anneal_temperatures,
            quenched_fermi_levels,
            marker="o",
            label="$E_F$ upon cooling (@ $T$ = 300K)",
            color="k",
            alpha=0.9,
        )
        ax.set_xlabel("Anneal Temperature (K)")
        ax.set_ylabel("Fermi Level wrt VBM (eV)")
        ax.set_xlim(300, 1400)
        ax.axvspan(500 + 273.15, 700 + 273.15, alpha=0.2, color="#33A7CC", label="Typical Anneal Range")
        ax.fill_between(
            self.anneal_temperatures,
            (1.5 - belas_linear_fit(self.anneal_temperatures)) / 2,
            0,
            alpha=0.2,
            color="C0",
            label="VBM (T @ $T_{anneal}$)",
            linewidth=0.25,
        )
        ax.fill_between(
            self.anneal_temperatures,
            1.5 - (1.5 - belas_linear_fit(self.anneal_temperatures)) / 2,
            1.5,
            alpha=0.2,
            color="C1",
            label="CBM (T @ $T_{anneal}$)",
            linewidth=0.25,
        )

        ax.legend(fontsize=8)

        ax.imshow(  # show VB in blue from -0.3 to 0 eV:
            [(1, 1), (0, 0)],
            cmap=plt.cm.Blues,
            extent=(ax.get_xlim()[0], ax.get_xlim()[1], -0.3, 0),
            vmin=0,
            vmax=3,
            interpolation="bicubic",
            rasterized=True,
            aspect="auto",
        )

        ax.imshow(
            [
                (
                    0,
                    0,
                ),
                (1, 1),
            ],
            cmap=plt.cm.Oranges,
            extent=(ax.get_xlim()[0], ax.get_xlim()[1], 1.5, 1.8),
            vmin=0,
            vmax=3,
            interpolation="bicubic",
            rasterized=True,
            aspect="auto",
        )
        ax.set_ylim(-0.2, 1.7)

        return f

    def test_calculated_fermi_level_k10(self):
        """
        Test calculating the Fermi level using a 10x10x10 k-point mesh DOS
        calculation for primitive CdTe; specifying just the path to the DOS
        vasprun.xml.
        """
        k10_dos_vr_path = os.path.join(self.module_path, "data/CdTe/CdTe_prim_k101010_dos_vr.xml.gz")

        for i, bulk_dos in enumerate([k10_dos_vr_path, get_vasprun(k10_dos_vr_path, parse_dos=True)]):
            print(f"Testing k10 DOS with thermo for {'str input' if i == 0 else 'DOS object input'}")
            quenched_fermi_levels = []
            for anneal_temp in self.reduced_anneal_temperatures:
                gap_shift = belas_linear_fit(anneal_temp) - 1.5
                (
                    fermi_level,
                    e_conc,
                    h_conc,
                    conc_df,
                ) = self.defect_thermo.get_quenched_fermi_level_and_concentrations(
                    # quenching to 300K (default)
                    bulk_dos=bulk_dos,
                    limit="Te-rich",
                    annealing_temperature=anneal_temp,
                    delta_gap=gap_shift,
                )
                quenched_fermi_levels += [fermi_level]

            # (approx) same result as with k181818 NKRED=2 (0.31825 eV with this DOS)
            # remember this is LZ thermo, not FNV thermo shown in thermodynamics tutorial
            assert np.isclose(np.mean(quenched_fermi_levels[6:8]), 0.31825, atol=1e-3)

    @custom_mpl_image_compare(filename="CdTe_LZ_Te_rich_concentrations.png")
    def test_calculated_concentrations(self):
        annealing_n = np.array(
            [self.annealing_dict[k]["annealing_e_conc"] for k in self.anneal_temperatures]
        )
        annealing_p = np.array(
            [self.annealing_dict[k]["annealing_h_conc"] for k in self.anneal_temperatures]
        )
        quenched_n = np.array([self.annealing_dict[k]["e_conc"] for k in self.anneal_temperatures])
        quenched_p = np.array([self.annealing_dict[k]["h_conc"] for k in self.anneal_temperatures])

        def _array_from_conc_df(name):
            return np.array(
                [
                    self.annealing_dict[temp]["conc_df"][
                        (self.annealing_dict[temp]["conc_df"]["Defect"] == name)
                        & (self.annealing_dict[temp]["conc_df"]["Charge"] == 0)
                    ]["Total Concentration (cm^-3)"].to_numpy()[0]
                    for temp in self.anneal_temperatures
                ]
            )  # TODO: Need to uppdate here and in tutorial??

        plt.style.use(STYLE)
        f, ax = plt.subplots()

        wienecke_data = np.array(
            [
                [675.644735186816, 15.19509584755584],
                [774.64775443452, 15.983458618047331],
                [773.2859479179771, 15.780402388747808],
                [876.594540193735, 16.456749859094277],
                [866.7316643602969, 16.470175483037483],
                [931.3592904767895, 16.68944378653258],
                [972.2040508240029, 16.939464368267398],
                [1043.955214492389, 17.234473455894925],
                [1030.0320795068562, 17.11399747952909],
                [1077.6449867907913, 17.335494943226077],
                [1082.4820732167568, 17.165318826904443],
            ]
        )
        emanuelsson_data = np.array([[750 + 273.15, np.log10(1.2e17)]])
        expt_data = np.append(wienecke_data, emanuelsson_data, axis=0)

        ax.plot(self.anneal_temperatures, quenched_p, label="p", alpha=0.85, linestyle="--")
        ax.plot(self.anneal_temperatures, quenched_n, label="n", alpha=0.85, linestyle="--")

        ax.plot(
            self.anneal_temperatures,
            annealing_p,
            label="p ($T_{anneal}$)",
            alpha=0.5,
            c="C0",
            linestyle="--",
        )
        ax.plot(
            self.anneal_temperatures,
            annealing_n,
            label="n ($T_{anneal}$)",
            alpha=0.5,
            c="C1",
            linestyle="--",
        )

        ax.scatter(expt_data[:, 0], 10 ** (expt_data[:, 1]), marker="x", label="Expt", c="C0", alpha=0.5)

        ax.plot(
            self.anneal_temperatures,
            _array_from_conc_df("v_Cd"),
            marker="o",
            label=r"$V_{Cd}$",
            linestyle="--",
            c="#0D7035",
            alpha=0.7,
        )
        ax.plot(
            self.anneal_temperatures,
            _array_from_conc_df("Te_Cd"),
            marker="o",
            label=r"$Te_{Cd}$",
            c="#F08613",
            alpha=0.7,
        )
        ax.plot(
            self.anneal_temperatures,
            _array_from_conc_df("Te_i_Td_Te2.83"),
            marker="o",
            label="$Te_i$",
            linestyle=":",
            c="#F0B713",
            alpha=0.7,
        )
        ax.plot(
            self.anneal_temperatures,
            _array_from_conc_df("Cd_i_Td_Te2.83"),
            marker="o",
            label="$Cd_i(Te)$",
            linestyle=":",
            c="#35AD88",
            alpha=0.7,
        )
        ax.plot(
            self.anneal_temperatures,
            _array_from_conc_df("v_Te"),
            marker="o",
            label=r"$V_{Te}$",
            linestyle="--",
            c="#D95F02",
            alpha=0.7,
        )
        ax.plot(
            self.anneal_temperatures,
            _array_from_conc_df("Cd_i_Td_Cd2.83"),
            marker="o",
            label="$Cd_i(Cd)$",
            linestyle=":",
            c="#35AD88",
            alpha=0.3,
        )

        ax.axvspan(500 + 273.15, 700 + 273.15, alpha=0.2, color="#33A7CC")

        ax.set_xlabel("Anneal Temperature (K)")
        ax.set_ylabel(r"Concentration (cm$^{-3}$)")
        ax.set_yscale("log")
        ax.set_xlim(300, 1400)
        ax.set_ylim(1e12, 1e18)
        # typical anneal range is 500 - 700, so shade in this region:
        ax.axvspan(500 + 273.15, 700 + 273.15, alpha=0.2, color="#33A7CC")
        ax.legend(fontsize=8)

        return f


# TODO: Test all DefectThermodynamics methods (doping windows/limits, etc)
# TODO: Test check_compatibility
# TODO: Test how attributes change when reloaded from JSON (e.g.
#  entry.corrections_metadata["kumagai_charge_correction"]["pydefect_ExtendedFnvCorrection"] currently
#  changes irreversibly to dict due to being a dataclass)
