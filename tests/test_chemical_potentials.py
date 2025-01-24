"""
Tests for the `doped.chemical_potentials` module.
"""

import os
import sys
import unittest
import warnings
from copy import deepcopy
from functools import wraps
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from test_analysis import if_present_rm
from test_plotting import custom_mpl_image_compare
from test_thermodynamics import _run_func_and_capture_stdout_warnings

from doped import chemical_potentials
from doped.utils.symmetry import get_primitive_structure


def _compare_chempot_dicts(dict1, dict2):
    for key, val in dict1.items():
        if isinstance(val, dict):
            _compare_chempot_dicts(val, dict2[key])
        else:
            assert np.isclose(val, dict2[key], atol=1e-5)


def parameterized_subtest(api_key_dict=None):
    """
    A test decorator to allow running competing phases tests with both the
    legacy and new Materials Project API keys.
    """
    if api_key_dict is None:  # set to SK MP Imperial email (GitHub) A/C keys by default
        api_key_dict = {"legacy": "c2LiJRMiBeaN5iXsH", "new": "UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv"}

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self, *args, **kwargs):
            for name, api_key in api_key_dict.items():
                with self.subTest(api_key=api_key):
                    print(f"Testing with {name} API")
                    test_func(self, api_key, *args, **kwargs)

        return wrapper

    return decorator


cwd = os.path.dirname(os.path.abspath(__file__))


class CompetingPhasesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.legacy_api_key = "c2LiJRMiBeaN5iXsH"  # SK MP Imperial email (GitHub) A/C
        self.api_key = "UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv"  # SK MP Imperial email (GitHub) A/C
        self.EXAMPLE_DIR = os.path.join(cwd, "../examples")
        self.cdte = Structure.from_file(os.path.join(self.EXAMPLE_DIR, "CdTe/relaxed_primitive_POSCAR"))
        self.na2fepo4f = Structure.from_file(os.path.join(cwd, "data/Na2FePO4F_MP_POSCAR"))
        self.cu2sise3 = Structure.from_file(os.path.join(cwd, "data/Cu2SiSe3_MP_POSCAR"))
        self.cu2sise4 = self.cu2sise3.get_primitive_structure().copy()
        self.cu2sise4.append("Se", [0.5, 0.5, 0.5])
        self.cu2sise4.append("Se", [0.5, 0.75, 0.5])

        self.zro2_entry_list = [  # without full_phase_diagram
            "ZrO2",
            "Zr",
            "O2",
            "Zr3O",
            "Zr3O",
            "Zr3O",
            "Zr2O",
            "ZrO2",
            "ZrO2",
            "Zr",
            "ZrO2",
            "ZrO2",
            "ZrO2",
        ]

    def tearDown(self) -> None:
        if_present_rm("CompetingPhases")

    def _check_ZrO2_cp_init(self, cp, num_stable_entries=4):
        for i, entry in enumerate(cp.entries):
            print(entry.name, entry.energy)
            eah = entry.data.get(cp.property_key_dict["energy_above_hull"])
            assert eah == 0 if i < num_stable_entries else eah > 0  # Zr4O is on hull

            mag = entry.data["total_magnetization"]
            is_molecule = entry.data["molecule"]

            assert is_molecule if entry.name == "O2" else not is_molecule
            assert np.isclose(
                mag, 2 if entry.name == "O2" else 0, atol=1e-3
            )  # only O2 is magnetic (triplet) here
            if entry.name == "O2":
                assert np.isclose(entry.data["energy_per_atom"], -4.94795546875)
                assert np.isclose(entry.data["energy"], -4.94795546875 * 2)

    @parameterized_subtest()
    def test_init(self, api_key):
        cp = chemical_potentials.CompetingPhases("ZrO2", e_above_hull=0.03, api_key=api_key)

        assert len(cp.entries) == 13
        assert [entry.name for entry in cp.entries] == self.zro2_entry_list
        self._check_ZrO2_cp_init(cp)
        assert "Zr4O" not in [e.name for e in cp.entries]  # not bordering or potentially with EaH

    @parameterized_subtest()
    def test_init_full_phase_diagram(self, api_key):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", e_above_hull=0.03, api_key=api_key, full_phase_diagram=True
        )

        assert len(cp.entries) == 14  # Zr4O now present
        zro2_full_pd_entry_list = self.zro2_entry_list[:4] + ["Zr4O"] + self.zro2_entry_list[4:]
        assert [entry.name for entry in cp.entries] == zro2_full_pd_entry_list
        self._check_ZrO2_cp_init(cp, num_stable_entries=5)  # Zr4O is on hull

    @parameterized_subtest()
    def test_init_ZnSe(self, api_key):
        """
        As noted by Savya Aggarwal, the legacy MP API code didn't return ZnSe2
        as a competing phase despite being on the hull and bordering ZnSe,
        because the legacy MP API database wrongly had the data['e_above_hull']
        value as 0.147 eV/atom (when it should be 0 eV/atom).

        https://legacy.materialsproject.org/materials/mp-1102515/
        https://next-gen.materialsproject.org/materials/mp-1102515?formula=ZnSe2

        Updated code which re-calculates the energy above hull avoids this issue.
        """
        cp = chemical_potentials.CompetingPhases("ZnSe", api_key=api_key)
        assert any(e.name == "ZnSe2" for e in cp.entries)
        assert len(cp.entries) in {11, 12}  # ZnSe2 present; 2 new Zn entries (mp-264...) with new MP API
        znse2_entry = next(e for e in cp.entries if e.name == "ZnSe2")
        assert znse2_entry.data.get(cp.property_key_dict["energy_above_hull"]) == 0
        assert not znse2_entry.data["molecule"]
        assert np.isclose(znse2_entry.energy_per_atom, -3.394683861)
        assert np.isclose(znse2_entry.energy, -3.394683861 * 12)

    @parameterized_subtest()
    def test_init_YTOS(self, api_key):
        # 144 phases on Y-Ti-O-S MP phase diagram
        cp = chemical_potentials.CompetingPhases("Y2Ti2S2O5", e_above_hull=0.1, api_key=api_key)
        assert len(cp.entries) in {109, 113}  # legacy and new MP APIs
        self.check_O2_entry(cp)

        cp = chemical_potentials.CompetingPhases(
            "Y2Ti2S2O5", e_above_hull=0.1, full_phase_diagram=True, api_key=api_key
        )
        # 144/149 phases on Y-Ti-O-S legacy/new MP phase diagram, 4 extra O2 phases removed
        assert len(cp.entries) in {140, 145}  # legacy and new MP APIs
        self.check_O2_entry(cp)

    def check_O2_entry(self, cp):
        # assert only one O2 phase present (molecular entry):
        result = [e for e in cp.entries if e.name == "O2"]
        assert len(result) == 1
        assert result[0].name == "O2"
        assert result[0].data["total_magnetization"] == 2
        assert result[0].data[cp.property_key_dict["energy_above_hull"]] == 0
        assert result[0].data["molecule"]
        assert np.isclose(result[0].data["energy_per_atom"], -4.94795546875)

    @parameterized_subtest()
    def test_entry_naming(self, api_key):
        """
        Test the naming functions for competing phase entries in ``doped``,
        including rounding to "_0" and increasing the number of digits if
        duplicates are encountered.
        """
        cdte_cp = chemical_potentials.CompetingPhases("CdTe", api_key=api_key)
        if len(api_key) != 32:
            assert [entry.data["doped_name"] for entry in cdte_cp.entries] == [
                "CdTe_F-43m_EaH_0",
                "Cd_P6_3/mmc_EaH_0",
                "Te_P3_121_EaH_0",
                "Te_P3_221_EaH_0",
                "Cd_Fm-3m_EaH_0.001",
                "Cd_R-3m_EaH_0.001",
                "CdTe_P6_3mc_EaH_0.003",
                "CdTe_Cmc2_1_EaH_0.006",
                "Cd_P6_3/mmc_EaH_0.018",
                "Te_C2/m_EaH_0.044",
                "Te_Pm-3m_EaH_0.047",
                "Te_Pmma_EaH_0.047",
                "Te_Pmc2_1_EaH_0.049",
            ]
        else:  # slightly different for new MP API, Te entries the same
            for i in [
                "Te_P3_121_EaH_0",
                "Te_P3_221_EaH_0",
                "Te_C2/m_EaH_0.044",
                "Te_Pm-3m_EaH_0.047",
                "Te_Pmma_EaH_0.047",
                "Te_Pmc2_1_EaH_0.049",
            ]:
                assert i in [entry.data["doped_name"] for entry in cdte_cp.entries]

        # test case when the EaH rounding needs to be dynamically updated:
        # (this will be quite a rare case, as it requires two phases with the same formula, space group
        # and energy above hull to 1 meV/atom
        cds_cp = chemical_potentials.CompetingPhases("CdS", api_key=api_key)
        assert "S_Pnnm_EaH_0.014" in [entry.data["doped_name"] for entry in cds_cp.entries]
        new_entry = deepcopy(
            next(entry for entry in cds_cp.entries if entry.data["doped_name"] == "S_Pnnm_EaH_0.014")
        )  # duplicate entry to force renaming
        if len(api_key) != 32:
            new_entry.data["e_above_hull"] += 2e-4
        else:
            new_entry.data["energy_above_hull"] += 2e-4
        chemical_potentials._name_entries_and_handle_duplicates([*cds_cp.entries, new_entry])
        entry_names = [entry.data["doped_name"] for entry in [*cds_cp.entries, new_entry]]
        assert "S_Pnnm_EaH_0.014" not in entry_names
        assert "S_Pnnm_EaH_0.0141" in entry_names
        assert "S_Pnnm_EaH_0.0143" in entry_names

    @parameterized_subtest()
    def test_unstable_host(self, api_key):
        """
        Test generating CompetingPhases with a composition that's unstable on
        the Materials Project database.
        """
        for cp_settings in [
            {"composition": "Na2FePO4F", "e_above_hull": 0.02, "api_key": api_key},
            {
                "composition": "Na2FePO4F",
                "e_above_hull": 0.02,
                "api_key": api_key,
                "full_phase_diagram": True,
            },
        ]:
            print(f"Testing with settings: {cp_settings}")
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(**cp_settings)
            print([str(warning.message) for warning in w])  # for debugging
            if len(api_key) != 32:  # recalculated energy for Na2FePO4F on new MP API, now on hull
                assert len([warning for warning in w if "You are using" not in str(warning.message)]) == 1
                for sub_message in [
                    "Note that the Materials Project (MP) database entry for Na2FePO4F is not stable with "
                    "respect to competing phases, having an energy above hull of 0.1701 eV/atom.",
                    "Formally, this means that the host material is unstable and so has no chemical "
                    "potential limits; though in reality there may be errors in the MP energies",
                    "Here we downshift the host compound entry to the convex hull energy, and then "
                    "determine the possible competing phases with the same approach as usual",
                ]:
                    print(sub_message)
                    assert any(sub_message in str(warning.message) for warning in w)
            if cp_settings.get("full_phase_diagram"):
                assert len(cp.entries) in {128, 172}  # legacy, new MP APIs
            else:
                assert len(cp.entries) in {50, 68}  # legacy, new MP APIs
            self.check_O2_entry(cp)

    def test_unknown_host(self):
        """
        Test generating CompetingPhases with a composition that's not on the
        Materials Project database.
        """
        for cp_settings in [
            {"composition": "Cu2SiSe4", "api_key": self.legacy_api_key},
            {"composition": "Cu2SiSe4", "api_key": self.legacy_api_key, "e_above_hull": 0.0},
            {"composition": "Cu2SiSe4", "api_key": self.legacy_api_key, "full_phase_diagram": True},
        ]:
            print(f"Testing with settings: {cp_settings}")
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(**cp_settings)
            print([str(warning.message) for warning in w])  # for debugging
            assert len([warning for warning in w if "You are using" not in str(warning.message)]) == 1
            assert "Note that no Materials Project (MP) database entry exists for Cu2SiSe4. Here" in str(
                w[-1].message
            )
            if cp_settings.get("full_phase_diagram"):
                assert len(cp.entries) == 29
            elif cp_settings.get("e_above_hull") == 0.0:
                assert len(cp.entries) == 8
            else:
                assert len(cp.entries) == 26

            # check naming of fake entry
            assert "Cu2SiSe4_NA_EaH_0" in [entry.data["doped_name"] for entry in cp.entries]

            # TODO: Test file generation functions for an unknown host!

    @parameterized_subtest()
    def test_convergence_setup(self, api_key):
        cp = chemical_potentials.CompetingPhases("ZrO2", e_above_hull=0.03, api_key=api_key)
        # potcar spec doesn't need potcars set up for pmg and it still works
        cp.convergence_setup(potcar_spec=True)
        assert len(cp.metallic_entries) == 6
        assert cp.metallic_entries[0].data["band_gap"] == 0
        assert not cp.nonmetallic_entries[0].data["molecule"]
        # this shouldn't exist - don't need to convergence test for molecules
        assert not os.path.exists("CompetingPhases/O2_Pmmm_EaH_0")

        # test if it writes out the files correctly
        Zro2_EaH_0pt009_folder = "CompetingPhases/ZrO2_Pbca_EaH_0.009/kpoint_converge/k2,1,1/"
        assert os.path.exists(Zro2_EaH_0pt009_folder)
        with open(f"{Zro2_EaH_0pt009_folder}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[3] == "2 1 1\n"

        with open(f"{Zro2_EaH_0pt009_folder}/POTCAR.spec", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[0] == "Zr_sv\n"

        with open(f"{Zro2_EaH_0pt009_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert any(line == "GGA = Ps\n" for line in contents)
            assert any(line == "NSW = 0\n" for line in contents)

    @parameterized_subtest()
    def test_vasp_std_setup(self, api_key):
        cp = chemical_potentials.CompetingPhases("ZrO2", e_above_hull=0.03, api_key=api_key)
        cp.vasp_std_setup(potcar_spec=True)
        assert len(cp.nonmetallic_entries) == 6
        assert len(cp.metallic_entries) == 6
        assert len(cp.molecular_entries) == 1
        assert cp.molecular_entries[0].name == "O2"
        assert cp.molecular_entries[0].data["total_magnetization"] == 2
        assert cp.molecular_entries[0].data["molecule"]
        assert not cp.nonmetallic_entries[0].data["molecule"]

        ZrO2_EaH_0_std_folder = "CompetingPhases/ZrO2_P2_1c_EaH_0/vasp_std/"
        assert os.path.exists(ZrO2_EaH_0_std_folder)
        with open(f"{ZrO2_EaH_0_std_folder}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert "KPOINTS from doped, with reciprocal_density = 64/Å" in contents[0]
            assert contents[3] == "4 4 4\n"

        with open(f"{ZrO2_EaH_0_std_folder}/POTCAR.spec", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents == ["Zr_sv\n", "O"]

        with open(f"{ZrO2_EaH_0_std_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert all(x in contents for x in ["AEXX = 0.25\n", "ISIF = 3\n", "GGA = Pe\n"])

        O2_EaH_0_std_folder = "CompetingPhases/O2_mmm_EaH_0/vasp_std"
        assert os.path.exists(O2_EaH_0_std_folder)
        with open(f"{O2_EaH_0_std_folder}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[3] == "1 1 1\n"

        struct = Structure.from_file(f"{O2_EaH_0_std_folder}/POSCAR")
        assert np.isclose(struct.sites[0].frac_coords, [0.49983339, 0.5, 0.50016672]).all()
        assert np.isclose(struct.sites[1].frac_coords, [0.49983339, 0.5, 0.5405135]).all()

    def test_api_keys_errors(self):
        api_key_error_start = ValueError(
            "The supplied API key (either ``api_key`` or 'PMG_MAPI_KEY' in ``~/.pmgrc.yaml`` or "
            "``~/.config/.pmgrc.yaml``;"
        )
        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="test",
            )
        assert str(api_key_error_start) in str(e.value)
        #
        assert (
            "is not a valid Materials Project API "
            "key, which is required by doped for competing phase generation. See the doped "
            "installation instructions for details:\n"
            "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
            "-project-api"
        ) in str(e.value)

        # test all works fine with key from new MP API:
        assert chemical_potentials.CompetingPhases("ZrO2", api_key="UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv")

    def test_legacy_API_message(self):
        """
        Quick test to check that the message about doped now supporting the new
        Materials Project API is printed to stdout as expected.
        """
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.

        try:
            chemical_potentials.CompetingPhases("Si", api_key=self.legacy_api_key)
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        print(output)  # for debugging
        assert (
            "Note that doped now supports the new Materials Project API, which can be used by updating "
            "your API key in ~/.pmgrc.yaml or ~/.config/.pmgrc.yaml: "
            "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
            "-project-api"
        ) in output

    @parameterized_subtest()
    def test_structure_input(self, api_key):
        for struct, name in [
            (self.cdte, "CdTe_F-43m_EaH_0"),
            (self.cdte * 2, "CdTe_F-43m_EaH_0"),  # supercell
            (self.na2fepo4f, "Na2FePO4F_Pbcn_EaH_0.17"),
            (self.cu2sise4, "Cu2SiSe4_P1_EaH_0"),
        ]:
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(
                    struct.composition.reduced_formula, api_key=api_key
                )
                cp_struct_input = chemical_potentials.CompetingPhases(struct, api_key=api_key)

            _check_structure_input(cp, cp_struct_input, struct, name, w, api_key)


def _check_structure_input(cp, cp_struct_input, struct, name, w, api_key, extrinsic=False):
    print([str(warning.message) for warning in w])  # for debugging
    user_warnings = [warning for warning in w if warning.category is UserWarning]
    if "Na2FePO4F" in name and len(api_key) != 32:  # stable in new MP
        assert len(user_warnings) == 2
        assert "Note that the Materials Project (MP) database entry for Na2FePO4F is not stable" in str(
            user_warnings[0].message
        )
    elif "Cu2SiSe4" in name:
        assert len(user_warnings) == 2
        assert "Note that no Materials Project (MP) database entry exists for Cu2SiSe4" in str(
            user_warnings[0].message
        )
    else:
        assert not user_warnings

    struct_entries = cp_struct_input.entries if not extrinsic else cp_struct_input.intrinsic_entries
    cp_entries = cp.entries if not extrinsic else cp.intrinsic_entries
    for entry in struct_entries:
        if entry.name != "Cu2SiSe4":  # differs in this case due to doubled formula in unit cell
            assert entry in cp_entries  # structure not compared with ``__eq__`` for entries
        if entry.name == struct.composition.reduced_formula:
            if "Na2FePO4F" not in name or len(api_key) != 32:
                assert entry.data["doped_name"] == name
            else:
                assert entry.data["doped_name"] == "Na2FePO4F_Pbcn_EaH_0"  # stable in new MP
            if entry.name != "CdTe" or len(struct) != 16:
                assert entry.structure == struct
            else:  # with supercell input, structure reduced to the primitive cell
                assert entry.structure == get_primitive_structure(struct)

    for entry in cp_entries:
        if entry.name != struct.composition.reduced_formula:
            assert entry in struct_entries

    assert len(struct_entries) <= len(cp_entries)
    assert (
        len([entry for entry in struct_entries if entry.name == struct.composition.reduced_formula]) == 1
    )


class ExtrinsicCompetingPhasesTestCase(unittest.TestCase):  # same setUp and tearDown as above
    # TODO: Need to add tests for co-doping, full_sub_approach, full_phase_diagram etc!!
    def setUp(self):
        CompetingPhasesTestCase.setUp(self)

    def tearDown(self):
        CompetingPhasesTestCase.tearDown(self)

    @parameterized_subtest()
    def test_init(self, api_key):
        ex_cp = chemical_potentials.ExtrinsicCompetingPhases(
            "ZrO2", extrinsic_species="La", e_above_hull=0, api_key=api_key
        )
        assert len(ex_cp.entries) == 2
        assert ex_cp.entries[0].name == "La"  # definite ordering
        assert ex_cp.entries[1].name == "La2Zr2O7"  # definite ordering
        assert all(chemical_potentials._get_e_above_hull(entry.data) == 0 for entry in ex_cp.entries)

        # names of intrinsic entries: ['Zr', 'O2', 'Zr3O', 'ZrO2']
        assert len(ex_cp.intrinsic_entries) == 4
        assert [entry.name for entry in ex_cp.intrinsic_entries] == self.zro2_entry_list[:4]
        assert all(
            chemical_potentials._get_e_above_hull(entry.data) == 0 for entry in ex_cp.intrinsic_entries
        )

        cp = chemical_potentials.ExtrinsicCompetingPhases(
            "ZrO2", extrinsic_species="La", api_key=api_key
        )  # default e_above_hull=0.05
        assert len(cp.entries) == 3
        assert cp.entries[2].name == "La"  # definite ordering, same 1st & 2nd as before
        assert all(chemical_potentials._get_e_above_hull(entry.data) == 0 for entry in cp.entries[:2])
        assert all(chemical_potentials._get_e_above_hull(entry.data) != 0 for entry in cp.entries[2:])
        assert len(cp.intrinsic_entries) == 18

    @parameterized_subtest()
    def test_structure_input(self, api_key):
        for struct, name in [
            (self.cdte, "CdTe_F-43m_EaH_0"),
            (self.cdte * 2, "CdTe_F-43m_EaH_0"),  # supercell
            (self.na2fepo4f, "Na2FePO4F_Pbcn_EaH_0.17"),
            (self.cu2sise4, "Cu2SiSe4_P1_EaH_0"),
        ]:
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.ExtrinsicCompetingPhases(
                    struct.composition.reduced_formula, api_key=api_key, extrinsic_species={"K"}
                )
                cp_struct_input = chemical_potentials.ExtrinsicCompetingPhases(
                    struct, api_key=api_key, extrinsic_species={"K"}
                )

            _check_structure_input(cp, cp_struct_input, struct, name, w, api_key, extrinsic=True)

            for entries_list in [cp_struct_input.entries, cp.entries]:
                assert len(entries_list) >= 1
                for extrinsic_entry in entries_list:
                    assert "K" in extrinsic_entry.data["doped_name"]
                    assert "K" in extrinsic_entry.name


class ChemPotAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        self.EXAMPLE_DIR = os.path.join(cwd, "../examples")
        self.DATA_DIR = os.path.join(cwd, "data")

        self.zro2_path = os.path.join(self.EXAMPLE_DIR, "ZrO2_CompetingPhases")
        self.la_zro2_path = os.path.join(self.EXAMPLE_DIR, "La_ZrO2_CompetingPhases")

        self.zro2_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        self.la_zro2_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.la_zro2_path)

        self.zro2_parsed_chempots = loadfn(f"{self.zro2_path}/zro2_chempots.json")
        self.la_zro2_parsed_chempots = loadfn(f"{self.la_zro2_path}/zro2_la_chempots.json")
        self.y_zro2_parsed_chempots = loadfn(f"{self.la_zro2_path}/zro2_y_chempots.json")

        self.zro2_entry_list = [  # for testing ordering
            "ZrO2",
            "Zr",
            "O2",
            "Zr3O",
            "ZrO2",
            "Zr3O",
            "Zr2O",
            "Zr",
        ]

        self.zro2_chempots_df_dict = {
            "Zr": {"ZrO2-O2": -10.97543, "Zr3O-ZrO2": -0.19954},
            "O": {"ZrO2-O2": 0.0, "Zr3O-ZrO2": -5.38794},
        }
        self.la_zro2_chempots_df_dict = {
            "Zr": {"ZrO2-O2": -10.97543, "Zr3O-ZrO2": -0.19954},
            "O": {"ZrO2-O2": 0.0, "Zr3O-ZrO2": -5.38794},
            "La": {"ZrO2-O2": -9.462985919999998, "Zr3O-ZrO2": -1.3810859199999967},
            "La-Limiting Phase": {"ZrO2-O2": "La2Zr2O7", "Zr3O-ZrO2": "La2Zr2O7"},
        }

    def tearDown(self):
        for i in ["cpa.json"]:
            if_present_rm(i)

        if_present_rm(os.path.join(self.DATA_DIR, "ZrO2_LaTeX_Tables/test.tex"))

    def test_cpa_chempots(self):
        assert isinstance(next(iter(self.zro2_cpa.intrinsic_chempots["elemental_refs"].keys())), str)
        for chempots_df in [self.zro2_cpa.chempots_df, self.zro2_cpa.calculate_chempots()]:
            assert next(iter(chempots_df["O"])) == 0

        for chempots_df in [
            self.la_zro2_cpa.chempots_df,
            self.la_zro2_cpa.calculate_chempots(extrinsic_species="La"),
        ]:
            assert next(iter(chempots_df["La-Limiting Phase"])) == "La2Zr2O7"
            assert np.isclose(next(iter(chempots_df["La"])), -9.46298748)

    def test_unstable_host_chempots(self):
        """
        Test the chemical potentials parsing when the host phase is unstable.
        """
        with warnings.catch_warnings(record=True) as w:
            unstable_cpa = chemical_potentials.CompetingPhasesAnalyzer("Zr2O", self.zro2_path)

        print([str(warning.message) for warning in w])  # for debugging
        assert (
            "Zr2O is not stable with respect to competing phases, having an energy "
            "above hull of 0.0194 eV/atom.\nFormally, this means that"
        ) in str(w[0].message)
        assert (
            "just a metastable phase.\nHere we will determine a single chemical potential 'limit' "
            "corresponding to the least unstable (i.e. closest) point on the convex hull for the host "
            "material, as an approximation for the true chemical potentials."
        ) in str(w[0].message)
        assert unstable_cpa.chempots_df.index.tolist() == ["Zr2O-ZrO2"]
        assert np.isclose(next(iter(unstable_cpa.chempots_df["Zr"])), -0.1997, atol=1e-3)
        assert np.isclose(next(iter(unstable_cpa.chempots_df["O"])), -5.3878, atol=1e-3)

        assert unstable_cpa.chempots["elemental_refs"] == self.zro2_parsed_chempots["elemental_refs"]
        assert len(unstable_cpa.chempots["limits"]) == 1
        assert len(unstable_cpa.chempots["limits_wrt_el_refs"]) == 1
        assert np.isclose(unstable_cpa.chempots["limits"]["Zr2O-ZrO2"]["Zr"], -10.0434, atol=1e-3)
        assert np.isclose(unstable_cpa.chempots["limits"]["Zr2O-ZrO2"]["O"], -12.3944, atol=1e-3)
        assert np.isclose(
            unstable_cpa.chempots["limits_wrt_el_refs"]["Zr2O-ZrO2"]["Zr"], -0.1997, atol=1e-3
        )
        assert np.isclose(
            unstable_cpa.chempots["limits_wrt_el_refs"]["Zr2O-ZrO2"]["O"], -5.3878, atol=1e-3
        )

    def test_ext_cpa_chempots(self):
        # test accessing cpa.chempots without previously calling cpa.calculate_chempots()
        _compare_chempot_dicts(self.zro2_cpa.chempots, self.zro2_parsed_chempots)

        assert (
            self.la_zro2_cpa.chempots["elemental_refs"] == self.la_zro2_parsed_chempots["elemental_refs"]
        )

    def test_sort_by(self):
        chempot_df = self.zro2_cpa.calculate_chempots(sort_by="Zr")
        assert np.isclose(next(iter(chempot_df["Zr"])), -0.199544, atol=1e-4)
        assert np.isclose(list(chempot_df["Zr"])[1], -10.975428439999998, atol=1e-4)

        with pytest.raises(KeyError):
            self.zro2_cpa.calculate_chempots(sort_by="M")

    def test_vaspruns(self):
        cpa = self.zro2_cpa
        assert len(cpa.elements) == 2

        self._general_cpa_check(cpa)
        assert cpa.chempots_df.to_dict() == self.zro2_chempots_df_dict

        cpa_w_subfolder = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2", self.zro2_path, subfolder="vasp_std"
        )
        self._general_cpa_check(cpa_w_subfolder)
        self._compare_cpas(cpa, cpa_w_subfolder)

        with pytest.raises(FileNotFoundError) as e:
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", entries="path", subfolder="vasp_std")
        assert "No such file or directory" in str(e.value)

        with pytest.raises(TypeError) as e:
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", entries=0, subfolder="vasp_std")
        assert "`entries` must be either a path to a directory" in str(e.value)
        assert "got type <class 'int'>" in str(e.value)

        ext_cpa = self.la_zro2_cpa
        assert len(ext_cpa.elements) == 3
        assert len(ext_cpa.extrinsic_elements) == 1
        # sorted by num_species, then alphabetically, then by num_atoms_in_fu, then by
        # formation_energy
        assert [entry.reduced_formula for entry in ext_cpa.entries] == [
            "ZrO2",
            "La",
            "Zr",
            "O2",
            "La2O3",
            "Zr3O",
            "La2Zr2O7",
            "ZrO2",
            "Zr3O",
            "Zr2O",
            "Zr",
        ]
        assert ext_cpa.chempots_df.to_dict() == self.la_zro2_chempots_df_dict

        # check if it works from a list
        all_paths = []
        for entry_folder in os.listdir(self.zro2_path):
            if os.path.isdir(os.path.join(self.zro2_path, entry_folder)) and "vasp_std" in os.listdir(
                os.path.join(self.zro2_path, entry_folder)
            ):
                all_paths.extend(
                    os.path.join(self.zro2_path, entry_folder, "vasp_std", vr_file)
                    for vr_file in os.listdir(os.path.join(self.zro2_path, entry_folder, "vasp_std"))
                    if vr_file.startswith("vasprun.xml")
                )
        lst_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", entries=all_paths)
        assert len(lst_cpa.elements) == 2
        assert len(lst_cpa.vasprun_paths) == 8
        self._compare_cpas(lst_cpa, cpa)
        self._general_cpa_check(lst_cpa)

        all_folders = [path.rsplit("/vasprun.xml")[0] for path in all_paths]
        lst_fols_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", entries=all_folders)
        assert len(lst_fols_cpa.elements) == 2
        self._compare_cpas(lst_fols_cpa, cpa)
        self._general_cpa_check(lst_fols_cpa)

        # TODO: all_folders = [path.split("/relax")[0] for path in all_paths] should work in future code
        #  (i.e. only needing to specify top folder and auto detecting the subfolders)

    def test_vaspruns_hidden_files(self):
        with open(f"{self.zro2_path}/._OUTCAR", "w") as f:
            f.write("test pop")
        with open(f"{self.zro2_path}/._vasprun.xml", "w") as f:
            f.write("test pop")
        with open(f"{self.zro2_path}/._LOCPOT", "w") as f:
            f.write("test pop")
        with open(f"{self.zro2_path}/.DS_Store", "w") as f:
            f.write("test pop")

        with warnings.catch_warnings(record=True) as w:
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert not w

        for i in ["._OUTCAR", "._vasprun.xml", "._LOCPOT", ".DS_Store"]:
            if_present_rm(f"{self.zro2_path}/{i}")

    def test_vaspruns_none_parsed(self):
        with warnings.catch_warnings(record=True) as w, pytest.raises(FileNotFoundError) as e:
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", "./")
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        print(e.value)  # for debugging
        assert "No vasprun files have been parsed," in str(e.value)

    def test_latex_table(self):
        cpa = self.zro2_cpa

        def _test_latex_table(cpa=cpa, ref_filename="default.tex", **kwargs):
            out, text, w = _run_func_and_capture_stdout_warnings(cpa.to_LaTeX_table, **kwargs)
            assert not out
            assert not w

            with open(f"{self.DATA_DIR}/ZrO2_LaTeX_Tables/test.tex", "w+") as f:
                f.write(text)

            with (
                open(f"{self.DATA_DIR}/ZrO2_LaTeX_Tables/{ref_filename}") as reference_f,
                open(f"{self.DATA_DIR}/ZrO2_LaTeX_Tables/test.tex") as test_f,
            ):
                assert reference_f.read() == test_f.read()

        for kwargs, ref_filename in [
            ({}, "default.tex"),
            ({"splits": 2}, "splits_2.tex"),
            ({"prune_polymorphs": False}, "no_prune.tex"),
        ]:
            _test_latex_table(ref_filename=ref_filename, **kwargs)

        _test_latex_table(self.la_zro2_cpa, "La_default.tex")

        with pytest.raises(ValueError):
            cpa.to_LaTeX_table(splits=3)

    def test_elements(self):
        struct, mag = chemical_potentials.make_molecule_in_a_box("O2")
        assert mag == 2
        assert type(struct) == Structure

        with pytest.raises(ValueError):
            chemical_potentials.make_molecule_in_a_box("R2")

    def test_combine_extrinsic(self):
        d = chemical_potentials.combine_extrinsic(
            self.la_zro2_parsed_chempots, self.y_zro2_parsed_chempots, "Y"
        )
        assert len(d["elemental_refs"].keys()) == 4
        limits = list(d["limits"].keys())
        assert limits[0].rsplit("-", 1)[1] == "Y2Zr2O7"

    def test_combine_extrinsic_errors(self):
        d = {"a": 1}
        with pytest.raises(KeyError):
            chemical_potentials.combine_extrinsic(d, self.y_zro2_parsed_chempots, "Y")

        with pytest.raises(KeyError):
            chemical_potentials.combine_extrinsic(self.la_zro2_parsed_chempots, d, "Y")

        with pytest.raises(ValueError):
            chemical_potentials.combine_extrinsic(
                self.la_zro2_parsed_chempots, self.y_zro2_parsed_chempots, "R"
            )

    def test_get_formation_energy_df(self):
        cpa = self.zro2_cpa

        def _check_zro2_form_e_df(
            form_e_df, skip_rounding=False, include_dft_energies=False, prune_polymorphs=False
        ):
            if not prune_polymorphs:
                assert len(form_e_df) == len(cpa.entries)  # all entries
            else:
                assert (
                    len(form_e_df) == 5
                )  # only ground states of each phase (including Zr2O with EaH > 0)
                assert len(set(form_e_df.index.to_numpy())) == len(form_e_df)  # no duplicates

            assert form_e_df.index.to_numpy().tolist() == (
                self.zro2_entry_list if not prune_polymorphs else self.zro2_entry_list[:4] + ["Zr2O"]
            )
            space_groups = ["P2_1/c", "P6_3/mmc", "P4/mmm", "R-3c", "Pbca", "P6_322", "P312", "Ibam"]
            assert form_e_df["Space Group"].to_numpy().tolist() == (
                space_groups if not prune_polymorphs else space_groups[:4] + ["P312"]
            )
            assert np.allclose(form_e_df["Energy above Hull (eV/atom)"].to_numpy()[:4], 0)  # stable phases

            for formula, series in form_e_df.iterrows():
                comp = Composition(formula)
                assert np.isclose(
                    series["Formation Energy (eV/fu)"],
                    series["Formation Energy (eV/atom)"] * comp.num_atoms,
                    atol=2e-3,
                )
                if include_dft_energies:
                    assert np.isclose(
                        series["DFT Energy (eV/fu)"],
                        series["DFT Energy (eV/atom)"] * comp.num_atoms,
                        atol=2e-3,
                    )

            assert ("DFT Energy (eV/fu)" in form_e_df.columns) == include_dft_energies
            assert ("DFT Energy (eV/atom)" in form_e_df.columns) == include_dft_energies

            # assert values are all rounded to 3 dp:
            assert form_e_df.round(3).equals(form_e_df) == (not skip_rounding)

        for kwargs in [
            {},
            {"skip_rounding": True},
            {"include_dft_energies": True},
            {"skip_rounding": True, "include_dft_energies": True},
            {"prune_polymorphs": True},
            {"prune_polymorphs": True, "skip_rounding": True, "include_dft_energies": True},
        ]:
            _check_zro2_form_e_df(cpa.get_formation_energy_df(**kwargs), **kwargs)

        la_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            composition="ZrO2",
            entries=self.la_zro2_path,
        )
        la_form_e_df = la_cpa.get_formation_energy_df()
        assert len(la_form_e_df) == len(la_cpa.entries)
        assert la_form_e_df.index.to_numpy()[1] == "La"
        assert la_form_e_df.loc["La"].tolist() == ["P6_3/mmc", 0.0, 0.0, 0.0, "10x10x3"]
        assert la_form_e_df.iloc[4].tolist() == ["Ia-3", 0.0, -18.017, -3.603, "3x3x3"]  # La2O3
        assert la_form_e_df.iloc[6].tolist() == ["Fd-3m", 0.0, -40.877, -3.716, "3x3x3"]  # La2Zr2O7

    def test_repr(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        assert (
            "doped CompetingPhasesAnalyzer for bulk composition ZrO2 with 8 entries (in self.entries):"
            in repr(cpa)
        )
        for entry in cpa.entries:
            assert entry.data.get("doped_name", "N/A") in repr(cpa)
        assert "Available attributes:" in repr(cpa)
        assert "Available methods:" in repr(cpa)

        la_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            composition="ZrO2",
            entries=self.la_zro2_path,
        )
        assert (
            "doped CompetingPhasesAnalyzer for bulk composition ZrO2 with 11 entries (in self.entries):"
            in repr(la_cpa)
        )
        for entry in la_cpa.entries:
            assert entry.data.get("doped_name", "N/A") in repr(la_cpa)
        assert "Available attributes:" in repr(la_cpa)
        assert "Available methods:" in repr(la_cpa)

    def _compare_cpas(self, cpa_a, cpa_b):
        def cleanse_entries(entries):
            """
            ``Vasprun.get_computed_entry`` sets the ``entry_id`` to
            f"vasprun-{datetime.now(tz=timezone.utc)}", so remove to allow
            comparison.
            """
            for entry in entries:
                entry.entry_id = None
            return entries

        for attr in [
            "entries",
            "chempots",
            "extrinsic_elements",
            "elements",
            "vasprun_paths",
            "parsed_folders",
            "unstable_host",
            "bulk_entry",
            "composition",
            "phase_diagram",
            "chempots_df",
        ]:
            print(f"Checking {attr}")
            if attr == "chempots_df":
                assert cpa_a.chempots_df.equals(cpa_b.chempots_df)
            elif attr == "phase_diagram":
                assert cleanse_entries(cpa_a.phase_diagram.entries) == cleanse_entries(
                    cpa_b.phase_diagram.entries
                )
            elif attr == "entries":
                assert cleanse_entries(cpa_a.entries) == cleanse_entries(cpa_b.entries)
            else:
                assert getattr(cpa_a, attr) == getattr(cpa_b, attr)

    def _general_cpa_check(self, cpa):
        cpa_dict = cpa.as_dict()
        cpa_from_dict = chemical_potentials.CompetingPhasesAnalyzer.from_dict(cpa_dict)
        self._compare_cpas(cpa, cpa_from_dict)

        dumpfn(cpa_dict, "cpa.json")
        reloaded_cpa = loadfn("cpa.json")
        self._compare_cpas(cpa, reloaded_cpa)

    def test_general_cpa_reloading(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        self._general_cpa_check(cpa)

        la_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.la_zro2_path)
        self._general_cpa_check(la_cpa)


class TestChemicalPotentialGrid(unittest.TestCase):
    def setUp(self):
        self.EXAMPLE_DIR = os.path.join(cwd, "../examples")
        self.chempots = loadfn(os.path.join(self.EXAMPLE_DIR, "Cu2SiSe3/Cu2SiSe3_chempots.json"))
        self.grid = chemical_potentials.ChemicalPotentialGrid(self.chempots)
        self.new_MP_api_key = "UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv"

    def test_init(self):
        assert isinstance(self.grid.vertices, pd.DataFrame)
        assert len(self.grid.vertices) == 7
        assert np.isclose(max(self.grid.vertices["μ_Cu"]), 0.0)
        assert np.isclose(max(self.grid.vertices["μ_Si"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(self.grid.vertices["μ_Se"]), 0.0)
        assert np.isclose(min(self.grid.vertices["μ_Cu"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["μ_Si"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["μ_Se"]), -0.758105, rtol=1e-5)

    def test_get_grid(self):
        grid_df = self.grid.get_grid(100)
        assert isinstance(grid_df, pd.DataFrame)
        assert len(self.grid.vertices) == 7
        assert np.isclose(max(self.grid.vertices["μ_Cu"]), 0.0)
        assert np.isclose(max(self.grid.vertices["μ_Si"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(self.grid.vertices["μ_Se"]), 0.0)
        assert np.isclose(min(self.grid.vertices["μ_Cu"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["μ_Si"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["μ_Se"]), -0.758105, rtol=1e-5)
        assert len(grid_df) == 3886

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_grid.png")
    def test_Na2FePO4F_chempot_grid(self):
        """
        Test ``ChemicalPotentialGrid`` generation and plotting for a complex
        quinary system (Na2FePO4F).
        """
        na2fepo4f_cp = chemical_potentials.CompetingPhases("Na2FePO4F", api_key=self.new_MP_api_key)
        na2fepo4f_doped_chempots = chemical_potentials.get_doped_chempots_from_entries(
            na2fepo4f_cp.entries, "Na2FePO4F"
        )
        chempot_grid = chemical_potentials.ChemicalPotentialGrid(na2fepo4f_doped_chempots)
        grid_df = chempot_grid.get_grid(100)

        # get the average Fe and P chempots, then plot a heatmap plot of the others at these fixed values:
        mean_mu_Fe = grid_df["μ_Fe"].mean()
        mean_mu_P = grid_df["μ_P"].mean()

        fixed_chempot_df = grid_df[
            (np.isclose(grid_df["μ_Fe"], mean_mu_Fe, atol=0.05))
            & (np.isclose(grid_df["μ_P"], mean_mu_P, atol=0.05))
        ]

        fig, ax = plt.subplots()
        sc = ax.scatter(
            fixed_chempot_df["μ_Na"], fixed_chempot_df["μ_O"], c=fixed_chempot_df["μ_F"], cmap="viridis"
        )
        fig.colorbar(sc, ax=ax, label="μ$_F$")
        ax.set_xlabel("μ$_{Na}$")
        ax.set_ylabel("μ$_{O}$")
        return fig

        # TODO: Use this as a plotting example in chemical potentials tutorial


# TODO: Use Cs2SnBr6 competing phase energies csv in JOSS data folder and LiPS4 data for test cases with
#  chempot plotting etc
