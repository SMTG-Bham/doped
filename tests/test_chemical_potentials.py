"""
Tests for the ``doped.chemical_potentials`` module.
"""

import os
import shutil
import unittest
import warnings
from copy import deepcopy

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


module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")
EXAMPLE_DIR = os.path.join(module_path, "../examples")


class CompetingPhasesTestCase(unittest.TestCase):
    def setUp(self):
        self.api_key = "UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv"  # SK MP Imperial email (GitHub) A/C
        self.cdte = Structure.from_file(os.path.join(EXAMPLE_DIR, "CdTe/relaxed_primitive_POSCAR"))
        self.na2fepo4f = Structure.from_file(os.path.join(data_dir, "Na2FePO4F_MP_POSCAR"))
        self.cu2sise3 = Structure.from_file(os.path.join(data_dir, "Cu2SiSe3_MP_POSCAR"))
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
            eah = entry.data.get("energy_above_hull")
            assert eah == 0 if i < num_stable_entries else eah > 0  # Zr4O is on hull

            mag = entry.data["summary"]["total_magnetization"]
            is_molecule = entry.data["molecule"]

            assert is_molecule if entry.name == "O2" else not is_molecule
            assert np.isclose(
                mag, 2 if entry.name == "O2" else 0, atol=1e-3
            )  # only O2 is magnetic (triplet) here
            if entry.name == "O2":
                assert np.isclose(entry.data["energy_per_atom"], -4.94795546875)
                assert np.isclose(entry.energy, -4.94795546875 * 2)

    def test_make_molecule_in_a_box(self):
        allowed_gaseous_elements = ["O2", "N2", "H2", "F2", "Cl2"]
        for element in allowed_gaseous_elements:
            structure, total_magnetization = chemical_potentials.make_molecule_in_a_box(element)
            if element == "O2":
                assert total_magnetization == 2
            else:
                assert total_magnetization == 0
            assert structure.composition.reduced_formula == element
            assert structure.num_sites == 2
            assert np.isclose(structure.volume, 30**3)

        with pytest.raises(ValueError) as exc:
            chemical_potentials.make_molecule_in_a_box("Te")
        assert "Element Te is not currently supported for molecule-in-a-box structure generation." in str(
            exc.value
        )

    def test_init(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=self.api_key)

        assert len(cp.entries) == 13
        assert [entry.name for entry in cp.entries] == self.zro2_entry_list
        self._check_ZrO2_cp_init(cp)
        assert "Zr4O" not in [e.name for e in cp.entries]  # not bordering or potentially with EaH
        assert not cp.MP_doc_dicts

    def test_init_full_phase_diagram(self):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", energy_above_hull=0.03, api_key=self.api_key, full_phase_diagram=True
        )

        assert len(cp.entries) == 14  # Zr4O now present
        zro2_full_pd_entry_list = self.zro2_entry_list[:4] + ["Zr4O"] + self.zro2_entry_list[4:]
        assert [entry.name for entry in cp.entries] == zro2_full_pd_entry_list
        self._check_ZrO2_cp_init(cp, num_stable_entries=5)  # Zr4O is on hull

    def test_init_ZnSe(self):
        """
        As noted by Savya Aggarwal, the legacy MP API code didn't return ZnSe2
        as a competing phase despite being on the hull and bordering ZnSe,
        because the legacy MP API database wrongly had the
        ``data['energy_above_hull']`` value as 0.147 eV/atom (when it should be
        0 eV/atom).

        https://legacy.materialsproject.org/materials/mp-1102515/
        https://next-gen.materialsproject.org/materials/mp-1102515?formula=ZnSe2

        Updated code which re-calculates the energy above hull avoids this
        issue, though ``pymatgen`` has now updated to no longer support the
        legacy MP database now anyway.
        """
        cp = chemical_potentials.CompetingPhases("ZnSe", api_key=self.api_key)
        assert any(e.name == "ZnSe2" for e in cp.entries)
        assert len(cp.entries) == 12  # ZnSe2 present; 2 new Zn entries (mp-264...) with new MP API
        znse2_entry = next(e for e in cp.entries if e.name == "ZnSe2")
        assert znse2_entry.data.get("energy_above_hull") == 0
        assert not znse2_entry.data["molecule"]
        assert np.isclose(znse2_entry.energy_per_atom, -3.394683861)
        assert np.isclose(znse2_entry.energy, -3.394683861 * 12)

    def test_init_YTOS(self):
        # 144 phases on Y-Ti-O-S MP phase diagram
        cp = chemical_potentials.CompetingPhases("Y2Ti2S2O5", energy_above_hull=0.1, api_key=self.api_key)
        assert len(cp.entries) == 113
        self.check_O2_entry(cp)

        cp = chemical_potentials.CompetingPhases(
            "Y2Ti2S2O5", energy_above_hull=0.1, full_phase_diagram=True, api_key=self.api_key
        )
        # 149 phases on Y-Ti-O-S MP full phase diagram, 4 extra O2 phases removed
        assert len(cp.entries) == 145
        self.check_O2_entry(cp)

    def check_O2_entry(self, cp):
        # assert only one O2 phase present (molecular entry):
        result = [e for e in cp.entries if e.name == "O2"]
        assert len(result) == 1
        assert result[0].name == "O2"
        assert result[0].data["summary"]["total_magnetization"] == 2
        assert result[0].data["energy_above_hull"] == 0
        assert result[0].data["molecule"]
        assert np.isclose(result[0].data["energy_per_atom"], -4.94795546875)

    def test_entry_naming(self):
        """
        Test the naming functions for competing phase entries in ``doped``,
        including rounding to "_0" and increasing the number of digits if
        duplicates are encountered.
        """
        cdte_cp = chemical_potentials.CompetingPhases("CdTe", api_key=self.api_key)
        assert [entry.data["doped_name"] for entry in cdte_cp.entries] == [
            "CdTe_F-43m_EaH_0",
            "Cd_Fm-3m_EaH_0",
            "Te_P3_121_EaH_0",
            "Te_P3_221_EaH_0",
            "CdTe_P6_3mc_EaH_0.006",
            "CdTe_Cmc2_1_EaH_0.009",
            "Cd_P6_3/mmc_EaH_0.014",
            "Cd_R-3m_EaH_0.018",
            "Cd_P6_3/mmc_EaH_0.034",
            "Te_C2/m_EaH_0.044",
            "Te_Pm-3m_EaH_0.047",
            "Te_Pmma_EaH_0.047",
            "Te_Pmc2_1_EaH_0.049",
        ]

        # test case when the EaH rounding needs to be dynamically updated:
        # (this will be quite a rare case, as it requires two phases with the same formula, space group
        # and energy above hull to 1 meV/atom
        cds_cp = chemical_potentials.CompetingPhases("CdS", api_key=self.api_key)
        assert "S_Pnnm_EaH_0.014" in [entry.data["doped_name"] for entry in cds_cp.entries]
        new_entry = deepcopy(
            next(entry for entry in cds_cp.entries if entry.data["doped_name"] == "S_Pnnm_EaH_0.014")
        )  # duplicate entry to force renaming
        new_entry.data["energy_above_hull"] += 2e-4
        chemical_potentials._name_entries_and_handle_duplicates([*cds_cp.entries, new_entry])
        entry_names = [entry.data["doped_name"] for entry in [*cds_cp.entries, new_entry]]
        assert "S_Pnnm_EaH_0.014" not in entry_names
        assert "S_Pnnm_EaH_0.0141" in entry_names
        assert "S_Pnnm_EaH_0.0143" in entry_names

    def test_unstable_host(self):
        """
        Test generating CompetingPhases with a composition that's unstable on
        the Materials Project database.
        """
        for cp_settings in [
            {"composition": "Na2FePO4F", "energy_above_hull": 0.02, "api_key": self.api_key},
            {
                "composition": "Na2FePO4F",
                "energy_above_hull": 0.02,
                "api_key": self.api_key,
                "full_phase_diagram": True,
            },
        ]:
            print(f"Testing with settings: {cp_settings}")
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(**cp_settings)
            print([str(warning.message) for warning in w])  # for debugging
            if cp_settings.get("full_phase_diagram"):
                assert len(cp.entries) == 172
            else:
                assert len(cp.entries) == 68
            self.check_O2_entry(cp)

    def test_unknown_host(self):
        """
        Test generating CompetingPhases with a composition that's not on the
        Materials Project database.
        """
        unknown_host_cp_kwargs = {"composition": "Cu2SiSe4", "api_key": self.api_key}
        for cp_settings in [
            {},
            {"energy_above_hull": 0.0},
            {"full_phase_diagram": True},
        ]:
            kwargs = {**unknown_host_cp_kwargs, **cp_settings}
            print(f"Testing with settings: {kwargs}")
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(**kwargs)
            print([str(warning.message) for warning in w])  # for debugging
            assert len(w) == 1
            assert "Note that no Materials Project (MP) database entry exists for Cu2SiSe4. Here" in str(
                w[-1].message
            )
            if kwargs.get("full_phase_diagram"):
                assert len(cp.entries) == 29
            elif kwargs.get("energy_above_hull") == 0.0:
                assert len(cp.entries) == 8
            else:
                assert len(cp.entries) == 26

            # check naming of fake entry
            assert "Cu2SiSe4_NA_EaH_0" in [entry.data["doped_name"] for entry in cp.entries]

            # TODO: Test file generation functions for an unknown host!

    def test_convergence_setup(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=self.api_key)
        # potcar spec doesn't need potcars set up for pmg and it still works
        cp.convergence_setup(potcar_spec=True)
        assert len(cp.metallic_entries) == 6
        assert cp.metallic_entries[0].data["summary"]["band_gap"] == 0
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

    def test_vasp_std_setup(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=self.api_key)
        cp.vasp_std_setup(potcar_spec=True)
        assert len(cp.nonmetallic_entries) == 6
        assert len(cp.metallic_entries) == 6
        assert len(cp.molecular_entries) == 1
        assert cp.molecular_entries[0].name == "O2"
        assert cp.molecular_entries[0].data["summary"]["total_magnetization"] == 2
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
        api_key_error = ValueError(
            "Invalid or old API key. Please obtain an updated API key at https://materialsproject.org/dashboard."
        )
        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="test",
            )
        assert str(api_key_error) in str(e.value)

        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="c2LiJRMiBeaN5iXsH",  # legacy API key
            )
        assert str(api_key_error) in str(e.value)

        # test all works fine with key from new MP API:
        assert chemical_potentials.CompetingPhases("ZrO2", api_key="UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv")

    def test_structure_input(self):
        for struct, name in [
            (self.cdte, "CdTe_F-43m_EaH_0"),
            (self.cdte * 2, "CdTe_F-43m_EaH_0"),  # supercell
            (self.na2fepo4f, "Na2FePO4F_Pbcn_EaH_0.17"),
            (self.cu2sise4, "Cu2SiSe4_P1_EaH_0"),
        ]:
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(
                    struct.composition.reduced_formula, api_key=self.api_key
                )
                cp_struct_input = chemical_potentials.CompetingPhases(struct, api_key=self.api_key)

            _check_structure_input(cp, cp_struct_input, struct, name, w, self.api_key)

    def test_MP_doc_dicts(self):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", MP_doc_dicts=True, energy_above_hull=0.03, api_key=self.api_key
        )
        assert cp.MP_doc_dicts
        assert len(cp.MP_doc_dicts) == 12  # just missing O2
        assert len(cp.entries) == 13
        assert set(cp.MP_doc_dicts.keys()) == {
            entry.data["material_id"] for entry in cp.entries if not entry.data["molecule"]
        }
        assert [entry.name for entry in cp.entries] == self.zro2_entry_list
        self._check_ZrO2_cp_init(cp)
        assert "Zr4O" not in [e.name for e in cp.entries]  # not bordering or potentially with EaH


def _check_structure_input(cp, cp_struct_input, struct, name, w, api_key, extrinsic=False):
    print([str(warning.message) for warning in w])  # for debugging
    user_warnings = [warning for warning in w if warning.category is UserWarning]
    if "Cu2SiSe4" in name:
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
            if "Na2FePO4F" not in name:
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

    def test_init(self):
        ex_cp = chemical_potentials.CompetingPhases(
            "ZrO2", extrinsic="La", energy_above_hull=0, api_key=self.api_key
        )
        assert len(ex_cp.extrinsic_entries) == 2
        assert len(ex_cp.entries) == 6
        assert ex_cp.extrinsic_entries[0].name == "La"  # definite ordering
        assert ex_cp.extrinsic_entries[1].name == "La2Zr2O7"  # definite ordering
        assert all(entry.data["energy_above_hull"] == 0 for entry in ex_cp.entries)

        # names of intrinsic entries: ['Zr', 'O2', 'Zr3O', 'ZrO2']
        assert len(ex_cp.intrinsic_entries) == 4
        assert [entry.name for entry in ex_cp.intrinsic_entries] == self.zro2_entry_list[:4]

        cp = chemical_potentials.CompetingPhases(
            "ZrO2", extrinsic="La", api_key=self.api_key
        )  # default energy_above_hull=0.05
        assert len(cp.extrinsic_entries) == 3
        assert len(cp.entries) == 21
        assert cp.extrinsic_entries[2].name == "La"  # definite ordering, same 1st & 2nd as before
        assert all(entry.data["energy_above_hull"] == 0 for entry in ex_cp.extrinsic_entries[:2])
        assert all(entry.data["energy_above_hull"] != 0 for entry in cp.extrinsic_entries[2:])
        assert len(cp.intrinsic_entries) == 18

    def test_structure_input(self):
        for struct, name in [
            (self.cdte, "CdTe_F-43m_EaH_0"),
            (self.cdte * 2, "CdTe_F-43m_EaH_0"),  # supercell
            (self.na2fepo4f, "Na2FePO4F_Pbcn_EaH_0.17"),
            (self.cu2sise4, "Cu2SiSe4_P1_EaH_0"),
        ]:
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(
                    struct.composition.reduced_formula, api_key=self.api_key, extrinsic={"K"}
                )
                cp_struct_input = chemical_potentials.CompetingPhases(
                    struct, api_key=self.api_key, extrinsic={"K"}
                )

            _check_structure_input(cp, cp_struct_input, struct, name, w, self.api_key, extrinsic=True)

            for entries_list in [cp_struct_input.extrinsic_entries, cp.extrinsic_entries]:
                assert len(entries_list) >= 1
                for extrinsic_entry in entries_list:
                    assert "K" in extrinsic_entry.data["doped_name"]
                    assert "K" in extrinsic_entry.name


class ChemPotAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        self.zro2_path = os.path.join(EXAMPLE_DIR, "ZrO2_CompetingPhases")
        self.la_zro2_path = os.path.join(EXAMPLE_DIR, "La_ZrO2_CompetingPhases")
        self.mgo_path = os.path.join(EXAMPLE_DIR, "MgO/CompetingPhases")

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
            "La": {"ZrO2-O2": -9.463, "Zr3O-ZrO2": -1.3811},
            "La-Limiting Phase": {"ZrO2-O2": "La2Zr2O7", "Zr3O-ZrO2": "La2Zr2O7"},
        }

    def tearDown(self):
        for i in ["cpa.json"]:
            if_present_rm(i)

        if_present_rm(os.path.join(data_dir, "ZrO2_LaTeX_Tables/test.tex"))

        if os.path.exists(f"{self.zro2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz"):
            if not os.path.exists(f"{self.zro2_path}/O2_EaH_0.0/vasp_std/mismatching_incar_vr.xml.gz"):
                shutil.move(
                    f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
                    f"{self.zro2_path}/O2_EaH_0.0/vasp_std/mismatching_incar_vr.xml.gz",
                )
            if not os.path.exists(f"{self.zro2_path}/O2_EaH_0.0/vasp_std/mismatching_potcar_vr.xml.gz"):
                shutil.move(
                    f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
                    f"{self.zro2_path}/O2_EaH_0.0/vasp_std/mismatching_potcar_vr.xml.gz",
                )
            shutil.move(
                f"{self.zro2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz",
                f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            )

        shutil.copyfile(
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            f"{self.la_zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        if_present_rm(
            os.path.join(
                data_dir,
                "Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0/duplicate_for_testing_vasprun.xml.gz",
            )
        )

    def test_cpa_chempots(self):
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

            with open(f"{data_dir}/ZrO2_LaTeX_Tables/test.tex", "w+") as f:
                f.write(text)

            with (
                open(f"{data_dir}/ZrO2_LaTeX_Tables/{ref_filename}") as reference_f,
                open(f"{data_dir}/ZrO2_LaTeX_Tables/test.tex") as test_f,
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
            if prune_polymorphs:
                assert (
                    len(form_e_df) == 5
                )  # only ground states of each phase (including Zr2O with EaH > 0)

            assert form_e_df.index.to_numpy().tolist() == (
                self.zro2_entry_list if not prune_polymorphs else self.zro2_entry_list[:4] + ["Zr2O"]
            )
            space_groups = ["P2_1/c", "P6_3/mmc", "P4/mmm", "R-3c", "Pbca", "P6_322", "P312", "Ibam"]
            assert form_e_df["Space Group"].to_numpy().tolist() == (
                space_groups if not prune_polymorphs else space_groups[:4] + ["P312"]
            )
            assert np.allclose(form_e_df["Energy above Hull (eV/atom)"].to_numpy()[:4], 0)  # stable phases

            _check_form_e_df(cpa, form_e_df, skip_rounding, include_dft_energies, prune_polymorphs)

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
        intrinsic_el_refs = cpa.intrinsic_chempots["elemental_refs"]
        assert isinstance(next(iter(intrinsic_el_refs.keys())), str)
        for chempots_df in [cpa.chempots_df, cpa.calculate_chempots()]:
            for el_ref in intrinsic_el_refs:
                assert el_ref in chempots_df.columns

        # test formation energy df:
        for kwargs in [
            {},
            {"skip_rounding": True},
            {"include_dft_energies": True},
            {"skip_rounding": True, "include_dft_energies": True},
            {"prune_polymorphs": True},
            {"prune_polymorphs": True, "skip_rounding": True, "include_dft_energies": True},
        ]:
            _check_form_e_df(cpa, cpa.get_formation_energy_df(**kwargs), **kwargs)

        # test chempots dict:
        assert isinstance(cpa.chempots, dict)
        # limits is equal to limits_wrt_el_refs + elemental_refs:
        for limit_name, limit_dict in cpa.chempots["limits"].items():
            for elt_name, elt_value in limit_dict.items():
                assert np.isclose(
                    elt_value,
                    cpa.chempots["limits_wrt_el_refs"][limit_name][elt_name]
                    + cpa.chempots["elemental_refs"][elt_name],
                )

        # test to/from dict:
        cpa_dict = cpa.as_dict()
        cpa_from_dict = chemical_potentials.CompetingPhasesAnalyzer.from_dict(cpa_dict)
        self._compare_cpas(cpa, cpa_from_dict)

        dumpfn(cpa_dict, "cpa.json")
        reloaded_cpa = loadfn("cpa.json")
        self._compare_cpas(cpa, reloaded_cpa)

    def test_general_cpa_reloading(self):
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        self._general_cpa_check(cpa)

        with warnings.catch_warnings(record=True) as w:
            la_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.la_zro2_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        self._general_cpa_check(la_cpa)

    def test_mismatching_incar_warnings(self):
        """
        Test warnings for mismatching INCAR settings.

        No warnings for ZrO2 / La_ZrO2 already checked in ``self.test_general_cpa_reloading()`` above.
        """
        # convert to mismatching O2 calc:
        shutil.move(
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz",
        )
        shutil.move(
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/mismatching_incar_vr.xml.gz",
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        print([str(warning.message) for warning in w])  # for debugging
        expected_mismatching_info = [
            "There are mismatching INCAR tags",
            "['O2']:",
            "Where ZrO2 was used as the reference entry calculation.",
            "[('HFSCREEN', 0.20786986, 0.2), ('LREAL', 'Auto      ! projection operators: autom', "
            "False)]",
        ]
        assert all(any(i in str(warning.message) for warning in w) for i in expected_mismatching_info)
        self._general_cpa_check(cpa)

        # test no warning with check_compatibility=False:
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer(
                "ZrO2", self.zro2_path, check_compatibility=False
            )
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        self._general_cpa_check(cpa)

        # test with extrinsic case:
        shutil.copyfile(
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",  # this is mismatching vr
            f"{self.la_zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            la_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.la_zro2_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert all(any(i in str(warning.message) for warning in w) for i in expected_mismatching_info)
        self._general_cpa_check(la_cpa)

        with warnings.catch_warnings(record=True) as w:
            mgo_cpa = chemical_potentials.CompetingPhasesAnalyzer("MgO", self.mgo_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert all(
            any(i in str(warning.message) for warning in w)
            for i in [
                "There are mismatching INCAR tags",
                "['Mg']:",
                "[('ENCUT', 585.0, 450.0)]",
                "Where MgO was used as the reference entry calculation.",
            ]
        )
        self._general_cpa_check(mgo_cpa)

    def test_mismatching_potcar_warnings(self):
        """
        Test warnings for mismatching POTCAR settings.

        No warnings for ZrO2 / La_ZrO2 already checked in ``self.test_general_cpa_reloading()`` above.
        """
        # convert to mismatching O2 calc, with fake "O_h" POTCAR:
        shutil.move(
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz",
        )
        shutil.move(
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/mismatching_potcar_vr.xml.gz",
            f"{self.zro2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.zro2_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert all(
            any(i in str(warning.message) for warning in w)
            for i in [
                "There are mismatching POTCAR symbols",
                "Where ZrO2 was used as the reference entry calculation.",
                "O2: [[{'titel': 'PAW_PBE O_h 08Apr2002_Fake', 'hash': None, 'summary_stats': {}}], "
                "[{'titel': 'PAW_PBE O 08Apr2002', 'hash': None, 'summary_stats': {}}]]",
            ]
        )
        self._general_cpa_check(cpa)

        # test no warning with check_compatibility=False:
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer(
                "ZrO2", self.zro2_path, check_compatibility=False
            )
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        self._general_cpa_check(cpa)

    def test_bulk_not_found(self):
        """
        Test case where bulk composition is not found in the supplied data.
        """
        with pytest.raises(ValueError) as exc:
            _cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.mgo_path)

        assert (
            "Could not find bulk phase for ZrO2 in the supplied data. Found intrinsic phase diagram "
            "entries for: {'O2'}"
        ) in str(exc.value)

    def test_Sn_in_Cs2AgBiBr6(self):
        r"""
        Test parsing competing phases calculations for Sn:Cs2AgBiBr6, where we
        have mismatching ``INCAR`` settings, mismatching ``POTCAR``\s, an
        incomplete ``vasprun.xml.gz`` and an unstable host (so a good test case
        for many warnings/issues to be handled).
        """
        shutil.copyfile(
            f"{data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0/vasprun.xml.gz",
            f"{data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0/duplicate_for_testing_vasprun.xml.gz",
        )
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer(
                "Cs2AgBiBr6", f"{data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases"
            )
        print([str(warning.message) for warning in w])  # for debugging
        for expected_warning in [
            f"Multiple `vasprun.xml` files found in directory: "
            f"{data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0",
            f"vasprun.xml file at {data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Bi_EaH=0/vasprun.xml.gz"
            f" is corrupted/incomplete. Attempting to continue parsing but may fail!",
            "There are mismatching INCAR tags for (some of) your competing phases calculations which are "
            "likely to cause errors in the parsed results (energies & thus chemical potential limits). "
            "Found the following differences:\n"
            "(in the format: 'Entries: (INCAR tag, value in entry calculation, value in reference "
            "calculation))':\n",
            "['Ag', 'AgBr', 'Bi', 'Br', 'Cs', 'Cs2AgBr3', 'Cs3Bi2Br9', 'CsAgBr3', 'Sn']:\n[('ADDGRID', "
            "True, False), ('HFSCREEN', 0.2, 0.207), ('LASPH', True, False)]",
            "Where Cs2AgBiBr6 was used as the reference entry calculation.",
            "In general, the same INCAR settings should be used in all final calculations for these tags "
            "which can affect energies!",
            "There are mismatching POTCAR symbols for (some of) your competing phases calculations which "
            "are likely to cause errors in the parsed results (energies & thus chemical potential "
            "limits). Found the following differences:",
            "(in the format: (entry POTCARs, reference POTCARs)):",
            "Bi: [[{'titel': 'PAW_PBE Bi_d 06Sep2000', 'hash': None, 'summary_stats': {}}], "
            "[{'titel': 'PAW_PBE Bi 08Apr2002', 'hash': None, 'summary_stats': {}}]]",
            "Cs3Bi2Br9: [[{'titel': 'PAW_PBE Bi_d 06Sep2000', 'hash': None, 'summary_stats': {}}], "
            "[{'titel': 'PAW_PBE Bi 08Apr2002', 'hash': None, 'summary_stats': {}}]]",
            "Where Cs2AgBiBr6 was used as the reference entry calculation.",
            "In general, the same POTCAR settings should be used in all final calculations for these tags "
            "which can affect energies!",
            "Cs2AgBiBr6 is not stable with respect to competing phases, having an energy above hull of "
            "0.0171 eV/atom.",
            "Formally, this means that (based on the supplied athermal calculation data) the host "
            "material is unstable and so has no chemical potential limits; though in reality the host may "
            "be stabilised by temperature effects etc, or just a metastable phase.",
            "Here we will determine a single chemical potential 'limit' corresponding to the least "
            "unstable (i.e. closest) point on the convex hull for the host material, as an approximation "
            "for the true chemical potentials.",
        ]:
            print(expected_warning)
            assert any(expected_warning in str(warning.message) for warning in w)
        self._general_cpa_check(cpa)

        assert cpa.chempots["elemental_refs"] == {
            "Cs": -0.9413,
            "Ag": -2.84693,
            "Sn": -4.54148,
            "Bi": -4.5954,
            "Br": -2.28653,
        }


def _check_form_e_df(
    cpa, form_e_df, skip_rounding=False, include_dft_energies=False, prune_polymorphs=False
):
    if not prune_polymorphs:
        assert len(form_e_df) == len(cpa.entries)  # all entries
    else:
        assert len(set(form_e_df.index.to_numpy())) == len(form_e_df)  # no duplicates

    assert set(form_e_df.index.to_numpy()) == {entry.name for entry in cpa.entries}
    assert np.allclose(form_e_df["Energy above Hull (eV/atom)"].to_numpy()[0], 0)  # at least one stable

    for formula, series in form_e_df.iterrows():
        comp = Composition(formula)
        assert np.isclose(
            series["Formation Energy (eV/fu)"],
            series["Formation Energy (eV/atom)"] * comp.num_atoms,
            atol=2e-3,
            rtol=1e-3,
        )
        if include_dft_energies:
            assert np.isclose(
                series["DFT Energy (eV/fu)"],
                series["DFT Energy (eV/atom)"] * comp.num_atoms,
                atol=2e-3,
                rtol=1e-3,
            )

    assert ("DFT Energy (eV/fu)" in form_e_df.columns) == include_dft_energies
    assert ("DFT Energy (eV/atom)" in form_e_df.columns) == include_dft_energies

    # assert values are all rounded to 3 dp:
    assert form_e_df.round(3).equals(form_e_df) == (not skip_rounding)


class TestChemicalPotentialGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chempots = loadfn(os.path.join(EXAMPLE_DIR, "Cu2SiSe3/Cu2SiSe3_chempots.json"))
        cls.grid = chemical_potentials.ChemicalPotentialGrid(cls.chempots)
        cls.api_key = "UsPX9Hwut4drZQXPTxk4CwlCstrAAjDv"
        cls.na2fepo4f_cp = chemical_potentials.CompetingPhases("Na2FePO4F", api_key=cls.api_key)
        na2fepo4f_doped_chempots = chemical_potentials.get_doped_chempots_from_entries(
            cls.na2fepo4f_cp.entries, "Na2FePO4F"
        )
        cls.na2fepo4f_grid = chemical_potentials.ChemicalPotentialGrid(
            na2fepo4f_doped_chempots,
        )
        cls.cpa_folder = os.path.join(data_dir, "ChemPotAnalyzers")
        cls.AgSbTe2_cpa = loadfn(os.path.join(cls.cpa_folder, "AgSbTe2_partial_cpa.json"))
        cls.LiPS4_cpa = loadfn(os.path.join(cls.cpa_folder, "LiPS4_cpa.json"))
        cls.Sn_in_Cs2AgBiBr6_ncl_cpa = loadfn(
            os.path.join(cls.cpa_folder, "Sn_in_Cs2AgBiBr6_ncl_cpa.json")
        )
        cls.Sn_in_Cs2AgBiBr6_std_cpa = loadfn(
            os.path.join(cls.cpa_folder, "Sn_in_Cs2AgBiBr6_std_cpa.json")
        )
        cls.zro2_path = os.path.join(EXAMPLE_DIR, "ZrO2_CompetingPhases")
        cls.zro2_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", cls.zro2_path)

    def tearDown(self):
        if_present_rm("test.png")

    def test_init(self):
        assert isinstance(self.grid.vertices, pd.DataFrame)
        assert len(self.grid.vertices) == 7
        assert np.isclose(max(self.grid.vertices["μ_Cu (eV)"]), 0.0)
        assert np.isclose(max(self.grid.vertices["μ_Si (eV)"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(self.grid.vertices["μ_Se (eV)"]), 0.0)
        assert np.isclose(min(self.grid.vertices["μ_Cu (eV)"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["μ_Si (eV)"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["μ_Se (eV)"]), -0.758105, rtol=1e-5)
        assert np.isclose(np.mean(self.grid.vertices["μ_Cu (eV)"]), -0.1917, rtol=1e-2)
        assert np.isclose(np.mean(self.grid.vertices["μ_Si (eV)"]), -1.0277, rtol=1e-2)
        assert np.isclose(np.mean(self.grid.vertices["μ_Se (eV)"]), -0.37004, rtol=1e-2)

    def test_get_grid(self):
        for cart in [True, False]:
            print(f"Testing grid with cartesian={cart}")
            grid_df = self.grid.get_grid(3800, cartesian=cart)
            assert isinstance(grid_df, pd.DataFrame)
            assert np.isclose(max(grid_df["μ_Cu (eV)"]), 0.0)
            assert np.isclose(max(grid_df["μ_Si (eV)"]), -0.0759, atol=1e-2)
            assert np.isclose(max(grid_df["μ_Se (eV)"]), 0.0)
            assert np.isclose(min(grid_df["μ_Cu (eV)"]), -0.463558, atol=1e-2)
            assert np.isclose(min(grid_df["μ_Si (eV)"]), -1.708951, atol=1e-2)
            assert np.isclose(min(grid_df["μ_Se (eV)"]), -0.758105, atol=1e-2)
            assert np.isclose(np.mean(grid_df["μ_Cu (eV)"]), -0.1966, atol=1e-3 if cart else 2e-2)
            assert np.isclose(np.mean(grid_df["μ_Si (eV)"]), -0.94906, atol=1e-3 if cart else 2e-1)
            assert np.isclose(np.mean(grid_df["μ_Se (eV)"]), -0.39294, atol=1e-3 if cart else 7e-2)

            assert len(grid_df) == (3792 if cart else 3744)

    def test_chempot_heatmap_3D_w_fixed_elements_error(self):
        with pytest.raises(ValueError) as exc:
            self.LiPS4_cpa.plot_chempot_heatmap(fixed_elements={"Li": -0.5})
        assert (
            "Chemical potential heatmap plotting requires 3-D data, requiring fixed chemical potential "
            "constraints for >ternary systems; such that the number of elements in the chemical system "
            "(3) minus the number of fixed chemical potentials (1) must be equal to 3." in str(exc.value)
        )

    def test_chempot_heatmap_3D_w_fixed_elements_error_wrong_element(self):
        with pytest.raises(ValueError) as exc:
            self.LiPS4_cpa.plot_chempot_heatmap(fixed_elements={"Cd": -0.5})
        assert "Chemical potential heatmap plotting requires 3-D data" in str(exc.value)
        assert "(3) minus the number of fixed chemical potentials (1)" in str(exc.value)

    def test_chempot_heatmap_4D_w_fixed_elements_error_wrong_element(self):
        with pytest.raises(ValueError) as exc:
            self.Sn_in_Cs2AgBiBr6_ncl_cpa.plot_chempot_heatmap(fixed_elements={"Cd": -0.5})
        assert "Cd (eV)' is not in list" in str(exc.value)

    def test_chempot_heatmap_4D_w_fixed_elements_error(self):
        with pytest.raises(ValueError) as exc:
            self.Sn_in_Cs2AgBiBr6_ncl_cpa.plot_chempot_heatmap(fixed_elements={"Cs": -0.5, "Ag": -0.5})
        assert "Chemical potential heatmap plotting requires 3-D data" in str(exc.value)
        assert "(4) minus the number of fixed chemical potentials (2)" in str(exc.value)

    def test_chempot_heatmap_2D_error(self):
        with pytest.raises(ValueError) as exc:  # this will likely change with updated code
            self.zro2_cpa.plot_chempot_heatmap()
        assert (
            "Chemical potential heatmap (i.e. 2D) plotting is not possible for a binary system! You "
            "can use ``cpd = ChemicalPotentialDiagram(cpa.entries); cpd.get_plot()`` to generate a "
            "line plot of the chemical potentials as shown in the doped competing phases tutorial."
            in str(exc.value)
        )

    def test_chempot_heatmap_4D_w_fixed_elements_outside_range(self):
        with pytest.raises(ValueError) as exc:
            self.Sn_in_Cs2AgBiBr6_ncl_cpa.plot_chempot_heatmap(fixed_elements={"Ag": -25})
        assert (
            "The input set of fixed chemical potentials does not intersect with the convex hull (i.e. "
            "stable chemical potential range) of the host material." in str(exc.value)
        )

    def plot_and_test_no_warnings(self, cpa, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            plot = cpa.plot_chempot_heatmap(**kwargs)
        print([str(warning.message) for warning in w])
        assert not w
        return plot

    @custom_mpl_image_compare(filename="AgSbTe2_chempot_heatmap_default.png")
    def test_AgSbTe2_chempot_heatmap_default(self):
        return self.plot_and_test_no_warnings(self.AgSbTe2_cpa)

    @custom_mpl_image_compare(
        filename="AgSbTe2_chempot_heatmap_custom.png",
        style=f"{module_path}/../doped/utils/displacement.mplstyle",
    )
    def test_AgSbTe2_chempot_heatmap_custom(self):
        plot = self.plot_and_test_no_warnings(
            self.AgSbTe2_cpa,
            dependent_element="Ag",
            xlim=(-0.5, 0.0),
            ylim=(-0.4, 0.0),
            cbar_range=(-0.4, 0.0),
            colormap="viridis",
            padding=0.05,
            title=True,
            label_positions=False,
            filename="test.png",
            style_file=f"{module_path}/../doped/utils/displacement.mplstyle",
        )
        assert os.path.exists("test.png")
        return plot

    @custom_mpl_image_compare(filename="LiPS4_chempot_heatmap_default.png")
    def test_LiPS4_chempot_heatmap_default(self):
        return self.plot_and_test_no_warnings(self.LiPS4_cpa)

    @custom_mpl_image_compare(filename="LiPS4_chempot_heatmap_custom.png")
    def test_LiPS4_chempot_heatmap_custom(self):
        return self.plot_and_test_no_warnings(
            self.LiPS4_cpa,
            dependent_element="Li",
            padding=0.1,
            title=False,
            label_positions=True,
        )

    @custom_mpl_image_compare(filename="Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_default.png")
    def test_Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_default(self):
        return self.plot_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_ncl_cpa, fixed_elements={"Cs": -3.3815}
        )

    @custom_mpl_image_compare(
        filename="Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_custom.png",
        style=f"{module_path}/../doped/utils/displacement.mplstyle",
    )
    def test_Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_custom(self):
        """
        Test customising the heatmap for a 4-D system, with custom label
        positions.

        Same example used in the plotting customisation tutorial.
        """
        return self.plot_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_ncl_cpa,
            fixed_elements={"Cs": -3.3815},
            dependent_element="Bi",  # change dependent (colourbar) element
            xlim=(-0.4, 0.0),
            ylim=(-0.6, -0.2),
            cbar_range=(-2, -1),
            colormap="navia",
            padding=0.05,
            title=True,
            label_positions={
                "CsAgBr3": (-0.3, 0.025),
                "AgBr": (-0.16, 0.0),
                "Cs3Bi2Br9": (-0.1, -0.05),
            },  # custom label positions
            style_file=f"{module_path}/../doped/utils/displacement.mplstyle",
        )

    @custom_mpl_image_compare(
        filename="Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_auto_centroid.png",
        style=f"{module_path}/../doped/utils/displacement.mplstyle",
    )
    def test_Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_auto_centroid(self):
        """
        Test customising the heatmap for a 4-D system, with custom label
        positions.

        Same example used in the plotting customisation tutorial.
        """
        return self.plot_and_test_no_warnings(self.Sn_in_Cs2AgBiBr6_ncl_cpa, dependent_element="Cs")

    @custom_mpl_image_compare(filename="Sn_in_Cs2AgBiBr6_std_chempot_heatmap_custom.png")
    def test_Sn_in_Cs2AgBiBr6_std_chempot_heatmap_custom(self):
        return self.plot_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_std_cpa,
            fixed_elements={"Cs": -3.3815},
            xlim=(-0.45, 0),
            ylim=(-2.35, -0.9),
            cbar_range=(-0.57, -0.3),
            label_positions=False,
        )

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_heatmap.png")
    def test_5D_fixed_elements_heatmap(self):
        na2fepo4f_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "Na2FePO4F", entries=self.na2fepo4f_cp.entries
        )
        return self.plot_and_test_no_warnings(na2fepo4f_cpa, fixed_elements={"Na": -1.9, "P": -1.3})

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_grid.png")
    def test_Na2FePO4F_chempot_grid(self):
        """
        Test ``ChemicalPotentialGrid`` generation and plotting for a complex
        quinary system (Na2FePO4F).
        """
        grid_df = self.na2fepo4f_grid.get_grid(1e8, drop_duplicates=False)
        return _plot_Na2FePO4F_chempot_grid(grid_df, atol=0.01)

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_grid_cartesian.png")
    def test_Na2FePO4F_chempot_grid_cartesian(self):
        """
        Test ``ChemicalPotentialGrid`` generation and plotting for a complex
        quinary system (Na2FePO4F).
        """
        grid_df = self.na2fepo4f_grid.get_grid(2e5, cartesian=True)
        return _plot_Na2FePO4F_chempot_grid(grid_df)


def _plot_Na2FePO4F_chempot_grid(grid_df, atol=0.05):
    # get the average Fe and P chempots, then plot a heatmap plot of the others at these fixed values:
    mean_mu_Fe = grid_df["μ_Fe (eV)"].mean()
    mean_mu_P = grid_df["μ_P (eV)"].mean()

    fixed_chempot_df = grid_df[
        (np.isclose(grid_df["μ_Fe (eV)"], mean_mu_Fe, atol=atol))
        & (np.isclose(grid_df["μ_P (eV)"], mean_mu_P, atol=atol))
    ]

    fig, ax = plt.subplots()
    sc = ax.scatter(
        fixed_chempot_df["μ_Na (eV)"],
        fixed_chempot_df["μ_O (eV)"],
        c=fixed_chempot_df["μ_F (eV)"],
        cmap="viridis",
    )
    fig.colorbar(sc, ax=ax, label="μ$_F$ (eV)")
    ax.set_xlabel("μ$_{Na}$ (eV)")
    ax.set_ylabel("μ$_{O}$ (eV)")
    return fig


# TODO: Use this as an advanced plotting example in advanced tutorial, linking in the
#  chempots/Fermisolver tutorials, and add heatmap plotting examples to the chemical potentials
#  tutorial/Fermisolver tutorial.
# TODO: Use Cs2SnBr6 competing phase energies csv in JOSS data folder and LiPS4 data for test cases with
#  chempot plotting etc
