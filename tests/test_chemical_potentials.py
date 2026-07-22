"""
Tests for the ``doped.chemical_potentials`` module.
"""

import glob
import inspect
import os
import shutil
import tempfile
import unittest
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.entries import ComputedEntry
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Potcar
from test_utils import (
    EXAMPLE_DIR,
    _potcars_available,
    _print_warning_info,
    _run_func_and_capture_stdout_warnings,
    _run_heavy_tests,
    api_key,
    custom_mpl_image_compare,
    data_dir,
    if_present_rm,
    module_path,
    plot_chempot_heatmap_and_test_no_warnings,
    vasp_data_dir,
)

from doped import chemical_potentials
from doped.io.vasp.outputs import _find_calc_outputs, _get_calc_files_df
from doped.utils.symmetry import get_primitive_structure


def _compare_chempot_dicts(dict1, dict2):
    for key, val in dict1.items():
        if isinstance(val, dict):
            _compare_chempot_dicts(val, dict2[key])
        else:
            assert np.isclose(val, dict2[key], atol=1e-5)


def _canonicalise_chempot_dict(chempots):
    """
    Return a copy of a chempot dict (e.g. ``cpa.intrinsic_chempots``) with
    limit keys canonicalised (bordering phases sorted alphabetically), so that
    two dicts from different parses can be compared even when the underlying
    ``PhaseDiagram`` enumerates phases at each facet in a different order.
    """
    out = {}
    for outer_k, outer_v in chempots.items():
        if isinstance(outer_v, dict):
            out[outer_k] = {
                ("-".join(sorted(k.split("-"))) if isinstance(k, str) and "-" in k else k): v
                for k, v in outer_v.items()
            }
        else:
            out[outer_k] = outer_v
    return out


def _check_entries_dict_behaviour(obj):
    first_key = obj.entries[0].data["doped_name"]
    second_key = obj.entries[1].data["doped_name"]
    assert obj[first_key] is obj.entries[0]
    assert obj[1] is obj.entries[1]
    assert obj[:1] == [obj.entries[0]]
    assert len(obj) == len(obj.entries)
    assert first_key in obj
    assert obj.entries[1] in obj
    assert list(obj) == [entry.data["doped_name"] for entry in obj.entries]
    assert obj.get(second_key) is obj.entries[1]
    assert obj.get("Missing_Key") is None
    assert obj.get("Missing_Key", obj.entries[0]) is obj.entries[0]
    assert list(obj.keys()) == [entry.data["doped_name"] for entry in obj.entries]
    assert list(obj.values()) == obj.entries
    assert list(obj.items()) == [(entry.data["doped_name"], entry) for entry in obj.entries]

    with pytest.raises(KeyError):
        _ = obj["Missing_Key"]


class CompetingPhasesTestCase(unittest.TestCase):
    def setUp(self):
        self.cdte = Structure.from_file(os.path.join(EXAMPLE_DIR, "CdTe/relaxed_primitive_POSCAR"))
        self.na2fepo4f = Structure.from_file(os.path.join(data_dir, "Na2FePO4F_MP_POSCAR"))
        self.cu2sise3 = Structure.from_file(os.path.join(data_dir, "Cu2SiSe3_MP_POSCAR"))
        self.cu2sise4 = self.cu2sise3.get_primitive_structure().copy()
        self.cu2sise4.append("Se", [0.5, 0.5, 0.5])
        self.cu2sise4.append("Se", [0.5, 0.75, 0.5])

        self.ZrO2_entry_list = [  # without full_phase_diagram
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
        if_present_rm("CustomOutputDir")
        if_present_rm("cp.json")

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

        _check_entries_dict_behaviour(cp)  # test dict behaviour

    def _compare_cps(self, cp_a, cp_b):
        def cleanse_entries(entries):
            for entry in entries:
                entry.entry_id = None
            return entries

        for attr in cp_a.__dict__:
            val_a = getattr(cp_a, attr)
            val_b = getattr(cp_b, attr)

            if attr in {"entries", "intrinsic_entries", "extrinsic_entries", "MP_full_pd_entries"}:
                assert cleanse_entries(val_a) == cleanse_entries(val_b)
            elif hasattr(val_a, "as_dict") and hasattr(val_b, "as_dict"):
                assert val_a.as_dict() == val_b.as_dict()
            else:
                assert val_a == val_b

    def _check_cp_json_roundtrip(self, cp):
        cp_dict = cp.as_dict()
        cp_from_dict = chemical_potentials.CompetingPhases.from_dict(cp_dict)
        self._compare_cps(cp, cp_from_dict)

        dumpfn(cp_dict, "cp.json")
        reloaded_cp = loadfn("cp.json")
        self._compare_cps(cp, reloaded_cp)

    def test_make_molecule_in_a_box(self):
        allowed_gaseous_elements = ["O2", "N2", "H2", "F2", "Cl2"]
        for element in allowed_gaseous_elements:
            structure = chemical_potentials.make_molecule_in_a_box(element)
            assert structure.composition.reduced_formula == element
            assert structure.num_sites == 2
            assert np.isclose(structure.volume, 30**3)

        # Triplet O2 vs closed-shell X2 magnetization is stored on the molecular
        # ``ComputedStructureEntry`` (used by ``_set_spin_polarisation``), not on
        # the bare ``Structure`` from ``make_molecule_in_a_box``:
        o2_mol = chemical_potentials.make_molecular_entry(
            ComputedEntry(
                "O",
                -1.0,
                data={"energy_per_atom": -1.0, "formula_pretty": "O2"},
            )
        )
        assert o2_mol.data["summary"]["total_magnetization"] == 2
        h2_mol = chemical_potentials.make_molecular_entry(
            ComputedEntry(
                "H",
                -0.5,
                data={"energy_per_atom": -0.5, "formula_pretty": "H2"},
            )
        )
        assert h2_mol.data["summary"]["total_magnetization"] == 0

        # elements without tabulated diatomic bond lengths now fall back to ``ShakeNBreak``'s
        # ``get_dimer_bond_length`` (rather than raising ``ValueError``):
        te_mol = chemical_potentials.make_molecule_in_a_box("Te")
        assert te_mol.num_sites == 2
        assert np.isclose(te_mol.volume, 30**3)
        assert 2.0 < te_mol.get_distance(0, 1) < 3.2  # reasonable Te dimer bond length

    def test_init(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)

        assert len(cp.entries) == 13
        assert [entry.name for entry in cp.entries] == self.ZrO2_entry_list
        self._check_ZrO2_cp_init(cp)
        assert "Zr4O" not in [e.name for e in cp.entries]  # not bordering or potentially with EaH
        assert not cp.MP_doc_dicts
        self._check_cp_json_roundtrip(cp)

    def test_init_full_phase_diagram(self):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", energy_above_hull=0.03, api_key=api_key, full_phase_diagram=True
        )

        assert len(cp.entries) == 14  # Zr4O now present
        ZrO2_full_pd_entry_list = [*self.ZrO2_entry_list[:4], "Zr4O", *self.ZrO2_entry_list[4:]]
        assert [entry.name for entry in cp.entries] == ZrO2_full_pd_entry_list
        self._check_ZrO2_cp_init(cp, num_stable_entries=5)  # Zr4O is on hull
        self._check_cp_json_roundtrip(cp)

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
        cp = chemical_potentials.CompetingPhases("ZnSe", api_key=api_key)
        assert any(e.name == "ZnSe2" for e in cp.entries)
        assert len(cp.entries) == 12  # ZnSe2 present; 2 new Zn entries (mp-264...) with new MP API
        znse2_entry = next(e for e in cp.entries if e.name == "ZnSe2")
        assert znse2_entry.data.get("energy_above_hull") == 0
        assert not znse2_entry.data["molecule"]
        assert np.isclose(znse2_entry.energy_per_atom, -3.394683861)
        assert np.isclose(znse2_entry.energy, -3.394683861 * 12)
        self._check_cp_json_roundtrip(cp)

    def test_init_YTOS(self):
        # 144 phases on Y-Ti-O-S MP phase diagram
        cp = chemical_potentials.CompetingPhases("Y2Ti2S2O5", energy_above_hull=0.1, api_key=api_key)
        assert len(cp.entries) == 113
        self.check_O2_entry(cp)

        cp = chemical_potentials.CompetingPhases(
            "Y2Ti2S2O5", energy_above_hull=0.1, full_phase_diagram=True, api_key=api_key
        )
        # 149 phases on Y-Ti-O-S MP full phase diagram, 4 extra O2 phases removed
        assert len(cp.entries) == 145
        self.check_O2_entry(cp)
        self._check_cp_json_roundtrip(cp)

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
        cdte_cp = chemical_potentials.CompetingPhases("CdTe", api_key=api_key)
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
        cds_cp = chemical_potentials.CompetingPhases("CdS", api_key=api_key)
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
            {"composition": "Na2FePO4F", "energy_above_hull": 0.02, "api_key": api_key},
            {
                "composition": "Na2FePO4F",
                "energy_above_hull": 0.02,
                "api_key": api_key,
                "full_phase_diagram": True,
            },
        ]:
            print(f"Testing with settings: {cp_settings}")
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(**cp_settings)
                cp.write_kpoint_convergence_files(potcar_spec=True)
                cp.write_relaxation_files(potcar_spec=True)
                cp.write_singlepoint_files(soc=False, potcar_spec=True)
            _print_warning_info(w)  # for debugging
            if cp_settings.get("full_phase_diagram"):
                assert len(cp.entries) == 172
            else:
                assert len(cp.entries) == 68
            self.check_O2_entry(cp)
            self._check_cp_json_roundtrip(cp)

    def test_unknown_host(self):
        """
        Test generating CompetingPhases with a composition that's not on the
        Materials Project database.
        """
        unknown_host_cp_kwargs = {"composition": "Cu2SiSe4", "api_key": api_key}
        for cp_settings in [
            {},
            {"energy_above_hull": 0.0},
            {"full_phase_diagram": True},
        ]:
            kwargs = {**unknown_host_cp_kwargs, **cp_settings}
            print(f"Testing with settings: {kwargs}")
            potcar_spec = not _potcars_available()
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(**kwargs)
                cp.write_kpoint_convergence_files(potcar_spec=potcar_spec)
                cp.write_relaxation_files(potcar_spec=potcar_spec)
                cp.write_singlepoint_files(soc=False, potcar_spec=potcar_spec)
            _print_warning_info(w)  # for debugging
            user_warnings = [x for x in w if x.category is UserWarning]
            assert "Note that no Materials Project (MP) database entry exists for Cu2SiSe4. Here" in str(
                user_warnings[0].message
            )
            no_structure_warnings = [
                uw for uw in user_warnings if "no structure is available" in str(uw.message).lower()
            ]
            assert len(no_structure_warnings) == 3  # one per write method
            for uw in no_structure_warnings:
                msg = str(uw.message)
                assert "placeholder" in msg.lower()
                assert "incar" in msg.lower()
                assert "potcar" in msg.lower()
                assert "non-metallic" in msg.lower()
                assert "non-magnetic" in msg.lower()
            bulk_ph = [e for e in cp.entries if e.name == "Cu2SiSe4" and not hasattr(e, "structure")]
            assert len(bulk_ph) == 1
            assert os.path.isdir("CompetingPhases")
            cu2sise4_folder = "CompetingPhases/Cu2SiSe4_NA_EaH_0"
            assert os.path.isdir(cu2sise4_folder)

            def _check_potcar(directory, potcar_spec=potcar_spec):
                """
                Check POTCAR/POTCAR.spec in ``directory`` for Cu, Si, Se.
                """
                if potcar_spec:
                    with open(os.path.join(directory, "POTCAR.spec"), encoding="utf-8") as f:
                        pot_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                    assert pot_lines == ["Cu", "Si", "Se"]
                else:
                    written_potcar = Potcar.from_file(os.path.join(directory, "POTCAR"))
                    for sym in ("Cu", "Si", "Se"):
                        assert any(
                            potcar_symbol.startswith(sym) for potcar_symbol in written_potcar.symbols
                        )

            # check Relax inputs
            relax_dir = f"{cu2sise4_folder}/Relax"
            assert os.path.isfile(f"{relax_dir}/INCAR")
            if potcar_spec:
                assert not os.path.isfile(f"{relax_dir}/POTCAR")
            with open(f"{relax_dir}/INCAR", encoding="utf-8") as f:
                incar_std_lines = f.readlines()
            assert any(line == "GGA = Pe\n" for line in incar_std_lines)
            assert any(line == "ISIF = 3\n" for line in incar_std_lines)
            assert any(line.strip().startswith("AEXX = 0.25") for line in incar_std_lines)
            _check_potcar(relax_dir)
            assert not os.path.exists(f"{relax_dir}/POSCAR")  # no structure available
            assert not os.path.exists(f"{relax_dir}/KPOINTS")  # no structure available

            # check kpoint_converge inputs
            kpt_incars = sorted(glob.glob(f"{cu2sise4_folder}/kpoint_converge/k*/INCAR"))
            assert kpt_incars
            kpt_dir = os.path.dirname(kpt_incars[0])
            with open(kpt_incars[0], encoding="utf-8") as f:
                incar_k_lines = f.readlines()
            assert any(line == "GGA = Ps\n" for line in incar_k_lines)
            assert any(line == "NSW = 0\n" for line in incar_k_lines)
            assert any(line == "ISMEAR = 0\n" for line in incar_k_lines)
            _check_potcar(kpt_dir)

            # check SinglePoint inputs for unknown-host placeholder
            sp_dir = f"{cu2sise4_folder}/SinglePoint"
            assert os.path.isfile(f"{sp_dir}/INCAR")
            with open(f"{sp_dir}/INCAR", encoding="utf-8") as f:
                incar_sp_lines = f.readlines()
            assert any(line == "NSW = 0\n" for line in incar_sp_lines)
            assert any(line.strip().startswith("AEXX = 0.25") for line in incar_sp_lines)
            assert not any("ISIF" in line for line in incar_sp_lines)
            _check_potcar(sp_dir)
            assert not os.path.exists(f"{sp_dir}/POSCAR")  # no structure
            assert not os.path.exists(f"{sp_dir}/KPOINTS")  # no structure

            assert len(os.listdir("CompetingPhases")) > 0  # other phases still get inputs

            # check all other written phase folders (excluding unknown-host placeholder) have full inputs
            other_phase_folders = [
                os.path.join("CompetingPhases", folder)
                for folder in os.listdir("CompetingPhases")
                if os.path.join("CompetingPhases", folder) != cu2sise4_folder
                and os.path.isdir(os.path.join("CompetingPhases", folder))
            ]
            assert other_phase_folders
            for phase_folder in other_phase_folders:
                # Relax inputs
                phase_relax = os.path.join(phase_folder, "Relax")
                assert os.path.isdir(phase_relax)
                assert os.path.isfile(os.path.join(phase_relax, "INCAR"))
                assert os.path.isfile(os.path.join(phase_relax, "KPOINTS"))
                assert os.path.isfile(os.path.join(phase_relax, "POSCAR"))

                # SinglePoint inputs: POSCARs are not written by default:
                phase_sp = os.path.join(phase_folder, "SinglePoint")
                assert os.path.isdir(phase_sp)
                assert os.path.isfile(os.path.join(phase_sp, "INCAR"))
                assert os.path.isfile(os.path.join(phase_sp, "KPOINTS"))
                assert not os.path.isfile(os.path.join(phase_sp, "POSCAR"))

                # kpoint_converge inputs
                phase_k_dirs = sorted(glob.glob(f"{phase_folder}/kpoint_converge/k*"))
                assert phase_k_dirs
                for k_dir in phase_k_dirs:
                    assert os.path.isfile(os.path.join(k_dir, "INCAR"))
                    assert os.path.isfile(os.path.join(k_dir, "KPOINTS"))
                    assert os.path.isfile(os.path.join(k_dir, "POSCAR"))

            if kwargs.get("full_phase_diagram"):
                assert len(cp.entries) == 29
            elif kwargs.get("energy_above_hull") == 0.0:
                assert len(cp.entries) == 8
            else:
                assert len(cp.entries) == 26

            # check naming of fake entry
            assert "Cu2SiSe4_NA_EaH_0" in [entry.data["doped_name"] for entry in cp.entries]
            shutil.rmtree("CompetingPhases")  # clean up for next iteration of test

    def test_write_kpoint_convergence_files(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        # potcar spec doesn't need potcars set up for pmg and it still works
        if_present_rm("CompetingPhases")
        dict_sets_no_write = cp.get_kpoint_convergence_sets()
        assert dict_sets_no_write
        assert not os.path.exists("CompetingPhases")
        no_write_key = "CompetingPhases/ZrO2_Pbca_EaH_0.009/kpoint_converge/k2,1,1"
        assert no_write_key in dict_sets_no_write
        no_write_dict_set = dict_sets_no_write[no_write_key]
        assert no_write_dict_set.kpoints.kpts[0] == (2, 1, 1)
        assert no_write_dict_set.potcar_symbols[0] == "Zr_sv"
        assert no_write_dict_set.incar["GGA"] == "Ps"
        assert no_write_dict_set.incar["NSW"] == 0

        dict_sets = cp.write_kpoint_convergence_files(potcar_spec=True)
        assert dict_sets
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in dict_sets.values())
        assert len(cp.metallic_entries) == 6
        assert cp.metallic_entries[0].data["summary"]["band_gap"] == 0
        assert not cp.nonmetallic_entries[0].data["molecule"]
        # this shouldn't exist - don't need to convergence test for molecules
        assert not os.path.exists("CompetingPhases/O2_Pmmm_EaH_0")

        # test if it writes out the files correctly
        Zro2_EaH_0pt009_folder = "CompetingPhases/ZrO2_Pbca_EaH_0.009/kpoint_converge/k2,1,1/"
        assert os.path.exists(Zro2_EaH_0pt009_folder)
        assert "CompetingPhases/ZrO2_Pbca_EaH_0.009/kpoint_converge/k2,1,1" in dict_sets
        dict_set = dict_sets["CompetingPhases/ZrO2_Pbca_EaH_0.009/kpoint_converge/k2,1,1"]
        assert dict_set.kpoints.kpts[0] == (2, 1, 1)
        assert dict_set.potcar_symbols[0] == "Zr_sv"
        assert dict_set.incar["GGA"] == "Ps"
        assert dict_set.incar["NSW"] == 0
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

        # existing folders should warn and be overwritten with new settings (one ``UserWarning`` per
        # existing folder, plus possible KPAR ``UserWarning``s for Γ-only molecular phases)
        _result, _stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_kpoint_convergence_files,
            potcar_spec=True,
            user_incar_settings={"NSW": 7, "GGA": "Ps"},
        )
        overwrite_w = [ww for ww in w if "already exists. Overwriting files." in str(ww.message)]
        other_w = [ww for ww in w if ww not in overwrite_w]
        assert len(overwrite_w) > 0
        assert all(issubclass(ww.category, UserWarning) for ww in overwrite_w)
        assert all(
            issubclass(ww.category, UserWarning) and "KPOINTS are Γ-only" in str(ww.message)
            for ww in other_w
        )
        with open(f"{Zro2_EaH_0pt009_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert any(line == "NSW = 7\n" for line in contents)
            assert not any(line == "NSW = 0\n" for line in contents)

    def test_write_relaxation_files(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        if_present_rm("CompetingPhases")
        dict_sets_no_write = cp.get_relaxation_sets()
        assert len(dict_sets_no_write) == len(cp)  # one per entry
        assert not os.path.exists("CompetingPhases")
        no_write_key = "CompetingPhases/ZrO2_P2_1c_EaH_0/Relax"
        assert no_write_key in dict_sets_no_write
        no_write_dict_set = dict_sets_no_write[no_write_key]
        assert no_write_dict_set.kpoints.kpts[0] == (4, 4, 4)
        assert no_write_dict_set.potcar_symbols == ["Zr_sv", "O"]
        assert no_write_dict_set.incar["AEXX"] == 0.25
        assert no_write_dict_set.incar["ISIF"] == 3
        assert no_write_dict_set.incar["GGA"] == "Pe"

        dict_sets = cp.write_relaxation_files(potcar_spec=True)
        assert len(dict_sets) == len(cp)  # one per entry
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in dict_sets.values())
        assert len(cp.nonmetallic_entries) == 6
        assert len(cp.metallic_entries) == 6
        assert len(cp.molecular_entries) == 1
        assert cp.molecular_entries[0].name == "O2"
        assert cp.molecular_entries[0].data["summary"]["total_magnetization"] == 2
        assert cp.molecular_entries[0].data["molecule"]
        assert not cp.nonmetallic_entries[0].data["molecule"]

        ZrO2_EaH_0_std_folder = "CompetingPhases/ZrO2_P2_1c_EaH_0/Relax/"
        assert os.path.exists(ZrO2_EaH_0_std_folder)
        assert "CompetingPhases/ZrO2_P2_1c_EaH_0/Relax" in dict_sets
        dict_set = dict_sets["CompetingPhases/ZrO2_P2_1c_EaH_0/Relax"]
        assert dict_set.kpoints.kpts[0] == (4, 4, 4)
        assert dict_set.potcar_symbols == ["Zr_sv", "O"]
        assert dict_set.incar["AEXX"] == 0.25
        assert dict_set.incar["ISIF"] == 3
        assert dict_set.incar["GGA"] == "Pe"
        with open(f"{ZrO2_EaH_0_std_folder}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert "KPOINTS from doped, with reciprocal_density = 64.0/Å" in contents[0]
            assert contents[3] == "4 4 4\n"

        with open(f"{ZrO2_EaH_0_std_folder}/POTCAR.spec", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents == ["Zr_sv\n", "O"]

        with open(f"{ZrO2_EaH_0_std_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert all(x in contents for x in ["AEXX = 0.25\n", "ISIF = 3\n", "GGA = Pe\n"])

        O2_EaH_0_std_folder = "CompetingPhases/O2_mmm_EaH_0/Relax"
        assert os.path.exists(O2_EaH_0_std_folder)
        o2_dict_set = dict_sets["CompetingPhases/O2_mmm_EaH_0/Relax"]
        assert o2_dict_set.kpoints.kpts[0] == (1, 1, 1)
        with open(f"{O2_EaH_0_std_folder}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[3] == "1 1 1\n"

        struct = Structure.from_file(f"{O2_EaH_0_std_folder}/POSCAR")
        assert np.isclose(struct.sites[0].frac_coords, [0.49983339, 0.5, 0.50016672]).all()
        assert np.isclose(struct.sites[1].frac_coords, [0.49983339, 0.5, 0.5405135]).all()
        assert struct == o2_dict_set.poscar.structure

        # existing folders should warn and be overwritten with new settings (one ``UserWarning`` per
        # existing folder, plus possible KPAR ``UserWarning``s for Γ-only molecular phases)
        _result, _stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_relaxation_files, potcar_spec=True, user_incar_settings={"ISIF": 2}
        )
        overwrite_w = [ww for ww in w if "already exists. Overwriting files." in str(ww.message)]
        other_w = [ww for ww in w if ww not in overwrite_w]
        assert len(overwrite_w) > 0
        assert all(issubclass(ww.category, UserWarning) for ww in overwrite_w)
        assert all(
            issubclass(ww.category, UserWarning) and "KPOINTS are Γ-only" in str(ww.message)
            for ww in other_w
        )
        with open(f"{ZrO2_EaH_0_std_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert "ISIF = 2\n" in contents
            assert "ISIF = 3\n" not in contents

    def test_custom_output_path(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        custom_dir = "CustomOutputDir"

        conv_sets = cp.write_kpoint_convergence_files(potcar_spec=True, output_path=custom_dir)
        assert conv_sets
        assert not os.path.exists("CompetingPhases")
        assert all(key.startswith(f"{custom_dir}/") for key in conv_sets)
        sample_key = next(iter(conv_sets))
        assert os.path.exists(sample_key)
        assert os.path.isfile(os.path.join(sample_key, "INCAR"))
        if_present_rm(custom_dir)

        std_sets = cp.write_relaxation_files(potcar_spec=True, output_path=custom_dir)
        assert std_sets
        assert not os.path.exists("CompetingPhases")
        assert all(key.startswith(f"{custom_dir}/") for key in std_sets)
        sample_key = next(iter(std_sets))
        assert os.path.exists(sample_key)
        assert os.path.isfile(os.path.join(sample_key, "INCAR"))
        if_present_rm(custom_dir)

        sp_sets = cp.write_singlepoint_files(soc=False, potcar_spec=True, output_path=custom_dir)
        assert sp_sets
        assert not os.path.exists("CompetingPhases")
        assert all(key.startswith(f"{custom_dir}/") for key in sp_sets)
        sample_key = next(iter(sp_sets))
        assert os.path.exists(sample_key)
        assert os.path.isfile(os.path.join(sample_key, "INCAR"))

    @pytest.mark.filterwarnings("always::DeprecationWarning")  # deliberate deprecated-API test
    def test_deprecated_convergence_setup(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        if_present_rm("CompetingPhases")
        dict_sets, _stdout, w = _run_func_and_capture_stdout_warnings(
            cp.convergence_setup, potcar_spec=True
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "convergence_setup" in str(w[0].message)
        assert "deprecated" in str(w[0].message)
        assert dict_sets
        assert os.path.exists("CompetingPhases")

    @pytest.mark.filterwarnings("always::DeprecationWarning")  # deliberate deprecated-API test
    def test_deprecated_vasp_std_setup(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        if_present_rm("CompetingPhases")
        dict_sets, _stdout, w = _run_func_and_capture_stdout_warnings(cp.vasp_std_setup, potcar_spec=True)
        deprecation_w = [ww for ww in w if issubclass(ww.category, DeprecationWarning)]
        other_w = [ww for ww in w if ww not in deprecation_w]
        assert len(deprecation_w) == 1
        assert "vasp_std_setup" in str(deprecation_w[0].message)
        assert "deprecated" in str(deprecation_w[0].message)
        # only other possible warning is the Γ-only KPAR warning for the O2 molecule
        assert all(
            issubclass(ww.category, UserWarning) and "KPOINTS are Γ-only" in str(ww.message)
            for ww in other_w
        )
        assert dict_sets
        assert os.path.exists("CompetingPhases")

    def test_get_kpoint_convergence_sets(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        dict_sets = cp.get_kpoint_convergence_sets()
        assert dict_sets
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in dict_sets.values())
        assert not os.path.exists("CompetingPhases")

    def test_get_relaxation_sets(self):
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        dict_sets = cp.get_relaxation_sets()
        assert dict_sets
        assert len(dict_sets) == len(cp)
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in dict_sets.values())
        assert not os.path.exists("CompetingPhases")

    def test_get_singlepoint_sets(self):
        r"""
        Test ``get_singlepoint_sets`` returns correct ``DopedDictSet``\s
        without writing files.
        """
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        # ZrO2 has Zr (Z=40), so SOC defaults to True
        dict_sets = cp.get_singlepoint_sets()
        assert dict_sets
        assert len(dict_sets) == len(cp)
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in dict_sets.values())
        assert not os.path.exists("CompetingPhases")

        # check SOC defaults on for ZrO2 (Zr Z=40 >= 31) -> vasp_ncl subfolder
        assert all("vasp_ncl" in key for key in dict_sets)
        sample_key = "CompetingPhases/ZrO2_P2_1c_EaH_0/vasp_ncl"
        assert sample_key in dict_sets
        ds = dict_sets[sample_key]
        assert ds.incar["LSORBIT"] is True
        assert ds.incar["NSW"] == 0
        assert "IBRION" not in ds.incar  # removed by pymatgen when NSW=0
        assert "EDIFFG" not in ds.incar  # removed (None value)
        assert "POTIM" not in ds.incar  # removed (None value)
        assert "ISIF" not in ds.incar  # not a relaxation
        assert ds.incar["AEXX"] == 0.25  # HSE06 by default

        # explicitly disable SOC -> SinglePoint subfolder, no LSORBIT
        dict_sets_no_soc = cp.get_singlepoint_sets(soc=False)
        assert all("SinglePoint" in key for key in dict_sets_no_soc)
        assert all("vasp_ncl" not in key for key in dict_sets_no_soc)
        sample_ds = dict_sets_no_soc["CompetingPhases/ZrO2_P2_1c_EaH_0/SinglePoint"]
        assert "LSORBIT" not in sample_ds.incar
        assert sample_ds.incar["NSW"] == 0

        # explicitly enable SOC
        dict_sets_soc = cp.get_singlepoint_sets(soc=True)
        assert all("vasp_ncl" in key for key in dict_sets_soc)
        for ds in dict_sets_soc.values():
            assert ds.incar["LSORBIT"] is True

    def test_write_singlepoint_files(self):
        """
        Test ``write_singlepoint_files`` generates correct input files.
        """
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        if_present_rm("CompetingPhases")

        # ZrO2 defaults to SOC (Zr Z=40), so subfolder is vasp_ncl
        dict_sets = cp.write_singlepoint_files(potcar_spec=True)
        assert len(dict_sets) == len(cp)
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in dict_sets.values())

        ZrO2_ncl_folder = "CompetingPhases/ZrO2_P2_1c_EaH_0/vasp_ncl/"
        assert os.path.exists(ZrO2_ncl_folder)
        dict_set = dict_sets["CompetingPhases/ZrO2_P2_1c_EaH_0/vasp_ncl"]
        assert dict_set.incar["LSORBIT"] is True
        assert dict_set.incar["NSW"] == 0

        with open(f"{ZrO2_ncl_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert any("NSW = 0" in line for line in contents)
            assert any("LSORBIT = True" in line for line in contents)
            assert any("AEXX = 0.25" in line for line in contents)
            assert not any("ISIF" in line for line in contents)  # not a relaxation
            assert not any("EDIFFG" in line for line in contents)  # removed
            assert not any("POTIM" in line for line in contents)  # removed

        with open(f"{ZrO2_ncl_folder}/POTCAR.spec", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents == ["Zr_sv\n", "O"]

        # POSCAR is not written by default (single-point calcs use user-supplied relaxed structures)
        assert not os.path.exists(f"{ZrO2_ncl_folder}/POSCAR")
        # KPOINTS is still written by default
        assert os.path.exists(f"{ZrO2_ncl_folder}/KPOINTS")

        # molecule entry should have KPAR=1 and gamma-only kpoints
        O2_ncl_folder = "CompetingPhases/O2_mmm_EaH_0/vasp_ncl"
        assert os.path.exists(O2_ncl_folder)
        o2_dict_set = dict_sets["CompetingPhases/O2_mmm_EaH_0/vasp_ncl"]
        assert o2_dict_set.kpoints.kpts[0] == (1, 1, 1)
        assert o2_dict_set.incar["KPAR"] == 1
        assert not os.path.exists(f"{O2_ncl_folder}/POSCAR")

        # test without SOC -> SinglePoint subfolder
        if_present_rm("CompetingPhases")
        _dict_sets_no_soc = cp.write_singlepoint_files(soc=False, potcar_spec=True)
        ZrO2_sp_folder = "CompetingPhases/ZrO2_P2_1c_EaH_0/SinglePoint/"
        assert os.path.exists(ZrO2_sp_folder)
        assert not os.path.exists("CompetingPhases/ZrO2_P2_1c_EaH_0/vasp_ncl")
        with open(f"{ZrO2_sp_folder}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert any("NSW = 0" in line for line in contents)
            assert not any("LSORBIT" in line for line in contents)
        assert not os.path.exists(f"{ZrO2_sp_folder}/POSCAR")  # no POSCAR by default

        # overwrite warning (one ``UserWarning`` per existing folder, plus possible KPAR
        # ``UserWarning``s for Γ-only molecular phases)
        _result, _stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_singlepoint_files, soc=False, potcar_spec=True
        )
        overwrite_w = [ww for ww in w if "already exists. Overwriting files." in str(ww.message)]
        other_w = [ww for ww in w if ww not in overwrite_w]
        assert len(overwrite_w) > 0
        assert all(issubclass(ww.category, UserWarning) for ww in overwrite_w)
        assert all(
            issubclass(ww.category, UserWarning) and "KPOINTS are Γ-only" in str(ww.message)
            for ww in other_w
        )

        # user_incar_settings override
        if_present_rm("CompetingPhases")
        dict_sets_custom = cp.write_singlepoint_files(
            soc=False, potcar_spec=True, user_incar_settings={"ALGO": "Normal"}
        )
        ds_custom = dict_sets_custom["CompetingPhases/ZrO2_P2_1c_EaH_0/SinglePoint"]
        assert ds_custom.incar["ALGO"] == "Normal"
        assert ds_custom.incar["NSW"] == 0  # singlepoint settings still applied

        # ``poscar=True`` explicitly writes POSCAR files
        if_present_rm("CompetingPhases")
        cp.write_singlepoint_files(soc=False, potcar_spec=True, poscar=True)
        assert os.path.isfile(f"{ZrO2_sp_folder}/POSCAR")
        assert os.path.isfile(f"{ZrO2_sp_folder}/INCAR")
        assert os.path.isfile(f"{ZrO2_sp_folder}/KPOINTS")

    def test_singlepoint_custom_output_path(self):
        """
        Test ``write_singlepoint_files`` with custom ``output_path``.
        """
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        custom_dir = "CustomOutputDir"
        if_present_rm(custom_dir)

        sp_sets = cp.write_singlepoint_files(soc=False, potcar_spec=True, output_path=custom_dir)
        assert sp_sets
        assert not os.path.exists("CompetingPhases")
        assert all(key.startswith(f"{custom_dir}/") for key in sp_sets)
        sample_key = next(iter(sp_sets))
        assert os.path.exists(sample_key)
        assert os.path.isfile(os.path.join(sample_key, "INCAR"))

    def test_subfolder_parameter(self):
        """
        Test subfolder parameter for relaxation and singlepoint methods.
        """
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)

        # relaxation: default subfolder is Relax
        relax_sets = cp.get_relaxation_sets()
        assert all("Relax" in key for key in relax_sets)

        # relaxation: custom subfolder
        relax_sets_custom = cp.get_relaxation_sets(subfolder="my_relax")
        assert all("my_relax" in key for key in relax_sets_custom)
        assert all("Relax" not in key for key in relax_sets_custom)

        # write_relaxation_files: custom subfolder
        if_present_rm("CompetingPhases")
        cp.write_relaxation_files(potcar_spec=True, subfolder="relax_v2")
        assert os.path.isdir("CompetingPhases/ZrO2_P2_1c_EaH_0/relax_v2")
        assert os.path.isfile("CompetingPhases/ZrO2_P2_1c_EaH_0/relax_v2/INCAR")

        # singlepoint: custom subfolder overrides default (vasp_ncl for SOC)
        sp_sets = cp.get_singlepoint_sets(subfolder="my_sp")
        assert all("my_sp" in key for key in sp_sets)
        assert all("vasp_ncl" not in key for key in sp_sets)

        # singlepoint soc=False: custom subfolder overrides default (SinglePoint)
        sp_sets_nosoc = cp.get_singlepoint_sets(soc=False, subfolder="sp_nosoc")
        assert all("sp_nosoc" in key for key in sp_sets_nosoc)
        assert all("SinglePoint" not in key for key in sp_sets_nosoc)

        # write_singlepoint_files: custom subfolder
        if_present_rm("CompetingPhases")
        cp.write_singlepoint_files(soc=False, potcar_spec=True, subfolder="sp_custom")
        assert os.path.isdir("CompetingPhases/ZrO2_P2_1c_EaH_0/sp_custom")
        assert os.path.isfile("CompetingPhases/ZrO2_P2_1c_EaH_0/sp_custom/INCAR")
        if_present_rm("CompetingPhases")

    def test_warnings_output(self):
        """
        Test warning/print outputs from ``CompetingPhases`` generation and
        input-file methods.
        """
        if_present_rm("CompetingPhases")
        cp, stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.CompetingPhases, "ZrO2", energy_above_hull=0.03, api_key=api_key
        )
        assert not stdout
        assert not w

        # SOC auto-detection prints info message (ZrO2 has Zr Z=40 >= 31)
        _result, stdout, w = _run_func_and_capture_stdout_warnings(cp.get_singlepoint_sets)
        assert "Spin-orbit coupling (SOC) is being used by default" in stdout
        assert "Z >= 31" in stdout
        assert not w

        # no SOC message when soc is explicitly set
        _result, stdout, w = _run_func_and_capture_stdout_warnings(cp.get_singlepoint_sets, soc=True)
        assert "Spin-orbit coupling (SOC) is being used by default" not in stdout
        assert not w
        _result, stdout, w = _run_func_and_capture_stdout_warnings(cp.get_singlepoint_sets, soc=False)
        assert "Spin-orbit coupling (SOC) is being used by default" not in stdout
        assert not w

        # overwrite warning from write_relaxation_files
        _result, stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_relaxation_files, potcar_spec=True
        )
        assert not w
        _result, stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_relaxation_files, potcar_spec=True
        )
        assert len(w) == len(cp)  # warning for each overwritten folder
        assert any(
            "already exists. Overwriting files." in str(warning.message)
            for warning in w
            if warning.category is UserWarning
        )

        # overwrite warning from write_singlepoint_files
        if_present_rm("CompetingPhases")
        _result, stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_singlepoint_files, potcar_spec=True
        )
        assert not w
        _result, stdout, w = _run_func_and_capture_stdout_warnings(
            cp.write_singlepoint_files, potcar_spec=True
        )
        assert len(w) == len(cp)  # warning for each overwritten folder
        assert any(
            "already exists. Overwriting files." in str(warning.message)
            for warning in w
            if warning.category is UserWarning
        )
        if_present_rm("CompetingPhases")

    def test_default_soc(self):
        """
        Test SOC default logic based on atomic numbers.
        """
        # ZrO2: Zr Z=40 >= 31, SOC should default True -> vasp_ncl
        cp_heavy = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        dict_sets = cp_heavy.get_singlepoint_sets()
        assert all("vasp_ncl" in key for key in dict_sets)

        # MgO: Mg Z=12, O Z=8, both < 31, SOC should default False -> SinglePoint
        cp_light = chemical_potentials.CompetingPhases("MgO", energy_above_hull=0, api_key=api_key)
        dict_sets = cp_light.get_singlepoint_sets()
        assert all("SinglePoint" in key for key in dict_sets)
        for ds in dict_sets.values():
            assert "LSORBIT" not in ds.incar

    def test_api_keys_errors(self):
        expected_error_substr = "is not a valid Materials Project API key"
        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="test",
            )
        assert expected_error_substr in str(e.value)

        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="c2LiJRMiBeaN5iXsH",  # legacy API key (16 chars, not 32)
            )
        assert expected_error_substr in str(e.value)

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
                    struct.composition.reduced_formula, api_key=api_key
                )
            with warnings.catch_warnings(record=True) as w2:  # ensure duplicate warnings not ignored
                cp_struct_input = chemical_potentials.CompetingPhases(struct, api_key=api_key)

            _check_structure_input(cp, cp_struct_input, struct, name, w + w2, api_key)

    def test_init_get_entries_kwargs_passthrough(self):
        """
        Extra ``**kwargs`` to ``CompetingPhases`` should be forwarded to the
        ``get_entries_in_chemsys`` / ``get_entries`` helpers (and onto the
        underlying ``MPRester`` query).

        Here we use ``additional_criteria={"thermo_types": ["R2SCAN"]}`` to
        restrict the MP query to R2SCAN thermo entries, and verify that the
        resulting entries differ from the default (GGA/GGA+U/R2SCAN) query.
        """
        cp_default = chemical_potentials.CompetingPhases(  # GGA/GGA+U/R2SCAN
            "ZrO2", energy_above_hull=0.03, api_key=api_key
        )
        cp_r2scan = chemical_potentials.CompetingPhases(
            "ZrO2",
            energy_above_hull=0.03,
            api_key=api_key,
            additional_criteria={"thermo_types": ["R2SCAN"]},  # R2SCAN only
        )
        assert cp_r2scan._get_entries_kwargs == {"additional_criteria": {"thermo_types": ["R2SCAN"]}}
        # R2SCAN energies differ from default GGA(+U), so entries should differ:
        assert len(cp_default) == 13
        assert len(cp_r2scan) == 14  # different number of entries within EaH tolerance

        for entry in cp_default.entries:
            assert entry.energy_per_atom not in [
                r2scan_ent.energy_per_atom for r2scan_ent in cp_r2scan.entries
            ]

    def test_single_extrinsic_phase_limits_default(self):
        """
        Check the default for ``single_extrinsic_phase_limits`` is ``False``.
        """
        for fn in (
            chemical_potentials.CompetingPhases.__init__,
            chemical_potentials.CompetingPhasesAnalyzer.__init__,
            chemical_potentials.CompetingPhasesAnalyzer.calculate_chempots,
        ):
            assert inspect.signature(fn).parameters["single_extrinsic_phase_limits"].default is False

    # TODO: remove with the ``full_sub_approach`` shim in v4.1
    def test_full_sub_approach_kwarg_raises(self):
        """
        ``full_sub_approach`` was renamed to ``single_extrinsic_phase_limits``
        (with inverted polarity) in doped v4.0; passing the old name to
        ``CompetingPhases`` should raise ``ValueError``.
        """
        with pytest.raises(ValueError, match=r"full_sub_approach.*single_extrinsic_phase_limits"):
            chemical_potentials.CompetingPhases("ZrO2", full_sub_approach=True)

    # TODO: remove with the ``full_sub_approach`` shim in v4.1
    @pytest.mark.filterwarnings("always::DeprecationWarning")  # deliberate deprecated-API test
    def test_from_dict_full_sub_approach_translates(self):
        """
        Loading a ``CompetingPhases`` saved under the old ``full_sub_approach``
        API translates to ``single_extrinsic_phase_limits`` (with inverted
        polarity), with a ``DeprecationWarning``.
        """
        cp = chemical_potentials.CompetingPhases("ZrO2", energy_above_hull=0.03, api_key=api_key)
        cp_dict = cp.as_dict()
        # simulate a save under the legacy API: ``full_sub_approach`` was the inverse polarity:
        legacy_dict = {**cp_dict, "full_sub_approach": not cp_dict.pop("single_extrinsic_phase_limits")}

        for legacy_value in (True, False):
            legacy_dict["full_sub_approach"] = legacy_value
            cp_loaded, _stdout, w = _run_func_and_capture_stdout_warnings(
                chemical_potentials.CompetingPhases.from_dict, legacy_dict
            )
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "full_sub_approach" in str(warning.message)
                for warning in w
            )
            assert len(w) == 1
            assert cp_loaded.single_extrinsic_phase_limits is (not legacy_value)
            assert "full_sub_approach" not in cp_loaded.__dict__  # avoid __getattr__ delegation

    def test_MP_doc_dicts(self):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", MP_doc_dicts=True, energy_above_hull=0.03, api_key=api_key
        )
        assert cp.MP_doc_dicts
        assert len(cp.MP_doc_dicts) == 12  # just missing O2
        assert len(cp.entries) == 13
        assert set(cp.MP_doc_dicts.keys()) == {
            entry.data["material_id"] for entry in cp.entries if not entry.data["molecule"]
        }
        assert [entry.name for entry in cp.entries] == self.ZrO2_entry_list
        self._check_ZrO2_cp_init(cp)
        assert "Zr4O" not in [e.name for e in cp.entries]  # not bordering or potentially with EaH


def _check_structure_input(cp, cp_struct_input, struct, name, w, api_key, extrinsic=False):
    _print_warning_info(w)  # for debugging
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
    def setUp(self):
        CompetingPhasesTestCase.setUp(self)
        self.La_ZrO2_cp = chemical_potentials.CompetingPhases(
            "ZrO2", extrinsic="La", api_key=api_key
        )  # default energy_above_hull=0.05

    def tearDown(self):
        CompetingPhasesTestCase.tearDown(self)

    def test_init(self):
        assert len(self.La_ZrO2_cp.extrinsic_entries) == 3
        assert len(self.La_ZrO2_cp.entries) == 21
        assert self.La_ZrO2_cp.extrinsic_entries[2].name == "La"  # definite ordering, same 1,2 as before
        assert all(entry.data["energy_above_hull"] == 0 for entry in self.La_ZrO2_cp.extrinsic_entries[:2])
        assert all(entry.data["energy_above_hull"] != 0 for entry in self.La_ZrO2_cp.extrinsic_entries[2:])
        assert len(self.La_ZrO2_cp.intrinsic_entries) == 18

        ex_cp = chemical_potentials.CompetingPhases(
            "ZrO2", extrinsic="La", energy_above_hull=0, api_key=api_key
        )
        assert len(ex_cp.extrinsic_entries) == 2
        assert len(ex_cp.entries) == 6
        assert ex_cp.extrinsic_entries[0].name == "La"  # definite ordering
        assert ex_cp.extrinsic_entries[1].name == "La2Zr2O7"  # definite ordering
        assert all(entry.data["energy_above_hull"] == 0 for entry in ex_cp.entries)

        # names of intrinsic entries: ['Zr', 'O2', 'Zr3O', 'ZrO2']
        assert len(ex_cp.intrinsic_entries) == 4
        assert [entry.name for entry in ex_cp.intrinsic_entries] == self.ZrO2_entry_list[:4]

    def test_structure_input(self):
        for struct, name in [
            (self.cdte, "CdTe_F-43m_EaH_0"),
            (self.cdte * 2, "CdTe_F-43m_EaH_0"),  # supercell
            (self.na2fepo4f, "Na2FePO4F_Pbcn_EaH_0.17"),
            (self.cu2sise4, "Cu2SiSe4_P1_EaH_0"),
        ]:
            with warnings.catch_warnings(record=True) as w:
                cp = chemical_potentials.CompetingPhases(
                    struct.composition.reduced_formula, api_key=api_key, extrinsic={"K"}
                )
            with warnings.catch_warnings(record=True) as w2:  # ensure duplicate warnings not ignored
                cp_struct_input = chemical_potentials.CompetingPhases(
                    struct, api_key=api_key, extrinsic={"K"}
                )

            _check_structure_input(cp, cp_struct_input, struct, name, w + w2, api_key, extrinsic=True)

            for entries_list in [cp_struct_input.extrinsic_entries, cp.extrinsic_entries]:
                assert len(entries_list) >= 1
                for extrinsic_entry in entries_list:
                    assert "K" in extrinsic_entry.data["doped_name"]
                    assert "K" in extrinsic_entry.name

    def test_extrinsic_only_setup(self):
        extrinsic_folder_names = [
            chemical_potentials._get_competing_phase_folder_name(entry)
            for entry in self.La_ZrO2_cp.extrinsic_entries
        ]
        intrinsic_folder_names = [
            chemical_potentials._get_competing_phase_folder_name(entry)
            for entry in self.La_ZrO2_cp.intrinsic_entries
        ]

        if_present_rm("CompetingPhases")
        conv_dict_sets = self.La_ZrO2_cp.write_kpoint_convergence_files(
            kpoints_metals=(5, 10, 5),
            kpoints_nonmetals=(5, 10, 5),
            potcar_spec=True,
            extrinsic_only=True,
        )
        assert conv_dict_sets
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in conv_dict_sets.values())
        assert all(
            any(f"/{name}/" in f"/{key}/" for name in extrinsic_folder_names) for key in conv_dict_sets
        )
        assert all(
            not any(f"/{name}/" in f"/{key}/" for name in intrinsic_folder_names) for key in conv_dict_sets
        )
        for name in extrinsic_folder_names:
            assert os.path.isdir(f"CompetingPhases/{name}/kpoint_converge")
        for name in intrinsic_folder_names:
            assert not os.path.exists(f"CompetingPhases/{name}")

        if_present_rm("CompetingPhases")
        std_dict_sets = self.La_ZrO2_cp.write_relaxation_files(
            potcar_spec=True,
            extrinsic_only=True,
        )
        assert std_dict_sets
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in std_dict_sets.values())
        assert all(
            any(f"/{name}/" in f"/{key}/" for name in extrinsic_folder_names) for key in std_dict_sets
        )
        assert all(
            not any(f"/{name}/" in f"/{key}/" for name in intrinsic_folder_names) for key in std_dict_sets
        )
        for name in extrinsic_folder_names:
            assert os.path.isdir(f"CompetingPhases/{name}/Relax")
        for name in intrinsic_folder_names:
            assert not os.path.exists(f"CompetingPhases/{name}")

        # test extrinsic_only with singlepoint files
        if_present_rm("CompetingPhases")
        sp_dict_sets = self.La_ZrO2_cp.write_singlepoint_files(
            potcar_spec=True,
            extrinsic_only=True,
        )
        assert sp_dict_sets
        assert all(isinstance(v, chemical_potentials.DopedDictSet) for v in sp_dict_sets.values())
        assert all(
            any(f"/{name}/" in f"/{key}/" for name in extrinsic_folder_names) for key in sp_dict_sets
        )
        assert all(
            not any(f"/{name}/" in f"/{key}/" for name in intrinsic_folder_names) for key in sp_dict_sets
        )
        # La_ZrO2 has Zr (Z=40), so SOC defaults on -> vasp_ncl
        for name in extrinsic_folder_names:
            assert os.path.isdir(f"CompetingPhases/{name}/vasp_ncl")
        for name in intrinsic_folder_names:
            assert not os.path.exists(f"CompetingPhases/{name}")
        for ds in sp_dict_sets.values():
            assert ds.incar["NSW"] == 0
            assert ds.incar["LSORBIT"] is True

    def test_extrinsic_soc_default(self):
        """
        Test SOC default considers extrinsic species atomic numbers.
        """
        # MgO (Mg Z=12, O Z=8) with Cd dopant (Z=48): max Z across all
        # species is 48 >= 31, so SOC should default to True (vasp_ncl)
        cp = chemical_potentials.CompetingPhases(
            "MgO", extrinsic="Cd", energy_above_hull=0, api_key=api_key
        )
        # SOC auto-enabled prints an info message
        result = _run_func_and_capture_stdout_warnings(cp.get_singlepoint_sets)
        dict_sets = result[0]
        stdout = result[1]
        assert "Spin-orbit coupling (SOC) is being used by default" in stdout
        assert all("vasp_ncl" in key for key in dict_sets)
        for ds in dict_sets.values():
            assert ds.incar["LSORBIT"] is True

        # no info message when soc is explicitly set
        result = _run_func_and_capture_stdout_warnings(cp.get_singlepoint_sets, soc=True)
        assert "Spin-orbit coupling (SOC) is being used by default" not in result[1]
        result = _run_func_and_capture_stdout_warnings(cp.get_singlepoint_sets, soc=False)
        assert "Spin-orbit coupling (SOC) is being used by default" not in result[1]

        # MgO without extrinsic: all light elements, SOC defaults False
        cp_light = chemical_potentials.CompetingPhases("MgO", energy_above_hull=0, api_key=api_key)
        dict_sets_light = cp_light.get_singlepoint_sets()
        assert all("SinglePoint" in key for key in dict_sets_light)

    def _check_extrinsic_cp_entries(self, cp, host_elements, extrinsic_elements, codoping=False):
        """
        Generic invariants for the entries in a ``CompetingPhases`` object
        with extrinsic species: intrinsic entries are pure host phases,
        extrinsic entries each contain at least one extrinsic species, all
        elemental references are present, and multi-extrinsic ("codoping")
        phases appear iff ``codoping=True``.
        """
        host_set = {Element(e) for e in host_elements}
        extrinsic_set = {Element(e) for e in extrinsic_elements}

        assert all(set(e.composition.elements).issubset(host_set) for e in cp.intrinsic_entries)
        assert {e.composition.reduced_formula for e in cp.intrinsic_entries}.issuperset(
            {
                cp.composition.reduced_formula,
                *(
                    el.symbol
                    for el in host_set
                    if el.symbol not in chemical_potentials.elemental_diatomic_bond_lengths
                ),
            }
        )

        # extrinsic entries each contain at least one extrinsic species:
        assert all(extrinsic_set & set(e.composition.elements) for e in cp.extrinsic_entries)
        # all extrinsic elemental references are present:
        assert {e.composition.reduced_formula for e in cp.extrinsic_entries}.issuperset(
            {el.symbol for el in extrinsic_set}
        )

        # entries == intrinsic + extrinsic, each entry has an EaH and a doped_name:
        assert len(cp.entries) == len(cp.intrinsic_entries) + len(cp.extrinsic_entries)
        assert all(isinstance(e.data["energy_above_hull"], float) for e in cp.entries)
        assert all(isinstance(e.data["doped_name"], str) for e in cp.entries)

        # multi-extrinsic ("codoping") phases only appear when codoping=True:
        codoping_entries = [
            e for e in cp.entries if sum(1 for ext in extrinsic_set if ext in e.composition.elements) >= 2
        ]
        if codoping:
            assert len(codoping_entries) >= 1
        else:
            assert codoping_entries == []

    def _check_cpa_from_cp_entries(self, cp, host_elements, extrinsic_elements):
        """
        Build ``CompetingPhasesAnalyzer`` from ``cp.entries`` with both
        ``single_extrinsic_phase_limits=False`` (default, recommended) and
        ``True``, and verify parsing/chempot invariants.

        Returns ``(cpa_default, cpa_single_extrinsic_phase_limits)``.
        """
        composition = cp.composition.reduced_formula
        extrinsic_set = {Element(e) for e in extrinsic_elements}

        cpa_default = chemical_potentials.CompetingPhasesAnalyzer(
            composition,
            list(cp.entries),
        )
        cpa_single_extrinsic_phase_limits = chemical_potentials.CompetingPhasesAnalyzer(
            composition,
            list(cp.entries),
            single_extrinsic_phase_limits=True,
        )

        for cpa in (cpa_default, cpa_single_extrinsic_phase_limits):
            assert set(cpa.intrinsic_elements) == set(host_elements)
            assert set(cpa.extrinsic_elements) == set(extrinsic_elements)
            # column ordering: host first, then extrinsic, matching ``cpa.elements``:
            assert list(cpa.chempots_df.columns) == [
                el for el in cpa.elements if el in cpa.chempots_df.columns
            ]
            assert set(cpa.chempots_df.columns) == set(host_elements) | set(extrinsic_elements)
            assert list(cpa.chempots_df.columns[: len(cpa.intrinsic_elements)]) == cpa.intrinsic_elements
            assert cpa.chempots_df.notna().all().all()
            assert len(cpa.chempots_df) >= 1

            for limit in cpa.chempots_df.index:
                phases = limit.split("-")
                for ext_el in extrinsic_set:
                    n_ext = sum(1 for p in phases if ext_el in Composition(p).elements)
                    if cpa.single_extrinsic_phase_limits:
                        assert n_ext == 1, (
                            f"Single-extrinsic-phase limit {limit!r} should have (only) 1 phase w/{ext_el}"
                        )
                    else:
                        assert n_ext >= 1, (
                            f"Should have at least one extrinsic phase for element {ext_el} in {limit!r}"
                        )

        assert len(cpa_default.chempots_df) >= len(cpa_single_extrinsic_phase_limits.chempots_df)

        return cpa_default, cpa_single_extrinsic_phase_limits

    def _check_cp_to_cpa_combinations(self, composition, host_elements, extrinsic, cases):
        """
        Run a parametrised ``CompetingPhases`` -> ``CompetingPhasesAnalyzer``
        roundtrip over ``cases``, where each case is ``(kwargs,
        expected_counts)`` and ``expected_counts`` is either ``(n_entries,
        n_intrinsic, n_extrinsic)`` or ``None`` (skip count assertions).

        For every case: builds ``CompetingPhases`` with ``kwargs``, asserts
        that the init flags match, asserts entry counts (if given), runs
        ``_check_extrinsic_cp_entries`` and ``_check_cpa_from_cp_entries``, and
        verifies the CPA's intrinsic chempots are stable across all cases (the
        intrinsic phase diagram is independent of any extrinsic flags).
        """
        intrinsic_chempots_ref = None
        for kwargs, expected_counts in cases:
            cp = chemical_potentials.CompetingPhases(
                composition,
                energy_above_hull=0.03,
                extrinsic=extrinsic,
                api_key=api_key,
                **kwargs,
            )
            codoping = kwargs.get("codoping", False)
            assert cp.single_extrinsic_phase_limits is (
                kwargs.get("single_extrinsic_phase_limits", False) and not codoping
            )
            assert cp.codoping is codoping
            assert cp.full_phase_diagram is kwargs.get("full_phase_diagram", False)
            if expected_counts is not None:
                n_entries, n_intrinsic, n_extrinsic = expected_counts
                assert len(cp.entries) == n_entries
                assert len(cp.intrinsic_entries) == n_intrinsic
                assert len(cp.extrinsic_entries) == n_extrinsic
            self._check_extrinsic_cp_entries(cp, host_elements, extrinsic, codoping=codoping)

            cpa_default, cpa_single_extrinsic_phase_limits = self._check_cpa_from_cp_entries(
                cp, host_elements, extrinsic
            )

            for cpa in [cpa_default, cpa_single_extrinsic_phase_limits]:
                if intrinsic_chempots_ref is None:
                    intrinsic_chempots_ref = cpa.intrinsic_chempots
                else:
                    _compare_chempot_dicts(
                        _canonicalise_chempot_dict(cpa.intrinsic_chempots),
                        _canonicalise_chempot_dict(intrinsic_chempots_ref),
                    )

    def test_BaSnO3_K_In_single_extrinsic_phase_limits_codoping_full_phase_diagram(self):
        """
        Test ``CompetingPhases`` generation and ``CompetingPhasesAnalyzer``
        parsing roundtrip for BaSnO3 with K and In as extrinsic species,
        covering the default approach, ``single_extrinsic_phase_limits=True``,
        ``codoping=True`` and ``full_phase_diagram=True`` cases (and
        combinations thereof).

        Co-doping competing phases (entries containing more than one extrinsic
        species, e.g. ``KInO2``) should only be generated when
        ``codoping=True`` (which also forces
        ``single_extrinsic_phase_limits=False``).
        ``full_phase_diagram=True`` includes all phases on the MP phase diagram
        (within ``energy_above_hull``) rather than only those potentially
        bordering the host, so increases both the intrinsic and extrinsic entry
        counts.
        """
        cases = [
            # (kwargs, (n_entries, n_intrinsic, n_extrinsic))
            # default (full phase diagram) approach: no codoping entries:
            ({}, (47, 23, 24)),
            # single_extrinsic_phase_limits=True (PyCDT-style restriction): fewer extrinsic phases:
            ({"single_extrinsic_phase_limits": True}, (44, 23, 21)),
            # codoping=True: forces single_extrinsic_phase_limits=False and includes KInO2 codoping phase:
            ({"codoping": True}, (48, 23, 25)),
            # full_phase_diagram=True + single_extrinsic_phase_limits=True: all MP phases included
            # for intrinsic chemical system; single-phase restriction keeps the extrinsic system small:
            ({"full_phase_diagram": True, "single_extrinsic_phase_limits": True}, (58, 37, 21)),
            # full_phase_diagram=True (default extrinsic): extended extrinsic phase set, still
            # no codoping entries (codoping=False prunes joint K-In phases):
            ({"full_phase_diagram": True}, (82, 37, 45)),
            # full_phase_diagram=True + codoping=True: largest case, with codoping entries
            # (e.g. ``KInO2``, ``K17In41``) and full intrinsic phase diagram:
            ({"full_phase_diagram": True, "codoping": True}, (88, 37, 51)),
        ]
        self._check_cp_to_cpa_combinations("BaSnO3", ["Ba", "Sn", "O"], ["K", "In"], cases)

    def test_BaSnO3_K_only_cp_to_cpa_combinations(self):
        """
        Test ``CompetingPhases`` -> ``CompetingPhasesAnalyzer`` roundtrip for
        BaSnO3 with a single extrinsic species (K), covering the
        ``single_extrinsic_phase_limits`` and ``full_phase_diagram``
        combinations (``codoping`` is meaningless with one extrinsic species).
        """
        cases = [
            ({}, (41, 23, 18)),
            ({"single_extrinsic_phase_limits": True}, (38, 23, 15)),
            ({"full_phase_diagram": True, "single_extrinsic_phase_limits": True}, (52, 37, 15)),
            ({"full_phase_diagram": True}, (65, 37, 28)),
        ]
        self._check_cp_to_cpa_combinations("BaSnO3", ["Ba", "Sn", "O"], ["K"], cases)

    def test_BaSnO3_4_extrinsic_cp_to_cpa_combinations(self):
        """
        Test ``CompetingPhases`` -> ``CompetingPhasesAnalyzer`` roundtrip for
        BaSnO3 with four extrinsic species, covering the
        ``single_extrinsic_phase_limits``, ``codoping`` and
        ``full_phase_diagram`` combinations.
        """
        cases = [
            # (kwargs, (n_entries, n_intrinsic, n_extrinsic))
            ({}, (74, 23, 51)),
            ({"single_extrinsic_phase_limits": True}, (67, 23, 44)),
            ({"codoping": True}, (79, 23, 56)),
            ({"full_phase_diagram": True, "single_extrinsic_phase_limits": True}, (81, 37, 44)),
            ({"full_phase_diagram": True}, (145, 37, 108)),
            ({"full_phase_diagram": True, "codoping": True}, (201, 37, 164)),
        ]
        self._check_cp_to_cpa_combinations("BaSnO3", ["Ba", "Sn", "O"], ["K", "In", "Na", "Mg"], cases)

    @pytest.mark.skipif(not _run_heavy_tests(), reason="Skipping heavy test")
    def test_Na2FePO4F_extrinsic_cp_to_cpa_combinations_heavy(self):
        """
        Heavy local-only test: ``CompetingPhases`` ->
        ``CompetingPhasesAnalyzer`` roundtrip for Na2FePO4F with two extrinsic
        species, covering ``single_extrinsic_phase_limits``, ``codoping`` and
        ``full_phase_diagram`` combinations.

        Marked heavy because Na2FePO4F has a 5-element host chemsys, so
        intrinsic + extrinsic queries pull hundreds of MP entries.
        """
        cases = [
            # (kwargs, (n_entries, n_intrinsic, n_extrinsic))
            ({}, (125, 82, 43)),
            ({"single_extrinsic_phase_limits": True}, (113, 82, 31)),
            ({"full_phase_diagram": True, "single_extrinsic_phase_limits": True}, (231, 200, 31)),
            ({"full_phase_diagram": True}, (320, 200, 120)),
            ({"codoping": True}, (133, 82, 51)),
            ({"full_phase_diagram": True, "codoping": True}, (344, 200, 144)),
        ]
        self._check_cp_to_cpa_combinations("Na2FePO4F", ["Na", "Fe", "P", "O", "F"], ["K", "In"], cases)


class ChemPotAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        self.ZrO2_path = os.path.join(EXAMPLE_DIR, "ZrO2_CompetingPhases")
        self.La_ZrO2_path = os.path.join(EXAMPLE_DIR, "La_ZrO2_CompetingPhases")
        self.MgO_path = os.path.join(EXAMPLE_DIR, "MgO/CompetingPhases")

        self.ZrO2_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path)
        self.La_ZrO2_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.La_ZrO2_path)

        self.ZrO2_parsed_chempots = loadfn(f"{self.ZrO2_path}/ZrO2_chempots.json")
        self.La_ZrO2_parsed_chempots = loadfn(f"{self.La_ZrO2_path}/ZrO2_la_chempots.json")
        self.y_ZrO2_parsed_chempots = loadfn(f"{self.La_ZrO2_path}/ZrO2_y_chempots.json")

        self.ZrO2_entry_list = [  # for testing ordering
            "ZrO2",
            "Zr",
            "O2",
            "Zr3O",
            "ZrO2",
            "Zr3O",
            "Zr2O",
            "Zr",
        ]

        self.ZrO2_chempots_df_dict = {
            "Zr": {"ZrO2-O2": -10.97543, "Zr3O-ZrO2": -0.19954},
            "O": {"ZrO2-O2": 0.0, "Zr3O-ZrO2": -5.38794},
        }
        self.La_ZrO2_chempots_df_dict = {
            "Zr": {"La2Zr2O7-ZrO2-O2": -10.97543, "La2Zr2O7-Zr3O-ZrO2": -0.19954},
            "O": {"La2Zr2O7-ZrO2-O2": 0.0, "La2Zr2O7-Zr3O-ZrO2": -5.38794},
            "La": {"La2Zr2O7-ZrO2-O2": -9.463, "La2Zr2O7-Zr3O-ZrO2": -1.38107},
        }

    def tearDown(self):
        for i in ["cpa.json"]:
            if_present_rm(i)

        if_present_rm(os.path.join(data_dir, "ZrO2_LaTeX_Tables/test.tex"))

        if os.path.exists(f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz"):
            if not os.path.exists(f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/mismatching_incar_vr.xml.gz"):
                shutil.move(
                    f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
                    f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/mismatching_incar_vr.xml.gz",
                )
            if not os.path.exists(f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/mismatching_potcar_vr.xml.gz"):
                shutil.move(
                    f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
                    f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/mismatching_potcar_vr.xml.gz",
                )
            shutil.move(
                f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz",
                f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            )

        shutil.copyfile(
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            f"{self.La_ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        if_present_rm(
            os.path.join(
                data_dir,
                "Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0/duplicate_for_testing_vasprun.xml.gz",
            )
        )

    def test_cpa_chempots(self):
        for chempots_df in [self.ZrO2_cpa.chempots_df, self.ZrO2_cpa.calculate_chempots()]:
            assert next(iter(chempots_df["O"])) == 0
            assert list(chempots_df.columns) == self.ZrO2_cpa.elements == ["Zr", "O"]

        for chempots_df in [
            self.La_ZrO2_cpa.chempots_df,
            self.La_ZrO2_cpa.calculate_chempots(extrinsic="La"),
        ]:
            assert all(limit.startswith("La2Zr2O7-") for limit in chempots_df.index)
            assert np.isclose(chempots_df["La"].loc["La2Zr2O7-ZrO2-O2"], -9.46298748)
            # columns and chempot dicts ordered host-first then extrinsic, matching ``self.elements``:
            assert list(chempots_df.columns) == self.La_ZrO2_cpa.elements == ["Zr", "O", "La"]
            for limit_key in ["limits", "limits_wrt_el_refs"]:
                for chempot_dict in self.La_ZrO2_cpa.chempots[limit_key].values():
                    assert list(chempot_dict.keys()) == self.La_ZrO2_cpa.elements

    def test_extrinsic_chempots_match_pycdt_single_extrinsic_phase_limits(self):
        """
        Confirm that doped's algebraic single-extrinsic-phase μ_extrinsic
        calculation gives the same result as the geometric PyCDT workflow
        (``full_sub_approach=False``, equivalent to
        ``single_extrinsic_phase_limits=True``): build a phase diagram from
        intrinsic + extrinsic entries, enumerate facets, keep only facets
        bordered by exactly one extrinsic phase (i.e. equilibria where the host
        phases still pin μ_host as in an intrinsic facet), and read
        ``μ_extrinsic`` off those facets.
        """
        cpa = self.La_ZrO2_cpa
        la = Element("La")

        intrinsic_entries = [e for e in cpa.entries if la not in e.composition.elements]
        extrinsic_entries = [e for e in cpa.entries if la in e.composition.elements]
        sub_pd = PhaseDiagram(intrinsic_entries + extrinsic_entries)
        mu_la_ref = sub_pd.el_refs[la].energy_per_atom

        # PyCDT ``full_sub_approach=False`` (``single_extrinsic_phase_limits=True``): enumerate
        # extrinsic-PD facets, keep those with exactly one La-bearing phase, and compare μ_La (relative to
        # elemental La) to doped. Limit keys are phase names joined with "-" (e.g.
        # ``ZrO2-O2-La2Zr2O7``); the same vertex is ``frozenset(facet_name.split("-"))``:
        pycdt_mu_la: dict[frozenset, float] = {}
        for facet_name, mu_dict in sub_pd.get_all_chempots(cpa.composition).items():
            phases = facet_name.split("-")
            if sum(la in Composition(p).elements for p in phases) != 1:
                continue  # codoping/multiple extrinsic phases at limit, excluded by the
                # single-extrinsic-phase-limits approximation
            pycdt_mu_la[frozenset(phases)] = mu_dict[la] - mu_la_ref

        doped_by_phases = {
            frozenset(limit.split("-")): float(row["La"]) for limit, row in cpa.chempots_df.iterrows()
        }
        assert set(pycdt_mu_la) == set(doped_by_phases)
        for k, mu in pycdt_mu_la.items():
            assert np.isclose(mu, doped_by_phases[k], atol=1e-3)

    def test_joint_extrinsic_chempots(self):
        """
        ``single_extrinsic_phase_limits=False`` (default, recommended) builds
        the joint (intrinsic + extrinsic) phase diagram and reads ``μ_host``
        and ``μ_extrinsic`` together at every facet (limit).

        Cross-check the result against a direct
        :meth:`~pymatgen.analysis.PhaseDiagram.get_all_chempots` call on the
        same entries, and against the single-extrinsic-phase-limit approach
        (whose ``μ_X`` values must agree at any shared facet — true for La-ZrO2
        since La2Zr2O7 binds ``μ_La`` at both intrinsic facets).
        """
        cpa = self.La_ZrO2_cpa
        full_df = cpa.chempots_df.copy()
        la = Element("La")
        single_extrinsic_phase_limits_df = cpa.calculate_chempots(single_extrinsic_phase_limits=True)
        # restore default-mode chempots on cpa (calculate_chempots mutates self.chempots):
        cpa.calculate_chempots(verbose=False)

        # output should be a μ column for every host + extrinsic element, with no `-Limiting Phase` cols:
        assert set(full_df.columns) == {"Zr", "O", "La"}
        assert all("La" in chempot_dict for chempot_dict in cpa.chempots["limits_wrt_el_refs"].values())
        assert "La" in cpa.chempots["elemental_refs"]

        # cross-check against a direct ``PhaseDiagram.get_all_chempots()`` call on the same entries:
        full_pd = PhaseDiagram(
            cpa.phase_diagram.entries,
            [*map(Element, cpa.composition.elements), la],
        )
        direct_chempots = full_pd.get_all_chempots(cpa.composition.reduced_composition)
        elemental_refs = {str(el): ent.energy_per_atom for el, ent in full_pd.el_refs.items()}
        direct_by_phases = {  # match by frozenset of phases at the vertex (facet-name ordering may differ)
            frozenset(facet.split("-")): {
                el.symbol: round(mu - elemental_refs[el.symbol], 4) for el, mu in mu_dict.items()
            }
            for facet, mu_dict in direct_chempots.items()
        }
        assert {frozenset(lim.split("-")) for lim in full_df.index} == set(direct_by_phases)
        for limit, row in full_df.iterrows():
            direct_row = direct_by_phases[frozenset(limit.split("-"))]
            for el in ("Zr", "O", "La"):
                assert np.isclose(direct_row[el], row[el], atol=1e-3)

        # for La-ZrO2, La2Zr2O7 binds μ_La at both intrinsic facets, so each single-extrinsic-phase facet
        # should appear in the full-approach output (lifted with the limiting phase added) with matching μ:
        for limit, row in single_extrinsic_phase_limits_df.iterrows():
            phases = frozenset(limit.split("-"))
            full_limit = next(lim for lim in full_df.index if frozenset(lim.split("-")) == phases)
            for el in ("Zr", "O", "La"):
                assert np.isclose(row[el], full_df.loc[full_limit, el], atol=1e-3)

        cpa_parsed_default = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2",
            self.La_ZrO2_path,
            single_extrinsic_phase_limits=False,  # default
        )
        pd.testing.assert_frame_equal(
            cpa_parsed_default.chempots_df,
            full_df,
            check_like=True,
            rtol=1e-5,
            atol=1e-5,
        )
        _compare_chempot_dicts(cpa_parsed_default.chempots, cpa.chempots)

    def _fabricate_Y_analogue_entries(self):
        """
        Build Y analogues of the La-bearing phases in ``self.La_ZrO2_cpa`` to
        get a 2-extrinsic dataset for testing multi-extrinsic behaviour.
        """
        extra_entries = []
        for entry in self.La_ZrO2_cpa.entries:
            if "La" in entry.composition:
                comp = Composition(
                    {"Y" if str(el) == "La" else str(el): n for el, n in entry.composition.items()}
                )
                # offset so Y phases differ slightly
                energy = entry.energy + 0.5 if len(entry.composition) > 1 else entry.energy
                ce = ComputedEntry(comp, energy)
                ce.parameters = dict(entry.parameters or {})
                ce.data = dict(entry.data or {})
                extra_entries.append(ce)
        return extra_entries

    def test_no_extrinsic_falls_through(self):
        """
        ``single_extrinsic_phase_limits`` with no extrinsic species should be a
        no-op equivalent to the intrinsic-only calculation, in either mode.
        """
        intrinsic_only = self.ZrO2_cpa.calculate_chempots(verbose=False)
        full = self.ZrO2_cpa.calculate_chempots(single_extrinsic_phase_limits=False, verbose=False)
        single = self.ZrO2_cpa.calculate_chempots(single_extrinsic_phase_limits=True, verbose=False)
        pd.testing.assert_frame_equal(intrinsic_only, full)
        pd.testing.assert_frame_equal(intrinsic_only, single)

    def test_single_extrinsic_phase_limits_with_multiple_extrinsic_species(self):
        """
        With >=2 extrinsic species and only single-extrinsic competing phases,
        the ``single_extrinsic_phase_limits=True`` approach should compute
        ``μ_X`` per species independently and produce limit keys with each
        species' limiting phase appearing exactly once, and matching the result
        for the extrinsic species parsed separately.
        """
        La_limiting_phase = "La2Zr2O7"
        Y_limiting_phase = "Y2Zr2O7"
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2",
            list(self.La_ZrO2_cpa.entries) + self._fabricate_Y_analogue_entries(),
            single_extrinsic_phase_limits=True,
        )
        assert set(cpa.extrinsic_elements) == {"La", "Y"}
        assert {"La", "Y"} <= set(cpa.chempots_df.columns)
        # columns and chempot dicts ordered host-first then extrinsic, matching ``self.elements``:
        assert list(cpa.chempots_df.columns) == cpa.elements
        assert cpa.elements == ["Zr", "O", "Y", "La"]
        assert cpa.elements[: len(cpa.intrinsic_elements)] == cpa.intrinsic_elements
        for limit_key in ["limits", "limits_wrt_el_refs"]:
            for chempot_dict in cpa.chempots[limit_key].values():
                assert list(chempot_dict.keys()) == cpa.elements

        # each extrinsic element's limiting phase must appear at most once in every limit key
        for limit_key in cpa.chempots["limits_wrt_el_refs"]:
            for limiting_phase in (La_limiting_phase, Y_limiting_phase):
                assert limit_key.count(limiting_phase) <= 1, (
                    f"Limit key {limit_key!r} contains {limiting_phase!r} more than once"
                )
            assert La_limiting_phase in limit_key
            assert Y_limiting_phase in limit_key

        # μ_La must match the single-extrinsic La-only result:
        for multi_limit, multi_row in cpa.chempots_df.iterrows():
            multi_phases = set(multi_limit.split("-"))
            assert any(
                set(limit.split("-")) <= multi_phases
                and np.isclose(row[elt], multi_row[elt], atol=1e-3)
                and not np.isclose(multi_row["Y"], multi_row["La"], atol=1e-2)
                for limit, row in self.La_ZrO2_cpa.chempots_df.iterrows()
                for elt in ["Zr", "O", "La"]
            )

        # ``sort_by`` should work for any element (host or extrinsic), with chempots_df rows and
        # ``self.chempots["limits"]`` / ``["limits_wrt_el_refs"]`` keys all sharing the same order:
        for sort_el in ["Zr", "O", "La", "Y"]:
            sorted_df = cpa.calculate_chempots(sort_by=sort_el, verbose=False)
            assert sorted_df[sort_el].tolist() == sorted(sorted_df[sort_el].tolist(), reverse=True)
            assert (
                sorted_df.index.tolist()
                == list(cpa.chempots["limits"].keys())
                == list(cpa.chempots["limits_wrt_el_refs"].keys())
            )
            # column ordering preserved (host first, then extrinsic) regardless of sort_by:
            assert list(sorted_df.columns) == cpa.elements

        with pytest.raises(KeyError):
            cpa.calculate_chempots(sort_by="Cu", verbose=False)

    def test_calculate_chempots_does_not_corrupt_intrinsic_chempots(self):
        """
        Repeated ``calculate_chempots`` calls (with extrinsic / subset /
        ``single_extrinsic_phase_limits`` / ``sort_by`` variations) must always
        leave ``self.intrinsic_chempots`` and ``self.intrinsic_chempots_df``
        consistent with the host-only intrinsic result -- only the row/key
        ordering may change with ``sort_by``.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2",
            list(self.La_ZrO2_cpa.entries) + self._fabricate_Y_analogue_entries(),
        )
        # reference intrinsic values (host-only CPA, parsed independently):
        ref_chempots = deepcopy(self.ZrO2_cpa.chempots)
        ref_chempots_df = self.ZrO2_cpa.chempots_df.copy()

        def _check_intrinsic_unchanged():
            _compare_chempot_dicts(
                _canonicalise_chempot_dict(cpa.intrinsic_chempots),
                _canonicalise_chempot_dict(ref_chempots),
            )
            assert set(cpa.intrinsic_chempots_df.columns) == set(ref_chempots_df.columns)
            assert {"-".join(sorted(k.split("-"))) for k in cpa.intrinsic_chempots_df.index} == {
                "-".join(sorted(k.split("-"))) for k in ref_chempots_df.index
            }
            for limit, row in cpa.intrinsic_chempots_df.iterrows():
                ref_row = ref_chempots_df.loc[
                    next(
                        ref_lim
                        for ref_lim in ref_chempots_df.index
                        if set(ref_lim.split("-")) == set(limit.split("-"))
                    )
                ]
                for el in cpa.intrinsic_elements:
                    assert np.isclose(row[el], ref_row[el], atol=1e-5)

        for kwargs in [
            {},
            {"extrinsic": "La"},
            {"extrinsic": ["La", "Y"]},
            {"single_extrinsic_phase_limits": True},
            {"sort_by": "Zr"},
            {"sort_by": "O"},
            {"sort_by": "La"},  # extrinsic sort: must still leave intrinsic_chempots intact
            {"extrinsic": "Y", "sort_by": "Y"},
            {"extrinsic": "La", "single_extrinsic_phase_limits": True, "sort_by": "O"},
        ]:
            cpa.calculate_chempots(verbose=False, **kwargs)
            _check_intrinsic_unchanged()

    def test_calculate_chempots_extrinsic_subset(self):
        """
        With ``single_extrinsic_phase_limits=False`` (default) and
        ``extrinsic`` set to a subset of the parsed extrinsic species, only
        that subset's competing phases should enter the joint phase diagram,
        and the result should match the equivalent single-extrinsic CPA.
        """
        # construct a multi-extrinsic CPA (La + fabricated Y analogues), then check that requesting
        # ``extrinsic="La"`` (and ``extrinsic=["La"]``, ``extrinsic=Element("La")``) reproduces the
        # La-only result, while requesting ``extrinsic="Y"`` reproduces the Y-only result and excludes
        # La phases:
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2",
            list(self.La_ZrO2_cpa.entries) + self._fabricate_Y_analogue_entries(),
        )
        assert set(cpa.extrinsic_elements) == {"La", "Y"}

        # La-only Y-only references for comparison: build a separate CPA from each subset of entries
        Y_only_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2",
            [e for e in cpa.entries if "La" not in e.composition],
        )
        assert set(Y_only_cpa.extrinsic_elements) == {"Y"}

        for extrinsic_arg in ["La", ["La"], Element("La")]:
            subset_df = cpa.calculate_chempots(extrinsic=extrinsic_arg, verbose=False)
            # columns restricted to intrinsic + La (no Y), ordered host-first then extrinsic:
            assert list(subset_df.columns) == ["Zr", "O", "La"]
            # μ values must match the La-only CPA's chempots_df:
            pd.testing.assert_frame_equal(
                subset_df,
                self.La_ZrO2_cpa.chempots_df,
                check_like=True,
                rtol=1e-5,
                atol=1e-5,
            )
            # ``self.chempots`` keys/columns mirror the requested subset (no Y):
            for limit_key in ["limits", "limits_wrt_el_refs"]:
                for chempot_dict in cpa.chempots[limit_key].values():
                    assert "Y" not in chempot_dict
                    assert set(chempot_dict.keys()) == {"Zr", "O", "La"}
            # no Y-bearing phases should appear in any limit key:
            assert all("Y" not in limit for limit in cpa.chempots["limits_wrt_el_refs"])

        # symmetric check with the Y subset:
        Y_subset_df = cpa.calculate_chempots(extrinsic="Y", verbose=False)
        assert list(Y_subset_df.columns) == ["Zr", "O", "Y"]
        pd.testing.assert_frame_equal(
            Y_subset_df,
            Y_only_cpa.chempots_df,
            check_like=True,
            rtol=1e-5,
            atol=1e-5,
        )
        assert all("La" not in limit for limit in cpa.chempots["limits_wrt_el_refs"])

        # ``intrinsic_chempots`` must be unaffected by the subset request:
        _compare_chempot_dicts(cpa.intrinsic_chempots, self.La_ZrO2_cpa.intrinsic_chempots)

    def test_calculate_chempots_missing_extrinsic_elemental_reference(self):
        """
        Extrinsic μ limits require a parsed elemental reference for that
        species.
        """
        with pytest.raises(ValueError, match="Elemental reference phase for the specified extrinsic"):
            self.ZrO2_cpa.calculate_chempots(extrinsic="La", verbose=False)

    def test_from_entries_warns_and_prunes_phases_without_elemental_reference(self):
        """
        Compounds containing an element without a parsed unary reference are
        dropped, with a warning, so intrinsic chemical potentials still build.
        """
        entries_no_la_metal = [
            e for e in self.La_ZrO2_cpa.entries if e.composition.reduced_formula != "La"
        ]
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", entries_no_la_metal)

        assert any(
            "No elemental reference phase (required for chemical potential analysis) was parsed for "
            "element(s): ['La']" in str(warning.message)
            for warning in w
        )
        assert not any("La" in e.composition for e in cpa.entries)
        assert "La" not in cpa.extrinsic_elements
        assert len(cpa.elements) == 2
        pd.testing.assert_frame_equal(cpa.chempots_df, self.ZrO2_cpa.chempots_df)

    def test_from_entries_raises_when_host_element_lacks_elemental_reference(self):
        entries_no_o2 = [e for e in self.ZrO2_cpa.entries if e.composition.reduced_formula != "O2"]
        with pytest.raises(
            ValueError,
            match="No elemental reference phase was parsed for host element",
        ):
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", entries_no_o2)

    def test_unstable_host_chempots(self):
        """
        Test the chemical potentials parsing when the host phase is unstable.
        """
        with warnings.catch_warnings(record=True) as w:
            unstable_cpa = chemical_potentials.CompetingPhasesAnalyzer("Zr2O", self.ZrO2_path)

        _print_warning_info(w)  # for debugging
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

        assert unstable_cpa.chempots["elemental_refs"] == self.ZrO2_parsed_chempots["elemental_refs"]
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
        _compare_chempot_dicts(self.ZrO2_cpa.chempots, self.ZrO2_parsed_chempots)

        assert (
            self.La_ZrO2_cpa.chempots["elemental_refs"] == self.La_ZrO2_parsed_chempots["elemental_refs"]
        )

    def test_sort_by(self):
        limits_order_zr_rich = ["Zr3O-ZrO2", "ZrO2-O2"]
        chempot_df = self.ZrO2_cpa.calculate_chempots(sort_by="Zr")
        assert np.isclose(next(iter(chempot_df["Zr"])), -0.199544, atol=1e-4)
        assert np.isclose(list(chempot_df["Zr"])[1], -10.975428439999998, atol=1e-4)
        assert chempot_df.index.tolist() == limits_order_zr_rich
        assert list(self.ZrO2_cpa.chempots["limits"].keys()) == limits_order_zr_rich

        limits_order_o_rich = ["ZrO2-O2", "Zr3O-ZrO2"]
        chempot_df_o = self.ZrO2_cpa.calculate_chempots(sort_by="O")
        assert np.isclose(next(iter(chempot_df_o["O"])), 0.0, atol=1e-4)
        assert np.isclose(list(chempot_df_o["O"])[1], -5.38794, atol=1e-4)
        assert chempot_df_o.index.tolist() == limits_order_o_rich
        assert list(self.ZrO2_cpa.chempots["limits"].keys()) == limits_order_o_rich

        with pytest.raises(KeyError):
            self.ZrO2_cpa.calculate_chempots(sort_by="M")

    def test_vaspruns(self):
        cpa = self.ZrO2_cpa
        assert len(cpa.elements) == 2

        self._general_cpa_check(cpa)
        assert cpa.chempots_df.to_dict() == self.ZrO2_chempots_df_dict

        cpa_w_subfolder = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2", self.ZrO2_path, subfolder="vasp_std"
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

        ext_cpa = self.La_ZrO2_cpa
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
        assert ext_cpa.chempots_df.to_dict() == self.La_ZrO2_chempots_df_dict

        # check if it works from a list
        all_paths = []
        for entry_folder in os.listdir(self.ZrO2_path):
            if os.path.isdir(os.path.join(self.ZrO2_path, entry_folder)) and "vasp_std" in os.listdir(
                os.path.join(self.ZrO2_path, entry_folder)
            ):
                all_paths.extend(
                    os.path.join(self.ZrO2_path, entry_folder, "vasp_std", vr_file)
                    for vr_file in os.listdir(os.path.join(self.ZrO2_path, entry_folder, "vasp_std"))
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

    def test_vaspruns_hidden_files(self):
        with open(f"{self.ZrO2_path}/._OUTCAR", "w") as f:
            f.write("test pop")
        with open(f"{self.ZrO2_path}/._vasprun.xml", "w") as f:
            f.write("test pop")
        with open(f"{self.ZrO2_path}/._LOCPOT", "w") as f:
            f.write("test pop")
        with open(f"{self.ZrO2_path}/.DS_Store", "w") as f:
            f.write("test pop")

        with warnings.catch_warnings(record=True) as w:
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path)
        _print_warning_info(w)  # for debugging
        assert not w

        for i in ["._OUTCAR", "._vasprun.xml", "._LOCPOT", ".DS_Store"]:
            if_present_rm(f"{self.ZrO2_path}/{i}")

    def test_vaspruns_none_parsed(self):
        with (
            tempfile.TemporaryDirectory() as empty_dir,
            pytest.raises(FileNotFoundError, match=r"No vasprun\.xml"),
        ):
            chemical_potentials.CompetingPhasesAnalyzer("ZrO2", empty_dir)

    def test_recursive_vasprun_discovery(self):
        """
        Test that vaspruns are found recursively and subfolder auto-detection
        (subfolder=None) picks vasp_std when present.
        """
        cpa_default = chemical_potentials.CompetingPhasesAnalyzer(
            "ZrO2", self.ZrO2_path, subfolder="vasp_std"
        )
        cpa_auto = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path, subfolder=None)
        self._compare_cpas(cpa_default, cpa_auto)

    def _build_mixed_subfolder_tree(self, tmp_path):
        """
        Build a directory tree with vaspruns in different subfolders.

        Layout under ``tmp_path``::

            ZrO2_EaH_0.0/vasp_ncl/vasprun.xml.gz  (from ZrO2_EaH_0.0)
            Zr_EaH_0.0/vasp_ncl/vasprun.xml.gz    (from Zr_EaH_0.0)
            O2_EaH_0.0/vasp_ncl/vasprun.xml.gz     (from O2_EaH_0.0)
            Zr3O_EaH_0.0/vasp_std/vasprun.xml.gz  (should be IGNORED by vasp_ncl)

        Returns the set of vasp_ncl vasprun paths (as resolved strings).
        """
        src = self.ZrO2_path
        ncl_phases = ["ZrO2_EaH_0.0", "Zr_EaH_0.0", "O2_EaH_0.0"]
        std_decoy = "Zr3O_EaH_0.0"

        ncl_paths = []
        for phase in ncl_phases:
            dest_dir = os.path.join(tmp_path, phase, "vasp_ncl")
            os.makedirs(dest_dir)
            dest = os.path.join(dest_dir, "vasprun.xml.gz")
            shutil.copy2(os.path.join(src, phase, "vasp_std", "vasprun.xml.gz"), dest)
            ncl_paths.append(os.path.realpath(dest))

        decoy_dir = os.path.join(tmp_path, std_decoy, "vasp_std")
        os.makedirs(decoy_dir)
        shutil.copy2(
            os.path.join(src, std_decoy, "vasp_std", "vasprun.xml.gz"),
            os.path.join(decoy_dir, "vasprun.xml.gz"),
        )
        return set(ncl_paths)

    def test_explicit_subfolder_ignores_others(self):
        """
        When ``subfolder="vasp_ncl"`` is given, vaspruns under ``vasp_std``
        must not be parsed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            ncl_paths = self._build_mixed_subfolder_tree(tmp_dir)
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", tmp_dir, subfolder="vasp_ncl")
            assert set(map(os.path.realpath, cpa.vasprun_paths)) == ncl_paths
            assert len(cpa.vasprun_paths) == 3

    def test_subfolder_auto_detect_picks_highest_priority(self):
        """
        With both ``vasp_ncl`` and ``vasp_std`` subfolders present,
        ``subfolder=None`` must auto-pick ``vasp_ncl`` (highest priority).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            ncl_paths = self._build_mixed_subfolder_tree(tmp_dir)
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", tmp_dir)  # subfolder=None by default
            assert set(map(os.path.realpath, cpa.vasprun_paths)) == ncl_paths
            assert len(cpa.vasprun_paths) == 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            ncl_paths = self._build_mixed_subfolder_tree(tmp_dir)
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", tmp_dir, subfolder=None)
            assert set(map(os.path.realpath, cpa.vasprun_paths)) == ncl_paths
            assert len(cpa.vasprun_paths) == 3

    def test_subfolder_not_found_warning_and_fallback(self):
        """
        If the requested subfolder doesn't exist, a warning is emitted and all
        discovered vaspruns are used as a fallback.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = self.ZrO2_path
            for phase in ["ZrO2_EaH_0.0", "Zr_EaH_0.0", "O2_EaH_0.0"]:
                dest_dir = os.path.join(tmp_dir, phase, "vasp_std")
                os.makedirs(dest_dir)
                shutil.copy2(
                    os.path.join(src, phase, "vasp_std", "vasprun.xml.gz"),
                    os.path.join(dest_dir, "vasprun.xml.gz"),
                )

            with warnings.catch_warnings(record=True) as w:
                cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", tmp_dir, subfolder="vasp_ncl")
            _print_warning_info(w)
            assert any("No vasprun.xml files found in 'vasp_ncl'" in str(wn.message) for wn in w)
            assert len(cpa.vasprun_paths) == 3

    def test_no_subfolder_flat_layout(self):
        """
        When vaspruns live directly in phase folders (no subfolder),
        ``subfolder=None`` should detect ``"."`` and parse all of them.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = self.ZrO2_path
            for phase in ["ZrO2_EaH_0.0", "Zr_EaH_0.0", "O2_EaH_0.0"]:
                dest_dir = os.path.join(tmp_dir, phase)
                os.makedirs(dest_dir)
                shutil.copy2(
                    os.path.join(src, phase, "vasp_std", "vasprun.xml.gz"),
                    os.path.join(dest_dir, "vasprun.xml.gz"),
                )

            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", tmp_dir)
            assert len(cpa.vasprun_paths) == 3

    def test_find_calc_outputs_shared_helper(self):
        """
        Direct tests for the shared ``_find_calc_outputs`` and
        ``_get_calc_files_df`` helpers in ``doped.io.vasp.outputs``.
        """
        from pathlib import Path

        calc_df = _get_calc_files_df(Path(self.ZrO2_path))
        assert not calc_df.empty
        assert set(calc_df["filename"].unique()) == {"vasprun.xml.gz"}
        assert len(calc_df["folder_in_root"].unique()) == 8

        calc_df, folders, subfolder = _find_calc_outputs(self.ZrO2_path)
        assert not calc_df.empty
        assert len(folders) == 8
        assert subfolder == "vasp_std"

        calc_df, folders, subfolder = _find_calc_outputs(self.ZrO2_path, subfolder="vasp_gam")
        assert not calc_df.empty
        assert subfolder == "vasp_gam"

        with tempfile.TemporaryDirectory() as empty_dir:
            calc_df, folders, subfolder = _find_calc_outputs(empty_dir)
            assert calc_df.empty
            assert folders == []
            assert subfolder == "."

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._build_mixed_subfolder_tree(tmp_dir)
            calc_df, folders, subfolder = _find_calc_outputs(tmp_dir)  # auto subfolder detection
            assert subfolder == "vasp_ncl"

    def test_latex_table(self):
        cpa = self.ZrO2_cpa

        def _test_latex_table(cpa=cpa, ref_filename="default.tex", **kwargs):
            return_str, stdout, w = _run_func_and_capture_stdout_warnings(cpa.to_LaTeX_table, **kwargs)
            assert not stdout
            assert not w

            with open(f"{data_dir}/ZrO2_LaTeX_Tables/test.tex", "w+") as f:
                f.write(return_str)

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

        _test_latex_table(self.La_ZrO2_cpa, "La_default.tex")

        with pytest.raises(ValueError):
            cpa.to_LaTeX_table(splits=3)

    def test_get_formation_energy_df(self):
        cpa = self.ZrO2_cpa

        def _check_ZrO2_form_e_df(
            form_e_df, skip_rounding=False, include_raw_energies=False, prune_polymorphs=False
        ):
            if prune_polymorphs:
                assert (
                    len(form_e_df) == 5
                )  # only ground states of each phase (including Zr2O with EaH > 0)

            assert form_e_df.index.to_numpy().tolist() == (
                self.ZrO2_entry_list if not prune_polymorphs else [*self.ZrO2_entry_list[:4], "Zr2O"]
            )
            space_groups = ["P2_1/c", "P6_3/mmc", "P4/mmm", "R-3c", "Pbca", "P6_322", "P312", "Ibam"]
            assert form_e_df["Space Group"].to_numpy().tolist() == (
                space_groups if not prune_polymorphs else [*space_groups[:4], "P312"]
            )
            assert np.allclose(form_e_df["Energy above Hull (eV/atom)"].to_numpy()[:4], 0)  # stable phases

            _check_form_e_df(cpa, form_e_df, skip_rounding, include_raw_energies, prune_polymorphs)

        for kwargs in [
            {},
            {"skip_rounding": True},
            {"include_raw_energies": True},
            {"skip_rounding": True, "include_raw_energies": True},
            {"prune_polymorphs": True},
            {"prune_polymorphs": True, "skip_rounding": True, "include_raw_energies": True},
        ]:
            _check_ZrO2_form_e_df(cpa.get_formation_energy_df(**kwargs), **kwargs)

        la_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            composition="ZrO2",
            entries=self.La_ZrO2_path,
        )
        la_form_e_df = la_cpa.get_formation_energy_df()
        assert len(la_form_e_df) == len(la_cpa.entries)
        assert la_form_e_df.index.to_numpy()[1] == "La"
        assert la_form_e_df.loc["La"].tolist() == ["P6_3/mmc", 0.0, 0.0, 0.0, "10x10x3"]
        assert la_form_e_df.iloc[4].tolist() == ["Ia-3", 0.0, -18.017, -3.603, "3x3x3"]  # La2O3
        assert la_form_e_df.iloc[6].tolist() == ["Fd-3m", 0.0, -40.877, -3.716, "3x3x3"]  # La2Zr2O7

    def test_repr(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path)
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
            entries=self.La_ZrO2_path,
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
            elif attr in ("vasprun_paths", "parsed_folders"):
                assert sorted(getattr(cpa_a, attr)) == sorted(getattr(cpa_b, attr))
            else:
                assert getattr(cpa_a, attr) == getattr(cpa_b, attr)

    def _general_cpa_check(self, cpa):
        intrinsic_el_refs = cpa.intrinsic_chempots["elemental_refs"]
        assert isinstance(next(iter(intrinsic_el_refs.keys())), str)
        for chempots_df in [cpa.chempots_df, cpa.calculate_chempots()]:
            for el_ref in intrinsic_el_refs:
                assert el_ref in chempots_df.columns

        _check_entries_dict_behaviour(cpa)  # test dict behaviour

        # test formation energy df:
        for kwargs in [
            {},
            {"skip_rounding": True},
            {"include_raw_energies": True},
            {"skip_rounding": True, "include_raw_energies": True},
            {"prune_polymorphs": True},
            {"prune_polymorphs": True, "skip_rounding": True, "include_raw_energies": True},
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
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path)
        _print_warning_info(w)  # for debugging
        assert not w
        self._general_cpa_check(cpa)

        with warnings.catch_warnings(record=True) as w:
            la_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.La_ZrO2_path)
        _print_warning_info(w)  # for debugging
        assert not w
        self._general_cpa_check(la_cpa)

    def test_mismatching_incar_warnings(self):
        """
        Test warnings for mismatching INCAR settings.

        No warnings for ZrO2 / La_ZrO2 already checked in ``self.test_general_cpa_reloading()`` above.
        """
        # convert to mismatching O2 calc:
        shutil.move(
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz",
        )
        shutil.move(
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/mismatching_incar_vr.xml.gz",
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path)
        _print_warning_info(w)  # for debugging
        expected_mismatching_info = [
            "There are mismatching INCAR tags",
            "['O2']:",
            "Where ZrO2 was used as the reference entry calculation.",
            "[('HFSCREEN', 0.20786986, 0.2), ('LREAL', 'Auto      ! projection operators: autom', False)]",
        ]
        assert all(any(i in str(warning.message) for warning in w) for i in expected_mismatching_info)
        self._general_cpa_check(cpa)

        # test no warning with check_compatibility=False:
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer(
                "ZrO2", self.ZrO2_path, check_compatibility=False
            )
        _print_warning_info(w)  # for debugging
        assert not w
        self._general_cpa_check(cpa)

        # test with extrinsic case:
        shutil.copyfile(
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",  # this is mismatching vr
            f"{self.La_ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            la_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.La_ZrO2_path)
        _print_warning_info(w)  # for debugging
        assert all(any(i in str(warning.message) for warning in w) for i in expected_mismatching_info)
        self._general_cpa_check(la_cpa)

        with warnings.catch_warnings(record=True) as w:
            MgO_cpa = chemical_potentials.CompetingPhasesAnalyzer("MgO", self.MgO_path)
        _print_warning_info(w)  # for debugging
        assert all(
            any(i in str(warning.message) for warning in w)
            for i in [
                "There are mismatching INCAR tags",
                "['Mg']:",
                "[('ENCUT', 585.0, 450.0)]",
                "Where MgO was used as the reference entry calculation.",
            ]
        )
        self._general_cpa_check(MgO_cpa)

    def test_mismatching_potcar_warnings(self):
        """
        Test warnings for mismatching POTCAR settings.

        No warnings for ZrO2 / La_ZrO2 already checked in ``self.test_general_cpa_reloading()`` above.
        """
        # convert to mismatching O2 calc, with fake "O_h" POTCAR:
        shutil.move(
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/orig_vr.xml.gz",
        )
        shutil.move(
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/mismatching_potcar_vr.xml.gz",
            f"{self.ZrO2_path}/O2_EaH_0.0/vasp_std/vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.ZrO2_path)
        _print_warning_info(w)  # for debugging
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
                "ZrO2", self.ZrO2_path, check_compatibility=False
            )
        _print_warning_info(w)  # for debugging
        assert not w
        self._general_cpa_check(cpa)

    def test_bulk_not_found(self):
        """
        Test case where bulk composition is not found in the supplied data.
        """
        with pytest.raises(ValueError) as exc:
            _cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", self.MgO_path)

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
            f"{vasp_data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0/vasprun.xml.gz",
            f"{vasp_data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0/duplicate_for_testing_vasprun.xml.gz",
        )
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer(
                "Cs2AgBiBr6", f"{vasp_data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases"
            )
        _print_warning_info(w)  # for debugging
        for expected_warning in [
            f"Multiple `vasprun.xml` files found in competing phase directory: "
            f"{vasp_data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Br_EaH=0",
            f"vasprun.xml file at {vasp_data_dir}/Sn_in_Cs2AgBiBr6_CompetingPhases/Bi_EaH=0/vasprun.xml.gz"
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
    cpa, form_e_df, skip_rounding=False, include_raw_energies=False, prune_polymorphs=False
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
        if include_raw_energies:
            assert np.isclose(
                series["Raw Energy (eV/fu)"],
                series["Raw Energy (eV/atom)"] * comp.num_atoms,
                atol=2e-3,
                rtol=1e-3,
            )

    assert ("Raw Energy (eV/fu)" in form_e_df.columns) == include_raw_energies
    assert ("Raw Energy (eV/atom)" in form_e_df.columns) == include_raw_energies

    # assert values are all rounded to 3 dp:
    assert form_e_df.round(3).equals(form_e_df) == (not skip_rounding)


class TestChemicalPotentialGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chempots = loadfn(os.path.join(EXAMPLE_DIR, "Cu2SiSe3/Cu2SiSe3_chempots.json"))
        cls.grid = chemical_potentials.ChemicalPotentialGrid(cls.chempots)
        cls.na2fepo4f_cp = chemical_potentials.CompetingPhases("Na2FePO4F", api_key=api_key)
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
        cls.ZrO2_path = os.path.join(EXAMPLE_DIR, "ZrO2_CompetingPhases")
        cls.ZrO2_cpa = chemical_potentials.CompetingPhasesAnalyzer("ZrO2", cls.ZrO2_path)
        # Note we also have Cs2SnBr6 competing phase energies csv in JOSS data folder if needed for tests

        # CuSe2 + Ge: the four `single_extrinsic_phase_limits` modes give visibly different chempots and
        # heatmaps; cache the two ``CompetingPhases`` queries so the four tests below share them rather
        # than re-querying MP each time:
        cls.CuSe2_Ge_cp = chemical_potentials.CompetingPhases(
            "CuSe2", energy_above_hull=0, extrinsic="Ge", api_key=api_key
        )
        cls.CuSe2_Ge_cp_single = chemical_potentials.CompetingPhases(
            "CuSe2",
            energy_above_hull=0,
            extrinsic="Ge",
            single_extrinsic_phase_limits=True,
            api_key=api_key,
        )

        # BaSnO3 + K, In: cache the two ``CompetingPhases`` queries (default and ``codoping=True``)
        # shared by the 2x2 codoping/``single_extrinsic_phase_limits`` heatmap tests below:
        cls.BaSnO3_K_In_cp = chemical_potentials.CompetingPhases(
            "BaSnO3", energy_above_hull=0, extrinsic=["K", "In"], api_key=api_key
        )
        cls.BaSnO3_K_In_codoping_cp = chemical_potentials.CompetingPhases(
            "BaSnO3", energy_above_hull=0, extrinsic=["K", "In"], codoping=True, api_key=api_key
        )

    def tearDown(self):
        if_present_rm("test.png")
        if_present_rm("cpg.json")

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
            assert np.isclose(np.mean(grid_df["μ_Cu (eV)"]), -0.19661, atol=1e-3 if cart else 2e-2)
            assert np.isclose(np.mean(grid_df["μ_Si (eV)"]), -0.94969, atol=1e-3 if cart else 2e-1)
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

    def test_chempot_heatmap_5D_w_fixed_elements_error_wrong_element(self):
        with pytest.raises(ValueError) as exc:
            self.Sn_in_Cs2AgBiBr6_ncl_cpa.plot_chempot_heatmap(fixed_elements={"Cd": -0.5})
        assert "Cd (eV)' is not in list" in str(exc.value)

    def test_chempot_heatmap_5D_w_fixed_elements_error(self):
        # ``Sn_in_Cs2AgBiBr6_ncl_cpa`` is 5-D (Cs, Ag, Bi, Br, Sn); fixing 3 elements leaves only
        # 2-D and so should trigger the dimensionality error:
        with pytest.raises(ValueError) as exc:
            self.Sn_in_Cs2AgBiBr6_ncl_cpa.plot_chempot_heatmap(
                fixed_elements={"Cs": -0.5, "Ag": -0.5, "Sn": 0.0}
            )
        assert "Chemical potential heatmap plotting requires 3-D data" in str(exc.value)
        assert "(5) minus the number of fixed chemical potentials (3)" in str(exc.value)

    def test_chempot_heatmap_5D_w_fixed_elements_outside_range(self):
        with pytest.raises(ValueError) as exc:
            self.Sn_in_Cs2AgBiBr6_ncl_cpa.plot_chempot_heatmap(fixed_elements={"Ag": -25, "Sn": 0.0})
        assert (
            "The input set of fixed chemical potentials does not intersect with the convex hull (i.e. "
            "stable chemical potential range) of the host material." in str(exc.value)
        )

    def test_chempot_heatmap_2D_error(self):
        with pytest.raises(ValueError) as exc:  # this will likely change with updated code
            self.ZrO2_cpa.plot_chempot_heatmap()
        assert (
            "Chemical potential heatmap (i.e. 2D) plotting is not possible for a binary system! You "
            "can use ``cpd = ChemicalPotentialDiagram(cpa.entries); cpd.get_plot()`` to generate a "
            "line plot of the chemical potentials as shown in the doped competing phases tutorial."
            in str(exc.value)
        )

    @custom_mpl_image_compare(filename="AgSbTe2_chempot_heatmap_default.png")
    def test_AgSbTe2_chempot_heatmap_default(self):
        return plot_chempot_heatmap_and_test_no_warnings(self.AgSbTe2_cpa)

    @custom_mpl_image_compare(
        filename="AgSbTe2_chempot_heatmap_custom.png",
        style=f"{module_path}/../doped/utils/displacement.mplstyle",
    )
    def test_AgSbTe2_chempot_heatmap_custom(self):
        plot = plot_chempot_heatmap_and_test_no_warnings(
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

    @custom_mpl_image_compare(
        filename="AgSbTe2_chempot_heatmap_custom.png",
        style=f"{module_path}/../doped/utils/displacement.mplstyle",
    )
    def test_AgSbTe2_chempot_heatmap_custom_w_direct_function(self):
        plot, output, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.plot_chempot_heatmap,
            self.AgSbTe2_cpa.chempots,
            composition="AgSbTe2",
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
        assert not w
        assert not output
        assert os.path.exists("test.png")
        return plot

    @custom_mpl_image_compare(filename="LiPS4_chempot_heatmap_default.png")
    def test_LiPS4_chempot_heatmap_default(self):
        return plot_chempot_heatmap_and_test_no_warnings(self.LiPS4_cpa)

    @custom_mpl_image_compare(filename="LiPS4_chempot_heatmap_custom.png")
    def test_LiPS4_chempot_heatmap_custom(self):
        return plot_chempot_heatmap_and_test_no_warnings(
            self.LiPS4_cpa,
            dependent_element="Li",
            padding=0.1,
            title=False,
            label_positions=True,
        )

    @custom_mpl_image_compare(filename="Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_default.png")
    def test_Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_default(self):
        # Sn is fixed at its elemental ref (= 0) so the plot reduces to the intrinsic 4-D system,
        # for which this baseline was generated:
        return plot_chempot_heatmap_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_ncl_cpa, fixed_elements={"Cs": -3.3815, "Sn": 0.0}
        )

    @custom_mpl_image_compare(
        filename="Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_custom.png",
        style=f"{module_path}/../doped/utils/displacement.mplstyle",
    )
    def test_Sn_in_Cs2AgBiBr6_ncl_chempot_heatmap_custom(self):
        """
        Test customising the heatmap for an extrinsic system (with the
        extrinsic chempot fixed at its elemental ref so the plot reduces to the
        intrinsic 4-D host stability region), with custom label positions.

        Same example used in the plotting customisation tutorial.
        """
        return plot_chempot_heatmap_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_ncl_cpa,
            fixed_elements={"Cs": -3.3815, "Sn": 0.0},
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
        Test the auto-centroid path with a non-default ``dependent_element``;
        Sn fixed at its elemental ref reduces this to the intrinsic 4-D system,
        with one host element auto-fixed to its centroid value.
        """
        return plot_chempot_heatmap_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_ncl_cpa,
            dependent_element="Cs",
            fixed_elements={"Sn": 0.0},
        )

    @custom_mpl_image_compare(filename="Sn_in_Cs2AgBiBr6_std_chempot_heatmap_custom.png")
    def test_Sn_in_Cs2AgBiBr6_std_chempot_heatmap_custom(self):
        return plot_chempot_heatmap_and_test_no_warnings(
            self.Sn_in_Cs2AgBiBr6_std_cpa,
            fixed_elements={"Cs": -3.3815, "Sn": 0.0},
            xlim=(-0.45, 0),
            ylim=(-2.35, -0.9),
            cbar_range=(-0.57, -0.3),
            label_positions=False,
        )

    @custom_mpl_image_compare(filename="CdTe_Cs_extrinsic_chempot_heatmap.png")
    def test_CdTe_Cs_extrinsic_chempot_heatmap(self):
        """
        Test heatmap plotting for a binary host with an extrinsic species (CdTe
        + Cs).
        """
        cp = chemical_potentials.CompetingPhases(
            "CdTe ",
            energy_above_hull=0,
            extrinsic=["Cs"],
        )
        cpa = chemical_potentials.CompetingPhasesAnalyzer("CdTe", cp.entries)
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap()
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="BaSnO3_K_In_extrinsic_chempot_heatmap_K_fixed.png")
    def test_BaSnO3_K_In_extrinsic_chempot_heatmap(self):
        """
        Test heatmap plotting for a ternary host with two extrinsic species
        (BaSnO3 + K + In; 5-D system). The default ``dependent_element`` falls
        back to the most electronegative host element (O), and the remaining
        two dimensions are auto-fixed at the centroid of the chemical stability
        region.

        ``KInO2`` is a co-doping competing phase that is present but is skipped
        under the single-extrinsic-phase-limits approximation.
        """
        cp = chemical_potentials.CompetingPhases(
            "BaSnO3", energy_above_hull=0.03, extrinsic=["K", "In"], api_key=api_key, codoping=True
        )
        with warnings.catch_warnings(record=True) as w:
            cpa = chemical_potentials.CompetingPhasesAnalyzer("BaSnO3", cp.entries)
        _print_warning_info(w)
        assert not w
        return plot_chempot_heatmap_and_test_no_warnings(cpa, fixed_elements={"K": -2.0})

    @custom_mpl_image_compare(filename="CuSe2_Ge_extrinsic_chempot_heatmap.png")
    def test_CuSe2_Ge_extrinsic_chempot_heatmap(self):
        """
        CuSe2 + Ge with the default (``single_extrinsic_phase_limits=False``
        everywhere): gives 4 chempot limits.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer("CuSe2", self.CuSe2_Ge_cp.entries)
        assert set(cpa.chempots_df.columns) == {"Cu", "Se", "Ge"}
        assert len(cpa.chempots_df) == 4
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap()
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CuSe2_Ge_extrinsic_chempot_heatmap_cpa_single_extrinsic.png")
    def test_CuSe2_Ge_extrinsic_chempot_heatmap_cpa_single_extrinsic(self):
        """
        CuSe2 + Ge with default ``CompetingPhases``, then parsing with
        ``CompetingPhasesAnalyzer(single_extrinsic_phase_limits=True)``, gives
        2 chempot limits.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "CuSe2", self.CuSe2_Ge_cp.entries, single_extrinsic_phase_limits=True
        )
        assert set(cpa.chempots_df.columns) == {"Cu", "Se", "Ge"}
        assert len(cpa.chempots_df) == 2
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap()
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CuSe2_Ge_extrinsic_chempot_heatmap_cpa_single_extrinsic.png")
    def test_CuSe2_Ge_extrinsic_chempot_heatmap_cp_single(self):
        """
        CuSe2 + Ge with ``single_extrinsic_phase_limits=True`` at both
        ``CompetingPhases`` and ``CompetingPhasesAnalyzer`` init: 2 chempot
        limits, matching
        ``test_CuSe2_Ge_extrinsic_chempot_heatmap_cpa_single_extrinsic`` (the
        entry pruning at CP construction is consistent with the single-phase-
        limit filter at CPA).
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "CuSe2", self.CuSe2_Ge_cp_single.entries, single_extrinsic_phase_limits=True
        )
        assert set(cpa.chempots_df.columns) == {"Cu", "Se", "Ge"}
        assert len(cpa.chempots_df) == 2
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap()
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CuSe2_Ge_extrinsic_chempot_heatmap_cpa_single_extrinsic.png")
    def test_CuSe2_Ge_extrinsic_chempot_heatmap_cp_single_cpa_default(self):
        """
        CuSe2 + Ge with ``single_extrinsic_phase_limits=True`` at
        ``CompetingPhases`` initialisation (entries pruned to single-extrinsic-
        phase candidates), but not at parsing with ``CompetingPhasesAnalyzer``,
        so we still get the limit at the `intersection` of ``Cu2GeSe3`` and
        ``Ge4Se9`` (``"Cu2GeSe3-CuSe2-Ge4Se9"``), giving 3 chempot limits.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer("CuSe2", self.CuSe2_Ge_cp_single.entries)
        assert set(cpa.chempots_df.columns) == {"Cu", "Se", "Ge"}
        assert len(cpa.chempots_df) == 3
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap()
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="BaSnO3_K_In_extrinsic_chempot_heatmap_default.png")
    def test_BaSnO3_K_In_extrinsic_chempot_heatmap_default(self):
        """
        BaSnO3 + K, In with default ``CompetingPhases`` (no codoping) and
        default ``CompetingPhasesAnalyzer``: 18 chempot limits.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer("BaSnO3", self.BaSnO3_K_In_cp.entries)
        assert set(cpa.chempots_df.columns) == {"Ba", "Sn", "O", "K", "In"}
        assert len(cpa.chempots_df) == 18
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap(fixed_elements={"In": -1})
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="BaSnO3_K_In_extrinsic_chempot_heatmap_default.png")
    def test_BaSnO3_K_In_extrinsic_chempot_heatmap_cpa_single(self):
        """
        BaSnO3 + K, In with default ``CompetingPhases`` (no codoping) and
        ``single_extrinsic_phase_limits=True`` at ``CompetingPhasesAnalyzer``.

        ``μ_host`` pinned at intrinsic limits, with only single-extrinsic-phase
        facets retained — gives fewer limits than the default-mode case.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "BaSnO3", self.BaSnO3_K_In_cp.entries, single_extrinsic_phase_limits=True
        )
        assert set(cpa.chempots_df.columns) == {"Ba", "Sn", "O", "K", "In"}
        # single-phase parsing prunes joint-extrinsic facets, so fewer than the 18 default-mode limits:
        assert 1 <= len(cpa.chempots_df) < 18
        assert len(cpa.chempots_df) == 7
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap(fixed_elements={"In": -1})
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="BaSnO3_K_In_extrinsic_chempot_heatmap_codoping.png")
    def test_BaSnO3_K_In_extrinsic_chempot_heatmap_codoping(self):
        """
        BaSnO3 + K, In with ``codoping=True`` at ``CompetingPhases`` (adds
        joint K-In phases like ``KInO2``) and default
        ``CompetingPhasesAnalyzer``: 25 chempot limits, including codoping-
        specific facets.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer("BaSnO3", self.BaSnO3_K_In_codoping_cp.entries)
        assert set(cpa.chempots_df.columns) == {"Ba", "Sn", "O", "K", "In"}
        assert len(cpa.chempots_df) == 25
        # codoping-specific limits (those including ``KInO2``) only appear with codoping=True:
        assert any("KInO2" in limit for limit in cpa.chempots_df.index)
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap(fixed_elements={"In": -1})
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="BaSnO3_K_In_extrinsic_chempot_heatmap_default.png")
    def test_BaSnO3_K_In_extrinsic_chempot_heatmap_codoping_cpa_single(self):
        """
        BaSnO3 + K, In with ``codoping=True`` at ``CompetingPhases`` and
        ``single_extrinsic_phase_limits=True`` at ``CompetingPhasesAnalyzer``.

        The single-phase parser prunes the joint K-In phases that codoping
        added, so this gives the same results as
        ``single_extrinsic_phase_limits=True`` with no co-doping.
        """
        cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "BaSnO3",
            self.BaSnO3_K_In_codoping_cp.entries,
            single_extrinsic_phase_limits=True,
        )
        assert set(cpa.chempots_df.columns) == {"Ba", "Sn", "O", "K", "In"}
        # codoping-only facets should be pruned by single-phase parsing:
        assert not any("KInO2" in limit for limit in cpa.chempots_df.index)
        assert len(cpa.chempots_df) == 7
        with warnings.catch_warnings(record=True) as w:
            fig = cpa.plot_chempot_heatmap(fixed_elements={"In": -1})
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_heatmap.png")
    def test_5D_fixed_elements_heatmap(self):
        na2fepo4f_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "Na2FePO4F", entries=self.na2fepo4f_cp.entries
        )
        return plot_chempot_heatmap_and_test_no_warnings(
            na2fepo4f_cpa, fixed_elements={"Na": -1.9, "P": -1.3}
        )

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_heatmap_no_bordering.png")
    def test_no_bordering_heatmap(self):
        na2fepo4f_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            "Na2FePO4F", entries=self.na2fepo4f_cp.entries
        )
        return plot_chempot_heatmap_and_test_no_warnings(
            na2fepo4f_cpa, fixed_elements={"Na": -1.9, "P": -1.3}, bordering_phases=False
        )

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_grid.png")
    def test_Na2FePO4F_chempot_grid(self):
        """
        Test |ChemicalPotentialGrid| generation and plotting for a complex
        quinary system (Na2FePO4F).
        """
        grid_df = self.na2fepo4f_grid.get_grid(1e8, drop_duplicates=False)
        return _plot_Na2FePO4F_chempot_grid(grid_df, atol=0.01)

    @custom_mpl_image_compare(filename="Na2FePO4F_chempot_grid_cartesian.png")
    def test_Na2FePO4F_chempot_grid_cartesian(self):
        """
        Test |ChemicalPotentialGrid| generation and plotting for a complex
        quinary system (Na2FePO4F).
        """
        grid_df = self.na2fepo4f_grid.get_grid(2e5, cartesian=True)
        return _plot_Na2FePO4F_chempot_grid(grid_df)

    def test_to_from_dict(self):
        chempots = loadfn(os.path.join(EXAMPLE_DIR, "Cu2SiSe3/Cu2SiSe3_chempots.json"))
        grid = chemical_potentials.ChemicalPotentialGrid(chempots)

        grid_dict = grid.as_dict()
        grid_from_dict = chemical_potentials.ChemicalPotentialGrid.from_dict(grid_dict)
        assert grid.vertices.equals(grid_from_dict.vertices)

        dumpfn(grid_dict, "cpg.json")
        reloaded_grid = loadfn("cpg.json")
        assert isinstance(reloaded_grid, chemical_potentials.ChemicalPotentialGrid)
        assert grid.vertices.equals(reloaded_grid.vertices)


def _plot_Na2FePO4F_chempot_grid(grid_df, atol=0.05):
    # get the average Fe and P chempots, then plot a heatmap plot of the others at these fixed values:
    middle_mu_Fe = (
        grid_df["μ_Fe (eV)"].min() + (grid_df["μ_Fe (eV)"].max() - grid_df["μ_Fe (eV)"].min()) / 2
    )
    middle_mu_P = grid_df["μ_P (eV)"].min() + (grid_df["μ_P (eV)"].max() - grid_df["μ_P (eV)"].min()) / 2

    fixed_chempot_df = grid_df[
        (np.isclose(grid_df["μ_Fe (eV)"], middle_mu_Fe, atol=atol))
        & (np.isclose(grid_df["μ_P (eV)"], middle_mu_P, atol=atol))
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


class TestGetXRichPoorLimit(unittest.TestCase):
    """
    Test ``get_X_rich_poor_limit``, particularly behaviour when several facets
    share the same μ_X extremum.
    """

    def test_rich_tie_first_refinement_max_mu_other(self):
        chempots = {
            "limits": {
                "A": {"Cu": -1.0, "O": -5.0},
                "B": {"Cu": -1.0, "O": -4.0},
            },  # Cu-rich tie: falls back to max μ_O -> B (more O-rich)
            "elemental_refs": {},
            "limits_wrt_el_refs": {},
        }  # typical chempots dict structure
        assert (
            chemical_potentials.get_X_rich_poor_limit(
                "Cu", chempots, bulk_composition="Cu2O", warn_if_multiple=False
            )
            == "B"
        )

    def test_rich_tie_first_refinement_max_mu_other_w_slight_diff(self):
        chempots = {
            "limits": {
                "A": {"Cu": -1.002, "O": -5.0},
                "B": {"Cu": -1.0, "O": -4.0},
            }
        }  # Cu-rich tie (slightly different but within default `tol`): falls back to max μ_O -> B
        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit,
            "Cu",
            chempots,
            bulk_composition="Cu2O",
            warn_if_multiple=True,
        )
        assert result == "B"
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)

    def test_poor_tie_first_refinement_min_mu_other(self):
        chempots = {
            "limits": {
                "A": {"Cu": -5.0, "O": -2.0},
                "B": {"Cu": -5.0, "O": -3.0},
            },
        }
        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit,
            "Cu",
            chempots,
            rich=False,
            bulk_composition="Cu2O",  # warn by default
        )
        assert result == "B"
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)

    def test_no_tie_same_as_simple_extremum(self):
        chempots = {
            "limits": {
                "Cd-CdTe": {"Cd": -0.5, "Te": -1.5},
                "CdTe-Te": {"Cd": -2.0, "Te": -1.0},
            },
        }
        with warnings.catch_warnings(record=True) as w:
            assert chemical_potentials.get_X_rich_poor_limit("Te", chempots) == "CdTe-Te"
            assert chemical_potentials.get_X_rich_poor_limit("Te", chempots, rich=False) == "Cd-CdTe"

        _print_warning_info(w)  # for debugging
        assert not w

    def test_tie_w_manual_tol(self):
        chempots = {  # same chempots dict as test above, but now with tol > 0.5 eV
            "limits": {
                "Cd-CdTe": {"Cd": -0.5, "Te": -1.5},
                "CdTe-Te": {"Cd": -2.0, "Te": -1.0},
            },
        }
        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit, "Te", chempots, tol=0.6
        )
        assert result == "Cd-CdTe"
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)

        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit, "Te-poor", chempots, tol=0.6
        )
        assert result == "CdTe-Te"
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)

    def test_rich_poor_string_input_overrides_rich(self):
        chempots = {
            "limits": {
                "Cd-CdTe": {"Cd": -0.5, "Te": -3.0},
                "CdTe-Te": {"Cd": -2.0, "Te": -1.0},
            },
        }
        with warnings.catch_warnings(record=True) as w:
            # "X-rich"/"X-poor" string should override the ``rich`` kwarg:
            assert chemical_potentials.get_X_rich_poor_limit("Te-rich", chempots, rich=False) == "CdTe-Te"
            assert chemical_potentials.get_X_rich_poor_limit("Te-poor", chempots, rich=True) == "Cd-CdTe"

        _print_warning_info(w)  # for debugging
        assert not w

    def test_invalid_X_raises(self):
        chempots = {"limits": {"A": {"Cu": -1.0}}, "elemental_refs": {}, "limits_wrt_el_refs": {}}
        with pytest.raises(ValueError, match="Invalid input for X"):
            chemical_potentials.get_X_rich_poor_limit("NotAnElement", chempots)

    def test_bulk_first_in_tiebreak_order(self):
        # Same μ_Cu; bulk Cu2O -> intrinsic O before Mn (despite Mn being more electronegatively-similar);
        # first refinement uses max μ_O -> Cu2O-MnO3 here
        chempots = {
            "limits": {  # hypothetical examples
                "Cu2O-MnO2": {"Cu": -1.0, "O": -5.0, "Mn": -1.0},
                "Cu2O-MnO3": {"Cu": -1.0, "O": -4.0, "Mn": -10.0},
            },
        }
        with warnings.catch_warnings(record=True) as w:
            lim = chemical_potentials.get_X_rich_poor_limit(
                "Cu", chempots, bulk_composition="Cu2O", warn_if_multiple=False
            )
        _print_warning_info(w)  # for debugging
        assert not w
        assert lim == "Cu2O-MnO3"

    def test_element_ordering_CuZnMn(self):
        """
        In ``get_X_rich_poor_limit`` we auto-detect the bulk composition as the
        phase appearing in every limit name.

        Here "Cu" is in both (elemental host composition), "Mn"/"Zn" are not
        (extrinsic). In the tie on μ_Cu with no other intrinsic element, the
        ordering then falls to extrinsic elements sorted by electronegativity
        similarity to Cu. Cu(1.90), Mn(1.55), Zn(1.65) -> Zn is more
        electronegatively-similar to Cu than Mn, so Zn is considered first; the
        max μ_Zn among tied limits lives in "Cu-Zn" here.
        """
        chempots = {
            "limits": {
                "Cu-Mn": {"Cu": -1.0, "Mn": -0.5, "Zn": -5.0},
                "Cu-Zn": {"Cu": -1.0, "Mn": -5.0, "Zn": -0.5},
            },
        }
        assert chemical_potentials.get_X_rich_poor_limit("Cu", chempots) == "Cu-Zn"

    def test_element_ordering_AgBiS2(self):
        """
        Test that in a tie-break with a multi-cation composition, A-rich (Ag-
        rich) falls back to the most B-rich (where A and B are the cations; Bi-
        rich here) option of the A-rich options.
        """
        chempots = {
            "limits": {
                "AgBiS2-Ag2S-Bi2S3": {"Ag": -1.0, "Bi": -0.5, "S": -5.0},
                "AgBiS2-AgS2-Bi5S7": {"Ag": -1.005, "Bi": -5.0, "S": -0.5},
            },
        }
        assert chemical_potentials.get_X_rich_poor_limit("Ag", chempots) == "AgBiS2-Ag2S-Bi2S3"

    def test_lexicographic_fallback_tiny_range(self):
        # All μ values tied to within tol for every element -> must fall through to the
        # lexicographic ``max``/``min`` fallback at the end of the function.
        chempots = {
            "limits": {
                "A": {"Cu": -1.0, "O": -1.0},
                "B": {"Cu": -1.0, "O": -1.0},
            },
        }
        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit,
            "Cu",
            chempots,
            bulk_composition="Cu2O",
        )
        assert result == "B"  # max between A and B
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)

        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit,
            "Cu",
            chempots,
            rich=False,
            bulk_composition="Cu2O",
        )
        assert result == "A"  # min between A and B
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)

    def test_raises_missing_element(self):
        chempots = {"limits": {"A": {"O": -1.0}}, "elemental_refs": {}, "limits_wrt_el_refs": {}}
        with pytest.raises(ValueError, match="Could not find Cu"):
            chemical_potentials.get_X_rich_poor_limit("Cu", chempots)

    @pytest.mark.filterwarnings("always::DeprecationWarning")  # deliberate deprecated-API test
    def test_deprecated_aliases_warn_and_forward(self):
        chempots = {
            "limits": {
                "Cd-CdTe": {"Cd": -0.5, "Te": -3.0},
                "CdTe-Te": {"Cd": -2.0, "Te": -1.0},
            },
        }
        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_limit, "Te", chempots
        )
        assert result == "CdTe-Te"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "get_X_rich_limit" in str(w[0].message)
        assert "deprecated" in str(w[0].message)

        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_poor_limit, "Te", chempots
        )
        assert result == "Cd-CdTe"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "get_X_poor_limit" in str(w[0].message)
        assert "deprecated" in str(w[0].message)


class TestSb2Si2Te6Chempots(unittest.TestCase):
    """
    Build a hypothetical ``Sb2Si2Te6`` chempots dict (via the MP Sb-Si-Te
    entries and a below-hull bulk entry) and exercise the downstream
    ``plot_chempot_heatmap`` and ``get_X_rich_poor_limit`` code paths.

    The heatmap plots previously failed for this system because the stability
    region is a simple simplex (a single triangular domain with only three
    vertices, which causes Delaunay triangulation to fail) that the old
    grid/labelling code did not handle natively.
    """

    @classmethod
    def setUpClass(cls):
        Sb_Si_Te_entries = chemical_potentials.get_entries_in_chemsys("Sb-Si-Te", api_key=api_key)
        phase_diagram = chemical_potentials.PhaseDiagram(Sb_Si_Te_entries)
        # fake bulk entry 50 meV below the convex hull so SbSiTe3 becomes
        # stable and we can extract its chempot limits:
        bulk_entry = ComputedEntry(
            Composition("Sb2Si2Te6"),
            phase_diagram.get_hull_energy(Composition("Sb2Si2Te6")) - 0.05,
            data={
                "energy_above_hull": 0.0,
                "material_id": "mp-0",
                "molecule": False,
                "summary": {
                    "band_gap": None,
                    "total_magnetization": None,
                    "database_IDs": {},
                },
            },
        )
        cls.chempots = chemical_potentials.get_doped_chempots_from_entries(
            [bulk_entry, *Sb_Si_Te_entries], Composition("Sb2Si2Te6")
        )

    def _plot_heatmap_no_warnings(self, **kwargs):
        # simple-simplex stability region: previously failed due to the grid/
        # labelling code not handling single-triangle domains natively.
        with warnings.catch_warnings(record=True) as w:
            fig = chemical_potentials.plot_chempot_heatmap(
                self.chempots, composition="Sb2Si2Te6", **kwargs
            )
        _print_warning_info(w)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="Sb2Si2Te6_chempot_heatmap_default.png")
    def test_Sb2Si2Te6_chempot_heatmap_default(self):
        return self._plot_heatmap_no_warnings()

    @custom_mpl_image_compare(filename="Sb2Si2Te6_chempot_heatmap_default.png")
    def test_Sb2Si2Te6_chempot_heatmap_cartesian(self):
        """
        Should give same plot, just testing Cartesian grid generation.
        """
        return self._plot_heatmap_no_warnings(cartesian=True)

    def test_Si_rich_limit_degeneracy(self):
        """
        Si-rich is degenerate between ``SbTe2-SiSbTe3-Si`` and
        ``SiSbTe3-SiTe2-Si`` (both at μ_Si = 0).

        The tie-break sorts the remaining bulk elements by electronegativity
        similarity to Si (χ=1.90): Sb (χ=2.05, Δ=0.15) is closer than Te
        (χ=2.10, Δ=0.20), so Sb is considered first, and the most Sb-rich
        tied limit (``SbTe2-SiSbTe3-Si``, μ_Sb ≈ -0.408) wins over
        ``SiSbTe3-SiTe2-Si`` (μ_Sb ≈ -0.483).
        """
        # chempots_df = pd.DataFrame.from_dict(self.chempots["limits_wrt_el_refs"], orient="index")
        # print("\nSb2Si2Te6 chempots (wrt elemental refs):")
        # print(chempots_df)  # for debugging/checking
        result, _stdout, w = _run_func_and_capture_stdout_warnings(
            chemical_potentials.get_X_rich_poor_limit, "Si-rich", self.chempots
        )
        assert result == "SbTe2-SiSbTe3-Si"
        assert len(w) == 1
        assert "Multiple chemical potential limits are degenerate" in str(w[0].message)
