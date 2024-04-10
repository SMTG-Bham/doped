"""
Tests for the `doped.chemical_potentials` module.
"""

import os
import shutil
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from test_analysis import if_present_rm

from doped import chemical_potentials


def _compare_chempot_dicts(dict1, dict2):
    for key, val in dict1.items():
        if isinstance(val, dict):
            _compare_chempot_dicts(val, dict2[key])
        else:
            assert np.isclose(val, dict2[key], atol=1e-5)


class ChemPotsTestCase(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).parents[1].joinpath("examples/competing_phases")
        self.stable_system = "ZrO2"
        self.unstable_system = "Zr2O"
        self.extrinsic_species = "La"
        self.csv_path = self.path / "zro2_competing_phase_energies.csv"
        self.csv_path_ext = self.path / "zro2_la_competing_phase_energies.csv"
        self.parsed_chempots = loadfn(self.path / "zro2_chempots.json")
        self.parsed_ext_chempots = loadfn(self.path / "zro2_la_chempots.json")

    def tearDown(self) -> None:
        if Path("chempot_limits.csv").is_file():
            os.remove("chempot_limits.csv")

        if Path("competing_phases.csv").is_file():
            os.remove("competing_phases.csv")

        if Path("input.dat").is_file():
            os.remove("input.dat")

        for i in os.listdir(self.path / "ZrO2"):
            if i.startswith("."):
                if_present_rm(self.path / "ZrO2" / i)

        return super().tearDown()

    def test_cpa_csv(self):
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)

        assert len(stable_cpa.elemental) == 2
        assert len(self.ext_cpa.elemental) == 3
        assert any(entry["Formula"] == "O2" for entry in stable_cpa.data)
        assert np.isclose(
            next(
                entry["DFT Energy (eV/fu)"]
                for entry in self.ext_cpa.data
                if entry["Formula"] == "La2Zr2O7"
            ),
            -119.619571095,
        )

    # test chempots
    def test_cpa_chempots(self):
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        chempot_df = stable_cpa.calculate_chempots()
        assert next(iter(chempot_df["O"])) == 0
        # check if it's no longer Element
        assert isinstance(next(iter(stable_cpa.intrinsic_chempots["elemental_refs"].keys())), str)

        self.unstable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.unstable_system)
        self.unstable_cpa.from_csv(self.csv_path)
        with pytest.raises(ValueError):
            self.unstable_cpa.calculate_chempots()

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        chempot_df = self.ext_cpa.calculate_chempots()
        assert next(iter(chempot_df["La-Limiting Phase"])) == "La2Zr2O7"
        assert np.isclose(next(iter(chempot_df["La"])), -9.46298748)

    def test_ext_cpa_chempots(self):
        # test accessing cpa.chempots without previously calling cpa.calculate_chempots()
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        _compare_chempot_dicts(stable_cpa.chempots, self.parsed_chempots)

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        assert self.ext_cpa.chempots["elemental_refs"] == self.parsed_ext_chempots["elemental_refs"]

    def test_sort_by(self):
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        chempot_df = stable_cpa.calculate_chempots(sort_by="Zr")
        assert np.isclose(next(iter(chempot_df["Zr"])), -0.199544, atol=1e-4)
        assert np.isclose(list(chempot_df["Zr"])[1], -10.975428439999998, atol=1e-4)

        with pytest.raises(KeyError):
            stable_cpa.calculate_chempots(sort_by="M")

    # test vaspruns
    def test_vaspruns(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"
        cpa.from_vaspruns(path=path, folder="relax", csv_path=self.csv_path)
        assert len(cpa.elemental) == 2
        assert cpa.data[0]["Formula"] == "O2"

        cpa_no = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        with pytest.raises(FileNotFoundError):
            cpa_no.from_vaspruns(path="path")

        with pytest.raises(ValueError):
            cpa_no.from_vaspruns(path=0)

        ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        path2 = self.path / "La_ZrO2"
        ext_cpa.from_vaspruns(path=path2, folder="relax", csv_path=self.csv_path_ext)
        assert len(ext_cpa.elemental) == 3
        # sorted by num_species, then alphabetically, then by num_atoms_in_fu, then by
        # formation_energy
        assert [entry["Formula"] for entry in ext_cpa.data] == [
            "La",
            "O2",
            "Zr",
            "Zr",
            "La2O3",
            "Zr2O",
            "Zr3O",
            "Zr3O",
            "ZrO2",
            "ZrO2",
            "La2Zr2O7",
        ]
        # spot check some values
        # [{'formula': 'La', 'kpoints': '10x10x3', 'energy_per_fu': -5.00458616,
        # 'energy_per_atom': -5.00458616, 'energy': -20.01834464, 'formation_energy': 0.0},
        # {'formula': 'O2', 'kpoints': '2x2x2', 'energy_per_fu': -14.01320413, 'energy_per_atom':
        # -7.006602065, 'energy': -14.01320413, 'formation_energy': 0.0}, {'formula': 'Zr',
        # 'kpoints': '9x9x5', 'energy_per_fu': -9.84367624, 'energy_per_atom': -9.84367624,
        # 'energy': -19.68735248, 'formation_energy': 0.0}, {'formula': 'Zr', 'kpoints': '6x6x6',
        # 'energy_per_fu': -9.818516226666667, 'energy_per_atom': -9.818516226666667, 'energy':
        # -58.91109736, 'formation_energy': 0.025160013333334064}, {'formula': 'La2O3',
        # 'kpoints': '3x3x3', 'energy_per_fu': -49.046189555, 'energy_per_atom': -9.809237911,
        # 'energy': -392.36951644, 'formation_energy': -18.01721104}, {'formula': 'Zr2O',
        # 'kpoints': '5x5x2', 'energy_per_fu': -32.42291351666667, 'energy_per_atom':
        # -10.807637838888889, 'energy': -194.5374811, 'formation_energy': -5.728958971666668},
        # {'formula': 'Zr3O', 'kpoints': '5x5x5', 'energy_per_fu': -42.524204305,
        # 'energy_per_atom': -10.63105107625, 'energy': -85.04840861, 'formation_energy':
        # -5.986573519999993}, {'formula': 'Zr3O', 'kpoints': '5x5x5', 'energy_per_fu':
        # -42.472744875, 'energy_per_atom': -10.61818621875, 'energy': -84.94548975,
        # 'formation_energy': -5.935114089999992}, {'formula': 'ZrO2', 'kpoints': '3x3x3',
        # 'energy_per_fu': -34.83230881, 'energy_per_atom': -11.610769603333333, 'energy':
        # -139.32923524, 'formation_energy': -10.975428440000002}, {'formula': 'ZrO2', 'kpoints':
        # '3x3x1', 'energy_per_fu': -34.807990365, 'energy_per_atom': -11.602663455, 'energy':
        # -278.46392292, 'formation_energy': -10.951109995000003}, {'formula': 'La2Zr2O7',
        # 'kpoints': '3x3x3', 'energy_per_fu': -119.619571095, 'energy_per_atom':
        # -10.874506463181818, 'energy': -239.23914219, 'formation_energy': -40.87683184}]
        assert np.isclose(ext_cpa.data[0]["DFT Energy (eV/fu)"], -5.00458616)
        assert np.isclose(ext_cpa.data[0]["DFT Energy (eV/atom)"], -5.00458616)
        assert np.isclose(ext_cpa.data[0]["DFT Energy (eV)"], -20.01834464)
        assert np.isclose(ext_cpa.data[0]["Formation Energy (eV/fu)"], 0.0)
        assert np.isclose(ext_cpa.data[0]["Formation Energy (eV/atom)"], 0.0)
        assert np.isclose(ext_cpa.data[-1]["DFT Energy (eV/fu)"], -119.619571095)
        assert np.isclose(ext_cpa.data[-1]["DFT Energy (eV/atom)"], -10.874506463181818)
        assert np.isclose(ext_cpa.data[-1]["DFT Energy (eV)"], -239.23914219)
        assert np.isclose(ext_cpa.data[-1]["Formation Energy (eV/fu)"], -40.87683184)
        assert np.isclose(ext_cpa.data[-1]["Formation Energy (eV/atom)"], -40.87683184 / 11)
        assert np.isclose(ext_cpa.data[6]["DFT Energy (eV/fu)"], -42.524204305)
        assert np.isclose(ext_cpa.data[6]["DFT Energy (eV/atom)"], -10.63105107625)
        assert np.isclose(ext_cpa.data[6]["DFT Energy (eV)"], -85.04840861)
        assert np.isclose(ext_cpa.data[6]["Formation Energy (eV/fu)"], -5.986573519999993)
        assert np.isclose(ext_cpa.data[6]["Formation Energy (eV/atom)"], -5.986573519999993 / 4)

        # check if it works from a list
        all_paths = []
        for p in path.iterdir():
            if not p.name.startswith("."):
                pp = p / "relax" / "vasprun.xml"
                ppgz = p / "relax" / "vasprun.xml.gz"
                if pp.is_file():
                    all_paths.append(pp)
                elif ppgz.is_file():
                    all_paths.append(ppgz)
        lst_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        lst_cpa.from_vaspruns(path=all_paths)
        assert len(lst_cpa.elemental) == 2
        assert len(lst_cpa.vasprun_paths) == 8

        all_fols = [p / "relax" for p in path.iterdir() if not p.name.startswith(".")]
        lst_fols_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        lst_fols_cpa.from_vaspruns(path=all_fols)
        assert len(lst_fols_cpa.elemental) == 2

    def test_vaspruns_hidden_files(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"

        with open(f"{path}/._OUTCAR", "w") as f:
            f.write("test pop")
        with open(f"{path}/._vasprun.xml", "w") as f:
            f.write("test pop")
        with open(f"{path}/._LOCPOT", "w") as f:
            f.write("test pop")
        with open(f"{path}/.DS_Store", "w") as f:
            f.write("test pop")

        with warnings.catch_warnings(record=True) as w:
            cpa.from_vaspruns(path=path, folder="relax", csv_path=self.csv_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert not w

    def test_vaspruns_none_parsed(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = Path(__file__).parents[1].joinpath("examples/competing_phases")

        with warnings.catch_warnings(record=True) as w, pytest.raises(FileNotFoundError) as e:
            cpa.from_vaspruns(path=path, folder="relax", csv_path=self.csv_path)
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert "Failed to parse the following `vasprun.xml` files:\n(files: error)\n" in str(w[0].message)
        assert "Is a directory" in str(w[0].message)
        assert "No vasprun files have been parsed," in str(e.value)

    def test_cplap_input(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        cpa.from_csv(self.csv_path)
        cpa.cplap_input(dependent_variable="O")

        assert Path("input.dat").is_file()

        with open("input.dat", encoding="utf-8") as file:
            contents = file.readlines()

        # assert these lines are in the file:
        for i in [
            "2  # number of elements in bulk\n",
            "1 Zr 2 O -10.975428440000002  # number of atoms, element, formation energy (bulk)\n",
            "O  # dependent variable (element)\n",
            "2  # number of bordering phases\n",
            "1  # number of elements in phase:\n",
            "2 O 0.0  # number of atoms, element, formation energy\n",
            "2  # number of elements in phase:\n",
            "3 Zr 1 O -5.986573519999993  # number of atoms, element, formation energy\n",
        ]:
            assert i in contents

        assert (
            "1 Zr 2 O -10.951109995000005\n" not in contents
        )  # shouldn't be included as is a higher energy polymorph of the bulk phase
        assert all("2 Zr 1 O" not in i for i in contents)  # non-bordering phase

    def test_latex_table(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"
        cpa.from_vaspruns(path=path, folder="relax", csv_path=self.csv_path)

        string = cpa.to_LaTeX_table(splits=1)
        assert (
            string[28:209]
            == "\\caption{Formation energies per formula unit ($\\Delta E_f$) of \\ce{ZrO2} and all "
            "competing phases, with k-meshes used in calculations. Only the lowest energy polymorphs "
            "are included"
        )
        assert len(string) == 589
        assert string.split("hline")[1] == "\nFormula & k-mesh & $\\Delta E_f$ (eV/fu) \\\\ \\"
        assert string.split("hline")[2][2:45] == "\\ce{ZrO2} & 3$\\times$3$\\times$3 & -10.975 \\"

        string = cpa.to_LaTeX_table(splits=2)
        assert (
            string[28:209]
            == "\\caption{Formation energies per formula unit ($\\Delta E_f$) of \\ce{ZrO2} and all "
            "competing phases, with k-meshes used in calculations. Only the lowest energy polymorphs "
            "are included"
        )
        assert (
            string.split("hline")[1]
            == "\nFormula & k-mesh & $\\Delta E_f$ (eV/fu) & Formula & k-mesh & $\\Delta E_f$ ("
            "eV/fu) \\\\ \\"
        )

        assert string.split("hline")[2][2:45] == "\\ce{ZrO2} & 3$\\times$3$\\times$3 & -10.975 &"
        assert len(string) == 586

        # test without kpoints:
        for entry_dict in cpa.data:
            entry_dict.pop("k-points")
        string = cpa.to_LaTeX_table(splits=1)
        assert (
            string[28:173]
            == "\\caption{Formation energies per formula unit ($\\Delta E_f$) of \\ce{ZrO2} and all "
            "competing phases. Only the lowest energy polymorphs are included"
        )
        assert len(string) == 433
        assert string.split("hline")[1] == "\nFormula & $\\Delta E_f$ (eV/fu) \\\\ \\"
        assert string.split("hline")[2][2:23] == "\\ce{ZrO2} & -10.975 \\"

        cpa_csv = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        cpa_csv.from_csv(self.csv_path)
        with pytest.raises(ValueError):
            cpa_csv.to_LaTeX_table(splits=3)

    def test_to_csv(self):
        self.tearDown()  # clear out previous csvs
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        stable_cpa_data = stable_cpa._get_and_sort_formation_energy_data()

        stable_cpa.to_csv("competing_phases.csv")
        reloaded_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        reloaded_cpa.from_csv("competing_phases.csv")
        reloaded_cpa_data = reloaded_cpa._get_and_sort_formation_energy_data()
        assert pd.DataFrame(stable_cpa_data).equals(pd.DataFrame(reloaded_cpa_data))

        # check chem limits the same:
        _compare_chempot_dicts(stable_cpa.chempots, reloaded_cpa.chempots)

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        self.ext_cpa.to_csv("competing_phases.csv")
        reloaded_ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        reloaded_ext_cpa.from_csv("competing_phases.csv")
        reloaded_ext_cpa._get_and_sort_formation_energy_data()

        assert reloaded_ext_cpa.formation_energy_df.equals(self.ext_cpa.formation_energy_df)

        # test pruning:
        self.ext_cpa.to_csv("competing_phases.csv", prune_polymorphs=True)
        reloaded_ext_cpa.from_csv("competing_phases.csv")
        assert len(reloaded_ext_cpa.data) == 8  # polymorphs pruned
        assert len(self.ext_cpa.data) == 11

        formulas = [i["Formula"] for i in reloaded_ext_cpa.data]
        assert len(formulas) == len(set(formulas))  # no polymorphs

        reloaded_cpa.to_csv("competing_phases.csv", prune_polymorphs=True)
        reloaded_cpa.from_csv("competing_phases.csv")

        # check chem limits the same:
        _compare_chempot_dicts(reloaded_cpa.chempots, stable_cpa.chempots)

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        assert self.ext_cpa.chempots["elemental_refs"] == self.parsed_ext_chempots["elemental_refs"]

        # test correct sorting:
        self.ext_cpa.to_csv("competing_phases.csv", prune_polymorphs=True, sort_by_energy=True)
        reloaded_ext_cpa_energy_sorted = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        reloaded_ext_cpa_energy_sorted.from_csv("competing_phases.csv")
        assert len(reloaded_ext_cpa.data) == 8  # polymorphs pruned

        assert reloaded_ext_cpa_energy_sorted.data != reloaded_ext_cpa.data  # different order
        sorted_data = sorted(
            reloaded_ext_cpa.data, key=lambda x: x["Formation Energy (eV/fu)"], reverse=True
        )
        chemical_potentials._move_dict_to_start(
            sorted_data, "Formula", self.ext_cpa.bulk_composition.reduced_formula
        )
        assert reloaded_ext_cpa_energy_sorted.data == sorted_data  # energy sorted data

    def test_from_csv_minimal(self):
        """
        Test that CompetingPhasesAnalyzer.from_csv() works with minimal data.
        """
        self.tearDown()  # clear out previous csvs
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"
        cpa.from_vaspruns(path=path, folder="relax")
        formation_energy_data = cpa._get_and_sort_formation_energy_data()
        formation_energy_df = pd.DataFrame(formation_energy_data)

        # drop all but the formula and energy_per_fu columns:
        for i in ["DFT Energy (eV/fu)", "DFT Energy (eV/atom)"]:
            minimal_formation_energy_df = formation_energy_df[["Formula", i]]
            minimal_formation_energy_df.to_csv("competing_phases.csv", index=False)

            reloaded_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
            reloaded_cpa.from_csv("competing_phases.csv")
            assert not cpa.formation_energy_df.equals(
                reloaded_cpa.formation_energy_df
            )  # no kpoints or raw energy, but should have formula, energy_per_fu, energy_per_atom,
            # elemental amounts (i.e. Zr and O here) and formation_energy:
            minimal_columns = [
                "Formula",
                "DFT Energy (eV/fu)",
                "DFT Energy (eV/atom)",
                "Formation Energy (eV/fu)",
                "Formation Energy (eV/atom)",
                "Zr",  # ordered by appearance in bulk composition
                "O",
            ]
            trimmed_df = cpa.formation_energy_df[
                [column for column in cpa.formation_energy_df.columns if column in minimal_columns]
            ]
            # trimmed_df is formation_energy_df with min columns but in same order as original

            trimmed_df = trimmed_df.round(5)  # round to avoid slight numerical differences
            reloaded_cpa.formation_energy_df = reloaded_cpa.formation_energy_df.round(5)
            print(trimmed_df, reloaded_cpa.formation_energy_df)
            print(trimmed_df.columns, reloaded_cpa.formation_energy_df.columns)
            assert trimmed_df.equals(reloaded_cpa.formation_energy_df)

            # check chem limits the same:
            _compare_chempot_dicts(cpa.chempots, reloaded_cpa.chempots)

        # test ValueError without energy_per_fu/energy_per_atom column:
        too_minimal_formation_energy_df = formation_energy_df[["Formula"]]
        too_minimal_formation_energy_df.to_csv("competing_phases.csv", index=False)
        reloaded_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        with pytest.raises(ValueError) as exc:
            reloaded_cpa.from_csv("competing_phases.csv")
        assert "Supplied csv does not contain the minimal columns required" in str(exc.value)


class BoxedMoleculesTestCase(unittest.TestCase):
    def test_elements(self):
        s, f, m = chemical_potentials.make_molecule_in_a_box("O2")
        assert f == "O2"
        assert m == 2
        assert type(s) == Structure

        with pytest.raises(ValueError):
            chemical_potentials.make_molecule_in_a_box("R2")


class FormationEnergyTestCase(unittest.TestCase):
    def test_fe(self):
        elemental = {"O": -7.006602065, "Zr": -9.84367624}
        data = [
            {
                "Formula": "O2",
                "DFT Energy (eV/fu)": -14.01320413,
                "DFT Energy (eV/atom)": -7.006602065,
                "DFT Energy (eV)": -14.01320413,
            },
            {
                "Formula": "Zr",
                "DFT Energy (eV/fu)": -9.84367624,
                "DFT Energy (eV/atom)": -9.84367624,
                "DFT Energy (eV)": -19.68735248,
            },
            {
                "Formula": "Zr3O",
                "DFT Energy (eV/fu)": -42.524204305,
                "DFT Energy (eV/atom)": -10.63105107625,
                "DFT Energy (eV)": -85.04840861,
            },
            {
                "Formula": "ZrO2",
                "DFT Energy (eV/fu)": -34.5391058,
                "DFT Energy (eV/atom)": -11.5130352,
                "DFT Energy (eV)": -138.156423,
            },
            {
                "Formula": "ZrO2",
                "DFT Energy (eV/fu)": -34.83230881,
                "DFT Energy (eV/atom)": -11.610769603333331,
                "DFT Energy (eV)": -139.32923524,
            },
            {
                "Formula": "Zr2O",
                "DFT Energy (eV/fu)": -32.42291351666667,
                "DFT Energy (eV/atom)": -10.807637838888889,
                "DFT Energy (eV)": -194.5374811,
            },
        ]

        formation_energy_df = chemical_potentials._calculate_formation_energies(data, elemental)
        assert formation_energy_df["Formula"][0] == "O2"  # definite order
        assert formation_energy_df["Formation Energy (eV/fu)"][0] == 0
        assert formation_energy_df["Formula"][1] == "Zr"
        assert formation_energy_df["Formation Energy (eV/fu)"][1] == 0
        # lowest energy ZrO2:
        assert np.isclose(formation_energy_df["Formation Energy (eV/fu)"][4], -10.975428440000002)


class CombineExtrinsicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parents[1].joinpath("examples/competing_phases")
        first = self.path / "zro2_la_chempots.json"
        second = self.path / "zro2_y_chempots.json"
        self.first = loadfn(first)
        self.second = loadfn(second)
        self.extrinsic_species = "Y"
        return super().setUp()

    def test_combine_extrinsic(self):
        d = chemical_potentials.combine_extrinsic(self.first, self.second, self.extrinsic_species)
        assert len(d["elemental_refs"].keys()) == 4
        limits = list(d["limits"].keys())
        assert limits[0].rsplit("-", 1)[1] == "Y2Zr2O7"

    def test_combine_extrinsic_errors(self):
        d = {"a": 1}
        with pytest.raises(KeyError):
            chemical_potentials.combine_extrinsic(d, self.second, self.extrinsic_species)

        with pytest.raises(KeyError):
            chemical_potentials.combine_extrinsic(self.first, d, self.extrinsic_species)

        with pytest.raises(ValueError):
            chemical_potentials.combine_extrinsic(self.first, self.second, "R")


class CompetingPhasesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parents[0]
        self.api_key = "c2LiJRMiBeaN5iXsH"  # SK MP Imperial email A/C
        self.cp = chemical_potentials.CompetingPhases("ZrO2", e_above_hull=0.03, api_key=self.api_key)

        return super().setUp()

    def tearDown(self) -> None:
        if Path("competing_phases").is_dir():
            shutil.rmtree("competing_phases")

        return super().tearDown()

    def test_init(self):
        assert len(self.cp.entries) == 13
        assert self.cp.entries[0].name == "O2"
        assert self.cp.entries[0].data["total_magnetization"] == 2
        assert self.cp.entries[0].data["e_above_hull"] == 0
        assert self.cp.entries[0].data["molecule"]
        assert np.isclose(self.cp.entries[0].data["energy_per_atom"], -4.94795546875)
        assert np.isclose(self.cp.entries[0].data["energy"], -9.8959109375)
        assert self.cp.entries[1].name == "Zr"
        assert np.isclose(self.cp.entries[1].data["total_magnetization"], 0, atol=1e-3)
        assert self.cp.entries[1].data["e_above_hull"] == 0
        assert not self.cp.entries[1].data["molecule"]
        assert self.cp.entries[2].name == "Zr3O"
        assert self.cp.entries[2].data["e_above_hull"] == 0
        assert self.cp.entries[3].name == "ZrO2"
        assert self.cp.entries[3].data["e_above_hull"] == 0
        assert self.cp.entries[4].name == "Zr3O"
        assert self.cp.entries[5].name == "Zr3O"
        assert self.cp.entries[6].name == "Zr2O"
        assert self.cp.entries[7].name == "ZrO2"
        assert self.cp.entries[8].name == "ZrO2"
        assert self.cp.entries[9].name == "Zr"
        assert self.cp.entries[10].name == "ZrO2"
        assert self.cp.entries[11].name == "ZrO2"
        assert self.cp.entries[12].name == "ZrO2"

        assert "Zr4O" not in [e.name for e in self.cp.entries]

    def test_init_ZnSe(self):
        """
        As noted by Savya Aggarwal, the legacy MP API code didn't return ZnSe2
        as a competing phase despite being on the hull and bordering ZnSe,
        because the legacy MP API database wrongly had the data['e_above_hull']
        value as 0.147 eV/atom (when it should be 0 eV/atom).
        https://legacy.materialsproject.org/materials/mp-1102515/ https://next-
        gen.materialsproject.org/materials/mp-1102515?formula=ZnSe2.

        Updated code which re-calculates the energy above hull avoids this
        issue.
        """
        cp = chemical_potentials.CompetingPhases("ZnSe", api_key=self.api_key)
        assert any(e.name == "ZnSe2" for e in cp.entries)
        assert len(cp.entries) == 14  # ZnSe2 now present
        znse2_entry = next(e for e in cp.entries if e.name == "ZnSe2")
        assert znse2_entry.data["e_above_hull"] == 0
        assert not znse2_entry.data["molecule"]
        assert np.isclose(znse2_entry.data["energy_per_atom"], -3.080017)
        assert np.isclose(znse2_entry.data["energy"], -3.080017 * 12)

    def test_init_full_phase_diagram(self):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", e_above_hull=0.03, api_key=self.api_key, full_phase_diagram=True
        )
        assert len(cp.entries) == 14  # Zr4O now present
        assert cp.entries[0].name == "O2"
        assert cp.entries[0].data["total_magnetization"] == 2
        assert cp.entries[0].data["e_above_hull"] == 0
        assert cp.entries[0].data["molecule"]
        assert np.isclose(cp.entries[0].data["energy_per_atom"], -4.94795546875)
        assert np.isclose(cp.entries[0].data["energy"], -9.8959109375)
        assert cp.entries[1].name == "Zr"
        assert np.isclose(cp.entries[1].data["total_magnetization"], 0, atol=1e-3)
        assert cp.entries[1].data["e_above_hull"] == 0
        assert not cp.entries[1].data["molecule"]
        assert cp.entries[2].name == "Zr3O"
        assert cp.entries[2].data["e_above_hull"] == 0
        assert cp.entries[3].name == "Zr4O"  # new entry!
        assert cp.entries[3].data["e_above_hull"] == 0
        assert cp.entries[4].name == "ZrO2"
        assert cp.entries[4].data["e_above_hull"] == 0
        assert cp.entries[5].name == "Zr3O"
        assert cp.entries[6].name == "Zr3O"
        assert cp.entries[7].name == "Zr2O"
        assert cp.entries[8].name == "ZrO2"
        assert cp.entries[9].name == "ZrO2"
        assert cp.entries[10].name == "Zr"
        assert cp.entries[11].name == "ZrO2"
        assert cp.entries[12].name == "ZrO2"
        assert cp.entries[13].name == "ZrO2"

    def test_init_ytos(self):
        # 144 phases on Y-Ti-O-S MP phase diagram
        cp = chemical_potentials.CompetingPhases("Y2Ti2S2O5", e_above_hull=0.1, api_key=self.api_key)
        assert len(cp.entries) == 115  # 115 phases with default algorithm
        # assert only one O2 phase present (molecular entry):
        o2_entries = [e for e in cp.entries if e.name == "O2"]
        assert len(o2_entries) == 1
        assert o2_entries[0].name == "O2"
        assert o2_entries[0].data["total_magnetization"] == 2
        assert o2_entries[0].data["e_above_hull"] == 0
        assert o2_entries[0].data["molecule"]
        assert np.isclose(o2_entries[0].data["energy_per_atom"], -4.94795546875)

        cp = chemical_potentials.CompetingPhases(
            "Y2Ti2S2O5", e_above_hull=0.1, full_phase_diagram=True, api_key=self.api_key
        )
        assert len(cp.entries) == 140  # 144 phases on Y-Ti-O-S MP phase diagram,
        # 4 extra O2 phases removed

        # assert only one O2 phase present (molecular entry):
        o2_entries = [e for e in cp.entries if e.name == "O2"]
        assert len(o2_entries) == 1
        assert o2_entries[0].name == "O2"
        assert o2_entries[0].data["total_magnetization"] == 2
        assert o2_entries[0].data["e_above_hull"] == 0
        assert o2_entries[0].data["molecule"]
        assert np.isclose(o2_entries[0].data["energy_per_atom"], -4.94795546875)

    def test_api_keys_errors(self):
        nonvalid_api_key_error = ValueError(
            "API key test is not a valid legacy Materials Project API key. These are "
            "available at https://legacy.materialsproject.org/open"
        )
        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="test",
            )
        assert str(nonvalid_api_key_error) in str(e.value)

        new_api_key_error = ValueError(
            "You are trying to use the new Materials Project (MP) API which is not supported "
            "by doped. Please use the legacy MP API (https://legacy.materialsproject.org/open)."
        )
        with pytest.raises(ValueError) as e:
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="testabcdefghijklmnopqrstuvwxyz12",
            )
        assert str(new_api_key_error) in str(e.value)

    def test_convergence_setup(self):
        # potcar spec doesn't need potcars set up for pmg and it still works
        self.cp.convergence_setup(potcar_spec=True)
        assert len(self.cp.metals) == 6
        assert self.cp.metals[0].data["band_gap"] == 0
        assert not self.cp.nonmetals[0].data["molecule"]
        # this shouldn't exist - don't need to convergence test for molecules
        assert not Path("competing_phases/O2_EaH_0.0").is_dir()

        # test if it writes out the files correctly
        path1 = "competing_phases/ZrO2_EaH_0.0088/kpoint_converge/k2,1,1/"
        assert Path(path1).is_dir()
        with open(f"{path1}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[3] == "2 1 1\n"

        with open(f"{path1}/POTCAR.spec", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[0] == "Zr_sv\n"

        with open(f"{path1}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert any(line == "GGA = Ps\n" for line in contents)
            assert any(line == "NSW = 0\n" for line in contents)

    def test_vasp_std_setup(self):
        self.cp.vasp_std_setup(potcar_spec=True)
        assert len(self.cp.nonmetals) == 6
        assert len(self.cp.metals) == 6
        assert len(self.cp.molecules) == 1
        assert self.cp.molecules[0].name == "O2"
        assert self.cp.molecules[0].data["total_magnetization"] == 2
        assert self.cp.molecules[0].data["molecule"]
        assert not self.cp.nonmetals[0].data["molecule"]

        path1 = "competing_phases/ZrO2_EaH_0/vasp_std/"
        assert Path(path1).is_dir()
        with open(f"{path1}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert "KPOINTS from doped, with reciprocal_density = 64/Å" in contents[0]
            assert contents[3] == "4 4 4\n"

        with open(f"{path1}/POTCAR.spec", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents == ["Zr_sv\n", "O"]

        with open(f"{path1}/INCAR", encoding="utf-8") as file:
            contents = file.readlines()
            assert all(x in contents for x in ["AEXX = 0.25\n", "ISIF = 3\n", "GGA = Pe\n"])

        path2 = "competing_phases/O2_EaH_0/vasp_std"
        assert Path(path2).is_dir()
        with open(f"{path2}/KPOINTS", encoding="utf-8") as file:
            contents = file.readlines()
            assert contents[3] == "1 1 1\n"

        struct = Structure.from_file(f"{path2}/POSCAR")
        assert np.isclose(struct.sites[0].frac_coords, [0.49983339, 0.5, 0.50016672]).all()
        assert np.isclose(struct.sites[1].frac_coords, [0.49983339, 0.5, 0.5405135]).all()


class ExtrinsicCompetingPhasesTestCase(unittest.TestCase):
    # TODO: Need to add tests for co-doping, full_sub_approach, full_phase_diagram etc!!
    def setUp(self) -> None:
        self.path = Path(__file__).parents[0]
        self.api_key = "c2LiJRMiBeaN5iXsH"  # SK MP Imperial email A/C
        self.ex_cp = chemical_potentials.ExtrinsicCompetingPhases(
            "ZrO2", extrinsic_species="La", e_above_hull=0, api_key=self.api_key
        )
        return super().setUp()

    def tearDown(self) -> None:
        if Path("competing_phases").is_dir():
            shutil.rmtree("competing_phases")

        return super().tearDown()

    def test_init(self):
        assert len(self.ex_cp.entries) == 2
        assert self.ex_cp.entries[0].name == "La"  # definite ordering
        assert self.ex_cp.entries[1].name == "La2Zr2O7"  # definite ordering
        assert [(entry.data["e_above_hull"] == 0) for entry in self.ex_cp.entries]

        # names of intrinsic entries: ['O2', 'Zr', 'Zr3O', 'ZrO2']
        assert len(self.ex_cp.intrinsic_entries) == 4
        assert self.ex_cp.intrinsic_entries[0].name == "O2"
        assert self.ex_cp.intrinsic_entries[1].name == "Zr"
        assert self.ex_cp.intrinsic_entries[2].name == "Zr3O"
        assert self.ex_cp.intrinsic_entries[3].name == "ZrO2"
        assert all(entry.data["e_above_hull"] == 0 for entry in self.ex_cp.intrinsic_entries)

        cp = chemical_potentials.ExtrinsicCompetingPhases(
            "ZrO2", extrinsic_species="La", api_key=self.api_key
        )  # default e_above_hull=0.1
        assert len(cp.entries) == 5
        assert cp.entries[2].name == "La"  # definite ordering, same 1st & 2nd as before
        assert cp.entries[3].name == "LaZr9O20"  # definite ordering
        assert cp.entries[4].name == "LaZr9O20"  # definite ordering
        assert all(entry.data["e_above_hull"] == 0 for entry in cp.entries[:2])
        assert not any(entry.data["e_above_hull"] == 0 for entry in cp.entries[2:])
        assert len(cp.intrinsic_entries) == 28
