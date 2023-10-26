"""
Tests for the `doped.chemical_potentials` module.
"""
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
from monty.serialization import loadfn
from pymatgen.core.structure import Structure

from doped import chemical_potentials


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
        assert any(entry["formula"] == "O2" for entry in stable_cpa.data)
        assert np.isclose(
            [entry["energy_per_fu"] for entry in self.ext_cpa.data if entry["formula"] == "La2Zr2O7"][0],
            -119.619571095,
        )

    # test chempots
    def test_cpa_chempots(self):
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        chempot_df = stable_cpa.calculate_chempots()
        assert list(chempot_df["O"])[0] == 0
        # check if it's no longer Element
        assert type(list(stable_cpa.intrinsic_chem_limits["elemental_refs"].keys())[0]) == str

        self.unstable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.unstable_system)
        self.unstable_cpa.from_csv(self.csv_path)
        with self.assertRaises(ValueError):
            self.unstable_cpa.calculate_chempots()

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        chempot_df = self.ext_cpa.calculate_chempots()
        assert list(chempot_df["La_limiting_phase"])[0] == "La2Zr2O7"
        assert np.isclose(list(chempot_df["La"])[0], -9.46298748)

    def test_cpa_chem_limits(self):
        # test accessing cpa.chem_limits without previously calling cpa.calculate_chempots()
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        assert stable_cpa.chem_limits == self.parsed_chempots

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        assert self.ext_cpa.chem_limits["elemental_refs"] == self.parsed_ext_chempots["elemental_refs"]

    def test_sort_by(self):
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        chempot_df = stable_cpa.calculate_chempots(sort_by="Zr")
        assert np.isclose(list(chempot_df["Zr"])[0], -0.199544)
        assert np.isclose(list(chempot_df["Zr"])[1], -10.975428439999998)

        with self.assertRaises(KeyError):
            stable_cpa.calculate_chempots(sort_by="M")

    # test vaspruns
    def test_vaspruns(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"
        cpa.from_vaspruns(path=path, folder="relax", csv_fname=self.csv_path)
        assert len(cpa.elemental) == 2
        assert cpa.data[0]["formula"] == "O2"

        cpa_no = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        with self.assertRaises(FileNotFoundError):
            cpa_no.from_vaspruns(path="path")

        with self.assertRaises(ValueError):
            cpa_no.from_vaspruns(path=0)

        ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        path2 = self.path / "La_ZrO2"
        ext_cpa.from_vaspruns(path=path2, folder="relax", csv_fname=self.csv_path_ext)
        assert len(ext_cpa.elemental) == 3
        # sorted by num_species, then alphabetically, then by num_atoms_in_fu, then by
        # formation_energy
        assert [entry["formula"] for entry in ext_cpa.data] == [
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
        assert np.isclose(ext_cpa.data[0]["energy_per_fu"], -5.00458616)
        assert np.isclose(ext_cpa.data[0]["energy_per_atom"], -5.00458616)
        assert np.isclose(ext_cpa.data[0]["energy"], -20.01834464)
        assert np.isclose(ext_cpa.data[0]["formation_energy"], 0.0)
        assert np.isclose(ext_cpa.data[-1]["energy_per_fu"], -119.619571095)
        assert np.isclose(ext_cpa.data[-1]["energy_per_atom"], -10.874506463181818)
        assert np.isclose(ext_cpa.data[-1]["energy"], -239.23914219)
        assert np.isclose(ext_cpa.data[-1]["formation_energy"], -40.87683184)
        assert np.isclose(ext_cpa.data[6]["energy_per_fu"], -42.524204305)
        assert np.isclose(ext_cpa.data[6]["energy_per_atom"], -10.63105107625)
        assert np.isclose(ext_cpa.data[6]["energy"], -85.04840861)
        assert np.isclose(ext_cpa.data[6]["formation_energy"], -5.986573519999993)

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

        all_fols = []
        for p in path.iterdir():
            if not p.name.startswith("."):
                pp = p / "relax"
                all_fols.append(pp)
        lst_fols_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        lst_fols_cpa.from_vaspruns(path=all_fols)
        assert len(lst_fols_cpa.elemental) == 2

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
            "1 Zr 2 O -10.975428440000002  # num_atoms, element, formation_energy (bulk)\n",
            "O  # dependent variable (element)\n",
            "2  # number of bordering phases\n",
            "1  # number of elements in phase:\n",
            "2 O 0.0  # num_atoms, element, formation_energy\n",
            "2  # number of elements in phase:\n",
            "3 Zr 1 O -5.986573519999993  # num_atoms, element, formation_energy\n",
        ]:
            assert i in contents

        assert (
            "1 Zr 2 O -10.951109995000005\n" not in contents
        )  # shouldn't be included as is a higher energy polymorph of the bulk phase
        assert all("2 Zr 1 O" not in i for i in contents)  # non-bordering phase

    def test_latex_table(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"
        cpa.from_vaspruns(path=path, folder="relax", csv_fname=self.csv_path)

        with self.assertRaises(ValueError):
            cpa.to_LaTeX_table(columns="M")

        string = cpa.to_LaTeX_table(columns=1)
        assert (
            string[28:209]
            == "\\caption{Formation energies ($\\Delta E_f$) per formula unit of \\ce{ZrO2} and all "
            "competing phases, with k-meshes used in calculations. Only the lowest energy polymorphs "
            "are included"
        )
        assert len(string) == 586
        assert string.split("hline")[1] == "\nFormula & k-mesh & $\\Delta E_f$ (eV) \\\\ \\"
        assert string.split("hline")[2][2:45] == "\\ce{ZrO2} & 3$\\times$3$\\times$3 & -10.975 \\"

        string = cpa.to_LaTeX_table(columns=2)
        assert (
            string[28:209]
            == "\\caption{Formation energies ($\\Delta E_f$) per formula unit of \\ce{ZrO2} and all "
            "competing phases, with k-meshes used in calculations. Only the lowest energy polymorphs "
            "are included"
        )
        assert (
            string.split("hline")[1]
            == "\nFormula & k-mesh & $\\Delta E_f$ (eV) & Formula & k-mesh & $\\Delta E_f$ (eV)\\\\ \\"
        )

        assert string.split("hline")[2][2:45] == "\\ce{ZrO2} & 3$\\times$3$\\times$3 & -10.975 &"
        assert len(string) == 579

        cpa_csv = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        cpa_csv.from_csv(self.csv_path)
        with self.assertRaises(ValueError):
            cpa_csv.to_LaTeX_table(columns=2)


class BoxedMoleculesTestCase(unittest.TestCase):
    def test_elements(self):
        s, f, m = chemical_potentials.make_molecule_in_a_box("O2")
        assert f == "O2"
        assert m == 2
        assert type(s) == Structure

        with self.assertRaises(ValueError):
            chemical_potentials.make_molecule_in_a_box("R2")


class FormationEnergyTestCase(unittest.TestCase):
    def test_fe(self):
        elemental = {"O": -7.006602065, "Zr": -9.84367624}
        data = [
            {
                "formula": "O2",
                "energy_per_fu": -14.01320413,
                "energy_per_atom": -7.006602065,
                "energy": -14.01320413,
            },
            {
                "formula": "Zr",
                "energy_per_fu": -9.84367624,
                "energy_per_atom": -9.84367624,
                "energy": -19.68735248,
            },
            {
                "formula": "Zr3O",
                "energy_per_fu": -42.524204305,
                "energy_per_atom": -10.63105107625,
                "energy": -85.04840861,
            },
            {
                "formula": "ZrO2",
                "energy_per_fu": -34.5391058,
                "energy_per_atom": -11.5130352,
                "energy": -138.156423,
            },
            {
                "formula": "ZrO2",
                "energy_per_fu": -34.83230881,
                "energy_per_atom": -11.610769603333331,
                "energy": -139.32923524,
            },
            {
                "formula": "Zr2O",
                "energy_per_fu": -32.42291351666667,
                "energy_per_atom": -10.807637838888889,
                "energy": -194.5374811,
            },
        ]

        formation_energy_df = chemical_potentials._calculate_formation_energies(data, elemental)
        assert formation_energy_df["formula"][0] == "O2"  # definite order
        assert formation_energy_df["formation_energy"][0] == 0
        assert formation_energy_df["formula"][1] == "Zr"
        assert formation_energy_df["formation_energy"][1] == 0
        # lowest energy ZrO2:
        assert np.isclose(formation_energy_df["formation_energy"][4], -10.975428440000002)


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
        facets = list(d["facets"].keys())
        assert facets[0].rsplit("-", 1)[1] == "Y2Zr2O7"

    def test_combine_extrinsic_errors(self):
        d = {"a": 1}
        with self.assertRaises(KeyError):
            chemical_potentials.combine_extrinsic(d, self.second, self.extrinsic_species)

        with self.assertRaises(KeyError):
            chemical_potentials.combine_extrinsic(self.first, d, self.extrinsic_species)

        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError) as e:
            nonvalid_api_key_error = ValueError(
                "API key test is not a valid legacy Materials Project API key. These are "
                "available at https://legacy.materialsproject.org/open"
            )
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="test",
            )
            assert nonvalid_api_key_error in e.exception

        with self.assertRaises(ValueError) as e:
            new_api_key_error = ValueError(
                "You are trying to use the new Materials Project (MP) API which is not supported "
                "by doped. Please use the legacy MP API (https://legacy.materialsproject.org/open)."
            )
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="testabcdefghijklmnopqrstuvwxyz12",
            )
            assert new_api_key_error in e.exception

    def test_convergence_setup(self):
        # potcar spec doesn't need potcars set up for pmg and it still works
        self.cp.convergence_setup(potcar_spec=True)
        assert len(self.cp.metals) == 6
        assert self.cp.metals[0].data["band_gap"] == 0
        assert not self.cp.nonmetals[0].data["molecule"]
        # this shouldn't exist - don't need to convergence test for molecules
        assert not Path("competing_phases/O2_EaH_0").is_dir()

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
            assert contents[0] == "pymatgen with grid density = 911 / number of atoms\n"
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
