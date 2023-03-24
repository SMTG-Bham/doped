import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from doped import competing_phases
from pymatgen.core.structure import Structure
from monty.serialization import loadfn, dumpfn


class ChemPotsTestCase(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).parents[1].joinpath("examples/competing_phases")
        self.stable_system = "ZrO2"
        self.unstable_system = "Zr2O"
        self.extrinsic_species = "La"
        self.csv_path = self.path / "zro2_competing_phase_energies.csv"
        self.csv_path_ext = self.path / "zro2_la_competing_phase_energies.csv"

    def tearDown(self) -> None:
        if Path("chempot_limits.csv").is_file():
            os.remove("chempot_limits.csv")

        if Path("competing_phases.csv").is_file():
            os.remove("competing_phases.csv")

        if Path("input.dat").is_file():
            os.remove("input.dat")

        return super().tearDown()

    def test_cpa_csv(self):
        stable_cpa = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        self.ext_cpa = competing_phases.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)

        self.assertEqual(len(stable_cpa.elemental), 2)
        self.assertEqual(len(self.ext_cpa.elemental), 3)
        self.assertTrue(any(entry["formula"] == "O2" for entry in stable_cpa.data))
        self.assertAlmostEqual(
            [
                entry["energy_per_fu"]
                for entry in self.ext_cpa.data
                if entry["formula"] == "La2Zr2O7"
            ][0],
            -119.619571095,
        )

    # test chempots
    def test_cpa_chempots(self):
        stable_cpa = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        df1 = stable_cpa.calculate_chempots()
        self.assertEqual(list(df1["O"])[0], 0)
        # check if it's no longer Element
        self.assertEqual(
            type(list(stable_cpa.intrinsic_chem_limits["elemental_refs"].keys())[0]),
            str,
        )

        self.unstable_cpa = competing_phases.CompetingPhasesAnalyzer(
            self.unstable_system
        )
        self.unstable_cpa.from_csv(self.csv_path)
        with self.assertRaises(ValueError):
            self.unstable_cpa.calculate_chempots()

        self.ext_cpa = competing_phases.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        df = self.ext_cpa.calculate_chempots()
        self.assertEqual(list(df["La_limiting_phase"])[0], "La2Zr2O7")
        self.assertAlmostEqual(list(df["La"])[0], -9.46298748)

    # test vaspruns
    def test_vaspruns(self):
        cpa = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO"
        cpa.from_vaspruns(path=path, folder="relax", csv_fname=self.csv_path)
        self.assertEqual(len(cpa.elemental), 2)
        self.assertEqual(cpa.data[0]["formula"], "O2")

        cpa_no = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        with self.assertRaises(FileNotFoundError):
            cpa_no.from_vaspruns(path="path")

        with self.assertRaises(ValueError):
            cpa_no.from_vaspruns(path=0)

        ext_cpa = competing_phases.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        path2 = self.path / "LaZrO"
        ext_cpa.from_vaspruns(path=path2, folder="relax", csv_fname=self.csv_path_ext)
        self.assertEqual(len(ext_cpa.elemental), 3)
        # sorted by num_species, then alphabetically, then by num_atoms_in_fu, then by
        # formation_energy
        self.assertEqual(
            [entry["formula"] for entry in ext_cpa.data],
            [
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
            ],
        )
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
        self.assertEqual(ext_cpa.data[0]["energy_per_fu"], -5.00458616)
        self.assertEqual(ext_cpa.data[0]["energy_per_atom"], -5.00458616)
        self.assertEqual(ext_cpa.data[0]["energy"], -20.01834464)
        self.assertEqual(ext_cpa.data[0]["formation_energy"], 0.0)
        self.assertEqual(ext_cpa.data[-1]["energy_per_fu"], -119.619571095)
        self.assertEqual(ext_cpa.data[-1]["energy_per_atom"], -10.874506463181818)
        self.assertEqual(ext_cpa.data[-1]["energy"], -239.23914219)
        self.assertEqual(ext_cpa.data[-1]["formation_energy"], -40.87683184)
        self.assertEqual(ext_cpa.data[6]["energy_per_fu"], -42.524204305)
        self.assertEqual(ext_cpa.data[6]["energy_per_atom"], -10.63105107625)
        self.assertEqual(ext_cpa.data[6]["energy"], -85.04840861)
        self.assertEqual(ext_cpa.data[6]["formation_energy"], -5.986573519999993)

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
        lst_cpa = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        lst_cpa.from_vaspruns(path=all_paths)
        self.assertEqual(len(lst_cpa.elemental), 2)
        self.assertEqual(len(lst_cpa.vasprun_paths), 8)

        all_fols = []
        for p in path.iterdir():
            if not p.name.startswith("."):
                pp = p / "relax"
                all_fols.append(pp)
        lst_fols_cpa = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        lst_fols_cpa.from_vaspruns(path=all_fols)
        self.assertEqual(len(lst_fols_cpa.elemental), 2)

    def test_cplap_input(self):
        cpa = competing_phases.CompetingPhasesAnalyzer(self.stable_system)
        cpa.from_csv(self.csv_path)
        cpa.cplap_input(dependent_variable="O")

        self.assertTrue(Path("input.dat").is_file())

        with open("input.dat", "r") as f:
            contents = f.readlines()

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
            self.assertIn(i, contents)

        self.assertNotIn(
            "1 Zr 2 O -10.951109995000005\n", contents
        )  # shouldn't be included as is a higher energy polymorph of the bulk phase
        self.assertFalse(any("2 Zr 1 O" in i for i in contents))  # non-bordering phase


class BoxedMoleculesTestCase(unittest.TestCase):
    def test_elements(self):
        s, f, m = competing_phases.make_molecule_in_a_box("O2")
        self.assertEqual(f, "O2")
        self.assertEqual(m, 2)
        self.assertEqual(type(s), Structure)

        with self.assertRaises(UnboundLocalError):
            s2, f2, m2 = competing_phases.make_molecule_in_a_box("R2")


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

        df = competing_phases._calculate_formation_energies(data, elemental)
        print(df)
        self.assertEqual(df["formula"][0], "O2")  # definite order
        self.assertEqual(df["formation_energy"][0], 0)
        self.assertEqual(df["formula"][1], "Zr")
        self.assertEqual(df["formation_energy"][1], 0)
        self.assertAlmostEqual(df["formation_energy"][4], -10.975428440000002)  # lowest energy ZrO2


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
        d = competing_phases.combine_extrinsic(
            self.first, self.second, self.extrinsic_species
        )
        self.assertEqual(len(d["elemental_refs"].keys()), 4)
        facets = list(d["facets"].keys())
        self.assertEqual(facets[0].rsplit("-", 1)[1], "Y2Zr2O7")

    def test_combine_extrinsic_errors(self):
        d = {"a": 1}
        with self.assertRaises(KeyError):
            competing_phases.combine_extrinsic(d, self.second, self.extrinsic_species)

        with self.assertRaises(KeyError):
            competing_phases.combine_extrinsic(self.first, d, self.extrinsic_species)

        with self.assertRaises(ValueError):
            competing_phases.combine_extrinsic(self.first, self.second, "R")


# not sure how legit this is but since I can't figure out how to
# patch mock get_entries_in_chemsys that's in the init of the class
class MockedIshCompetingPhases(competing_phases.CompetingPhases):
    # sets up a fake class that can read in entries
    # that were already collected from MP
    def __init__(self, entries, system, e_above_hull, full_phase_diagram, api_key=None):
        self.entries = entries
        self.eah = "e_above_hull"  # for legacy MPRester
        super().__init__(
            system,
            e_above_hull=e_above_hull,
            full_phase_diagram=full_phase_diagram,
            api_key=api_key,
        )


class CompetingPhasesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # this json was generated 09/02/2023 using
        # CompetingPhases(['Zr', 'O'], e_above_hull=0.01)
        self.entries = loadfn("entries_test_data.json")
        self.system = ["Zr", "O"]
        self.e_above_hull = 0.01

        # not sure if this actually works tbh
        path = Path("~/.pmgrc.yaml").expanduser()
        if path.is_file():
            path2 = str(path) + ".bak"
            os.rename(str(path), path2)
        return super().setUp()

    def test_init(self):
        mcp = MockedIshCompetingPhases(
            self.entries, self.system, self.e_above_hull, full_phase_diagram=True
        )
        self.assertEqual(len(mcp.entries), 13)
        self.assertEqual(len(mcp.parsed_entries), 11)
        self.assertEqual(len(mcp.competing_phases), 9)
        self.assertEqual(mcp.competing_phases[0]["magnetisation"], 2)

    def test_api_keys_errors(self):
        with self.assertRaisesRegex(
            ValueError, "API key test is not a valid Materials Project API key."
        ):
            mcp = MockedIshCompetingPhases(
                self.entries,
                self.system,
                self.e_above_hull,
                full_phase_diagram=True,
                api_key="test",
            )

        with self.assertRaisesRegex(
            ValueError,
            "You are trying to use the new Materials Project API key without the mp-api client, which is not supported by doped. Please use the legacy API key.",
        ):
            mcp = MockedIshCompetingPhases(
                self.entries,
                self.system,
                self.e_above_hull,
                full_phase_diagram=True,
                api_key="testabcdefghijklmnopqrstuvwxyz12",
            )

    def test_convergence_setup(self):
        mcp = MockedIshCompetingPhases(
            self.entries, self.system, self.e_above_hull, full_phase_diagram=True
        )

        # potcar spec doesnt need potcars set up for pmg and it still works
        mcp.convergence_setup(potcar_spec=True)
        self.assertEqual(len(mcp.metals), 6)
        self.assertEqual(mcp.metals[0]["band_gap"], 0)
        self.assertEqual(mcp.nonmetals[0]["molecule"], False)
        # this shouldnt exist - dont need to convergence test for molecules
        self.assertFalse(Path("competing_phases/O2_EaH_0.0").is_dir())

        # test if it writes out the files correctly
        path1 = "competing_phases/ZrO2_EaH_0.0088/kpoint_converge/k2,1,1/"
        self.assertTrue(Path(path1).is_dir())
        with open(f"{path1}/KPOINTS", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents[3], "2 1 1\n")

        with open(f"{path1}/POTCAR.spec", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents[0], "Zr_sv\n")

        with open(f"{path1}/INCAR", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents[4], "GGA = Ps\n")
            self.assertEqual(contents[6], "ISIF = 2\n")

    def test_vasp_std_setup(self):
        mcp = MockedIshCompetingPhases(
            self.entries, self.system, self.e_above_hull, full_phase_diagram=True
        )

        mcp.vasp_std_setup(potcar_spec=True)
        self.assertEqual(len(mcp.nonmetals), 2)
        self.assertEqual(mcp.molecules[0]["formula"], "O2")
        self.assertEqual(mcp.molecules[0]["magnetisation"], 2)
        self.assertEqual(mcp.nonmetals[0]["molecule"], False)

        path1 = "competing_phases/ZrO2_EaH_0.0/vasp_std/"
        self.assertTrue(Path(path1).is_dir())
        with open(f"{path1}/KPOINTS", "r") as f:
            contents = f.readlines()
            self.assertEqual(
                contents[0], "pymatgen with grid density = 911 / number of atoms\n"
            )
            self.assertEqual(contents[3], "4 4 4\n")

        with open(f"{path1}/POTCAR.spec", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents, ["Zr_sv\n", "O"])

        with open(f"{path1}/INCAR", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents[0], "AEXX = 0.25\n")
            self.assertEqual(contents[8], "ISIF = 3\n")
            self.assertEqual(contents[5], "GGA = Pe\n")

        path2 = "competing_phases/O2_EaH_0.0/vasp_std"
        self.assertTrue(Path(path2).is_dir())
        with open(f"{path2}/KPOINTS", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents[3], "1 1 1\n")

        with open(f"{path2}/POSCAR", "r") as f:
            contents = f.readlines()
            self.assertEqual(contents[-1], "0.500000 0.500000 0.540667 O\n")

    def test_mp_phase_diagram(self):
        with self.assertRaises(ValueError):
            mcp = MockedIshCompetingPhases(
                self.entries, self.system, self.e_above_hull, full_phase_diagram=False
            )

        mcp = MockedIshCompetingPhases(
            self.entries, "ZrO2", self.e_above_hull, full_phase_diagram=False
        )
        self.assertEqual(len(mcp.entries), 13)
        self.assertEqual(len(mcp.parsed_entries), 9)
        self.assertEqual(len(mcp.competing_phases), 7)
        self.assertNotIn("Zr4O", [e["formula"] for e in mcp.competing_phases])

    def tearDown(self) -> None:
        if Path("competing_phases").is_dir():
            shutil.rmtree("competing_phases")

        path = Path("~/.pmgrc.yaml.bak").expanduser()
        if path.is_file():
            path2 = str(path)
            os.rename(path2, path2[:-4])

        return super().tearDown()


class MockedIshExtrinsicCompetingPhases(competing_phases.ExtrinsicCompetingPhases):
    # sets up a fake class that can read in entries
    # that were already collected from MP
    def __init__(
        self,
        og_competing_phases,
        ext_comp_phases,
        system,
        extrinsic_species,
        e_above_hull,
    ):
        self.og_competing_phases = og_competing_phases
        self.ext_competing_phases = ext_comp_phases
        super().__init__(system, extrinsic_species, e_above_hull)


class ExtrinsicCompetingPhasesTestCase(unittest.TestCase):
    # this could be done a lot better but it works for now, also included entries in the test_data folder to make it easier to test later on if someone manages to refractor the actual code to make it more testable
    def setUp(self) -> None:
        # this json was generated 09/02/2023 using
        # ExtrinsicCompetingPhases(['Zr', 'O'], 'La', e_above_hull=0.015)
        self.og_competing_phases = loadfn("phases_test_data.json")
        self.ext_competing_phases = loadfn("phases_la_test_data.json")
        self.system = ["Zr", "O"]
        self.extrinsic_species = "La"
        self.e_above_hull = 0.01
        return super().setUp()

    def test_init(self):
        macp = MockedIshExtrinsicCompetingPhases(
            self.og_competing_phases,
            self.ext_competing_phases,
            system=self.system,
            extrinsic_species=self.extrinsic_species,
            e_above_hull=self.e_above_hull,
        )
        self.assertEqual(len(macp.competing_phases), 4)
        self.assertEqual(macp.competing_phases[0]["formula"], "La")
