import os
import shutil
import unittest
from pathlib import Path

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
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        df1 = stable_cpa.calculate_chempots()
        self.assertEqual(list(df1["O"])[0], 0)
        # check if it's no longer Element
        self.assertEqual(
            type(list(stable_cpa.intrinsic_chem_limits["elemental_refs"].keys())[0]),
            str,
        )

        self.unstable_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.unstable_system
        )
        self.unstable_cpa.from_csv(self.csv_path)
        with self.assertRaises(ValueError):
            self.unstable_cpa.calculate_chempots()

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        df = self.ext_cpa.calculate_chempots()
        self.assertEqual(list(df["La_limiting_phase"])[0], "La2Zr2O7")
        self.assertAlmostEqual(list(df["La"])[0], -9.46298748)

    def test_cpa_chem_limits(self):
        # test accessing cpa.chem_limits without previously calling cpa.calculate_chempots()
        stable_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        self.assertDictEqual(stable_cpa.chem_limits, self.parsed_chempots)

        self.ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        self.ext_cpa.from_csv(self.csv_path_ext)
        self.assertDictEqual(
            self.ext_cpa.chem_limits["elemental_refs"],
            self.parsed_ext_chempots["elemental_refs"],
        )

    # test vaspruns
    def test_vaspruns(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        path = self.path / "ZrO2"
        cpa.from_vaspruns(path=path, folder="relax", csv_fname=self.csv_path)
        self.assertEqual(len(cpa.elemental), 2)
        self.assertEqual(cpa.data[0]["formula"], "O2")

        cpa_no = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        with self.assertRaises(FileNotFoundError):
            cpa_no.from_vaspruns(path="path")

        with self.assertRaises(ValueError):
            cpa_no.from_vaspruns(path=0)

        ext_cpa = chemical_potentials.CompetingPhasesAnalyzer(
            self.stable_system, self.extrinsic_species
        )
        path2 = self.path / "La_ZrO2"
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
        self.assertAlmostEqual(ext_cpa.data[0]["energy_per_fu"], -5.00458616)
        self.assertAlmostEqual(ext_cpa.data[0]["energy_per_atom"], -5.00458616)
        self.assertAlmostEqual(ext_cpa.data[0]["energy"], -20.01834464)
        self.assertAlmostEqual(ext_cpa.data[0]["formation_energy"], 0.0)
        self.assertAlmostEqual(ext_cpa.data[-1]["energy_per_fu"], -119.619571095)
        self.assertAlmostEqual(ext_cpa.data[-1]["energy_per_atom"], -10.874506463181818)
        self.assertAlmostEqual(ext_cpa.data[-1]["energy"], -239.23914219)
        self.assertAlmostEqual(ext_cpa.data[-1]["formation_energy"], -40.87683184)
        self.assertAlmostEqual(ext_cpa.data[6]["energy_per_fu"], -42.524204305)
        self.assertAlmostEqual(ext_cpa.data[6]["energy_per_atom"], -10.63105107625)
        self.assertAlmostEqual(ext_cpa.data[6]["energy"], -85.04840861)
        self.assertAlmostEqual(ext_cpa.data[6]["formation_energy"], -5.986573519999993)

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
        self.assertEqual(len(lst_cpa.elemental), 2)
        self.assertEqual(len(lst_cpa.vasprun_paths), 8)

        all_fols = []
        for p in path.iterdir():
            if not p.name.startswith("."):
                pp = p / "relax"
                all_fols.append(pp)
        lst_fols_cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        lst_fols_cpa.from_vaspruns(path=all_fols)
        self.assertEqual(len(lst_fols_cpa.elemental), 2)

    def test_cplap_input(self):
        cpa = chemical_potentials.CompetingPhasesAnalyzer(self.stable_system)
        cpa.from_csv(self.csv_path)
        cpa.cplap_input(dependent_variable="O")

        self.assertTrue(Path("input.dat").is_file())

        with open("input.dat", "r") as file:
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
            self.assertIn(i, contents)

        self.assertNotIn(
            "1 Zr 2 O -10.951109995000005\n", contents
        )  # shouldn't be included as is a higher energy polymorph of the bulk phase
        self.assertFalse(any("2 Zr 1 O" in i for i in contents))  # non-bordering phase


class BoxedMoleculesTestCase(unittest.TestCase):
    def test_elements(self):
        s, f, m = chemical_potentials.make_molecule_in_a_box("O2")
        self.assertEqual(f, "O2")
        self.assertEqual(m, 2)
        self.assertEqual(type(s), Structure)

        with self.assertRaises(UnboundLocalError):
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

        df = chemical_potentials._calculate_formation_energies(data, elemental)
        print(df)
        self.assertEqual(df["formula"][0], "O2")  # definite order
        self.assertEqual(df["formation_energy"][0], 0)
        self.assertEqual(df["formula"][1], "Zr")
        self.assertEqual(df["formation_energy"][1], 0)
        self.assertAlmostEqual(
            df["formation_energy"][4], -10.975428440000002
        )  # lowest energy ZrO2


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
        d = chemical_potentials.combine_extrinsic(
            self.first, self.second, self.extrinsic_species
        )
        self.assertEqual(len(d["elemental_refs"].keys()), 4)
        facets = list(d["facets"].keys())
        self.assertEqual(facets[0].rsplit("-", 1)[1], "Y2Zr2O7")

    def test_combine_extrinsic_errors(self):
        d = {"a": 1}
        with self.assertRaises(KeyError):
            chemical_potentials.combine_extrinsic(
                d, self.second, self.extrinsic_species
            )

        with self.assertRaises(KeyError):
            chemical_potentials.combine_extrinsic(self.first, d, self.extrinsic_species)

        with self.assertRaises(ValueError):
            chemical_potentials.combine_extrinsic(self.first, self.second, "R")


class CompetingPhasesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parents[0]
        self.api_key = "c2LiJRMiBeaN5iXsH"  # SK MP Imperial email A/C
        self.cp = chemical_potentials.CompetingPhases(
            "ZrO2", e_above_hull=0.03, api_key=self.api_key
        )

        return super().setUp()

    def tearDown(self) -> None:
        if Path("competing_phases").is_dir():
            shutil.rmtree("competing_phases")

        return super().tearDown()

    def test_init(self):
        self.assertEqual(len(self.cp.entries), 13)
        self.assertEqual(self.cp.entries[0].name, "O2")
        self.assertEqual(self.cp.entries[0].data["total_magnetization"], 2)
        self.assertEqual(self.cp.entries[0].data["e_above_hull"], 0)
        self.assertTrue(self.cp.entries[0].data["molecule"])
        self.assertAlmostEqual(
            self.cp.entries[0].data["energy_per_atom"], -4.94795546875
        )
        self.assertAlmostEqual(self.cp.entries[0].data["energy"], -9.8959109375)
        self.assertEqual(self.cp.entries[1].name, "Zr")
        self.assertAlmostEqual(
            self.cp.entries[1].data["total_magnetization"], 0, places=3
        )
        self.assertEqual(self.cp.entries[1].data["e_above_hull"], 0)
        self.assertFalse(self.cp.entries[1].data["molecule"])
        self.assertEqual(self.cp.entries[2].name, "Zr3O")
        self.assertEqual(self.cp.entries[2].data["e_above_hull"], 0)
        self.assertEqual(self.cp.entries[3].name, "ZrO2")
        self.assertEqual(self.cp.entries[3].data["e_above_hull"], 0)
        self.assertEqual(self.cp.entries[4].name, "Zr3O")
        self.assertEqual(self.cp.entries[5].name, "Zr3O")
        self.assertEqual(self.cp.entries[6].name, "Zr2O")
        self.assertEqual(self.cp.entries[7].name, "ZrO2")
        self.assertEqual(self.cp.entries[8].name, "ZrO2")
        self.assertEqual(self.cp.entries[9].name, "Zr")
        self.assertEqual(self.cp.entries[10].name, "ZrO2")
        self.assertEqual(self.cp.entries[11].name, "ZrO2")
        self.assertEqual(self.cp.entries[12].name, "ZrO2")

        self.assertNotIn("Zr4O", [e.name for e in self.cp.entries])

    def test_init_full_phase_diagram(self):
        cp = chemical_potentials.CompetingPhases(
            "ZrO2", e_above_hull=0.03, api_key=self.api_key, full_phase_diagram=True
        )
        self.assertEqual(len(cp.entries), 14)  # Zr4O now present
        self.assertEqual(cp.entries[0].name, "O2")
        self.assertEqual(cp.entries[0].data["total_magnetization"], 2)
        self.assertEqual(cp.entries[0].data["e_above_hull"], 0)
        self.assertTrue(cp.entries[0].data["molecule"])
        self.assertAlmostEqual(cp.entries[0].data["energy_per_atom"], -4.94795546875)
        self.assertAlmostEqual(cp.entries[0].data["energy"], -9.8959109375)
        self.assertEqual(cp.entries[1].name, "Zr")
        self.assertAlmostEqual(cp.entries[1].data["total_magnetization"], 0, places=3)
        self.assertEqual(cp.entries[1].data["e_above_hull"], 0)
        self.assertFalse(cp.entries[1].data["molecule"])
        self.assertEqual(cp.entries[2].name, "Zr3O")
        self.assertEqual(cp.entries[2].data["e_above_hull"], 0)
        self.assertEqual(cp.entries[3].name, "Zr4O")  # new entry!
        self.assertEqual(cp.entries[3].data["e_above_hull"], 0)
        self.assertEqual(cp.entries[4].name, "ZrO2")
        self.assertEqual(cp.entries[4].data["e_above_hull"], 0)
        self.assertEqual(cp.entries[5].name, "Zr3O")
        self.assertEqual(cp.entries[6].name, "Zr3O")
        self.assertEqual(cp.entries[7].name, "Zr2O")
        self.assertEqual(cp.entries[8].name, "ZrO2")
        self.assertEqual(cp.entries[9].name, "ZrO2")
        self.assertEqual(cp.entries[10].name, "Zr")
        self.assertEqual(cp.entries[11].name, "ZrO2")
        self.assertEqual(cp.entries[12].name, "ZrO2")
        self.assertEqual(cp.entries[13].name, "ZrO2")

    def test_init_ytos(self):
        # 144 phases on Y-Ti-O-S MP phase diagram
        cp = chemical_potentials.CompetingPhases(
            "Y2Ti2S2O5", e_above_hull=0.1, api_key=self.api_key
        )
        self.assertEqual(len(cp.entries), 115)  # 115 phases with default algorithm
        # assert only one O2 phase present (molecular entry):
        o2_entries = [e for e in cp.entries if e.name == "O2"]
        self.assertEqual(len(o2_entries), 1)
        self.assertEqual(o2_entries[0].name, "O2")
        self.assertEqual(o2_entries[0].data["total_magnetization"], 2)
        self.assertEqual(o2_entries[0].data["e_above_hull"], 0)
        self.assertTrue(o2_entries[0].data["molecule"])
        self.assertAlmostEqual(o2_entries[0].data["energy_per_atom"], -4.94795546875)

        cp = chemical_potentials.CompetingPhases(
            "Y2Ti2S2O5", e_above_hull=0.1, full_phase_diagram=True, api_key=self.api_key
        )
        self.assertEqual(
            len(cp.entries), 140
        )  # 144 phases on Y-Ti-O-S MP phase diagram,
        # 4 extra O2 phases removed

        # assert only one O2 phase present (molecular entry):
        o2_entries = [e for e in cp.entries if e.name == "O2"]
        self.assertEqual(len(o2_entries), 1)
        self.assertEqual(o2_entries[0].name, "O2")
        self.assertEqual(o2_entries[0].data["total_magnetization"], 2)
        self.assertEqual(o2_entries[0].data["e_above_hull"], 0)
        self.assertTrue(o2_entries[0].data["molecule"])
        self.assertAlmostEqual(o2_entries[0].data["energy_per_atom"], -4.94795546875)

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
            self.assertIn(nonvalid_api_key_error, e.exception)

        with self.assertRaises(ValueError) as e:
            new_api_key_error = ValueError(
                "You are trying to use the new Materials Project (MP) API which is not supported "
                "by doped. Please use the legacy MP API (https://legacy.materialsproject.org/open)."
            )
            chemical_potentials.CompetingPhases(
                "ZrO2",
                api_key="testabcdefghijklmnopqrstuvwxyz12",
            )
            self.assertIn(new_api_key_error, e.exception)

    def test_convergence_setup(self):
        # potcar spec doesnt need potcars set up for pmg and it still works
        self.cp.convergence_setup(potcar_spec=True)
        self.assertEqual(len(self.cp.metals), 6)
        self.assertEqual(self.cp.metals[0].data["band_gap"], 0)
        self.assertFalse(self.cp.nonmetals[0].data["molecule"])
        # this shouldnt exist - dont need to convergence test for molecules
        self.assertFalse(Path("competing_phases/O2_EaH_0").is_dir())

        # test if it writes out the files correctly
        path1 = "competing_phases/ZrO2_EaH_0.0088/kpoint_converge/k2,1,1/"
        self.assertTrue(Path(path1).is_dir())
        with open(f"{path1}/KPOINTS", "r") as file:
            contents = file.readlines()
            self.assertEqual(contents[3], "2 1 1\n")

        with open(f"{path1}/POTCAR.spec", "r") as file:
            contents = file.readlines()
            self.assertEqual(contents[0], "Zr_sv\n")

        with open(f"{path1}/INCAR", "r") as file:
            contents = file.readlines()
            self.assertTrue(any("GGA = Ps\n" == line for line in contents))
            self.assertTrue(any("NSW = 0\n" == line for line in contents))

    def test_vasp_std_setup(self):
        self.cp.vasp_std_setup(potcar_spec=True)
        self.assertEqual(len(self.cp.nonmetals), 6)
        self.assertEqual(len(self.cp.metals), 6)
        self.assertEqual(len(self.cp.molecules), 1)
        self.assertEqual(self.cp.molecules[0].name, "O2")
        self.assertEqual(self.cp.molecules[0].data["total_magnetization"], 2)
        self.assertTrue(self.cp.molecules[0].data["molecule"])
        self.assertFalse(self.cp.nonmetals[0].data["molecule"])

        path1 = "competing_phases/ZrO2_EaH_0/vasp_std/"
        self.assertTrue(Path(path1).is_dir())
        with open(f"{path1}/KPOINTS", "r") as file:
            contents = file.readlines()
            self.assertEqual(
                contents[0], "pymatgen with grid density = 911 / number of atoms\n"
            )
            self.assertEqual(contents[3], "4 4 4\n")

        with open(f"{path1}/POTCAR.spec", "r") as file:
            contents = file.readlines()
            self.assertEqual(contents, ["Zr_sv\n", "O"])

        with open(f"{path1}/INCAR", "r") as file:
            contents = file.readlines()
            self.assertEqual(contents[0], "AEXX = 0.25\n")
            self.assertEqual(contents[8], "ISIF = 3\n")
            self.assertEqual(contents[5], "GGA = Pe\n")

        path2 = "competing_phases/O2_EaH_0/vasp_std"
        self.assertTrue(Path(path2).is_dir())
        with open(f"{path2}/KPOINTS", "r") as file:
            contents = file.readlines()
            self.assertEqual(contents[3], "1 1 1\n")

        with open(f"{path2}/POSCAR", "r") as file:
            contents = file.readlines()
            self.assertEqual(contents[-1], "0.500000 0.500000 0.540667 O\n")


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
        self.assertEqual(len(self.ex_cp.entries), 2)
        self.assertEqual(self.ex_cp.entries[0].name, "La")  # definite ordering
        self.assertEqual(self.ex_cp.entries[1].name, "La2Zr2O7")  # definite ordering
        self.assertTrue(
            [entry.data["e_above_hull"] == 0 for entry in self.ex_cp.entries]
        )

        # names of intrinsic entries: ['O2', 'Zr', 'Zr3O', 'ZrO2']
        self.assertEqual(len(self.ex_cp.intrinsic_entries), 4)
        self.assertEqual(self.ex_cp.intrinsic_entries[0].name, "O2")
        self.assertEqual(self.ex_cp.intrinsic_entries[1].name, "Zr")
        self.assertEqual(self.ex_cp.intrinsic_entries[2].name, "Zr3O")
        self.assertEqual(self.ex_cp.intrinsic_entries[3].name, "ZrO2")
        self.assertTrue(
            all(
                entry.data["e_above_hull"] == 0
                for entry in self.ex_cp.intrinsic_entries
            )
        )

        cp = chemical_potentials.ExtrinsicCompetingPhases(
            "ZrO2", extrinsic_species="La", api_key=self.api_key
        )  # default e_above_hull=0.1
        self.assertEqual(len(cp.entries), 5)
        self.assertEqual(
            cp.entries[2].name, "La"
        )  # definite ordering, same 1st & 2nd as before
        self.assertEqual(cp.entries[3].name, "LaZr9O20")  # definite ordering
        self.assertEqual(cp.entries[4].name, "LaZr9O20")  # definite ordering
        self.assertTrue(
            all(entry.data["e_above_hull"] == 0 for entry in cp.entries[:2])
        )
        self.assertFalse(
            any(entry.data["e_above_hull"] == 0 for entry in cp.entries[2:])
        )
        self.assertEqual(len(cp.intrinsic_entries), 28)
