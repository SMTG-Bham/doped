import unittest

from monty.serialization import loadfn

from doped.interface.fermi_solver import _get_label_and_charge, FermiSolver
from doped.thermodynamics import DefectThermodynamics


def test_get_label_and_charge(self):
    self.assertEqual(_get_label_and_charge("defect_1"), ("defect", 1))
    self.assertEqual(_get_label_and_charge("defect_-1"), ("defect", -1))
    self.assertEqual(_get_label_and_charge("defect_0"), ("defect", 0))
    self.assertEqual(_get_label_and_charge("defect"), ("defect", None))


class TestFermiSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.thermodynamics = loadfn("../examples/CdTe/CdTe_thermo_v2.3_wout_meta.json")
        self.bulk_dos = "dos.vr"
        self.solver = FermiSolver(self.thermodynamics, self.bulk_dos)

    def test_initialization(self):
        self.assertEqual(self.solver.defect_thermodynamics, self.thermodynamics)
        self.assertEqual(self.solver.bulk_dos, self.bulk_dos)

    def test_equilibrium_solve(self):
        with self.assertRaises(NotImplementedError):
            self.solver.equilibrium_solve({"Fe": 1.0}, 300)

    def test_pseudo_equilibrium_solve(self):
        with self.assertRaises(NotImplementedError):
            self.solver.pseudo_equilibrium_solve({"Fe": 1.0}, 300, 400)

    def test_scan_dopant_concentration(self):
        with self.assertRaises(NotImplementedError):
            self.solver.scan_dopant_concentration({"Fe": 1.0}, 300, [1.0, 2.0])

    def test_scan_dopant_concentration_with_anneal_and_quench(self):
        with self.assertRaises(NotImplementedError):
            self.solver.scan_dopant_concentration_with_anneal_and_quench(
                {"Fe": 1.0}, [300], [400], [1.0, 2.0]
            )

    def test_scan_temperature(self):
        with self.assertRaises(NotImplementedError):
            self.solver.scan_temperature({"Fe": 1.0}, [300, 400])

    def test__solve_and_append_chemical_potentials(self):
        with self.assertRaises(NotImplementedError):
            self.solver._solve_and_append_chemical_potentials({"Fe": 1.0}, 300)

    def test__get_interpolated_chempots(self):
        start = {"Fe": 1.0}
        end = {"Fe": 2.0}
        n_points = 10

        interpolated_chem_pots, interpolation = self.solver._get_interpolated_chempots(
            start, end, n_points
        )

        self.assertEqual(len(interpolated_chem_pots), n_points)
        self.assertEqual(len(interpolation), n_points)

        for chem_pot in interpolated_chem_pots:
            self.assertTrue("Fe" in chem_pot)
            self.assertTrue(1.0 <= chem_pot["Fe"] <= 2.0)

    def test_interpolate_chemical_potentials_value_error(self):
        chem_pot_start = {"Fe": 1.0}
        chem_pot_end = {"Fe": 2.0}
        n_points = 10
        temperature = 300.0
        annealing_temperatures = [300, 400]

        with self.assertRaises(ValueError):
            self.solver.interpolate_chemical_potentials(
                chem_pot_start,
                chem_pot_end,
                n_points,
                temperature,
                annealing_temperatures=annealing_temperatures,
            )

        quenching_temperatures = [300, 400]

        with self.assertRaises(ValueError):
            self.solver.interpolate_chemical_potentials(
                chem_pot_start,
                chem_pot_end,
                n_points,
                temperature,
                quenching_temperatures=quenching_temperatures,
            )


if __name__ == "__main__":
    unittest.main()
