import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from py_sc_fermi.defect_system import DefectSystem
from doped.thermodynamics import DefectThermodynamics
from doped.interface.py_sc_fermi import _get_label_and_charge
from doped.interface.py_sc_fermi import FermiSolver


class TestGetLabelAndCharge(unittest.TestCase):
    def test_basic(self):
        label, charge = _get_label_and_charge("defect_1")
        self.assertEqual(label, "defect")
        self.assertEqual(charge, 1)

    def test_with_multiple_underscores(self):
        label, charge = _get_label_and_charge("defect_type_1")
        self.assertEqual(label, "defect_type")
        self.assertEqual(charge, 1)


class TestFermiSolver(unittest.TestCase):
    def setUp(self):
        self.defect_thermodynamics = DefectThermodynamics.from_json(
            "data/interface_pyscfermi_test/CdTe_example_thermo.json"
        )
        self.dos_vasprun = "data/interface_pyscfermi_test/vasprun_nsp.xml"
        self.solver = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, dos_vasprun=self.dos_vasprun, volume=None
        )

    def test_post_init(self):
        # Assert
        self.assertIsNotNone(self.solver.bulk_dos)
        self.assertIsNotNone(self.solver.volume)

    def test_defect_picker(self):
        # Arrange

        # Act
        result = self.solver._defect_picker("v_Cd_0")
        test_defect_entry = self.solver.defect_thermodynamics.defect_entries[0]

        # Assert
        self.assertEqual(result, test_defect_entry)

    def test_defect_system_from_chemical_potentials(self):
        # Arrange
        chemical_potentials = {"Cd": -2.0, "Te": -2.5}
        temperature = 300.0

        # Act
        defect_system = self.solver.defect_system_from_chemical_potentials(
            chemical_potentials, temperature
        )

        # Assert
        self.assertIsInstance(defect_system, DefectSystem)
        self.assertEqual(len(defect_system.defect_species), 4)
        self.assertEqual(defect_system.temperature, temperature)
        self.assertEqual(defect_system.volume, self.solver.volume)

    def test_update_defect_system_temperature(self):
        # Arrange
        chemical_potentials = {"Cd": -2.0, "Te": -2.5}
        temperature = 300.0
        defect_system = self.solver.defect_system_from_chemical_potentials(
            chemical_potentials, temperature
        )
        new_temperature = 400.0

        # Act
        updated_defect_system = self.solver.update_defect_system_temperature(
            defect_system, new_temperature
        )

        # Assert
        self.assertEqual(updated_defect_system.temperature, new_temperature)

    def test_scan_annealing_temperature(self):
        chemical_potentials = {"Cd": -2.0, "Te": -2.5}  # Fill this with real data
        temperature = 300  # Example value
        annealing_temperature_range = [500]  # Example values
        result = self.solver.scan_annealing_temperature(
            chemical_potentials, temperature, annealing_temperature_range
        )

        # Then you can make assertions about the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            result.columns.tolist(),
            ["Defect", "Concentration", "Temperature", "Anneal Temperature", "Fermi Level"],
        )
        self.assertEqual(result.shape, (6, 5))

    def test_scan_temperature(self):
        # Arrange
        chemical_potentials = {"Cd": -2.0, "Te": -2.5}
        temperature_range = [500]
        annealing_temperature = 300
        fix_defect_species = True
        exceptions = []
        file_name = None
        cpus = 1
        suppress_warnings = True

        # Act
        result = self.solver.scan_temperature(
            chemical_potentials,
            temperature_range,
            annealing_temperature,
            fix_defect_species,
            exceptions,
            file_name,
            cpus,
            suppress_warnings,
        )

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            result.columns.tolist(), ["Defect", "Concentration", "Temperature", "Fermi Level"]
        )
        self.assertEqual(result.shape, (6, 4))

    def test_get_concentrations(self):
        # Create a DefectSystem object with known properties
        class DefectSystem:
            @property
            def temperature(self):
                return 300

            def concentration_dict(self):
                return {
                    "Defect A": 1.0,
                    "Defect B": 2.0,
                    "Fermi Energy": 1.5,
                }

        defect_system = DefectSystem()

        # Call the method with the DefectSystem object
        concentration_data, fermi_level_data = FermiSolver._get_concentrations(defect_system)

        # Check the returned data
        self.assertEqual(
            concentration_data,
            [
                {"Defect": "Defect A", "Concentration": 1.0, "Temperature": 300},
                {"Defect": "Defect B", "Concentration": 2.0, "Temperature": 300},
            ],
        )
        self.assertEqual(
            fermi_level_data,
            [
                {"Fermi Level": 1.5, "Temperature": 300},
            ],
        )

    def test_interpolate_chemical_potentials(self):
        # Arrange
        chem_pot_start = {"Cd": -2.0, "Te": -2.5}
        chem_pot_end = {"Cd": -1.0, "Te": -1.5}
        n_points = 10
        annealing_temperature = 300
        fix_defect_species = True
        exceptions = []
        file_name = None
        cpus = 1
        suppress_warnings = True

        # Act
        result = self.solver.interpolate_chemical_potentials(
            chem_pot_start,
            chem_pot_end,
            n_points,
            annealing_temperature,
            fix_defect_species,
            exceptions,
            file_name,
            cpus,
            suppress_warnings,
        )

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        # add more assertions here...


if __name__ == "__main__":
    unittest.main()
