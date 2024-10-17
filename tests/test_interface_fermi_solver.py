
import unittest
from unittest.mock import MagicMock, patch, Mock, PropertyMock
import numpy as np
import builtins
import warnings
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.electronic_structure.core import Spin
from doped.thermodynamics import FermiSolver, get_py_sc_fermi_dos_from_fermi_dos
from monty.serialization import loadfn

# Check if py_sc_fermi is available
try:
    import py_sc_fermi

    py_sc_fermi_available = True
except ImportError:
    py_sc_fermi_available = False


class TestGetPyScFermiDosFromFermiDos(unittest.TestCase):
    """Tests for the get_py_sc_fermi_dos_from_fermi_dos function."""

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos(self):
        """Test conversion of FermiDos to py_sc_fermi DOS with default parameters."""
        from py_sc_fermi.dos import DOS

        # Create a mock FermiDos object
        mock_fermi_dos = MagicMock(spec=FermiDos)
        mock_fermi_dos.densities = {
            Spin.up: np.array([1.0, 2.0, 3.0]),
            Spin.down: np.array([0.5, 1.0, 1.5]),
        }
        mock_fermi_dos.energies = np.array([0.0, 0.5, 1.0])
        mock_fermi_dos.get_cbm_vbm.return_value = (None, 0.0)  # VBM = 0.0
        mock_fermi_dos.nelecs = 10  # Mock number of electrons
        mock_fermi_dos.get_gap.return_value = 0.5  # Mock bandgap

        # Test with default values
        result = get_py_sc_fermi_dos_from_fermi_dos(mock_fermi_dos)

        # Assertions
        self.assertEqual(result.nelect, 10)
        self.assertEqual(result.bandgap, 0.5)
        np.testing.assert_array_equal(result.edos, np.array([0.0, 0.5, 1.0]))
        self.assertTrue(result.spin_polarised)

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos_with_custom_parameters(self):
        """Test conversion with custom vbm, nelect, and bandgap."""
        from py_sc_fermi.dos import DOS

        # Create a mock FermiDos object
        mock_fermi_dos = MagicMock(spec=FermiDos)
        mock_fermi_dos.densities = {Spin.up: np.array([1.0, 2.0, 3.0])}
        mock_fermi_dos.energies = np.array([0.0, 0.5, 1.0])
        mock_fermi_dos.get_cbm_vbm.return_value = (None, 0.0)
        mock_fermi_dos.nelecs = 10
        mock_fermi_dos.get_gap.return_value = 0.5

        # Test with custom parameters
        result = get_py_sc_fermi_dos_from_fermi_dos(mock_fermi_dos, vbm=0.1, nelect=12, bandgap=0.5)

        # Assertions
        self.assertEqual(result.nelect, 12)
        self.assertEqual(result.bandgap, 0.5)
        np.testing.assert_array_equal(result.edos, np.array([-0.1, 0.4, 0.9]))
        self.assertFalse(result.spin_polarised)


class TestFermiSolverWithLoadedData(unittest.TestCase):
    """Tests for FermiSolver initialization with loaded data."""

    def setUp(self):
        # Adjust the paths to your actual data files
        self.defect_thermodynamics = loadfn("../examples/CdTe/CdTe_example_thermo.json")
        self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(
            "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        )
        self.defect_thermodynamics.chempots = loadfn("../examples/CdTe/CdTe_chempots.json")

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_valid_initialization_doped_backend(self, mock_find_spec):
        """Test initialization with doped backend."""
        mock_find_spec.return_value = None  # Simulate py_sc_fermi not installed

        # Ensure bulk_dos is set
        self.assertIsNotNone(self.defect_thermodynamics.bulk_dos, "bulk_dos is not set.")

        # Initialize FermiSolver
        solver = FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="doped")

        # Assertions
        self.assertEqual(solver.backend, "doped")
        self.assertEqual(solver.defect_thermodynamics, self.defect_thermodynamics)
        self.assertIsNotNone(solver.volume)

    @patch("doped.thermodynamics.importlib.util.find_spec")
    @patch("doped.thermodynamics.FermiSolver._activate_py_sc_fermi_backend")
    def test_valid_initialization_py_sc_fermi_backend(self, mock_activate_backend, mock_find_spec):
        """Test initialization with py-sc-fermi backend."""
        mock_find_spec.return_value = True  # Simulate py_sc_fermi installed
        mock_activate_backend.return_value = None

        # Initialize FermiSolver
        solver = FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi")

        # Assertions
        self.assertEqual(solver.backend, "py-sc-fermi")
        self.assertEqual(solver.defect_thermodynamics, self.defect_thermodynamics)
        self.assertIsNotNone(solver.volume)
        mock_activate_backend.assert_called_once()

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_missing_bulk_dos(self, mock_find_spec):
        """Test initialization failure due to missing bulk_dos."""
        mock_find_spec.return_value = None

        # Remove bulk_dos
        self.defect_thermodynamics.bulk_dos = None

        with self.assertRaises(ValueError) as context:
            FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="doped")

        self.assertIn("No bulk DOS calculation", str(context.exception))

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_invalid_backend(self, mock_find_spec):
        """Test initialization failure due to invalid backend."""
        mock_find_spec.return_value = None

        with self.assertRaises(ValueError) as context:
            FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="invalid_backend")

        self.assertIn("Unrecognised `backend`", str(context.exception))


class TestFermiSolverActivateBackend(unittest.TestCase):
    """Tests for FermiSolver's _activate_py_sc_fermi_backend method."""

    def setUp(self):
        # Adjust the paths to your actual data files
        self.defect_thermodynamics = loadfn("../examples/CdTe/CdTe_example_thermo.json")
        vasprun_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(vasprun_path)
        self.defect_thermodynamics.chempots = loadfn("../examples/CdTe/CdTe_chempots.json")
        self.solver = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi", skip_check=True
        )

    def test_activate_backend_py_sc_fermi_installed(self):
        """Test backend activation when py_sc_fermi is installed."""
        with patch.dict(
            "sys.modules",
            {
                "py_sc_fermi": MagicMock(),
                "py_sc_fermi.defect_charge_state": MagicMock(),
                "py_sc_fermi.defect_species": MagicMock(),
                "py_sc_fermi.defect_system": MagicMock(),
                "py_sc_fermi.dos": MagicMock(),
            },
        ):
            from py_sc_fermi.defect_charge_state import DefectChargeState
            from py_sc_fermi.defect_species import DefectSpecies
            from py_sc_fermi.defect_system import DefectSystem
            from py_sc_fermi.dos import DOS

            # Activate backend
            self.solver._activate_py_sc_fermi_backend()

            # Assertions
            self.assertIsNotNone(self.solver._DefectSystem)
            self.assertIsNotNone(self.solver._DefectSpecies)
            self.assertIsNotNone(self.solver._DefectChargeState)
            self.assertIsNotNone(self.solver._DOS)
            self.assertIsNotNone(self.solver.py_sc_fermi_dos)
            self.assertIsNotNone(self.solver.multiplicity_scaling)
            self.assertEqual(self.solver._DefectSystem, DefectSystem)
            self.assertEqual(self.solver._DefectSpecies, DefectSpecies)
            self.assertEqual(self.solver._DefectChargeState, DefectChargeState)
            self.assertEqual(self.solver._DOS, DOS)

    def test_activate_backend_py_sc_fermi_not_installed(self):
        """Test backend activation failure when py_sc_fermi is not installed."""
        original_import = builtins.__import__

        def mocked_import(name, globals, locals, fromlist, level):
            if name.startswith("py_sc_fermi"):
                raise ImportError("py-sc-fermi is not installed")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mocked_import):
            with self.assertRaises(ImportError) as context:
                self.solver._activate_py_sc_fermi_backend()

            self.assertIn("py-sc-fermi is not installed", str(context.exception))

    def test_activate_backend_error_message(self):
        """Test custom error message when backend activation fails."""
        original_import = builtins.__import__

        def mocked_import(name, globals, locals, fromlist, level):
            if name.startswith("py_sc_fermi"):
                raise ImportError("py-sc-fermi is not installed")
            return original_import(name, globals, locals, fromlist, level)

        error_message = "Custom error: py_sc_fermi activation failed"

        with patch("builtins.__import__", side_effect=mocked_import):
            with self.assertRaises(ImportError) as context:
                self.solver._activate_py_sc_fermi_backend(error_message=error_message)

            self.assertIn(error_message, str(context.exception))

    def test_activate_backend_non_integer_volume_scaling(self):
        """Test warning when volume scaling is non-integer."""
        with patch.dict(
            "sys.modules",
            {
                "py_sc_fermi": MagicMock(),
                "py_sc_fermi.defect_charge_state": MagicMock(),
                "py_sc_fermi.defect_species": MagicMock(),
                "py_sc_fermi.defect_system": MagicMock(),
                "py_sc_fermi.dos": MagicMock(),
            },
        ):
            from py_sc_fermi.dos import DOS

            with patch("doped.thermodynamics.get_py_sc_fermi_dos_from_fermi_dos", return_value=DOS()):
                # Set non-integer volume scaling
                self.solver.volume = 100.0
                first_defect_entry = next(iter(self.defect_thermodynamics.defect_entries.values()))

                # Patch the volume property
                with patch.object(
                    type(first_defect_entry.sc_entry.structure), "volume", new_callable=PropertyMock
                ) as mock_volume:
                    mock_volume.return_value = 150.0

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        self.solver._activate_py_sc_fermi_backend()

                        # Assertions
                        self.assertTrue(len(w) > 0)
                        self.assertIn("non-integer", str(w[-1].message))


class TestFermiSolverMethods(unittest.TestCase):
    """Tests for methods of the FermiSolver class."""

    def setUp(self):
        """Set up actual DefectThermodynamics object and FermiSolver instances."""
        # Load the DefectThermodynamics object from your data files
        self.defect_thermodynamics = loadfn("../examples/CdTe/CdTe_example_thermo.json")
        vasprun_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(vasprun_path)
        self.defect_thermodynamics.chempots = loadfn("../examples/CdTe/CdTe_chempots.json")

        # Create FermiSolver instances for both backends
        self.solver_doped = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics,
            backend="doped",
            skip_check=True,  # Skip VBM checks during tests
        )

        self.solver_py_sc_fermi = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi", skip_check=True
        )
        # Mock the _DOS attribute for py-sc-fermi backend if needed
        self.solver_py_sc_fermi._DOS = MagicMock()

    # Tests for _check_required_backend_and_error

    def test_check_required_backend_and_error_doped_missing_bulk_dos(self):
        """Test that RuntimeError is raised when bulk_dos is missing for doped backend."""
        # Remove bulk_dos to simulate missing DOS
        self.solver_doped.defect_thermodynamics.bulk_dos = None

        with self.assertRaises(RuntimeError) as context:
            self.solver_doped._check_required_backend_and_error("doped")
        self.assertIn("This function is only supported for the doped backend", str(context.exception))

    def test_check_required_backend_and_error_doped_correct(self):
        """Test that no error is raised when bulk_dos is present for doped backend."""
        # bulk_dos is correctly set
        try:
            self.solver_doped._check_required_backend_and_error("doped")
        except RuntimeError as e:
            self.fail(f"RuntimeError raised unexpectedly: {e}")

    def test_check_required_backend_and_error_py_sc_fermi_missing_DOS(self):
        """Test that RuntimeError is raised when _DOS is missing for py-sc-fermi backend."""
        # Remove _DOS to simulate missing DOS
        self.solver_py_sc_fermi._DOS = None

        with self.assertRaises(RuntimeError) as context:
            self.solver_py_sc_fermi._check_required_backend_and_error("py-sc-fermi")
        self.assertIn(
            "This function is only supported for the py-sc-fermi backend", str(context.exception)
        )

    def test_check_required_backend_and_error_py_sc_fermi_correct(self):
        """Test that no error is raised when _DOS is present for py-sc-fermi backend."""
        # _DOS is correctly set
        try:
            self.solver_py_sc_fermi._check_required_backend_and_error("py-sc-fermi")
        except RuntimeError as e:
            self.fail(f"RuntimeError raised unexpectedly: {e}")

    # Tests for _get_fermi_level_and_carriers

    def test_get_fermi_level_and_carriers(self):
        """Test _get_fermi_level_and_carriers returns correct values for doped backend."""
        # Use actual method
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")
        fermi_level, electrons, holes = self.solver_doped._get_fermi_level_and_carriers(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.defect_thermodynamics.el_refs,
            temperature=300,
            effective_dopant_concentration=None,
        )

        # Assertions
        self.assertIsInstance(fermi_level, float)
        self.assertIsInstance(electrons, float)
        self.assertIsInstance(holes, float)

    # Tests for _get_single_chempot_dict

    def test_get_single_chempot_dict_correct(self):
        """Test that the correct chemical potential dictionary is returned."""
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")

        expected_chempots = self.defect_thermodynamics.chempots["limits_wrt_el_refs"]["CdTe-Te"]
        self.assertEqual(single_chempot_dict, expected_chempots)
        self.assertEqual(el_refs, self.defect_thermodynamics.el_refs)

    def test_get_single_chempot_dict_limit_not_found(self):
        """Test that ValueError is raised when the specified limit is not found."""
        with self.assertRaises(ValueError) as context:
            self.solver_doped._get_single_chempot_dict(limit="nonexistent_limit")
        self.assertIn("Limit 'nonexistent_limit' not found", str(context.exception))

    # Tests for equilibrium_solve

    def test_equilibrium_solve_doped_backend(self):
        """Test equilibrium_solve method for doped backend."""
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")

        # Call the method
        concentrations = self.solver_doped.equilibrium_solve(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.defect_thermodynamics.el_refs,
            temperature=300,
            effective_dopant_concentration=1e16,
            append_chempots=True,
        )

        # Assertions
        self.assertIn("Fermi Level", concentrations.columns)
        self.assertIn("Electrons (cm^-3)", concentrations.columns)
        self.assertIn("Holes (cm^-3)", concentrations.columns)
        self.assertIn("Temperature", concentrations.columns)
        self.assertIn("Dopant (cm^-3)", concentrations.columns)
        # Check that concentrations are reasonable numbers
        self.assertTrue(np.all(concentrations["Concentration (cm^-3)"] >= 0))
        # Check appended chemical potentials
        for element in single_chempot_dict:
            self.assertIn(f"μ_{element}", concentrations.columns)
            self.assertEqual(concentrations[f"μ_{element}"].iloc[0], single_chempot_dict[element])

    def test_equilibrium_solve_py_sc_fermi_backend(self):
        """Test equilibrium_solve method for py-sc-fermi backend."""
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")

        # Mock _generate_defect_system
        self.solver_py_sc_fermi._generate_defect_system = MagicMock()
        # Mock defect_system object
        defect_system = MagicMock()
        defect_system.concentration_dict.return_value = {
            "Fermi Energy": 0.5,
            "n0": 1e18,
            "p0": 1e15,
            "defect1": 1e15,
            "defect2": 1e14,
            "Dopant": 1e16,
        }
        defect_system.temperature = 300

        self.solver_py_sc_fermi._generate_defect_system.return_value = defect_system

        # Call the method
        concentrations = self.solver_py_sc_fermi.equilibrium_solve(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.defect_thermodynamics.el_refs,
            temperature=300,
            effective_dopant_concentration=1e16,
            append_chempots=True,
        )

        # Assertions
        self.assertIn("Fermi Level", concentrations.columns)
        self.assertIn("Electrons (cm^-3)", concentrations.columns)
        self.assertIn("Holes (cm^-3)", concentrations.columns)
        self.assertIn("Temperature", concentrations.columns)
        self.assertIn("Dopant (cm^-3)", concentrations.columns)
        # Check defects are included
        self.assertIn("defect1", concentrations.index)
        self.assertIn("defect2", concentrations.index)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            self.assertIn(f"μ_{element}", concentrations.columns)
            self.assertEqual(concentrations[f"μ_{element}"].iloc[0], single_chempot_dict[element])


class TestEquilibriumCalculations(unittest.TestCase):
    """Tests for equilibrium and pseudo-equilibrium calculations in FermiSolver."""

    def setUp(self):
        """Set up actual DefectThermodynamics object and FermiSolver instances."""
        # Load the DefectThermodynamics object from your data files
        self.defect_thermodynamics = loadfn("../examples/CdTe/CdTe_example_thermo.json")
        vasprun_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(vasprun_path)
        self.defect_thermodynamics.chempots = loadfn("../examples/CdTe/CdTe_chempots.json")

        # Create FermiSolver instances for both backends
        self.solver_doped = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics,
            backend="doped",
            skip_check=True,  # Skip VBM checks during tests
        )

        self.solver_py_sc_fermi = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi", skip_check=True
        )
        # Mock the _DOS attribute for py-sc-fermi backend if needed
        self.solver_py_sc_fermi._DOS = MagicMock()

    # Tests for pseudo_equilibrium_solve

    def test_pseudo_equilibrium_solve_doped_backend(self):
        """Test pseudo_equilibrium_solve method for doped backend."""
        single_chempot_dict, el_refs = self.solver_doped._get_single_chempot_dict(limit="Te-rich")

        # Call the method
        concentrations = self.solver_doped.pseudo_equilibrium_solve(
            annealing_temperature=800,
            single_chempot_dict=single_chempot_dict,
            el_refs=el_refs,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
            append_chempots=True,
        )

        # Assertions
        self.assertIn("Fermi Level", concentrations.columns)
        self.assertIn("Electrons (cm^-3)", concentrations.columns)
        self.assertIn("Holes (cm^-3)", concentrations.columns)
        self.assertIn("Annealing Temperature", concentrations.columns)
        self.assertIn("Quenched Temperature", concentrations.columns)
        # Check that concentrations are reasonable numbers
        self.assertTrue(np.all(concentrations["Concentration (cm^-3)"] >= 0))
        # Check appended chemical potentials
        for element in single_chempot_dict:
            self.assertIn(f"μ_{element}", concentrations.columns)
            self.assertEqual(concentrations[f"μ_{element}"].iloc[0], single_chempot_dict[element])

    def test_pseudo_equilibrium_solve_py_sc_fermi_backend(self):
        """Test pseudo_equilibrium_solve method for py-sc-fermi backend with fixed_defects."""
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")

        # Mock _generate_annealed_defect_system
        self.solver_py_sc_fermi._generate_annealed_defect_system = MagicMock()
        # Mock defect_system object
        defect_system = MagicMock()
        defect_system.concentration_dict.return_value = {
            "Fermi Energy": 0.6,
            "n0": 1e17,
            "p0": 1e16,
            "defect1": 1e15,
            "defect2": 1e14,
            "Dopant": 1e16,
        }
        defect_system.temperature = 300

        self.solver_py_sc_fermi._generate_annealed_defect_system.return_value = defect_system

        # Call the method with fixed_defects
        concentrations = self.solver_py_sc_fermi.pseudo_equilibrium_solve(
            annealing_temperature=800,
            single_chempot_dict=single_chempot_dict,
            el_refs=el_refs,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
            fixed_defects={"defect1": 1e15},
            fix_charge_states=True,
            append_chempots=True,
        )

        # Assertions
        self.assertIn("Fermi Level", concentrations.columns)
        self.assertIn("Electrons (cm^-3)", concentrations.columns)
        self.assertIn("Holes (cm^-3)", concentrations.columns)
        self.assertIn("Annealing Temperature", concentrations.columns)
        self.assertIn("Quenched Temperature", concentrations.columns)
        # Check defects are included
        self.assertIn("defect1", concentrations.index)
        self.assertIn("defect2", concentrations.index)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            self.assertIn(f"μ_{element}", concentrations.columns)
            self.assertEqual(concentrations[f"μ_{element}"].iloc[0], single_chempot_dict[element])

    # Tests for scan_temperature

    @patch("doped.thermodynamics.tqdm")
    def test_scan_temperature_equilibrium(self, mock_tqdm):
        """Test scan_temperature method under thermodynamic equilibrium."""
        single_chempot_dict, el_refs = self.solver_doped._get_single_chempot_dict(limit="Te-rich")

        temperatures = [300, 400, 500]

        # Mock tqdm to return the temperatures directly
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        concentrations = self.solver_doped.scan_temperature(
            temperature_range=temperatures,
            chempots=single_chempot_dict,
            el_refs=el_refs,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        self.assertTrue(set(temperatures).issubset(concentrations["Temperature"].unique()))

    @patch("doped.thermodynamics.tqdm")
    def test_scan_temperature_pseudo_equilibrium(self, mock_tqdm):
        """Test scan_temperature method under pseudo-equilibrium conditions."""
        single_chempot_dict, el_refs = self.solver_doped._get_single_chempot_dict(limit="Te-rich")

        annealing_temperatures = [800, 900]
        quenched_temperatures = [300, 350]

        # Mock tqdm to return the product of temperatures directly
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        concentrations = self.solver_doped.scan_temperature(
            annealing_temperature_range=annealing_temperatures,
            quenched_temperature_range=quenched_temperatures,
            chempots=single_chempot_dict,
            el_refs=el_refs,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        self.assertTrue(
            set(annealing_temperatures).issubset(concentrations["Annealing Temperature"].unique())
        )
        self.assertTrue(
            set(quenched_temperatures).issubset(concentrations["Quenched Temperature"].unique())
        )

    # Tests for scan_dopant_concentration

    @patch("doped.thermodynamics.tqdm")
    def test_scan_dopant_concentration_equilibrium(self, mock_tqdm):
        """Test scan_dopant_concentration method under thermodynamic equilibrium."""
        single_chempot_dict, el_refs = self.solver_doped._get_single_chempot_dict(limit="Te-rich")

        dopant_concentrations = [1e15, 1e16, 1e17]

        # Mock tqdm to return the dopant concentrations directly
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        concentrations = self.solver_doped.scan_dopant_concentration(
            effective_dopant_concentration_range=dopant_concentrations,
            chempots=single_chempot_dict,
            el_refs=el_refs,
            temperature=300,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        self.assertTrue(set(dopant_concentrations).issubset(concentrations["Dopant (cm^-3)"].unique()))

    @patch("doped.thermodynamics.tqdm")
    def test_scan_dopant_concentration_pseudo_equilibrium(self, mock_tqdm):
        """Test scan_dopant_concentration method under pseudo-equilibrium conditions."""
        single_chempot_dict, el_refs = self.solver_doped._get_single_chempot_dict(limit="Te-rich")

        dopant_concentrations = [1e15, 1e16, 1e17]

        # Mock tqdm to return the dopant concentrations directly
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        concentrations = self.solver_doped.scan_dopant_concentration(
            effective_dopant_concentration_range=dopant_concentrations,
            chempots=single_chempot_dict,
            el_refs=el_refs,
            annealing_temperature=800,
            quenched_temperature=300,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        self.assertTrue(set(dopant_concentrations).issubset(concentrations["Dopant (cm^-3)"].unique()))
        self.assertIn("Annealing Temperature", concentrations.columns)
        self.assertIn("Quenched Temperature", concentrations.columns)

    @patch("doped.thermodynamics.tqdm")
    def test_interpolate_chempots_with_limits(self, mock_tqdm):
        """Test interpolate_chempots method using limits."""
        # Mock tqdm to avoid progress bar output
        mock_tqdm.side_effect = lambda x: x

        n_points = 5
        limits = ["Cd-rich", "Te-rich"]

        # Call the method
        concentrations = self.solver_doped.interpolate_chempots(
            n_points=n_points,
            limits=limits,
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        # Check that the concentrations have been calculated at n_points
        unique_chempot_sets = concentrations[
            [f"μ_{el}" for el in self.defect_thermodynamics.chempots["elemental_refs"]]
        ].drop_duplicates()
        self.assertEqual(len(unique_chempot_sets), n_points)

    @patch("doped.thermodynamics.tqdm")
    def test_interpolate_chempots_with_chempot_dicts(self, mock_tqdm):
        """Test interpolate_chempots method with manually specified chemical potentials."""
        mock_tqdm.side_effect = lambda x: x

        n_points = 3
        chempots_list = [
            {"Cd": -0.5, "Te": -1.0},
            {"Cd": -1.0, "Te": -0.5},
        ]

        # Call the method
        concentrations = self.solver_doped.interpolate_chempots(
            n_points=n_points,
            chempots=chempots_list,
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        unique_chempot_sets = concentrations[["μ_Cd", "μ_Te"]].drop_duplicates()
        self.assertEqual(len(unique_chempot_sets), n_points)

    def test_interpolate_chempots_invalid_chempots_list_length(self):
        """Test that ValueError is raised when chempots list does not contain exactly two dictionaries."""
        with self.assertRaises(ValueError):
            self.solver_doped.interpolate_chempots(
                n_points=5,
                chempots=[{"Cd": -0.5}],  # Only one chempot dict provided
                annealing_temperature=800,
                quenched_temperature=300,
            )

    def test_interpolate_chempots_missing_limits(self):
        """Test that ValueError is raised when limits are missing and chempots is in doped format."""
        with self.assertRaises(ValueError):
            self.solver_doped.interpolate_chempots(
                n_points=5,
                chempots=self.defect_thermodynamics.chempots,
                annealing_temperature=800,
                quenched_temperature=300,
                limits=None,  # Limits are not provided
            )

    # Tests for scan_chemical_potential_grid

    @patch("doped.thermodynamics.tqdm")
    def test_scan_chemical_potential_grid(self, mock_tqdm):
        """Test scan_chemical_potential_grid method."""
        mock_tqdm.side_effect = lambda x: x

        n_points = 5

        # Provide chempots with multiple limits
        chempots = loadfn("../examples/Cu2SiSe3/Cu2SiSe3_chempots.json")

        # Call the method
        concentrations = self.solver_doped.scan_chemical_potential_grid(
            chempots=chempots,
            n_points=n_points,
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(concentrations, pd.DataFrame)
        self.assertTrue(len(concentrations) > 0)
        unique_chempot_sets = concentrations[
            [f"μ_{el}" for el in self.defect_thermodynamics.chempots["elemental_refs"]]
        ].drop_duplicates()
        self.assertTrue(len(unique_chempot_sets) > 0)

    def test_scan_chemical_potential_grid_no_chempots(self):
        """Test that ValueError is raised when no chempots are provided and none are set."""
        # Temporarily remove chempots from defect_thermodynamics
        original_chempots = self.defect_thermodynamics.chempots
        self.defect_thermodynamics.chempots = None

        with self.assertRaises(ValueError):
            self.solver_doped.scan_chemical_potential_grid(
                n_points=5,
                annealing_temperature=800,
                quenched_temperature=300,
            )

        # Restore original chempots
        self.defect_thermodynamics.chempots = original_chempots

    # Tests for min_max_X

    @patch("doped.thermodynamics.tqdm")
    def test_min_max_X_maximize_electrons(self, mock_tqdm):
        """Test min_max_X method to maximize electron concentration."""
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        result = self.solver_doped.min_max_X(
            target="Electrons (cm^-3)",
            min_or_max="max",
            annealing_temperature=800,
            quenched_temperature=300,
            tolerance=0.05,
            n_points=5,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0)
        self.assertIn("Electrons (cm^-3)", result.columns)

    @patch("doped.thermodynamics.tqdm")
    def test_min_max_X_minimize_holes(self, mock_tqdm):
        """Test min_max_X method to minimize hole concentration."""
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        result = self.solver_doped.min_max_X(
            target="Holes (cm^-3)",
            min_or_max="min",
            annealing_temperature=800,
            quenched_temperature=300,
            tolerance=0.05,
            n_points=5,
            effective_dopant_concentration=1e16,
        )

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0)
        self.assertIn("Holes (cm^-3)", result.columns)

    # Additional tests for internal methods

    def test_get_interpolated_chempots(self):
        """Test _get_interpolated_chempots method."""
        chempot_start = {"Cd": -0.5, "Te": -1.0}
        chempot_end = {"Cd": -1.0, "Te": -0.5}
        n_points = 3

        interpolated_chempots = self.solver_doped._get_interpolated_chempots(
            chempot_start, chempot_end, n_points
        )

        # Assertions
        self.assertEqual(len(interpolated_chempots), n_points)
        self.assertEqual(interpolated_chempots[0], chempot_start)
        self.assertEqual(interpolated_chempots[-1], chempot_end)
        # Check middle point
        expected_middle = {"Cd": -0.75, "Te": -0.75}
        self.assertEqual(interpolated_chempots[1], expected_middle)

    def test_parse_and_check_grid_like_chempots(self):
        """Test _parse_and_check_grid_like_chempots method."""
        chempots = self.defect_thermodynamics.chempots

        parsed_chempots, el_refs = self.solver_doped._parse_and_check_grid_like_chempots(chempots)

        # Assertions
        self.assertIsInstance(parsed_chempots, dict)
        self.assertIsInstance(el_refs, dict)
        self.assertIn("limits", parsed_chempots)
        self.assertIn("elemental_refs", parsed_chempots)

    def test_parse_and_check_grid_like_chempots_invalid_chempots(self):
        """Test that ValueError is raised when chempots is None."""
        # Temporarily remove chempots from defect_thermodynamics
        original_chempots = self.defect_thermodynamics.chempots
        self.defect_thermodynamics.chempots = None

        with self.assertRaises(ValueError):
            self.solver_doped._parse_and_check_grid_like_chempots()

        # Restore original chempots
        self.defect_thermodynamics.chempots = original_chempots


if __name__ == "__main__":
    unittest.main()
