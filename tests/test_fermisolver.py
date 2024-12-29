"""
Tests for FermiSolver class in doped.thermodynamics module.
"""

import builtins
import itertools
import os
import unittest
import warnings

# Check if py_sc_fermi is available
from importlib.util import find_spec
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import FermiDos

from doped.thermodynamics import (
    FermiSolver,
    _get_py_sc_fermi_dos_from_fermi_dos,
    get_e_h_concs,
    get_fermi_dos,
)

py_sc_fermi_available = bool(find_spec("py_sc_fermi"))
module_path = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(module_path, "../examples")


class TestGetPyScFermiDosFromFermiDos(unittest.TestCase):
    """
    Tests for the ``_get_py_sc_fermi_dos_from_fermi_dos`` function.
    """

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos(self):
        """
        Test conversion of FermiDos to py_sc_fermi DOS with default parameters.
        """
        # Create a mock FermiDos object
        mock_fermi_dos = MagicMock(spec=FermiDos)
        mock_fermi_dos.densities = {
            Spin.up: np.array([1.0, 2.0, 3.0, 4.0]),
            Spin.down: np.array([0.5, 1.0, 1.5, 2.0]),
        }
        mock_fermi_dos.energies = np.array([0.0, 0.5, 1.0, 1.5])
        mock_fermi_dos.get_cbm_vbm.return_value = (1.0, 0.5)  # VBM = 0.5
        mock_fermi_dos.nelecs = 7.5  # Mock number of electrons
        mock_fermi_dos.get_gap.return_value = 0.4999  # Mock bandgap
        mock_fermi_dos.volume = 1
        mock_fermi_dos.idx_vbm = 1
        mock_fermi_dos.idx_cbm = 2
        mock_fermi_dos.de = (
            np.hstack((mock_fermi_dos.energies[1:], mock_fermi_dos.energies[-1])) - mock_fermi_dos.energies
        )
        e_vbm = mock_fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1]
        tdos = sum(mock_fermi_dos.densities.values())
        mock_fermi_dos.tdos = (
            tdos
            * mock_fermi_dos.nelecs
            / (tdos * mock_fermi_dos.de)[mock_fermi_dos.energies <= e_vbm].sum()
        )
        mock_fermi_dos.A_to_cm = 1e-8
        gap = mock_fermi_dos.get_gap(tol=1e-4, abs_tol=True)

        # Test with default values
        pyscfermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(mock_fermi_dos)
        print(pyscfermi_dos._n0_index())

        # Assertions
        assert pyscfermi_dos.nelect == mock_fermi_dos.nelecs
        assert pyscfermi_dos.bandgap == gap
        assert pyscfermi_dos.spin_polarised
        np.testing.assert_array_equal(pyscfermi_dos.edos, mock_fermi_dos.energies - e_vbm)

        # test carrier concentrations (indirectly tests DOS densities, this is the relevant property
        # from the DOS objects):
        pyscfermi_scale = 1e24 / mock_fermi_dos.volume
        for e_fermi, temperature in itertools.product(
            np.linspace(-0.25, gap + 0.25, 10), np.linspace(300, 1000.0, 10)
        ):
            pyscfermi_h_e = pyscfermi_dos.carrier_concentrations(e_fermi, temperature)  # rel to VBM
            doped_e_h = get_e_h_concs(mock_fermi_dos, e_fermi + e_vbm, temperature)  # raw Fermi eigenvalue
            assert np.allclose(
                (pyscfermi_h_e[1] * pyscfermi_scale, pyscfermi_h_e[0] * pyscfermi_scale),
                doped_e_h,
                rtol=0.25,
                atol=1e4,
            ), f"e_fermi={e_fermi}, temperature={temperature}"
            # tests: absolute(a - b) <= (atol + rtol * absolute(b)), so rtol of 15% but with a base atol
            # of 1e4 to allow larger relative mismatches for very small densities (more sensitive to
            # differences in integration schemes) -- main difference seems to be hard chopping of
            # integrals in py-sc-fermi at the expected VBM/CBM indices (but ``doped`` is agnostic to
            # these to improve robustness), makes more difference at low temperatures so only T >= 300K
            # tested here

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos_with_custom_parameters(self):
        """
        Test conversion with custom vbm, nelect, and bandgap.
        """
        # Create a mock FermiDos object
        mock_fermi_dos = MagicMock(spec=FermiDos)
        mock_fermi_dos.densities = {Spin.up: np.array([1.0, 2.0, 3.0])}
        mock_fermi_dos.energies = np.array([0.0, 0.5, 1.0])
        mock_fermi_dos.get_cbm_vbm.return_value = (None, 0.0)
        mock_fermi_dos.nelecs = 10
        mock_fermi_dos.get_gap.return_value = 0.5

        # Test with custom parameters
        pyscfermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(
            mock_fermi_dos, vbm=0.1, nelect=12, bandgap=0.5
        )

        # Assertions
        assert pyscfermi_dos.nelect == 12
        assert pyscfermi_dos.bandgap == 0.5
        np.testing.assert_array_equal(pyscfermi_dos.edos, np.array([-0.1, 0.4, 0.9]))
        assert not pyscfermi_dos.spin_polarised

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos_from_CdTe_dos(self):
        """
        Test conversion of FermiDos to py_sc_fermi DOS with default parameters.
        """
        fermi_dos = get_fermi_dos(
            os.path.join(EXAMPLE_DIR, "CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz")
        )
        pyscfermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(fermi_dos)
        assert pyscfermi_dos.nelect == fermi_dos.nelecs
        assert pyscfermi_dos.nelect == 18
        assert np.isclose(pyscfermi_dos.bandgap, fermi_dos.get_gap(tol=1e-4, abs_tol=True))
        assert np.isclose(pyscfermi_dos.bandgap, 1.526, atol=1e-3)
        assert not pyscfermi_dos.spin_polarised  # SOC DOS

        e_vbm = fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1]
        gap = fermi_dos.get_gap(tol=1e-4, abs_tol=True)
        np.testing.assert_array_equal(pyscfermi_dos.edos, fermi_dos.energies - e_vbm)

        # test carrier concentrations (indirectly tests DOS densities, this is the relevant property
        # from the DOS objects):
        pyscfermi_scale = 1e24 / fermi_dos.volume
        for e_fermi, temperature in itertools.product(
            np.linspace(-0.5, gap + 0.5, 10), np.linspace(300, 2000.0, 10)
        ):
            pyscfermi_h_e = pyscfermi_dos.carrier_concentrations(e_fermi, temperature)  # rel to VBM
            doped_e_h = get_e_h_concs(fermi_dos, e_fermi + e_vbm, temperature)  # raw Fermi eigenvalue
            assert np.allclose(
                (pyscfermi_h_e[1] * pyscfermi_scale, pyscfermi_h_e[0] * pyscfermi_scale),
                doped_e_h,
                rtol=0.15,
                atol=1e4,
            ), f"e_fermi={e_fermi}, temperature={temperature}"
            # tests: absolute(a - b) <= (atol + rtol * absolute(b)), so rtol of 15% but with a base atol
            # of 1e4 to allow larger relative mismatches for very small densities (more sensitive to
            # differences in integration schemes) -- main difference seems to be hard chopping of
            # integrals in py-sc-fermi at the expected VBM/CBM indices (but ``doped`` is agnostic to
            # these to improve robustness), makes more difference at low temperatures so only T >= 300K
            # tested here


# TODO: Use pytest fixtures to reduce code redundancy here?
class TestFermiSolverWithLoadedData(unittest.TestCase):
    """
    Tests for FermiSolver initialization with loaded data.
    """

    def setUp(self):
        # Adjust the paths to your actual data files
        self.defect_thermodynamics = loadfn("../examples/CdTe/CdTe_example_thermo.json")
        vasprun_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(vasprun_path)
        self.defect_thermodynamics.chempots = loadfn("../examples/CdTe/CdTe_chempots.json")
        self.solver_py_sc_fermi = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi", skip_check=True
        )
        self.solver_doped = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="doped", skip_check=True
        )
        # Mock the _DOS attribute for py-sc-fermi backend if needed
        self.solver_py_sc_fermi._DOS = MagicMock()

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_valid_initialization_doped_backend(self, mock_find_spec):
        """
        Test initialization with doped backend.
        """
        mock_find_spec.return_value = None  # Simulate py_sc_fermi not installed

        # Ensure bulk_dos is set
        assert self.defect_thermodynamics.bulk_dos is not None, "bulk_dos is not set."

        # Initialize FermiSolver
        solver = FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="doped")

        # Assertions
        assert solver.backend == "doped"
        assert solver.defect_thermodynamics == self.defect_thermodynamics
        assert solver.volume is not None

    @patch("doped.thermodynamics.importlib.util.find_spec")
    @patch("doped.thermodynamics.FermiSolver._activate_py_sc_fermi_backend")
    def test_valid_initialization_py_sc_fermi_backend(self, mock_activate_backend, mock_find_spec):
        """
        Test initialization with py-sc-fermi backend.
        """
        mock_find_spec.return_value = True  # Simulate py_sc_fermi installed
        mock_activate_backend.return_value = None

        # Initialize FermiSolver
        solver = FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi")

        # Assertions
        assert solver.backend == "py-sc-fermi"
        assert solver.defect_thermodynamics == self.defect_thermodynamics
        assert solver.volume is not None
        mock_activate_backend.assert_called_once()

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_missing_bulk_dos(self, mock_find_spec):
        """
        Test initialization failure due to missing bulk_dos.
        """
        mock_find_spec.return_value = None

        # Remove bulk_dos
        self.defect_thermodynamics.bulk_dos = None

        with pytest.raises(ValueError) as context:
            FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="doped")

        assert "No bulk DOS calculation" in str(context.value)

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_invalid_backend(self, mock_find_spec):
        """
        Test initialization failure due to invalid backend.
        """
        mock_find_spec.return_value = None

        with pytest.raises(ValueError) as context:
            FermiSolver(defect_thermodynamics=self.defect_thermodynamics, backend="invalid_backend")

        assert "Unrecognised `backend`" in str(context.value)

    def test_activate_backend_py_sc_fermi_installed(self):
        """
        Test backend activation when py_sc_fermi is installed.
        """
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
            self.solver_py_sc_fermi._activate_py_sc_fermi_backend()

            # Assertions
            assert self.solver_py_sc_fermi._DefectSystem is not None
            assert self.solver_py_sc_fermi._DefectSpecies is not None
            assert self.solver_py_sc_fermi._DefectChargeState is not None
            assert self.solver_py_sc_fermi._DOS is not None
            assert self.solver_py_sc_fermi.py_sc_fermi_dos is not None
            assert self.solver_py_sc_fermi.multiplicity_scaling is not None
            assert self.solver_py_sc_fermi._DefectSystem == DefectSystem
            assert self.solver_py_sc_fermi._DefectSpecies == DefectSpecies
            assert self.solver_py_sc_fermi._DefectChargeState == DefectChargeState
            assert self.solver_py_sc_fermi._DOS == DOS

    def test_activate_backend_py_sc_fermi_not_installed(self):
        """
        Test backend activation failure when py_sc_fermi is not installed.
        """
        original_import = builtins.__import__

        def mocked_import(name, globals, locals, fromlist, level):
            if name.startswith("py_sc_fermi"):
                raise ImportError("py-sc-fermi is not installed")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mocked_import):
            with pytest.raises(ImportError) as context:
                self.solver_py_sc_fermi._activate_py_sc_fermi_backend()

            assert "py-sc-fermi is not installed" in str(context.value)

    def test_activate_backend_error_message(self):
        """
        Test custom error message when backend activation fails.
        """
        original_import = builtins.__import__

        def mocked_import(name, globals, locals, fromlist, level):
            if name.startswith("py_sc_fermi"):
                raise ImportError("py-sc-fermi is not installed")
            return original_import(name, globals, locals, fromlist, level)

        error_message = "Custom error: py_sc_fermi activation failed"

        with patch("builtins.__import__", side_effect=mocked_import):
            with pytest.raises(ImportError) as context:
                self.solver_py_sc_fermi._activate_py_sc_fermi_backend(error_message=error_message)

            assert error_message in str(context.value)

    def test_activate_backend_non_integer_volume_scaling(self):
        """
        Test warning when volume scaling is non-integer.
        """
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

            with patch("doped.thermodynamics._get_py_sc_fermi_dos_from_fermi_dos", return_value=DOS()):
                # Set non-integer volume scaling
                self.solver_py_sc_fermi.volume = 100.0
                first_defect_entry = next(iter(self.defect_thermodynamics.defect_entries.values()))

                # Patch the volume property
                with patch.object(
                    type(first_defect_entry.sc_entry.structure), "volume", new_callable=PropertyMock
                ) as mock_volume:
                    mock_volume.return_value = 150.0

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        self.solver_py_sc_fermi._activate_py_sc_fermi_backend()

                        # Assertions
                        assert len(w) > 0
                        assert "non-integer" in str(w[-1].message)

    # Tests for _check_required_backend_and_error

    def test_check_required_backend_and_error_doped_missing_bulk_dos(self):
        """
        Test that RuntimeError is raised when bulk_dos is missing for doped
        backend.
        """
        # Remove bulk_dos to simulate missing DOS
        self.solver_doped.defect_thermodynamics.bulk_dos = None

        with pytest.raises(RuntimeError) as context:
            self.solver_doped._check_required_backend_and_error("doped")
        assert "This function is only supported for the doped backend" in str(context.value)

    def test_check_required_backend_and_error_doped_correct(self):
        """
        Test that no error is raised when bulk_dos is present for doped
        backend.
        """
        # bulk_dos is correctly set
        try:
            self.solver_doped._check_required_backend_and_error("doped")
        except RuntimeError as e:
            self.fail(f"RuntimeError raised unexpectedly: {e}")

    def test_check_required_backend_and_error_py_sc_fermi_missing_DOS(self):
        """
        Test that RuntimeError is raised when _DOS is missing for py-sc-fermi
        backend.
        """
        # Remove _DOS to simulate missing DOS
        self.solver_py_sc_fermi._DOS = None

        with pytest.raises(RuntimeError) as context:
            self.solver_py_sc_fermi._check_required_backend_and_error("py-sc-fermi")
        assert "This function is only supported for the py-sc-fermi backend" in str(context.value)

    def test_check_required_backend_and_error_py_sc_fermi_correct(self):
        """
        Test that no error is raised when _DOS is present for py-sc-fermi
        backend.
        """
        # _DOS is correctly set
        try:
            self.solver_py_sc_fermi._check_required_backend_and_error("py-sc-fermi")
        except RuntimeError as e:
            self.fail(f"RuntimeError raised unexpectedly: {e}")

    # Tests for _get_fermi_level_and_carriers

    def test_get_fermi_level_and_carriers(self):
        """
        Test _get_fermi_level_and_carriers returns correct values for doped
        backend.
        """
        # Use actual method
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")
        fermi_level, electrons, holes = self.solver_doped._get_fermi_level_and_carriers(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.defect_thermodynamics.el_refs,
            temperature=300,
            effective_dopant_concentration=None,
        )

        # Assertions
        assert isinstance(fermi_level, float)
        assert isinstance(electrons, float)
        assert isinstance(holes, float)

    # Tests for _get_single_chempot_dict

    def test_get_single_chempot_dict_correct(self):
        """
        Test that the correct chemical potential dictionary is returned.
        """
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")

        expected_chempots = self.defect_thermodynamics.chempots["limits_wrt_el_refs"]["CdTe-Te"]
        assert single_chempot_dict == expected_chempots
        assert el_refs == self.defect_thermodynamics.el_refs

    def test_get_single_chempot_dict_limit_not_found(self):
        """
        Test that ValueError is raised when the specified limit is not found.
        """
        with pytest.raises(ValueError) as context:
            self.solver_doped._get_single_chempot_dict(limit="nonexistent_limit")
        assert "Limit 'nonexistent_limit' not found" in str(context.value)

    # Tests for equilibrium_solve

    def test_equilibrium_solve_doped_backend(self):
        """
        Test equilibrium_solve method for doped backend.
        """
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
        assert "Fermi Level" in concentrations.columns
        assert "Electrons (cm^-3)" in concentrations.columns
        assert "Holes (cm^-3)" in concentrations.columns
        assert "Temperature" in concentrations.columns
        assert "Dopant (cm^-3)" in concentrations.columns
        # Check that concentrations are reasonable numbers
        assert np.all(concentrations["Concentration (cm^-3)"] >= 0)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element}" in concentrations.columns
            assert concentrations[f"μ_{element}"].iloc[0] == single_chempot_dict[element]

    def test_equilibrium_solve_py_sc_fermi_backend(self):
        """
        Test equilibrium_solve method for py-sc-fermi backend.
        """
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
        assert "Fermi Level" in concentrations.columns
        assert "Electrons (cm^-3)" in concentrations.columns
        assert "Holes (cm^-3)" in concentrations.columns
        assert "Temperature" in concentrations.columns
        assert "Dopant (cm^-3)" in concentrations.columns
        # Check defects are included
        assert "defect1" in concentrations.index
        assert "defect2" in concentrations.index
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element}" in concentrations.columns
            assert concentrations[f"μ_{element}"].iloc[0] == single_chempot_dict[element]

    # Tests for pseudo_equilibrium_solve

    def test_pseudo_equilibrium_solve_doped_backend(self):
        """
        Test pseudo_equilibrium_solve method for doped backend.
        """
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
        assert "Fermi Level" in concentrations.columns
        assert "Electrons (cm^-3)" in concentrations.columns
        assert "Holes (cm^-3)" in concentrations.columns
        assert "Annealing Temperature" in concentrations.columns
        assert "Quenched Temperature" in concentrations.columns
        # Check that concentrations are reasonable numbers
        assert np.all(concentrations["Concentration (cm^-3)"] >= 0)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element}" in concentrations.columns
            assert concentrations[f"μ_{element}"].iloc[0] == single_chempot_dict[element]

    def test_pseudo_equilibrium_solve_py_sc_fermi_backend(self):
        """
        Test pseudo_equilibrium_solve method for py-sc-fermi backend with
        fixed_defects.
        """
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
        assert "Fermi Level" in concentrations.columns
        assert "Electrons (cm^-3)" in concentrations.columns
        assert "Holes (cm^-3)" in concentrations.columns
        assert "Annealing Temperature" in concentrations.columns
        assert "Quenched Temperature" in concentrations.columns
        # Check defects are included
        assert "defect1" in concentrations.index
        assert "defect2" in concentrations.index
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element}" in concentrations.columns
            assert concentrations[f"μ_{element}"].iloc[0] == single_chempot_dict[element]

    # Tests for scan_temperature

    @patch("doped.thermodynamics.tqdm")
    def test_scan_temperature_equilibrium(self, mock_tqdm):
        """
        Test scan_temperature method under thermodynamic equilibrium.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        assert set(temperatures).issubset(concentrations["Temperature"].unique())

    @patch("doped.thermodynamics.tqdm")
    def test_scan_temperature_pseudo_equilibrium(self, mock_tqdm):
        """
        Test scan_temperature method under pseudo-equilibrium conditions.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        assert set(annealing_temperatures).issubset(concentrations["Annealing Temperature"].unique())
        assert set(quenched_temperatures).issubset(concentrations["Quenched Temperature"].unique())

    def test_scan_temperature_error_catch(self):
        with pytest.raises(ValueError) as exc:
            self.solver_doped.scan_temperature(
                annealing_temperature_range=[300, 350],
                temperature_range=[300, 400],
                chempots="Te-rich",
            )
        assert "Both ``annealing_temperature_range`` and ``temperature_range`` were set" in str(exc.value)

    # Tests for scan_dopant_concentration:
    @patch("doped.thermodynamics.tqdm")
    def test_scan_dopant_concentration_equilibrium(self, mock_tqdm):
        """
        Test scan_dopant_concentration method under thermodynamic equilibrium.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        assert set(dopant_concentrations).issubset(concentrations["Dopant (cm^-3)"].unique())

    @patch("doped.thermodynamics.tqdm")
    def test_scan_dopant_concentration_pseudo_equilibrium(self, mock_tqdm):
        """
        Test scan_dopant_concentration method under pseudo-equilibrium
        conditions.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        assert set(dopant_concentrations).issubset(concentrations["Dopant (cm^-3)"].unique())
        assert "Annealing Temperature" in concentrations.columns
        assert "Quenched Temperature" in concentrations.columns

    @patch("doped.thermodynamics.tqdm")
    def test_interpolate_chempots_with_limits(self, mock_tqdm):
        """
        Test interpolate_chempots method using limits.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        # Check that the concentrations have been calculated at n_points
        unique_chempot_sets = concentrations[
            [f"μ_{el}" for el in self.defect_thermodynamics.chempots["elemental_refs"]]
        ].drop_duplicates()
        assert len(unique_chempot_sets) == n_points

    @patch("doped.thermodynamics.tqdm")
    def test_interpolate_chempots_with_chempot_dicts(self, mock_tqdm):
        """
        Test interpolate_chempots method with manually specified chemical
        potentials.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        unique_chempot_sets = concentrations[["μ_Cd", "μ_Te"]].drop_duplicates()
        assert len(unique_chempot_sets) == n_points

    def test_interpolate_chempots_invalid_chempots_list_length(self):
        """
        Test that ValueError is raised when chempots list does not contain
        exactly two dictionaries.
        """
        with pytest.raises(ValueError):
            self.solver_doped.interpolate_chempots(
                n_points=5,
                chempots=[{"Cd": -0.5}],  # Only one chempot dict provided
                annealing_temperature=800,
                quenched_temperature=300,
            )

    def test_interpolate_chempots_missing_limits(self):
        """
        Test that ValueError is raised when limits are missing and chempots is
        in doped format.
        """
        with pytest.raises(ValueError):
            self.solver_doped.interpolate_chempots(
                n_points=5,
                chempots=self.defect_thermodynamics.chempots,
                annealing_temperature=800,
                quenched_temperature=300,
                limits=None,  # Limits are not provided
            )

    # Tests for min_max_X

    @patch("doped.thermodynamics.tqdm")
    def test_min_max_X_maximize_electrons(self, mock_tqdm):
        """
        Test min_max_X method to maximize electron concentration.
        """
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
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Electrons (cm^-3)" in result.columns

    @patch("doped.thermodynamics.tqdm")
    def test_min_max_X_minimize_holes(self, mock_tqdm):
        """
        Test min_max_X method to minimize hole concentration.
        """
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
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Holes (cm^-3)" in result.columns

    # Additional tests for internal methods

    def test_get_interpolated_chempots(self):
        """
        Test _get_interpolated_chempots method.
        """
        chempot_start = {"Cd": -0.5, "Te": -1.0}
        chempot_end = {"Cd": -1.0, "Te": -0.5}
        n_points = 3

        interpolated_chempots = self.solver_doped._get_interpolated_chempots(
            chempot_start, chempot_end, n_points
        )

        # Assertions
        assert len(interpolated_chempots) == n_points
        assert interpolated_chempots[0] == chempot_start
        assert interpolated_chempots[-1] == chempot_end
        # Check middle point
        expected_middle = {"Cd": -0.75, "Te": -0.75}
        assert interpolated_chempots[1] == expected_middle

    def test_parse_and_check_grid_like_chempots(self):
        """
        Test _parse_and_check_grid_like_chempots method.
        """
        chempots = self.defect_thermodynamics.chempots

        parsed_chempots, el_refs = self.solver_doped._parse_and_check_grid_like_chempots(chempots)

        # Assertions
        assert isinstance(parsed_chempots, dict)
        assert isinstance(el_refs, dict)
        assert "limits" in parsed_chempots
        assert "elemental_refs" in parsed_chempots

    def test_parse_and_check_grid_like_chempots_invalid_chempots(self):
        """
        Test that ValueError is raised when chempots is None.
        """
        # Temporarily remove chempots from defect_thermodynamics
        original_chempots = self.defect_thermodynamics.chempots
        self.defect_thermodynamics.chempots = None

        with pytest.raises(ValueError):
            self.solver_doped._parse_and_check_grid_like_chempots()

        # Restore original chempots
        self.defect_thermodynamics.chempots = original_chempots


class TestFermiSolverWithLoadedData3D(unittest.TestCase):
    """
    Tests for FermiSolver initialization with loaded data.
    """

    def setUp(self):
        # Adjust the paths to your actual data files
        self.defect_thermodynamics = loadfn("../examples/Cu2SiSe3/Cu2SiSe3_thermo.json")
        vasprun_path = "../examples/Cu2SiSe3/vasprun.xml.gz"
        self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(vasprun_path)
        self.defect_thermodynamics.chempots = loadfn("../examples/Cu2SiSe3/Cu2SiSe3_chempots.json")
        self.solver_py_sc_fermi = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="py-sc-fermi", skip_check=True
        )
        self.solver_doped = FermiSolver(
            defect_thermodynamics=self.defect_thermodynamics, backend="doped", skip_check=True
        )
        # Mock the _DOS attribute for py-sc-fermi backend if needed
        self.solver_py_sc_fermi._DOS = MagicMock()

    @patch("doped.thermodynamics.tqdm")
    def test_min_max_X_maximize_electrons(self, mock_tqdm):
        """
        Test min_max_X method to maximize electron concentration.
        """
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
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Electrons (cm^-3)" in result.columns

    @patch("doped.thermodynamics.tqdm")
    def test_min_max_X_minimize_holes(self, mock_tqdm):
        """
        Test min_max_X method to minimize hole concentration.
        """
        mock_tqdm.side_effect = lambda x: x

        # Call the method
        self.solver_doped.min_max_X(
            target="Holes (cm^-3)",
            min_or_max="min",
            annealing_temperature=800,
            quenched_temperature=300,
            tolerance=0.05,
            n_points=5,
            effective_dopant_concentration=1e16,
        )

    # Tests for scan_chemical_potential_grid

    @patch("doped.thermodynamics.tqdm")
    def test_scan_chemical_potential_grid(self, mock_tqdm):
        """
        Test scan_chemical_potential_grid method.
        """
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
        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        unique_chempot_sets = concentrations[
            [f"μ_{el}" for el in self.defect_thermodynamics.chempots["elemental_refs"]]
        ].drop_duplicates()
        assert len(unique_chempot_sets) > 0

    def test_scan_chemical_potential_grid_no_chempots(self):
        """
        Test that ValueError is raised when no chempots are provided and none
        are set.
        """
        # Temporarily remove chempots from defect_thermodynamics
        original_chempots = self.defect_thermodynamics.chempots
        self.defect_thermodynamics.chempots = None

        with pytest.raises(ValueError):
            self.solver_doped.scan_chemical_potential_grid(
                n_points=5,
                annealing_temperature=800,
                quenched_temperature=300,
            )

        # Restore original chempots
        self.defect_thermodynamics.chempots = original_chempots


if __name__ == "__main__":
    unittest.main()
