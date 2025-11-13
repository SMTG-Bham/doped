"""
Tests for the ``FermiSolver`` class in ``doped.thermodynamics``.

The ``scan_chemical_potential_grid`` tests here indirectly test the ``ChemicalPotentialGrid`` class in
``doped.chemical_potentials``.
"""

import builtins
import itertools
import os
import unittest
import warnings
from copy import deepcopy
from functools import wraps

# Check if py_sc_fermi is available
from importlib.util import find_spec
from unittest.mock import MagicMock, PropertyMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from monty.serialization import loadfn
from pymatgen.electronic_structure.dos import Dos, FermiDos, Spin
from test_thermodynamics import (
    STYLE,
    anneal_temperatures,
    belas_linear_fit,
    custom_mpl_image_compare,
    data_dir,
    reduced_anneal_temperatures,
)

from doped.thermodynamics import (
    DefectThermodynamics,
    FermiSolver,
    _get_py_sc_fermi_dos_from_fermi_dos,
    get_e_h_concs,
    get_fermi_dos,
    get_interpolated_chempots,
)
from doped.utils.plotting import format_defect_name

py_sc_fermi_available = bool(find_spec("py_sc_fermi"))
module_path = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(module_path, "../examples")


class TestGetPyScFermiDosFromFermiDos(unittest.TestCase):
    """
    Tests for the ``_get_py_sc_fermi_dos_from_fermi_dos`` function.
    """

    @classmethod
    def setUpClass(cls):
        cls.CdTe_fermi_dos = get_fermi_dos(
            os.path.join(EXAMPLE_DIR, "CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz")
        )

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos(self):
        """
        Test conversion of ``FermiDos`` to ``py_sc_fermi`` DOS with default
        parameters.
        """
        dos = Dos(
            energies=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
            densities={
                Spin.up: np.array([1.0, 2.0, 0.0, 0.0, 3.0, 4.0]),
                Spin.down: np.array([0.5, 1.0, 0.0, 0.0, 1.5, 2.0]),
            },
            efermi=0.75,
        )
        fermi_dos = FermiDos(dos, structure=self.CdTe_fermi_dos.structure)
        e_cbm, e_vbm = fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)
        gap = fermi_dos.get_gap(tol=1e-4, abs_tol=True)

        # Test with default values
        pyscfermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(fermi_dos)
        assert pyscfermi_dos.nelect == fermi_dos.nelecs
        assert pyscfermi_dos.bandgap == gap
        assert pyscfermi_dos.spin_polarised
        np.testing.assert_array_equal(pyscfermi_dos.edos, fermi_dos.energies - e_vbm)

        # test carrier concentrations (indirectly tests DOS densities, this is the relevant property
        # from the DOS objects):
        pyscfermi_scale = 1e24 / fermi_dos.volume
        for e_fermi, temperature in itertools.product(
            np.linspace(-0.25, gap + 0.25, 10), np.linspace(300, 1000.0, 10)
        ):
            pyscfermi_h_e = pyscfermi_dos.carrier_concentrations(e_fermi, temperature)  # rel to VBM
            doped_e_h = get_e_h_concs(fermi_dos, e_fermi + e_vbm, temperature)  # raw Fermi eigenvalue
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
        dos = Dos(
            energies=np.array([0.0, 0.5, 1.0]),
            densities={Spin.up: np.array([1.0, 2.0, 3.0])},
            efermi=0.1,
        )
        fermi_dos = FermiDos(dos, structure=self.CdTe_fermi_dos.structure)

        # Test with custom parameters; overrides values in the ``FermiDos`` object
        pyscfermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(fermi_dos, vbm=0.1, nelect=12, bandgap=0.5)
        assert pyscfermi_dos.nelect == 12
        assert pyscfermi_dos.bandgap == 0.5
        np.testing.assert_array_equal(pyscfermi_dos.edos, np.array([-0.1, 0.4, 0.9]))
        assert not pyscfermi_dos.spin_polarised

    @unittest.skipIf(not py_sc_fermi_available, "py_sc_fermi is not available")
    def test_get_py_sc_fermi_dos_from_CdTe_dos(self):
        """
        Test conversion of FermiDos to py_sc_fermi DOS with default parameters.
        """
        pyscfermi_dos = _get_py_sc_fermi_dos_from_fermi_dos(self.CdTe_fermi_dos)
        assert pyscfermi_dos.nelect == self.CdTe_fermi_dos.nelecs
        assert pyscfermi_dos.nelect == 18
        assert np.isclose(pyscfermi_dos.bandgap, self.CdTe_fermi_dos.get_gap(tol=1e-4, abs_tol=True))
        assert np.isclose(pyscfermi_dos.bandgap, 1.5263, atol=1e-3)
        assert not pyscfermi_dos.spin_polarised  # SOC DOS

        e_vbm = self.CdTe_fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1]
        gap = self.CdTe_fermi_dos.get_gap(tol=1e-4, abs_tol=True)
        np.testing.assert_array_equal(pyscfermi_dos.edos, self.CdTe_fermi_dos.energies - e_vbm)

        # test carrier concentrations (indirectly tests DOS densities, this is the relevant property
        # from the DOS objects):
        pyscfermi_scale = 1e24 / self.CdTe_fermi_dos.volume
        for e_fermi, temperature in itertools.product(
            np.linspace(-0.5, gap + 0.5, 10), np.linspace(300, 2000.0, 10)
        ):
            pyscfermi_h_e = pyscfermi_dos.carrier_concentrations(e_fermi, temperature)  # rel to VBM
            doped_e_h = get_e_h_concs(
                self.CdTe_fermi_dos, e_fermi + e_vbm, temperature
            )  # raw Fermi eigenvalue
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


def parameterize_backend():
    """
    A test decorator to allow easy running of ``FermiSolver`` tests with both
    the ``doped`` and ``py-sc-fermi`` backends.
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self, *args, **kwargs):
            for backend in ["doped", "py-sc-fermi"]:
                with self.subTest(backend=backend):
                    print(f"Testing with {backend} backend")
                    test_func(self, backend, *args, **kwargs)

        return wrapper

    return decorator


def check_concentrations_df(solver, concentrations, free_defects=None):
    """
    Convenience function to test that the defect concentrations in a given
    ``DataFrame`` match the corresponding Fermi levels and chemical potentials.

    Note: Will need to update when testing ``free_defects``, ``fixed_defects``.
    """
    free_defects = free_defects or []
    annealing = "Annealing Temperature (K)" in concentrations.columns
    formation_energy = None

    for defect, row in concentrations.iterrows():
        print(f"Checking {defect}")
        total_concentration = 0
        formal_chempots = {
            mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
        }
        this_row_formation_energy = sum(formal_chempots.values())
        if formation_energy is not None:
            assert np.isclose(this_row_formation_energy, formation_energy, atol=1e-4)
        formation_energy = this_row_formation_energy

        temperature = row["Annealing Temperature (K)"] if annealing else row["Temperature (K)"]
        if annealing:
            dopant_concentration = row["Dopant (cm^-3)"] if "Dopant (cm^-3)" in row.index else None
            fermi_level = solver.defect_thermodynamics.get_equilibrium_fermi_level(
                temperature=temperature,
                chempots=formal_chempots,
                effective_dopant_concentration=dopant_concentration,
            )

        for defect_entry in solver.defect_thermodynamics.all_entries[defect]:
            total_concentration += defect_entry.equilibrium_concentration(
                temperature=temperature,
                fermi_level=fermi_level if annealing else row["Fermi Level (eV wrt VBM)"],
                chempots=formal_chempots,
                el_refs=solver.defect_thermodynamics.el_refs,
            )

        # higher rtol required with large temperatures, concentrations more sensitive to rounded numbers:
        rtol = 1e-3 * (np.exp(temperature / 300))
        if defect in free_defects:
            assert row["Concentration (cm^-3)"] < total_concentration
        else:
            assert np.isclose(total_concentration, row["Concentration (cm^-3)"], rtol=rtol)


# TODO: Add actual tests for fixed_defects, free_defects and fix_charge_states
class TestFermiSolverWithLoadedData(unittest.TestCase):
    """
    Tests for ``FermiSolver`` initialization with loaded data.
    """

    @classmethod
    def setUpClass(cls):
        cls.CdTe_thermo = loadfn(os.path.join(EXAMPLE_DIR, "CdTe/CdTe_LZ_thermo_wout_meta.json.gz"))
        cls.CdTe_fermi_dos = get_fermi_dos(
            os.path.join(EXAMPLE_DIR, "CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz")
        )
        cls.CdTe_thermo.chempots = loadfn(os.path.join(EXAMPLE_DIR, "CdTe/CdTe_chempots.json"))

        cls.CdTe_700K_fermi_level, cls.CdTe_700K_e, cls.CdTe_700K_h = (
            cls.CdTe_thermo.get_equilibrium_fermi_level(
                limit="Cd-rich", temperature=700, return_concs=True
            )
        )
        cls.CdTe_700K_conc_df = cls.CdTe_thermo.get_equilibrium_concentrations(
            fermi_level=cls.CdTe_700K_fermi_level,
            limit="Cd-rich",
            temperature=700,
            per_charge=False,
            skip_formatting=True,
        )  # currently FermiSolver only supports per charge=False

        cls.CdTe_300K_eff_1e16_fermi_level, cls.CdTe_300K_eff_1e16_e, cls.CdTe_300K_eff_1e16_h = (
            cls.CdTe_thermo.get_equilibrium_fermi_level(
                limit="Te-rich", temperature=300, effective_dopant_concentration=1e16, return_concs=True
            )
        )
        cls.CdTe_300K_eff_1e16_conc_df = cls.CdTe_thermo.get_equilibrium_concentrations(
            fermi_level=cls.CdTe_300K_eff_1e16_fermi_level,
            limit="Te-rich",
            temperature=300,
            per_charge=False,
            skip_formatting=True,
        )  # currently FermiSolver only supports per charge=False

        (
            cls.CdTe_anneal_800K_eff_1e16_fermi_level,
            cls.CdTe_anneal_800K_eff_1e16_e,
            cls.CdTe_anneal_800K_eff_1e16_h,
            cls.CdTe_anneal_800K_eff_1e16_conc_df,
        ) = cls.CdTe_thermo.get_fermi_level_and_concentrations(
            annealing_temperature=800,
            effective_dopant_concentration=1e16,
            limit="Te-rich",
            per_charge=False,  # currently FermiSolver only supports per charge=False
            skip_formatting=True,
        )
        # drop Dopant row, included as column instead
        cls.CdTe_anneal_800K_eff_1e16_conc_df = cls.CdTe_anneal_800K_eff_1e16_conc_df.drop(
            "Dopant", errors="ignore"
        )
        cls.CdTe_anneal_800K_eff_1e16_conc_df["Dopant (cm^-3)"] = 1e16

        (
            cls.CdTe_anneal_1400K_quenched_150K_fermi_level,
            cls.CdTe_anneal_1400K_quenched_150K_e,
            cls.CdTe_anneal_1400K_quenched_150K_h,
            cls.CdTe_anneal_1400K_quenched_150K_conc_df,
        ) = cls.CdTe_thermo.get_fermi_level_and_concentrations(
            annealing_temperature=1400,
            quenched_temperature=150,
            limit="Cd-rich",
            per_charge=False,  # currently FermiSolver only supports per charge=False
            skip_formatting=True,
        )

        cls.Sb2S3_thermo = loadfn(os.path.join(EXAMPLE_DIR, "Sb2S3/Sb2S3_thermo.json.gz"))

    def setUp(self):
        self.CdTe_thermo.bulk_dos = self.CdTe_fermi_dos
        self.solver_py_sc_fermi = FermiSolver(
            defect_thermodynamics=self.CdTe_thermo, backend="py-sc-fermi"
        )
        self.solver_doped = FermiSolver(defect_thermodynamics=self.CdTe_thermo, backend="doped")
        # Mock the _DOS attribute for py-sc-fermi backend if needed
        self.solver_py_sc_fermi._DOS = MagicMock()

    def test_default_initialization(self):
        """
        Test default initialization, which uses ``doped`` backend.
        """
        solver = FermiSolver(defect_thermodynamics=self.CdTe_thermo)
        assert solver.backend == "doped"
        assert solver.defect_thermodynamics == self.CdTe_thermo
        assert solver.volume is not None

    @patch("doped.thermodynamics.importlib.util.find_spec")
    def test_valid_initialization_doped_backend(self, mock_find_spec):
        """
        Test initialization with ``doped`` backend.
        """
        mock_find_spec.return_value = None  # Simulate py_sc_fermi not installed

        # Ensure bulk_dos is set
        assert self.CdTe_thermo.bulk_dos is not None, "bulk_dos is not set."

        # Initialize FermiSolver
        solver = FermiSolver(defect_thermodynamics=self.CdTe_thermo, backend="doped")
        assert solver.backend == "doped"
        assert solver.defect_thermodynamics == self.CdTe_thermo
        assert solver.volume is not None

    @patch("doped.thermodynamics.importlib.util.find_spec")
    @patch("doped.thermodynamics.FermiSolver._activate_py_sc_fermi_backend")
    def test_valid_initialization_py_sc_fermi_backend(self, mock_activate_backend, mock_find_spec):
        """
        Test initialization with ``py-sc-fermi`` backend.
        """
        mock_find_spec.return_value = True  # Simulate py_sc_fermi installed
        mock_activate_backend.return_value = None

        # Initialize FermiSolver
        solver = FermiSolver(defect_thermodynamics=self.CdTe_thermo, backend="py-sc-fermi")
        assert solver.backend == "py-sc-fermi"
        assert solver.defect_thermodynamics == self.CdTe_thermo
        assert solver.volume is not None
        mock_activate_backend.assert_called_once()

    def test_missing_bulk_dos(self):
        """
        Test initialization failure due to missing bulk_dos.
        """
        self.CdTe_thermo.bulk_dos = None  # Remove bulk_dos

        with pytest.raises(ValueError) as context:
            FermiSolver(defect_thermodynamics=self.CdTe_thermo, backend="doped")

        assert "No bulk DOS calculation" in str(context.value)

    def test_invalid_backend(self):
        """
        Test initialization failure due to invalid backend.
        """
        with pytest.raises(ValueError) as context:
            FermiSolver(defect_thermodynamics=self.CdTe_thermo, backend="invalid_backend")

        assert "Unrecognised `backend`" in str(context.value)

    def test_activate_backend_py_sc_fermi_installed(self):
        """
        Test backend activation when ``py_sc_fermi`` is installed.
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

            assert self.solver_py_sc_fermi._DefectSystem == DefectSystem
            assert self.solver_py_sc_fermi._DefectSpecies == DefectSpecies
            assert self.solver_py_sc_fermi._DefectChargeState == DefectChargeState
            assert self.solver_py_sc_fermi._DOS == DOS
            assert self.solver_py_sc_fermi.py_sc_fermi_dos is not None
            assert self.solver_py_sc_fermi.multiplicity_scaling is not None

    def test_activate_backend_py_sc_fermi_not_installed(self):
        """
        Test backend activation failure when ``py_sc_fermi`` is not installed.
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
                first_defect_entry = next(iter(self.CdTe_thermo.defect_entries.values()))

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

    def test_check_required_backend_and_error_doped_correct(self):
        """
        Test that no error is raised with ``_check_required_backend_and_error``
        for ``doped`` backend.
        """
        try:
            self.solver_doped._check_required_backend_and_error("doped")
        except RuntimeError as e:
            self.fail(f"RuntimeError raised unexpectedly: {e}")

    def test_check_required_backend_and_error_py_sc_fermi_missing_DOS(self):
        """
        Test that ``RuntimeError`` is raised when ``_DOS`` is missing for ``py-
        sc-fermi`` backend.
        """
        # first test that no error is raised when _DOS is present
        self.solver_py_sc_fermi._check_required_backend_and_error("py-sc-fermi")

        # Remove _DOS to simulate missing DOS
        self.solver_py_sc_fermi._DOS = None

        with pytest.raises(RuntimeError) as context:
            self.solver_py_sc_fermi._check_required_backend_and_error("py-sc-fermi")
        assert "This function is currently only supported for the py-sc-fermi backend" in str(
            context.value
        )

    def test_check_required_backend_and_error_py_sc_fermi_doped_backend(self):
        """
        Test that ``RuntimeError`` is raised when
        ``_check_required_backend_and_error`` is called when ``py-sc-fermi``
        backend functionality is required, but the backend is set to ``doped``.
        """
        with pytest.raises(RuntimeError) as context:
            self.solver_doped._check_required_backend_and_error("py-sc-fermi")
        assert "This function is currently only supported for the py-sc-fermi backend" in str(
            context.value
        )

    def test_get_fermi_level_and_carriers(self):
        """
        Test ``_get_fermi_level_and_carriers`` returns correct values for
        ``doped`` backend.

        This method is only used for the ``doped`` backend.
        """
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")
        fermi_level, electrons, holes = self.solver_doped._get_fermi_level_and_carriers(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.CdTe_thermo.el_refs,
            temperature=300,
            effective_dopant_concentration=None,
        )

        doped_fermi_level, doped_e, doped_h = self.CdTe_thermo.get_equilibrium_fermi_level(
            limit="Te-rich", temperature=300, return_concs=True
        )
        assert np.isclose(fermi_level, doped_fermi_level, rtol=1e-3)
        assert np.isclose(electrons, doped_e, rtol=1e-3)
        assert np.isclose(holes, doped_h, rtol=1e-3)

    def test_get_single_chempot_dict_correct(self):
        """
        Test that the correct chemical potential dictionary is returned.
        """
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")
        assert single_chempot_dict == self.CdTe_thermo.chempots["limits_wrt_el_refs"]["CdTe-Te"]
        assert el_refs == self.CdTe_thermo.el_refs

    @parameterize_backend()
    def test_get_single_chempot_dict_limit_not_found(self, backend):
        """
        Test that ``ValueError`` is raised when the specified limit is not
        found.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as context:
            solver._get_single_chempot_dict(limit="nonexistent_limit")
        assert "Limit 'nonexistent_limit' not found" in str(context.value)

    @parameterize_backend()
    def test_equilibrium_solve(self, backend):
        """
        Test ``equilibrium_solve`` method for both backends.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Te-rich")

        concentrations = solver.equilibrium_solve(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.CdTe_thermo.el_refs,
            temperature=300,
            effective_dopant_concentration=1e16,
            append_chempots=True,
        )

        for i in [
            "Fermi Level (eV wrt VBM)",
            "Electrons (cm^-3)",
            "Holes (cm^-3)",
            "Temperature (K)",
        ]:
            assert i in concentrations.columns, f"Missing column: {i}"

        # Check that concentrations are reasonable numbers
        assert np.all(concentrations["Concentration (cm^-3)"] >= 0)
        assert np.isclose(concentrations["Temperature (K)"].iloc[0], 300)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element} (eV)" in concentrations.columns
            assert concentrations[f"μ_{element} (eV)"].iloc[0] == single_chempot_dict[element]

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        assert np.isclose(
            concentrations["Fermi Level (eV wrt VBM)"].iloc[0], self.CdTe_300K_eff_1e16_fermi_level
        )
        assert np.isclose(
            concentrations["Electrons (cm^-3)"].iloc[0], self.CdTe_300K_eff_1e16_e, rtol=1e-3
        )
        assert np.isclose(concentrations["Holes (cm^-3)"].iloc[0], self.CdTe_300K_eff_1e16_h, rtol=1e-3)

        pd.testing.assert_series_equal(
            self.CdTe_300K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

        # check against pseudo_equilibrium_solve
        pseudo_concentrations = solver.pseudo_equilibrium_solve(
            single_chempot_dict=single_chempot_dict,
            effective_dopant_concentration=1e16,
            annealing_temperature=300,  # set to 300 to match equilibrium_solve
        )
        pseudo_concentrations = pseudo_concentrations.rename(
            columns={"Annealing Temperature (K)": "Temperature (K)"}
        )  # retains ordering
        pseudo_concentrations = pseudo_concentrations.drop(columns=["Quenched Temperature (K)"])
        pd.testing.assert_frame_equal(
            concentrations, pseudo_concentrations, rtol=1e-3, check_dtype=False
        )  # Temperature can be int/float

    @parameterize_backend()
    def test_equilibrium_solve_700K_no_eff_dopant(self, backend):
        """
        Test ``equilibrium_solve`` method for both backends, this time with
        ``temperature=700``, no effective dopant and Cd-rich conditions.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Cd-rich")
        concentrations = solver.equilibrium_solve(
            single_chempot_dict=single_chempot_dict,
            el_refs=self.CdTe_thermo.el_refs,
            temperature=700,
            append_chempots=True,
        )

        for i in [
            "Fermi Level (eV wrt VBM)",
            "Electrons (cm^-3)",
            "Holes (cm^-3)",
            "Temperature (K)",
        ]:
            assert i in concentrations.columns, f"Missing column: {i}"

        assert np.isclose(concentrations["Temperature (K)"].iloc[0], 700)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element} (eV)" in concentrations.columns
            assert concentrations[f"μ_{element} (eV)"].iloc[0] == single_chempot_dict[element]

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        assert np.isclose(concentrations["Fermi Level (eV wrt VBM)"].iloc[0], self.CdTe_700K_fermi_level)
        assert np.isclose(concentrations["Electrons (cm^-3)"].iloc[0], self.CdTe_700K_e, rtol=1e-3)
        assert np.isclose(concentrations["Holes (cm^-3)"].iloc[0], self.CdTe_700K_h, rtol=1e-3)

        pd.testing.assert_series_equal(
            self.CdTe_700K_conc_df["Concentration (cm^-3)"],
            concentrations["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

        # check against pseudo_equilibrium_solve
        pseudo_concentrations = solver.pseudo_equilibrium_solve(
            single_chempot_dict=single_chempot_dict,
            annealing_temperature=700,  # match equilibrium_solve
            quenched_temperature=700,
        )
        pseudo_concentrations = pseudo_concentrations.rename(
            columns={"Annealing Temperature (K)": "Temperature (K)"}
        )  # retains ordering
        pseudo_concentrations = pseudo_concentrations.drop(columns=["Quenched Temperature (K)"])
        pd.testing.assert_frame_equal(
            concentrations, pseudo_concentrations, rtol=1e-3, check_dtype=False
        )  # Temperature can be int/float

    def test_equilibrium_solve_mocked_py_sc_fermi_backend(self):
        """
        Test equilibrium_solve method for a mocked ``py-sc-fermi`` backend (so
        test works even when ``py-sc-fermi`` is not installed).
        """
        single_chempot_dict, el_refs = self.solver_py_sc_fermi._get_single_chempot_dict(limit="Te-rich")

        # Mock _generate_defect_system
        self.solver_py_sc_fermi._generate_defect_system = MagicMock()
        # Mock defect_system object:
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
            el_refs=self.CdTe_thermo.el_refs,
            temperature=300,
            effective_dopant_concentration=1e16,
            append_chempots=True,
        )

        for i in [
            "Fermi Level (eV wrt VBM)",
            "Electrons (cm^-3)",
            "Holes (cm^-3)",
            "Temperature (K)",
        ]:
            assert i in concentrations.columns, f"Missing column: {i}"

        # Check defects are included
        assert "defect1" in concentrations.index
        assert "defect2" in concentrations.index
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element} (eV)" in concentrations.columns
            assert concentrations[f"μ_{element} (eV)"].iloc[0] == single_chempot_dict[element]

    @parameterize_backend()
    def test_pseudo_equilibrium_solve(self, backend):
        """
        Test ``pseudo_equilibrium_solve`` method for both backends.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Te-rich")

        concentrations = solver.pseudo_equilibrium_solve(
            annealing_temperature=800,
            single_chempot_dict=single_chempot_dict,
            el_refs=el_refs,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
            append_chempots=True,
        )

        for i in [
            "Fermi Level (eV wrt VBM)",
            "Electrons (cm^-3)",
            "Holes (cm^-3)",
            "Annealing Temperature (K)",
            "Quenched Temperature (K)",
        ]:
            assert i in concentrations.columns, f"Missing column: {i}"

        # Check that concentrations are reasonable numbers
        assert np.all(concentrations["Concentration (cm^-3)"] >= 0)
        assert np.isclose(concentrations["Quenched Temperature (K)"].iloc[0], 300)
        assert np.isclose(concentrations["Annealing Temperature (K)"].iloc[0], 800)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element} (eV)" in concentrations.columns
            assert concentrations[f"μ_{element} (eV)"].iloc[0] == single_chempot_dict[element]

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        assert np.isclose(
            concentrations["Fermi Level (eV wrt VBM)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_fermi_level
        )
        assert np.isclose(
            concentrations["Electrons (cm^-3)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_e, rtol=1e-3
        )
        assert np.isclose(
            concentrations["Holes (cm^-3)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_h, rtol=1e-3
        )

        pd.testing.assert_series_equal(
            self.CdTe_anneal_800K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

    @parameterize_backend()
    def test_pseudo_equilibrium_solve_1400K_no_eff_dopant(self, backend):
        """
        Test ``pseudo_equilibrium_solve`` method for both backends, now with
        ``annealing_temperature=1400``, ``quenched_temperature=150``,
        ``limit="Cd-rich"`` and no effective dopant.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Cd-rich")

        concentrations = solver.pseudo_equilibrium_solve(
            annealing_temperature=1400,
            single_chempot_dict=single_chempot_dict,
            el_refs=el_refs,
            quenched_temperature=150,
            append_chempots=True,
        )

        for i in [
            "Fermi Level (eV wrt VBM)",
            "Electrons (cm^-3)",
            "Holes (cm^-3)",
            "Annealing Temperature (K)",
            "Quenched Temperature (K)",
        ]:
            assert i in concentrations.columns, f"Missing column: {i}"

        # Check that concentrations are reasonable numbers
        assert np.all(concentrations["Concentration (cm^-3)"] >= 0)
        assert np.isclose(concentrations["Quenched Temperature (K)"].iloc[0], 150)
        assert np.isclose(concentrations["Annealing Temperature (K)"].iloc[0], 1400)
        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element} (eV)" in concentrations.columns
            assert concentrations[f"μ_{element} (eV)"].iloc[0] == single_chempot_dict[element]

        assert np.isclose(
            concentrations["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_1400K_quenched_150K_fermi_level,
        )
        assert np.isclose(
            concentrations["Electrons (cm^-3)"].iloc[0], self.CdTe_anneal_1400K_quenched_150K_e, rtol=1e-3
        )
        assert np.isclose(
            concentrations["Holes (cm^-3)"].iloc[0], self.CdTe_anneal_1400K_quenched_150K_h, rtol=1e-3
        )

        pd.testing.assert_series_equal(
            self.CdTe_anneal_1400K_quenched_150K_conc_df["Concentration (cm^-3)"],
            concentrations["Concentration (cm^-3)"],
            rtol=3e-3,  # higher rtol required with large annealing, low quenching w/py-sc-fermi
        )  # also checks the index and ordering

    def test_pseudo_equilibrium_solve_mocked_py_sc_fermi_backend(self):
        """
        Test ``pseudo_equilibrium_solve`` method for a mocked ``py-sc-fermi``
        backend (so test works even when ``py-sc-fermi`` is not installed),
        with ``fixed_defects``.
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

        for i in [
            "Fermi Level (eV wrt VBM)",
            "Electrons (cm^-3)",
            "Holes (cm^-3)",
            "Annealing Temperature (K)",
            "Quenched Temperature (K)",
        ]:
            assert i in concentrations.columns, f"Missing column: {i}"

        # Check defects are included
        assert "defect1" in concentrations.index
        assert "defect2" in concentrations.index

        assert np.isclose(concentrations["Quenched Temperature (K)"].iloc[0], 300)
        assert np.isclose(concentrations["Annealing Temperature (K)"].iloc[0], 800)

        # Check appended chemical potentials
        for element in single_chempot_dict:
            assert f"μ_{element} (eV)" in concentrations.columns
            assert concentrations[f"μ_{element} (eV)"].iloc[0] == single_chempot_dict[element]

    @parameterize_backend()
    def test_scan_temperature_equilibrium(self, backend):
        """
        Test ``scan_temperature`` method under thermodynamic equilibrium.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        temperatures = [200, 300, 400, 500, 2000]

        concentrations = solver.scan_temperature(
            temperature_range=temperatures,
            limit="Te-rich",  # test using limit option
            el_refs=self.CdTe_thermo.chempots["elemental_refs"],
            effective_dopant_concentration=1e16,
        )
        assert len(concentrations) > 0
        assert set(temperatures).issubset(concentrations["Temperature (K)"].unique())

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        concentrations_300K = concentrations[concentrations["Temperature (K)"] == 300]
        assert np.isclose(
            concentrations_300K["Fermi Level (eV wrt VBM)"].iloc[0], self.CdTe_300K_eff_1e16_fermi_level
        )
        assert np.isclose(
            concentrations_300K["Electrons (cm^-3)"].iloc[0], self.CdTe_300K_eff_1e16_e, rtol=1e-3
        )
        assert np.isclose(
            concentrations_300K["Holes (cm^-3)"].iloc[0], self.CdTe_300K_eff_1e16_h, rtol=1e-3
        )

        pd.testing.assert_series_equal(
            self.CdTe_300K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations_300K["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

        # v_Cd concentration is basically the same at all <=500K temperatures here, as it's the only
        # significant compensating species to the effective dopant concentration:
        concentrations_lte_500K = concentrations[concentrations["Temperature (K)"] <= 500]
        assert np.allclose(concentrations_lte_500K.loc["v_Cd", "Concentration (cm^-3)"], 5e15, rtol=1e-3)
        for defect in set(concentrations.index.values):
            print(f"Checking {defect}")
            assert np.isclose(
                concentrations[concentrations["Temperature (K)"] == 200].loc[
                    defect, "Concentration (cm^-3)"
                ],
                concentrations[concentrations["Temperature (K)"] == 400].loc[
                    defect, "Concentration (cm^-3)"
                ],
                atol=1e-40,
                rtol=1e-3,
            ) == (defect == "v_Cd")

    @parameterize_backend()
    def test_scan_temperature_pseudo_equilibrium(self, backend):
        """
        Test ``scan_temperature`` method under pseudo-equilibrium conditions.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Te-rich")
        annealing_temperatures = [800, 900]
        quenched_temperatures = [300, 350]

        concentrations = solver.scan_temperature(
            annealing_temperature_range=annealing_temperatures,
            quenched_temperature_range=quenched_temperatures,
            chempots=single_chempot_dict,
            el_refs=el_refs,
            effective_dopant_concentration=1e16,
        )

        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        assert set(annealing_temperatures).issubset(concentrations["Annealing Temperature (K)"].unique())
        assert set(quenched_temperatures).issubset(concentrations["Quenched Temperature (K)"].unique())

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        concentrations_800K = concentrations[
            (concentrations["Annealing Temperature (K)"] == 800)
            & (concentrations["Quenched Temperature (K)"] == 300)
        ]
        assert np.isclose(
            concentrations_800K["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_800K_eff_1e16_fermi_level,
        )
        assert np.isclose(
            concentrations_800K["Electrons (cm^-3)"].iloc[0],
            self.CdTe_anneal_800K_eff_1e16_e,
            rtol=1e-3,
        )
        assert np.isclose(
            concentrations_800K["Holes (cm^-3)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_h, rtol=1e-3
        )
        pd.testing.assert_series_equal(
            self.CdTe_anneal_800K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations_800K["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

    @parameterize_backend()
    def test_scan_temperature_pseudo_equilibrium_free_defects(self, backend):
        """
        Test ``scan_temperature`` method under pseudo-equilibrium conditions,
        using the ``free_defects`` parameter.

        Note that here we use the ``free_defects`` parameter which is currently
        only supported by the ``py-sc-fermi`` backend, but this is fine for
        both initial backends, as we automatically switch to the ``py-sc-fermi``
        backend if required (and ``py-sc-fermi`` installed).
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Te-rich")
        annealing_temperatures = [800, 900]
        quenched_temperatures = [300, 350]

        with warnings.catch_warnings(record=True) as w:
            concentrations = solver.scan_temperature(
                annealing_temperature_range=annealing_temperatures,
                quenched_temperature_range=quenched_temperatures,
                chempots=single_chempot_dict,
                el_refs=el_refs,
                free_defects=["v_Cd"],
            )
        print([str(warning.message) for warning in w])  # for debugging
        assert not w

        assert isinstance(concentrations, pd.DataFrame)
        assert len(concentrations) > 0
        assert set(annealing_temperatures).issubset(concentrations["Annealing Temperature (K)"].unique())
        assert set(quenched_temperatures).issubset(concentrations["Quenched Temperature (K)"].unique())

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations, free_defects=["v_Cd"])

        # test Fermi level, carrier and defect concentrations against known values
        concentrations_800K = concentrations[
            (concentrations["Annealing Temperature (K)"] == 800)
            & (concentrations["Quenched Temperature (K)"] == 300)
        ]
        assert not np.isclose(  # changed from before
            concentrations_800K["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_800K_eff_1e16_fermi_level,
        )
        assert np.isclose(
            concentrations_800K["Fermi Level (eV wrt VBM)"].iloc[0], 0.936, atol=1e-3
        )  # much more n-type now, with v_Cd as free defect

    @parameterize_backend()
    def test_scan_func_error_catch(self, backend):
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as exc:
            solver.scan_temperature(
                annealing_temperature_range=[300, 350],
                temperature_range=[300, 400],
            )
        assert "Both ``annealing_temperature_range`` and ``temperature_range`` were set" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            solver.scan_temperature(
                quenched_temperature_range=[300, 350],
            )
        assert "Quenched temperature was set but no annealing temperature was given!" in str(exc.value)

        # test for other methods:
        for func, additional_kwargs in {
            "scan_dopant_concentration": {"effective_dopant_concentration_range": [1e15, 1e16, 1e17]},
            "interpolate_chempots": {"limits": ["Cd-rich", "Te-rich"]},
            "scan_chempots": {},
            "scan_chemical_potential_grid": {},
            "optimise": {"target": "Electrons (cm^-3)"},
        }.items():
            with pytest.raises(ValueError) as exc:
                getattr(solver, func)(
                    annealing_temperature=400,
                    temperature=400,
                    **additional_kwargs,
                )
            assert "Both ``annealing_temperature`` and ``temperature`` were set" in str(exc.value)

            with pytest.raises(ValueError) as exc:
                getattr(solver, func)(
                    quenched_temperature=400,
                    **additional_kwargs,
                )
            assert "Quenched temperature was set but no annealing temperature was given!" in str(exc.value)

    @parameterize_backend()
    def test_scan_temperature_no_chempots_error_catch(self, backend):
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as exc:
            solver.scan_temperature(
                annealing_temperature_range=[300, 350],
            )
        assert (
            "Limit 'None' not found in the chemical potentials dictionary! You must specify an "
            "appropriate limit or provide an appropriate chempots dictionary"
        ) in str(exc.value)

    @parameterize_backend()
    def test_scan_dopant_concentration_no_chempots_error_catch(self, backend):
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as exc:
            solver.scan_dopant_concentration(
                annealing_temperature=400,
                effective_dopant_concentration_range=[1e16, 1e18],
            )
        assert (
            "Limit 'None' not found in the chemical potentials dictionary! You must specify an "
            "appropriate limit or provide an appropriate chempots dictionary"
        ) in str(exc.value)

    # Note that the following methods use the ``self.defect_thermodynamics.chempots`` by default,
    # and so do not throw an error if no chempots are provided:
    # scan_chempots, scan_chemical_potential_grid, optimise

    @parameterize_backend()
    def test_scan_chemical_potential_grid_non_2D_data(self, backend):
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as exc:
            solver.scan_chemical_potential_grid()
        assert (
            "Chemical potential grid generation is only possible for systems with "
            "two or more independent variables (chemical potentials), i.e. ternary or "
            "higher-dimensional systems. Stable chemical potential ranges are just a line for binary "
            "systems, for which ``FermiSolver.interpolate_chempots()`` can be used." in str(exc.value)
        )

    @parameterize_backend()
    def test_scan_dopant_concentration_equilibrium(self, backend):
        """
        Test ``scan_dopant_concentration`` method under thermodynamic
        equilibrium.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Te-rich")

        dopant_concentrations = [0, 1e15, 1e16, 1e17]
        concentrations = solver.scan_dopant_concentration(
            effective_dopant_concentration_range=dopant_concentrations,
            limit="Te-rich",
            el_refs=el_refs,
            temperature=300,
        )
        assert len(concentrations) > 0
        assert set(dopant_concentrations).issubset(concentrations["Dopant (cm^-3)"].unique())

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        concentrations_1e16 = concentrations[concentrations["Dopant (cm^-3)"] == 1e16]
        assert np.isclose(
            concentrations_1e16["Fermi Level (eV wrt VBM)"].iloc[0], self.CdTe_300K_eff_1e16_fermi_level
        )
        assert np.isclose(
            concentrations_1e16["Electrons (cm^-3)"].iloc[0], self.CdTe_300K_eff_1e16_e, rtol=1e-3
        )
        assert np.isclose(
            concentrations_1e16["Holes (cm^-3)"].iloc[0], self.CdTe_300K_eff_1e16_h, rtol=1e-3
        )

        pd.testing.assert_series_equal(
            self.CdTe_300K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations_1e16["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

        # test dopant concentration = 0 values:
        concentrations = solver.scan_dopant_concentration(
            effective_dopant_concentration_range=dopant_concentrations,
            limit="Cd-rich",  # Cd-rich for 700 K no eff dopant tests
            temperature=700,
        )
        concentrations_0 = concentrations[concentrations["Dopant (cm^-3)"] == 0]
        assert np.isclose(concentrations_0["Fermi Level (eV wrt VBM)"].iloc[0], self.CdTe_700K_fermi_level)
        assert np.isclose(concentrations_0["Electrons (cm^-3)"].iloc[0], self.CdTe_700K_e, rtol=1e-3)
        assert np.isclose(concentrations_0["Holes (cm^-3)"].iloc[0], self.CdTe_700K_h, rtol=1e-3)
        pd.testing.assert_series_equal(
            self.CdTe_700K_conc_df["Concentration (cm^-3)"],
            concentrations_0["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

    @parameterize_backend()
    def test_scan_dopant_concentration_pseudo_equilibrium(self, backend):
        """
        Test ``scan_dopant_concentration`` method under pseudo-equilibrium
        conditions.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Te-rich")
        dopant_concentrations = [0, 1e15, 1e16, 1e17]

        concentrations = solver.scan_dopant_concentration(
            effective_dopant_concentration_range=dopant_concentrations,
            chempots=single_chempot_dict,
            el_refs=el_refs,
            annealing_temperature=800,
            quenched_temperature=300,
        )
        assert len(concentrations) > 0
        assert set(dopant_concentrations).issubset(concentrations["Dopant (cm^-3)"].unique())
        assert "Annealing Temperature (K)" in concentrations.columns
        assert "Quenched Temperature (K)" in concentrations.columns
        assert "Temperature (K)" not in concentrations.columns

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        concentrations_1e16 = concentrations[concentrations["Dopant (cm^-3)"] == 1e16]
        assert np.isclose(
            concentrations_1e16["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_800K_eff_1e16_fermi_level,
        )
        assert np.isclose(
            concentrations_1e16["Electrons (cm^-3)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_e, rtol=1e-3
        )
        assert np.isclose(
            concentrations_1e16["Holes (cm^-3)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_h, rtol=1e-3
        )

        pd.testing.assert_series_equal(
            self.CdTe_anneal_800K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations_1e16["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

        # test dopant concentration = 0 values:
        concentrations = solver.scan_dopant_concentration(
            effective_dopant_concentration_range=dopant_concentrations,
            limit="Cd-rich",  # Cd-rich for no eff dopant tests
            annealing_temperature=1400,
            quenched_temperature=150,
        )
        concentrations_0 = concentrations[concentrations["Dopant (cm^-3)"] == 0]
        assert np.isclose(
            concentrations_0["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_1400K_quenched_150K_fermi_level,
        )
        assert np.isclose(
            concentrations_0["Electrons (cm^-3)"].iloc[0],
            self.CdTe_anneal_1400K_quenched_150K_e,
            rtol=1e-3,
        )
        assert np.isclose(
            concentrations_0["Holes (cm^-3)"].iloc[0], self.CdTe_anneal_1400K_quenched_150K_h, rtol=1e-3
        )
        pd.testing.assert_series_equal(
            self.CdTe_anneal_1400K_quenched_150K_conc_df["Concentration (cm^-3)"],
            concentrations_0["Concentration (cm^-3)"],
            rtol=3e-3,  # higher rtol required with large annealing, low quenching w/py-sc-fermi
        )  # also checks the index and ordering

    @parameterize_backend()
    def test_interpolate_chempots_with_limits(self, backend):
        """
        Test ``interpolate_chempots()`` using limits.

        Note that this and the other ``interpolate_chempots()`` tests
        implicitly test the simpler ``scan_chempots()`` method, as it
        is used within this function.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        n_points = 5
        limits = ["Cd-rich", "Te-rich"]

        concentrations = solver.interpolate_chempots(
            n_points=n_points,
            limits=limits,
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
        )
        assert len(concentrations) > 0
        # Check that the concentrations have been calculated at n_points
        unique_chempot_sets = concentrations[
            [f"μ_{el} (eV)" for el in self.CdTe_thermo.chempots["elemental_refs"]]
        ].drop_duplicates()
        assert len(unique_chempot_sets) == n_points

        for mu_col in [col for col in concentrations.columns if "μ_" in col]:
            elt = mu_col.strip("μ_").split()[0]
            chempot_vals = [
                limit_dict[elt]
                for limit_dict in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values()
            ]
            assert set(np.round(unique_chempot_sets[mu_col].to_numpy(), 3)) == set(
                np.round(
                    np.linspace(
                        min(chempot_vals),
                        max(chempot_vals),
                        n_points,
                    ),
                    3,
                )
            )

        # check concentrations match Fermi level and chemical potentials:
        check_concentrations_df(solver, concentrations)

        # test Fermi level, carrier and defect concentrations against known values
        concentrations_Te_rich = concentrations[concentrations["μ_Te (eV)"] == 0]
        assert np.isclose(
            concentrations_Te_rich["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_800K_eff_1e16_fermi_level,
        )
        assert np.isclose(
            concentrations_Te_rich["Electrons (cm^-3)"].iloc[0],
            self.CdTe_anneal_800K_eff_1e16_e,
            rtol=1e-3,
        )
        assert np.isclose(
            concentrations_Te_rich["Holes (cm^-3)"].iloc[0], self.CdTe_anneal_800K_eff_1e16_h, rtol=1e-3
        )

        pd.testing.assert_series_equal(
            self.CdTe_anneal_800K_eff_1e16_conc_df["Concentration (cm^-3)"],
            concentrations_Te_rich["Concentration (cm^-3)"],
            rtol=1e-3,
        )  # also checks the index and ordering

        # test dopant concentration = 0 values:
        concentrations = solver.interpolate_chempots(
            n_points=n_points,
            limits=limits,
            annealing_temperature=1400,
            quenched_temperature=150,
        )
        concentrations_Cd_rich = concentrations[concentrations["μ_Cd (eV)"] == 0]
        assert np.isclose(
            concentrations_Cd_rich["Fermi Level (eV wrt VBM)"].iloc[0],
            self.CdTe_anneal_1400K_quenched_150K_fermi_level,
        )
        assert np.isclose(
            concentrations_Cd_rich["Electrons (cm^-3)"].iloc[0],
            self.CdTe_anneal_1400K_quenched_150K_e,
            rtol=1e-3,
        )
        assert np.isclose(
            concentrations_Cd_rich["Holes (cm^-3)"].iloc[0],
            self.CdTe_anneal_1400K_quenched_150K_h,
            rtol=1e-3,
        )
        pd.testing.assert_series_equal(
            self.CdTe_anneal_1400K_quenched_150K_conc_df["Concentration (cm^-3)"],
            concentrations_Cd_rich["Concentration (cm^-3)"],
            rtol=3e-3,  # higher rtol required with large annealing, low quenching w/py-sc-fermi
        )  # also checks the index and ordering

    @parameterize_backend()
    def test_interpolate_chempots_with_chempot_dicts(self, backend):
        """
        Test interpolate_chempots method with manually specified chemical
        potentials.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        n_points = 30
        chempots_list = [
            {"Cd": -0.5, "Te": -1.0},
            {"Cd": -1.0, "Te": -0.5},
        ]

        concentrations = solver.interpolate_chempots(
            n_points=n_points,
            chempots=chempots_list,
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
        )
        assert len(concentrations) > 0
        unique_chempot_sets = concentrations[["μ_Cd (eV)", "μ_Te (eV)"]].drop_duplicates()
        assert len(unique_chempot_sets) == n_points

    @parameterize_backend()
    def test_interpolate_chempots_invalid_chempots_list_length(self, backend):
        """
        Test that ``ValueError`` is raised when chempots list does not contain
        exactly two dictionaries.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as exc:
            solver.interpolate_chempots(
                chempots=[{"Cd": -0.5}],  # Only one chempot dict provided
                annealing_temperature=800,
                quenched_temperature=300,
            )
        assert (
            "If `chempots` is a list, it must contain two dictionaries representing the starting and "
            "ending chemical potentials. The provided list has 1 entries!" in str(exc.value)
        )

    @parameterize_backend()
    def test_interpolate_chempots_missing_limits(self, backend):
        """
        Test that ``ValueError`` is raised when limits are missing and
        ``chempots`` is in ``doped`` format.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        no_err_out = solver.interpolate_chempots(
            annealing_temperature=800,
            quenched_temperature=300,
            limits=None,  # Limits are not provided
        )
        assert no_err_out is not None  # no error for binary system, just takes the two limits

        # but error when limits not set and number of limits != 2:
        three_lim_chempots = deepcopy(self.CdTe_thermo.chempots)
        extra_lim = {"Cd": -3, "Te": 4}
        three_lim_chempots["limits"]["extra"] = extra_lim
        three_lim_chempots["limits_wrt_el_refs"]["extra"] = extra_lim
        with pytest.raises(ValueError) as exc:
            solver.interpolate_chempots(
                chempots=three_lim_chempots,
                annealing_temperature=800,
                quenched_temperature=300,
                limits=None,  # Limits are not provided
            )

        assert (
            "If `chempots` is not provided as a list, then `limits` must be a list containing two "
            "strings representing the chemical potential limits to interpolate between. The provided "
            "`limits` is: None." in str(exc.value)
        )

    @parameterize_backend()
    def test_optimise_electrons(self, backend):
        """
        Test ``optimise`` method to min/max electron concentration.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        for min_max in ["min", "max"]:
            with patch("builtins.print") as mock_print:
                result = solver.optimise(
                    target="Electrons (cm^-3)",
                    min_or_max=min_max,
                    annealing_temperature=800,
                    quenched_temperature=300,
                    tolerance=0.05,
                    effective_dopant_concentration=1e16,
                )
            mock_print.assert_called_once_with(
                f"Searching for chemical potentials which {min_max}imise the target column: ['Electrons "
                f"(cm^-3)']..."
            )

            assert result.equals(
                solver.optimise(
                    target="e",  # target as column substring also fine
                    min_or_max=min_max,
                    annealing_temperature=800,
                    tolerance=0.05,
                    effective_dopant_concentration=1e16,
                )
            )

            # should correspond to Cd-rich for max electrons, Te-rich for min electrons
            limit = "Cd-rich" if min_max == "max" else "Te-rich"
            single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit=limit)
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }
            assert formal_chempots == single_chempot_dict
            _check_output_concentrations(solver, result)

            # quick test for anneal at 1400 K with no eff dopant:
            for cartesian in [True, False]:
                print(f"Testing with cartesian={cartesian}")
                result = solver.optimise(
                    target="Electrons (cm^-3)",
                    min_or_max=min_max,
                    annealing_temperature=1400,
                    quenched_temperature=150,
                    tolerance=0.05,
                )
                row = result.iloc[0]
                formal_chempots = {
                    mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
                }
                _check_output_concentrations(solver, result)

                # doesn't actually occur at the Te-rich limit for min electrons:
                assert (formal_chempots == single_chempot_dict) == (min_max != "min")

    @parameterize_backend()
    def test_optimise_electrons_non_limit_extremum(self, backend):
        """
        Test ``optimise`` method for a binary system (CdTe) where the extremum
        occurs at an internal chemical potential point, not at a limiting
        chemical potential point.
        """
        # Note: This function takes a bit of time, but the majority (>90%) is ``concentration_dict()``
        # from ``py-sc-fermi``; while ``doped`` backend is much faster
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        rtol = 0.1
        known_min_e = 1.50946e-23  # when using interpolate_chempots with 100 n_points
        result = solver.optimise(
            target="Electrons (cm^-3)",
            min_or_max="min",
            annealing_temperature=1400,
            quenched_temperature=150,
            tolerance=rtol,
            n_points=5,
            cartesian=True,  # should have no effect here, for binary system
        )  # requires 3 iterations for convergence within tolerance
        assert np.isclose(result["Electrons (cm^-3)"].iloc[0], known_min_e, atol=1e-40, rtol=rtol)

        result = solver.optimise(
            target="Electrons (cm^-3)",
            min_or_max="min",
            annealing_temperature=1400,
            quenched_temperature=150,
            tolerance=rtol / 100,
            n_points=5,
            cartesian=False,  # should have no effect here, for binary system
        )  # requires 4 iterations for convergence within tolerance
        # small non-convergence in py-sc-fermi for small concentrations:
        tight_rtol = rtol / 100 if backend == "doped" else rtol / 10
        assert np.isclose(result["Electrons (cm^-3)"].iloc[0], known_min_e, atol=1e-40, rtol=tight_rtol)

        _check_output_concentrations(solver, result)  # test dataframe output

    @parameterize_backend()
    def test_optimise_holes(self, backend):
        """
        Test ``optimise`` method to min/max hole concentration.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        for min_max in ["min", "max"]:
            with patch("builtins.print") as mock_print:
                result = solver.optimise(
                    target="Holes (cm^-3)",
                    min_or_max=min_max,
                    annealing_temperature=800,
                    quenched_temperature=300,
                    tolerance=0.05,
                    n_points=5,
                    effective_dopant_concentration=1e16,
                )
            mock_print.assert_called_once_with(
                f"Searching for chemical potentials which {min_max}imise the target column: ['Holes ("
                f"cm^-3)']..."
            )

            # should correspond to Te-rich for max holes, Cd-rich for min holes
            limit = "Te-rich" if min_max == "max" else "Cd-rich"
            single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit=limit)
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }
            assert formal_chempots == single_chempot_dict
            _check_output_concentrations(solver, result)

            # quick test for anneal at 1400 K with no eff dopant:
            result = solver.optimise(
                target="Holes (cm^-3)",
                min_or_max=min_max,
                annealing_temperature=1400,
                quenched_temperature=150,
                tolerance=0.05,
                n_points=5,
            )
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }
            _check_output_concentrations(solver, result)

            # doesn't actually occur at the Te-rich limit for max quenched holes:
            # e.g. see SK Thesis Fig. 6.18 (though without eff dopant conc, but similar behaviour)
            assert (formal_chempots == single_chempot_dict) == (min_max != "max")

    @parameterize_backend()
    def test_optimise_defect(self, backend):
        """
        Test ``optimise`` method to min/max a defect concentration.

        Here we use the somewhat-odd example of V_S in Sb2S3, where
        the concentration is (just about) minimised for an intermediate
        chemical potential (see Figure 1 in 10.1021/acsenergylett.4c02722,
        also the subject of "More Se Vacancies in Sb2Se3 under Se-Rich
        Conditions..." -- 10.1002/smll.202102429, to a more extreme extent).
        """
        solver = FermiSolver(self.Sb2S3_thermo, backend=backend)
        for annealing in [True, False]:
            for cartesian in [True, False]:
                print(f"Testing with annealing={annealing}, cartesian={cartesian}")
                temp_arg_name = "annealing_temperature" if annealing else "temperature"
                with patch("builtins.print") as mock_print:
                    result = solver.optimise(
                        target="V_S_3",  # highest concentration V_S
                        min_or_max="min",
                        cartesian=cartesian,
                        n_points=10 if backend != "doped" else 25,  # py-sc-fermi quite slow
                        **{temp_arg_name: 603},
                    )
                mock_print.assert_called_once_with(
                    "Searching for chemical potentials which minimise the target defect(s): ['V_S_3']..."
                )
                row = result.iloc[0]
                formal_chempots = {
                    mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
                }

                # confirm that formal_chempots does not correspond to a X-rich limit:
                for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values():
                    for el_key in limit:
                        assert not np.isclose(formal_chempots[el_key], limit[el_key], atol=5e-2)

                assert np.isclose(formal_chempots["Sb"], -0.430, atol=2e-2)
                assert np.isclose(formal_chempots["S"], -0.129, atol=2e-2)

                _check_output_concentrations(solver, result)

                # test that for a different defect (S_Sb), the extremum _is_ at a limiting chempot:
                result = solver.optimise(
                    "S_Sb_1",
                    min_or_max="max",
                    cartesian=cartesian,
                    n_points=10 if backend != "doped" else 25,  # py-sc-fermi quite slow
                    **{temp_arg_name: 603},
                )
                row = result.iloc[0]
                formal_chempots = {
                    mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
                }
                assert any(
                    all(np.isclose(formal_chempots[el_key], limit[el_key], atol=5e-2) for el_key in limit)
                    for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values()
                )
                _check_output_concentrations(solver, result)

    @parameterize_backend()
    def test_optimise_multiple_defects(self, backend):
        """
        Test ``optimise`` method to min/max a defect concentration, where now
        we use a defect name substring to match multiple defects.

        Here we use the vacanies in Sb2S3 as an example case; see
        10.1021/acsenergylett.4c02722 for reference.
        """
        solver = FermiSolver(self.Sb2S3_thermo, backend=backend)
        for annealing in [True, False]:
            temp_arg_name = "annealing_temperature" if annealing else "temperature"
            with patch("builtins.print") as mock_print:
                result = solver.optimise(
                    target="V_S",  # V_S and V_Sb (renamed defect folders, not default doped names)
                    min_or_max="min",
                    n_points=10 if backend != "doped" else 25,  # py-sc-fermi quite slow
                    **{temp_arg_name: 603},
                )
            mock_print.assert_called_once_with(
                "Searching for chemical potentials which minimise the target defect(s): ['V_S_1', "
                "'V_S_2', 'V_S_3', 'V_Sb_1', 'V_Sb_2']..."
            )
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }

            # here the minimising chempots are an intermediate chempot again, but slightly different
            # to before:
            for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values():
                for el_key in limit:  # confirm that formal_chempots don't correspond to X-rich limits
                    assert not np.isclose(formal_chempots[el_key], limit[el_key], atol=5e-2)

            assert np.isclose(formal_chempots["Sb"], -0.269, atol=1e-2)
            assert np.isclose(formal_chempots["S"], -0.237, atol=1e-2)

            _check_output_concentrations(solver, result)

            # test that for a different defect (S_Sb), the extremum _is_ at a limiting chempot:
            with patch("builtins.print") as mock_print:
                result = solver.optimise(
                    "S_Sb",  # less ambiguity this time
                    min_or_max="max",
                    n_points=10 if backend != "doped" else 25,  # py-sc-fermi quite slow
                    **{temp_arg_name: 603},
                )
            mock_print.assert_called_once_with(
                "Searching for chemical potentials which maximise the target defect(s): ['S_Sb_1', "
                "'S_Sb_2']..."
            )
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }
            single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="S-rich")
            assert all(  # for S_Sb total concentration, maximised at S-rich limit:
                np.isclose(formal_chempots[el_key], single_chempot_dict[el_key], atol=5e-2)
                for el_key in single_chempot_dict
            )
            _check_output_concentrations(solver, result)

    @parameterize_backend()
    def test_optimise_fermi_level(self, backend):
        """
        Test ``optimise`` method to min/max the Fermi level.

        Here we use CdTe as an example, where the most p-type Fermi level upon
        quenching is actually not at the Te-rich limit (but close); see SK
        Thesis Fig. 6.17. For most n-type Fermi level, it is at the Cd-rich
        limit as is typical.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        for min_max in ["min", "max"]:
            print(f"Testing {min_max}imising Fermi level...")
            with patch("builtins.print") as mock_print:
                result = solver.optimise(
                    target="Fermi Level (eV wrt VBM)",
                    min_or_max=min_max,
                    annealing_temperature=973,  # SK Thesis Fig. 6.17
                )
            mock_print.assert_called_once_with(
                f"Searching for chemical potentials which {min_max}imise the target column: ['Fermi "
                f"Level (eV wrt VBM)']..."
            )
            assert result.equals(
                solver.optimise(
                    target="fermi",  # target as column substring also fine
                    min_or_max=min_max,
                    annealing_temperature=973,  # SK Thesis Fig. 6.17
                )
            )

            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }

            if min_max == "min":  # formal_chempots does not correspond to a X-rich limit, for min Fermi:
                for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values():
                    for el_key in limit:
                        assert not np.isclose(formal_chempots[el_key], limit[el_key], atol=5e-2)
            else:  # for max (i.e. most n-type Fermi level), it is at Cd-rich
                assert any(
                    all(np.isclose(formal_chempots[el_key], limit[el_key], atol=5e-2) for el_key in limit)
                    for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values()
                )

            assert np.isclose(formal_chempots["Cd"], -1.05 if min_max == "min" else 0, atol=1e-2)

            _check_output_concentrations(solver, result)  # SK Thesis Fig. 6.17

    @parameterize_backend()
    def test_optimise_chempot(self, backend):
        """
        Test ``optimise`` method to min/max a chemical potential.

        Usually not a desired usage, but could be a handy convenience method in
        some cases.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        for min_max in ["min", "max"]:
            print(f"Testing {min_max}imising chemical potential...")
            with patch("builtins.print") as mock_print:
                result = solver.optimise(target="μ_Te (eV)", min_or_max=min_max, annealing_temperature=973)
            mock_print.assert_called_once_with(
                f"Searching for chemical potentials which {min_max}imise the target column: ['μ_Te ("
                f"eV)']..."
            )
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }

            # corresponds to Te-rich/poor limits:
            limit = "Te-rich" if min_max == "max" else "Cd-rich"
            single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit=limit)
            assert all(
                np.isclose(formal_chempots[el_key], single_chempot_dict[el_key], atol=5e-2)
                for el_key in single_chempot_dict
            )

            _check_output_concentrations(solver, result)

    @parameterize_backend()
    def test_optimise_invalid_target(self, backend):
        """
        Test ``optimise`` method error with an invalid input target.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with pytest.raises(ValueError) as exc:
            solver.optimise(target="WTF?")
        assert (
            "Target 'WTF?' not found in results DataFrame! Must be a column or defect name/substring! "
            "See docstring for more info."
        ) in str(exc.value)

    @parameterize_backend()
    def test_optimise_multiple_column_matches(self, backend):
        """
        Test ``optimise`` method error with an input target which matches
        multiple columns.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with warnings.catch_warnings(record=True) as w:
            solver.optimise(target="cm^-3")
        print([str(warning.message) for warning in w])  # for debugging
        assert (
            "Multiple columns with the name 'cm^-3' found in the results DataFrame! Choosing the first "
            "match"
        ) in str(w[-1].message)

    def test_get_interpolated_chempots(self):
        """
        Test ``get_interpolated_chempots`` function.
        """
        chempot_start = {"Cd": -0.5, "Te": -1.0}
        chempot_end = {"Cd": -1.0, "Te": -0.5}
        n_points = 3
        interpolated_chempots = get_interpolated_chempots(chempot_start, chempot_end, n_points)

        assert len(interpolated_chempots) == n_points
        assert interpolated_chempots[0] == chempot_start
        assert interpolated_chempots[-1] == chempot_end
        expected_middle = {"Cd": -0.75, "Te": -0.75}  # Check middle point
        assert interpolated_chempots[1] == expected_middle

    @parameterize_backend()
    def test_parse_and_check_grid_like_chempots(self, backend):
        """
        Test ``_parse_and_check_grid_like_chempots`` method.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        chempots = self.CdTe_thermo.chempots
        parsed_chempots, el_refs = solver._parse_and_check_grid_like_chempots(chempots)

        assert isinstance(parsed_chempots, dict)
        assert isinstance(el_refs, dict)
        assert "limits" in parsed_chempots
        assert "elemental_refs" in parsed_chempots

        # test error with single-chempot-limit system:
        Se_pnict_thermo = loadfn(os.path.join(data_dir, "Se_Pnict_Thermo.json.gz"))
        solver = FermiSolver(
            Se_pnict_thermo, bulk_dos=self.CdTe_fermi_dos, skip_dos_check=True, backend=backend
        )
        with pytest.raises(ValueError) as exc:
            solver._parse_and_check_grid_like_chempots()
        assert "Only one chemical potential limit is present in " in str(exc.value)

    @parameterize_backend()
    def test_parse_and_check_grid_like_chempots_invalid_chempots(self, backend):
        """
        Test that ``ValueError`` is raised when ``chempots`` is ``None``.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        # Temporarily remove chempots from defect_thermodynamics:
        solver = deepcopy(self.solver_doped)
        solver.defect_thermodynamics.chempots = None

        with pytest.raises(ValueError):
            solver._parse_and_check_grid_like_chempots()

    @parameterize_backend()
    def test_skip_dos_check(self, backend):
        """
        Test the ``FermiDos`` vs ``DefectThermodynamics`` VBM check, and how it
        is skipped with ``skip_dos_check``.

        Main test code in ``test_thermodynamics.py``.
        """
        fd_up_fdos = deepcopy(self.CdTe_thermo.bulk_dos)
        fd_up_fdos.energies -= 0.1
        defect_thermo = deepcopy(self.CdTe_thermo)

        from test_thermodynamics import _check_CdTe_mismatch_fermi_dos_warning

        with warnings.catch_warnings(record=True) as w:
            FermiSolver(defect_thermodynamics=defect_thermo, bulk_dos=fd_up_fdos, backend=backend)
        _check_CdTe_mismatch_fermi_dos_warning(None, w)

        with warnings.catch_warnings(record=True) as w:
            FermiSolver(
                defect_thermodynamics=defect_thermo,
                bulk_dos=fd_up_fdos,
                skip_dos_check=True,
                backend=backend,
            )
        print([str(warning.message) for warning in w])
        assert not w

    @parameterize_backend()
    def test_delta_gap_calculated_fermi_level_k10(self, backend):
        """
        Test calculating the Fermi level using a 10x10x10 k-point mesh DOS
        calculation for primitive CdTe; with ``delta_gap``.

        Mirrors the direct functions test (``test_calculated_fermi_level_k10``)
        in ``test_thermodynamics``, but now with ``FermiSolver`` functions.
        """
        solver = deepcopy(self.solver_doped if backend == "doped" else self.solver_py_sc_fermi)
        solver.defect_thermodynamics.bulk_dos = os.path.join(
            module_path, "data/CdTe/CdTe_prim_k101010_dos_vr.xml.gz"
        )

        quenched_fermi_levels = []
        for anneal_temp in reduced_anneal_temperatures:
            gap_shift = belas_linear_fit(anneal_temp) - 1.5
            result = solver.pseudo_equilibrium_solve(
                # quenching to 300K (default)
                single_chempot_dict=self.CdTe_thermo.chempots["limits_wrt_el_refs"]["CdTe-Te"],
                annealing_temperature=anneal_temp,
                delta_gap=gap_shift,
            )
            quenched_fermi_levels += [result.iloc[0]["Fermi Level (eV wrt VBM)"]]

        # (approx) same result as with k181818 NKRED=2 (0.31825 eV with this DOS)
        # remember this is LZ thermo, not FNV thermo shown in thermodynamics tutorial
        assert np.isclose(
            np.mean(quenched_fermi_levels[6:8]), 0.31825, atol=1e-3 if backend == "doped" else 1e-2
        )

    @custom_mpl_image_compare(filename="CdTe_LZ_Te_rich_concentrations_vs_μ_Te.png")
    def test_delta_gap_interpolate_chempots_CdTe(self):
        """
        Mirrors ``test_CdTe_concentrations_vs_chempots`` in
        ``test_thermodynamics``, but now using ``FermiSolver`` (with
        ``delta_gap``).

        Tests both backends, but only uses the ``doped`` results for plotting
        for full consistency with ``test_thermodynamics`` test (due to site
        competition differences etc).
        """
        kwargs = {
            "limits": ["Cd-rich", "Te-rich"],
            "annealing_temperature": 875,  # typical for CdTe
            "delta_gap": belas_linear_fit(875) - 1.5,
        }

        doped_output_df = self.solver_doped.interpolate_chempots(**kwargs)
        py_sc_fermi_output_df = self.solver_py_sc_fermi.interpolate_chempots(**kwargs)
        pd.testing.assert_frame_equal(doped_output_df, py_sc_fermi_output_df, rtol=2e-2, atol=1e16)
        pd.testing.assert_series_equal(
            doped_output_df["Fermi Level (eV wrt VBM)"],
            py_sc_fermi_output_df["Fermi Level (eV wrt VBM)"],
            atol=2e2,
        )

        plt.style.use(STYLE)
        f, ax = plt.subplots()
        for defect_index in doped_output_df.index.unique():
            matching_rows = doped_output_df[doped_output_df.index == defect_index]
            ax.plot(
                matching_rows["μ_Te (eV)"],
                matching_rows["Concentration (cm^-3)"],
                label=format_defect_name(defect_index, wout_charge=True, include_site_info_in_name=True),
            )

        ax.plot(
            doped_output_df["μ_Te (eV)"], doped_output_df["Electrons (cm^-3)"], label="Electrons", ls="--"
        )
        ax.plot(doped_output_df["μ_Te (eV)"], doped_output_df["Holes (cm^-3)"], label="Holes", ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e7, 1e19)
        ax.set_ylabel("Concentration (cm$^{-3}$)")
        ax.set_xlabel("Te Chemical Potential (eV)")
        ax.legend()

        return f

    @custom_mpl_image_compare(filename="CdTe_LZ_Te_rich_Fermi_levels.png")
    def test_delta_gap_scan_temperature(self):
        """
        Mirrors ``test_calculated_fermi_levels`` in ``test_thermodynamics``,
        but now using ``FermiSolver`` (with ``delta_gap`` as a function!).

        Tests both backends, but only uses the ``doped`` results for plotting
        for full consistency with ``test_thermodynamics`` test (due to site
        competition differences etc).
        """
        kwargs = {
            "limit": "Te-rich",
            "annealing_temperature_range": anneal_temperatures,
            "delta_gap": lambda T: belas_linear_fit(T) - 1.5,
        }

        doped_output_df = self.solver_doped.scan_temperature(**kwargs)
        py_sc_fermi_output_df = self.solver_py_sc_fermi.scan_temperature(**kwargs)
        pd.testing.assert_frame_equal(doped_output_df, py_sc_fermi_output_df, rtol=0.2, atol=2e17)
        pd.testing.assert_series_equal(
            doped_output_df["Fermi Level (eV wrt VBM)"],
            py_sc_fermi_output_df["Fermi Level (eV wrt VBM)"],
            atol=2e2,
        )
        kwargs["temperature_range"] = kwargs.pop("annealing_temperature_range")

        # remember this is LZ thermo, not FNV thermo shown in thermodynamics tutorial
        assert np.isclose(
            np.mean(
                doped_output_df["Fermi Level (eV wrt VBM)"][
                    (doped_output_df["Annealing Temperature (K)"] >= 800)
                    & (doped_output_df["Annealing Temperature (K)"] <= 950)
                ]
            ),
            0.318674,
            atol=1e-3,
        )

        plt.style.use(STYLE)
        f, ax = plt.subplots()
        # ax.plot(  # cut because requires using scissored DOS to get correct annealing Fermi level
        #     doped_output_df["Annealing Temperature (K)"].unique(),
        #     doped_annealing_output_df["Fermi Level (eV wrt VBM)"].unique(),
        #     marker="o",
        #     label="$E_F$ during annealing (@ $T_{anneal}$)",
        #     color="k",
        #     alpha=0.25,
        # )
        ax.plot(
            doped_output_df["Annealing Temperature (K)"].unique(),
            doped_output_df["Fermi Level (eV wrt VBM)"].unique(),
            marker="o",
            label="$E_F$ upon cooling (@ $T$ = 300K)",
            color="k",
            alpha=0.9,
        )
        ax.set_xlabel("Anneal Temperature (K)")
        ax.set_ylabel("Fermi Level wrt VBM (eV)")
        ax.set_xlim(300, 1400)
        ax.axvspan(500 + 273.15, 700 + 273.15, alpha=0.2, color="#33A7CC", label="Typical Anneal Range")
        ax.fill_between(
            doped_output_df["Annealing Temperature (K)"],
            (1.5 - belas_linear_fit(doped_output_df["Annealing Temperature (K)"])) / 2,
            0,
            alpha=0.2,
            color="C0",
            label="VBM (T @ $T_{anneal}$)",
            linewidth=0.25,
        )
        ax.fill_between(
            doped_output_df["Annealing Temperature (K)"],
            1.5 - (1.5 - belas_linear_fit(doped_output_df["Annealing Temperature (K)"])) / 2,
            1.5,
            alpha=0.2,
            color="C1",
            label="CBM (T @ $T_{anneal}$)",
            linewidth=0.25,
        )

        ax.legend(fontsize=8)

        ax.imshow(  # show VB in blue from -0.3 to 0 eV:
            [(1, 1), (0, 0)],
            cmap=plt.cm.Blues,
            extent=(ax.get_xlim()[0], ax.get_xlim()[1], -0.3, 0),
            vmin=0,
            vmax=3,
            interpolation="bicubic",
            rasterized=True,
            aspect="auto",
        )

        ax.imshow(
            [
                (
                    0,
                    0,
                ),
                (1, 1),
            ],
            cmap=plt.cm.Oranges,
            extent=(ax.get_xlim()[0], ax.get_xlim()[1], 1.5, 1.8),
            vmin=0,
            vmax=3,
            interpolation="bicubic",
            rasterized=True,
            aspect="auto",
        )
        ax.set_ylim(-0.2, 1.7)

        return f


def _check_output_concentrations(solver, result):
    row = result.iloc[0]  # same parameters for each concentration row from ``optimise()`` method
    kwargs = {
        "single_chempot_dict": {
            mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
        }
    }
    if "Annealing Temperature (K)" in row.index:
        kwargs["annealing_temperature"] = row["Annealing Temperature (K)"]
        kwargs["quenched_temperature"] = row["Quenched Temperature (K)"]
    else:
        kwargs["temperature"] = row["Temperature (K)"]
    if "Dopant (cm^-3)" in row.index:
        kwargs["effective_dopant_concentration"] = row["Dopant (cm^-3)"]

    expected_concentrations = solver._solve(
        **kwargs,
        append_chempots=True,
    )
    print(f"Comparing:\n{result}\nto\n{expected_concentrations}\nwith kwargs: {kwargs}")
    pd.testing.assert_frame_equal(result, expected_concentrations, check_dtype=False)


def _plot_Cu_i_data(Cu_i_data):
    Cu_i_concs = Cu_i_data["Concentration (cm^-3)"]
    plt.style.use(STYLE)
    fig, ax = plt.subplots()
    from matplotlib.colors import LogNorm

    sc = ax.scatter(
        Cu_i_data["μ_Se (eV)"],
        Cu_i_data["μ_Cu (eV)"],
        c=Cu_i_concs,
        cmap="viridis",
        norm=LogNorm(min(Cu_i_concs), min(Cu_i_concs) * 1.1),  # small range for better contrast
    )
    fig.colorbar(sc, ax=ax, label="Cu$_i$ Concentration (cm$^{-3}$)")
    ax.set_xlabel("$\mu_{Se}$ (eV)")
    ax.set_ylabel("$\mu_{Cu}$ (eV)")
    return fig


# TODO: Test free_defects with substring matching (and fixed_defects later when supported)
# TODO: Use plots in FermiSolver tutorial as quick test cases here
class TestFermiSolverWithLoadedData3D(unittest.TestCase):
    """
    Tests for ``FermiSolver`` with loaded data, for a ternary system (Cu2SiSe3
    in this case).
    """

    @classmethod
    def setUpClass(cls):
        cls.Cu2SiSe3_thermo = loadfn(os.path.join(EXAMPLE_DIR, "Cu2SiSe3/Cu2SiSe3_thermo.json"))
        cls.Cu2SiSe3_fermi_dos = get_fermi_dos(os.path.join(EXAMPLE_DIR, "Cu2SiSe3/vasprun.xml.gz"))
        cls.Cu2SiSe3_thermo.chempots = loadfn(os.path.join(EXAMPLE_DIR, "Cu2SiSe3/Cu2SiSe3_chempots.json"))

        cls.fake_no_v_Cu_Cu2SiSe3_thermo = DefectThermodynamics(
            [
                entry
                for name, entry in cls.Cu2SiSe3_thermo.defect_entries.items()
                if not name.startswith("v_Cu")
            ],
            chempots=cls.Cu2SiSe3_thermo.chempots,
        )

        cls.Y_doped_Cd2Sb2O7_thermo = loadfn(
            os.path.join(data_dir, "Y_doped_Cd2Sb2O7_thermo_w_DOS.json.gz")
        )

    def setUp(self):
        self.Cu2SiSe3_thermo.bulk_dos = self.Cu2SiSe3_fermi_dos
        self.solver_py_sc_fermi = FermiSolver(
            defect_thermodynamics=self.Cu2SiSe3_thermo, backend="py-sc-fermi"
        )
        self.solver_doped = FermiSolver(defect_thermodynamics=self.Cu2SiSe3_thermo, backend="doped")
        # Mock the _DOS attribute for py-sc-fermi backend if needed
        self.solver_py_sc_fermi._DOS = MagicMock()

    @parameterize_backend()
    def test_optimise_maximize_electrons(self, backend):
        """
        Test ``optimise`` method to maximize electron concentration.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with patch("builtins.print") as mock_print:
            result = solver.optimise(
                target="Electrons (cm^-3)",
                min_or_max="max",
                annealing_temperature=800,
                quenched_temperature=300,
                tolerance=0.05,
                effective_dopant_concentration=1e16,
            )
        mock_print.assert_called_once_with(
            "Searching for chemical potentials which maximise the target column: ['Electrons (cm^-3)']..."
        )
        row = result.iloc[0]
        formal_chempots = {
            mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
        }

        # corresponds to Cu-rich limit: (minimising v_Cu)
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Cu-rich")
        assert all(
            np.isclose(formal_chempots[el_key], single_chempot_dict[el_key], atol=5e-2)
            for el_key in single_chempot_dict
        )

        _check_output_concentrations(solver, result)

    @parameterize_backend()
    def test_optimise_minimize_holes(self, backend):
        """
        Test ``optimise`` method to minimize hole concentration.
        """
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        with patch("builtins.print") as mock_print:
            result = solver.optimise(
                target="Holes (cm^-3)",
                min_or_max="min",
                annealing_temperature=800,
                effective_dopant_concentration=1e16,
            )
        mock_print.assert_called_once_with(
            "Searching for chemical potentials which minimise the target column: ['Holes (cm^-3)']..."
        )
        row = result.iloc[0]
        formal_chempots = {
            mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
        }

        # corresponds to Cu-rich limit: (minimising v_Cu)
        single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit="Cu-rich")
        assert all(
            np.isclose(formal_chempots[el_key], single_chempot_dict[el_key], atol=5e-2)
            for el_key in single_chempot_dict
        )

        _check_output_concentrations(solver, result)

    def test_optimise_maximize_electrons_Y_doped_Cd2Sb2O7_fixed_elements(self):
        """
        Test ``optimise`` method to maximize electron concentration.

        Previously reported by Peter Russell to fail, which was due to rounding
        errors. Fixed now, with ``fixed_elements`` input supported.
        """
        solver = FermiSolver(defect_thermodynamics=self.Y_doped_Cd2Sb2O7_thermo, backend="doped")
        result = solver.optimise(
            target="Electrons (cm^-3)",
            min_or_max="max",
            fixed_elements={"O": -1.32},
        )
        row = result.iloc[0]
        assert np.isclose(row["Electrons (cm^-3)"], 2.0775e18, rtol=0.01)
        formal_chempots = {
            mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
        }

        # confirm that formal_chempots does not correspond to a X-rich limit:
        for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values():
            for el_key in limit:
                assert not np.isclose(formal_chempots[el_key], limit[el_key], atol=2e-2)

        assert np.isclose(formal_chempots["O"], -1.32, atol=1e-2)
        assert np.isclose(formal_chempots["Sb"], -2.0659, atol=1e-3)
        assert np.isclose(formal_chempots["Cd"], -1.2767, atol=1e-3)
        assert np.isclose(formal_chempots["Y"], -8.0855, atol=1e-3)

        _check_output_concentrations(solver, result)

    @parameterize_backend()
    def test_optimise_defect_3D_non_limiting_chempot(self, backend):
        """
        Test ``optimise`` method for a ternary system (Cu2SiSe3) where the
        extremum occurs at an internal chemical potential point, not at a
        limiting chemical potential point.

        For Cu2SiSe3, the defect/carrier concentrations are dominated
        by ``v_Cu``, which means that extrema for all defects/carriers/Fermi
        levels basically always occur at limiting chemical potentials.
        If we create a 'fake' defect thermo with ``v_Cu`` removed however,
        we then get a minimum ``Cu_i`` concentration for chemical potentials
        along the tangent line between two of the limiting chemical
        potentials, which is tested here.

        See ``test_plot_scan_chemical_potential_grid`` for visualisation
        confirming minimum not at limiting chemical potentials.
        """
        solver = FermiSolver(
            self.fake_no_v_Cu_Cu2SiSe3_thermo, bulk_dos=self.Cu2SiSe3_fermi_dos, backend=backend
        )

        for annealing in [True, False]:
            temp_arg_name = "annealing_temperature" if annealing else "temperature"
            with patch("builtins.print") as mock_print:
                result = solver.optimise(
                    target="Int_Cu",  # older doped names
                    min_or_max="min",
                    **{temp_arg_name: 1000},
                )
            mock_print.assert_called_once_with(
                "Searching for chemical potentials which minimise the target defect(s): ['Int_Cu']..."
            )
            row = result.iloc[0]
            formal_chempots = {
                mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
            }

            # confirm that formal_chempots does not correspond to a X-rich limit:
            for limit in solver.defect_thermodynamics.chempots["limits_wrt_el_refs"].values():
                for el_key in limit:
                    assert not np.isclose(formal_chempots[el_key], limit[el_key], atol=2e-2)

            assert np.isclose(formal_chempots["Cu"], -0.179, atol=1e-3)
            assert np.isclose(formal_chempots["Se"], -0.569, atol=1e-3)
            assert np.isclose(formal_chempots["Si"], -0.457, atol=1e-3)

            _check_output_concentrations(solver, result)
            if backend != "doped":
                return  # skip second half of test for py-sc-fermi backend, it's quite slow

            # test that when v_Cu is included, the extremum _is_ at a limiting chempot:
            w_v_Cu_solver = self.solver_doped
            for min_max in ["min", "max"]:
                print(f"Testing {min_max}imising chemical potential...")
                with patch("builtins.print") as mock_print:
                    result = w_v_Cu_solver.optimise(
                        "Int_Cu",
                        min_or_max=min_max,
                        **{temp_arg_name: 1000},
                    )
                mock_print.assert_called_once_with(
                    f"Searching for chemical potentials which {min_max}imise the target defect(s): ["
                    f"'Int_Cu']..."
                )
                row = result.iloc[0]
                formal_chempots = {
                    mu_col.strip("μ_").split()[0]: row[mu_col] for mu_col in row.index if "μ_" in mu_col
                }

                # corresponds to Cu-rich/poor limits:
                limit = "Cu-rich" if min_max == "max" else "Cu-poor"
                single_chempot_dict, el_refs = solver._get_single_chempot_dict(limit=limit)
                assert all(
                    np.isclose(formal_chempots[el_key], single_chempot_dict[el_key], atol=2e-2)
                    for el_key in single_chempot_dict
                )
                _check_output_concentrations(w_v_Cu_solver, result)

    @custom_mpl_image_compare(filename="fake_no_v_Cu_Cu2SiSe3_Cu_i_chempot_grid.png")
    def test_plot_scan_chemical_potential_grid(self, backend="doped"):
        """
        Test ``scan_chemical_potential_grid`` method, by plotting the output
        with the ``fake_no_v_Cu_Cu2SiSe3_thermo`` system.

        Note this code takes >7 mins with with the ``py-sc-fermi``
        backend, but only 45 seconds with ``doped`` backend.
        """
        solver = FermiSolver(
            self.fake_no_v_Cu_Cu2SiSe3_thermo, bulk_dos=self.Cu2SiSe3_fermi_dos, backend=backend
        )
        data = solver.scan_chemical_potential_grid(
            n_points=500,
            annealing_temperature=1000,  # the temperature at which to anneal the system
        )
        return _plot_Cu_i_data(data.loc["Int_Cu"])

    @custom_mpl_image_compare(filename="fake_no_v_Cu_Cu2SiSe3_Cu_i_chempot_grid_cartesian.png")
    def test_plot_scan_chemical_potential_grid_cartesian(self, backend="doped"):
        solver = FermiSolver(
            self.fake_no_v_Cu_Cu2SiSe3_thermo, bulk_dos=self.Cu2SiSe3_fermi_dos, backend=backend
        )
        data = solver.scan_chemical_potential_grid(
            n_points=500,
            annealing_temperature=1000,  # the temperature at which to anneal the system
            cartesian=True,
        )
        return _plot_Cu_i_data(data.loc["Int_Cu"])

    def test_scan_chemical_potential_grid_Y_doped_Cd2Sb2O7_fixed_elements(self):
        solver = FermiSolver(defect_thermodynamics=self.Y_doped_Cd2Sb2O7_thermo, backend="doped")
        concentrations = solver.scan_chemical_potential_grid(
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
            fixed_elements={"O": -1.32},
        )
        assert len(concentrations) > 0
        unique_chempot_sets = concentrations[
            [f"μ_{el} (eV)" for el in self.Y_doped_Cd2Sb2O7_thermo.chempots["elemental_refs"]]
        ].drop_duplicates()
        assert len(unique_chempot_sets) > 0

    @parameterize_backend()
    def test_scan_chemical_potential_grid(self, backend):
        """
        Test ``scan_chemical_potential_grid`` method.
        """
        # Note: Plotting tests are likely the easiest for testing the actual output of this function,
        # as done in ``test_plot_scan_chemical_potential_grid``. This function is also implicitly tested
        # via the ``optimise()`` tests
        solver = self.solver_doped if backend == "doped" else self.solver_py_sc_fermi
        n_points = 50
        concentrations = solver.scan_chemical_potential_grid(
            # use self.defect_thermodynamics.chempots by default
            n_points=n_points,
            annealing_temperature=800,
            quenched_temperature=300,
            effective_dopant_concentration=1e16,
        )
        assert len(concentrations) > 0
        unique_chempot_sets = concentrations[
            [f"μ_{el} (eV)" for el in self.Cu2SiSe3_thermo.chempots["elemental_refs"]]
        ].drop_duplicates()
        assert len(unique_chempot_sets) > 0

    def test_scan_chemical_potential_grid_wrong_chempots(self):
        """
        Test that ``ValueError`` is raised when no chempots are provided and
        ``None`` are available in ``self.Cu2SiSe3_thermo``, or only a single
        limit is provided.
        """
        # Temporarily remove chempots from defect_thermodynamics
        solver = deepcopy(self.solver_doped)
        solver.defect_thermodynamics.chempots = None

        for chempot_kwargs in [
            {},
            {"chempots": {"Cu": -0.5, "Si": -1.0, "Se": 2}},
        ]:
            print(f"Testing with {chempot_kwargs}")
            with pytest.raises(ValueError) as exc:
                solver.scan_chemical_potential_grid(
                    annealing_temperature=800,
                    quenched_temperature=300,
                    **chempot_kwargs,
                )
            print(str(exc.value))
            assert (
                "Only one chemical potential limit is present in "
                "`chempots`/`self.defect_thermodynamics.chempots`, which makes no sense for a chemical "
                "potential grid scan"
            ) in str(exc.value)


if __name__ == "__main__":
    unittest.main()
