"""
Tests for doped.interface.fermi_solver module.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.electronic_structure.dos import FermiDos

from doped.interface.fermi_solver import FermiSolver, FermiSolverDoped, FermiSolverPyScFermi


class FermiSolverTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = "../examples/CdTe/CdTe_LZ_thermo_wout_meta.json"
        self.bulk_dos_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.thermo = loadfn(self.thermo_path)
        self.fs = FermiSolver(self.thermo, self.bulk_dos_path)

    def test_init(self):
        assert self.fs.defect_thermodynamics == self.thermo
        assert self.fs.bulk_dos == self.bulk_dos_path

    def test_equilibrium_solve(self):
        # test that equilibrium solve raises not implemented error
        with pytest.raises(NotImplementedError) as exc:
            self.fs.equilibrium_solve()

        print(exc.value)  # for debugging
        assert (
            "This method is implemented in the derived class, use FermiSolverDoped or "
            "FermiSolverPyScFermi instead."
        ) in str(exc.value)

    def test_pseudoequilibrium_solve(self):
        # test that pseudoequilibrium solve raises not implemented error
        with pytest.raises(NotImplementedError) as exc:
            self.fs.pseudo_equilibrium_solve()

        print(exc.value)  # for debugging
        assert (
            "This method is implemented in the derived class, use FermiSolverDoped or "
            "FermiSolverPyScFermi instead."
        ) in str(exc.value)

    def test_assert_scan_temperature_raises(self):

        with pytest.raises(ValueError) as exc:
            self.fs.scan_temperature(
                {},
                temperature_range=[1, 2, 3],
                annealing_temperature_range=[1, 2, 3],
                quenching_temperature_range=None,
            )
        print(exc.value)  # for debugging

        with pytest.raises(ValueError) as exc:
            self.fs.scan_temperature(
                {},
                temperature_range=[1, 2, 3],
                annealing_temperature_range=None,
                quenching_temperature_range=[1, 2, 3],
            )
        print(exc.value)  # for debugging

        with pytest.raises(ValueError) as exc:
            self.fs.scan_temperature(
                {},
                temperature_range=[],
                annealing_temperature_range=None,
                quenching_temperature_range=None,
            )
        print(exc.value)  # for debugging

    def test_get_interpolated_chempots(self):

        int_chempots = self.fs._get_interpolated_chempots({"a": -1, "b": -1}, {"a": 1, "b": 1}, 11)
        assert len(int_chempots) == 11
        assert int_chempots[0]["a"] == -1
        assert int_chempots[0]["b"] == -1
        assert int_chempots[-1]["a"] == 1
        assert int_chempots[-1]["b"] == 1
        assert int_chempots[5]["a"] == 0
        assert int_chempots[5]["b"] == 0


class FermiSolverDopedTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = "../examples/CdTe/CdTe_LZ_thermo_wout_meta.json"
        self.bulk_dos_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.thermo = loadfn(self.thermo_path)
        self.fs = FermiSolverDoped(self.thermo, self.bulk_dos_path)

    def test__init__(self):
        assert self.fs.defect_thermodynamics == self.thermo
        assert isinstance(self.fs.bulk_dos, FermiDos)
        assert self.fs.bulk_dos.nelecs == 18.0

    def test_assert_scan_dopant_concentration_raises(self):
        with pytest.raises(TypeError) as exc:  # missing arguments
            self.fs.scan_dopant_concentration()

        print(exc.value)  # for debugging

        # TODO: This doesn't raise a NotImplementedError because it's a method for the parent
        #  `FermiSolver` class. Is this not also possible with `FermiSolverDoped`? As it's just fixing
        #  the dopant concentrations? Thought we could do it with both backends without issue
        with pytest.raises(NotImplementedError) as exc:
            self.fs.scan_dopant_concentration(
                chempots=None, effective_dopant_concentration_range=[1, 2, 3]
            )

        print(exc.value)  # for debugging

    def test_get_fermi_level_and_carriers(self):
        # set return value for the thermodynamics method `get_equilibrium_fermi_level`,
        # does it need to be a mock?
        with patch(
            "doped.thermodynamics.DefectThermodynamics.get_equilibrium_fermi_level",
            return_value=(0.5, 0.5, 0.5),
        ):
            fermi_level, electrons, holes = self.fs._get_fermi_level_and_carriers({}, 300)
            assert fermi_level == 0.5
            assert electrons == 0.5
            assert holes == 0.5

    def test_equilibrium_solve(self):

        equilibrium_results = self.fs.equilibrium_solve({"Cd": -2, "Te": -2}, 300)
        assert np.isclose(equilibrium_results["Fermi Level"][0], 0.798379, rtol=1e-4)
        assert np.isclose(equilibrium_results["Electrons (cm^-3)"][0], 57861.403451, rtol=1e-4)
        assert np.isclose(equilibrium_results["Holes (cm^-3)"][0], 390268.197798, rtol=1e-4)
        assert np.isclose(equilibrium_results["Temperature"][0], 300, rtol=1e-4)
        assert np.isclose(equilibrium_results["Concentration (cm^-3)"][0], 1.423000e-24, rtol=1e-4)

    def test_pseudo_equilibrium_solve(self):
        pseudo_equilibrium_results = self.fs.pseudo_equilibrium_solve(
            {"Cd": -2, "Te": -2}, annealing_temperature=900, quenched_temperature=300
        )

        assert np.isclose(pseudo_equilibrium_results["Fermi Level"][0], 0.456269, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Electrons (cm^-3)"][0], 0.10356, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Holes (cm^-3)"][0], 218051006212.72186, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Annealing Temperature"][0], 900.0, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Quenched Temperature"][0], 300.0, rtol=1e-4)
        assert np.isclose(
            pseudo_equilibrium_results["Concentration (cm^-3)"][0], 5.426998900437918e20, rtol=1e-4
        )


class FermiSolverPyScFermiTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = "../examples/CdTe/CdTe_LZ_thermo_wout_meta.json"
        self.bulk_dos_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.thermo = loadfn(self.thermo_path)
        self.fs = FermiSolverPyScFermi(self.thermo, self.bulk_dos_path, multiplicity_scaling=32)

    def test__init__(self):
        assert self.fs.defect_thermodynamics == self.thermo

    def test_equilibrium_solve(self):

        equilibrium_results = self.fs.equilibrium_solve({"Cd": -2, "Te": -2}, 300)
        assert np.isclose(equilibrium_results["Fermi Level"][0], 0.8878258982206311, rtol=1e-4)
        assert np.isclose(equilibrium_results["Electrons (cm^-3)"][0], 4490462.40649167, rtol=1e-4)
        assert np.isclose(equilibrium_results["Holes (cm^-3)"][0], 12219.64626151177, rtol=1e-4)
        assert np.isclose(equilibrium_results["Temperature"][0], 300, rtol=1e-4)

    def test_pseudo_equilibrium_solve(self):
        pseudo_equilibrium_results = self.fs.pseudo_equilibrium_solve(
            {"Cd": -2, "Te": -2}, annealing_temperature=900, quenched_temperature=300
        )

        assert np.isclose(pseudo_equilibrium_results["Fermi Level"][0], 0.8900579775197812, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Electrons (cm^-3)"][0], 4895401.840081683, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Holes (cm^-3)"][0], 11208.85760769246, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Annealing Temperature"][0], 900.0, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Quenched Temperature"][0], 300.0, rtol=1e-4)
