"""
Tests for doped.interface.fermi_solver module.
"""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from monty.serialization import loadfn
from pymatgen.electronic_structure.dos import FermiDos

from doped.interface.fermi_solver import (
    ChemicalPotentialGrid,
    FermiSolver,
    FermiSolverDoped,
    FermiSolverPyScFermi,
)

py_sc_fermi_installed = bool(importlib.util.find_spec("py_sc_fermi"))


class FermiSolverTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = "../examples/CdTe/CdTe_LZ_thermo_wout_meta.json.gz"
        self.bulk_dos_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.thermo = loadfn(self.thermo_path)
        self.fs = FermiSolver(self.thermo, self.bulk_dos_path)

    def test_init(self):
        assert self.fs.defect_thermodynamics == self.thermo
        assert self.fs.bulk_dos == self.bulk_dos_path

    def test__get_limits(self):
        assert self.fs._get_limits("Cd-rich") == {"Cd": 0.0, "Te": -1.2513173828125002}

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
                chempots={},
                temperature_range=[1, 2, 3],
                annealing_temperature_range=[1, 2, 3],
                quenching_temperature_range=None,
            )
        print(exc.value)  # for debugging

        with pytest.raises(ValueError) as exc:
            self.fs.scan_temperature(
                chempots={},
                temperature_range=[1, 2, 3],
                annealing_temperature_range=None,
                quenching_temperature_range=[1, 2, 3],
            )
        print(exc.value)  # for debugging

        with pytest.raises(ValueError) as exc:
            self.fs.scan_temperature(
                chempots={},
                temperature_range=[],
                annealing_temperature_range=None,
                quenching_temperature_range=None,
            )
        print(exc.value)  # for debugging

        with pytest.raises(ValueError) as exc:
            self.fs.scan_temperature(
                limit=None,
                chempots=None,
                temperature_range=None,
                annealing_temperature_range=[],
                quenching_temperature_range=None,
            )
        print(exc.value)

    def test_get_interpolated_chempots(self):
        int_chempots = self.fs._get_interpolated_chempots({"a": -1, "b": -1}, {"a": 1, "b": 1}, 11)
        assert len(int_chempots) == 11
        assert int_chempots[0]["a"] == -1
        assert int_chempots[0]["b"] == -1
        assert int_chempots[-1]["a"] == 1
        assert int_chempots[-1]["b"] == 1
        assert int_chempots[5]["a"] == 0
        assert int_chempots[5]["b"] == 0

    def test__solve_and_append_chempots(self):
        with patch(
            "doped.interface.fermi_solver.FermiSolver.equilibrium_solve",
            return_value={
                "Fermi Level": [0.5],
                "Electrons (cm^-3)": [0.5],
                "Holes (cm^-3)": [0.5],
                "Temperature": [300],
            },
        ):
            results = self.fs._solve_and_append_chempots({}, 300)
            assert results["Fermi Level"] == [0.5]
            assert results["Electrons (cm^-3)"] == [0.5]
            assert results["Holes (cm^-3)"] == [0.5]
            assert results["Temperature"] == [300]

    def test_solve_and_append_chempots_pseudo(self):
        with patch(
            "doped.interface.fermi_solver.FermiSolver.pseudo_equilibrium_solve",
            return_value={
                "Fermi Level": [0.5],
                "Electrons (cm^-3)": [0.5],
                "Holes (cm^-3)": [0.5],
                "Annealing Temperature": [300],
                "Quenched Temperature": [300],
            },
        ):
            results = self.fs._solve_and_append_chempots_pseudo(
                {}, quenched_temperature=300, annealing_temperature=300
            )
            assert results["Fermi Level"] == [0.5]
            assert results["Electrons (cm^-3)"] == [0.5]
            assert results["Holes (cm^-3)"] == [0.5]
            assert results["Annealing Temperature"] == [300]
            assert results["Quenched Temperature"] == [300]

    def test__add_effective_dopant_concentration_and_solve(self):
        with patch(
            "doped.interface.fermi_solver.FermiSolver.equilibrium_solve",
            return_value={
                "Fermi Level": [0.5],
                "Electrons (cm^-3)": [0.5],
                "Holes (cm^-3)": [0.5],
                "Temperature": [300],
            },
        ):
            results = self.fs._add_effective_dopant_concentration_and_solve(
                {}, effective_dopant_concentration=-1, temperature=1
            )
            assert results["Dopant (cm^-3)"] == 1

    def test__add_effective_dopant_concentration_and_solve_raises(self):
        with patch(
            "doped.interface.fermi_solver.FermiSolver.equilibrium_solve",
            return_value={
                "Fermi Level": [0.5],
                "Electrons (cm^-3)": [0.5],
                "Holes (cm^-3)": [0.5],
                "Temperature": [300],
            },
        ):
            with pytest.raises(ValueError) as exc:
                self.fs._add_effective_dopant_concentration_and_solve({}, temperature=1)
            print(exc.value)

    def test__add_effective_dopant_concentration_and_solve_pseudo(self):
        with patch(
            "doped.interface.fermi_solver.FermiSolver.pseudo_equilibrium_solve",
            return_value={
                "Fermi Level": [0.5],
                "Electrons (cm^-3)": [0.5],
                "Holes (cm^-3)": [0.5],
                "Annealing Temperature": [300],
                "Quenched Temperature": [300],
            },
        ):
            results = self.fs._add_effective_dopant_concentration_and_solve_pseudo(
                {}, effective_dopant_concentration=-1, quenched_temperature=300, annealing_temperature=300
            )
            assert results["Dopant (cm^-3)"] == 1

    def test__add_effective_dopant_concentration_and_solve_pseudo_raises(self):
        with patch(
            "doped.interface.fermi_solver.FermiSolver.pseudo_equilibrium_solve",
            return_value={
                "Fermi Level": [0.5],
                "Electrons (cm^-3)": [0.5],
                "Holes (cm^-3)": [0.5],
                "Annealing Temperature": [300],
                "Quenched Temperature": [300],
            },
        ):
            with pytest.raises(ValueError) as exc:
                self.fs._add_effective_dopant_concentration_and_solve_pseudo(
                    {}, quenched_temperature=300, annealing_temperature=300
                )
            print(exc.value)


class FermiSolverDopedTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = "../examples/CdTe/CdTe_LZ_thermo_wout_meta.json.gz"
        self.bulk_dos_vr_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.thermo = loadfn(self.thermo_path)
        self.fs = FermiSolverDoped(self.thermo, self.bulk_dos_vr_path)

    def test__init__(self):
        assert self.fs.defect_thermodynamics == self.thermo
        assert isinstance(self.fs.bulk_dos, FermiDos)
        assert self.fs.bulk_dos.nelecs == 18.0

    def test_get_fermi_level_and_carriers(self):
        with patch(
            "doped.thermodynamics.DefectThermodynamics.get_equilibrium_fermi_level",
            return_value=(0.5, 0.5, 0.5),
        ):
            fermi_level, electrons, holes = self.fs._get_fermi_level_and_carriers({}, 300)
            assert fermi_level == 0.5
            assert electrons == 0.5
            assert holes == 0.5

    def test_equilibrium_solve(self):
        equilibrium_results = self.fs.equilibrium_solve({"Cd": 0.0, "Te": -1.2513173828125002}, 300)
        assert np.isclose(equilibrium_results["Fermi Level"][0], 0.8638811445611926, rtol=1e-4)
        assert np.isclose(equilibrium_results["Electrons (cm^-3)"][0], 1771777.3797365315, rtol=1e-4)
        assert np.isclose(equilibrium_results["Holes (cm^-3)"][0], 30972.77981875159, rtol=1e-4)
        assert np.isclose(equilibrium_results["Temperature"][0], 300, rtol=1e-4)
        assert np.isclose(equilibrium_results["Concentration (cm^-3)"][0], 9.413e-06, rtol=1e-4)

    def test_pseudo_equilibrium_solve(self):
        pseudo_equilibrium_results = self.fs.pseudo_equilibrium_solve(
            {"Cd": 0.0, "Te": -1.2513173828125002}, annealing_temperature=900, quenched_temperature=300
        )

        assert np.isclose(pseudo_equilibrium_results["Fermi Level"][0], 1.4121535299345318, rtol=1e-4)
        assert np.isclose(
            pseudo_equilibrium_results["Electrons (cm^-3)"][0], 2864732025029575.0, rtol=1e-4
        )
        assert np.isclose(
            pseudo_equilibrium_results["Holes (cm^-3)"][0], 1.9072582125691956e-05, rtol=1e-4
        )
        assert np.isclose(pseudo_equilibrium_results["Annealing Temperature"][0], 900.0, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Quenched Temperature"][0], 300.0, rtol=1e-4)
        assert np.isclose(
            pseudo_equilibrium_results["Concentration (cm^-3)"][0], 723879713527.0284, rtol=1e-4
        )


@unittest.skipIf(not py_sc_fermi_installed, "py_sc_fermi is not installed")
class FermiSolverPyScFermiTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = "../examples/CdTe/CdTe_LZ_thermo_wout_meta.json.gz"
        self.bulk_dos_vr_path = "../examples/CdTe/CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"
        self.thermo = loadfn(self.thermo_path)
        self.fs = FermiSolverPyScFermi(self.thermo, self.bulk_dos_vr_path, multiplicity_scaling=32)

    def test__init__(self):
        assert self.fs.defect_thermodynamics == self.thermo

    def test_equilibrium_solve(self):
        equilibrium_results = self.fs.equilibrium_solve({"Cd": 0.0, "Te": -1.2513173828125002}, 300)
        assert np.isclose(equilibrium_results["Fermi Level"][0], 8.638481e-01, rtol=1e-4)
        assert np.isclose(equilibrium_results["Electrons (cm^-3)"][0], 1.776159e06, rtol=1e-4)
        assert np.isclose(equilibrium_results["Holes (cm^-3)"][0], 3.089355e04, rtol=1e-4)
        assert np.isclose(equilibrium_results["Temperature"][0], 300, rtol=1e-4)

    def test_pseudo_equilibrium_solve(self):
        pseudo_equilibrium_results = self.fs.pseudo_equilibrium_solve(
            {"Cd": 0.0, "Te": -1.2513173828125002},
            annealing_temperature=900,
            quenched_temperature=300,
        )
        assert np.isclose(pseudo_equilibrium_results["Fermi Level"][0], 1.412081, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Electrons (cm^-3)"][0], 2.867398e15, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Holes (cm^-3)"][0], 1.905303e-05, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Annealing Temperature"][0], 900.0, rtol=1e-4)
        assert np.isclose(pseudo_equilibrium_results["Quenched Temperature"][0], 300.0, rtol=1e-4)

    def test_generate_defect_system(self):
        defect_system = self.fs.generate_defect_system(300, {"Cd": 0.0, "Te": -1.2513173828125002})
        assert defect_system.temperature == 300
        assert defect_system.dos.bandgap == 1.4969
        assert defect_system.dos.nelect == 18.0
        assert np.isclose(defect_system.volume, 70.0401764998796, rtol=1e-8)

        assert defect_system.defect_species_by_name("v_Cd").nsites == 1
        assert defect_system.defect_species_by_name("v_Cd").charge_states[0].energy == 3.149852420000008
        assert defect_system.defect_species_by_name("v_Cd").charge_states[0].degeneracy == 6.0

    def test_generate_annealed_defect_system(self):
        defect_system = self.fs.generate_annealed_defect_system(
            chempots={"Cd": 0.0, "Te": -1.2513173828125002},
            annealing_temperature=900,
            quenched_temperature=300,
        )
        assert defect_system.temperature == 300
        assert defect_system.dos.bandgap == 1.4969
        assert defect_system.dos.nelect == 18.0
        assert np.isclose(defect_system.volume, 70.0401764998796, rtol=1e-8)

        assert defect_system.defect_species_by_name("v_Cd").nsites == 1
        assert defect_system.defect_species_by_name("v_Cd").charge_states[0].energy == 3.149852420000008
        assert defect_system.defect_species_by_name("v_Cd").charge_states[0].degeneracy == 6.0


class TestChemicalPotentialGrid(unittest.TestCase):
    def setUp(self):
        self.chempots = loadfn("../examples/Cu2SiSe3/Cu2SiSe3_chempots.json")
        self.grid = ChemicalPotentialGrid.from_chempots(self.chempots)

    def test_init(self):
        assert isinstance(self.grid.vertices, pd.DataFrame)
        assert len(self.grid.vertices) == 7
        assert np.isclose(max(self.grid.vertices["Cu"]), 0.0)
        assert np.isclose(max(self.grid.vertices["Si"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(self.grid.vertices["Se"]), 0.0)
        assert np.isclose(min(self.grid.vertices["Cu"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["Si"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["Se"]), -0.758105, rtol=1e-5)

    def test_get_grid(self):
        grid_df = self.grid.get_grid(100)
        assert isinstance(grid_df, pd.DataFrame)
        assert len(self.grid.vertices) == 7
        assert np.isclose(max(self.grid.vertices["Cu"]), 0.0)
        assert np.isclose(max(self.grid.vertices["Si"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(self.grid.vertices["Se"]), 0.0)
        assert np.isclose(min(self.grid.vertices["Cu"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["Si"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(self.grid.vertices["Se"]), -0.758105, rtol=1e-5)
        assert len(grid_df) == 3886


class FermiSolver3DTestCase(unittest.TestCase):
    def setUp(self):
        self.thermo_path = loadfn("../examples/Cu2SiSe3/Cu2SiSe3_thermo.json")
        self.bulk_dos_path = "../examples/Cu2SiSe3/vasprun.xml.gz"
        self.fs = FermiSolverDoped(self.thermo_path, self.bulk_dos_path)
        self.py_fs = FermiSolverPyScFermi(self.thermo_path, self.bulk_dos_path)

    def test_scan_chemical_potential_grid(self):

        results = self.fs.scan_chemical_potential_grid(
            n_points=5, annealing_temperature=1000, quenching_temperature=300
        )
        assert len(results) == 156
        assert np.isclose(max(results["Cu"]), 0.0)
        assert np.isclose(max(results["Si"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(results["Se"]), 0.0)
        assert np.isclose(min(results["Cu"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(results["Si"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(results["Se"]), -0.758105, rtol=1e-5)
        assert np.isclose(max(results["Fermi Level"]), 0.014048268, rtol=1e-4)
        assert np.isclose(min(results["Fermi Level"]), -0.074598, rtol=1e-4)

    def test_min_max_X(self):

        results = self.fs.min_max_X(
            target="Electrons (cm^-3)",
            min_or_max="max",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], 0.0)

        results = self.fs.min_max_X(
            target="Electrons (cm^-3)",
            min_or_max="min",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], -0.46355805460937427)

        results = self.fs.min_max_X(
            target="Holes (cm^-3)",
            min_or_max="max",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], -0.46355805460937427)

        results = self.fs.min_max_X(
            target="Holes (cm^-3)",
            min_or_max="min",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], 0)

        results = self.py_fs.min_max_X(
            target="v_Se",
            min_or_max="max",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Se"][0], -0.758105, rtol=1e-4)

    @unittest.skipIf(not py_sc_fermi_installed, "py_sc_fermi is not installed")
    def test_min_max_X_pyscf(self):

        results = self.py_fs.min_max_X(
            target="Electrons (cm^-3)",
            min_or_max="max",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], 0.0)

        results = self.py_fs.min_max_X(
            target="Electrons (cm^-3)",
            min_or_max="min",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], -0.46355805460937427, rtol=1e-5)

        results = self.py_fs.min_max_X(
            target="Holes (cm^-3)",
            min_or_max="max",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], -0.46355805460937427)

        results = self.py_fs.min_max_X(
            target="Holes (cm^-3)",
            min_or_max="min",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Cu"][0], 0)

        results = self.py_fs.min_max_X(
            target="v_Se",
            min_or_max="max",
            tolerance=0.001,
            annealing_temperature=1000,
            quenching_temperature=300,
            n_points=5,
            processes=1,
        )
        assert np.isclose(results["Se"][0], -0.758105, rtol=1e-4)

    @unittest.skipIf(not py_sc_fermi_installed, "py_sc_fermi is not installed")
    def test_scan_chemical_potential_grid_pyscf(self):

        results = self.py_fs.scan_chemical_potential_grid(
            n_points=5, annealing_temperature=1000, quenching_temperature=300
        )
        assert len(results) == 156
        assert np.isclose(max(results["Cu"]), 0.0)
        assert np.isclose(max(results["Si"]), -0.077858, rtol=1e-5)
        assert np.isclose(max(results["Se"]), 0.0)
        assert np.isclose(min(results["Cu"]), -0.463558, rtol=1e-5)
        assert np.isclose(min(results["Si"]), -1.708951, rtol=1e-5)
        assert np.isclose(min(results["Se"]), -0.758105, rtol=1e-5)
        assert np.isclose(max(results["Fermi Level"]), 0.0155690, rtol=1e-4)
        assert np.isclose(min(results["Fermi Level"]), -0.07285, rtol=1e-4)
