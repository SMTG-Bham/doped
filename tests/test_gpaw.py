"""
Tests for the GPAW interface in ``doped.io.gpaw``.
"""

import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.structure import Structure
from test_utils import gpaw_data_dir

from doped.io.gpaw.inputs import DefectRelaxSet
from doped.io.gpaw.outputs import _find_gpaw_output
from doped.parsing import DefectParser, DefectsParser


class GPAWTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = "gpaw_test_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create a simple structure for testing input generation
        self.structure = Structure.from_file(os.path.join(self.data_dir, "Cu_prim_POSCAR"))

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_gpaw_defect_relax_set(self):
        # Test with Structure
        relax_set = DefectRelaxSet(self.structure, charge_state=1)
        relax_set.write_input(self.output_dir)

        assert os.path.exists(os.path.join(self.output_dir, "relax.py"))
        assert os.path.exists(os.path.join(self.output_dir, "structure.cif"))
        written_structure = Structure.from_file(os.path.join(self.output_dir, "structure.cif"))
        assert written_structure.matches(self.structure)

        with open(os.path.join(self.output_dir, "relax.py")) as f:
            content = f.read()
            assert "charge=1" in content
            assert "mode=PW(ecut=400)" in content  # Default

    def test_gpaw_defect_relax_set_custom(self):
        # Test with custom settings
        gpaw_settings = {
            "mode": {"name": "pw", "ecut": 400},
            "xc": "PBE",
            "kpts": {"size": (2, 2, 2), "gamma": True},
        }
        relax_set = DefectRelaxSet(self.structure, charge_state=-1, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)

        with open(os.path.join(self.output_dir, "relax.py")) as f:
            content = f.read()
            assert "charge=-1" in content
            assert "mode=PW(ecut=400)" in content
            assert "'size': (2, 2, 2)" in content
            assert "from gpaw import GPAW, PW, LCAO, FD" in content

    def test_gpaw_defect_relax_set_lcao(self):
        # Test with LCAO mode
        gpaw_settings = {
            "mode": {"name": "lcao", "basis": "dzp"},
        }
        relax_set = DefectRelaxSet(self.structure, charge_state=0, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)

        with open(os.path.join(self.output_dir, "relax.py")) as f:
            content = f.read()
            assert "mode=LCAO(basis='dzp')" in content
            assert "from gpaw import GPAW, PW, LCAO, FD" in content

    def test_gpaw_singlepoint_set(self):
        singlepoint_set = DefectRelaxSet(
            self.structure,
            charge_state=1,
            calculation_type="singlepoint",
        )
        singlepoint_set.write_input(self.output_dir)

        script_path = os.path.join(self.output_dir, "singlepoint.py")
        assert os.path.exists(script_path)
        with open(script_path) as file:
            content = file.read()

        assert "charge=1" in content
        assert "calc.write('singlepoint.gpw')" in content
        assert "ase.optimize" not in content
        assert "dyn.run" not in content

    def test_gpaw_initial_magnetic_moments(self):
        magnetic_moments = [1.0] + [0.0] * (len(self.structure) - 1)
        input_set = DefectRelaxSet(
            self.structure,
            gpaw_settings={"initial_magnetic_moments": magnetic_moments},
            calculation_type="singlepoint",
        )
        input_set.write_input(self.output_dir)

        with open(os.path.join(self.output_dir, "singlepoint.py")) as file:
            content = file.read()

        assert f"atoms.set_initial_magnetic_moments({magnetic_moments!r})" in content
        assert "initial_magnetic_moments=" not in content

        invalid_set = DefectRelaxSet(
            self.structure,
            gpaw_settings={"initial_magnetic_moments": [*magnetic_moments, 0.0]},
        )
        with pytest.raises(ValueError, match="one value per atom"):
            invalid_set.write_input(self.output_dir)

    def test_gpaw_defects_set(self):
        """
        Test the generic ``DefectsSet`` workflow for GPAW: building input sets
        for a full ``DefectsGenerator`` output and writing them to the
        ``<defect name>/`` folder structure, as with VASP.
        """
        from doped.generation import DefectsGenerator
        from doped.io.gpaw.inputs import DefectsSet

        defect_gen = DefectsGenerator(self.structure)
        defects_set = DefectsSet(defect_gen, gpaw_settings={"mode": {"name": "pw", "ecut": 500}})
        assert len(defects_set.defect_sets) == len(defect_gen.defect_entries)

        defects_set.write_files(output_path=self.output_dir, processes=1)
        written = set(os.listdir(self.output_dir))
        assert set(defect_gen.defect_entries) <= written  # one folder per defect, doped-named
        assert "bulk" in written  # the neutral bulk reference, written once
        assert any(name.endswith(".json.gz") for name in written)  # provenance

        defect_species = next(iter(defect_gen.defect_entries))
        defect_folder = os.path.join(self.output_dir, defect_species)
        assert sorted(os.listdir(defect_folder)) == ["relax.py", "structure.cif"]
        with open(os.path.join(defect_folder, "relax.py")) as script_file:
            script = script_file.read()
        assert "mode=PW(ecut=500)" in script  # settings forwarded to each defect
        assert f"charge={defect_gen.defect_entries[defect_species].charge_state}" in script

    def test_find_gpaw_output(self):
        calc_dir = os.path.join(self.output_dir, "calculation")
        os.makedirs(calc_dir)

        custom_output = os.path.join(calc_dir, "custom.gpw")
        Path(custom_output).touch()
        assert _find_gpaw_output(calc_dir) == custom_output

        relaxed_output = os.path.join(calc_dir, "relaxed.gpw")
        Path(relaxed_output).touch()
        assert _find_gpaw_output(calc_dir) == relaxed_output
        assert _find_gpaw_output(custom_output) == custom_output

    def test_gpaw_kumagai_correction_mgo(self):
        """
        Test that the GPAW parser correctly extracts electrostatic potentials
        and calculates the eFNV (Kumagai) correction for multiple charge states
        using real static ``.gpw`` files (both ``v_Mg`` and ``Mg_O`` defects).
        """
        pytest.importorskip("gpaw")
        # Path to the static test data directories
        gpaw_mgo_dir = os.path.join(gpaw_data_dir, "MgO")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "Bulk test directory missing!"

        dp_gpaw = DefectsParser(
            output_path=gpaw_mgo_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
            calculator="gpaw",
            json_filename=False,
        )

        defect_dict = dp_gpaw.defect_dict

        # Expected Kumagai corrections mapped by DEFECT NAME
        # (to handle multiple defects with the same charge)
        expected_corrections = {
            "v_Mg_+1": -0.05491517,
            "v_Mg_-2": 1.20301268,
            "Mg_O_+1": 0.36016471,
        }
        assert expected_corrections.keys() <= defect_dict.keys()

        for defect_name, expected_energy in expected_corrections.items():
            entry = defect_dict[defect_name]
            charge = entry.charge_state
            assert "kumagai_charge_correction" in entry.corrections
            calculated_energy = float(entry.corrections["kumagai_charge_correction"])

            np.testing.assert_allclose(
                calculated_energy,
                expected_energy,
                atol=1e-3,
                err_msg=f"Failed for defect {defect_name} (Charge {charge})!",
            )

        # --- Explicitly Test the Unrelaxed Mg_O +1 State ---
        mg_o_unrelaxed_dir = os.path.join(gpaw_mgo_dir, "Mg_O_unrelaxed")
        assert os.path.exists(mg_o_unrelaxed_dir), "Unrelaxed Mg_O +1 test directory missing!"

        mg_o_unrelaxed_entry = DefectParser.from_paths(
            defect_path=mg_o_unrelaxed_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
            charge_state=1,
            calculator="gpaw",
        ).defect_entry
        mg_o_unrelaxed_entry.get_kumagai_correction()

        assert "kumagai_charge_correction" in mg_o_unrelaxed_entry.corrections
        calculated_mg_o_unrelaxed = float(mg_o_unrelaxed_entry.corrections["kumagai_charge_correction"])

        np.testing.assert_allclose(
            calculated_mg_o_unrelaxed,
            0.39874949,
            atol=1e-3,
            err_msg="Failed for unrelaxed Mg_O +1 state!",
        )

    def test_gpaw_freysoldt_correction_mgo(self):
        """
        Test that the GPAW parser supports the Freysoldt (FNV) correction via
        manual invocation after parsing, using the MgO test data.
        """
        pytest.importorskip("gpaw")
        gpaw_mgo_dir = os.path.join(gpaw_data_dir, "MgO")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "MgO bulk test directory missing!"

        dp_gpaw = DefectsParser(
            output_path=gpaw_mgo_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
            calculator="gpaw",
            json_filename=False,
        )

        defect_dict = dp_gpaw.defect_dict

        print("\n--- Calculated Freysoldt (FNV) Corrections ---")

        # eFNV and FNV must agree for a physically-valid case; ``v_Mg_-2`` is the only one here
        # (``v_Mg_+1`` is 3 holes in the VB and ``Mg_O_+1`` electrons in the CB, i.e.
        # delocalised-carrier states for which the point-charge correction model does not apply):
        expected_fnv = {
            "v_Mg_+1": -0.10899449,
            "v_Mg_-2": 1.27743123,
            "Mg_O_+1": 0.12075791,
        }
        assert expected_fnv.keys() <= defect_dict.keys()

        for defect_name, expected_energy in expected_fnv.items():
            defect_entry = defect_dict[defect_name]
            defect_entry.corrections.pop("kumagai_charge_correction", None)
            defect_entry.corrections_metadata.pop("kumagai_charge_correction", None)
            # ``doped`` parses the site potentials and uses eFNV by preference, so the planar-averaged
            # potentials are given here explicitly (loaded from the ``.gpw`` files by the GPAW backend):
            defect_entry.get_freysoldt_correction(
                defect_planar_averaged_potentials=defect_entry.calculation_metadata["defect_path"],
                bulk_planar_averaged_potentials=gpaw_bulk_dir,
            )
            calculated_energy = float(defect_entry.corrections["freysoldt_charge_correction"])

            print(f"{defect_name} (Charge {defect_entry.charge_state}): {calculated_energy:.4f} eV")

            np.testing.assert_allclose(
                calculated_energy,
                expected_energy,
                atol=1e-3,
                err_msg=f"FNV value mismatch for {defect_name}!",
            )

    def test_gpaw_calculator_metadata_and_potentials_from_input(self):
        """
        Test that parsed GPAW entries record ``calculation_metadata
        ["calculator"] = "gpaw"``, so that ``doped``'s backend lookups (e.g.
        for the charge-correction potentials) dispatch to
        ``doped.io.gpaw.outputs`` rather than defaulting to VASP, and that the
        corresponding ``get_potentials_from_input()`` gives the same potentials
        as those parsed up-front.
        """
        pytest.importorskip("gpaw")
        from doped.io import get_calculation_outputs
        from doped.io.gpaw.outputs import get_potentials_from_input

        gpaw_mgo_dir = os.path.join(gpaw_data_dir, "MgO")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")
        dp_gpaw = DefectsParser(
            output_path=gpaw_mgo_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
            calculator="gpaw",
            json_filename=False,
        )

        for defect_entry in dp_gpaw.defect_dict.values():
            assert defect_entry.calculation_metadata["calculator"] == "gpaw"

        # re-loading from the calculation directory reproduces the up-front parse:
        entry = dp_gpaw.defect_dict["v_Mg_-2"]
        site_potentials = get_potentials_from_input(gpaw_bulk_dir, potential_type="site", dir_type="bulk")
        np.testing.assert_allclose(site_potentials, entry.calculation_metadata["bulk_site_potentials"])
        planar_potentials = get_potentials_from_input(
            gpaw_bulk_dir, potential_type="planar", dir_type="bulk"
        )
        bulk_outputs = get_calculation_outputs(gpaw_bulk_dir, calculator="gpaw", label="bulk")
        # the lazy getter keys axes by string, ``CalculationOutputs`` by int, as with VASP:
        assert {int(axis) for axis in planar_potentials} == set(bulk_outputs.planar_averaged_potentials)
        for axis, potentials in planar_potentials.items():
            np.testing.assert_allclose(potentials, bulk_outputs.planar_averaged_potentials[int(axis)])

        # already-parsed potentials are returned as-is, and anything else is rejected:
        assert get_potentials_from_input(planar_potentials) is planar_potentials
        with pytest.raises(TypeError, match=r"bulk potentials input must be either a path"):
            get_potentials_from_input(12345, dir_type="bulk")

    def test_gpaw_calculation_outputs(self):
        """
        Test the ``doped.io`` backend protocol entry point,
        ``get_calculation_outputs()``, which is what lets GPAW calculations be
        parsed with ``doped``'s generic machinery.
        """
        pytest.importorskip("gpaw")
        from doped.io import get_calculation_outputs

        gpaw_mgo_dir = os.path.join(gpaw_data_dir, "MgO")
        bulk_outputs = get_calculation_outputs(
            os.path.join(gpaw_mgo_dir, "bulk"), calculator="gpaw", label="bulk"
        )
        assert bulk_outputs.calculator == "gpaw"
        assert bulk_outputs.charge == 0
        assert len(bulk_outputs.structure) == len(bulk_outputs.site_potentials)
        assert set(bulk_outputs.planar_averaged_potentials) == {0, 1, 2}
        assert bulk_outputs.converged_electronic
        assert bulk_outputs.nelect > 0
        assert bulk_outputs.band_gap > 0  # band edges are only taken for the (neutral) bulk
        assert bulk_outputs.vbm < bulk_outputs.efermi < bulk_outputs.cbm
        assert "gpaw_parameters" in bulk_outputs.run_metadata
        # eigenvalues are (n_kpoints, n_bands, 2), i.e. energy and occupancy, as pymatgen expects:
        assert all(eigs.ndim == 3 and eigs.shape[-1] == 2 for eigs in bulk_outputs.eigenvalues.values())
        assert len(bulk_outputs.kpoint_weights) == len(bulk_outputs.kpoint_coords)

        defect_outputs = get_calculation_outputs(
            os.path.join(gpaw_mgo_dir, "v_Mg_-2"), calculator="gpaw", label="defect"
        )
        assert defect_outputs.charge == -2
        assert defect_outputs.band_gap is None  # Fermi-level band edges are meaningless when charged
        assert len(defect_outputs.structure) == len(bulk_outputs.structure) - 1  # a vacancy

    def test_gpaw_generic_defects_parser(self):
        """
        Test the structure-derived defect naming and the calculation metadata
        that ``doped``'s calculator-agnostic ``DefectsParser`` provides for
        GPAW calculations.
        """
        pytest.importorskip("gpaw")
        from doped.parsing import DefectsParser

        gpaw_mgo_dir = os.path.join(gpaw_data_dir, "MgO")
        dp = DefectsParser(  # default ``processes``, so this also covers the multiprocessing path
            output_path=gpaw_mgo_dir,
            bulk_path=os.path.join(gpaw_mgo_dir, "bulk"),
            dielectric=8.8963,
            calculator="gpaw",
            json_filename=False,  # don't write a parsed-dict JSON into the test data directory
        )

        # names come from structure analysis, so the unrelaxed folder no longer collides:
        assert set(dp.defect_dict) == {"v_Mg_+1", "v_Mg_-2", "Mg_O_+1", "Mg_O_unrelaxed_+1"}

        expected_corrections = {
            "v_Mg_+1": -0.05491517,
            "v_Mg_-2": 1.20301268,
            "Mg_O_+1": 0.36016471,
            "Mg_O_unrelaxed_+1": 0.39874916,
        }
        for name, expected_energy in expected_corrections.items():
            entry = dp.defect_dict[name]
            np.testing.assert_allclose(
                float(entry.corrections["kumagai_charge_correction"]),
                expected_energy,
                atol=1e-3,
                err_msg=f"eFNV value mismatch for {name}!",
            )
            assert entry.calculation_metadata["calculator"] == "gpaw"
            # metadata the GPAW-specific parser does not provide:
            run_metadata = entry.calculation_metadata["run_metadata"]
            assert run_metadata["defect_gpaw_parameters"]["mode"]["name"] == "pw"
            assert run_metadata["bulk_gpaw_parameters"]["charge"] == 0
            # the bulk & defect calculations here differ only in charge, which is excluded:
            assert entry.calculation_metadata["mismatching_gpaw_parameters"] is False
            assert entry.calculation_metadata["relaxed point symmetry"]

    def test_gpaw_graphene_2d_handling(self):
        """
        Test that the GPAW parser handles highly anisotropic 2D supercells
        (Graphene) without crashing during the Kumagai correction / defect
        region radius calculation.

        Tests multiple defects spanning vacancies, interstitials, and
        substitutions to ensure robustness.
        """
        pytest.importorskip("gpaw")

        # Path to the static test data directories
        gpaw_graphene_dir = os.path.join(gpaw_data_dir, "Graphene")
        gpaw_bulk_dir = os.path.join(gpaw_graphene_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "Graphene bulk test directory missing!"

        # Initialize the parser
        dp_gpaw = DefectsParser(
            output_path=gpaw_graphene_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=np.diag([1e6, 1e6, 1.0]),
            calculator="gpaw",
            json_filename=False,
        )

        defect_dict = dp_gpaw.defect_dict

        # Expected corrections with a metallic in-plane response and vacuum-like out-of-plane response
        expected_corrections = {
            "v_C_+1": -2.92459981,
            "C_i_C3v_+4": -46.09056677,  # q^2 scaling...
            "N_C_-2": -2.31309662,
        }

        for defect_name, expected_energy in expected_corrections.items():
            assert defect_name in defect_dict, f"{defect_name} missing from parsed defects!"
            entry = defect_dict[defect_name]

            # Verify the Kumagai correction was calculated (even if physically inaccurate for 2D)
            assert "kumagai_charge_correction" in entry.corrections
            calculated_energy = float(entry.corrections["kumagai_charge_correction"])

            np.testing.assert_allclose(
                calculated_energy,
                expected_energy,
                atol=1e-3,
                err_msg=f"Graphene 2D Kumagai calculation failed for {defect_name}!",
            )
