"""
Tests for the GPAW interface in ``doped.gpaw``.
"""

import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from doped.gpaw import GPAWDefectRelaxSet, GPAWDefectsParser, _find_gpaw_output


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
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=1)
        relax_set.write_input(self.output_dir)

        assert os.path.exists(os.path.join(self.output_dir, "relax.py"))
        assert os.path.exists(os.path.join(self.output_dir, "structure.cif"))
        written_structure = Structure.from_file(os.path.join(self.output_dir, "structure.cif"))
        assert written_structure.matches(self.structure)

        with open(os.path.join(self.output_dir, "relax.py")) as f:
            content = f.read()
            assert "charge=1" in content
            assert "mode=PW(ecut=400)" in content  # Default
            assert "legacy_gpaw=True" in content

    def test_gpaw_defect_relax_set_custom(self):
        # Test with custom settings
        gpaw_settings = {
            "mode": {"name": "pw", "ecut": 400},
            "xc": "PBE",
            "kpts": {"size": (2, 2, 2), "gamma": True},
            "legacy_gpaw": False,
        }
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=-1, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)

        with open(os.path.join(self.output_dir, "relax.py")) as f:
            content = f.read()
            assert "charge=-1" in content
            assert "mode=PW(ecut=400)" in content
            assert "'size': (2, 2, 2)" in content
            assert "legacy_gpaw=False" in content
            assert "from gpaw import GPAW, PW, LCAO, FD" in content

    def test_gpaw_defect_relax_set_lcao(self):
        # Test with LCAO mode
        gpaw_settings = {
            "mode": {"name": "lcao", "basis": "dzp"},
        }
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=0, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)

        with open(os.path.join(self.output_dir, "relax.py")) as f:
            content = f.read()
            assert "mode=LCAO(basis='dzp')" in content
            assert "from gpaw import GPAW, PW, LCAO, FD" in content

    def test_gpaw_singlepoint_set(self):
        singlepoint_set = GPAWDefectRelaxSet(
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
        assert "legacy_gpaw=True" in content
        assert "calc.write('singlepoint.gpw.gz')" in content
        assert "ase.optimize" not in content
        assert "dyn.run" not in content

    def test_find_gpaw_output(self):
        calc_dir = os.path.join(self.output_dir, "calculation")
        os.makedirs(calc_dir)

        custom_output = os.path.join(calc_dir, "custom.gpw")
        Path(custom_output).touch()
        assert _find_gpaw_output(calc_dir) == custom_output

        relaxed_output = os.path.join(calc_dir, "relaxed.gpw.gz")
        Path(relaxed_output).touch()
        assert _find_gpaw_output(calc_dir) == relaxed_output
        assert _find_gpaw_output(custom_output) == custom_output

    def test_gpaw_kumagai_correction_mgo(self):
        """
        Test that the GPAW parser correctly extracts electrostatic potentials
        and calculates the eFNV (Kumagai) correction for multiple charge states
        using real static ``.gpw(.gz)`` files (both ``v_Mg`` and ``Mg_O``
        defects).
        """
        pytest.importorskip("gpaw")
        from doped.gpaw import GPAWParser, get_gpaw_defect_entry

        # Path to the static test data directories
        gpaw_mgo_dir = os.path.join(self.data_dir, "gpaw_mgo_test")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "Bulk test directory missing!"

        # Initialize the parser
        dp_gpaw = GPAWDefectsParser(output_path=gpaw_mgo_dir, bulk_path=gpaw_bulk_dir, dielectric=10.0)

        # Parse all relaxed defects in the folder
        defect_dict = dp_gpaw.parse_all()
        assert len(defect_dict) >= 4, "Not all defects were parsed!"

        # Expected Kumagai corrections mapped by DEFECT NAME
        # (to handle multiple defects with the same charge)
        expected_corrections = {
            "v_Mg_Oh_O2.10_+1": 0.303790,
            "v_Mg_Oh_O2.10_-1": 0.091121,
            "v_Mg_Oh_O2.10_-2": 0.575566,
            "Mg_O_Oh_Mg2.10O2.98Mg3.65b_+1": 0.224,
            "Mg_O_Oh_Mg2.10O2.98Mg3.65b_+2": 0.808,
            "Mg_O_Oh_Mg2.10O2.98Mg3.65b_+4": 3.052,
            "v_O_Oh_Mg2.10_+2": 0.918,
            "O_i_Cs_O1.71_-2": 0.571,
            "O_Mg_Oh_O2.10_-2": 0.502,
        }

        for defect_name, entry in defect_dict.items():
            charge = entry.charge_state

            # Neutral defects have no Kumagai correction
            if charge == 0:
                assert "kumagai_charge_correction" not in entry.corrections
                continue

            if defect_name in expected_corrections:
                expected_energy = expected_corrections[defect_name]
                assert "kumagai_charge_correction" in entry.corrections

                calculated_energy = float(entry.corrections["kumagai_charge_correction"])

                np.testing.assert_allclose(
                    calculated_energy,
                    expected_energy,
                    atol=1e-3,
                    err_msg=f"Failed for defect {defect_name} (Charge {charge})!",
                )

        # --- Explicitly Test the Unrelaxed v_Mg +1 State ---
        v_mg_unrelaxed_dir = os.path.join(gpaw_mgo_dir, "v_Mg_+1_unrelaxed")
        assert os.path.exists(v_mg_unrelaxed_dir), "Unrelaxed v_Mg +1 test directory missing!"

        bulk_parser = GPAWParser(os.path.join(gpaw_bulk_dir, "relaxed.gpw.gz"))
        v_mg_unrelaxed_entry = get_gpaw_defect_entry(
            defect_path=v_mg_unrelaxed_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=10.0,
            charge_state=1,
            bulk_parser=bulk_parser,
        )
        v_mg_unrelaxed_entry.get_kumagai_correction()

        assert "kumagai_charge_correction" in v_mg_unrelaxed_entry.corrections
        calculated_v_mg_unrelaxed = float(v_mg_unrelaxed_entry.corrections["kumagai_charge_correction"])

        np.testing.assert_allclose(
            calculated_v_mg_unrelaxed,
            0.303790,
            atol=1e-3,
            err_msg="Failed for unrelaxed v_Mg +1 state!",
        )

        # --- Explicitly Test the Unrelaxed Mg_O +1 State ---
        mg_o_unrelaxed_dir = os.path.join(gpaw_mgo_dir, "Mg_O_Oh_Mg2.10O2.98Mg3.65b_+1_unrelaxed")
        assert os.path.exists(mg_o_unrelaxed_dir), "Unrelaxed Mg_O +1 test directory missing!"

        bulk_parser = GPAWParser(os.path.join(gpaw_bulk_dir, "relaxed.gpw.gz"))
        mg_o_unrelaxed_entry = get_gpaw_defect_entry(
            defect_path=mg_o_unrelaxed_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=10.0,
            charge_state=1,
            bulk_parser=bulk_parser,
        )
        mg_o_unrelaxed_entry.get_kumagai_correction()

        assert "kumagai_charge_correction" in mg_o_unrelaxed_entry.corrections
        calculated_mg_o_unrelaxed = float(mg_o_unrelaxed_entry.corrections["kumagai_charge_correction"])

        # The value parsed from your terminal output for this specific unrelaxed calculation
        np.testing.assert_allclose(
            calculated_mg_o_unrelaxed, 0.155, atol=1e-3, err_msg="Failed for unrelaxed Mg_O +1 state!"
        )

    def test_gpaw_freysoldt_correction_mgo(self):
        """
        Test that the GPAW parser supports the Freysoldt (FNV) correction via
        manual invocation after parsing, using the MgO test data.
        """
        pytest.importorskip("gpaw")
        from doped.gpaw import GPAWParser, get_gpaw_defect_entry

        gpaw_mgo_dir = os.path.join(self.data_dir, "gpaw_mgo_test")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "MgO bulk test directory missing!"

        dp_gpaw = GPAWDefectsParser(output_path=gpaw_mgo_dir, bulk_path=gpaw_bulk_dir, dielectric=10.0)

        defect_dict = dp_gpaw.parse_all()

        print("\n--- Calculated Freysoldt (FNV) Corrections ---")

        # We only keep v_Mg here because FNV requires perfectly matched grids.
        # Mg_O triggered an (80,) vs (96,) grid mismatch with the bulk.
        expected_fnv = {
            "v_Mg_Oh_O2.10_-1": 0.4145,
            "v_Mg_Oh_O2.10_+1": -0.0999,
            "v_Mg_Oh_O2.10_-2": 1.1544,
        }

        for defect_name, defect_entry in defect_dict.items():
            if defect_entry.charge_state == 0:
                continue

            # Skip any defect not explicitly in our expected_fnv dictionary!
            # This prevents the parser from attempting FNV on the mismatched Mg_O grids.
            if defect_name not in expected_fnv:
                continue

            defect_entry.get_freysoldt_correction()
            calculated_energy = float(defect_entry.corrections["freysoldt_charge_correction"])

            print(f"{defect_name} (Charge {defect_entry.charge_state}): {calculated_energy:.4f} eV")

            np.testing.assert_allclose(
                calculated_energy,
                expected_fnv[defect_name],
                atol=1e-3,
                err_msg=f"FNV value mismatch for {defect_name}!",
            )

        # --- Explicitly Test the Unrelaxed v_Mg +1 State ---
        v_mg_unrelaxed_dir = os.path.join(gpaw_mgo_dir, "v_Mg_+1_unrelaxed")
        assert os.path.exists(v_mg_unrelaxed_dir), "Unrelaxed v_Mg +1 test directory missing!"

        bulk_parser = GPAWParser(os.path.join(gpaw_bulk_dir, "relaxed.gpw.gz"))
        v_mg_unrelaxed_entry = get_gpaw_defect_entry(
            defect_path=v_mg_unrelaxed_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=10.0,
            charge_state=1,
            bulk_parser=bulk_parser,
        )

        v_mg_unrelaxed_entry.get_freysoldt_correction()
        calculated_unrelaxed = float(v_mg_unrelaxed_entry.corrections["freysoldt_charge_correction"])

        print(f"v_Mg_+1_unrelaxed (Charge 1): {calculated_unrelaxed:.4f} eV")

        np.testing.assert_allclose(
            calculated_unrelaxed,
            -0.0999,
            atol=1e-3,
            err_msg="FNV value mismatch for unrelaxed defect!",
        )

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
        gpaw_graphene_dir = os.path.join(self.data_dir, "gpaw_graphene_test")
        gpaw_bulk_dir = os.path.join(gpaw_graphene_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "Graphene bulk test directory missing!"

        # Initialize the parser
        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_graphene_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=10.0,  # Dummy dielectric
        )

        # Parse the graphene defects
        defect_dict = dp_gpaw.parse_all()
        assert len(defect_dict) >= 5, "Not enough Graphene defects were parsed!"

        # Expected Kumagai corrections mapped by defect name (values from local run)
        # Note: The +4 charge state correction is ~41 eV due to the q^2 scaling
        # of the charge correction in a small 2D supercell.
        expected_corrections = {
            "v_C_D3h_C1.42_+1": 2.5146,
            "C_i_C3v_C2.00_+4": 41.0205,
            "N_i_C3v_C2.00_-3": 4.8420,
            "v_C_D3h_C1.42_-1": 1.5464,
            "N_C_D3h_C1.42_-2": 1.5279,
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
