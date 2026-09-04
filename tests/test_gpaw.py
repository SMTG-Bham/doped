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
        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_mgo_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
        )

        defect_dict = dp_gpaw.defect_dict

        # Expected Kumagai corrections mapped by DEFECT NAME
        # (to handle multiple defects with the same charge)
        expected_corrections = {
            "v_Mg_+1": 0.21323889,
            "v_Mg_-2": 0.52235841,
            "Mg_O_+1": 0.13920905,
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
        mg_o_unrelaxed_dir = os.path.join(gpaw_mgo_dir, "Mg_O_+1_unrelaxed")
        assert os.path.exists(mg_o_unrelaxed_dir), "Unrelaxed Mg_O +1 test directory missing!"

        bulk_parser = GPAWParser(os.path.join(gpaw_bulk_dir, "relaxed.gpw.gz"))
        mg_o_unrelaxed_entry = get_gpaw_defect_entry(
            defect_path=mg_o_unrelaxed_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
            charge_state=1,
            bulk_parser=bulk_parser,
        )
        bulk_parser.close()
        mg_o_unrelaxed_entry.get_kumagai_correction()

        assert "kumagai_charge_correction" in mg_o_unrelaxed_entry.corrections
        calculated_mg_o_unrelaxed = float(mg_o_unrelaxed_entry.corrections["kumagai_charge_correction"])

        np.testing.assert_allclose(
            calculated_mg_o_unrelaxed,
            -0.03567710,
            atol=1e-3,
            err_msg="Failed for unrelaxed Mg_O +1 state!",
        )

    def test_gpaw_freysoldt_correction_mgo(self):
        """
        Test that the GPAW parser supports the Freysoldt (FNV) correction via
        manual invocation after parsing, using the MgO test data.
        """
        pytest.importorskip("gpaw")
        gpaw_mgo_dir = os.path.join(self.data_dir, "gpaw_mgo_test")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")

        assert os.path.exists(gpaw_bulk_dir), "MgO bulk test directory missing!"

        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_mgo_dir,
            bulk_path=gpaw_bulk_dir,
            dielectric=8.8963,
        )

        defect_dict = dp_gpaw.defect_dict

        print("\n--- Calculated Freysoldt (FNV) Corrections ---")

        # We only keep v_Mg here because FNV requires perfectly matched grids.
        # Mg_O triggered an (80,) vs (96,) grid mismatch with the bulk.
        expected_fnv = {
            "v_Mg_+1": 1.11865072,
            "v_Mg_-2": -1.25915911,
        }
        assert expected_fnv.keys() <= defect_dict.keys()

        for defect_name, expected_energy in expected_fnv.items():
            defect_entry = defect_dict[defect_name]
            defect_entry.corrections.pop("kumagai_charge_correction", None)
            defect_entry.corrections_metadata.pop("kumagai_charge_correction", None)
            defect_entry.get_freysoldt_correction()
            calculated_energy = float(defect_entry.corrections["freysoldt_charge_correction"])

            print(f"{defect_name} (Charge {defect_entry.charge_state}): {calculated_energy:.4f} eV")

            np.testing.assert_allclose(
                calculated_energy,
                expected_energy,
                atol=1e-3,
                err_msg=f"FNV value mismatch for {defect_name}!",
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
            dielectric=np.diag([1e6, 1e6, 1.0]),
        )

        defect_dict = dp_gpaw.defect_dict

        # Expected corrections with a metallic in-plane response and vacuum-like
        # out-of-plane response. The large +4 value reflects q^2 scaling.
        expected_corrections = {
            "v_C_+1": 1.56463945,
            "C_i_C3v_+4": 27.89761687,
            "N_C_-2": -2.85529189,
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
