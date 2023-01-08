import os
import numpy as np
import unittest
from doped.pycdt.utils import parse_calculations
from pymatgen.core.structure import Structure


class DopedParsingTestCase(unittest.TestCase):
    def setUp(self):
        # get module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.BULK_DATA_DIR = os.path.join(self.EXAMPLE_DIR, "Bulk_Supercell/vasp_ncl")
        self.dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

    def test_vacancy_parsing_and_freysoldt(self):
        """Test parsing of Cd vacancy calculations and correct Freysoldt correction calculated"""
        parsed_vac_Cd_dict = {}

        for i in os.listdir(self.EXAMPLE_DIR):
            if "vac_1_Cd" in i:  # loop folders and parse those with "vac_1_Cd" in name
                defect_file_path = f"{self.EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=self.BULK_DATA_DIR,
                    dielectric=self.dielectric,
                    defect_charge=defect_charge,
                )
                bo = sdp.freysoldt_loader()
                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                parsed_vac_Cd_dict[
                    i
                ] = sdp.defect_entry  # Keep dictionary of parsed defect entries

        self.assertTrue(len(parsed_vac_Cd_dict) == 3)
        self.assertTrue(all(f"vac_1_Cd_{i}" in parsed_vac_Cd_dict for i in [0, -1, -2]))
        # Check that the correct Freysoldt correction is applied
        for name, energy, correction_dict in [
            (
                "vac_1_Cd_0",
                4.166,
                {"bandfilling_correction": -0.0, "bandedgeshifting_correction": 0.0},
            ),
            (
                "vac_1_Cd_-1",
                6.355,
                {
                    "charge_correction": 0.22517150393292082,
                    "bandfilling_correction": -0.0,
                    "bandedgeshifting_correction": 0.0,
                },
            ),
            (
                "vac_1_Cd_-2",
                8.398,
                {
                    "charge_correction": 0.7376460317828045,
                    "bandfilling_correction": -0.0,
                    "bandedgeshifting_correction": 0.0,
                },
            ),
        ]:
            self.assertAlmostEqual(parsed_vac_Cd_dict[name].energy, energy, places=3)
            for k in correction_dict:
                self.assertAlmostEqual(
                    parsed_vac_Cd_dict[name].corrections[k],
                    correction_dict[k],
                    places=3,
                )
            # assert auto-determined vacancy site is correct
            # should be: PeriodicSite: Cd (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
            if name == "vac_1_Cd_0":
                np.testing.assert_array_almost_equal(
                    parsed_vac_Cd_dict[name].site.frac_coords, [0.5, 0.5, 0.5]
                )
            else:
                np.testing.assert_array_almost_equal(
                    parsed_vac_Cd_dict[name].site.frac_coords, [0, 0, 0]
                )

    def test_interstitial_parsing_and_kumagai(self):
        """Test parsing of Te (split-)interstitial and Kumagai-Oba (eFNV) correction"""
        for i in os.listdir(self.EXAMPLE_DIR):
            if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                defect_file_path = f"{self.EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=self.BULK_DATA_DIR,
                    dielectric=self.dielectric,
                    defect_charge=defect_charge,
                )
                sdp.kumagai_loader()
                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                te_i_2_ent = sdp.defect_entry

        self.assertAlmostEqual(te_i_2_ent.energy, -6.221, places=3)
        self.assertAlmostEqual(te_i_2_ent.uncorrected_energy, -7.105, places=3)
        correction_dict = {
            "charge_correction": 0.8834518111049584,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for k in correction_dict:
            self.assertAlmostEqual(
                te_i_2_ent.corrections[k], correction_dict[k], places=3
            )
        # assert auto-determined interstital site is correct
        # should be: PeriodicSite: Te (12.2688, 12.2688, 8.9972) [0.9375, 0.9375, 0.6875]
        np.testing.assert_array_almost_equal(
            te_i_2_ent.site.frac_coords, [0.9375, 0.9375, 0.6875]
        )
        os.remove("bulk_voronoi_nodes.json")

    def test_substitution_parsing_and_kumagai(self):
        """Test parsing of Te_Cd_1 and Kumagai-Oba (eFNV) correction"""
        for i in os.listdir(self.EXAMPLE_DIR):
            if "as_1_Te" in i:  # loop folders and parse those with "Int_Te" in name
                defect_file_path = f"{self.EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=self.BULK_DATA_DIR,
                    dielectric=self.dielectric,
                    defect_charge=defect_charge,
                )
                sdp.kumagai_loader()
                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                te_cd_1_ent = sdp.defect_entry

        self.assertAlmostEqual(te_cd_1_ent.energy, -2.665996, places=3)
        self.assertAlmostEqual(te_cd_1_ent.uncorrected_energy, -2.906, places=3)
        correction_dict = {
            "charge_correction": 0.24005014473002428,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for k in correction_dict:
            self.assertAlmostEqual(
                te_cd_1_ent.corrections[k], correction_dict[k], places=3
            )
        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        np.testing.assert_array_almost_equal(
            te_cd_1_ent.site.frac_coords, [0.5000, 0.5000, 0.5000]
        )

    def test_extrinsic_interstitial_defect_ID(self):
        """Test parsing of extrinsic F in YTOS interstitial"""
        bulk_sc_structure = Structure.from_file(
            f"{self.EXAMPLE_DIR}/YTOS_Extrinsic_Fluorine_Interstitial/Bulk_POSCAR"
        )
        initial_defect_structure = Structure.from_file(
            f"{self.EXAMPLE_DIR}/YTOS_Extrinsic_Fluorine_Interstitial/Relaxed_CONTCAR"
        )
        (
            def_type,
            comp_diff,
        ) = parse_calculations.get_defect_type_and_composition_diff(
            bulk_sc_structure, initial_defect_structure
        )
        self.assertEqual(def_type, "interstitial")
        self.assertDictEqual(comp_diff, {"F": 1})
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = parse_calculations.get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
        self.assertEqual(bulk_site_idx, None)
        self.assertEqual(defect_site_idx, len(unrelaxed_defect_structure) - 1)

        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        np.testing.assert_array_almost_equal(
            unrelaxed_defect_structure[defect_site_idx].frac_coords,
            [0.00, 0.00, 0.48],
            decimal=2,  # approx match, not exact because relaxed bulk supercell
        )

    def test_extrinsic_substitution_defect_ID(self):
        """Test parsing of extrinsic U_on_Cd in CdTe"""
        bulk_sc_structure = Structure.from_file(
            f"{self.EXAMPLE_DIR}/CdTe_bulk_supercell_POSCAR"
        )
        initial_defect_structure = Structure.from_file(
            f"{self.EXAMPLE_DIR}/U_on_Cd_POSCAR"
        )
        (
            def_type,
            comp_diff,
        ) = parse_calculations.get_defect_type_and_composition_diff(
            bulk_sc_structure, initial_defect_structure
        )
        self.assertEqual(def_type, "substitution")
        self.assertDictEqual(comp_diff, {"Cd": -1, "U": 1})
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = parse_calculations.get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
        self.assertEqual(bulk_site_idx, 0)
        self.assertEqual(defect_site_idx, 63)  # last site in structure

        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        np.testing.assert_array_almost_equal(
            unrelaxed_defect_structure[defect_site_idx].frac_coords,
            [0.00, 0.00, 0.00],
            decimal=2,  # exact match because perfect supercell
        )


if __name__ == "__main__":
    unittest.main()
