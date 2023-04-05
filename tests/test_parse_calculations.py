import os
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from pymatgen.core.structure import Structure

from doped.pycdt.utils import parse_calculations


class DopedParsingTestCase(unittest.TestCase):
    def setUp(self):
        # get module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CDTE_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.CDTE_BULK_DATA_DIR = os.path.join(
            self.CDTE_EXAMPLE_DIR, "Bulk_Supercell/vasp_ncl"
        )
        self.cdte_dielectric = np.array(
            [[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]]
        )  # CdTe

        self.ytos_dielectric = [
            [40.71948719643814, -9.282128210266565e-14, 1.26076160303219e-14],
            [-9.301652644020242e-14, 40.71948719776858, 4.149879443489052e-14],
            [5.311743673463141e-15, 2.041077680836527e-14, 25.237620491130023],
        ]
        # from legacy Materials Project

    def test_dielectric_initialisation(self):
        """
        Test that dielectric can be supplied as float or int or 3x1 array/list or 3x3 array/list
        """
        defect_file_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl"
        # test float
        sdp = parse_calculations.SingleDefectParser.from_paths(
            path_to_defect=defect_file_path,
            path_to_bulk=self.CDTE_BULK_DATA_DIR,
            dielectric=9.13,
            defect_charge=-2,
        )
        self.assertTrue(
            np.allclose(sdp.defect_entry.parameters["dielectric"], 9.13)
        )  # # float/int stay the same

        # test int
        sdp = parse_calculations.SingleDefectParser.from_paths(
            path_to_defect=defect_file_path,
            path_to_bulk=self.CDTE_BULK_DATA_DIR,
            dielectric=9,
            defect_charge=-2,
        )
        self.assertTrue(
            np.allclose(
                sdp.defect_entry.parameters["dielectric"],
                9,  # float/int stay the same
            )
        )

        # test 3x1 array
        sdp = parse_calculations.SingleDefectParser.from_paths(
            path_to_defect=defect_file_path,
            path_to_bulk=self.CDTE_BULK_DATA_DIR,
            dielectric=np.array([9.13, 9.13, 9.13]),
            defect_charge=-2,
        )
        self.assertTrue(
            np.allclose(sdp.defect_entry.parameters["dielectric"], self.cdte_dielectric)
        )

        # test 3x1 list
        sdp = parse_calculations.SingleDefectParser.from_paths(
            path_to_defect=defect_file_path,
            path_to_bulk=self.CDTE_BULK_DATA_DIR,
            dielectric=[9.13, 9.13, 9.13],
            defect_charge=-2,
        )
        self.assertTrue(
            np.allclose(sdp.defect_entry.parameters["dielectric"], self.cdte_dielectric)
        )

        # test 3x3 array
        sdp = parse_calculations.SingleDefectParser.from_paths(
            path_to_defect=defect_file_path,
            path_to_bulk=self.CDTE_BULK_DATA_DIR,
            dielectric=self.cdte_dielectric,
            defect_charge=-2,
        )
        self.assertTrue(
            np.allclose(sdp.defect_entry.parameters["dielectric"], self.cdte_dielectric)
        )

        # test 3x3 list
        sdp = parse_calculations.SingleDefectParser.from_paths(
            path_to_defect=defect_file_path,
            path_to_bulk=self.CDTE_BULK_DATA_DIR,
            dielectric=self.cdte_dielectric.tolist(),
            defect_charge=-2,
        )
        self.assertTrue(
            np.allclose(sdp.defect_entry.parameters["dielectric"], self.cdte_dielectric)
        )

    def test_vacancy_parsing_and_freysoldt(self):
        """Test parsing of Cd vacancy calculations and correct Freysoldt correction calculated"""
        parsed_vac_Cd_dict = {}

        for i in os.listdir(self.CDTE_EXAMPLE_DIR):
            if "vac_1_Cd" in i:  # loop folders and parse those with "vac_1_Cd" in name
                defect_file_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=self.CDTE_BULK_DATA_DIR,
                    dielectric=self.cdte_dielectric,
                    defect_charge=defect_charge,
                )
                sdp.freysoldt_loader()
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
            for correction_name, correction_energy in correction_dict.items():
                self.assertAlmostEqual(
                    parsed_vac_Cd_dict[name].corrections[correction_name],
                    correction_energy,
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
        with patch("builtins.print") as mock_print:
            for i in os.listdir(self.CDTE_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_file_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                    defect_charge = int(i[-2:].replace("_", ""))
                    # parse with no transformation.json:
                    sdp = parse_calculations.SingleDefectParser.from_paths(
                        path_to_defect=defect_file_path,
                        path_to_bulk=self.CDTE_BULK_DATA_DIR,
                        dielectric=self.cdte_dielectric,
                        defect_charge=defect_charge,
                    )
                    sdp.kumagai_loader()
                    sdp.get_stdrd_metadata()
                    sdp.get_bulk_gap_data()
                    sdp.run_compatibility()
                    te_i_2_ent = sdp.defect_entry

        mock_print.assert_called_once_with(
            "Saving parsed Voronoi sites (for interstitial site-matching) "
            "to bulk_voronoi_sites.json to speed up future parsing."
        )

        self.assertAlmostEqual(te_i_2_ent.energy, -6.221, places=3)
        self.assertAlmostEqual(te_i_2_ent.uncorrected_energy, -7.105, places=3)
        correction_dict = {
            "charge_correction": 0.8834518111049584,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                te_i_2_ent.corrections[correction_name], correction_energy, places=3
            )

        # assert auto-determined interstitial site is correct
        # should be: PeriodicSite: Te (12.2688, 12.2688, 8.9972) [0.9375, 0.9375, 0.6875]
        np.testing.assert_array_almost_equal(
            te_i_2_ent.site.frac_coords, [0.9375, 0.9375, 0.6875]
        )

        # run again to check parsing of previous Voronoi sites
        with patch("builtins.print") as mock_print:
            for i in os.listdir(self.CDTE_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_file_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                    defect_charge = int(i[-2:].replace("_", ""))
                    # parse with no transformation.json:
                    sdp = parse_calculations.SingleDefectParser.from_paths(
                        path_to_defect=defect_file_path,
                        path_to_bulk=self.CDTE_BULK_DATA_DIR,
                        dielectric=self.cdte_dielectric,
                        defect_charge=defect_charge,
                    )
                    sdp.kumagai_loader()
                    sdp.get_stdrd_metadata()
                    sdp.get_bulk_gap_data()
                    sdp.run_compatibility()
                    te_i_2_ent = sdp.defect_entry

        mock_print.assert_not_called()
        os.remove("bulk_voronoi_nodes.json")

    def test_substitution_parsing_and_kumagai(self):
        """Test parsing of Te_Cd_1 and Kumagai-Oba (eFNV) correction"""
        for i in os.listdir(self.CDTE_EXAMPLE_DIR):
            if "as_1_Te" in i:  # loop folders and parse those with "as_1_Te" in name
                defect_file_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=self.CDTE_BULK_DATA_DIR,
                    dielectric=self.cdte_dielectric,
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
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                te_cd_1_ent.corrections[correction_name], correction_energy, places=3
            )
        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        np.testing.assert_array_almost_equal(
            te_cd_1_ent.site.frac_coords, [0.5000, 0.5000, 0.5000]
        )

    def test_extrinsic_interstitial_defect_ID(self):
        """Test parsing of extrinsic F in YTOS interstitial"""
        bulk_sc_structure = Structure.from_file(f"{self.EXAMPLE_DIR}/YTOS/Bulk/POSCAR")
        initial_defect_structure = Structure.from_file(
            f"{self.EXAMPLE_DIR}/YTOS/Int_F_-1/Relaxed_CONTCAR"
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

        # assert auto-determined interstitial site is correct
        print(unrelaxed_defect_structure[defect_site_idx].frac_coords)
        self.assertAlmostEqual(
            unrelaxed_defect_structure[
                defect_site_idx
            ].distance_and_image_from_frac_coords(
                [-0.0005726049122470, -0.0001544430438804, 0.47800736578014720]
            )[
                0
            ],
            0.0,
            places=2,
        )  # approx match, not exact because relaxed bulk supercell

    def test_extrinsic_substitution_defect_ID(self):
        """Test parsing of extrinsic U_on_Cd in CdTe"""
        bulk_sc_structure = Structure.from_file(
            f"{self.CDTE_EXAMPLE_DIR}/CdTe_bulk_supercell_POSCAR"
        )
        initial_defect_structure = Structure.from_file(
            f"{self.CDTE_EXAMPLE_DIR}/U_on_Cd_POSCAR"
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
        np.testing.assert_array_almost_equal(
            unrelaxed_defect_structure[defect_site_idx].frac_coords,
            [0.00, 0.00, 0.00],
            decimal=2,  # exact match because perfect supercell
        )

    def test_extrinsic_interstitial_parsing_and_kumagai(self):
        """Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV) correction"""

        for i in os.listdir(self.EXAMPLE_DIR):
            if "YTOS" in i:
                defect_file_path = f"{self.EXAMPLE_DIR}/{i}/Int_F_-1/"
                defect_charge = -1
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=f"{self.EXAMPLE_DIR}/{i}/Bulk/",
                    dielectric=self.ytos_dielectric,
                    defect_charge=defect_charge,
                )
                sdp.kumagai_loader()
                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                int_F_minus1_ent = sdp.defect_entry

        self.assertAlmostEqual(int_F_minus1_ent.energy, 0.767, places=3)
        self.assertAlmostEqual(int_F_minus1_ent.uncorrected_energy, 0.7515, places=3)
        correction_dict = {
            "charge_correction": 0.0155169495708003,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                int_F_minus1_ent.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            int_F_minus1_ent.site.distance_and_image_from_frac_coords([0, 0, 0.4847])[
                0
            ],
            0.0,
            places=2,
        )  # approx match, not exact because relaxed bulk supercell

        os.remove("bulk_voronoi_nodes.json")

    def test_extrinsic_substitution_parsing_and_freysoldt_and_kumagai(self):
        """
        Test parsing of extrinsic F-on-O substitution in YTOS,
        w/Kumagai-Oba (eFNV) and Freysoldt (FNV) corrections
        """

        # first using Freysoldt (FNV) correction
        for i in os.listdir(self.EXAMPLE_DIR):
            if "YTOS" in i:
                defect_file_path = f"{self.EXAMPLE_DIR}/{i}/F_O_1/"
                defect_charge = +1
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=f"{self.EXAMPLE_DIR}/{i}/Bulk/",
                    dielectric=self.ytos_dielectric,
                    defect_charge=defect_charge,
                )
                sdp.freysoldt_loader()
                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                F_O_1_ent = sdp.defect_entry

        self.assertAlmostEqual(F_O_1_ent.energy, 0.0308, places=3)
        self.assertAlmostEqual(F_O_1_ent.uncorrected_energy, -0.0852, places=3)
        correction_dict = {
            "charge_correction": 0.11602321245859282,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                F_O_1_ent.corrections[correction_name], correction_energy, places=3
            )
        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            F_O_1_ent.site.distance_and_image_from_frac_coords([0, 0, 0])[0],
            0.0,
            places=2,
        )

        # now using Kumagai-Oba (eFNV) correction
        for i in os.listdir(self.EXAMPLE_DIR):
            if "YTOS" in i:
                defect_file_path = f"{self.EXAMPLE_DIR}/{i}/F_O_1/"
                defect_charge = +1
                # parse with no transformation.json:
                sdp = parse_calculations.SingleDefectParser.from_paths(
                    path_to_defect=defect_file_path,
                    path_to_bulk=f"{self.EXAMPLE_DIR}/{i}/Bulk/",
                    dielectric=self.ytos_dielectric,
                    defect_charge=defect_charge,
                )
                sdp.kumagai_loader()
                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                F_O_1_ent = sdp.defect_entry

        self.assertAlmostEqual(F_O_1_ent.energy, -0.0031, places=3)
        self.assertAlmostEqual(F_O_1_ent.uncorrected_energy, -0.0852, places=3)
        correction_dict = {
            "charge_correction": 0.08214,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                F_O_1_ent.corrections[correction_name], correction_energy, places=3
            )
        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            F_O_1_ent.site.distance_and_image_from_frac_coords([0, 0, 0])[0],
            0.0,
            places=2,
        )

    def test_voronoi_structure_mismatch_and_reparse(self):
        """
        Test that a mismatch in bulk_supercell structure from previously parsed
        Voronoi nodes json file with current defect bulk supercell is detected and
        re-parsed
        """
        with patch("builtins.print") as mock_print:
            for i in os.listdir(self.CDTE_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_file_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                    defect_charge = int(i[-2:].replace("_", ""))
                    # parse with no transformation.json:
                    sdp = parse_calculations.SingleDefectParser.from_paths(
                        path_to_defect=defect_file_path,
                        path_to_bulk=self.CDTE_BULK_DATA_DIR,
                        dielectric=self.cdte_dielectric,
                        defect_charge=defect_charge,
                    )

        mock_print.assert_called_once_with(
            "Saving parsed Voronoi sites (for interstitial site-matching) "
            "to bulk_voronoi_sites.json to speed up future parsing."
        )

        with warnings.catch_warnings(record=True) as w:
            for i in os.listdir(self.EXAMPLE_DIR):
                if "YTOS" in i:
                    defect_file_path = f"{self.EXAMPLE_DIR}/{i}/Int_F_-1/"
                    defect_charge = -1
                    # parse with no transformation.json:
                    sdp = parse_calculations.SingleDefectParser.from_paths(
                        path_to_defect=defect_file_path,
                        path_to_bulk=f"{self.EXAMPLE_DIR}/{i}/Bulk/",
                        dielectric=self.ytos_dielectric,
                        defect_charge=defect_charge,
                    )

        warning_message = (
            "Previous bulk_voronoi_nodes.json detected, but does not "
            "match current bulk supercell. Recalculating Voronoi nodes."
        )
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        self.assertEqual(len(user_warnings), 1)
        self.assertIn(warning_message, str(user_warnings[0].message))
        os.remove("bulk_voronoi_nodes.json")


if __name__ == "__main__":
    unittest.main()
