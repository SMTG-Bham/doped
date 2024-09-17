"""
Tests for the `doped.utils.displacements` module.
"""

import os
import shutil
import unittest

import matplotlib as mpl
import numpy as np
import pytest
from test_thermodynamics import custom_mpl_image_compare, data_dir

from doped.core import DefectEntry
from doped.utils.displacements import calc_displacements_ellipsoid, calc_site_displacements

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


def if_present_rm(path):
    """
    Remove file or directory if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class DefectDisplacementsTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.v_Cd_0_defect_entry = DefectEntry.from_json(f"{data_dir}/v_Cd_defect_entry.json.gz")
        self.v_Cd_m1_defect_entry = DefectEntry.from_json(f"{data_dir}/v_Cd_m1_defect_entry.json.gz")
        self.F_i_m1_defect_entry = DefectEntry.from_json(f"{data_dir}/YTOS_Int_F_-1_defect_entry.json.gz")
        self.Te_Cd_1_defect_entry = DefectEntry.from_json(f"{data_dir}/Te_Cd_+1_defect_entry.json.gz")
        self.Te_i_1_defect_entry = DefectEntry.from_json(f"{data_dir}/Int_Te_3_1_defect_entry.json.gz")

    def test_calc_site_displacements(self):
        """
        Test calc_site_displacements() function.
        """
        defect_entry = self.v_Cd_0_defect_entry  # Neutral Cd vacancy
        disp_dict = calc_site_displacements(defect_entry)
        for i, disp in [
            (0, [0.0572041, 0.00036486, -0.01794981]),
            (15, [0.11715445, -0.03659073, 0.01312027]),
        ]:
            np.allclose(disp_dict["Displacement"][i], np.array(disp))
        # Check distance
        for i, dist in [
            (0, 0.0),
            (15, 6.543643778883931),
        ]:
            assert np.isclose(disp_dict["Distance to defect"][i], dist)
        # Test displacements added to defect_entry:
        disp_dict_metadata = defect_entry.calculation_metadata["site_displacements"]
        for i, disp in [
            (0, [0.0572041, 0.00036486, -0.01794981]),
            (15, [0.11715445, -0.03659073, 0.01312027]),
        ]:
            np.allclose(disp_dict_metadata["displacements"][i], np.array(disp))
        # Test displacement of vacancy removed before adding to calculation_metadata
        assert len(disp_dict_metadata["distances"]) == 63  # Cd Vacancy so 63 sites
        # Test relative displacements from defect
        disp_dict = calc_site_displacements(defect_entry, relative_to_defect=True)
        for i, disp in [
            (0, 0.0),
            (1, -0.1165912674943227),
        ]:
            assert np.isclose(disp_dict["Displacement wrt defect"][i], disp)
        # Test projection along Te dimer direction (1,1,0)
        disp_dict = calc_site_displacements(defect_entry, vector_to_project_on=[1, 1, 0])
        for i, dist, disp in [
            (32, 2.177851642, 0.980779911),  # index, distance, displacement
            (33, 2.234858368, -0.892888144),
        ]:
            assert np.isclose(disp_dict["Displacement projected along vector"][i], disp)
            assert np.isclose(disp_dict["Distance to defect"][i], dist)
        # Test projection along (-1,-1,-1) for V_Cd^-1
        defect_entry = self.v_Cd_m1_defect_entry
        disp_dict = calc_site_displacements(defect_entry, vector_to_project_on=[-1, -1, -1])
        indexes = (32, 33, 34, 35)  # Defect NNs
        distances = (2.5850237041739614, 2.5867590623267396, 2.5867621810347914, 3.0464198655727284)
        disp_parallel = (0.10857369429255698, 0.10824441910793342, 0.10824525022932621, 0.2130514712472405)
        disp_perpendicular = (
            0.22517568241969493,
            0.22345433515763177,
            0.22345075557153446,
            0.0002337424502264542,
        )
        for index, dist, disp_paral, disp_perp in zip(
            indexes, distances, disp_parallel, disp_perpendicular
        ):
            assert np.isclose(disp_dict["Distance to defect"][index], dist)
            assert np.isclose(disp_dict["Displacement projected along vector"][index], disp_paral)
            assert np.isclose(disp_dict["Displacement perpendicular to vector"][index], disp_perp)

        # Substitution:
        disp_dict = calc_site_displacements(self.Te_Cd_1_defect_entry)
        for i, disp in [
            (0, [0.00820645, 0.00821417, -0.00815738]),
            (15, [-0.00639524, 0.00639969, -0.01407927]),
        ]:
            np.allclose(disp_dict["Displacement"][i], np.array(disp))
        # Interstitial:
        disp_dict = calc_site_displacements(self.Te_i_1_defect_entry)
        for i, disp in [
            (0, [-0.03931121, 0.01800569, 0.04547194]),
            (15, [-0.04850126, -0.01378455, 0.05439607]),
        ]:
            np.allclose(disp_dict["Displacement"][i], np.array(disp))

    def test_plot_site_displacements_error(self):
        # Check ValueError raised if user sets both separated_by_direction and vector_to_project_on
        defect_entry = self.v_Cd_0_defect_entry
        with pytest.raises(ValueError):
            defect_entry.plot_site_displacements(
                separated_by_direction=True, vector_to_project_on=[0, 0, 1]
            )
        # Same but if user sets separated_by_direction and relative_to_defect
        with pytest.raises(ValueError):
            defect_entry.plot_site_displacements(separated_by_direction=True, relative_to_defect=True)
        # Same but if user sets vector_to_project_on and relative_to_defect
        with pytest.raises(ValueError):
            defect_entry.plot_site_displacements(vector_to_project_on=[0, 0, 1], relative_to_defect=True)

    def test_calc_displacements_ellipsoid(self):
        # Vacancy:
        # These benchmarks are for the displacement ellipsoid of V_Cd^0 in CdTe at quantile=0.8:
        ellipsoid_center_V_Cd_0 = [7.0425819, 6.02166313, 6.95936032]
        ellipsoid_radii_V_Cd_0 = [4.87240561, 5.05708172, 7.17470889]
        ellipsoid_rotation_V_Cd_0 = [
            [-0.60311542, 0.65630014, 0.45334526],
            [-0.72881472, -0.68438333, 0.02117917],
            [0.32416183, -0.31763121, 0.89108334],
        ]

        # Substitution:
        # These benchmarks are for the displacement ellipsoid of Te_Cd^+1 in CdTe at quantile=0.8:
        ellipsoid_center_Te_Cd_1 = [6.18957603, 6.17932465, 6.84044422]
        ellipsoid_radii_Te_Cd_1 = [3.10352118, 5.06896107, 5.11217386]
        ellipsoid_rotation_Te_Cd_1 = [
            [-0.57487044, -0.57786455, 0.57930695],
            [0.23007278, -0.79357926, -0.56329254],
            [0.78523276, -0.19053747, 0.58915616],
        ]

        # Interstitial:
        # These benchmarks are for the displacement ellipsoid of Int_Te_3_1 in CdTe at quantile=0.8:
        ellipsoid_center_Te_i_1 = [6.09696696, 7.48234777, 6.19752711]
        ellipsoid_radii_Te_i_1 = [3.48790624, 5.56101187, 6.8367233]
        ellipsoid_rotation_Te_i_1 = [
            [-0.43707588, -0.59878091, -0.67113791],
            [-0.38107693, 0.79918498, -0.46484808],
            [0.81470549, 0.05258129, -0.57748607],
        ]

        for entry, ellipsoid_center_benchmark, ellipsoid_radii_benchmark, ellipsoid_rotation_benchmark in [
            (
                self.v_Cd_0_defect_entry,
                ellipsoid_center_V_Cd_0,
                ellipsoid_radii_V_Cd_0,
                ellipsoid_rotation_V_Cd_0,
            ),
            (
                self.Te_Cd_1_defect_entry,
                ellipsoid_center_Te_Cd_1,
                ellipsoid_radii_Te_Cd_1,
                ellipsoid_rotation_Te_Cd_1,
            ),
            (
                self.Te_i_1_defect_entry,
                ellipsoid_center_Te_i_1,
                ellipsoid_radii_Te_i_1,
                ellipsoid_rotation_Te_i_1,
            ),
        ]:
            print("Testing displacement ellipsoid for", entry.name)
            ellipsoid_center, ellipsoid_radii, ellipsoid_rotation = calc_displacements_ellipsoid(
                entry, quantile=0.8
            )
            np.allclose(ellipsoid_center, np.array(ellipsoid_center_benchmark))
            np.allclose(ellipsoid_radii, np.array(ellipsoid_radii_benchmark))
            np.allclose(ellipsoid_rotation, np.array(ellipsoid_rotation_benchmark))

    @custom_mpl_image_compare(filename="v_Cd_0_disp_proj_plot.png")
    def test_plot_site_displacements_proj(self):
        # Vacancy, displacement separated by direction:
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False
        )

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot.png")
    def test_plot_site_displacements(self):
        # Vacancy, total displacement
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=False, use_plotly=False
        )

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements.png")
    def test_plot_site_displacements_ytos(self):
        # Interstitial, total displacement
        return self.F_i_m1_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False
        )
