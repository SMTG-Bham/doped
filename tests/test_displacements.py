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
from doped.utils.displacements import (
    calc_displacements_ellipsoid,
    calc_site_displacements,
    plot_displacements_ellipsoid,
)

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally
module_path = os.path.dirname(os.path.abspath(__file__))
STYLE = f"{module_path}/../doped/utils/displacement.mplstyle"


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
        self.v_Cd_0_defect_entry = DefectEntry.from_json(f"{data_dir}/v_Cd_defect_entry.json.gz")
        self.v_Cd_m1_defect_entry = DefectEntry.from_json(f"{data_dir}/v_Cd_m1_defect_entry.json.gz")
        self.F_i_m1_defect_entry = DefectEntry.from_json(f"{data_dir}/YTOS_Int_F_-1_defect_entry.json.gz")
        self.Te_Cd_1_defect_entry = DefectEntry.from_json(f"{data_dir}/Te_Cd_+1_defect_entry.json.gz")
        self.Te_i_1_defect_entry = DefectEntry.from_json(f"{data_dir}/Int_Te_3_1_defect_entry.json.gz")

    def test_calc_site_displacements(self):
        """
        Test ``calc_site_displacements()`` function.
        """
        for relaxed_distances in [False, True]:
            print(f"Testing calc_site_displacements with relaxed_distances={relaxed_distances}")
            defect_entry = self.v_Cd_0_defect_entry  # Neutral Cd vacancy
            disp_df = calc_site_displacements(defect_entry, relaxed_distances=relaxed_distances)
            for i, disp in [
                (0, [0.0572041, 0.00036486, -0.01794981]),
                (15, [0.11715445, -0.03659073, 0.01312027]),
            ]:
                np.allclose(disp_df["Displacement"].iloc[i], np.array(disp))
            # Check distance
            for i, dist in [
                (0, 0.0),
                (15, 6.54),
            ]:
                assert np.isclose(disp_df["Distance to defect"].iloc[i], dist, atol=2e-2)
            # Test displacements added to defect_entry:
            disp_metadata = defect_entry.calculation_metadata["site_displacements"]
            for i, disp in [
                (0, [0.0572041, 0.00036486, -0.01794981]),
                (15, [0.11715445, -0.03659073, 0.01312027]),
            ]:
                np.allclose(disp_metadata["displacements"][i], np.array(disp))
            # Test displacement of vacancy removed before adding to calculation_metadata
            assert len(disp_metadata["distances"]) == 63  # Cd Vacancy so 63 sites
            # Test relative displacements from defect
            disp_df = calc_site_displacements(
                defect_entry, relaxed_distances=relaxed_distances, relative_to_defect=True
            )
            disp_tuples = [
                (0, 0.0),
            ] + (
                [
                    (1, -0.1166),
                ]
                if relaxed_distances
                else [
                    (4, -0.0796),
                ]
            )
            for i, disp in disp_tuples:
                assert np.isclose(disp_df["Displacement wrt defect"].iloc[i], disp, atol=1e-3)

            # Test projection along Te dimer direction (1,1,0)
            disp_df = calc_site_displacements(
                defect_entry, relaxed_distances=relaxed_distances, vector_to_project_on=[1, 1, 0]
            )
            if relaxed_distances:
                disp_tuples = [
                    (32, 2.177851642, 0.980779911),  # index, distance, displacement
                    (33, 2.234858368, -0.892888144),
                ]
            else:
                disp_tuples = [
                    (32, 2.83337, 0.980779911),  # index, distance, displacement
                    (33, 2.83337, -0.892888144),
                ]
            for i, dist, disp in disp_tuples:
                assert np.isclose(disp_df["Displacement projected along vector"].iloc[i], disp, atol=1e-3)
                assert np.isclose(disp_df["Distance to defect"].iloc[i], dist, atol=1e-3)

            # Test projection along (-1,-1,-1) for V_Cd^-1
            defect_entry = self.v_Cd_m1_defect_entry
            disp_df = calc_site_displacements(
                defect_entry, relaxed_distances=relaxed_distances, vector_to_project_on=[-1, -1, -1]
            )
            indexes = (32, 33, 34, 35)  # Defect NNs
            distances = (2.5850237041739614, 2.5867590623267396, 2.5867621810347914, 3.0464198655727284)
            disp_parallel = (
                0.10857369429255698,
                0.10824441910793342,
                0.10824525022932621,
                0.2130514712472405,
            )
            disp_perpendicular = (
                0.22517568241969493,
                0.22345433515763177,
                0.22345075557153446,
                0.0002337424502264542,
            )
            for index, dist, disp_paral, disp_perp in zip(
                indexes, distances, disp_parallel, disp_perpendicular, strict=False
            ):
                if relaxed_distances:
                    assert np.isclose(disp_df["Distance to defect"].iloc[index], dist, atol=1e-3)
                else:  # all the same NN distance:
                    assert np.isclose(disp_df["Distance to defect"].iloc[index], 2.83337, atol=1e-3)
                assert np.isclose(
                    disp_df["Displacement projected along vector"].iloc[index], disp_paral, atol=1e-3
                )
                assert np.isclose(
                    disp_df["Displacement perpendicular to vector"].iloc[index], disp_perp, atol=1e-2
                )

            # Substitution:
            disp_df = calc_site_displacements(self.Te_Cd_1_defect_entry)
            for i, disp in [
                (0, [0.00820645, 0.00821417, -0.00815738]),
                (15, [-0.00639524, 0.00639969, -0.01407927]),
            ]:
                np.allclose(disp_df["Displacement"].iloc[i], np.array(disp), atol=1e-3)
            # Interstitial:
            disp_df = calc_site_displacements(self.Te_i_1_defect_entry)
            for i, disp in [
                (0, [-0.03931121, 0.01800569, 0.04547194]),
                (15, [-0.04850126, -0.01378455, 0.05439607]),
            ]:
                np.allclose(disp_df["Displacement"].iloc[i], np.array(disp), atol=1e-3)

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
        ellipsoid_center_V_Cd_0 = [7.13105322, 6.00301352, 7.01083535]
        ellipsoid_radii_V_Cd_0 = [4.93846557, 5.13530794, 7.24309775]
        ellipsoid_rotation_V_Cd_0 = [
            [-0.6268144, 0.6346784, 0.45198123],
            [0.71042757, 0.70376392, -0.0030035],
            [0.31999434, -0.31921729, 0.89202239],
        ]

        # Substitution:
        # These benchmarks are for the displacement ellipsoid of Te_Cd^+1 in CdTe at quantile=0.8:
        ellipsoid_center_Te_Cd_1 = [6.19574587, 6.19525728, 6.88797036]
        ellipsoid_radii_Te_Cd_1 = [3.15467871, 5.13765138, 5.14004549]
        ellipsoid_rotation_Te_Cd_1 = [
            [0.5772016, 0.57738077, 0.57746841],
            [0.26610123, 0.80155036, 0.53545043],
            [0.77202879, 0.15539778, 0.61629788],
        ]

        # Interstitial:
        # These benchmarks are for the displacement ellipsoid of Int_Te_3_1 in CdTe at quantile=0.8:
        ellipsoid_center_Te_i_1 = [6.09363088, 7.49090509, 6.25422689]
        ellipsoid_radii_Te_i_1 = [3.55290552, 5.58410512, 6.92612648]
        ellipsoid_rotation_Te_i_1 = [
            [0.44059141, 0.59640957, 0.6709507],
            [0.38012734, 0.80103898, 0.46242811],
            [0.81325421, 0.05130485, 0.57964248],
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
            for relaxed_distances in [False, True]:
                ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, anisotropy_df = (
                    calc_displacements_ellipsoid(entry, quantile=0.8, relaxed_distances=relaxed_distances)
                )
                if relaxed_distances:
                    assert np.allclose(ellipsoid_center, np.array(ellipsoid_center_benchmark), atol=1e-3)
                    assert np.allclose(ellipsoid_radii, np.array(ellipsoid_radii_benchmark), atol=1e-3)
                    assert np.allclose(
                        np.abs(ellipsoid_rotation), np.abs(ellipsoid_rotation_benchmark), atol=1e-3
                    )  # use absolute values, as vectors can be inverted

                else:
                    assert np.allclose(ellipsoid_center, np.array(ellipsoid_center_benchmark), atol=2.0)
                    assert np.allclose(ellipsoid_radii, np.array(ellipsoid_radii_benchmark), atol=2.0)

                assert anisotropy_df["Longest Radius"].to_numpy()[0] == ellipsoid_radii[2]
                assert (
                    anisotropy_df["2nd_Longest/Longest"].to_numpy()[0]
                    == ellipsoid_radii[1] / ellipsoid_radii[2]
                )
                assert (
                    anisotropy_df["3rd_Longest/Longest"].to_numpy()[0]
                    == ellipsoid_radii[0] / ellipsoid_radii[2]
                )

    @custom_mpl_image_compare(filename="v_Cd_0_disp_proj_plot.png", style=STYLE)
    def test_plot_site_displacements_proj(self):
        # Vacancy, displacement separated by direction:
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False
        )

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot.png", style=STYLE)
    def test_plot_site_displacements(self):
        # Vacancy, total displacement
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=False, use_plotly=False
        )

    @custom_mpl_image_compare(filename="v_Cd_0_disp_proj_plot_relaxed_dists.png", style=STYLE)
    def test_plot_site_displacements_proj_relaxed_dists(self):
        # Vacancy, displacement separated by direction:
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False, relaxed_distances=True
        )

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot_relaxed_dists.png", style=STYLE)
    def test_plot_site_displacements_relaxed_dists(self):
        # Vacancy, total displacement
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=False, use_plotly=False, relaxed_distances=True
        )

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements.png", style=STYLE)
    def test_plot_site_displacements_ytos(self):
        # Interstitial, total displacement
        return self.F_i_m1_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False
        )

    @custom_mpl_image_compare(filename="v_Cd_0_disp_ellipsoid_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_ellipsoid(self):
        return plot_displacements_ellipsoid(self.v_Cd_0_defect_entry, plot_ellipsoid=True)

    @custom_mpl_image_compare(filename="v_Cd_0_disp_anisotropy_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_anisotropy(self):
        return plot_displacements_ellipsoid(
            self.v_Cd_0_defect_entry, plot_ellipsoid=False, plot_anisotropy=True
        )

    @custom_mpl_image_compare(filename="Te_Cd_1_disp_ellipsoid_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_ellipsoid_Te_Cd_1(self):
        return plot_displacements_ellipsoid(self.Te_Cd_1_defect_entry, plot_anisotropy=True)[0]

    @custom_mpl_image_compare(filename="Te_Cd_1_disp_anisotropy_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_anisotropy_Te_Cd_1(self):
        return plot_displacements_ellipsoid(self.Te_Cd_1_defect_entry, plot_anisotropy=True)[1]

    @custom_mpl_image_compare(filename="F_i_-1_disp_ellipsoid_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_ellipsoid_F_i_m1(self):
        return plot_displacements_ellipsoid(self.F_i_m1_defect_entry, plot_ellipsoid=True)

    @custom_mpl_image_compare(filename="F_i_-1_disp_anisotropy_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_anisotropy_F_i_m1(self):
        return plot_displacements_ellipsoid(
            self.F_i_m1_defect_entry, plot_ellipsoid=False, plot_anisotropy=True
        )
