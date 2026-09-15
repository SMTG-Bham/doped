"""
Tests for the `doped.utils.displacements` module.
"""

import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from pymatgen.core.structure import Structure
from test_utils import STYLE, custom_mpl_image_compare, data_dir

from doped.core import DefectEntry
from doped.utils.displacements import (
    calc_displacements_ellipsoid,
    calc_site_displacements,
    plot_displacements_ellipsoid,
    plot_site_displacements,
)
from doped.utils.symmetry import remove_translation_drift, translate_structure

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


class DefectDisplacementsTestCase(unittest.TestCase):
    def setUp(self):
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
            disp_vec_tuples = [
                (0, [0.0, 0.0, 0.0]),  # the vacancy site itself; zero displacement by construction
            ] + (  # rows are sorted by distance to defect, so index 15 is a different site in each case:
                [(15, [-0.00113, 0.00022, -0.14417])]
                if relaxed_distances
                else [(15, [-0.00116, 0.00114, -0.14415])]
            )
            for i, disp in disp_vec_tuples:
                assert np.allclose(disp_df["Displacement vector"].iloc[i], np.array(disp), atol=1e-5)
            # Check distance
            for i, dist in [
                (0, 0.0),
                (15, 6.54),
            ]:
                assert np.isclose(disp_df["Distance to defect"].iloc[i], dist, atol=2e-2)
            # Test displacements added to defect_entry:
            disp_metadata = defect_entry.calculation_metadata["site_displacements"]
            assert np.allclose(  # the vacancy site is row 0; sorted by distance, checked as 0.0 above
                np.array(disp_metadata["displacements"]),
                np.stack(list(disp_df["Displacement vector"]))[1:],
            )
            assert np.allclose(
                np.array(disp_metadata["distances"]), disp_df["Distance to defect"].to_numpy()[1:]
            )
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
                    (1, -0.0768),
                ]
                if relaxed_distances
                else [
                    (4, -0.0783),
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
                    (32, 2.203030, 0.939950),  # index, distance, displacement
                    (33, 2.209070, -0.933720),
                ]
            else:
                disp_tuples = [
                    (32, 2.83337, 0.939950),  # index, distance, displacement
                    (33, 2.83337, -0.933720),
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
                    disp_df["Absolute displacement perpendicular to vector"].iloc[index],
                    disp_perp,
                    atol=1e-2,
                )

        # substitution and interstitial cases; with the default ``relaxed_distances``, so outside the loop
        # above (which only affects the tabulated distances, and so the row ordering):
        for defect_entry, disp_vec_tuples in [
            (
                self.Te_Cd_1_defect_entry,
                [(0, [-0.0624, -0.06237, 0.01063]), (15, [0.0036, 0.0182, -0.00363])],
            ),
            (
                self.Te_i_1_defect_entry,
                [(0, [-0.23097, -0.32803, 0.23296]), (15, [-0.05906, 0.0545, -0.0305])],
            ),
        ]:
            disp_df = calc_site_displacements(defect_entry)
            for i, disp in disp_vec_tuples:
                assert np.allclose(disp_df["Displacement vector"].iloc[i], np.array(disp), atol=1e-4)

    def test_remove_translation_drift(self):
        """
        Test ``remove_translation_drift``, which should recover a rigid
        translation of the defect supercell exactly for an unrelaxed defect
        (where the mean host atom displacement `is` the translation), including
        for defects with contested bulk sites -- an antisite (whose substituent
        has no bulk site of its own, so claims that of a host atom) and a split
        interstitial (whose two halves claim the same bulk site) -- which must
        be excluded from the drift estimate.

        Also that it gives the same result from translated and untranslated
        inputs for real (relaxed) entries.
        """
        bulk_supercell = self.v_Cd_0_defect_entry.bulk_supercell
        cd_index, te_index = (bulk_supercell.indices_from_symbol(symbol)[0] for symbol in ("Cd", "Te"))
        cd_frac_coords, te_frac_coords = (bulk_supercell[i].frac_coords for i in (cd_index, te_index))
        shift = np.array([0.004, -0.006, 0.009])  # fractional; ~0.05-0.12 Å in this 13 Å supercell

        def _max_min_image_diff(frac_coords_a, frac_coords_b):
            diff = np.asarray(frac_coords_a) - np.asarray(frac_coords_b)
            return np.abs(diff - np.round(diff)).max()

        vacancy = bulk_supercell.copy()
        vacancy.remove_sites([cd_index])  # first Cd index
        antisite = bulk_supercell.copy()
        antisite.replace(cd_index, "Te")
        split_interstitial = bulk_supercell.copy()
        split_interstitial.remove_sites([te_index])
        for sign in (1, -1):  # Te-Te dumbbell centred on the bulk Te site, ~1.2 Å long
            split_interstitial.append("Te", te_frac_coords + sign * np.array([0.045, 0, 0]))

        for defect_supercell, defect_frac_coords in [
            (vacancy, cd_frac_coords),
            (antisite, cd_frac_coords),
            (split_interstitial, te_frac_coords),
        ]:
            unshifted_supercell, unshifted_frac_coords = remove_translation_drift(
                translate_structure(defect_supercell, shift, frac_coords=True),
                bulk_supercell,
                defect_frac_coords + shift,
            )
            assert (
                _max_min_image_diff(unshifted_supercell.frac_coords, defect_supercell.frac_coords) < 1e-8
            )
            # the returned coords always follow the atoms; a _vacancy_ site is defined in the bulk frame
            # instead, so its caller keeps its own input coords (see ``get_defect_in_supercell``):
            assert _max_min_image_diff(unshifted_frac_coords, defect_frac_coords) < 1e-8

        # supercells with no species in common give an informative error, not an ``IndexError``:
        with pytest.raises(ValueError, match="No host atoms could be matched"):
            remove_translation_drift(
                Structure(bulk_supercell.lattice, ["Ar"], [[0, 0, 0]]), bulk_supercell, [0, 0, 0]
            )

        for defect_entry in [
            self.v_Cd_0_defect_entry,
            self.Te_Cd_1_defect_entry,
            self.Te_i_1_defect_entry,
        ]:
            frac_coords = np.array(defect_entry.sc_defect_frac_coords)
            reference = remove_translation_drift(
                defect_entry.defect_supercell, defect_entry.bulk_supercell, frac_coords
            )
            shifted = remove_translation_drift(
                translate_structure(defect_entry.defect_supercell, shift, frac_coords=True),
                defect_entry.bulk_supercell,
                frac_coords + shift,
            )
            assert _max_min_image_diff(shifted[0].frac_coords, reference[0].frac_coords) < 1e-8
            assert _max_min_image_diff(shifted[1], reference[1]) < 1e-8

    def test_plot_site_displacements_error(self):
        # Check ValueError raised if user sets both separated_by_direction and vector_to_project_on
        defect_entry = self.v_Cd_0_defect_entry
        with pytest.raises(ValueError):
            defect_entry.plot_site_displacements(
                separated_by_direction=True, vector_to_project_on=[0, 0, 1]
            )
        # test now fine if user sets separated_by_direction and relative_to_defect (latter ignored):
        defect_entry.plot_site_displacements(separated_by_direction=True, relative_to_defect=True)
        # test now fine if user sets vector_to_project_on and relative_to_defect (latter ignored):
        defect_entry.plot_site_displacements(vector_to_project_on=[0, 0, 1], relative_to_defect=True)

        # wrong-length ax sequences should raise ValueError:
        _, (ax1, ax2) = plt.subplots(1, 2)
        with pytest.raises(ValueError):  # separated_by_direction needs 3 axes, not 2
            defect_entry.plot_site_displacements(separated_by_direction=True, ax=[ax1, ax2])
        _, ax1 = plt.subplots(1, 1)
        with pytest.raises(ValueError):  # vector_to_project_on needs 2 axes, not 1
            defect_entry.plot_site_displacements(vector_to_project_on=[0, 0, 1], ax=[ax1])
        plt.close("all")

    def test_calc_displacements_ellipsoid(self):
        # Vacancy:
        # These benchmarks are for the displacement ellipsoid of V_Cd^0 in CdTe at quantile=0.8:
        ellipsoid_center_V_Cd_0 = [6.92183865, 6.16328614, 5.12604008]
        ellipsoid_radii_V_Cd_0 = [3.88736226, 4.79169454, 5.65805019]
        ellipsoid_rotation_V_Cd_0 = [
            [-0.63676437, 0.63729803, 0.43403037],
            [0.30650482, -0.30730649, 0.90089817],
            [0.70752098, 0.70669226, 0.0003469],
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
        ellipsoid_center_Te_i_1 = [7.48324291, 6.98981838, 5.60624404]
        ellipsoid_radii_Te_i_1 = [2.98045949, 4.35544932, 8.89576523]
        ellipsoid_rotation_Te_i_1 = [
            [0.70693397, -0.0003162, 0.70727948],
            [2.9e-06, 0.9999999, 0.00044417],
            [0.70727955, 0.00031195, -0.7069339],
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

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot.png", style=STYLE)
    def test_plot_site_displacements(self):
        return self.v_Cd_0_defect_entry.plot_site_displacements(use_plotly=False)

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot_total_disp.png", style=STYLE)
    def test_plot_site_displacements_total_disp(self):  # Vacancy, total displacement
        return self.v_Cd_0_defect_entry.plot_site_displacements(use_plotly=False, relative_to_defect=False)

    @custom_mpl_image_compare(filename="v_Cd_0_disp_proj_plot.png", style=STYLE)
    def test_plot_site_displacements_proj(self):
        # Vacancy, displacement separated by direction:
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False
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

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot_relaxed_dists.png", style=STYLE)
    def test_plot_site_displacements_relaxed_dists_relative_to_defect(self):
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=False, use_plotly=False, relaxed_distances=True, relative_to_defect=True
        )

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements_separated.png", style=STYLE)
    def test_plot_site_displacements_ytos(self):  # Interstitial, total displacement
        return self.F_i_m1_defect_entry.plot_site_displacements(
            separated_by_direction=True, use_plotly=False
        )

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements.png", style=STYLE)
    def test_plot_site_displacements_ytos_relative_to_defect(self):
        return self.F_i_m1_defect_entry.plot_site_displacements(
            use_plotly=False,
            relative_to_defect=True,  # default
        )

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements_total_disp.png", style=STYLE)
    def test_plot_site_displacements_ytos_relative_to_defect_total_disp(self):
        return self.F_i_m1_defect_entry.plot_site_displacements(
            use_plotly=False,
            relative_to_defect=False,  # total displacements
        )

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements_along_111.png", style=STYLE)
    def test_plot_site_displacements_ytos_vector_to_project_on(self):
        # test using direct function here:
        return plot_site_displacements(self.F_i_m1_defect_entry, vector_to_project_on=[1, 1, 1])

    @custom_mpl_image_compare(filename="v_Cd_0_disp_ellipsoid_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid(self):
        return plot_displacements_ellipsoid(self.v_Cd_0_defect_entry, plot_ellipsoid=True)

    @custom_mpl_image_compare(filename="v_Cd_0_disp_anisotropy_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_anisotropy(self):
        return plot_displacements_ellipsoid(
            self.v_Cd_0_defect_entry, plot_ellipsoid=False, plot_anisotropy=True
        )

    @custom_mpl_image_compare(filename="Te_Cd_1_disp_ellipsoid_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_Te_Cd_1(self):
        return plot_displacements_ellipsoid(self.Te_Cd_1_defect_entry, plot_anisotropy=True)[0]

    @custom_mpl_image_compare(filename="Te_Cd_1_disp_anisotropy_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_Te_Cd_1_anisotropy(self):
        return plot_displacements_ellipsoid(self.Te_Cd_1_defect_entry, plot_anisotropy=True)[1]

    @custom_mpl_image_compare(filename="F_i_-1_disp_ellipsoid_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_F_i_m1(self):
        return plot_displacements_ellipsoid(self.F_i_m1_defect_entry, plot_ellipsoid=True)

    @custom_mpl_image_compare(filename="F_i_-1_disp_anisotropy_plot.png", style=STYLE)
    def test_plot_displacements_ellipsoid_F_i_m1_anisotropy(self):
        return plot_displacements_ellipsoid(
            self.F_i_m1_defect_entry, plot_ellipsoid=False, plot_anisotropy=True
        )

    @custom_mpl_image_compare(filename="v_Cd_0_and_m1_disp_on_provided_axes.png", style=STYLE)
    def test_plot_site_displacements_provided_ax(self):
        """
        Test that ``plot_site_displacements`` plots onto a user-provided
        ``ax``, enabling side-by-side comparison of two defect entries on a
        single figure.
        """
        styled_fig_size = plt.rcParams["figure.figsize"]
        fig, axes = plt.subplots(1, 2, figsize=(2 * styled_fig_size[0], styled_fig_size[1]), sharey=True)

        self.v_Cd_0_defect_entry.plot_site_displacements(
            separated_by_direction=False, use_plotly=False, ax=axes[0], style_file=STYLE
        )
        self.v_Cd_m1_defect_entry.plot_site_displacements(
            separated_by_direction=False, use_plotly=False, ax=axes[1], style_file=STYLE
        )
        axes[0].set_title("V$_{Cd}^{0}$")
        axes[1].set_title("V$_{Cd}^{-1}$")
        fig.subplots_adjust(wspace=0.15)
        return fig

    @custom_mpl_image_compare(filename="v_Cd_0_disp_vector_to_project_on.png", style=STYLE)
    def test_plot_site_displacements_vector_to_project_on(self):
        """
        Test ``plot_site_displacements`` with ``vector_to_project_on``, which
        produces a 2-panel mpl figure (parallel + perpendicular).
        """
        return self.v_Cd_0_defect_entry.plot_site_displacements(
            vector_to_project_on=[0, 0, 1], style_file=STYLE
        )

    def test_plot_site_displacements_provided_plotly_fig(self):
        """
        Test that ``plot_site_displacements`` adds traces to a user-provided
        plotly ``fig``, for each plot mode.
        """
        from plotly.subplots import make_subplots

        # single panel: one trace per species (Cd, Te)
        existing_fig = go.Figure()
        result = self.v_Cd_0_defect_entry.plot_site_displacements(use_plotly=True, fig=existing_fig)
        assert result is existing_fig
        assert len(result.data) == 2  # one trace per species
        assert len({t.xaxis for t in result.data}) == 1  # all on the same (single) panel

        # separated_by_direction: 3 scatter traces (one per axis, no legend) + 2 legend-only traces
        multi_fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True)
        result2 = self.v_Cd_0_defect_entry.plot_site_displacements(
            use_plotly=True, separated_by_direction=True, fig=multi_fig
        )
        assert result2 is multi_fig
        assert len(result2.data) == 6  # 2x species (Cd, Te) for each x/y/z subplot
        scatter_xaxes = {t.xaxis for t in result2.data}
        assert scatter_xaxes == {"x", "x2", "x3"}  # one scatter per subplot
        scatter_xaxes_no_showlegend = {t.xaxis for t in result2.data if not t.showlegend}
        assert scatter_xaxes_no_showlegend == {"x2", "x3"}  # showlegend False for extra axes

        # vector_to_project_on: 2 species x 2 panels (parallel + perpendicular)
        vtp_fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True)
        result3 = self.v_Cd_0_defect_entry.plot_site_displacements(
            use_plotly=True, vector_to_project_on=[0, 0, 1], fig=vtp_fig
        )
        assert result3 is vtp_fig
        assert len(result3.data) == 4  # 2 species x 2 panels
        assert {t.xaxis for t in result3.data} == {"x", "x2"}  # one set per panel
