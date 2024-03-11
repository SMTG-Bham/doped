"""
Test functions for getting freysoldt and kumagai corrections.

These tests are templated off those originally written by the PyCDT (
https://doi.org/10.1016/j.cpc.2018.01.004)
developers.
"""

import os
import unittest
from typing import Any
from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
from pymatgen.core.sites import PeriodicSite
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.util.testing import PymatgenTest
from test_analysis import if_present_rm

from doped import analysis
from doped.core import DefectEntry, Vacancy
from doped.corrections import get_freysoldt_correction, get_kumagai_correction

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally

# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")


class FiniteSizeChargeCorrectionTest(PymatgenTest):
    """
    Test functions for getting freysoldt and kumagai corrections.
    """

    def setUp(self):
        self.dielectric = 15.0

        struct = PymatgenTest.get_structure("VO2")
        struct.make_supercell(3)
        vac = Vacancy(struct, struct.sites[0], charge=-3)

        ids = vac.defect_structure
        abc = struct.lattice.abc
        bldata = {
            i: np.array([1.0 for _ in np.arange(0.0, lattval, 0.2)]) for i, lattval in enumerate(abc)
        }
        dldata = {
            i: np.array([(-1 - np.cos(2 * np.pi * u / lattval)) for u in np.arange(0.0, lattval, 0.2)])
            for i, lattval in enumerate(abc)
        }
        # load necessary parameters for defect_entry to make use of Freysoldt and Kumagai corrections
        metadata = {
            "bulk_locpot_dict": bldata,
            "defect_locpot_dict": dldata,
            "defect_frac_sc_coords": struct.sites[0].frac_coords,
        }
        bulk_atomic_site_averages, defect_atomic_site_averages = [], []
        defect_site_with_sc_lattice = PeriodicSite(
            struct.sites[0].specie,
            struct.sites[0].coords,
            struct.lattice,
            coords_are_cartesian=True,
        )
        max_dist = 9.6
        pert_amnt = 1.0
        for site_ind, site in enumerate(struct.sites):
            if site.specie.symbol == "O":
                Oval = -30.6825
                bulk_atomic_site_averages.append(Oval)
                if site_ind:
                    dist_to_defect = site.distance_and_image(defect_site_with_sc_lattice)[0]
                    defect_site_val = (
                        Oval - 0.3 + pert_amnt * ((max_dist - dist_to_defect) / max_dist) ** 2
                    )
                    defect_atomic_site_averages.append(defect_site_val)
            else:
                Vval = -51.6833
                bulk_atomic_site_averages.append(Vval)
                if site_ind:
                    dist_to_defect = site.distance_and_image(defect_site_with_sc_lattice)[0]
                    defect_site_val = (
                        Vval - 0.3 + pert_amnt * ((max_dist - dist_to_defect) / max_dist) ** 2
                    )
                    defect_atomic_site_averages.append(defect_site_val)

        metadata.update(
            {
                "bulk_site_potentials": -1 * np.array(bulk_atomic_site_averages),
                "defect_site_potentials": -1 * np.array(defect_atomic_site_averages),
            }
        )

        self.defect_entry = DefectEntry(
            vac,
            charge_state=-3,
            sc_entry=ComputedStructureEntry(
                structure=ids,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            bulk_entry=ComputedStructureEntry(
                structure=struct,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            sc_defect_frac_coords=struct.sites[0].frac_coords,
            calculation_metadata=metadata,
        )

    def test_get_freysoldt_correction(self):
        fnv_corr_list = [get_freysoldt_correction(self.defect_entry, self.dielectric)]
        fnv_corr_list.append(get_freysoldt_correction(self.defect_entry, self.dielectric, axis=0))
        fnv_corr_list.append(get_freysoldt_correction(self.defect_entry, self.dielectric, axis=2))
        fnv_corr_list.append(
            get_freysoldt_correction(self.defect_entry, self.dielectric, axis=2, plot=True)[0]
        )

        for fnv_corr in fnv_corr_list:
            assert np.isclose(fnv_corr.correction_energy, 5.445950368792991)

        # test verbose option:
        with patch("builtins.print") as mock_print:
            get_freysoldt_correction(self.defect_entry, self.dielectric, verbose=True)
        mock_print.assert_called_once_with("Calculated Freysoldt (FNV) correction is 5.446 eV")

        with patch("builtins.print") as mock_print:
            get_freysoldt_correction(self.defect_entry, self.dielectric, verbose=False)
        mock_print.assert_not_called()

    def test_get_kumagai_correction(self):
        efnv_corr_list = [get_kumagai_correction(self.defect_entry, self.dielectric)]
        efnv_corr_list.append(get_kumagai_correction(self.defect_entry, self.dielectric, plot=True)[0])
        for efnv_corr in efnv_corr_list:
            assert np.isclose(efnv_corr.correction_energy, 1.2651776920778381)

        # test verbose option:
        with patch("builtins.print") as mock_print:
            get_kumagai_correction(self.defect_entry, self.dielectric, verbose=True)
        mock_print.assert_called_once_with("Calculated Kumagai (eFNV) correction is 1.265 eV")

        with patch("builtins.print") as mock_print:
            get_kumagai_correction(self.defect_entry, self.dielectric, verbose=False)
        mock_print.assert_not_called()


class CorrectionsPlottingTestCase(unittest.TestCase):
    module_path: str
    example_dir: str
    CdTe_example_dir: str
    ytos_example_dir: str
    ytos_dielectric: list
    CdTe_bulk_data_dir: str
    CdTe_dielectric: np.ndarray
    v_Cd_dict: dict[Any, Any]
    F_O_1_entry: DefectEntry
    Te_i_2_ent: DefectEntry

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_bulk_data_dir, "voronoi_nodes.json"))

    @classmethod
    def setUpClass(cls) -> None:
        # prepare parsed defect data (from doped parsing example)
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.example_dir = os.path.join(cls.module_path, "../examples")
        cls.CdTe_example_dir = os.path.join(cls.module_path, "../examples/CdTe")
        cls.ytos_example_dir = os.path.join(cls.module_path, "../examples/YTOS")
        cls.CdTe_bulk_data_dir = os.path.join(cls.CdTe_example_dir, "CdTe_bulk/vasp_ncl")
        cls.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

        cls.v_Cd_dict = {}  # dictionary of parsed vacancy defect entries

        for i in os.listdir(cls.CdTe_example_dir):  # loops through the example directory
            if (
                os.path.isdir(f"{cls.CdTe_example_dir}/{i}") and "v_Cd" in i
            ):  # and parses folders that have "v_Cd" in their name
                print(f"Parsing {i}...")
                defect_path = f"{cls.CdTe_example_dir}/{i}/vasp_ncl"
                cls.v_Cd_dict[i] = analysis.defect_entry_from_paths(
                    defect_path,
                    cls.CdTe_bulk_data_dir,
                    cls.CdTe_dielectric,
                    charge_state=int(i.split("_")[-1]),  # test manually specifying charge states here
                )

        cls.ytos_dielectric = [40.7, 40.7, 25.2]  # from legacy Materials Project
        cls.F_O_1_entry = analysis.defect_entry_from_paths(
            defect_path=f"{cls.ytos_example_dir}/F_O_1",
            bulk_path=f"{cls.ytos_example_dir}/Bulk",
            dielectric=cls.ytos_dielectric,
        )

        cls.Te_i_2_ent = analysis.defect_entry_from_paths(
            defect_path=f"{cls.CdTe_example_dir}/Int_Te_3_2/vasp_ncl",
            bulk_path=f"{cls.CdTe_bulk_data_dir}",
            dielectric=9.13,
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="v_Cd_-2_FNV_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_freysoldt(self):
        """
        Test FNV correction plotting.
        """
        return get_freysoldt_correction(self.v_Cd_dict["v_Cd_-2"], 9.13, plot=True)[1]

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="v_Cd_-2_FNV_plot_custom.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_freysoldt_custom(self):
        """
        Test FNV correction plotting, with customisation.
        """
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        _corr, fig = get_freysoldt_correction(self.v_Cd_dict["v_Cd_-2"], 9.13, plot=True)

        # add logo to fig:
        doped_logo = mpl.pyplot.imread(f"{module_path}/../docs/doped_v2_logo.png")
        im = OffsetImage(doped_logo, zoom=0.05)
        ab = AnnotationBbox(im, (1, 0.3), xycoords="axes fraction", box_alignment=(1.1, -0.1))
        fig.gca().add_artist(ab)

        # now only show the c-axis subplot (which was the last axis and so has the logo) from the figure:
        fig.axes[0].set_visible(False)
        fig.axes[1].set_visible(False)

        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="F_O_+1_eFNV_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_kumagai(self):
        """
        Test eFNV correction plotting.
        """
        mpl.pyplot.clf()
        corr, fig = get_kumagai_correction(self.F_O_1_entry, dielectric=self.ytos_dielectric, plot=True)
        assert np.isclose(corr.correction_energy, 0.12691248591191384)
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="F_O_+1_eFNV_wacky_region_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_kumagai_adjusted_sampling_region(self):
        """
        Test eFNV correction plotting, with an adjusted sampling region.
        """
        mpl.pyplot.clf()
        orig_corr_error = self.F_O_1_entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
        corr, fig = get_kumagai_correction(
            self.F_O_1_entry, dielectric=self.ytos_dielectric, defect_region_radius=7.5, plot=True
        )
        assert np.isclose(corr.correction_energy, 0.11104892482814409)
        assert np.isclose(
            sum(self.F_O_1_entry.corrections.values()), 0.12691248591191384
        )  # correction not applied to DefectEntry when using get_kumagai_correction directly
        assert np.isclose(
            orig_corr_error, self.F_O_1_entry.corrections_metadata["kumagai_charge_correction_error"]
        )  # correction error unchanged as correction not applied
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="F_O_+1_eFNV_plot_custom.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_kumagai_custom(self):
        """
        Test eFNV correction plotting, with figure customisation.
        """
        mpl.pyplot.clf()
        _corr, fig = get_kumagai_correction(self.F_O_1_entry, dielectric=self.ytos_dielectric, plot=True)
        # add shading to plot:
        ax = fig.gca()
        ax.axvspan(5, 100, alpha=0.2, color="yellow")

        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="F_O_+1_eFNV_plot_excluded.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_kumagai_excluded(self):
        """
        Test eFNV correction plotting and correction, with excluded_indices
        parameter.
        """
        mpl.pyplot.clf()
        orig_corr_error = self.F_O_1_entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
        corr, fig = get_kumagai_correction(
            self.F_O_1_entry,
            dielectric=self.ytos_dielectric,
            defect_region_radius=7.5,
            plot=True,
            excluded_indices=np.arange(0, 109),
        )
        # gives only Oxygens in the correction/plot
        assert np.isclose(corr.correction_energy, 0.12218608369699298)  # different by ~4 meV to above
        assert np.isclose(
            sum(self.F_O_1_entry.corrections.values()), 0.12691248591191384
        )  # correction not applied to DefectEntry when using get_kumagai_correction directly
        assert np.isclose(
            orig_corr_error, self.F_O_1_entry.corrections_metadata["kumagai_charge_correction_error"]
        )  # correction error unchanged as correction not applied

        # F substitution is site number 109, so whether or not it's excluded gives same figure and result
        corr, fig = self.F_O_1_entry.get_kumagai_correction(  # test as DefectEntry method this time
            dielectric=self.ytos_dielectric,
            defect_region_radius=7.5,
            plot=True,
            excluded_indices=np.arange(0, 110),
        )
        assert np.isclose(corr.correction_energy, 0.12218608369699298)  # different by ~4 meV to above
        assert np.isclose(
            sum(self.F_O_1_entry.corrections.values()), 0.12218608369699298
        )  # correction now applied to DefectEntry when using DefectEntry.get_kumagai_correction()
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="Te_i_+2_eFNV_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_kumagai_Te_i_2(self):
        """
        Test eFNV correction plotting with Te_i^+2 (slightly trickier defect
        case).
        """
        mpl.pyplot.clf()
        return self.Te_i_2_ent.get_kumagai_correction(plot=True)[1]
