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
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import PeriodicSite, Structure
from test_utils import EXAMPLE_DIR, custom_mpl_image_compare, data_dir, module_path

from doped.analysis import DefectParser, DefectsParser
from doped.core import DefectEntry, Vacancy
from doped.corrections import get_freysoldt_correction, get_kumagai_correction

try:
    from pymatgen.core.entries import ComputedStructureEntry
except ImportError:  # pymatgen <2026.3; can remove when doped/SnB pmg requirement is >=2026.3.23
    from pymatgen.entries.computed_entries import ComputedStructureEntry

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


class FiniteSizeChargeCorrectionTest(unittest.TestCase):
    """
    Test functions for getting Freysoldt (FNV) and Kumagai (eFNV) corrections.
    """

    def setUp(self):
        self.dielectric = 15.0

        struct = Structure.from_file(os.path.join(data_dir, "VO2_POSCAR"))
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

    def test_defect_frac_coords_w_get_freysoldt_correction(self):
        defect_coords_kwarg = {"defect_frac_coords": [0.3, 0.4, 0.2]}
        fnv_corr_list = [
            get_freysoldt_correction(self.defect_entry, self.dielectric, **defect_coords_kwarg)
        ]
        fnv_corr_list.append(
            get_freysoldt_correction(self.defect_entry, self.dielectric, plot=True, **defect_coords_kwarg)[
                0
            ]
        )
        for fnv_corr in fnv_corr_list:
            assert np.isclose(fnv_corr.correction_energy, 7.499071576499095)

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

    def test_defect_coords_w_get_kumagai_correction(self):
        defect_coords_kwarg = {"defect_coords": [0.3, 0.4, 0.2]}
        efnv_corr_list = [
            get_kumagai_correction(self.defect_entry, self.dielectric, **defect_coords_kwarg)
        ]
        efnv_corr_list.append(
            get_kumagai_correction(self.defect_entry, self.dielectric, plot=True, **defect_coords_kwarg)[0]
        )
        for efnv_corr in efnv_corr_list:
            assert np.isclose(efnv_corr.correction_energy, 1.1194286233529542)

    def test_multiple_point_charges_w_get_kumagai_correction(self):
        """
        Test modelling the defect charge as multiple point charges in the eFNV
        correction, with ``point_charges``.
        """
        q = self.defect_entry.charge_state
        coords = np.array(self.defect_entry.sc_defect_frac_coords)
        single_corr = get_kumagai_correction(self.defect_entry, self.dielectric, verbose=False)

        # splitting the charge in two, 0.2 Å apart, closely reproduces the single point charge:
        shift = np.array([0.1 / self.defect_entry.sc_entry.structure.lattice.a, 0, 0])
        split_charges = [(q / 2, coords - shift), (q / 2, coords + shift)]
        split_corr = get_kumagai_correction(
            self.defect_entry, self.dielectric, verbose=False, point_charges=split_charges
        )
        assert np.isclose(split_corr.correction_energy, single_corr.correction_energy, atol=5e-4)
        assert split_corr.metadata["point_charges"] == [(qi, list(fc)) for qi, fc in split_charges]

        # split-vacancy-like [V-V_i-V]^-3 model, with an anisotropic dielectric tensor; the point charge
        # energy matches an independent pymatgen ``EwaldSummation`` calculation (in the eps^-1/2-scaled
        # frame) of 1.0993703688 eV, and is independent of the ordering and periodic images of the input
        # point charges (which are unwrapped to a single configuration):
        aniso_dielectric = np.array([[12.0, 1.5, 0.0], [1.5, 9.0, -2.0], [0.0, -2.0, 20.0]])
        nn_v_coords = np.array([0.0, 0.09978070333333333, 0.09978070333333333])
        for point_charges in [
            [(-3, coords), (3, (coords + nn_v_coords) / 2), (-3, nn_v_coords)],
            [
                (-3, nn_v_coords + np.array([1, 0, -1])),
                (-3, coords),
                (3, (coords + nn_v_coords) / 2 + np.array([0, 1, 0])),
            ],
        ]:
            three_site_corr = get_kumagai_correction(
                self.defect_entry, aniso_dielectric, verbose=False, point_charges=point_charges
            )
            efnv_corr_obj = three_site_corr.metadata["pydefect_ExtendedFnvCorrection"]
            assert np.isclose(efnv_corr_obj.point_charge_correction, 1.0993703688, atol=1e-8)
            assert np.isclose(three_site_corr.correction_energy, 1.2856151718, atol=1e-8)

        for point_charges, message in [
            ([(-1, coords), (-1, nn_v_coords)], "does not match the defect charge state"),
            ([(-1.5, coords), (-1.5, coords + np.array([1, 0, 0]))], "coincident"),  # periodic images
            ([(-1.5, coords), (-1.5, coords + shift / 4)], "coincident"),  # 0.05 Å apart
        ]:
            with pytest.raises(ValueError, match=message):
                get_kumagai_correction(
                    self.defect_entry, self.dielectric, verbose=False, point_charges=point_charges
                )

    def test_get_kumagai_correction_large_dielectric(self):
        """
        Test that the eFNV point charge energy is converged for large
        dielectric constants (scaling exactly as 1/eps), which was not the case
        with the default ``pydefect`` Ewald parameters.
        """
        pc_energies = {
            dielectric: get_kumagai_correction(self.defect_entry, dielectric, verbose=False)
            .metadata["pydefect_ExtendedFnvCorrection"]
            .point_charge_correction
            for dielectric in [1, 300]
        }
        assert np.isclose(pc_energies[300] * 300, pc_energies[1], rtol=1e-10)

    def test_get_kumagai_correction_slab_defect_region_radius(self):
        """
        Test that the eFNV defect region radius (default or user-supplied) is
        reduced when fewer than 3 atomic sites lie outside it (e.g. slab
        supercells with vacuum).
        """
        slab = Structure(  # 4x4 square C monolayer (2 Å spacing), with 20 Å c-axis -> vacuum
            Lattice.from_parameters(8.0, 8.0, 20.0, 90, 90, 90),
            ["C"] * 16,
            [[i / 4, j / 4, 0.5] for i in range(4) for j in range(4)],
        )
        vac = Vacancy(slab, slab.sites[0], charge=-1)
        defect_structure = vac.defect_structure
        dists = np.array([site.distance(slab.sites[0]) for site in defect_structure])
        slab_entry = DefectEntry(
            vac,
            charge_state=-1,
            sc_entry=ComputedStructureEntry(structure=defect_structure, energy=0.0),
            bulk_entry=ComputedStructureEntry(structure=slab, energy=0.0),
            sc_defect_frac_coords=slab.sites[0].frac_coords,
            calculation_metadata={
                "bulk_site_potentials": np.full(len(slab), 30.0),
                "defect_site_potentials": 30.0 + 0.2 * np.exp(-dists),  # decaying defect perturbation
            },
        )

        # default radius (half the 20 Å c-axis) exceeds all (in-plane) site distances, and only 1 site
        # (at 5.66 Å) lies outside a 5 Å radius, so both are reduced to just below the 4 next-furthest
        # sites (at 4.47 Å), giving 5 sampled sites:
        for kwargs in [{}, {"defect_region_radius": 5}]:
            with pytest.warns(UserWarning, match="reduced to 4.47 Å to include at least 3 sites"):
                efnv_corr = get_kumagai_correction(slab_entry, [10, 10, 1], verbose=False, **kwargs)
            efnv_corr_obj = efnv_corr.metadata["pydefect_ExtendedFnvCorrection"]
            sampled = [s for s in efnv_corr_obj.sites if s.distance > efnv_corr_obj.defect_region_radius]
            assert len(sampled) == 5
            assert np.isfinite(efnv_corr.correction_energy)


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

    @classmethod
    def setUpClass(cls) -> None:
        # prepare parsed defect data (from doped parsing example)
        cls.CdTe_example_dir = os.path.join(EXAMPLE_DIR, "CdTe")
        cls.ytos_example_dir = os.path.join(EXAMPLE_DIR, "YTOS")
        cls.CdTe_bulk_data_dir = os.path.join(cls.CdTe_example_dir, "v_Cd_example_data/CdTe_bulk/vasp_ncl")
        cls.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

        cls.v_Cd_dict = {}  # dictionary of parsed vacancy defect entries

        cls.v_Cd_dict = DefectsParser(
            f"{cls.CdTe_example_dir}/v_Cd_example_data",
            bulk_path=cls.CdTe_bulk_data_dir,
            dielectric=cls.CdTe_dielectric,
        ).defect_dict

        cls.ytos_dielectric = [40.7, 40.7, 25.2]  # from legacy Materials Project
        cls.F_O_1_entry = DefectParser.from_paths(
            defect_path=f"{cls.ytos_example_dir}/F_O_1",
            bulk_path=f"{cls.ytos_example_dir}/Bulk",
            dielectric=cls.ytos_dielectric,
        ).defect_entry

        cls.Te_i_2_ent = DefectParser.from_paths(
            defect_path=f"{cls.CdTe_example_dir}/Int_Te_3_2/vasp_ncl",
            bulk_path=f"{cls.CdTe_bulk_data_dir}",
            dielectric=9.13,
        ).defect_entry

    @custom_mpl_image_compare("v_Cd_-2_FNV_plot.png")
    def test_plot_freysoldt(self):
        """
        Test FNV correction plotting.
        """
        return get_freysoldt_correction(self.v_Cd_dict["v_Cd_-2"], 9.13, plot=True)[1]

    @custom_mpl_image_compare("v_Cd_-2_FNV_plot_custom.png")
    def test_plot_freysoldt_custom(self):
        """
        Test FNV correction plotting, with customisation.
        """
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        _corr, fig = get_freysoldt_correction(self.v_Cd_dict["v_Cd_-2"], 9.13, plot=True)

        # add logo to fig:
        doped_logo = mpl.pyplot.imread(f"{module_path}/../docs/Images/doped_v2_logo.png")
        im = OffsetImage(doped_logo, zoom=0.05)
        ab = AnnotationBbox(im, (1, 0.3), xycoords="axes fraction", box_alignment=(1.1, -0.1))
        fig.gca().add_artist(ab)

        # now only show the c-axis subplot (which was the last axis and so has the logo) from the figure:
        fig.axes[0].set_visible(False)
        fig.axes[1].set_visible(False)

        return fig

    @custom_mpl_image_compare("F_O_+1_eFNV_plot.png")
    def test_plot_kumagai(self):
        """
        Test eFNV correction plotting.
        """
        mpl.pyplot.clf()
        corr, fig = get_kumagai_correction(self.F_O_1_entry, dielectric=self.ytos_dielectric, plot=True)
        assert np.isclose(corr.correction_energy, 0.12697319510724614)
        return fig

    @custom_mpl_image_compare("F_O_+1_eFNV_wacky_region_plot.png")
    def test_plot_kumagai_adjusted_sampling_region(self):
        """
        Test eFNV correction plotting, with an adjusted sampling region.
        """
        mpl.pyplot.clf()
        orig_corr_error = self.F_O_1_entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
        corr, fig = get_kumagai_correction(
            self.F_O_1_entry, dielectric=self.ytos_dielectric, defect_region_radius=7.5, plot=True
        )
        assert np.isclose(corr.correction_energy, 0.11110646242343837)
        assert np.isclose(
            sum(self.F_O_1_entry.corrections.values()), 0.12697319510724614
        )  # correction not applied to DefectEntry when using get_kumagai_correction directly
        assert np.isclose(
            orig_corr_error, self.F_O_1_entry.corrections_metadata["kumagai_charge_correction_error"]
        )  # correction error unchanged as correction not applied
        return fig

    @custom_mpl_image_compare("F_O_+1_eFNV_plot_custom.png")
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

    @custom_mpl_image_compare("F_O_+1_eFNV_plot_excluded.png")
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
        assert np.isclose(corr.correction_energy, 0.12224386127739299)  # different by ~4 meV to above
        assert np.isclose(
            sum(self.F_O_1_entry.corrections.values()), 0.12697319510724614
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
        assert np.isclose(corr.correction_energy, 0.12224386127739299)  # different by ~4 meV to above
        assert np.isclose(
            sum(self.F_O_1_entry.corrections.values()), 0.12224386127739299
        )  # correction now applied to DefectEntry when using DefectEntry.get_kumagai_correction()
        return fig

    @custom_mpl_image_compare("Te_i_+2_eFNV_plot.png")
    def test_plot_kumagai_Te_i_2(self):
        """
        Test eFNV correction plotting with Te_i^+2 (slightly trickier defect
        case).
        """
        mpl.pyplot.clf()
        return self.Te_i_2_ent.get_kumagai_correction(plot=True)[1]
