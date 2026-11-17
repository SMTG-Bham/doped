"""
Test functions for getting freysoldt and kumagai corrections.

These tests are templated off those originally written by the PyCDT (
https://doi.org/10.1016/j.cpc.2018.01.004)
developers.
"""

import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
from pymatgen.core.entries import ComputedStructureEntry
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.io.vasp.outputs import Outcar
from test_utils import EXAMPLE_DIR, custom_mpl_image_compare, data_dir, module_path

from doped.analysis import DefectParser, DefectsParser
from doped.core import DefectEntry, Vacancy
from doped.corrections import get_freysoldt_correction, get_kumagai_correction
from doped.utils.parsing import _get_defect_supercell_sites
from doped.utils.supercells import _get_unwrapped_complex_frac_coords

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
            assert np.isclose(fnv_corr.correction_energy, 5.4464703687929905)

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
            assert np.isclose(fnv_corr.correction_energy, 7.499591576499095)

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

        # split-vacancy-like [V-V_i-V]^-3 model, with an anisotropic dielectric tensor
        aniso_dielectric = np.array([[12.0, 1.5, 0.0], [1.5, 9.0, -2.0], [0.0, -2.0, 20.0]])
        nn_v_coords = np.array([0.0, 0.09978070333333333, 0.09978070333333333])
        for point_charges in [
            [(-3, coords), (3, (coords + nn_v_coords) / 2), (-3, nn_v_coords)],
            [
                (-3, nn_v_coords + np.array([1, 0, -1])),
                (-3, coords),
                (3, (coords + nn_v_coords) / 2 + np.array([0, 1, 0])),
            ],
        ]:  # independent of the ordering and periodic images of the input point charges
            three_site_corr = get_kumagai_correction(
                self.defect_entry, aniso_dielectric, verbose=False, point_charges=point_charges
            )
            efnv_corr_obj = three_site_corr.metadata["pydefect_ExtendedFnvCorrection"]
            # matches an independent pymatgen ``EwaldSummation`` calculation:
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
        Test that the default eFNV defect region radius is reduced when fewer
        than 3 atomic sites lie outside it (e.g. slab supercells with vacuum),
        while a user-supplied radius is used as given (with a warning if fewer
        than 3 sites lie outside it, or an error if none do).
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

        # default radius (half the 20 Å c-axis) exceeds all (in-plane) site distances, so is reduced to
        # just below the 4 next-furthest sites (at 4.47 Å), giving 5 sampled sites, whereas a user-supplied
        # 5 Å radius is used as given, with only 1 site (at 5.66 Å) outside it:
        for kwargs, message, num_sampled, correction_energy in [
            ({}, "reduced to 4.47 Å to include at least 3 sites", 5, 2.6496062676),
            ({"defect_region_radius": 5}, "Only 1 atomic site lies outside the supplied", 1, 2.5472823613),
        ]:
            with pytest.warns(UserWarning, match=message):
                efnv_corr = get_kumagai_correction(slab_entry, [10, 10, 1], verbose=False, **kwargs)
            efnv_corr_obj = efnv_corr.metadata["pydefect_ExtendedFnvCorrection"]
            sampled = [s for s in efnv_corr_obj.sites if s.distance > efnv_corr_obj.defect_region_radius]
            assert len(sampled) == num_sampled
            assert np.isclose(efnv_corr_obj.point_charge_correction, -1.2461289533, atol=1e-8)
            assert np.isclose(efnv_corr.correction_energy, correction_energy, atol=1e-8)

        with pytest.raises(ValueError, match="No atomic sites lie outside the supplied"):
            get_kumagai_correction(slab_entry, [10, 10, 1], verbose=False, defect_region_radius=6)

    def test_unwrapped_complex_frac_coords_half_lattice_vector_separations(self):
        """
        Test that complexes with 3+ sites separated by exactly half lattice
        vectors (so that all pairs are Wigner-Seitz cell boundary ties) are
        unwrapped with a warning, rather than raising an error.
        """
        for frac_coords in [
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        ]:
            with pytest.warns(UserWarning, match="not a minimum image of the supercell"):
                unwrapped_frac_coords = _get_unwrapped_complex_frac_coords(
                    Lattice.cubic(10), np.array(frac_coords, dtype=float)
                )
            shifts = np.array(unwrapped_frac_coords) - frac_coords
            assert np.allclose(shifts, np.round(shifts))  # same sites, modulo lattice vectors


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

    @custom_mpl_image_compare("v_Cd_-2_FNV_plot_onto_axes.png")
    def test_plot_freysoldt_onto_axes(self):
        """
        Test FNV correction plotting onto user-provided ``Axes``.
        """
        plt = mpl.pyplot
        v_Cd_entry = self.v_Cd_dict["v_Cd_-2"]
        # extra row to check only the given ``Axes`` are drawn on:
        user_fig, user_axes = plt.subplots(2, 3, figsize=(12, 7))
        user_fig.subplots_adjust(wspace=0.3, hspace=0.35)  # room for the per-panel axis labels & titles
        num_figs = len(plt.get_fignums())

        _corr, fig = get_freysoldt_correction(v_Cd_entry, 9.13, plot=True, ax=user_axes[0], verbose=False)
        assert fig is user_fig
        assert len(plt.get_fignums()) == num_figs  # no new figures
        assert all(ax.lines for ax in user_axes[0])
        assert not any(ax.lines for ax in user_axes[1])

        # single axis, also via ``DefectEntry`` method:
        _corr, fig = v_Cd_entry.get_freysoldt_correction(plot=True, axis=0, ax=user_axes[1, 1])
        assert fig is user_fig
        assert len(plt.get_fignums()) == num_figs
        assert user_axes[1, 1].lines
        assert user_axes[1, 1].get_title() == r"$a$-axis"

        root_fig = (
            plt.figure()
        )  # ``Axes`` within a ``SubFigure``; root ``Figure`` should be returned & saved
        sub_axes = root_fig.subfigures(2, 1)[0].subplots(1, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            _corr, fig = get_freysoldt_correction(
                v_Cd_entry, 9.13, plot=True, ax=sub_axes, filename=f"{tmp_dir}/FNV.png", verbose=False
            )
            assert os.listdir(tmp_dir) == ["FNV.png"]
        assert fig is root_fig
        plt.close(root_fig)

        for ax, axis in [(user_axes[0], 0), (user_axes[1, 0], None)]:  # wrong number of ``Axes``
            with pytest.raises(ValueError, match="`Axes` with `ax`, but"):
                get_freysoldt_correction(v_Cd_entry, 9.13, plot=True, axis=axis, ax=ax, verbose=False)

        return user_fig

    @custom_mpl_image_compare("F_O_+1_eFNV_plot.png")  # same as ``test_plot_kumagai``
    def test_correction_plots_keep_user_figures_open(self):
        """
        Test that FNV/eFNV correction plotting doesn't close other open (user)
        figures.
        """
        plt = mpl.pyplot
        user_fig = plt.figure()
        _corr, fnv_fig = get_freysoldt_correction(
            self.v_Cd_dict["v_Cd_-2"], 9.13, plot=True, axis=0, verbose=False
        )
        assert plt.fignum_exists(user_fig.number)
        plt.figure(user_fig.number)  # make the user figure current again; eFNV must not draw on it
        _corr, fig = get_kumagai_correction(
            self.F_O_1_entry, dielectric=self.ytos_dielectric, plot=True, verbose=False
        )
        assert plt.fignum_exists(user_fig.number)
        assert fig is not user_fig
        assert not user_fig.axes  # nothing drawn on the user figure
        plt.close(user_fig)
        plt.close(fnv_fig)
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

    def test_kumagai_correction_w_outcar_objects(self):
        """
        Test that ``Outcar`` objects can be used as the ``defect_outcar`` and
        ``bulk_outcar`` inputs, giving the same result as their file paths.
        """
        outcar_paths = {
            "defect_outcar": f"{self.CdTe_example_dir}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
            "bulk_outcar": f"{self.CdTe_bulk_data_dir}/OUTCAR.gz",
        }
        corr_from_paths = get_kumagai_correction(self.Te_i_2_ent, verbose=False, **outcar_paths)
        corr_from_objs = get_kumagai_correction(
            self.Te_i_2_ent, verbose=False, **{k: Outcar(path) for k, path in outcar_paths.items()}
        )
        assert np.isclose(corr_from_objs.correction_energy, corr_from_paths.correction_energy, atol=1e-8)
        assert np.isclose(
            corr_from_objs.correction_energy,
            self.Te_i_2_ent.corrections["kumagai_charge_correction"],
            atol=1e-6,
        )

    @custom_mpl_image_compare("Te_i_+2_eFNV_plot.png")
    def test_plot_kumagai_Te_i_2(self):
        """
        Test eFNV correction plotting with Te_i^+2 (slightly trickier defect
        case).
        """
        mpl.pyplot.clf()
        # parsed as a Cd_i-Te_i-v_Cd complex with the default ``site_tol``, so the eFNV sampling region is
        # centred on the complex centroid, ~1.4 Å from the Te_i site, which changes the alignment term (the
        # point charge term is the same); 0.9040 eV for Te_i (``site_tol=None``), and 0.9109 eV with the
        # charge split over the Te trimer (``test_plot_kumagai_Te_i_2_multiple_point_charges``):
        corr, fig = self.Te_i_2_ent.get_kumagai_correction(plot=True)
        assert np.isclose(corr.correction_energy, 0.9002, atol=1e-4)
        return fig

    def test_kumagai_Te_i_2_charge_models(self):
        """
        Test the eFNV correction for Te_i^+2 (parsed as a Cd_i-Te_i-v_Cd
        complex) with different models of the defect charge: a single point
        charge at the complex centroid (default) or Te_i site, or multiple
        point charges on the Te trimer or the complex constituents.
        """
        ent = self.Te_i_2_ent
        struct = ent.defect_supercell
        assert [defect.name for defect in ent.defect.defects] == ["v_Cd", "Cd_i", "Te_i"]
        constituent_sites = _get_defect_supercell_sites(ent)
        formal_charges = {"Cd_i": 2, "Te_i": 2, "v_Cd": -2}  # Cd_i is the displaced Cd, 1.58 Å from v_Cd

        for kwargs, correction, point_charge_correction, alignment_correction in [
            ({}, 0.9002, 0.6839, 0.2163),  # single +2 point charge at complex centroid (default)
            ({"defect_coords": struct[64].frac_coords}, 0.9040, 0.6839, 0.2201),  # +2 at Te_i site
            (  # +2/3 on Te_i and its 2 Te nearest neighbours (Te trimer):
                {"point_charges": [(2 / 3, struct[i].frac_coords) for i in [64, 63, 53]]},
                0.9109,
                0.6598,
                0.2511,
            ),
            (  # formal charges on the complex constituents:
                {
                    "point_charges": [
                        (formal_charges[defect.name], site.frac_coords)
                        for defect, site in zip(ent.defect.defects, constituent_sites, strict=True)
                    ]
                },
                0.8888,
                0.6630,
                0.2258,
            ),
        ]:
            print(f"Testing with kwargs: {kwargs}")
            corr = get_kumagai_correction(ent, verbose=False, **kwargs)
            assert np.isclose(corr.correction_energy, correction, atol=1e-4)
            efnv = corr.metadata["pydefect_ExtendedFnvCorrection"]
            assert np.isclose(efnv.point_charge_correction, point_charge_correction, atol=1e-4)
            assert np.isclose(efnv.alignment_correction, alignment_correction, atol=1e-4)

    @custom_mpl_image_compare("Te_i_+2_eFNV_plot_multiple_point_charges.png")
    def test_plot_kumagai_Te_i_2_multiple_point_charges(self):
        """
        Test eFNV correction plotting with Te_i^+2 modelled as multiple point
        charges, split over the Te trimer (with one Te across the cell
        boundary, so also testing the unwrapping of the point charges).
        """
        mpl.pyplot.clf()
        struct = self.Te_i_2_ent.defect_supercell
        point_charges = [(2 / 3, struct[i].frac_coords) for i in [64, 63, 53]]  # Te_i & its 2 Te NNs
        corr, fig = get_kumagai_correction(self.Te_i_2_ent, point_charges=point_charges, plot=True)
        assert np.isclose(corr.correction_energy, 0.910851106265789, atol=1e-5)  # 0.904 eV for single PC
        assert corr.metadata["point_charges"] == [(q, list(fc)) for q, fc in point_charges]
        return fig
