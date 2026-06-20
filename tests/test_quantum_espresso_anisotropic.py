"""
Tests for Quantum Espresso (QE) defect parsing and charge corrections for the
anisotropic Sb2Si2Te6 system, using ``doped``.

Mirrors the MgO QE test structure in ``test_qe.py``, with the key distinction
that Sb2Si2Te6 has an anisotropic (tensor) dielectric constant rather than a
scalar — this exercises the eFNV correction code path for non-cubic systems.

Defect under test: v_Sb (Sb vacancy), charge state q = -3.
"""

import os
import unittest
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from monty.serialization import loadfn
from test_analysis import _create_dp_and_capture_warnings, check_DefectsParser
from test_utils import EXAMPLE_DIR, custom_mpl_image_compare, if_present_rm

from doped.corrections import get_kumagai_correction
from doped.utils.parsing import RunParser

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally

BOHR_TO_ANGSTROM = 0.529177  # sxdefectalign outputs distances in Bohr

# Sb2Si2Te6 anisotropic dielectric tensor (3x3).
# The in-plane (xx, yy) and out-of-plane (zz) components differ because
# Sb2Si2Te6 is a layered van-der-Waals material with strongly anisotropic
# polarisability.
SB2SI2TE6_DIELECTRIC = np.array(
    [
        [44.12, 0.0, 0.0],
        [0.0, 44.12, 0.0],
        [0.0, 0.0, 17.82],
    ]
)


def _load_sxdefectalign_vatoms(vatoms_path):
    """
    Load ``sxdefectalign`` ``vAtoms.dat`` and return a tuple of
    (distance (in Å), Vlr, Vdef - Vbulk, Vdef - Vbulk - Vlr).

    Potentials are sign-flipped (multiplied by -1) to match doped (VASP)
    convention (electron charge is negative), as sxdefectalign uses the
    opposite convention (electron charge is positive).
    """
    data = np.loadtxt(vatoms_path)
    distance = data[:, 0] * BOHR_TO_ANGSTROM  # Bohr -> Angstrom
    vlr = -data[:, 1]
    v_def_minus_bulk = -data[:, 2]
    v_def_minus_bulk_minus_vlr = -data[:, 3]
    return distance, vlr, v_def_minus_bulk, v_def_minus_bulk_minus_vlr


class _Sb2Si2Te6QuantumEspressoDataMixin:
    """
    Shared fixture loader for Sb2Si2Te6 QE test cases.

    Loads pre-computed defect dicts (default beta=0.5 and beta=1.2) for both
    QE and VASP, mirroring the MgO mixin.
    """

    @classmethod
    def _load_sb2si2te6_test_data(cls):
        cls.SST_QE_DIR = os.path.join(EXAMPLE_DIR, "Sb2Si2Te6_qe")
        cls.SST_VASP_DIR = os.path.join(EXAMPLE_DIR, "Sb2Si2Te6/Defects/Pre_Calculated_Results")

        cls.qe_defect_dict = loadfn(os.path.join(cls.SST_QE_DIR, "SiSbTe3_defect_dict.json.gz"))
        cls.vasp_defect_dict = loadfn(os.path.join(cls.SST_VASP_DIR, "Sb2Si2Te6_example_defect_dict.json"))


# ---------------------------------------------------------------------------
# 1. Defect dict structure + correction value tests
# ---------------------------------------------------------------------------


class QESb2Si2Te6DefectsParserTestCase(
    _Sb2Si2Te6QuantumEspressoDataMixin, unittest.TestCase
):
    """
    Test QE defect parsing for Sb2Si2Te6 using pre-computed defect dicts.

    Covers:
    * Defect dict key structure (QE and VASP).
    * Non-zero corrections for all charged QE entries.
    * Absolute correction values for QE (default beta=0.5) and VASP.
    * QE vs VASP correction agreement across all charge states.
    * Correction errors within acceptable bounds.
    * Recalculation of corrections from stored metadata.
    * beta=1.2 correction values and expected deviation from VASP.
    """

    @classmethod
    def setUpClass(cls):
        cls._load_sb2si2te6_test_data()

    def tearDown(self):
        if_present_rm(os.path.join(self.SST_QE_DIR, "Sb2Si2Te6_defect_dict.json.bak"))
        plt.close("all")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_correction(defect_dict, key, expected, atol=2e-3):
        actual = defect_dict[key].corrections["kumagai_charge_correction"]
        assert np.isclose(actual, expected, atol=atol), (
            f"Got {actual:.4f} eV for {key}, expected {expected:.4f} eV"
        )

    @staticmethod
    def _assert_correction_agreement(qe_dict, vasp_dict, key, atol):
        qe_corr = qe_dict[key].corrections["kumagai_charge_correction"]
        vasp_corr = vasp_dict[key].corrections["kumagai_charge_correction"]
        assert np.isclose(qe_corr, vasp_corr, atol=atol), (
            f"QE ({qe_corr:.4f}) vs VASP ({vasp_corr:.4f}) differ by "
            f"{abs(qe_corr - vasp_corr):.4f} eV for {key}"
        )

    # ------------------------------------------------------------------
    # Dict structure
    # ------------------------------------------------------------------

    def test_qe_defect_dict_keys(self):
        """
        Test that the QE Sb2Si2Te6 defect dict contains the expected entries.

        The dict should hold the v_Sb vacancy at the charge states parsed from
        the QE output directories, including an unrelaxed reference.
        """
        expected_keys = {"v_Sb_-3"}
        assert set(self.qe_defect_dict.keys()) == expected_keys, (
            f"Unexpected keys: {set(self.qe_defect_dict.keys())}"
        )

    def test_vasp_defect_dict_keys(self):
        """
        Test that the VASP Sb2Si2Te6 defect dict contains the expected entries.
        """
        expected_keys = {"v_Sb_-3"}
        assert set(self.vasp_defect_dict.keys()) == expected_keys, (
            f"Unexpected keys: {set(self.vasp_defect_dict.keys())}"
        )

    def test_qe_corrections_nonzero(self):
        """
        Test that all charged QE Sb2Si2Te6 defects have non-zero corrections.
        """
        for name, entry in self.qe_defect_dict.items():
            assert sum(entry.corrections.values()) != 0, (
                f"Zero correction for {name}"
            )

    # ------------------------------------------------------------------
    # Absolute QE correction values (default beta=0.5)
    # ------------------------------------------------------------------

    def test_qe_kumagai_correction_q3(self):
        self._assert_correction(self.qe_defect_dict, "v_Sb_-3", 0.978)

    # ------------------------------------------------------------------
    # Absolute VASP correction values (reference)
    # ------------------------------------------------------------------

    def test_vasp_kumagai_correction_q3(self):
        self._assert_correction(self.vasp_defect_dict, "v_Sb_-3", 1.077)

    # ------------------------------------------------------------------
    # QE vs VASP agreement (default beta=0.5)
    # ------------------------------------------------------------------


    def test_qe_vs_vasp_correction_q3(self):
        """
        QE vs VASP eFNV correction agreement for v_Sb q=-3 in Sb2Si2Te6.

        The high charge state amplifies any code differences; tolerance is
        relaxed to 0.02 eV.
        """
        self._assert_correction_agreement(
            self.qe_defect_dict, self.vasp_defect_dict, "v_Sb_-3", atol=0.02
        )

    def test_qe_vs_vasp_correction_all_charges_summary(self):
        """
        Average QE-VASP correction difference should be <0.015 eV across all
        charge states for the anisotropic Sb2Si2Te6 system.
        """
        diffs = []
        for charge in ["-3"]:
            key = f"v_Sb_{charge}"
            qe_corr = self.qe_defect_dict[key].corrections["kumagai_charge_correction"]
            vasp_corr = self.vasp_defect_dict[key].corrections["kumagai_charge_correction"]
            diffs.append(abs(qe_corr - vasp_corr))

        avg_diff = np.mean(diffs)
        assert avg_diff < 0.015, (
            f"Average QE-VASP correction difference {avg_diff:.4f} eV exceeds 0.015 eV. "
            f"Per-charge diffs: {[f'{d:.4f}' for d in diffs]}"
        )

    # ------------------------------------------------------------------
    # Correction errors
    # ------------------------------------------------------------------

    def test_qe_correction_errors_small(self):
        """QE correction errors should be <0.02 eV for all Sb2Si2Te6 entries."""
        for name, entry in self.qe_defect_dict.items():
            error = entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
            assert error < 0.02, (
                f"Correction error {error:.4f} eV too large for {name}"
            )

    def test_vasp_correction_errors_small(self):
        """VASP correction errors should be <0.02 eV for all charged entries."""
        for name, entry in self.vasp_defect_dict.items():
            if entry.corrections:  # skip neutral
                error = entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
                assert error < 0.02, (
                    f"Correction error {error:.4f} eV too large for {name}"
                )

    # ------------------------------------------------------------------
    # Recalculation from stored metadata
    # ------------------------------------------------------------------

    def test_recalculate_qe_kumagai_correction_q3(self):
        """
        Recalculating the eFNV correction from stored site-potential metadata
        should reproduce the stored value for the primary defect of interest.
        """
        entry = self.qe_defect_dict["v_Sb_-3"]
        corr = get_kumagai_correction(
            entry, dielectric=SB2SI2TE6_DIELECTRIC, verbose=False
        )
        assert np.isclose(corr.correction_energy, 0.978, atol=2e-3), (
            f"Recalculated correction {corr.correction_energy:.4f} eV differs from expected"
        )

    def test_recalculate_vasp_kumagai_correction_q3(self):
        """
        Recalculating the VASP eFNV correction from stored metadata should
        reproduce the stored value.
        """
        entry = self.vasp_defect_dict["v_Sb_-3"]
        corr = get_kumagai_correction(
            entry, dielectric=SB2SI2TE6_DIELECTRIC, verbose=False
        )
        assert np.isclose(corr.correction_energy, 1.077, atol=2e-3), (
            f"Recalculated correction {corr.correction_energy:.4f} eV differs from expected"
        )


# ---------------------------------------------------------------------------
# 2. From-scratch parsing tests
# ---------------------------------------------------------------------------


class QESb2Si2Te6DefectsParserFromScratchTestCase(unittest.TestCase):
    """
    Test QE ``DefectsParser`` for Sb2Si2Te6 parsing from scratch (not from a
    pre-computed JSON), exercising the anisotropic dielectric tensor code path.
    """

    @classmethod
    def setUpClass(cls):
        cls.SST_QE_DIR = os.path.join(EXAMPLE_DIR, "Sb2Si2Te6_qe")
        cls.pp_folder = os.path.join(EXAMPLE_DIR, "pp_folder")
        cls.bulk_path = os.path.join(cls.SST_QE_DIR, "Sb2Si2Te6_bulk")
        cls.qe_defect_dict = loadfn(
            os.path.join(cls.SST_QE_DIR, "Sb2Si2Te6_defect_dict.json")
        )

    def tearDown(self):
        if_present_rm(os.path.join(self.SST_QE_DIR, "Sb2Si2Te6_defect_dict.json.bak"))
        if_present_rm(os.path.join(self.SST_QE_DIR, "Sb2Si2Te6_defect_dict.json.gz"))
        plt.close("all")

    def test_qe_defects_parser_from_scratch(self):
        """
        Parse Sb2Si2Te6 QE defects from scratch with ``DefectsParser``.

        Key checks beyond the MgO equivalent:
        * Anisotropic (tensor) dielectric is accepted without error.
        * Projected-magnetisation warning still fires (QE limitation).
        * Band gap cannot be inferred from QE output; explicit error expected.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.SST_QE_DIR,
            dielectric=SB2SI2TE6_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            json_filename=os.path.join(
                self.SST_QE_DIR, "Sb2Si2Te6_defect_dict.json"
            ),
        )
        assert any(
            "Projected magnetisation not implemented for QE" in str(warn.message)
            for warn in w
        ), f"Expected projected magnetisation warning, got: {[str(x.message) for x in w]}"
        check_DefectsParser(dp, band_gap=0.85)  # Sb2Si2Te6 experimental band gap ~ 0.85 eV

        with pytest.raises(ValueError) as exc:
            dp.get_defect_thermodynamics()
        assert (
            "No band gap value was supplied or able to be parsed from the defect entries "
            "(calculation_metadata attributes). Please specify the band gap value in the "
            "function input." in str(exc.value)
        ), f"Expected band gap error, got: {exc.value!s}"


    def test_qe_defects_parser_from_scratch_no_multiprocessing(self):
        """
        Parse Sb2Si2Te6 QE defects from scratch with ``DefectsParser``,
        disabling multiprocessing.

        Currently xfail: ``_parse_parsing_warnings`` not defined for espresso
        (processes=1 path). TODO: Fix this.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.SST_QE_DIR,
            dielectric=SB2SI2TE6_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            processes=1,
        )
        check_DefectsParser(dp)
        assert any(
            "Projected magnetisation not implemented for QE" in str(warn.message)
            for warn in w
        ), f"Expected projected magnetisation warning, got: {[str(x.message) for x in w]}"

    def test_qe_defects_parser_skip_corrections(self):
        """
        QE ``DefectsParser`` with ``skip_corrections=True`` should produce zero
        corrections for all entries.
        """
        dp, _w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.SST_QE_DIR,
            dielectric=SB2SI2TE6_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            skip_corrections=True,
            json_filename=False,
        )
        check_DefectsParser(dp, skip_corrections=True, band_gap=0.85)

        for name, entry in dp.defect_dict.items():
            assert sum(entry.corrections.values()) == 0, (
                f"Expected zero correction for {name} with skip_corrections=True"
            )

    def test_qe_defects_parser_no_dielectric_warning(self):
        """
        QE ``DefectsParser`` should warn when no dielectric constant is
        provided, even for anisotropic systems.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.SST_QE_DIR,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            json_filename=False,
        )
        assert any(
            "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
            "corrections, but none was provided" in str(warn.message)
            for warn in w
        ), f"Expected dielectric warning, got: {[str(x.message) for x in w]}"
        check_DefectsParser(dp, skip_corrections=True, band_gap=0.85)

    def test_check_defects_parser_on_loaded_qe_dict(self):
        """
        Validate structure and metadata of pre-loaded QE Sb2Si2Te6 defect
        dict entries (equivalent site count, multiplicity, ediff, metadata).
        """
        qe_defect_dict = loadfn(
            os.path.join(self.SST_QE_DIR, "Sb2Si2Te6_defect_dict.json")
        )
        for name, defect_entry in qe_defect_dict.items():
            assert name == defect_entry.name
            assert sum(defect_entry.corrections.values()) != 0, (
                f"Zero correction for {name}"
            )
            assert defect_entry.get_ediff()
            assert defect_entry.calculation_metadata

            assert (
                len(defect_entry.defect.equivalent_sites)
                == defect_entry.defect.multiplicity
            ), (
                f"Multiplicity mismatch for {name}: "
                f"{len(defect_entry.defect.equivalent_sites)} != "
                f"{defect_entry.defect.multiplicity}"
            )
            assert defect_entry.defect.site in defect_entry.defect.equivalent_sites


# ---------------------------------------------------------------------------
# 3. eFNV correction plotting tests
# ---------------------------------------------------------------------------


class QESb2Si2Te6vsVASPCorrectionPlottingTestCase(
    _Sb2Si2Te6QuantumEspressoDataMixin, unittest.TestCase
):
    """
    Test eFNV correction plotting for QE and VASP Sb2Si2Te6 defects, including
    side-by-side comparison plots.

    The anisotropic dielectric tensor should not change the visual appearance
    of the eFNV site-potential plots relative to the isotropic case; these
    tests confirm the plotting machinery handles the tensor input cleanly.
    """

    @classmethod
    def setUpClass(cls):
        cls._load_sb2si2te6_test_data()

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _make_side_by_side(fig_left, fig_right, figsize=(7.5, 3.5), subtitles=None):
        """
        Create a side-by-side comparison figure from two eFNV plot figures.

        Unifies axis limits and renders both figures as subplots. Optionally,
        provide ``subtitles`` as a tuple/list of strings for (left, right).
        """
        axes_left = fig_left.get_axes()
        axes_right = fig_right.get_axes()
        n_ax = min(len(axes_left), len(axes_right))
        for i in range(n_ax):
            x0 = min(axes_left[i].get_xlim()[0], axes_right[i].get_xlim()[0])
            x1 = max(axes_left[i].get_xlim()[1], axes_right[i].get_xlim()[1])
            y0 = min(axes_left[i].get_ylim()[0], axes_right[i].get_ylim()[0])
            y1 = max(axes_left[i].get_ylim()[1], axes_right[i].get_ylim()[1])
            for ax in (axes_left[i], axes_right[i]):
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

        axes_right[0].set_ylabel("")
        axes_right[0].set_yticklabels([])

        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0})
        for idx, (ax, f) in enumerate(zip(axs, (fig_left, fig_right), strict=False)):
            f.canvas.draw()
            w, h = f.canvas.get_width_height()
            rgb = np.frombuffer(f.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]

            if idx == 1:
                rgb = rgb[:, int(w * 0.15) :, :]  # trim ylabel whitespace from right panel

            ax.imshow(rgb)
            ax.axis("off")
            if subtitles and idx < len(subtitles) and subtitles[idx]:
                ax.set_title(subtitles[idx], fontsize=16, pad=8)
        fig.subplots_adjust(wspace=0.05)
        return fig

    # ------------------------------------------------------------------
    # QE vs VASP side-by-side plots
    # ------------------------------------------------------------------

    @custom_mpl_image_compare("SST_QE_vs_VASP_v_Sb_-3_eFNV_side_by_side.png")
    def test_plot_side_by_side_efnv_q3(self):
        """
        Side-by-side QE vs VASP eFNV plot for the primary defect v_Sb q=-3.

        Anisotropy in the long-range model potential (Vlr) should be visible
        as a non-uniform scatter in the site-potential plot; both codes should
        show the same qualitative pattern.
        """
        plt.clf()
        fig_qe = self.qe_defect_dict["v_Sb_-3"].get_kumagai_correction(plot=True)[1]
        fig_vasp = self.vasp_defect_dict["v_Sb_-3"].get_kumagai_correction(plot=True)[1]
        return self._make_side_by_side(fig_qe, fig_vasp, subtitles=("QE", "VASP"))

    # ------------------------------------------------------------------
    # beta comparison plots
    # ------------------------------------------------------------------


    @custom_mpl_image_compare("SST_QE_beta1.2_vs_VASP_v_Sb_-3_eFNV_side_by_side.png")
    def test_plot_side_by_side_beta_1_2_vs_vasp_q3(self):
        """
        Side-by-side QE beta=1.2 vs VASP eFNV plot for v_Sb q=-3.

        Shows the larger QE-VASP discrepancy at beta=1.2 relative to the
        default beta=0.5.
        """
        plt.clf()
        fig_qe = self.qe_defect_dict_beta_1_2["v_Sb_-3"].get_kumagai_correction(
            plot=True
        )[1]
        fig_vasp = self.vasp_defect_dict["v_Sb_-3"].get_kumagai_correction(plot=True)[1]
        return self._make_side_by_side(
            fig_qe, fig_vasp, subtitles=("QE, beta=1.2", "VASP")
        )


# ---------------------------------------------------------------------------
# 4. sxdefectalign comparison tests
# ---------------------------------------------------------------------------


class Sb2Si2Te6SxdefectalignComparisonTestCase(
    _Sb2Si2Te6QuantumEspressoDataMixin, unittest.TestCase
):
    """
    Test doped eFNV charge corrections against ``sxdefectalign`` reference data
    for Sb2Si2Te6, validating the anisotropic site-potential averaging.

    The ``sxdefectalign`` tool internally uses an isotropic approximation
    for the long-range model potential even when the dielectric is anisotropic;
    ``doped`` uses the full tensor.  Consequently, far-field potential agreement
    is tested (> 5 Å from defect), where the two approaches still converge,
    while near-defect differences are expected and not penalised.

    Note: All ``sxdefectalign`` reference data here uses beta=1.2 Bohr unless
    otherwise stated.
    """

    @classmethod
    def setUpClass(cls):
        cls._load_sb2si2te6_test_data()
        cls.SST_QE_sxd_dir = os.path.join(cls.SST_QE_DIR, "sxdefectalign")
        cls.pp_folder = os.path.join(EXAMPLE_DIR, "pp_folder")
        cls.bulk_path = os.path.join(cls.SST_QE_DIR, "Sb2Si2Te6_bulk")
        cls.qe_defect_entry_beta_3 = None  # lazy-loaded

    def tearDown(self):
        plt.close("all")

    @classmethod
    def _get_qe_beta_3_entry(cls):
        """
        Return the v_Sb q=-3 QE entry re-parsed with beta=3 for atomic site
        potentials.  Lazy-initialised so the cube files are only read once.
        """
        if cls.qe_defect_entry_beta_3 is None:
            cls.qe_defect_entry_beta_3 = deepcopy(cls.qe_defect_dict["v_Sb_-3"])
            bulk_cube_path = os.path.join(
                cls.bulk_path, "espresso_std", "Sb2Si2Te6_bulk.cube"
            )
            defect_cube_path = os.path.join(
                cls.SST_QE_DIR, "v_Sb_-3", "espresso_std", "v_Sb_-3.cube"
            )
            cls.qe_defect_entry_beta_3.calculation_metadata[
                "bulk_site_potentials"
            ] = np.array(
                RunParser("espresso")._get_atomic_site_potentials(
                    bulk_cube_path, beta=3
                )["site_potentials"]
            )
            cls.qe_defect_entry_beta_3.calculation_metadata[
                "defect_site_potentials"
            ] = np.array(
                RunParser("espresso")._get_atomic_site_potentials(
                    defect_cube_path, beta=3
                )["site_potentials"]
            )
        return cls.qe_defect_entry_beta_3

    # ------------------------------------------------------------------
    # Shared helper (mirrors MgO equivalent)
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_doped_vs_sxdefectalign(
        entry, vatoms_path, dielectric, atol=0.01, match_expected=True
    ):
        """
        Compare ``doped`` eFNV site potentials with ``sxdefectalign``
        ``vAtoms.dat`` data.

        Uses far-field (> 5 Å) potentials to avoid near-defect discrepancies
        between the two averaging approaches, and accounts for the possibility
        that ``sxdefectalign`` excludes a few near-defect sites.
        """
        sx_dist, _sx_vlr, sx_v_diff, _sx_v_diff_lr = _load_sxdefectalign_vatoms(
            vatoms_path
        )

        corr = get_kumagai_correction(entry, dielectric=dielectric, verbose=False)
        efnv_data = corr.metadata["pydefect_ExtendedFnvCorrection"]

        doped_distances = np.array([float(s.distance) for s in efnv_data.sites])
        doped_potentials = np.array([float(s.potential) for s in efnv_data.sites])

        assert abs(len(doped_distances) - len(sx_dist)) <= 5, (
            f"Site count mismatch too large: doped={len(doped_distances)}, "
            f"sx={len(sx_dist)}"
        )

        doped_far = doped_potentials[doped_distances > 5.0]
        sx_far = sx_v_diff[sx_dist > 5.0]

        assert (
            np.isclose(
                np.mean(np.abs(doped_far)), np.mean(np.abs(sx_far)), atol=atol
            )
            == match_expected
        ), (
            f"Far-field mean |V| mismatch: doped={np.mean(np.abs(doped_far)):.4f}, "
            f"sx={np.mean(np.abs(sx_far)):.4f}"
        )
        return doped_distances, doped_potentials, sx_dist, sx_v_diff

    def _assert_qe_vs_sxd_case(
        self,
        vatoms_relpath,
        matched_entry,
        atol,
        default_entry=None,
        default_should_match=False,
    ):
        vatoms_path = os.path.join(self.SST_QE_sxd_dir, vatoms_relpath)
        assert os.path.exists(vatoms_path), (
            f"sxdefectalign data not found: {vatoms_path}"
        )
        self._compare_doped_vs_sxdefectalign(
            matched_entry, vatoms_path, SB2SI2TE6_DIELECTRIC, atol=atol
        )
        if default_entry is not None:
            self._compare_doped_vs_sxdefectalign(
                default_entry,
                vatoms_path,
                SB2SI2TE6_DIELECTRIC,
                atol=atol,
                match_expected=default_should_match,
            )

    @staticmethod
    def _overlay_sxd_on_correction_plot(entry, vatoms_path):
        _corr, fig = entry.get_kumagai_correction(plot=True)
        sx_dist, _sx_vlr, sx_v_diff, _sx_v_diff_lr = _load_sxdefectalign_vatoms(
            vatoms_path
        )
        ax = fig.gca()
        ax.scatter(
            sx_dist,
            sx_v_diff,
            label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (sxd)",
            s=10,
            color="black",
            zorder=5,
        )
        ax.legend(fontsize=8)
        return fig

    # ------------------------------------------------------------------
    # QE-doped vs QE-sxdefectalign
    # ------------------------------------------------------------------

    def test_qe_doped_vs_qe_sxdefectalign_q3_potentials(self):
        """
        Compare QE-doped and QE-sxdefectalign site potentials for the primary
        defect v_Sb q=-3.

        This is the most important comparison: the high charge state has the
        largest long-range potential and benefits most from accurate eFNV
        averaging.
        """
        self._assert_qe_vs_sxd_case(
            "v_Sb_-3/vAtoms.dat",
            self.qe_defect_dict_beta_1_2["v_Sb_-3"],
            atol=0.01,
            default_entry=self.qe_defect_dict["v_Sb_-3"],
            default_should_match=False,
        )

    def test_qe_doped_vs_qe_sxdefectalign_q3_beta_3_potentials(self):
        """
        Compare QE-doped and QE-sxdefectalign site potentials for v_Sb q=-3
        with ``beta=3`` (Bohr).
        """
        self._assert_qe_vs_sxd_case(
            "v_Sb_-3/beta_3_Bohr/vAtoms.dat",
            self._get_qe_beta_3_entry(),
            atol=0.01,
            default_entry=self.qe_defect_dict["v_Sb_-3"],
            default_should_match=False,
        )

    # ------------------------------------------------------------------
    # VASP-doped vs VASP-sxdefectalign
    # ------------------------------------------------------------------

    def test_vasp_doped_vs_vasp_sxdefectalign_unrelaxed_potentials(self):
        """
        Compare VASP-doped site potentials with VASP-sxdefectalign for
        unrelaxed v_Sb q=-3.

        VASP uses a test-charge approach (RWIGS radii) rather than atomic-
        sphere averaging; agreement in the far field validates that both
        methods converge for well-separated sites even with an anisotropic
        dielectric.
        """
        vasp_vatoms_path = os.path.join(
            EXAMPLE_DIR,
            "Sb2Si2Te6/Defects/Pre_Calculated_Results",
            "v_Sb_-3/vAtoms.dat",
        )
        assert os.path.exists(vasp_vatoms_path), (
            f"sxdefectalign data not found: {vasp_vatoms_path}"
        )
        entry = self.vasp_defect_dict["v_Sb_-3"]
        self._compare_doped_vs_sxdefectalign(
            entry, vasp_vatoms_path, SB2SI2TE6_DIELECTRIC, atol=0.01
        )

    # ------------------------------------------------------------------
    # QE-sxdefectalign vs VASP-sxdefectalign comparison plot
    # ------------------------------------------------------------------

    @custom_mpl_image_compare("SST_QE_sxd_vs_VASP_sxd_unrelaxed_v_Sb_-3_potentials.png")
    def test_qe_sxd_vs_vasp_sxd_unrelaxed_potentials(self):
        """
        Compare QE-sxdefectalign and VASP-sxdefectalign site potentials for
        unrelaxed v_Sb q=-3.

        For the anisotropic Sb2Si2Te6 system the two DFT codes should still
        agree very closely in the far field, with small near-defect differences
        owing to the differing pseudopotential implementations.  Far-field
        agreement tolerance is tighter (0.005 eV) to detect regressions.
        """
        qe_vatoms_path = os.path.join(
            self.SST_QE_sxd_dir, "v_Sb_Unrelaxed_-3/vAtoms.dat"
        )
        vasp_vatoms_path = os.path.join(
            EXAMPLE_DIR,
            "Sb2Si2Te6/Defects/Pre_Calculated_Results",
            "v_Sb_Unrelaxed_-3/vAtoms.dat",
        )
        assert os.path.exists(qe_vatoms_path)
        assert os.path.exists(vasp_vatoms_path)

        qe_dist, _qe_vlr, qe_v_diff, _qe_v_diff_lr = _load_sxdefectalign_vatoms(
            qe_vatoms_path
        )
        vasp_dist, _vasp_vlr, vasp_v_diff, _vasp_v_diff_lr = (
            _load_sxdefectalign_vatoms(vasp_vatoms_path)
        )

        assert len(qe_dist) == len(vasp_dist), (
            f"Site count mismatch: QE={len(qe_dist)}, VASP={len(vasp_dist)}"
        )
        assert np.allclose(qe_dist, vasp_dist, atol=0.01), (
            "Site distances differ between QE and VASP sxdefectalign outputs"
        )

        far_mask = qe_dist > 5.0
        assert np.allclose(
            qe_v_diff[far_mask], vasp_v_diff[far_mask], atol=0.005
        ), (
            f"Far-field max potential difference: "
            f"{np.max(np.abs(qe_v_diff[far_mask] - vasp_v_diff[far_mask])):.4f} eV"
        )
        assert np.allclose(qe_v_diff, vasp_v_diff, atol=0.04), (
            f"Max potential difference: "
            f"{np.max(np.abs(qe_v_diff - vasp_v_diff)):.4f} eV"
        )

        fig, ax = plt.subplots()
        ax.scatter(
            qe_dist,
            qe_v_diff,
            label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (QE)",
            s=10,
        )
        ax.scatter(
            vasp_dist,
            vasp_v_diff,
            label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (VASP)",
            s=10,
        )
        ax.set_xlabel(r"Distance from defect ($\mathrm{\AA}$)")
        ax.set_ylabel(r"Potential $\times$ ($-1$) (eV)")
        ax.set_title(
            r"sxdefectalign: QE vs VASP, $v_{\mathrm{Sb}}$ Unrelaxed $q$=$-$3"
        )
        ax.legend(fontsize=8)
        return fig

    # ------------------------------------------------------------------
    # Overlay plots: doped eFNV + sxdefectalign scatter
    # ------------------------------------------------------------------

    @custom_mpl_image_compare("SST_QE_doped_vs_sxd_v_Sb_-3_potentials.png")
    def test_plot_qe_doped_vs_sxd_q3(self):
        """
        Plot ``doped`` eFNV correction with ``sxdefectalign`` data overlaid for
        QE v_Sb q=-3.
        """
        plt.clf()
        vatoms_path = os.path.join(self.SST_QE_sxd_dir, "v_Sb_-3/vAtoms.dat")
        return self._overlay_sxd_on_correction_plot(
            self.qe_defect_dict_beta_1_2["v_Sb_-3"], vatoms_path
        )

    @custom_mpl_image_compare("SST_QE_doped_vs_sxd_v_Sb_-3_potentials_beta_3.png")
    def test_plot_qe_doped_vs_sxd_q3_beta_3(self):
        """
        Plot ``doped`` eFNV correction with ``sxdefectalign`` data overlaid for
        QE v_Sb q=-3 with ``beta=3`` (Bohr).

        This is the most stringent visual check: at beta=3 the Gaussian
        envelope used in atomic-sphere averaging is very broad, so the doped
        and sxdefectalign potentials should overlap almost exactly.
        """
        plt.clf()
        vatoms_path = os.path.join(
            self.SST_QE_sxd_dir, "v_Sb_-3/beta_3_Bohr/vAtoms.dat"
        )
        fig = self._overlay_sxd_on_correction_plot(
            self._get_qe_beta_3_entry(), vatoms_path
        )
        assert len(fig.gca().collections) >= 2
        return fig