"""
Tests for the `doped.utils.plotting` module, which also implicitly tests some
of the `doped.utils.mappings` and `doped.analysis` modules.

Note that some of the integration tests in `test_analysis.py` also implicitly
tests much of the plotting functionality.

The plotting tests are split into two ``TestCase`` classes:

- ``DefectTransitionLevelPlotsTestCase``: tests ``DefectThermodynamics.plot_transition_levels``
- ``DefectFormationEnergiesPlotsTestCase``: tests ``DefectThermodynamics.plot`` (formation energy diagrams)

Both build on ``DefectThermodynamicsSetupMixin`` (from ``test_thermodynamics``), only loading/defining
extra objects (in ``setUpClass``) where the mixin does not already provide them.
"""

import os
import warnings
from copy import deepcopy

import matplotlib as mpl
import pytest
from matplotlib.colors import ListedColormap
from monty.serialization import loadfn
from test_thermodynamics import DefectThermodynamicsSetupMixin
from test_utils import (
    EXAMPLE_DIR,
    STYLE,
    _print_warning_info,
    _run_func_and_capture_stdout_warnings,
    custom_mpl_image_compare,
    data_dir,
    if_present_rm,
    vasp_data_dir,
)

from doped.analysis import DefectParser
from doped.core import Interstitial, Substitution
from doped.thermodynamics import DefectThermodynamics
from doped.utils import plotting

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


class DefectTransitionLevelPlotsTestCase(DefectThermodynamicsSetupMixin):
    """
    Tests for ``DefectThermodynamics.plot_transition_levels`` (transition level
    diagrams).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # full intrinsic CdTe defect set, used for the ``..._all_intrinsic_...`` TL plots below
        # (except ``test_plot_transition_levels_CdTe_example_groundstate``, which uses the curated
        # example thermo, i.e. the mixin's ``CdTe_defect_thermo``):
        cls.CdTe_intrinsic_thermo = DefectThermodynamics(
            loadfn(os.path.join(data_dir, "CdTe_defect_dict_v2.3.json.gz"))
        )

        # extrinsic Se interstitials (a tough edge case test, with many shallow/unstable and metastable
        # (from essentially duplicated sites) states), derived from the amalgamated extrinsic Se thermo:
        Se_extrinsic_thermo = DefectThermodynamics.from_json(
            os.path.join(EXAMPLE_DIR, "Se", "Se_Amalgamated_Extrinsic_Thermo.json.gz")
        )
        cls.Se_extrinsic_interstitials_thermo = DefectThermodynamics(
            defect_entries=[
                entry
                for entry in Se_extrinsic_thermo.defect_entries.values()
                if isinstance(entry.defect, Interstitial)
            ],
            chempots=Se_extrinsic_thermo.chempots,
        )
        # extrinsic Se substitutions (a less-tough edge case test, with many shallow/unstable but much
        # less metastable states), derived from the amalgamated extrinsic Se thermo:
        cls.Se_extrinsic_substitutions_thermo = DefectThermodynamics(
            defect_entries=[
                entry
                for entry in Se_extrinsic_thermo.defect_entries.values()
                if isinstance(entry.defect, Substitution)
            ],
            chempots=Se_extrinsic_thermo.chempots,
        )

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_faded.png")
    def test_plot_transition_levels_CdTe_faded(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels
        )  # default: faded TLs drawn without labels (cleaner)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_faded_labels.png")
    def test_plot_transition_levels_CdTe_faded_labels(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels, all="faded_labels"
        )  # all="faded_labels": include labels for the faded metastable TLs as well
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_all.png")
    def test_plot_transition_levels_CdTe_all(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels, all=True
        )  # all single-electron (metastable) TLs at full strength
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_subset.png")
    def test_plot_transition_levels_CdTe_defect_subset(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels, defect_subset=["v_"]
        )  # only show defects whose name contains "v_" (i.e. the vacancy columns)
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_ylim.png")
    def test_plot_transition_levels_CdTe_ylim(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels, ylim=(0, 2)
        )  # custom energy-axis limits, shows additional TLs
        assert not w
        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_no_site_info.png")
    def test_plot_transition_levels_CdTe_no_site_info(self):
        # ``include_site_info=False``: never add site info, so the inequivalent ``Cd_i`` columns are
        # disambiguated with "-a"/"-b" suffixes instead of site info
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels, include_site_info=False
        )
        assert not w
        label_txt = [t.get_text() for t in fig.get_axes()[0].texts]
        assert any("-a" in t or "-b" in t for t in label_txt)  # suffix-disambiguated columns present
        assert all("T_{d}" not in t for t in label_txt)  # but no site info added to any column header
        return fig

    @custom_mpl_image_compare(filename="CdTe_example_transition_levels_plot_groundstate.png")
    def test_plot_transition_levels_CdTe_example_groundstate(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.plot_transition_levels, all=False
        )  # the one CdTe TL plot test retained on the curated example thermo; only ground-state TLs
        assert not w
        return fig

    @custom_mpl_image_compare(filename="YTOS_example_transition_levels_plot.png")
    def test_plot_transition_levels_YTOS(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.YTOS_defect_thermo.plot_transition_levels
        )
        assert not w
        return fig

    # Se extrinsic interstitials TL plots; many extrinsic interstitials, shallow/unstable states and
    # metastable charge states (so a tough edge case test):
    @custom_mpl_image_compare(filename="Se_extrinsic_interstitials_transition_levels_plot_faded.png")
    def test_plot_transition_levels_Se_extrinsic_interstitials_faded(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_interstitials_thermo.plot_transition_levels
        )
        assert not w
        return fig

    @custom_mpl_image_compare(filename="Se_extrinsic_interstitials_transition_levels_plot_groundstate.png")
    def test_plot_transition_levels_Se_extrinsic_interstitials_groundstate(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_interstitials_thermo.plot_transition_levels, all=False
        )
        assert not w
        return fig

    @custom_mpl_image_compare(
        filename="Se_extrinsic_interstitials_transition_levels_plot_faded_labels.png"
    )
    def test_plot_transition_levels_Se_extrinsic_interstitials_faded_labels(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_interstitials_thermo.plot_transition_levels, all="faded_labels"
        )  # Note: This gives some partially overlapping TL labels in this plot, basically unavoidable
        assert not w
        return fig

    @custom_mpl_image_compare(
        filename="Se_extrinsic_interstitials_transition_levels_plot_all_no_charge_labels.png"
    )
    def test_plot_transition_levels_Se_extrinsic_interstitials_all_no_charge_labels(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_interstitials_thermo.plot_transition_levels,
            all=True,
            show_charge_labels=False,
        )
        assert not w
        return fig

    @custom_mpl_image_compare(
        filename="Se_extrinsic_interstitials_transition_levels_plot_all_no_band_labels.png"
    )
    def test_plot_transition_levels_Se_extrinsic_interstitials_all_no_band_labels(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_interstitials_thermo.plot_transition_levels,
            all=True,
            show_band_labels=False,
        )
        assert not w
        return fig

    @custom_mpl_image_compare(filename="Se_extrinsic_interstitials_transition_levels_plot_all.png")
    def test_plot_transition_levels_Se_extrinsic_interstitials_all(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_interstitials_thermo.plot_transition_levels, all=True
        )
        assert not w
        return fig

    # Se extrinsic substitutions TL plots; many extrinsic substitutions with shallow/unstable states, but
    # much less metastable states than the extrinsic interstitials (which had essentially duplicated sites)
    @custom_mpl_image_compare(filename="Se_extrinsic_substitutions_transition_levels_plot_faded.png")
    def test_plot_transition_levels_Se_extrinsic_substitutions_faded(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_substitutions_thermo.plot_transition_levels
        )  # default ("faded"): metastable TLs faded without labels
        assert not w
        return fig

    @custom_mpl_image_compare(filename="Se_extrinsic_substitutions_transition_levels_plot_all.png")
    def test_plot_transition_levels_Se_extrinsic_substitutions_all(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_substitutions_thermo.plot_transition_levels, all=True
        )  # all single-electron (metastable) TLs at full strength
        assert not w
        return fig

    @custom_mpl_image_compare(
        filename="Se_extrinsic_substitutions_transition_levels_plot_faded_labels.png"
    )
    def test_plot_transition_levels_Se_extrinsic_substitutions_faded_labels(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_extrinsic_substitutions_thermo.plot_transition_levels, all="faded_labels"
        )  # all="faded_labels": include labels for the faded metastable TLs as well
        assert not w
        return fig

    # ``include_site_info`` for ``plot_transition_levels``; tested here with the pnictogen-in-Se thermo
    # (rather than CdTe, where ``include_site_info`` has no effect as there are no inequivalent sites)
    @custom_mpl_image_compare(filename="Se_pnictogen_transition_levels_plot.png")
    def test_plot_transition_levels_Se_pnict_new_names(self):
        """
        Test ``plot_transition_levels`` with pnictogen impurities in Se, with
        default ``include_site_info=False``.
        """
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_pnict_thermo.plot_transition_levels
        )
        assert not w
        label_txt = [t.get_text() for t in fig.get_axes()[0].texts]
        print(label_txt)
        for i in ["i_{", "{Se_{", "}}$", "-a", "-b"]:
            assert all(i not in text for text in label_txt)

        return fig

    @custom_mpl_image_compare(filename="Se_pnictogen_transition_levels_site_info_plot.png")
    def test_plot_transition_levels_Se_pnict_site_info(self):
        """
        Test ``plot_transition_levels`` with pnictogen impurities in Se, with
        ``include_site_info=True``.
        """
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.Se_pnict_thermo.plot_transition_levels, include_site_info=True
        )
        assert not w
        label_txt = [t.get_text() for t in fig.get_axes()[0].texts]
        print(label_txt)
        # site info being added; C2 for interstitials, none for _Se (only one site):
        defect_labels = [t for t in label_txt if "$" in t]  # defect-name labels (not charge/band labels)
        assert defect_labels  # sanity check that we found the defect-name labels
        assert all(any(i in text for i in ["{i_{C_{2}}}", "_{Se"]) for text in defect_labels)
        assert any("{i_{C_{2}}}" in text for text in defect_labels)  # at least one interstitial w/ C2

        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_all_kwargs.png")
    def test_plot_transition_levels_all_kwargs(self):
        """
        Exercise every keyword argument of ``plot_transition_levels`` at once
        (including a ``prune_to_stable_entries`` kwarg via ``**kwargs`` and
        saving to ``filename``).
        """
        save_path = os.path.join(data_dir, "_test_TLD_all_kwargs_saved.png")
        if_present_rm(save_path)
        try:
            fig, _output, w = _run_func_and_capture_stdout_warnings(
                self.CdTe_intrinsic_thermo.plot_transition_levels,
                all="faded_labels",  # but no unstable entries, so no faded labels to actually show
                defect_subset=["v_", "Te_Cd", "Cd_Te"],
                unstable_entries=False,
                include_site_info=True,  # no effect for CdTe (no inequivalent sites)
                ylim=(-0.2, self.CdTe_intrinsic_thermo.band_gap + 0.2),
                show_charge_labels=True,
                show_band_labels=False,
                label_fontsize=10,
                column_width=0.5,
                figsize=(8, 6),
                style_file=STYLE,
                filename=save_path,
                charge_stability_tolerance=-0.1,  # passed through **kwargs to prune_to_stable_entries
            )
            assert not w
            assert os.path.isfile(save_path)  # figure was saved to ``filename``
        finally:
            if_present_rm(save_path)
        return fig

    @custom_mpl_image_compare(filename="CdTe_all_intrinsic_transition_levels_plot_heatmap_overlay.png")
    def test_plot_transition_levels_modify_output_figure(self):
        """
        Show that the returned ``Figure`` can be further customised, here by
        adding a diverging symmetric heatmap behind the TL diagram which peaks
        in the centre of the band gap and minimises at the band edges (e.g. if
        one wanted to show trends in recombination/emission rates etc.).
        """
        import numpy as np

        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_intrinsic_thermo.plot_transition_levels
        )
        assert not w
        ax = fig.axes[0]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        band_gap = self.CdTe_intrinsic_thermo.band_gap

        # value peaks (=0) at the gap centre and decreases symmetrically towards the band edges:
        y = np.linspace(ylim[0], ylim[1], 256)
        profile = -np.abs(y - band_gap / 2)
        grid = np.tile(profile[:, np.newaxis], (1, 2))  # vary only along the Fermi-level (y) axis
        ax.imshow(
            grid,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            aspect="auto",
            cmap="Reds",
            vmin=-band_gap / 2,  # symmetric about the gap centre
            vmax=band_gap / 2,  # only ever reaches 50% vmax, to keep colours light and avoid greying out
            zorder=-10,  # behind the TL lines/labels
        )
        return fig

    def test_plot_transition_levels_returns_figure(self):
        from matplotlib.figure import Figure

        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.plot_transition_levels
        )
        assert not w
        assert isinstance(fig, Figure)
        # one axes:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        # x-axis ticks hidden, y-label set:
        assert ax.get_xticks().size == 0
        assert ax.get_ylabel() == "Fermi Level (eV)"

    def test_plot_transition_levels_no_charge_labels(self):
        fig, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.plot_transition_levels, show_charge_labels=False
        )
        assert not w
        ax = fig.axes[0]
        # no parenthesised charge labels like "(+1/0)" should remain
        text_strings = [t.get_text() for t in ax.texts]
        assert not any("/" in s and s.startswith("(") for s in text_strings)

    def test_plot_transition_levels_custom_ylim(self):
        # ``ylim`` touching the band edges (VBM at 0, CBM at band_gap) should not emit any warnings
        # (e.g. the singular-transform warning from a zero-height band-edge shaded region):
        band_gap = self.CdTe_defect_thermo.band_gap
        for ylim in [(0.0, 1.0), (0.0, band_gap), (-0.5, band_gap), (0.0, band_gap + 0.5)]:
            fig, _output, w = _run_func_and_capture_stdout_warnings(
                self.CdTe_defect_thermo.plot_transition_levels, show_charge_labels=False, ylim=ylim
            )
            assert not w, f"Unexpected warning(s) for ylim={ylim}: {[str(i.message) for i in w]}"
            ax = fig.axes[0]
            assert ax.get_ylim() == ylim

    def test_plot_transition_levels_invalid_all(self):
        with pytest.raises(ValueError, match="`all` must be False, True, 'faded' or 'faded_labels'"):
            self.CdTe_defect_thermo.plot_transition_levels(all="invalid")

    def test_plot_transition_levels_faded_labels_count(self):
        # default "faded": only solid (ground-state) TLs get labels
        fig_default, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.plot_transition_levels
        )
        assert not w
        # all="faded_labels": all TLs (including metastable) get labels
        fig_with_faded, _output, w = _run_func_and_capture_stdout_warnings(
            self.CdTe_defect_thermo.plot_transition_levels, all="faded_labels"
        )
        assert not w

        def n_charge_labels(fig):
            return sum(
                1 for t in fig.axes[0].texts if "/" in t.get_text() and t.get_text().startswith("(")
            )

        assert n_charge_labels(fig_with_faded) > n_charge_labels(fig_default)

    def test_plot_transition_levels_empty_subset(self):
        with pytest.raises(ValueError, match="No defects with transition levels"):
            self.CdTe_defect_thermo.plot_transition_levels(defect_subset=["non_existent_defect"])

    def test_plot_transition_levels_defect_subset_string(self):
        # bare string should be treated as single-element list
        fig_list = self.CdTe_defect_thermo.plot_transition_levels(defect_subset=["v_"])
        fig_str = self.CdTe_defect_thermo.plot_transition_levels(defect_subset="v_")
        # both should produce the same single-column plot:
        assert fig_list.axes[0].get_xlim() == fig_str.axes[0].get_xlim()

    def test_transition_level_diagram_data_helpers(self):
        from doped.utils.plotting import _format_TL_charge_label, _get_transition_level_data

        assert _format_TL_charge_label((1, 0)) == "(+1/0)"
        assert _format_TL_charge_label((1, 0), pos_meta=True) == "(+1*/0)"
        assert _format_TL_charge_label((-1, -2), neg_meta=True) == "(-1/-2*)"
        assert _format_TL_charge_label((2, -3)) == "(+2/-3)"

        ground = _get_transition_level_data(self.CdTe_defect_thermo, all=False)
        all_data = _get_transition_level_data(self.CdTe_defect_thermo, all=True)
        faded = _get_transition_level_data(self.CdTe_defect_thermo, all="faded")
        # each entry is a 5-tuple (TL_eV, charges, pos_meta, neg_meta, faded):
        for tls in faded.values():
            for tl in tls:
                assert len(tl) == 5
        # ground-state TLs are never faded:
        for tls in ground.values():
            assert all(not tl[4] for tl in tls)
        # all=True: also never faded
        for tls in all_data.values():
            assert all(not tl[4] for tl in tls)
        # "faded" mode should include all ground-state TLs as not-faded:
        for name, gs_tls in ground.items():
            non_faded_in_faded = [tl for tl in faded.get(name, []) if not tl[4]]
            assert len(non_faded_in_faded) >= len(gs_tls)
        # each entry is sorted by TL energy:
        for tls in all_data.values():
            assert tls == sorted(tls, key=lambda x: x[0])
        for tls in faded.values():
            assert tls == sorted(tls, key=lambda x: x[0])


class DefectFormationEnergiesPlotsTestCase(DefectThermodynamicsSetupMixin):
    """
    Tests for ``DefectThermodynamics.plot`` (defect formation energy diagrams).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.CdTe_LZ_defect_dict = loadfn(
            os.path.join(data_dir, "CdTe_LZ_defect_dict_v2.3_wout_meta.json.gz")
        )
        cls.orig_CdTe_LZ_thermo_wout_meta = DefectThermodynamics(
            cls.CdTe_LZ_defect_dict, chempots=cls.CdTe_chempots
        )

        cls.orig_Se_amalgamated_extrinsic_thermo = DefectThermodynamics.from_json(
            f"{EXAMPLE_DIR}/Se/Se_Amalgamated_Extrinsic_Thermo.json.gz"
        )
        cls.orig_Bi_Se_thermo = loadfn(f"{data_dir}/Bi_Se_BiSeI_thermo.json.gz")

        # constant colormap/linestyles for the extrinsic Se ``linestyles``/``colormap`` tests
        # (defined here rather than in ``setUp`` as they are not mutated per-test):
        # H, N, P, As, Sb, O, S, Te, F, Cl, Br:
        colors = mpl.colormaps.get("Dark2").colors
        H_color = colors[4]
        pnict_color = colors[2]
        chalc_color = colors[1]
        halogen_color = colors[0]
        cls.H_pnict_chalc_halogen_colormap = ListedColormap(
            [
                H_color,
                *[(*pnict_color, 1 - 0.2 * i) for i in range(4)],
                *[(*chalc_color, 1 - 0.3 * i) for i in range(3)],
                *[(*halogen_color, 1 - 0.3 * i) for i in range(3)],
            ]
        )
        # solid for first of each group, then dashed, dotted, double dash:
        cls.Se_ext_linestyles = ["-", "-", "--", ":", "-.", "-", "--", ":", "-", "--", ":"]

    def setUp(self):
        super().setUp()
        # per-test deep-copies of objects that some tests mutate (e.g. ``dist_tol``):
        self.CdTe_LZ_thermo_wout_meta = deepcopy(self.orig_CdTe_LZ_thermo_wout_meta)
        self.Se_amalgamated_extrinsic_thermo = deepcopy(self.orig_Se_amalgamated_extrinsic_thermo)
        self.Bi_Se_thermo = deepcopy(self.orig_Bi_Se_thermo)

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_plot_CdTe(self):
        return self.CdTe_defect_thermo.plot(self.CdTe_chempots, limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_listed_colormap.png")
    def test_plot_CdTe_custom_listed_colormap(self):
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["#D4447E", "#E9A66C", "#507BAA", "#5FABA2", "#63666A"])
        return self.CdTe_defect_thermo.plot(self.CdTe_chempots, colormap=cmap)[1]

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_viridis.png")
    def test_plot_CdTe_custom_str_colormap(self):
        return self.CdTe_defect_thermo.plot(self.CdTe_chempots, colormap="viridis")[1]

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_plot_CdTe_multiple_figs_no_chempot_table(self):
        # when limits not specified, plots all of them (second should be Te-rich here)
        # chempot_table true by default with multiple figures, test setting to False
        fig = self.CdTe_defect_thermo.plot(self.CdTe_chempots, chempot_table=False)[1]
        fig.gca().set_title("")  # remove the title by setting it to an empty string
        return fig

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_Cd_rich.png")
    def test_plot_CdTe_Cd_rich(self):
        # when limits not specified, plots all of them (first should be Cd-rich here)
        fig = self.CdTe_defect_thermo.plot(self.CdTe_chempots)[0]
        fig.gca().set_title("")  # remove the title by setting it to an empty string
        return fig

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_plot_YTOS(self):
        with warnings.catch_warnings(record=True) as w:
            plot = self.YTOS_defect_thermo.plot()  # no chempots
        _print_warning_info(w)  # for debugging
        assert len(w) == 2
        assert any("All formation energies for" in str(warn.message) for warn in w)
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        return plot

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_subset.png")
    def test_plot_CdTe_defect_subset(self):
        # subset filter on the standard formation-energy plot
        return self.CdTe_defect_thermo.plot(
            self.CdTe_chempots, limit="CdTe-Te", defect_subset=["v_", "Te_Cd"]
        )

    def test_plot_defect_subset_filters_lines(self):
        # standard formation energy plot with subset filter -- check legend is reduced
        fig_full = self.CdTe_defect_thermo.plot(self.CdTe_chempots, limit="CdTe-Te")
        fig_subset = self.CdTe_defect_thermo.plot(
            self.CdTe_chempots, limit="CdTe-Te", defect_subset=["v_"]
        )
        # legend should be smaller after filtering:
        full_legend_n = len(fig_full.axes[0].get_legend().get_texts())
        subset_legend_n = len(fig_subset.axes[0].get_legend().get_texts())
        assert subset_legend_n < full_legend_n
        assert subset_legend_n >= 1  # at least v_Cd

    def test_plot_limit_no_chempots_error(self):
        with pytest.raises(ValueError) as exc:
            self.CdTe_defect_thermo.plot(limit="Te-rich")
        assert (
            "You have specified a chemical potential limit, but no `chempots` have been supplied, "
            "so `limit` cannot be used here!"
        ) in str(exc.value)

    def test_plot_limit_user_chempots_error(self):
        with pytest.raises(ValueError) as exc:
            self.CdTe_defect_thermo.plot(chempots={"Cd": 0.5, "Te": -0.5}, limit="Te-rich")
        assert (
            "You have specified a chemical potential limit, but the supplied chempots are not in the "
            "doped format (i.e. with `limits` in the chempots dict), and instead correspond to just a "
            "single chemical potential limit, so `limit` cannot be used here!"
        ) in str(exc.value)

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_no_chempots.png")
    def test_default_CdTe_plot_no_chempots(self):
        with warnings.catch_warnings(record=True) as w:
            plot = self.CdTe_defect_thermo.plot()
        _print_warning_info(w)  # for debugging
        assert len(w) == 2  # formation energy warning for Int_Te_3, and no chempots
        assert any("All formation energies for" in str(warn.message) for warn in w)
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        return plot

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_explicit_limit(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_specified_chempots(self):
        return self.CdTe_defect_thermo.plot(chempots=self.CdTe_chempots, limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_manual_chempots(self):
        return self.CdTe_defect_thermo.plot(
            chempots={"Cd": -1.2513, "Te": 0}, el_refs=self.CdTe_chempots["elemental_refs"]
        )

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_manual_chempots_at_init(self):
        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()),
            chempots={"Cd": -1.2513, "Te": 0},
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        return defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1(self):
        self.CdTe_defect_thermo.chempots = {"Cd": -1.2513, "Te": 0}
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.chempots = {"Cd": -1.2513, "Te": 0}
        return self.CdTe_defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other2(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot(chempots={"Cd": -1.2513, "Te": 0})

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other3(self):
        self.CdTe_defect_thermo.chempots = {"Cd": -1.2513, "Te": 0}
        return self.CdTe_defect_thermo.plot(el_refs=self.CdTe_chempots["elemental_refs"])

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_edited_el_refs(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_edited_el_refs_other(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_Cd_rich.png")
    def test_default_CdTe_plot_Cd_rich(self):
        return self.CdTe_defect_thermo.plot(
            chempots=self.CdTe_chempots, limit="Cd-rich", chempot_table=True
        )

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_plot_neutral_v_O_V2O5(self):
        """
        Test plotting V2O5 defects, FNV correction.
        """
        dielectric = [4.186, 19.33, 17.49]
        bulk_path = f"{vasp_data_dir}/V2O5/V2O5_bulk"
        chempots = loadfn(f"{vasp_data_dir}/V2O5/chempots.json")

        defect_dict = {
            defect: DefectParser.from_paths(
                f"{vasp_data_dir}/V2O5/{defect}", bulk_path, dielectric=dielectric
            ).defect_entry
            for defect in [dir for dir in os.listdir(f"{vasp_data_dir}/V2O5") if "v_O" in dir]
        }  # charge auto-determined (as neutral)
        thermo = DefectThermodynamics(list(defect_dict.values()))
        return thermo.plot(chempots, limit="V2O5-O2")

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_V2O5_O_rich_plot(self):
        return self.V2O5_defect_thermo.plot(limit="O-rich")

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_V2O5_O_rich_plot_reduced_dist_tol(self):
        self.V2O5_defect_thermo.dist_tol = 1e-4
        # still two neutral vacancies merged as their site is the exact same
        return self.V2O5_defect_thermo.plot(limit="O-rich")

    @custom_mpl_image_compare(filename="neutral_v_O_plot_all_entries.png")
    def test_V2O5_O_rich_all_entries_plot(self):
        return self.V2O5_defect_thermo.plot(limit="O-rich", all_entries=True)

    @custom_mpl_image_compare(filename="neutral_v_O_plot_faded.png")
    def test_V2O5_O_rich_faded_plot(self):
        return self.V2O5_defect_thermo.plot(limit="O-rich", all_entries="faded")

    @custom_mpl_image_compare(filename="CdTe_LZ_all_default_Te_rich.png")
    def test_CdTe_LZ_all_defects_plot(self):
        return self.CdTe_LZ_thermo_wout_meta.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_LZ_all_Te_rich_dist_tol_1pt6.png")
    def test_CdTe_LZ_all_defects_plot_dist_tol_1pt6(self):
        # Matches SK Thesis Fig 6.1b
        lz_cdte_defect_thermo = self.CdTe_LZ_thermo_wout_meta
        lz_cdte_defect_thermo.dist_tol = 1.6  # increase to 1.6 Å to merge Te_i defects

        return lz_cdte_defect_thermo.plot(chempots=self.CdTe_chempots, limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_LZ_all_Te_rich_dist_tol_3.png")
    def test_CdTe_LZ_all_defects_plot_dist_tol_3(self):
        lz_cdte_defect_thermo = self.CdTe_LZ_thermo_wout_meta
        lz_cdte_defect_thermo.dist_tol = 3  # increase to 3 Å to merge Te_i and Cd_i defects

        return lz_cdte_defect_thermo.plot(chempots=self.CdTe_chempots, limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_LZ_all_no_site_info.png")
    def test_plot_CdTe_LZ_all_no_site_info(self):
        # uses "-a", "-b" etc. to disambiguate interstitial sites, rather than site info:
        return self.CdTe_LZ_thermo_wout_meta.plot(
            self.CdTe_chempots, limit="CdTe-Te", include_site_info=False
        )

    @custom_mpl_image_compare(filename="CdTe_FNV_all_default_Te_rich_old_names.png")
    def test_CdTe_FNV_all_defects_plot_default_old_names(self):
        # Tests naming handling when old/unrecognised defect names are used
        cdte_defect_dict = loadfn(os.path.join(data_dir, "CdTe_defect_dict_old_names.json.gz"))
        cdte_defect_thermo = DefectThermodynamics(cdte_defect_dict)
        with warnings.catch_warnings(record=True) as w:
            plot = cdte_defect_thermo.plot()
        _print_warning_info(w)  # for debugging
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        assert any(
            "All formation energies for as_1_Te_on_Cd are below zero across the entire band gap range. "
            "This is typically unphysical (see docs), and likely due to mis-specification of chemical "
            "potentials (see docstrings and/or tutorials)." in str(warn.message)
            for warn in w
        )
        assert any("All formation energies for Int_Te_3 are below zero" in str(warn.message) for warn in w)
        assert len(w) == 3
        return plot

    @custom_mpl_image_compare(filename="Sb2O5_default_TLD.png")
    def test_Sb2O5_default(self):
        print(self.Sb2O5_defect_thermo.transition_level_map)
        with warnings.catch_warnings(record=True) as w:
            plot = self.Sb2O5_defect_thermo.plot(limit="O-poor")
        _print_warning_info(w)  # for debugging
        assert not w
        return plot

    @custom_mpl_image_compare(filename="Sb2O5_merged_dist_tol_TLD.png")
    def test_Sb2O5_merged_dist_tol(self):
        print(self.Sb2O5_defect_thermo.transition_level_map)
        with warnings.catch_warnings(record=True) as w:
            self.Sb2O5_defect_thermo.dist_tol = 10
            print(self.Sb2O5_defect_thermo.transition_level_map)
            plot = self.Sb2O5_defect_thermo.plot(limit="O-poor")
        _print_warning_info(w)  # for debugging
        assert not w
        return plot

    @custom_mpl_image_compare(filename="CdTe_duplicate_entry_names.png")
    def test_handling_duplicate_entry_names_CdTe(self):
        """
        Test renaming behaviour when defect entries with the same names are
        provided.
        """
        defect_dict = loadfn(os.path.join(data_dir, "CdTe_defect_dict_v2.3.json.gz"))
        num_entries = len(defect_dict)
        for defect_entry in defect_dict.values():
            if "Cd_i" in defect_entry.name:
                defect_entry.name = f"Cd_i_{defect_entry.charge_state}"
            if "Te_i" in defect_entry.name:
                defect_entry.name = f"Te_i_{defect_entry.charge_state}"

        defect_thermo = DefectThermodynamics(defect_dict)
        assert len(defect_thermo.defect_entries) == num_entries
        print([entry.name for entry in defect_thermo.defect_entries.values()])

        return defect_thermo.plot(self.CdTe_chempots, limit="Cd-rich")

    @custom_mpl_image_compare(filename="Se_pnictogen_plot.png")
    def test_plotting_ext_Se_new_names(self):
        """
        Test plotting behaviour with pnictogen impurities in Se.

        Previously this would append site info to the defect names by default,
        but this is not necessary here as there are no inequivalent
        substitution/interstitial sites, so check that the legend is as
        expected with no site info.
        """
        fig = self.Se_pnict_thermo.plot()
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        for i in ["i_{", "{Se_{", "}}$", "-a", "-b"]:
            assert all(i not in text for text in legend_txt)

        return fig

    @custom_mpl_image_compare(filename="Se_site_info_old_names_plot.png")
    def test_ext_Se_old_names_include_site_info(self):
        """
        Test plotting extrinsic defects in Se with ``include_site_info=True``.

        In this case, the defect folder/entry names match the old ``doped``
        format, with e.g. ``sub_1_Br_on_Se_-1`` and ``inter_2_O_0`` etc.

        We have some duplicates (``inter_1_O_0`` and ``inter_2_O_0``) which
        are removed by including site info, but some (``inter_11_H_X``) which
        aren't, so ``_a`` & ``_b`` are appended to those names.
        """
        fig = self.Se_ext_no_pnict_thermo.plot(include_site_info=True)
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        for i in [
            "O$_{i_{2}}$",
            "Te$_{Se_{1}}$",
            "H$_{i_{11}}$",
            "H$_{i_{12}}$",
            "F$_{i_{1}}$",
        ]:
            assert i in legend_txt

        return fig

    @custom_mpl_image_compare(filename="Se_pnictogen_site_info_plot.png")
    def test_plotting_ext_Se_site_info(self):
        """
        Test plotting behaviour with pnictogen impurities in Se, with
        ``include_site_info=True``.
        """
        fig = self.Se_pnict_thermo.plot(include_site_info=True)
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        # site info being added; C2 for interstitials, none for _Se (only one site)
        assert all(any(i in text for i in ["{C_{2}}}", "_{Se"]) for text in legend_txt)

        return fig

    @custom_mpl_image_compare(filename="Se_ext_no_pnict_thermo_unstable_entries_True.png")
    def test_unstable_entries_True(self):
        """
        Test plotting behaviour with non-pnictogen impurities in Se, with
        ``unstable_entries=True`` (plot all entries regardless of
        stability/shallowness).
        """
        return self.Se_ext_no_pnict_thermo.plot(unstable_entries=True)

    @custom_mpl_image_compare(filename="Se_ext_no_pnict_thermo_unstable_entries_default.png")
    def test_unstable_entries_default(self):
        """
        Test plotting behaviour with non-pnictogen impurities in Se, with
        default ``unstable_entries`` ("not shallow").
        """
        return self.Se_ext_no_pnict_thermo.plot()  # unstable_entries="not shallow" by default

    @custom_mpl_image_compare(filename="Se_ext_no_pnict_thermo_unstable_entries_default.png")
    def test_unstable_entries_not_shallow(self):
        """
        Test plotting behaviour with non-pnictogen impurities in Se, with
        ``unstable_entries="not shallow"`` (same as above, checking explicit
        choice).
        """
        return self.Se_ext_no_pnict_thermo.plot(unstable_entries="not shallow")

    @custom_mpl_image_compare(filename="Se_ext_no_pnict_thermo_unstable_entries_False.png")
    def test_unstable_entries_False(self):
        """
        Test plotting behaviour with non-pnictogen impurities in Se, with
        ``unstable_entries=False`` (no shallow within default tol, and no non
        in-gap stable).

        Here we also test the renaming behaviour when defect clusters with the
        same (representative) name occur.
        """
        fig = self.Se_ext_no_pnict_thermo.plot(unstable_entries=False, ylim=(0, 2.5))
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        for i in [
            "O$_i$",
            "H$_i$$_{-a}$",
            "H$_i$$_{-b}$",
            "Br$_i$",
            "F$_i$",
        ]:
            assert i in legend_txt

        return fig

    def test_unstable_entries_error(self):
        """
        Test error with ``unstable_entries="wrong input"``.
        """
        with pytest.raises(ValueError) as e:
            self.Se_ext_no_pnict_thermo.plot(unstable_entries="wrong input")
        assert (
            "`unstable_entries` option must be either True, False or 'not shallow' -- not wrong input"
            in str(e.value)
        )

    @custom_mpl_image_compare(
        filename="Se_ext_no_pnict_thermo_unstable_entries_charge_stability_kwarg.png"
    )
    def test_unstable_entries_charge_stability_kwarg(self):
        """
        Test plotting behaviour with non-pnictogen impurities in Se, with
        ``unstable_entries=False`` and ``charge_stability_tolerance=0.4``.
        """
        return self.Se_ext_no_pnict_thermo.plot(
            unstable_entries=False, ylim=(0, 2.5), charge_stability_tolerance=0.4
        )

    @custom_mpl_image_compare(filename="Se_ext_no_pnict_thermo_unstable_entries_True.png")
    def test_unstable_entries_shallow_charge_stability_kwarg(self):
        """
        Test plotting behaviour with non-pnictogen impurities in Se, with
        ``shallow_charge_stability_tolerance=-0.2`` -> here equivalent to
        ``unstable_entries=True``.
        """
        return self.Se_ext_no_pnict_thermo.plot(shallow_charge_stability_tolerance=-0.2)

    @custom_mpl_image_compare(filename="Se_extrinsic_interstitials_linestyles_plot.png")
    def test_plotting_linestyles_and_colors_ext_Se(self):
        """
        Test plotting extrinsic interstitials in Se, using the ``linestyles``
        and ``colormap`` (with ``ListedColormap``) options.
        """
        from doped.core import Interstitial

        amalgamated_Se_extrinsic_interstitials_thermo = DefectThermodynamics(
            defect_entries=[
                entry
                for entry in self.Se_amalgamated_extrinsic_thermo.defect_entries.values()
                if isinstance(entry.defect, Interstitial)
            ],
            chempots=self.Se_amalgamated_extrinsic_thermo.chempots,
        )
        amalgamated_Se_extrinsic_interstitials_thermo.dist_tol = 2  # amalgamate Hi and Oi

        return amalgamated_Se_extrinsic_interstitials_thermo.plot(
            chempot_table=False,
            colormap=self.H_pnict_chalc_halogen_colormap,
            linestyles=self.Se_ext_linestyles,
        )

    @custom_mpl_image_compare(filename="Se_extrinsic_substitutions_linestyles_plot.png")
    def test_plotting_linestyles_and_colors_ext_Se_substitutions(self):
        """
        Test plotting extrinsic substitutions in Se, using the ``linestyles``
        and ``colormap`` (with ``ListedColormap``) options.

        Previously there was an issue where the extrinsic element wasn't being
        correctly parsed for substitution Defects (as ``Defect.defect_site``
        confusingly returns the original element, not the substitution
        element), so we test the correct behaviour here.
        """
        from doped.core import Substitution

        amalgamated_Se_extrinsic_substitutions_thermo = DefectThermodynamics(
            defect_entries=[
                entry
                for entry in self.Se_amalgamated_extrinsic_thermo.defect_entries.values()
                if isinstance(entry.defect, Substitution)
            ],
            chempots=self.Se_amalgamated_extrinsic_thermo.chempots,
        )

        return amalgamated_Se_extrinsic_substitutions_thermo.plot(
            chempot_table=False,
            colormap=self.H_pnict_chalc_halogen_colormap,
            linestyles=self.Se_ext_linestyles,
        )

    @custom_mpl_image_compare(filename="CdTe_LZ_all_default_Te_rich.png")
    def test_CdTe_LZ_site_info_plot(self):
        """
        Test CdTe plotting behaviour with ``include_site_info=True``.
        """
        return self.CdTe_LZ_thermo_wout_meta.plot(limit="Te-rich", include_site_info=True)

    @custom_mpl_image_compare(filename="Bi_Se_BiSeI_Se_rich.png")
    def test_Bi_Se_all_entries_plot(self):
        """
        Test plotting all entries in Bi_Se thermo.

        Previously had issue where the colours did not match the appropriate
        lines.
        """
        return self.Bi_Se_thermo.plot(limit="Se-rich", all_entries=True)

    @custom_mpl_image_compare(filename="Bi_Se_BiSeI_Se_rich.png")
    def test_Bi_Se_all_entries_manual_colormap_plot(self):
        from doped.utils.plotting import ListedColormap, get_colormap

        cmap = get_colormap("tab10")
        cmap = ListedColormap([(*color, 0.75) for color in cmap.colors])
        return self.Bi_Se_thermo.plot(limit="Se-rich", all_entries=True, colormap=cmap)

    @custom_mpl_image_compare(filename="Bi_Se_BiSeI_Se_rich.png")
    def test_Bi_Se_all_entries_unrecognised_colormap_plot(self):
        with warnings.catch_warnings(record=True) as w:
            fig = self.Bi_Se_thermo.plot(limit="Se-rich", all_entries=True, colormap="Well shiii")
        _print_warning_info(w)
        assert any("Colormap 'Well shiii' not found in" in str(warn.message) for warn in w)
        assert any("Defaulting to 'tab10' colormap" in str(warn.message) for warn in w)
        return fig

    def test_format_defect_name(self):
        """
        Test ``format_defect_name()`` function.
        """
        # test standard behaviour
        formatted_name = plotting.format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_info=False,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd}^{0}$"
        # test with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_info=True,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd_{1}}^{0}$"

        # test interstitial case with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_info=False,
        )
        assert formatted_name == "Cd$_i^{0}$"
        # test interstitial case with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_info=True,
        )
        assert formatted_name == "Cd$_{i_{1}}^{0}$"

        # test lowercase interstitial with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="int_Cd_1_0",
            include_site_info=False,
        )
        assert formatted_name == "Cd$_i^{0}$"
        # test lowercase interstitial with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="int_Cd_1_0",
            include_site_info=True,
        )
        assert formatted_name == "Cd$_{i_{1}}^{0}$"

        # test uppercase vacancy (pymatgen default name) with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="Vac_1_Cd_0",
            include_site_info=False,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd}^{0}$"
        # test uppercase vacancy (pymatgen default name) with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="Vac_1_Cd_0",
            include_site_info=True,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd_{1}}^{0}$"

        # test substitution with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_Ni_on_Li_0",
            include_site_info=False,
        )
        assert formatted_name == "Ni$_{Li}^{0}$"

        # test substitution with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_Ni_on_Li_0",
            include_site_info=True,
        )
        assert formatted_name == "Ni$_{Li_{1}}^{0}$"

        # test substitution with site number excluded, current doped format, two-letter subbed element
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_P_on_Na_-1",
            include_site_info=False,
        )
        assert formatted_name == "P$_{Na}^{-1}$"

        # test substitution with site number included, current doped format, two-letter subbed element
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_P_on_Na_-1 ",
            include_site_info=True,
        )
        assert formatted_name == "P$_{Na_{1}}^{-1}$"

        # test substitution with site number excluded, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="as_2_Na_on_P_0",
            include_site_info=False,
        )
        assert formatted_name == "Na$_{P}^{0}$"

        # test substitution with site number included, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="as_2_Na_on_P_0",
            include_site_info=True,
        )
        assert formatted_name == "Na$_{P_{2}}^{0}$"

        # test interstitial with site number excluded, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="inter_12_P_0",
            include_site_info=False,
        )
        assert formatted_name == "P$_i^{0}$"

        # test interstitial with site number included, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="inter_12_P_0",
            include_site_info=True,
        )
        assert formatted_name == "P$_{i_{12}}^{0}$"

        # test vacancy with site number excluded, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="vac_4_P_-2",
            include_site_info=False,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{P}^{-2}$"

        # test vacancy with site number included, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="vac_4_P_-2",
            include_site_info=True,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{P_{4}}^{-2}$"

        # check exceptions raised: invalid charge or defect_species
        # test error catching:
        wrong_charge_error = ValueError(
            "Problem reading defect name vac_1_Cd_a, should end with charge state after underscore"
        )
        with pytest.raises(ValueError) as e:
            plotting.format_defect_name(defect_species="vac_1_Cd_a", include_site_info=True)
        assert str(wrong_charge_error) in str(e.value)

        with pytest.raises(TypeError) as e:
            plotting.format_defect_name(defect_species=2, include_site_info=True)
        # check invalid defect type returns None
        assert plotting.format_defect_name(defect_species="kk_Cd_1_0", include_site_info=True) is None

        defect_species_name_dict = {
            "vac_Cd_mult32_0": "$\\it{V}\\!$ $_{Cd}^{0}$",
            "VSb_0": "$\\it{V}\\!$ $_{Sb}^{0}$",
            "VI_9": "$\\it{V}\\!$ $_{I}^{+9}$",
            "VCd_0": "$\\it{V}\\!$ $_{Cd}^{0}$",
            "V_Sb_0": "V$_{Sb}^{0}$",
            "V_I,_-2": "V$_{I}^{-2}$",
            "V_I_-2": "V$_{I}^{-2}$",
            "VacSb_2": "$\\it{V}\\!$ $_{Sb}^{+2}$",
            "VacI_2": "$\\it{V}\\!$ $_{I}^{+2}$",
            "Vac_Sb_3": "$\\it{V}\\!$ $_{Sb}^{+3}$",
            "Vac_I_1": "$\\it{V}\\!$ $_{I}^{+1}$",
            "VaSb_3": "$\\it{V}\\!$ $_{Sb}^{+3}$",
            "VaI_9": "$\\it{V}\\!$ $_{I}^{+9}$",
            "Va_Sb_10": "$\\it{V}\\!$ $_{Sb}^{+10}$",
            "Va_I_4": "$\\it{V}\\!$ $_{I}^{+4}$",
            "V_Cd_0": "V$_{Cd}^{0}$",  # V substitution on Cd
            "V_on_Cd_0": "V$_{Cd}^{0}$",
            "V_O_2": "V$_{O}^{+2}$",
            "V_S_-1": "V$_{S}^{-1}$",
            "V_i_0": "V$_i^{0}$",  # V interstitial
            "Int_V_1_2": "V$_i^{+2}$",
            "inter_3_V_-1": "V$_i^{-1}$",
            "iV_0": "V$_i^{0}$",
            "Cd_V_0": "Cd$_{V}^{0}$",  # substitution/antisite on a vanadium site
            "Cd_on_V_0": "Cd$_{V}^{0}$",
            "Te_V_-2": "Te$_{V}^{-2}$",
            "v_V_0": "$\\it{V}\\!$ $_{V}^{0}$",  # vacancy *of* vanadium
            "Vac_V_1": "$\\it{V}\\!$ $_{V}^{+1}$",
            "vac_1_V_0": "$\\it{V}\\!$ $_{V}^{0}$",
            "i_Sb_1": "Sb$_i^{+1}$",
            "Sb_i_3": "Sb$_i^{+3}$",
            "iSb_8": "Sb$_i^{+8}$",
            "IntSb_2": "Sb$_i^{+2}$",
            "Int_Sb_9": "Sb$_i^{+9}$",
            "Sb_Se_9": "Sb$_{Se}^{+9}$",
            "Sb_on_Se_9": "Sb$_{Se}^{+9}$",
            "Int_Li_mult64_-1": "Li$_i^{-1}$",
            "Int_Li_mult64_-2": "Li$_i^{-2}$",
            "Int_Li_mult64_0": "Li$_i^{0}$",
            "Int_Li_mult64_1": "Li$_i^{+1}$",
            "Int_Li_mult64_2": "Li$_i^{+2}$",
            "Sub_Li_on_Ni_mult32_-1": "Li$_{Ni}^{-1}$",
            "Sub_Li_on_Ni_mult32_-2": "Li$_{Ni}^{-2}$",
            "Sub_Li_on_Ni_mult32_0": "Li$_{Ni}^{0}$",
            "Sub_Li_on_Ni_mult32_+1": "Li$_{Ni}^{+1}$",
            "Sub_Li_on_Ni_mult32_2": "Li$_{Ni}^{+2}$",
            "Sub_Ni_on_Li_mult32_-1": "Ni$_{Li}^{-1}$",
            "Sub_Ni_on_Li_mult32_-2": "Ni$_{Li}^{-2}$",
            "Sub_Ni_on_Li_mult32_0": "Ni$_{Li}^{0}$",
            "Sub_Ni_on_Li_mult32_+1": "Ni$_{Li}^{+1}$",
            "Sub_Ni_on_Li_mult32_2": "Ni$_{Li}^{+2}$",
            "Vac_Li_mult32_-1": "$\\it{V}\\!$ $_{Li}^{-1}$",
            "Vac_Li_mult32_-2": "$\\it{V}\\!$ $_{Li}^{-2}$",
            "Vac_Li_mult32_0": "$\\it{V}\\!$ $_{Li}^{0}$",
            "Vac_Li_mult32_+1": "$\\it{V}\\!$ $_{Li}^{+1}$",
            "v_Cd_-1": "$\\it{V}\\!$ $_{Cd}^{-1}$",
            "v_Te_+2": "$\\it{V}\\!$ $_{Te}^{+2}$",
            "Cd_i_C3v_+2": "Cd$_i^{+2}$",
            "Cd_i_m32_2": "Cd$_i^{+2}$",
            "Cd_i_Td_Cd2.83_+2": "Cd$_i^{+2}$",
            "Cd_i_Td_Te2.83_2": "Cd$_i^{+2}$",
            "Te_i_C3vb_-2": "Te$_i^{-2}$",
            "Te_Cd_s32_2": "Te$_{Cd}^{+2}$",
            "Te_Cd_s32c_2": "Te$_{Cd}^{+2}$",
            "Cd_Te_+2": "Cd$_{Te}^{+2}$",
            "Cd_Te_2": "Cd$_{Te}^{+2}$",
            "as_2_Bi_on_O_-2": "Bi$_{O}^{-2}$",
            "S_Se_0": "S$_{Se}^{0}$",
            "Se_S_0": "Se$_{S}^{0}$",
            "Si_S_0": "Si$_{S}^{0}$",
            "S_Si_0": "S$_{Si}^{0}$",
            "inter_8_O_-1": "O$_i^{-1}$",
            "inter_14_O_0": "O$_i^{0}$",
            "inter_14_O_+2": "O$_i^{+2}$",
            "inter_14_Th_+2": "Th$_i^{+2}$",
            "vac_14_Th_+2": "$\\it{V}\\!$ $_{Th}^{+2}$",
            "As_i_C2_-3": "As$_i^{-3}$",
            "Ag_i_C3v_Ag2.40_0": "Ag$_i^{0}$",
            "Ga_O_C1_Ga1.93Ga2.08O2.62aas_0": "Ga$_{O}^{0}$",
        }

        for defect_species, expected_name in defect_species_name_dict.items():
            formatted_name = plotting.format_defect_name(
                defect_species=defect_species,
                include_site_info=False,
            )
            assert formatted_name == expected_name

        defect_species_w_site_info_name_dict = {
            "vac_Cd_mult32_0": "$\\it{V}\\!$ $_{Cd_{m32}}^{0}$",
            "Int_Li_mult64_-1": "Li$_{i_{m64}}^{-1}$",
            "Int_Li_mult64_-2": "Li$_{i_{m64}}^{-2}$",
            "Int_Li_mult64_0": "Li$_{i_{m64}}^{0}$",
            "Int_Li_mult64_1": "Li$_{i_{m64}}^{+1}$",
            "Int_Li_mult64_2": "Li$_{i_{m64}}^{+2}$",
            "Sub_Li_on_Ni_mult32_-1": "Li$_{Ni_{m32}}^{-1}$",
            "Sub_Li_on_Ni_mult32_-2": "Li$_{Ni_{m32}}^{-2}$",
            "Sub_Li_on_Ni_mult32_0": "Li$_{Ni_{m32}}^{0}$",
            "Sub_Li_on_Ni_mult32_+1": "Li$_{Ni_{m32}}^{+1}$",
            "Sub_Li_on_Ni_mult32_2": "Li$_{Ni_{m32}}^{+2}$",
            "Sub_Ni_on_Li_mult32_-1": "Ni$_{Li_{m32}}^{-1}$",
            "Sub_Ni_on_Li_mult32_-2": "Ni$_{Li_{m32}}^{-2}$",
            "Sub_Ni_on_Li_mult32_0": "Ni$_{Li_{m32}}^{0}$",
            "Sub_Ni_on_Li_mult32_1": "Ni$_{Li_{m32}}^{+1}$",
            "Sub_Ni_on_Li_mult32_2": "Ni$_{Li_{m32}}^{+2}$",
            "Vac_Li_mult32_-1": "$\\it{V}\\!$ $_{Li_{m32}}^{-1}$",
            "Vac_Li_mult32_-2": "$\\it{V}\\!$ $_{Li_{m32}}^{-2}$",
            "Vac_Li_mult32_0": "$\\it{V}\\!$ $_{Li_{m32}}^{0}$",
            "Vac_Li_mult32_1": "$\\it{V}\\!$ $_{Li_{m32}}^{+1}$",
            "v_Cd_-1": "$\\it{V}\\!$ $_{Cd}^{-1}$",
            "v_Te_2": "$\\it{V}\\!$ $_{Te}^{+2}$",
            "v_Tea_2": "$\\it{V}\\!$ $_{Te}^{+2}$",
            # Vanadium (capitalised "V") with site info; cf. vacancy *of* vanadium (lowercase "v"):
            "V_i_C3v_0": "V$_{i_{C_{3v}}}^{0}$",  # V interstitial
            "V_i_Td_V2.83_0": "V$_{i_{T_{d}-V2.83}}^{0}$",
            "Int_V_mult4_1": "V$_{i_{m4}}^{+1}$",
            "V_Cd_C2v_0": "V$_{Cd_{C_{2v}}}^{0}$",  # V substitution
            "Cd_V_C3v_0": "Cd$_{V_{C_{3v}}}^{0}$",  # substitution on a vanadium site
            "vac_1_V_0": "$\\it{V}\\!$ $_{V_{1}}^{0}$",  # vacancy *of* vanadium
            "Te_Cd_s32_2": "Te$_{Cd_{s32}}^{+2}$",
            "Te_Cd_s32c_2": "Te$_{Cd_{s32c}}^{+2}$",
            "Cd_Te_+2": "Cd$_{Te}^{+2}$",
            "Cd_Te_2": "Cd$_{Te}^{+2}$",
            "Cd_i_C3v_+2": "Cd$_{i_{C_{3v}}}^{+2}$",
            "Cd_i_Td_Te2.83_2": "Cd$_{i_{T_{d}-Te2.83}}^{+2}$",
            "Te_i_Td_Cd2.83_-2": "Te$_{i_{T_{d}-Cd2.83}}^{-2}$",
            "v_O_C3v_0": "$\\it{V}\\!$ $_{O_{C_{3v}}}^{0}$",
            "v_O_C3v_1": "$\\it{V}\\!$ $_{O_{C_{3v}}}^{+1}$",
            "v_O_D4h_2": "$\\it{V}\\!$ $_{O_{D_{4h}}}^{+2}$",
            "Y_O_D4h_0": "Y$_{O_{D_{4h}}}^{0}$",
            "Ti_Y_+2": "Ti$_{Y}^{+2}$",
            "Ti_O_D4h_0": "Ti$_{O_{D_{4h}}}^{0}$",
            "Y_i_C4v_O1.92_0": "Y$_{i_{C_{4v}-O1.92}}^{0}$",
            "Y_i_C4v_O2.68_1": "Y$_{i_{C_{4v}-O2.68}}^{+1}$",
            "Y_i_Cs_O1.71_-2": "Y$_{i_{C_{s}-O1.71}}^{-2}$",
            "Y_i_Cs_O1.95_2": "Y$_{i_{C_{s}-O1.95}}^{+2}$",
            "Y_i_D2d_0": "Y$_{i_{D_{2d}}}^{0}$",
            "v_O_C1_0": "$\\it{V}\\!$ $_{O_{C_{1}}}^{0}$",
            "v_O_C3_8": "$\\it{V}\\!$ $_{O_{C_{3}}}^{+8}$",
            "Li_i_C1_Li1.75_0": "Li$_{i_{C_{1}-Li1.75}}^{0}$",
            "Li_i_C1_O1.72_1": "Li$_{i_{C_{1}-O1.72}}^{+1}$",
            "Li_i_C1_O1.78_2": "Li$_{i_{C_{1}-O1.78}}^{+2}$",
            "Li_i_C2_Li1.84O1.84_0": "Li$_{i_{C_{2}-Li1.84O1.84}}^{0}$",
            "Li_i_C2_Li1.84O1.94_1": "Li$_{i_{C_{2}-Li1.84O1.94}}^{+1}$",
            "Li_i_C2_Li1.86_2": "Li$_{i_{C_{2}-Li1.86}}^{+2}$",
            "Li_i_C3_0": "Li$_{i_{C_{3}}}^{0}$",
            "Cu_i_Oh_0": "Cu$_{i_{O_{h}}}^{0}$",
            "Cu_i_Td_0": "Cu$_{i_{T_{d}}}^{0}$",
            "Cu_i_C3v_Ag1.56Cu1.56Ag2.99a_0": "Cu$_{i_{C_{3v}-Ag1.56Cu1.56Ag2.99a}}^{0}$",
            "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_0": "Cu$_{i_{C_{3v}-Ag1.56Cu1.56Ag2.99b}}^{0}$",
            "Cu_i_C3v_Ag1.80_0": "Cu$_{i_{C_{3v}-Ag1.80}}^{0}$",
            "Te_i_C1_Cd2.71Te2.71Cd4.25j_-2": "Te$_{i_{C_{1}-Cd2.71Te2.71Cd4.25j}}^{-2}$",
            "Te_i_C1_Cd2.71Te2.71Cd4.25l_-2": "Te$_{i_{C_{1}-Cd2.71Te2.71Cd4.25l}}^{-2}$",
            "v_Cd_C1_Te2.83Cd4.62Te5.42a_-1": "$\\it{V}\\!$ $_{Cd_{C_{1}-Te2.83Cd4.62Te5.42a}}^{-1}$",
            "v_Cd_C1_Te2.83Cd4.62Te5.42b_-1": "$\\it{V}\\!$ $_{Cd_{C_{1}-Te2.83Cd4.62Te5.42b}}^{-1}$",
            "v_Cd_C3v_Cd2.71_-1": "$\\it{V}\\!$ $_{Cd_{C_{3v}-Cd2.71}}^{-1}$",
            "v_Cd_C3v_Te2.83Cd4.25_-1": "$\\it{V}\\!$ $_{Cd_{C_{3v}-Te2.83Cd4.25}}^{-1}$",
            "v_Cd_C3v_Te2.83Cd4.62_-1": "$\\it{V}\\!$ $_{Cd_{C_{3v}-Te2.83Cd4.62}}^{-1}$",
            "inter_8_O_-1": "O$_{i_{8}}^{-1}$",
            "inter_14_O_0": "O$_{i_{14}}^{0}$",
            "inter_14_O_+2": "O$_{i_{14}}^{+2}$",
            "inter_14_Th_+2": "Th$_{i_{14}}^{+2}$",
            "vac_14_Th_+2": "$\\it{V}\\!$ $_{Th_{14}}^{+2}$",
            "As_i_C2_-3": "As$_{i_{C_{2}}}^{-3}$",
            "Ag_i_C3v_Ag2.40_0": "Ag$_{i_{C_{3v}-Ag2.40}}^{0}$",
            "Ga_O_C1_Ga1.93Ga2.08O2.62aas_0": "Ga$_{O_{C_{1}-Ga1.93Ga2.08O2.62aas}}^{0}$",
        }
        for (
            defect_species,
            expected_name,
        ) in defect_species_w_site_info_name_dict.items():
            formatted_name = plotting.format_defect_name(
                defect_species=defect_species,
                include_site_info=True,
            )
            assert formatted_name == expected_name
