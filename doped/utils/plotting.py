"""
Code for plotting defect formation energies and transition levels.
"""

import contextlib
import functools
import math
import os
import re
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, ticker
from matplotlib.colors import Colormap, ListedColormap, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table
from pymatgen.core.periodic_table import Element
from pymatgen.util.string import latexify
from pymatgen.util.typing import PathLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from doped.utils.symmetry import sch_symbols  # point group symbols

if TYPE_CHECKING:
    from doped.thermodynamics import DefectThermodynamics

# Recognised vacancy/interstitial substrings in defect names, used by ``format_defect_name`` to infer
# the defect type. Sorted longest-first so the most specific match is found first. Note capitalised "V"
# is treated as Vanadium when separated from the following element by an underscore (e.g. "V_Sb"), but
# as a vacancy when directly concatenated (e.g. "VSb"; handled in ``format_defect_name``). Likewise "I" is
# treated as Iodine (not an interstitial).
recognised_pre_vacancy_strings = sorted(
    ["v_", "v", "va_", "Va_", "va", "Va", "Vac", "vac", "Vac_", "vac_"],
    key=len,
    reverse=True,
)
recognised_post_vacancy_strings = sorted(
    ["_v", "v", "_vac", "_Vac", "vac", "Vac", "va", "Va", "_va", "_Va"],
    key=len,
    reverse=True,
)
recognised_pre_interstitial_strings = sorted(
    ["i", "i_", "Int", "int", "Int_", "int_", "Inter", "inter", "Inter_", "inter_"],
    key=len,
    reverse=True,
)
recognised_post_interstitial_strings = sorted(
    ["_i", "_int", "_Int", "int", "Int", "inter", "Inter", "_inter", "_Inter"],
    key=len,
    reverse=True,
)
lower_cap, upper_cap = (
    -100,
    100,
)  # default min/max x/y range for defect formation energy plots


@contextlib.contextmanager
def doped_plot_style(style_file: PathLike | None = None, style: str = "doped"):
    """
    Context manager applying a ``matplotlib`` plotting style, whether a user-
    supplied ``style_file`` or one of the ``doped`` defaults (``"doped"`` or
    ``"displacement"``).

    Installs ``doped``'s custom font if needed, applies the chosen ``mplstyle``
    within a ``plt.style.context`` (so artists built inside the ``with`` block
    are styled), and then wraps the ``draw()`` and ``print_figure()`` methods
    of any figures created within the block so the style is re-applied on every
    (re-)render -- including Jupyter's deferred end-of-cell display, where a
    bare ``plt.style.context`` would have been restored before the figure is
    drawn. This avoids the need for a session-wide ``plt.style.use``, so the
    user's global ``matplotlib`` style is left unchanged.

    Args:
        style_file (PathLike):
            Path to a ``.mplstyle`` file. If ``None`` (default), uses
            ``doped``'s bundled ``{style}.mplstyle`` (in ``doped/utils``).
        style (str):
            Name of the bundled ``doped`` style to use when ``style_file`` is
            ``None``; either ``"doped"`` (default) or ``"displacement"``.

    Yields:
        PathLike: The resolved style-file path.
    """
    style_file = _resolve_doped_style(style_file, style)
    pre_fignums = set(plt.get_fignums())
    with plt.style.context(style_file):
        yield style_file
    for num in set(plt.get_fignums()) - pre_fignums:  # figures created within the block
        _style_figure_draws(plt.figure(num), style_file)


def _resolve_doped_style(style_file: PathLike | None = None, style: str = "doped") -> PathLike:
    """
    Install ``doped``'s custom (Montserrat) font if not already present, and
    return the resolved style-file path (without applying the style).

    Args:
        style_file (PathLike):
            Path to a ``.mplstyle`` file. If ``None`` (default), uses
            ``doped``'s bundled ``{style}.mplstyle`` (in ``doped/utils``).
        style (str):
            Name of the bundled ``doped`` style to use when ``style_file`` is
            ``None``; either ``"doped"`` (default) or ``"displacement"``.

    Returns:
        PathLike: The resolved style-file path.
    """
    with contextlib.suppress(Exception):  # best-effort; lazy import avoids circular import
        from shakenbreak.plotting import _install_custom_font

        _install_custom_font()
    return style_file or os.path.join(os.path.dirname(__file__), f"{style}.mplstyle")


@functools.cache
def _doped_style_rc(style_file: str) -> dict:
    """
    Parse (and cache) the ``rcParams`` defined in a ``.mplstyle`` file.
    """
    return mpl.rc_params_from_file(style_file, use_default_template=False)


def _style_figure_draws(fig: "mpl.figure.Figure", style_file: PathLike) -> "mpl.figure.Figure":
    """
    Wrap a figure's render methods so the style is (re-)applied transiently on
    every render of ``fig`` -- including the deferred end-of-cell display in
    Jupyter and any later ``savefig`` -- without persisting it to the global
    ``rcParams`` (e.g. via ``plt.style.use``).

    This keeps render-time-resolved style settings (mathtext fonts, lazily
    generated tick labels, tight-bbox sizing, ...) correct even though the
    style context has been exited, while leaving the user's session
    ``rcParams`` untouched.

    Both ``fig.draw`` (interactive redraws) and ``fig.canvas.print_figure``
    (``savefig`` and inline display, including the ``bbox_inches="tight"``
    extent computation that bypasses ``fig.draw``) are wrapped.
    """
    if getattr(fig, "_doped_styled_draw", False):
        return fig  # already wrapped
    rc = _doped_style_rc(str(style_file))

    def _wrap(obj, method_name):
        """
        Replace ``obj.method_name`` with a version that applies the doped rc
        context.
        """
        orig = getattr(obj, method_name)

        @functools.wraps(orig)
        def _styled(*args, **kwargs):
            with mpl.rc_context(rc):
                return orig(*args, **kwargs)

        setattr(obj, method_name, _styled)

    _wrap(fig, "draw")
    _wrap(fig.canvas, "print_figure")
    fig._doped_styled_draw = True  # type: ignore[attr-defined]  # dynamic flag on mpl Figure
    return fig


def _chempot_warning(abs_chempots: dict | None) -> None:
    """
    Issue a warning if absolute (QM/DFT) chemical potentials are not provided.

    Args:
        abs_chempots (dict | None):
            Dictionary of chemical potentials (to be used for computing
            formation energies for plotting). If ``None``, a warning is raised
            indicating that absolute formation energies will be inaccurate.
    """
    if abs_chempots is None:
        warnings.warn(
            "You have not specified chemical potentials (`chempots`), so chemical potentials are set to "
            "zero for each species. This will give large errors in the absolute values of formation "
            "energies, but the transition level positions will be unaffected."
        )


def get_colormap(colormap: str | Colormap | None = None, default: str = "batlow") -> Colormap:
    """
    Get a colormap from a string or a ``Colormap`` object.

    If ``_alpha_X`` in the colormap name, sets the alpha value to X (0-1).

    ``cmcrameri`` colour maps citation: https://zenodo.org/records/8409685

    Args:
        colormap (str, matplotlib.colors.Colormap):
            Colormap to use, either as a string (which can be a colormap name
            from https://www.fabiocrameri.ch/colourmaps or
            https://matplotlib.org/stable/users/explain/colors/colormaps), or
            a ``Colormap`` / ``ListedColormap`` object. If ``None`` (default),
            uses ``default`` colormap (which is ``"batlow"`` by default).
            Append "S" to the colormap name if using a sequential colormap
            from https://www.fabiocrameri.ch/colourmaps.
        default (str):
            Default colormap to use if ``colormap`` is ``None``. Defaults to
            ``"batlow"`` from https://www.fabiocrameri.ch/colourmaps.
    """
    if colormap is None:
        colormap = default

    alpha = None
    if isinstance(colormap, str):  # get colormap from string
        if "_alpha_" in colormap:
            alpha = float(colormap.split("_alpha_")[-1])
            colormap = colormap.split("_alpha_")[0]

        # first check if it's a cmcrameri colormap:
        cmap = cmc.cmaps.get(colormap, None)
        if cmap is None:  # if not, check matplotlib colormaps
            cmap = colormaps.get(colormap, None)
        if cmap is None:
            if "_alpha_" in default:
                alpha = float(default.rsplit("_alpha_", maxsplit=1)[-1])
                default = default.split("_alpha_", maxsplit=1)[0]

            warnings.warn(
                f"Colormap '{colormap}' not found in `cmcrameri` "
                f"(https://www.fabiocrameri.ch/colourmaps) or `matplotlib` "
                f"(https://matplotlib.org/stable/users/explain/colors/colormaps) colormaps. "
                f"Defaulting to '{default}' colormap."
            )
            cmap = cmc.cmaps.get(default, colormaps.get(default, cmc.batlow))

        colormap = cmap

    if alpha is not None and isinstance(colormap, ListedColormap):  # apply alpha to listed colour map
        rgb = np.asarray(colormap.colors)[:, :3]
        colormap = ListedColormap(np.column_stack([rgb, np.full(len(rgb), alpha)]), name=colormap.name)

    assert isinstance(colormap, Colormap)  # always resolved to a Colormap above
    return colormap


def get_linestyles(linestyles: str | list[str] = "-", num_lines: int = 1) -> list[str]:
    """
    Get a list of linestyles to use for plotting, from a string or list of
    strings (linestyles).

    If a list is provided which doesn't match the number of lines, the list is
    repeated until it does.

    Args:
        linestyles (str, list[str]):
            Linestyles to use for plotting. If a string, uses that linestyle
            for all lines. If a list, uses each linestyle in the list for each
            line. Defaults to ``"-"``.
        num_lines (int):
            Number of lines to plot (and thus number of linestyles to output in
            list). Defaults to 1.
    """
    if isinstance(linestyles, str):
        return [linestyles] * num_lines

    # else ensure match number of lines to number of linestyles:
    return linestyles * (num_lines // len(linestyles)) + linestyles[: num_lines % len(linestyles)]


def _get_TLD_colors_and_linestyles(
    colormap: str | Colormap | None, linestyles: str | list[str], num_lines: int
) -> tuple[np.ndarray, list[str]]:
    """
    Helper function to get the colors and linestyles to use for defect
    formation energy lines on a transition level diagram plot.

    Args:
        colormap (str, matplotlib.colors.Colormap):
            Colormap to use for the formation energy lines.
        linestyles (str, list[str]):
            Linestyles to use for the formation energy lines.
        num_lines (int):
            Number of lines to plot (and thus number of colours and linestyles
            to output).

    Returns:
        colors (list[str | tuple[float, ...]]):
            List of colors to use for the formation energy lines.
        linestyles (list[str]):
            List of linestyles to use for the formation energy lines.
    """
    # future updated colour handling (based on defect type etc) should remove the need for this:
    if num_lines <= 10:
        default = "tab10_alpha_0.75"
    elif num_lines <= 20:
        default = "tab20"
    else:
        default = "batlow"  # set to colormap if not enough colours in listed colormaps

    cmap = get_colormap(colormap, default=default)
    base = (  # normalise to RGBA, as listed colormaps can mix RGB / RGBA colours (-> inhomogeneous array)
        to_rgba_array(cmap.colors) if isinstance(cmap, ListedColormap) else np.empty(0)
    )
    colors: np.ndarray  # typing
    if 0 < len(base) < 150:  # cmcrameri colormaps return 256 colours
        # repeat (tile) the listed colours (cycling) until we have one per line:
        colors = np.tile(base, (int(np.ceil(num_lines / len(base))), 1))[:num_lines]
    else:
        colors = cmap(np.linspace(0, 1, num_lines))

    return colors, get_linestyles(linestyles, num_lines)


def _plot_formation_energy_lines(
    xy: dict,
    colors: Sequence[str | tuple[float, ...]] | np.ndarray,
    linestyles: list[str],
    ax: plt.Axes,
    styled_linewidth: float,
    styled_markersize: float,
    **kwargs,
) -> None:
    r"""
    Plot defect formation energy lines on a given ``Axes`` object.

    Args:
        xy (dict):
            Dictionary of ``{defect_name: [[x_vals], [y_vals]]}`` for the
            formation energy lines to plot.
        colors (list):
            List of colors to use for the formation energy lines, matching the
            order of the ``xy`` dictionary.
        linestyles (list[str]):
            List of linestyles to use for the formation energy lines, matching
            the order of the ``xy`` dictionary.
        ax (plt.Axes):
            ``Axes`` object to plot the formation energy lines on.
        styled_linewidth (float):
            Linewidth to use (multiplied by ``1.2``) for the formation energy
            lines.
        styled_markersize (float):
            Marker size to use (multiplied by ``4/6``) for the formation energy
            lines.
        **kwargs:
            Additional keyword arguments to pass to ``ax.plot``.
    """
    for i, (x_vals, y_vals) in enumerate(xy.values()):
        ax.plot(
            x_vals,
            y_vals,
            color=colors[i],
            linestyle=linestyles[i],
            markeredgecolor=colors[i],
            lw=styled_linewidth * 1.2,
            markersize=styled_markersize * (4 / 6),
            **kwargs,
        )


def _plot_transition_level_markers(
    ax: plt.Axes,
    defect_thermodynamics: "DefectThermodynamics",
    defect_names: Iterable[str],
    colors: Sequence[str | tuple[float, ...]] | np.ndarray,
    abs_chempots: dict | None,
    styled_markersize: float,
    styled_font_size: float,
    all_entries: bool | str = False,
    auto_labels: bool = False,
) -> None:
    r"""
    Mark the charge transition levels (as points) on the formation energy
    diagram, one set per defect in ``defect_names`` (taking transition level
    positions from ``DefectThermodynamics.transition_level_map``), optionally
    annotating each with its charge transition label (e.g.
    ``$\epsilon$(+1/0)``) when ``auto_labels`` is ``True``.

    Args:
        ax (plt.Axes):
            ``Axes`` object to plot the transition level markers on.
        defect_thermodynamics (DefectThermodynamics):
            ``DefectThermodynamics`` object containing the transition level
            data (in the ``transition_level_map`` attribute).
        defect_names (Iterable[str]):
            List of defect names to plot transition level markers for, matching
            keys in ``DefectThermodynamics.transition_level_map``.
        colors (list[mpl.colors.Color]):
            List of colors to use for the transition level markers, matching
            the order of ``defect_names``.
        abs_chempots (dict | None):
            Dictionary of chemical potentials to use for the formation energy
            calculations.
        styled_markersize (float):
            Marker size to use (multiplied by ``4/6``).
        styled_font_size (float):
            Font size to use for transition level labels.
        all_entries (bool | str):
            Whether all entries or only the stable entries are being plotted
            (used to determine the color and alpha of the transition level
            markers). Defaults to ``False``.
        auto_labels (bool):
            Whether to automatically label the transition levels with their
            charge states. If there are many transition levels, this can be
            quite ugly. Defaults to ``False``.
    """
    tl_map: dict[str, dict[float, list[int]]] = defect_thermodynamics.transition_level_map
    for i, def_name in enumerate(defect_names):
        x_trans, y_trans, tl_labels, tl_label_type = [], [], [], []
        for x_val, chargeset in tl_map[def_name].items():
            x_trans.append(x_val)
            y_trans.append(
                next(
                    defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=abs_chempots, fermi_level=x_val
                    )
                    for defect_entry in defect_thermodynamics.stable_entries[def_name]
                    if defect_entry.charge_state == chargeset[0]  # formation energy of first entry in TL
                )
            )
            tl_labels.append(
                _format_TL_charge_label((max(chargeset), min(chargeset)), prefix=r"$\epsilon$")
            )
            tl_label_type.append("start_positive" if max(chargeset) > 0 else "end_negative")

        if not x_trans:
            continue

        color = "k" if all_entries is True else colors[i]
        ax.plot(
            x_trans,
            y_trans,
            marker="o",
            color=color,
            markeredgecolor=color,
            markersize=styled_markersize * (4 / 6),
            linestyle="",
            alpha=0.5 if all_entries is True else None,
        )
        if auto_labels:  # annotate each TL point with its charge transition label
            for coords, label, label_type in zip(
                zip(x_trans, y_trans, strict=True), tl_labels, tl_label_type, strict=True
            ):
                ax.annotate(
                    label,
                    coords,
                    textcoords="offset points",
                    xytext=(0, 5),  # offset (x, y) from the point
                    ha="right" if label_type == "start_positive" else "left",
                    size=styled_font_size * 0.9,
                    annotation_clip=True,  # only show label if coords in current axes
                )


def shade_band_edges(
    ax: plt.Axes,
    band_gap: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    orientation: str = "horizontal",
) -> None:
    """
    Shade the valence (blue) and conduction (orange) band-edge regions, each
    darkest at the band edge and fading away from it.

    This style was initially implemented in the deprecated ``AIDE`` defect
    package by Adam Jackson and Alex Ganose.

    For ``orientation="horizontal"`` (formation energy / transition level
    diagram), the Fermi level is on the x-axis: the VBM region spans ``x`` in
    ``[min(xlim), 0]`` and the CBM region ``x`` in ``[band_gap, max(xlim)]``
    (full vertical extent). For ``orientation="vertical"`` (vertical energy /
    transition level diagram), the Fermi level is on the y-axis: the VBM region
    spans ``y`` in ``[min(ylim), 0]`` and the CBM region ``y`` in
    ``[band_gap, max(ylim)]`` (full horizontal extent).

    Args:
        ax (plt.Axes):
            The ``matplotlib`` ``Axes`` object to apply the shading to.
        band_gap (float):
            The band gap of the material (in eV). This defines the start
            of the conduction band minimum (CBM) region.
        xlim (tuple[float, float]):
            Tuple of ``(min, max)`` limits for the x-axis. Used to determine
            the extent of the shaded regions, depending on the orientation.
        ylim (tuple[float, float]):
            Tuple of ``(min, max)`` limits for the y-axis. Used to determine
            the extent of the shaded regions, depending on the orientation.
            Ignored if ``orientation="horizontal"``.
        orientation (str):
            The orientation of the plot. Either ``"horizontal"`` (default) for
            standard formation energy diagrams, or ``"vertical"`` for vertical
            energy / transition level diagrams.
    """
    shared_kwargs: dict[str, Any] = {
        "vmin": 0,
        "vmax": 3,
        "interpolation": "bicubic",
        "rasterized": True,
        "aspect": "auto",
        "zorder": 0,
    }
    blues, oranges = colormaps["Blues"], colormaps["Oranges"]
    if orientation == "horizontal":  # gradient along x, fixed (large) vertical extent
        ylim = (lower_cap, upper_cap)
        if min(xlim) < 0:  # only draw if finite extent of band-edge region in plot
            ax.imshow([(0, 1), (0, 1)], cmap=blues, extent=(min(xlim), 0, *ylim), **shared_kwargs)
        if max(xlim) > band_gap:  # only draw if finite extent of band-edge region in plot
            ax.imshow([(1, 0), (1, 0)], cmap=oranges, extent=(band_gap, max(xlim), *ylim), **shared_kwargs)
    else:  # vertical: gradient along y, spanning the full x-range
        if min(ylim) < 0:  # only draw if finite extent of band-edge region in plot
            ax.imshow([(1, 1), (0, 0)], cmap=blues, extent=(*xlim, min(ylim), 0.0), **shared_kwargs)
        if max(ylim) > band_gap:  # only draw if finite extent of band-edge region in plot
            ax.imshow([(0, 0), (1, 1)], cmap=oranges, extent=(*xlim, band_gap, max(ylim)), **shared_kwargs)


def _set_TLD_axis_labels_limits_ticks(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """
    Format the axes for a formation energy diagram plot.

    Sets the x and y axis limits, labels and tick locators.

    Args:
        ax (plt.Axes):
            The ``matplotlib`` ``Axes`` object to format.
        xlim (tuple[float, float]):
            The minimum and maximum limits for the x-axis (Fermi level).
        ylim (tuple[float, float]):
            The minimum and maximum limits for the y-axis (Formation Energy).
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Fermi Level (eV)")
    ax.set_ylabel("Formation Energy (eV)")
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(4))
        axis.set_minor_locator(ticker.AutoMinorLocator(2))


def _set_title_and_save_figure(
    ax: plt.Axes,
    title: str | None = None,
    chempot_table: bool = False,
    filename: PathLike | None = None,
    styled_font_size: float = 9.0,
) -> None:
    """
    Set the plot title (if given) and save the figure to file (if requested).

    Args:
        ax (plt.Axes):
            The axes whose title to set and whose figure to save.
        title (str | None):
            Plot title, LaTeX-formatted via ``latexify``. If ``None`` or empty,
            no title is set. Defaults to ``None``.
        chempot_table (bool):
            Whether a chemical potential table is displayed above the plot, in
            which case the title is enlarged and padded to sit above it.
            Defaults to ``False``.
        filename (PathLike | None):
            If provided, the path to save the figure to (at 600 dpi, with tight
            bounding box and transparent background). Defaults to ``None``.
        styled_font_size (float):
            Base font size (pt) for the title. Defaults to ``9.0``.
    """
    if title:
        if chempot_table:
            ax.set_title(
                latexify(title),
                size=1.2 * styled_font_size,
                pad=28,
                fontdict={"fontweight": "bold"},
            )
        else:
            ax.set_title(latexify(title), size=styled_font_size, fontdict={"fontweight": "bold"})

    if filename is not None:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)
        fig.savefig(filename, dpi=600, bbox_inches="tight", transparent=True)


def format_defect_name(
    defect_species: str,
    include_site_info: bool = False,
    include_charge: bool = True,
    wout_charge: bool | None = None,
    include_site_info_in_name: bool | None = None,
) -> str | None:
    r"""
    Format defect name using LaTeX styling, intended for plot labelling/titles.

    For example, converts ``"Cd_i_C3v_0"`` to ``"$Cd_{i}^{0}$"`` or
    ``"$Cd_{i_{C3v}}^{0}$"``, if ``include_site_info`` is ``True``), or
    ``"$Cd_i$"`` if ``include_charge = False``.

    Note that capitalised ``"V"`` is treated as Vanadium when separated from
    the following element by an underscore (e.g. ``"V_Sb"``), but as a vacancy
    when directly concatenated (e.g. ``"VSb"``); lowercase ``"v"`` or
    ``"Va"``/``"Vac"`` are always vacancies. Likewise ``"I"`` is treated as
    Iodine (not an interstitial; lowercase ``"i"`` or ``"Int"`` are used for
    interstitials).

    Args:
        defect_species (str):
            Name of defect including charge state (e.g. ``"Cd_i_C3v_0"``).
        include_site_info (bool):
            Whether to include site info in name (e.g. ``"$Cd_{i}^{0}$"``
            or ``"$Cd_{i_{C3v}}^{0}$"``). Defaults to ``False``.
        include_charge (bool):
            Whether to include the charge state in the formatted
            ``defect_species`` name. Defaults to ``True``.
        wout_charge (bool):
            Deprecated alias for ``not include_charge`` (i.e. whether to
            *exclude* the charge state). Will be removed in ``doped`` v4.1;
            use ``include_charge`` instead.
        include_site_info_in_name (bool):
            Deprecated alias for ``include_site_info``. Will be removed in
            ``doped`` v4.1; use ``include_site_info`` instead.

    Returns:
        str | None:
            Formatted defect name, or ``None`` if it could not be parsed.
    """
    if not isinstance(defect_species, str):  # check inputs
        raise TypeError(f"`defect_species` {defect_species} should be a string")

    # capitalised "V" directly concatenated with an element (e.g. "VSb", "VI") treated as a vacancy.
    defect_species = re.sub(r"^V(?=[A-Z])", "v_", defect_species)
    # "V" separated from the element by an underscore (e.g. "V_Sb") is instead treated as Vanadium.

    if wout_charge is not None:
        warnings.warn(
            "The `wout_charge` argument of `format_defect_name` is deprecated and will be removed in "
            "doped v4.1. Use `include_charge` (with inverted meaning) instead.",
            DeprecationWarning,
            stacklevel=2,
        )  # TODO: Remove in v4.1
        include_charge = not wout_charge

    if include_site_info_in_name is not None:
        warnings.warn(
            "The `include_site_info_in_name` argument of `format_defect_name` is deprecated and will be "
            "removed in doped v4.1. Use `include_site_info` instead.",
            DeprecationWarning,
            stacklevel=2,
        )  # TODO: Remove in v4.1
        include_site_info = include_site_info_in_name

    if not include_charge:
        defect_species += "_99"  # add dummy charge for parsing; 99 red balloons go by...

    try:
        charge = int(defect_species.split("_")[-1])  # charge comes last
        charge_string = f"{charge:+}" if charge > 0 else f"{charge}"
    except ValueError as e:
        raise ValueError(
            f"Problem reading defect name {defect_species}, should end with charge state after underscore "
            f"(e.g. Te_i_Td_Te2.83_+1)"
        ) from e

    defect_name = None
    pre_charge_name = defect_species.rsplit("_", 1)[0]  # defect name without charge state
    trimmed_pre_charge_name = pre_charge_name  # later trimmed of any pre/post vacancy/interstitial strings

    doped_site_info = None
    # check if name is doped format, having site info as point group symbol (and more) after 2nd "_":
    with contextlib.suppress(IndexError):
        # from 2nd underscore to last underscore (before charge state) is site info:
        point_group_symbol = defect_species.split("_")[2]
        if point_group_symbol in sch_symbols and all(  # recognised point group symbol?
            i not in pre_charge_name
            for i in ["int", "Int", "vac", "Vac", "sub", "Sub", "as_"]  # no As_
        ):
            # format point group symbol to (e.g. C1 -> C_1) (already in math mode here)
            doped_site_info = f"{point_group_symbol[0]}_{{{point_group_symbol[1:]}}}"
            if defect_species.split("_")[3:-1]:  # if there is more site info after point group symbol
                doped_site_info += "-" + "-".join(defect_species.split("_")[3:-1])
            trimmed_pre_charge_name = pre_charge_name.replace(
                f"_{'_'.join(defect_species.split('_')[2:-1])}", ""
            )

    for substring in (  # trim any matching pre or post vacancy/interstitial strings from defect name
        recognised_pre_vacancy_strings
        + recognised_post_vacancy_strings
        + recognised_pre_interstitial_strings
        + recognised_post_interstitial_strings
    ):
        if substring in trimmed_pre_charge_name and not (
            substring.endswith("i") or substring.startswith("i")
        ):
            trimmed_pre_charge_name = trimmed_pre_charge_name.replace(substring, "")

    # Note throughout the element-matching below: ``trimmed_pre_charge_name`` (pre/post vacancy and
    # interstitial substrings removed) is used to `find` the constituent elements, while the untrimmed
    # ``pre_charge_name`` is passed on to `match` the defect format (so the defect type strings remain):
    possible_two_character_elements = _valid_symbols(
        trimmed_pre_charge_name[i : i + 2] for i in range(len(trimmed_pre_charge_name) - 1)
    )

    matching_kwargs: dict[str, Any] = {
        "charge_string": charge_string,
        "doped_site_info": doped_site_info,
        "name": pre_charge_name,
        "include_site_info": include_site_info,
    }
    if possible_two_character_elements:
        defect_name = _defect_name_from_matching_elements(
            possible_two_character_elements, **matching_kwargs
        )

        if defect_name is None and len(possible_two_character_elements) == 1:
            # possibly one single-character element and one two-character element
            possible_one_character_elements = _valid_symbols(
                trimmed_pre_charge_name.replace(possible_two_character_elements[0], "")
            )

            if possible_one_character_elements:
                # in this case, we don't know the order of the 1-character vs 2-character elements in the
                # name, so we try both orderings:
                defect_name = _defect_name_from_matching_elements(
                    possible_two_character_elements + possible_one_character_elements, **matching_kwargs
                )
                if defect_name is None:
                    defect_name = _defect_name_from_matching_elements(
                        possible_one_character_elements + possible_two_character_elements,
                        **matching_kwargs,
                    )

    if defect_name is None and (
        possible_one_character_elements := _valid_symbols(trimmed_pre_charge_name)
    ):
        defect_name = _defect_name_from_matching_elements(
            possible_one_character_elements, **matching_kwargs
        )

    if defect_name is None:  # try matching to PyCDT/old-doped style:
        defect_name = _pycdt_style_defect_name(defect_species, charge_string, include_site_info)

    return f"{defect_name.rsplit('^', 1)[0]}$" if (defect_name and not include_charge) else defect_name


def _pycdt_style_defect_name(
    defect_species: str,
    charge_string: str,
    include_site_info: bool,
) -> str | None:
    """
    Format a defect name from the old PyCDT/``doped`` style.

    Handles e.g. ``"vac_1_Cd"``, ``"Int_Cd_1"`` and ``"sub_2_Te_on_Cd"``,
    returning ``None`` if the name is not recognised.
    """
    try:
        defect_type = defect_species.split("_", maxsplit=1)[0]  # vac, as or int
        if (
            defect_type.capitalize() == "Int"
        ):  # for interstitials, name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
            site_element = defect_species.split("_")[1]  # element then site for interstitials
            site: str | None = defect_species.split("_")[2]
            defect_name = _int_name(site_element, charge_string, site if include_site_info else None)
        else:
            site = defect_species.split("_")[1]  # number indicating defect site (from doped)
            site_element = defect_species.split("_")[2]  # element at defect site

        site = site if include_site_info else None  # whether to include the site number
        if defect_type.lower() == "vac":
            defect_name = _vac_name(site_element, charge_string, site)
        elif defect_type.lower() in ["as", "sub"]:
            subs_element = defect_species.split("_")[4]
            defect_name = _sub_name(site_element, subs_element, charge_string, site)
        elif defect_type.capitalize() != "Int":
            raise ValueError(f"Defect type {defect_type} not recognized. Please check spelling.")
    except Exception:
        return None

    return defect_name


# LaTeX defect-name builders, used by ``format_defect_name`` (``site`` = optional site-info subscript):
def _vac_name(element: str, charge_string: str, site: str | None = None) -> str:
    """
    Build a LaTeX vacancy label for ``element``.
    """
    inner = f"{element}_{{{site}}}" if site else element
    return rf"$\it{{V}}\!$ $_{{{inner}}}^{{{charge_string}}}$"


def _int_name(element: str, charge_string: str, site: str | None = None) -> str:
    """
    Build a LaTeX interstitial label for ``element``.
    """
    sub = f"_{{i_{{{site}}}}}" if site else "_i"  # note: bare "_i" (no braces) when no site info
    return f"{element}${sub}^{{{charge_string}}}$"


def _sub_name(sub_element: str, orig_element: str, charge_string: str, site: str | None = None) -> str:
    """
    Build a LaTeX substitution/antisite label (``sub_element`` on
    ``orig_element``).
    """
    inner = f"{orig_element}_{{{site}}}" if site else orig_element
    return f"{sub_element}$_{{{inner}}}^{{{charge_string}}}$"


def _valid_symbols(strings) -> list[str]:
    """
    Unique valid element symbols (order-preserving) from ``strings``.
    """
    seen: list[str] = []
    for s in strings:
        if Element.is_valid_symbol(s) and s not in seen:
            seen.append(s)
    return seen


def _check_matching_defect_format(
    element: str,
    name: str,
    pre_def_type_list: list[str],
    post_def_type_list: list[str],
) -> int:
    """
    Score how well ``name`` matches a defect naming format for ``element``.

    Ignores site info (parsed separately), checking only ``element`` placement
    relative to the pre/post defect-type strings. Returns ``len(name)`` minus
    the match start index (so earlier matches score higher), or ``0`` if no
    match is found.
    """
    patterns = [f"{pre_def_type}{element}" for pre_def_type in pre_def_type_list] + [
        f"{element}{post_def_type}" for post_def_type in post_def_type_list
    ]
    if any(name.startswith(pattern) for pattern in patterns):
        return len(name)
    for i in range(len(name) - 1):
        if any(name[i : i + len(pattern)] == pattern for pattern in patterns):
            return len(name) - i
    return 0  # 0 -> False, no match found


def _check_matching_defect_format_with_old_site_info(
    element: str,
    name: str,
    pre_def_type_list: list[str],
    post_def_type_list: list[str],
) -> tuple[bool, str | None]:
    """
    Match ``name`` to an old-format vacancy or interstitial defect label for
    ``element``, with site info.

    Returns ``(matched, site_info)``, where ``site_info`` is the parsed (old-
    format) site string, or ``None`` if unmatched.
    """
    for site_info in _iter_old_site_info_matches(name):
        pre_matches = (
            fstring in name
            for pre_def_type in pre_def_type_list
            for fstring in [
                f"{pre_def_type}{site_info}{element}",
                f"{pre_def_type}{element}{site_info}",
                f"{pre_def_type}{site_info}_{element}",
                f"{pre_def_type}{element}_{site_info}",
            ]
        )
        post_matches = (
            fstring in name
            for post_def_type in post_def_type_list
            for fstring in [
                f"{element}{site_info}{post_def_type}",
                f"{site_info}{element}{post_def_type}",
                f"{element}{site_info}_{post_def_type}",
                f"{site_info}_{element}{post_def_type}",
            ]
        )
        if any(pre_matches) or any(post_matches):
            return True, site_info.replace("mult", "m")

    return False, None


def _try_vacancy_interstitial_match(
    element: str,
    name: str,
    include_site_info: bool,
    charge_string: str,
    doped_site_info: str | None,
) -> str | None:
    """
    Match ``name`` to a vacancy or interstitial defect label for ``element``,
    returning the formatted label, or ``None`` if no match found.
    """
    defect_name_without_site_info = defect_name_with_site_info = None

    match_found, site_info = _check_matching_defect_format_with_old_site_info(
        element, name, recognised_pre_vacancy_strings, recognised_post_vacancy_strings
    )
    if match_found:
        defect_name_with_site_info = _vac_name(element, charge_string, site_info)
        defect_name_without_site_info = _vac_name(element, charge_string)

    else:
        match_found, site_info = _check_matching_defect_format_with_old_site_info(
            element, name, recognised_pre_interstitial_strings, recognised_post_interstitial_strings
        )
        if match_found:
            defect_name_with_site_info = _int_name(element, charge_string, site_info)
            defect_name_without_site_info = _int_name(element, charge_string)

    if include_site_info and defect_name_with_site_info is not None:
        return defect_name_with_site_info

    vacancy_match_score = _check_matching_defect_format(
        element, name, recognised_pre_vacancy_strings, recognised_post_vacancy_strings
    )
    interstitial_match_score = _check_matching_defect_format(
        element, name, recognised_pre_interstitial_strings, recognised_post_interstitial_strings
    )

    if vacancy_match_score == 0 and interstitial_match_score == 0:  # no match
        if defect_name_without_site_info is not None:  # if match with old site-info format
            return defect_name_without_site_info

        return None

    name_func = _vac_name if vacancy_match_score > interstitial_match_score else _int_name
    if include_site_info and doped_site_info is not None:
        return name_func(element, charge_string, doped_site_info)

    return name_func(element, charge_string)


def _try_substitution_match(
    substituting_element: str,
    orig_site_element: str,
    name: str,
    include_site_info: bool,
    charge_string: str,
    doped_site_info: str | None,
) -> str | None:
    """
    Match ``name`` to a substitution/antisite defect label, returning the
    formatted name, else ``None`` if no match found.
    """
    defect_name = None
    if (
        f"{substituting_element}_{orig_site_element}" in name
        or f"{substituting_element}_on_{orig_site_element}" in name
    ):
        defect_name = _sub_name(
            substituting_element,
            orig_site_element,
            charge_string,
            doped_site_info if include_site_info else None,
        )

    if defect_name and include_site_info:  # if we have a match, check if we can add the site number
        for site_info in _iter_old_site_info_matches(name):
            if any(
                fstring in name
                for fstring in [
                    f"{site_info}_{substituting_element}_{orig_site_element}",
                    f"{substituting_element}_{orig_site_element}_{site_info}",
                    f"{site_info}_{substituting_element}_on_{orig_site_element}",
                    f"{substituting_element}_on_{orig_site_element}_{site_info}",
                ]
            ):
                defect_name = _sub_name(substituting_element, orig_site_element, charge_string, site_info)
                break

    return defect_name.replace("mult", "m") if defect_name is not None else None


def _defect_name_from_matching_elements(
    element_matches: list[str],
    charge_string: str,
    doped_site_info: str | None,
    name: str,
    include_site_info: bool,
) -> str | None:
    """
    Determine the (formatted) defect label from candidate ``element_matches``
    in the (unformatted) ``name``.
    """
    if len(element_matches) == 1:  # vacancy or interstitial?
        defect_name = _try_vacancy_interstitial_match(
            element_matches[0], name, include_site_info, charge_string, doped_site_info
        )
    elif len(element_matches) == 2:
        # try substitution/antisite match, if not try vacancy/interstitial with first element
        defect_name = _try_substitution_match(
            element_matches[0], element_matches[1], name, include_site_info, charge_string, doped_site_info
        )
        if defect_name is None:
            defect_name = _try_vacancy_interstitial_match(
                element_matches[0], name, include_site_info, charge_string, doped_site_info
            )
    else:
        # try use first match and see if we match vacancy or interstitial format; if not, try first and
        # second matches and see if we match substitution format; otherwise fail
        defect_name = _try_vacancy_interstitial_match(
            element_matches[0], name, include_site_info, charge_string, doped_site_info
        )
        if defect_name is None:
            defect_name = _try_substitution_match(
                element_matches[0],
                element_matches[1],
                name,
                include_site_info,
                charge_string,
                doped_site_info,
            )

    return defect_name


def _iter_old_site_info_matches(name: str):
    """
    Yield candidate site-info substrings (e.g. ``"1"``, ``"s2"``, ``"mult3"``)
    parsed from ``name`` using the old (pre-point-group) ``doped``/``PyCDT``
    naming formats, trying each preposition/postposition combination in turn.
    """
    for site_preposition in ["s", "m", "mult", ""]:  # possible site info prepositions
        for site_postposition in [r"[a-z]", ""]:  # possible site info postpositions
            # ([a-z_]+) -> 1st group; letters/underscores (no numbers), then the site info:
            # ({site_preposition}[0-9]+{site_postposition}) -> 2nd group; pre, number(s), post
            match = re.match(f"([a-z_]+)({site_preposition}[0-9]+{site_postposition})", name, re.I)
            if match:
                yield match.groups()[1]  # the site-info group (2nd)


def _try_format_defect_name(defect_entry_name: str, site_info: bool, include_charge: bool = False) -> str:
    """
    `Try` to format a defect entry name in LaTeX style (for plotting), falling
    back to the raw name if formatting fails.
    """
    try:
        formatted_name = format_defect_name(
            defect_species=defect_entry_name,
            include_site_info=site_info,
            include_charge=include_charge,
        )
        if formatted_name is None:  # fallback to exception handling below
            raise RuntimeError(f"Defect entry {defect_entry_name} could not be formatted.")
        return formatted_name

    except Exception:  # if formatting fails, just use the defect_species name
        return defect_entry_name


def format_defect_names(
    defect_names: list[str], include_charge: bool = False, include_site_info: bool | None = None
) -> list[str]:
    """
    Format a list of defect names into LaTeX-like labels for plotting (e.g.
    plot legends or transition level diagram column headers).

    Each name is formatted with :func:`format_defect_name`
    (e.g. ``"Cd_i_C3v_0"`` -> ``"$Cd_{i}^{0}$"``), and the labels are then made
    unique so that no two defects share the same label. The handling of
    crystallographic site info, used to disambiguate defects of the same type
    at inequivalent sites, is controlled by ``include_site_info``. If names for
    different defects are still not unique after formatting, and potentially
    including site info, then "-a", "-b", "-c"... suffixes are appended to
    differentiate defects.

    Args:
        defect_names (list[str]):
            List of defect names to format (e.g. ``["Cd_i_C3v_0", "Cd_Te",
            "v_Cd_-1", ...]``), as taken from ``DefectEntry.name`` or the keys
            of a ``DefectThermodynamics`` transition-level dictionary.
        include_charge (bool):
            Whether to include the charge states in the formatted defect names.
            Defaults to ``False``.
        include_site_info (bool, None):
            Whether to include crystallographic site info in the formatted
            names (e.g. ``"$Cd_{i_{C3v}}$"`` rather than ``"$Cd_{i}$"``):

            - ``None`` (default): site info is omitted, then added only to
              defect names that would otherwise collide (to disambiguate
              them), then "-a", "-b", "-c"... suffixes are appended as a last
              resort if names still collide.
            - ``False``: site info is omitted from all names, and "-a", "-b",
              "-c"... suffixes are appended if names collide.
            - ``True``: site info is shown on all names (when available), and
              "-a", "-b", "-c"... suffixes are appended if names still collide.

    Returns:
        list[str]:
            The formatted, unique defect labels, in the same order as the input
            ``defect_names``.
    """
    formatted_names = [
        _try_format_defect_name(name, include_site_info is True, include_charge) for name in defect_names
    ]

    if len(formatted_names) == len(set(formatted_names)):  # no duplicates, good to go
        return formatted_names

    # duplicates in defect names; rename to avoid overwriting:
    if include_site_info is None:  # first see if using site info with duplicates removes duplicate names
        site_info_names = [_try_format_defect_name(name, True, include_charge) for name in defect_names]
        formatted_names = [
            (
                site_info_name
                if site_info_names.count(site_info_name) < formatted_names.count(non_site_info_name)
                else non_site_info_name
            )
            for site_info_name, non_site_info_name in zip(site_info_names, formatted_names, strict=False)
        ]

        if len(formatted_names) == len(set(formatted_names)):
            return formatted_names

    # duplicates in entry names and site info doesn't (fully) solve it (or ``include_site_info=False``),
    # so append "a,b,c.." for different defect species with the same name:
    final_names: list[str] = []
    for defect_name in formatted_names:
        final_names.append(
            _uniquified_name(
                defect_name,
                variant_exists=lambda base: any(base in i for i in final_names),
                exact_exists=lambda n: n in final_names,
                suffix=lambda base, i: f"{base}$_{{-{chr(96 + i)}}}$",
                rename_prev=lambda old, new: final_names.__setitem__(final_names.index(old), new),
            )
        )

    return final_names


def get_legend_font_size() -> float:
    """
    Convenience function to get the current ``matplotlib`` legend font size, in
    points (pt).

    Returns:
        float: Current legend font size in points (pt).
    """
    font_size = plt.rcParams["legend.fontsize"]  # current legend font size from rcParams

    # if the font size is a string (like 'medium'), convert it using FontProperties
    if isinstance(font_size, str):
        return FontProperties(size=font_size).get_size_in_points()

    return font_size  # otherwise numeric, return as is


def _uniquified_name(
    base: str,
    variant_exists: Callable[[str], bool],
    exact_exists: Callable[[str], bool],
    suffix: Callable[[str, int], str],
    rename_prev: Callable[[str, str], None],
) -> str:
    """
    Generate a unique name by appending an "a, b, c..." suffix when ``base``
    (or a previously-suffixed variant) already exists.

    If ``base`` is the first duplicate (a direct match), the pre-existing entry
    is renamed to the "a" variant (via ``rename_prev``) and ``base`` becomes
    the "b" variant; otherwise the next free suffix is used.

    Args:
        base (str): The (unsuffixed) name to uniquify.
        variant_exists (Callable):
            Function to test whether ``base`` `or any suffixed variant` already
            exists.
        exact_exists (Callable):
            Function to test if a given (`exact`) name already exists.
        suffix (Callable):
            Function to build the suffixed name from ``(base, i)``
            (e.g. i=1 -> "{base}_a" etc).
        rename_prev (Callable):
            Function to rename a pre-existing entry from old to new name.

    Returns:
        str: The (possibly suffixed) unique name.
    """
    if not variant_exists(base):
        return base

    # defects with same name, rename to prevent overwriting; append "a,b,c.." for different species:
    i = 3
    if exact_exists(base):  # first repeat, direct match, rename previous entry to "a"
        rename_prev(base, suffix(base, 1))  # a
        name = suffix(base, 2)  # b
    else:
        name = suffix(base, i)  # c

    while exact_exists(name):
        i += 1
        name = suffix(base, i)  # d, e, f etc

    return name


def _rename_key_and_dicts(
    key: str,
    output_dicts: list,
) -> tuple[str, list]:
    """
    Given an input key, renames the key if it already exists in the
    ``output_dicts`` dictionaries (to ``key``_a, ``key``_b, ``key``_c etc),
    renames the corresponding keys in the dictionaries, and returns the renamed
    key and updated dictionaries.
    """
    output_dict = output_dicts[0]

    def _rename_prev(old, new):
        """
        Rename the ``old`` key to ``new`` in every dict in ``output_dicts``.
        """
        for single_output_dict in output_dicts:
            single_output_dict[new] = single_output_dict.pop(old)

    key = _uniquified_name(
        key,
        variant_exists=lambda base: (
            base in output_dict or any(f"{base}_{chr(96 + i)}" in output_dict for i in range(1, 27))
        ),
        exact_exists=lambda n: n in output_dict,
        suffix=lambda base, i: f"{base}_{chr(96 + i)}",
        rename_prev=_rename_prev,
    )
    return key, output_dicts


def _get_formation_energy_lines(
    defect_thermodynamics: "DefectThermodynamics",
    abs_chempots: dict | None,
    xlim: tuple[float, float],
    defect_subset: list[str] | str | None = None,
):
    """
    Compute formation energy vs Fermi level line data for plotting.

    ``((xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin)`` is
    returned, where ``xy`` holds the stable (ground-state) formation energy
    lines per defect, ``all_lines_xy`` holds the lines for `every` charge
    state, and the ``y_range_vals`` lists give the y-values at the x-limits
    (for axis scaling).
    """

    def _form_en(defect_entry, fermi_level):
        """
        Formation energy of ``defect_entry`` at the given Fermi level.
        """
        return defect_thermodynamics.get_formation_energy(
            defect_entry, chempots=abs_chempots, fermi_level=fermi_level
        )

    def _entry_with_charge(entries, charge):
        """
        First entry in ``entries`` with the given charge state.
        """
        return next(e for e in entries if e.charge_state == charge)

    xy: dict = {}  # {defect_name: [[x_vals], [y_vals]]} for stable (ground-state) lines
    all_lines_xy: dict = {}  # as above, but for all entries (every charge state)
    y_range_vals: list[float] = []  # y-values at the x-limits, used to set the y-axis range
    all_entries_y_range_vals: list[float] = []  # y-values at the x-limits, used to set the y-axis range
    ymin = 0

    all_entries = _filter_by_defect_subset(defect_thermodynamics.all_entries, defect_subset)
    for defect_entry_list in all_entries.values():
        for defect_entry in defect_entry_list:
            # all_lines name includes charge state; rename in case of duplicate entry names:
            defect_name_w_charge, [all_lines_xy] = _rename_key_and_dicts(defect_entry.name, [all_lines_xy])
            all_lines_xy[defect_name_w_charge] = [
                [lower_cap, upper_cap],
                [_form_en(defect_entry, lower_cap), _form_en(defect_entry, upper_cap)],
            ]
            all_entries_y_range_vals.extend(_form_en(defect_entry, x_window) for x_window in xlim)

    transition_level_map = _filter_by_defect_subset(
        defect_thermodynamics.transition_level_map, defect_subset
    )
    for def_name, def_tl in transition_level_map.items():
        xy[def_name] = [[], []]
        stable_entries = defect_thermodynamics.stable_entries[def_name]

        if def_tl:
            org_x = sorted(def_tl.keys())
            # lower x-bound, from the line of the most positive (stable) charge state:
            first_entry = _entry_with_charge(stable_entries, max(def_tl[org_x[0]]))
            xy[def_name][0].append(lower_cap)
            xy[def_name][1].append(_form_en(first_entry, lower_cap))
            y_range_vals.append(_form_en(first_entry, xlim[0]))

            for fl in org_x:  # iterate over stable charge state transitions
                form_en = _form_en(_entry_with_charge(stable_entries, max(def_tl[fl])), fl)
                xy[def_name][0].append(fl)
                xy[def_name][1].append(form_en)
                y_range_vals.append(form_en)

            # upper x-bound, from the line of the most negative (stable) charge state:
            last_entry = _entry_with_charge(stable_entries, min(def_tl[org_x[-1]]))
            xy[def_name][0].append(upper_cap)
            xy[def_name][1].append(_form_en(last_entry, upper_cap))
            y_range_vals.append(_form_en(last_entry, xlim[1]))

        else:  # no transition level -> only one stable charge state, single line across the range:
            defect_entry = stable_entries[0]
            xy[def_name] = [
                [lower_cap, upper_cap],
                [_form_en(defect_entry, lower_cap), _form_en(defect_entry, upper_cap)],
            ]
            y_range_vals.extend(_form_en(defect_entry, x_window) for x_window in xlim)

        # if xy corresponds to a line below 0 for all x in (0, band_gap), warn!
        assert defect_thermodynamics.band_gap is not None  # typing

        in_gap_fermi_levels = np.linspace(0, defect_thermodynamics.band_gap, 1000)
        in_gap_formation_energies = np.interp(in_gap_fermi_levels, xp=xy[def_name][0], fp=xy[def_name][1])
        if all(y < 0 for y in in_gap_formation_energies):  # Check if all y-values are below zero
            warnings.warn(
                f"All formation energies for {def_name} are below zero across the entire band gap range. "
                f"This is typically unphysical (see docs), and likely due to mis-specification of "
                f"chemical potentials (see docstrings and/or tutorials)."
            )
            ymin = min(ymin, *in_gap_formation_energies)

    if not y_range_vals:
        raise ValueError("No formation energy data available to plot.")

    return (xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin


def _get_ylim_from_y_range_vals(
    y_range_vals: list[float], ymin: float = 0, auto_labels: bool = False
) -> tuple[float, float]:
    """
    Determine y-axis limits from the formation-energy ``y_range_vals``.

    Adds headroom above the data, and extra space for transition-level labels
    when ``auto_labels`` is ``True``.
    """
    window = max(y_range_vals) - min(*y_range_vals, ymin)
    spacer = 0.1 * window
    ylim = (ymin, max(y_range_vals) + spacer)
    if auto_labels:  # need to manually set xlim or ylim if labels cross axes!!
        # Increase y_limit to give space for transition level labels
        ylim = (ymin, max(y_range_vals) * 1.17) if spacer / ylim[1] < 0.145 else ylim

    return ylim


def formation_energy_plot(
    defect_thermodynamics: "DefectThermodynamics",
    abs_chempots: dict | None = None,
    el_refs: dict | None = None,
    all_entries: bool | str = False,
    include_site_info: bool | None = None,
    chempot_table: bool = True,
    defect_subset: list[str] | str | None = None,
    colormap: str | Colormap | None = None,
    linestyles: str | list[str] = "-",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fermi_level: float | None = None,
    title: str | None = None,
    auto_labels: bool = False,
    filename: PathLike | None = None,
) -> "mpl.figure.Figure":
    """
    Produce defect formation energy vs Fermi level plot (i.e. defect formation
    energy / transition level diagram).

    This function is not intended to be directly called. The recommended usage
    is :meth:`~doped.thermodynamics.DefectThermodynamics.plot()` -- see
    docstring for details.

    Args:
        defect_thermodynamics (|DefectThermodynamics|):
            |DefectThermodynamics| object containing defect entries to plot.
        abs_chempots (dict):
            Dictionary of ``{Element: value}`` giving the absolute chemical
            potential of each element.
        el_refs (dict):
            Dictionary of ``{Element: value}`` giving the reference energy of
            each element.
        all_entries (bool, str):
            Whether to plot the formation energy lines of `all` defect entries,
            rather than the default of showing only the equilibrium states at
            each Fermi level position (traditional). If instead set to "faded",
            will plot the equilibrium states in bold, and all unstable states
            in faded grey. (Default: False)
        include_site_info (bool, None):
            Whether to include site info in defect names in the plot legend
            (e.g. ``$Cd_{i_{C3v}}$`` rather than ``$Cd_{i}$``). If ``None``
            (default), site info is omitted unless needed to disambiguate
            non-grouped defects with the same name (i.e. inequivalent sites for
            the same defect type). If ``False``, site info is never included.
            If ``True``, site info is shown for all defect names. In all cases,
            if duplicate defect names remain, "-a", "-b", "-c" etc. are
            appended to the names to differentiate them.
        chempot_table (bool):
            Whether to print the chemical potential table above the plot.
            (Default: True)
        defect_subset (list[str], str):
            If provided, only defects whose name contains at least one of the
            given substrings are plotted (e.g. ``["v_", "Te_Cd"]`` would keep
            all vacancies plus ``Te_Cd``). A bare string is treated as a
            single-element list. (Default: ``None`` -- all defects)
        colormap (str, matplotlib.colors.Colormap):
            Colormap to use for the formation energy lines, either as a string
            (which can be a colormap name from
            https://matplotlib.org/stable/users/explain/colors/colormaps or
            from https://www.fabiocrameri.ch/colourmaps -- append 'S' if using
            a sequential colormap from the latter) or a ``Colormap`` /
            ``ListedColormap`` object. If ``None`` (default), uses ``tab10``
            with ``alpha=0.75`` (if 10 or fewer lines to plot), ``tab20`` (if
            20 or fewer lines) or ``batlow`` (if more than 20 lines).
        linestyles (str, list[str]):
            Linestyles to use for the formation energy lines, either as a
            single linestyle (``str``) or list of linestyles (``list[str]``) in
            the order of appearance of lines in the plot legend. Default is
            ``"-"``; i.e. solid linestyle for all entries.
        xlim:
            Tuple (min,max) giving the range of the x-axis (Fermi level). May
            want to set manually when including transition level labels, to
            avoid crossing the axes. Default is to plot from -0.3 to +0.3 eV
            above the band gap.
        ylim:
            Tuple (min,max) giving the range for the y-axis (formation energy).
            May want to set manually when including transition level labels, to
            avoid crossing the axes. Default is from 0 to just above the
            maximum formation energy value in the band gap.
        fermi_level (float):
            If set, plots a dashed vertical line at this Fermi level value,
            typically used to indicate the equilibrium Fermi level position.
            (Default: None)
        title (str):
            Title for the plot. (Default: None)
        auto_labels (bool):
            Whether to automatically label the transition levels with their
            charge states. If there are many transition levels, this can be
            quite ugly. (Default: False)
        filename (PathLike):
            Filename to save the plot to. (Default: None (not saved)).

    Returns:
        ``matplotlib`` ``Figure`` object.
    """
    _chempot_warning(abs_chempots)
    if defect_thermodynamics.band_gap is None:
        raise ValueError(
            "`band_gap` is not set for `DefectThermodynamics`, cannot plot formation energies."
        )

    if xlim is None:
        xlim = (-0.3, defect_thermodynamics.band_gap + 0.3)

    (xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin = _get_formation_energy_lines(
        defect_thermodynamics, abs_chempots, xlim, defect_subset=defect_subset
    )  # get formation energy lines data

    plotting_xy = all_lines_xy if all_entries is True else xy
    colors, linestyles = _get_TLD_colors_and_linestyles(colormap, linestyles, len(plotting_xy))

    # generate plot:
    styled_fig_size = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=((2.6 / 3.5) * styled_fig_size[0], (1.95 / 3.5) * styled_fig_size[1]))
    # Gives a final figure width matching styled_fig_size, with dimensions matching the doped default
    styled_font_size = plt.rcParams["font.size"]
    styled_linewidth = plt.rcParams["lines.linewidth"]
    styled_markersize = plt.rcParams["lines.markersize"]

    _plot_formation_energy_lines(  # plot formation energies
        plotting_xy,
        colors=colors,
        linestyles=linestyles,
        ax=ax,
        styled_linewidth=styled_linewidth,
        styled_markersize=styled_markersize,
    )

    if all_entries == "faded":  # plot after, so legend line colours are correct
        _plot_formation_energy_lines(  # grey 'all_lines_xy' not included in legend
            all_lines_xy,
            colors=[(0.8, 0.8, 0.8)] * len(all_lines_xy),
            linestyles=[
                "-",
            ]
            * len(all_lines_xy),
            ax=ax,
            styled_linewidth=styled_linewidth,
            styled_markersize=styled_markersize,
            alpha=0.5,
            zorder=0.5,  # plot behind other lines, but above band edges
        )

    _plot_transition_level_markers(
        ax,
        defect_thermodynamics,
        (xy).keys(),
        colors,
        abs_chempots,
        styled_markersize=styled_markersize,
        styled_font_size=styled_font_size,
        all_entries=all_entries,
        auto_labels=auto_labels,
    )

    legend_txt = format_defect_names(
        list((plotting_xy).keys()),
        include_charge=all_entries is True,
        include_site_info=include_site_info,
    )
    user_figsize_legend_fontsize_ratio = (plt.rcParams["figure.figsize"][1] / get_legend_font_size()) / (
        3.5 / 9
    )
    ax.legend(
        legend_txt,
        loc="upper left",  # (of bbox)
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,  # adjust padding to move closer to the axes
        # max 10 labels per column with default settings:
        ncol=np.ceil(len(legend_txt) / (10 * user_figsize_legend_fontsize_ratio)),
    )

    if ylim is None:
        ylim = _get_ylim_from_y_range_vals(
            all_entries_y_range_vals if all_entries is True else y_range_vals,
            ymin=ymin,
            auto_labels=auto_labels,
        )

    shade_band_edges(ax, defect_thermodynamics.band_gap, xlim, ylim, orientation="horizontal")
    _set_TLD_axis_labels_limits_ticks(ax, xlim, ylim)

    # dashed line for E_formation = 0 (in case ymin < 0) and Fermi level if provided:
    ax.plot([xlim[0], xlim[1]], [0, 0], c="k", ls="--", alpha=0.7)
    if fermi_level is not None:
        ax.axvline(x=fermi_level, linestyle="-.", color="k")

    if chempot_table and abs_chempots:
        plot_chemical_potential_table(ax, chempots=abs_chempots, el_refs=el_refs)

    _set_title_and_save_figure(ax, title, chempot_table, filename, styled_font_size)

    return fig


def plot_chemical_potential_table(
    ax: plt.Axes,
    chempots: dict[str, float],
    cellLoc: Literal["left", "center", "right"] = "left",
    el_refs: dict[str, float] | None = None,
) -> Table:
    """
    Plot a table of chemical potentials above the plot in ``ax``.

    Args:
        ax (plt.Axes):
            Axes object to plot the table in.
        chempots (dict):
            Dictionary of chemical potentials of the form ``{Element: value}``.
        cellLoc (str):
            Alignment of text in cells. Default is "left".
        el_refs (dict):
            Dictionary of elemental reference energies of the form
            ``{Element: value}``. If provided, the chemical potentials are
            given with respect to these reference energies.

    Returns:
        The ``matplotlib.table.Table`` object (which has been added to the
        ``ax`` object).
    """
    if el_refs is not None:
        chempots = {el: energy - el_refs[el] for el, energy in chempots.items()}
    labels = [rf"$\mathregular{{\mu_{{{s}}}}}$," for s in sorted(chempots.keys())]
    labels[0] = f"({labels[0]}"
    labels[-1] = f"{labels[-1][:-1]})"  # [:-1] removes trailing comma
    labels = ["Chemical Potentials", *labels, " Units:"]

    text_list = [f"{chempots[el]:.2f}," for el in sorted(chempots.keys())]

    # add brackets to first and last entries:
    text_list[0] = f"({text_list[0]}"
    text_list[-1] = f"{text_list[-1][:-1]})"  # [:-1] removes trailing comma
    if el_refs is not None:
        text_list = ["(wrt Elemental refs)", *text_list, "  [eV]"]
    else:
        text_list = ["(from calculations)", *text_list, "  [eV]"]
    widths = [0.1] + [0.9 / len(chempots)] * (len(chempots) + 2)
    tab = ax.table(cellText=[text_list], colLabels=labels, colWidths=widths, loc="top", cellLoc=cellLoc)
    tab.auto_set_column_width(list(range(len(widths))))

    for cell in tab.get_celld().values():
        cell.set_linewidth(0)
        cell.set_facecolor("none")  # make transparent as with rest of plot

    return tab


class TransitionLevel(NamedTuple):
    """
    A charge transition level (TL), between charge states ``q_pos`` and
    ``q_neg`` (``q_neg = q_pos - 1`` for single-electron TLs).

    ``charges = (q_pos, q_neg)`` (more positive, then more negative charge
    state); ``pos_meta``/``neg_meta`` flag whether the more-positive/negative
    charge state is metastable. ``TL_eV`` is the TL position in eV from the
    VBM. ``faded`` is ``True`` if the TL should be drawn faded in the vertical
    TL diagram (a rendering flag, left at its ``False`` default outside of
    plotting).

    Shared by |get_TLs| (used for single-electron TLs) and the TL plotting
    routines.
    """

    TL_eV: float
    charges: tuple[int, int]
    pos_meta: bool
    neg_meta: bool
    faded: bool = False


class TransitionLevelLabel(NamedTuple):
    """
    A plot position for a charge transition level (TL) label.

    ``(x, y)`` is the label anchor position with alignments ``ha``/``va``;
    ``label`` and ``label_w`` are the label text and width; ``TL_eV`` is the
    TL position in eV from the VBM (same as for :class`TransitionLevel`);
    ``conn_y`` and ``conn_x`` are the source TL line ``y``/column-edge ``x``
    for an off-column label that needs a connector (both ``None`` for an inline
    label with no connector).
    """

    x: float
    y: float
    ha: str
    va: str
    label: str
    label_w: float
    TL_eV: float
    conn_y: float | None = None
    conn_x: float | None = None


def _get_transition_level_data(
    defect_thermodynamics: "DefectThermodynamics",
    all: bool | str = False,
):
    """
    Collect transition level data for :func:`transition_level_diagram`.

    Returns ``{defect_name: [TransitionLevel]}``, with ``TransitionLevel``
    being a named tuple: ``(TL_eV, charges, pos_meta, neg_meta, faded)`` (see
    definition above), sorted by TL energy (Fermi level). ``faded`` is ``True``
    if the TL should be drawn faded, only used when ``all == "faded"``.

    Args:
        defect_thermodynamics (|DefectThermodynamics|):
            Source of TL data.
        all (bool, str):

            - ``False``: only thermodynamic ground-state TLs (from
              ``transition_level_map``). ``faded`` is always ``False``.
            - ``True``: all single-electron TLs. ``faded`` is always ``False``.
            - ``"faded"`` / ``"faded_labels"``: ground-state TLs (solid) plus
              single-electron TLs that involve at least one metastable charge
              state (these latter are marked ``faded=True``).
    """

    def sorted_tls(tl_list: list[TransitionLevel]) -> list[TransitionLevel]:
        return sorted(tl_list, key=lambda x: x.TL_eV)  # sort by TL position wrt VBM

    # ground-state TLs (i.e. those visible on the formation energy diagram):
    gs_per_defect: dict[str, list[TransitionLevel]] = {}
    for defect_name, tl_dict in defect_thermodynamics.transition_level_map.items():
        gs_per_defect[defect_name] = [
            TransitionLevel(float(TL), (max(chargeset), min(chargeset)), False, False, False)
            for TL, chargeset in tl_dict.items()
        ]

    if all is False:
        return {name: sorted_tls(tls) for name, tls in gs_per_defect.items()}

    single_electron_TLs: dict[str, list[TransitionLevel]] = (
        defect_thermodynamics._get_single_electron_tls()  # already ``TransitionLevel`` tuples
    )  # all single-electron TLs (consecutive charge pairs with ``q_neg = q_pos - 1``)

    # defect order = transition_level_map order (which respects defect appearance order), with any
    # defects that are only present in single_electron_TLs appended afterwards:
    ordered_names = list(
        dict.fromkeys([*defect_thermodynamics.transition_level_map, *single_electron_TLs])
    )  # dict.fromkeys ensures unique keys, so only unique keys from single_electron_TLs are appended

    if all is True:
        return {name: sorted_tls(single_electron_TLs.get(name, [])) for name in ordered_names}

    # all not ``True`` or ``False``; in {"faded", "faded_labels"}: ground-state TLs solid, metastable
    # single-electron faded
    out = {}
    for name in ordered_names:
        merged = list(gs_per_defect.get(name, []))  # GS, not faded
        merged += [
            tl._replace(faded=True)
            for tl in single_electron_TLs.get(name, [])
            if tl.pos_meta or tl.neg_meta
        ]
        out[name] = sorted_tls(merged)
    return out


def _format_TL_charge_label(charges, pos_meta=False, neg_meta=False, prefix=""):
    """
    Format a charge transition label like ``"(+1/0)"`` or ``"(+1*/0)"``.

    ``charges = (q_pos, q_neg)`` where ``q_pos`` is the more positive charge
    state. ``pos_meta``/``neg_meta`` denote whether the more-positive/negative
    charge state is metastable, in which case a ``*`` is appended to that
    charge. ``prefix`` is prepended to the label (e.g. ``"ε"`` for
    transition-level naming).
    """
    q_pos, q_neg = charges
    return (
        f"{prefix}({'+' if q_pos > 0 else ''}{q_pos}{'*' if pos_meta else ''}"
        f"/{'+' if q_neg > 0 else ''}{q_neg}{'*' if neg_meta else ''})"
    )


def _filter_by_defect_subset(defect_dict: dict, defect_subset: list[str] | str | None) -> dict:
    """
    Filter a ``{defect_name: ...}`` dict to defects whose name contains at
    least one of the substrings in ``defect_subset``.

    If ``defect_subset`` is ``None`` (or empty), returns ``defect_dict``
    unchanged. A bare string is treated as a single-element list.
    """
    if not defect_subset:
        return defect_dict
    if isinstance(defect_subset, str):
        defect_subset = [defect_subset]
    return {k: v for k, v in defect_dict.items() if any(s in k for s in defect_subset)}


# rough character width as a fraction of font size (in points):
_CHAR_WIDTH_FRAC = 0.55

# assumed max width (in characters) of a TL charge label, for sizing the side padding around outer columns
_LABEL_MAX_CHARS = 9  # e.g. (-2*/-1*)


def _estimate_label_width(fontsize: float, n_chars: float, data_width: float, fig_width: float) -> float:
    """
    Estimate the horizontal extent of a text label, in data units (which is
    just ``1`` per defect for vertical TL diagram plots).

    The label width in inches is approximated as ``n_chars`` characters, each
    ~``_CHAR_WIDTH_FRAC * fontsize`` points wide (divided by 72 to convert
    points to inches), then scaled by the data-to-figure width ratio
    (``data_width / fig_width``) to convert inches to data units.
    """
    width_inches = n_chars * _CHAR_WIDTH_FRAC * fontsize / 72.0
    return width_inches * data_width / max(fig_width, 1.0)


def _label_axis_extent(x: float, alignment: str = "left", label_size: float = 8.1) -> tuple[float, float]:
    """
    Horizontal or vertical extent (``x_min``, ``x_max``)/(``y_min``, ``y_max``)
    of a label centred or anchored at ``x``, with alignment ``alignment`` and
    size (width/height) ``label_size``.
    """
    if alignment in ["bottom", "left"]:
        return x, x + label_size  # text extends to the right of the anchor
    if alignment in ["top", "right"]:
        return x - label_size, x  # text extends to the left of the anchor
    return x - 0.5 * label_size, x + 0.5 * label_size  # "center"


def _label_box(
    x_pos: float, y_pos: float, ha: str, va: str, label_width: float, label_height: float
) -> tuple[float, float, float, float]:
    """
    Bounding box ``(x_min, x_max, y_min, y_max)`` of a label anchored at
    ``(x_pos, y_pos)`` with the given alignments and dimensions.
    """
    x_min, x_max = _label_axis_extent(x_pos, ha, label_width)
    y_min, y_max = _label_axis_extent(y_pos, va, label_height)
    return x_min, x_max, y_min, y_max


def _TL_label_box(
    TL_label: TransitionLevelLabel,
    label_height: float = 0.0,
    label_width: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Bounding box ``(x_min, x_max, y_min, y_max)``, from ``_label_box`` for a
    given ``TransitionLevelLabel``, ``label_height`` and ``label_width``
    (defaults to ``TL_label.label_w``).
    """
    return _label_box(
        TL_label.x, TL_label.y, TL_label.ha, TL_label.va, label_width or TL_label.label_w, label_height
    )


def _boxes_overlap(b1: tuple, b2: tuple, x_buf: float = 0.0, y_buf: float = 0.0) -> bool:
    """
    Whether two label boxes ``(x_min, x_max, y_min, y_max)`` overlap.

    Label boxes are additionally treated as overlapping if they are within
    ``x_buf``/``y_buf`` of each other horizontally/vertically.
    """
    return (
        b1[1] + x_buf > b2[0] and b1[0] - x_buf < b2[1] and b1[3] + y_buf > b2[2] and b1[2] - y_buf < b2[3]
    )


def _box_overlap_fraction(b1: tuple, b2: tuple, x_buf: float = 0.0, y_buf: float = 0.0) -> float:
    """
    Quantify how much two label boxes ``(x_min, x_max, y_min, y_max)`` overlap.

    Returns the overlapping area as a fraction (``0.0``--``1.0``) of the smaller
    box's area, padding each box by ``x_buf``/``y_buf`` (consistent with
    :func:`_boxes_overlap`). ``0.0`` means no overlap, a small value means a
    slight clip, and ``1.0`` means one box fully covers the other. This lets
    the label placement optimiser prefer slight overlaps over full overlaps
    when some overlap is unavoidable.
    """
    dx = min(b1[1], b2[1]) - max(b1[0], b2[0]) + x_buf
    if dx <= 0:
        return 0.0  # break early
    dy = min(b1[3], b2[3]) - max(b1[2], b2[2]) + y_buf
    if dy <= 0:
        return 0.0  # break early
    # determine smaller box, assuming label height is equal:
    xrange1 = b1[1] - b1[0]
    xrange2 = b2[1] - b2[0]
    smaller_area = max(min(xrange1, xrange2) * (b1[3] - b1[2]), 1e-4)
    return min(dx * dy / smaller_area, 1.0)


def _connector_x0(x_pos: float, x_center: float, TL_line_left: float, TL_line_right: float) -> float:
    """
    X-coordinate at which an off-column label's connector meets its column; the
    near column edge (or the centre for a centred label).
    """
    return x_center if x_pos == x_center else (TL_line_right if x_pos > x_center else TL_line_left)


def _connector_endpoints(
    TL_label: TransitionLevelLabel,
) -> tuple[float, float, float, float] | None:
    """
    Connector segment endpoints ``(x0, y0, x1, y1)`` for an off-column label,
    running from the TL column edge ``(conn_x, conn_y)`` to the label anchor
    ``(x, y)``, or ``None`` for an inline label (``conn_y is None``, no
    connector).
    """
    if TL_label.conn_y is None:
        return None
    assert TL_label.conn_x is not None  # set alongside conn_y when the candidate is built
    return TL_label.conn_x, TL_label.conn_y, TL_label.x, TL_label.y


def _segment_intersects_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rx_min: float,
    rx_max: float,
    ry_min: float,
    ry_max: float,
) -> bool:
    """
    Liang-Barsky test: does the line segment ``(x0,y0)-(x1,y1)`` intersect the
    axis-aligned rectangle ``[rx_min, rx_max] x [ry_min, ry_max]``?
    """
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - rx_min), (dx, rx_max - x0), (-dy, y0 - ry_min), (dy, ry_max - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False  # parallel and outside
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


# --- Tuning constants for vertical transition-level (TL) diagram label placement ---
# These govern how `transition_level_diagram()` spaces/offsets TL charge labels relative to the TL lines
# and to each other. Each ``*_FRAC`` value is a fraction of a label's height (``label_h``, in eV).
_SIDE_LABEL_X_GAP = 0.06  # horizontal gap (x-units) from a column edge to an off-column label
_DIAG_LABEL_DY_FRAC = 1.6  # vertical offset of a diagonal side label from its TL line
_DIRECT_LABEL_DY_FRAC = 0.4  # anchor offset of a direct above/below label from its TL line
_LABEL_STACK_Y_BUFFER_FRAC = 0.3  # extra vertical buffer enforced between two stacked labels
_TL_LINE_EPS_FRAC = 0.05  # half-thickness used to treat a TL line as a thin rectangle
_TL_LINE_CLEARANCE_FRAC = 0.75  # required clearance of a label from a neighbouring (non-source) TL line
# Side-placement overlap penalties (integer cost-table weights). The box penalty is scaled by the overlap
# fraction (0 - 1; see ``_box_overlap_fraction``), while the connector penalty is fixed. Their ratio
# (10:4) sets box-vs-connector severity:
_LABEL_OVERLAP_PENALTY = 100  # cost weight for a *full* label box-box overlap (scaled by overlap fraction)
_CONNECTOR_OVERLAP_PENALTY = 40  # cost for a connector line crossing a label box


class _SideBoundTL(NamedTuple):
    """
    A side-bound TL (one whose label could not be placed inline) awaiting side
    label placement.

    ``col_idx`` is the defect column index and ``idx_in_col`` is the position
    of this TL within that column's results list (so the chosen label position
    can be written back). ``tl_eV`` is the TL energy (used for locality
    clustering), and ``candidates`` is its off-column
    :class:`TransitionLevelLabel` positions to choose from.
    """

    col_idx: int
    idx_in_col: int
    TL_eV: float
    candidates: list[TransitionLevelLabel]


def _cluster_side_bound_tls(
    side_bound: list[_SideBoundTL], y_threshold: float
) -> list[list[_SideBoundTL]]:
    """
    Group side-bound TLs into independent clusters by locality.

    Two side-bound TLs are linked (placed in the same cluster) if they sit in
    the same or a neighbouring defect column (``abs(col_i - col_j) <= 1``) and
    are within ``y_threshold`` of each other vertically
    (``abs(tl_eV_i - tl_eV_j) <= y_threshold``); clusters are the connected
    components (i.e. transitive closure) of these links.

    Only same-or-neighbouring columns can have overlapping side labels (their
    label boxes/connectors are confined to the column edges), and TLs separated
    by more than ``y_threshold`` vertically cannot overlap either, so distinct
    clusters never interact and can be optimised independently (see
    :func:`_optimise_side_placements`).

    Args:
        side_bound (list[_SideBoundTL]):
            The side-bound TLs to cluster (across all columns).
        y_threshold (float):
            Maximum vertical separation (in eV) for two TLs in neighbouring
            columns to be considered as possibly interacting.

    Returns:
        list[list[_SideBoundTL]]:
            A list of clusters, each a list of :class:`_SideBoundTL`.
    """
    n = len(side_bound)
    if n == 0:
        return []

    # build the link (adjacency) matrix from the proximity rule, then take its connected components:
    cols = np.array([sb.col_idx for sb in side_bound])
    tls = np.array([sb.TL_eV for sb in side_bound])
    linked = (np.abs(cols[:, None] - cols[None, :]) <= 1) & (
        np.abs(tls[:, None] - tls[None, :]) <= y_threshold
    )
    n_clusters, labels = connected_components(csr_matrix(linked), directed=False)

    clusters: list[list[_SideBoundTL]] = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(side_bound[i])
    return clusters


def _optimise_side_placements(
    side_candidates_per_tl: list[list[TransitionLevelLabel]],
    label_height: float,
    max_brute_force_ops: int = 6_000_000_000,
) -> list[TransitionLevelLabel]:
    r"""
    Pick one label position per side-bound-label TL so as to minimise total
    overlap cost.

    For each TL we have several candidate ``TransitionLevelLabel``
    (``(x, y, ha, va, conn_y)``) positions (``side_candidates_per_tl``). The
    cost of an assignment is the sum of pairwise overlap penalties between
    label boxes and connector lines. The combination minimising the total cost
    is returned.

    A greedy first-pick plus a few hill-climbing refinement passes is run
    first. If this achieves zero total overlap cost, it corresponds to a global
    optimum and is returned immediately -- the common case, since candidate
    positions are generated precisely to allow overlap-free placements. Only
    when residual overlap remains, we run an exact brute-force enumeration
    (vectorised with ``numpy``), and only if the estimated work is small
    enough; otherwise the greedy result is kept. The work estimate is
    ``space * (n*(n-1)/2)`` integer cost-table reads (the per-combination
    ``n*(n-1)/2`` pairwise sums, ``space`` being the product of candidate
    counts and ``n`` the number of side-bound-label TLs); the numpy build
    sustains ~3 G reads/s on a modern laptop, so the default
    ``max_brute_force_ops`` (with early skipping of independent i-j pairs;
    ~50% cost reduction) keeps worst-case brute force to ~1 s.

    Args:
        side_candidates_per_tl (list[list[TransitionLevelLabel]]):
            A list of lists of :class:`TransitionLevelLabel`s; one inner list
            per side-bound TL, containing its candidate
            :class:`TransitionLevelLabel` positions to choose from.
        label_height (float):
            Vertical height of a label in y-units (eV), used for stacking and
            collision buffers.
        max_brute_force_ops (int):
            Maximum estimated cost-table reads (see above) permitted for
            brute-force enumeration; above this the greedy plus hill-climbing
            fallback is used instead.

    Returns:
        list[TransitionLevelLabel]:
            The chosen label position for each side-bound TL, in the same order
            as ``side_candidates_per_tl``.
    """
    n = len(side_candidates_per_tl)
    if n == 0:
        return []
    counts = [len(c) for c in side_candidates_per_tl]
    space = math.prod(counts)
    per_combo_reads = n * (n - 1) // 2  # ``space * (n*(n-1)/2 pairwise)`` cost-table reads

    # add a small y-buffer (~30% of a label height) so two side labels packed almost touch-to-touch on
    # the same side are treated as overlapping (the same buffer applied to inline label-vs-label checks):
    y_buf = _LABEL_STACK_Y_BUFFER_FRAC * label_height

    # Precompute the box and connector geometry for each candidate, so the search below works purely
    # with cached coordinates and integer cost tables:
    cand_boxes = [[_TL_label_box(c, label_height) for c in cands] for cands in side_candidates_per_tl]
    cand_conns = [[_connector_endpoints(c) for c in cands] for cands in side_candidates_per_tl]

    def _pair_cost(i: int, a: int, j: int, b: int) -> int:
        """
        Pairwise cost between side pick ``a`` of TL ``i`` and pick ``b`` of TL
        ``j``: box-box overlap (scaled by how much the boxes overlap) plus
        `either` pick's connector crossing the `other` pick's label box
        (checked symmetrically).
        """
        box_i = cand_boxes[i][a]
        box_j = cand_boxes[j][b]
        # scale the box-overlap penalty by the overlap fraction, with `any` non-zero overlap costing at
        # least 1 (equivalent to 1% overlap):
        frac = _box_overlap_fraction(box_i, box_j, y_buf=y_buf)
        cost = 0 if frac == 0 else max(1, round(_LABEL_OVERLAP_PENALTY * frac))
        for conn, box in [(cand_conns[i][a], box_j), (cand_conns[j][b], box_i)]:
            if conn is not None and _segment_intersects_rect(*conn, *box):
                cost += _CONNECTOR_OVERLAP_PENALTY  # connector - label box overlap
        return cost

    # tabulate pairwise costs for all i < j as ``pair_lookup[(i, j)][a][b]``:
    pair_lookup = {
        (i, j): np.array(
            [[_pair_cost(i, a, j, b) for b in range(counts[j])] for a in range(counts[i])],
            dtype=np.int32,
        )
        for i in range(n)
        for j in range(i + 1, n)
    }

    # Greedy first-pick plus a few hill-climbing passes, reusing the precomputed geometry. The cost of
    # giving TL ``i`` pick ``a`` given the other chosen picks is the sum of the tabulated pairwise
    # penalties (symmetric, so the ``i > j`` case just transposes the lookup):
    def _conditional_cost(i: int, a: int, picks: list[int]) -> int:
        """
        Cost of pick ``a`` of TL ``i`` given the current other TL ``picks``
        (ignoring unset picks).
        """
        cost = 0
        for j, b in enumerate(picks):
            if b < 0 or j == i:  # b < 0 if not yet chosen, and ignore i-i pairs
                continue
            cost += pair_lookup[(i, j)][a][b] if i < j else pair_lookup[(j, i)][b][a]
        return int(cost)

    picks = [-1] * n  # -1 == not yet chosen
    for i in range(n):  # greedy first-pick: lowest-cost candidate given prior picks
        best_idx, best_cost = 0, float("inf")
        for a in range(counts[i]):
            c = _conditional_cost(i, a, picks)
            if c < best_cost:
                best_idx, best_cost = a, c
        picks[i] = best_idx

    for _ in range(max(n, 20)):  # hill-climbing refinement: swap to a lower-cost alternative until stable
        improved = False
        for i in range(n):
            best_alt, best_alt_cost = None, _conditional_cost(i, picks[i], picks)  # current best
            for a in range(counts[i]):  # for all other possible choices
                if a != picks[i]:  # if not already the current choice
                    c = _conditional_cost(i, a, picks)
                    if c < best_alt_cost:
                        best_alt, best_alt_cost = a, c
            if best_alt is not None:
                picks[i] = best_alt
                improved = True
        if not improved:  # no further improvement, break
            break

    greedy_cost = sum(int(mat[picks[i]][picks[j]]) for (i, j), mat in pair_lookup.items())
    if greedy_cost == 0 or space * per_combo_reads > max_brute_force_ops:
        return [side_candidates_per_tl[k][picks[k]] for k in range(n)]  # use greedy result

    # Brute force via numpy broadcasting: build the full cost grid (one axis per TL, candidate index along
    # that axis) by adding each (non-zero) pairwise table along its two axes, then take the global argmin:
    grid = np.zeros(counts, dtype=np.int32)  # n dimensions, each of length = num candidates for that TL
    for (i, j), mat in pair_lookup.items():  # Note: Could possibly vectorise for efficiency if required...
        if not mat.any():  # independent pair (no overlap for any candidate combination), nothing to add
            continue
        shape = [1] * n
        shape[i] = counts[i]  # each dimension of length i = num candidates per TL (counts[i])
        shape[j] = counts[j]  # i < j in pair_items, so need both to fully determine shape
        grid += mat.reshape(shape)  # Note: this grid building and reshaping dominates brute-force cost
    # argmin gives a flat index; unravel to per-axis candidate indices (one per TL / grid axis) to map back
    best_indices = np.unravel_index(int(np.argmin(grid)), grid.shape)
    # best_indices -> length n (corresponding to n dimensions/TLs to pick labels) w/indices of best
    # labels for each TL; return these optimal label choices:
    return [side_candidates_per_tl[k][int(best_indices[k])] for k in range(n)]


def _place_inline_labels_for_column(
    tls: list[TransitionLevel],
    x_center: float,
    half_w: float,
    band_gap: float,
    ylim: tuple[float, float],
    xlim: tuple[float, float],
    label_height: float,
    per_char_label_width: float,
    neighbor_columns: list[tuple[float, float, list[float]]] | None = None,
    skip_faded: bool = True,
    label_y_max: float | None = None,
    label_y_min: float | None = None,
) -> tuple[list[TransitionLevelLabel | None], list[_SideBoundTL]]:
    r"""
    Place the inline (directly above/below) labels for one defect column, and
    collect the remaining "side-bound" TLs for later global side placement.

    For each TL (in input order) the label is tried directly above, then
    directly below the TL line; the first such inline position that doesn't
    overlap another TL line in the same column, an already-placed (this-column)
    label, or a band edge (VBM at 0 eV, CBM at ``band_gap``) is committed. Any
    TL that cannot be placed inline becomes a :class:`_SideBoundTL` (with its
    off-column candidate positions generated here), returned for
    :func:`transition_level_diagram` to cluster and optimise globally via
    :func:`_optimise_side_placements`.

    Args:
        tls (list[TransitionLevel]):
            List of :class:`TransitionLevel` entries (for which to generate
            :class:`TransitionLevelLabel`s; i.e. label placements).
        x_center (float):
            X-coordinate of the centre of the defect column.
        half_w (float):
            Half-width of the TL lines in the column (in (normalised) x-units).
        band_gap (float):
            Band gap in eV (CBM position; VBM is at 0 eV), used as a collision
            boundary for label placement.
        ylim (tuple[float, float]):
            ``(y_min, y_max)`` axis limits in eV.
        xlim (tuple[float, float]):
            ``(x_min, x_max)`` axis limits in (normalised) x-units.
        label_height (float):
            Vertical height of a label in y-units (eV), used (only) for
            stacking and collision checks -- not directly used in plotting.
        per_char_label_width (float):
            Reference width per character (in (normalised) x-units, where each
            column has a width of ``1``) of TL labels, to be scaled per-label
            by the actual character count for collision checks.
        neighbor_columns (list[tuple[float, float, list[float]]] | None):
            List of ``(x_center, half_w, TL_y_positions)`` tuples for
            neighbouring columns, used to determine preference for direct right
            or left side label placement (i.e. to choose the 'spacier' side).
        skip_faded (bool):
            If ``True``, faded TLs do not get labels (their lines are still
            part of collision checks however). Their entries in the returned
            ``results`` list are ``None``.
        label_y_max (float | None):
            If provided, labels are not allowed to extend above this ``y``, so
            they cannot collide with the column header above ``ylim[1]``.
        label_y_min (float | None):
            If provided, labels are not allowed to extend below this ``y``.

    Returns:
        tuple[list[TransitionLevelLabel | None], list[_SideBoundTL]]:

            - ``results``: One entry per TL in input order, holding the
              committed inline :class:`TransitionLevelLabel` or ``None``
              (skipped/faded TLs, and side-bound TLs whose positions are filled
              in later by the caller).
            - ``side_bound``: One :class:`_SideBoundTL` per TL that could not
              be placed inline (with off-column candidate positions), for
              global clustering and side placement. ``col`` is left as ``-1``
              here and set by the caller for index tracking.
    """
    TL_y_positions = [tl.TL_eV for tl in tls]
    TL_line_left = x_center - half_w
    TL_line_right = x_center + half_w
    side_x_right = TL_line_right + _SIDE_LABEL_X_GAP
    side_x_left = TL_line_left - _SIDE_LABEL_X_GAP

    # labels may extend up to `label_y_max` (slightly past ylim[1], into the buffer below the column
    # header) and down to `label_y_min` (slightly past ylim[0]); this lets TLs that sit in/near the CBM
    # (orange) or VBM (blue) zones have their labels placed directly above/below their line:
    y_max_allowed = label_y_max if label_y_max is not None else ylim[1]
    y_min_allowed = label_y_min if label_y_min is not None else ylim[0]

    # ``placed`` accumulates this column's committed inline label boxes for in-column collision checks
    placed: list[TransitionLevelLabel] = []
    results: list[TransitionLevelLabel | None] = []

    def collides_with_band(TL_label: TransitionLevelLabel) -> bool:
        """
        Whether a label straddles a band edge or exceeds the plot bounds.
        """
        lbl_left, lbl_right, y_min, y_max = _TL_label_box(TL_label, label_height)
        if y_min < band_gap < y_max or y_min < 0.0 < y_max:  # straddles band edge
            return True
        if y_max > y_max_allowed or y_min < y_min_allowed:  # exceeds plot y-axis bounds:
            return True
        return lbl_left < xlim[0] or lbl_right > xlim[1]  # exceeds plot x-axis bounds?

    def collides_with_tl_line(TL_label: TransitionLevelLabel, source_y: float) -> bool:
        """
        Whether a label collides with a TL line (excluding ``source_y``).
        """
        lbl_left, lbl_right, y_min, y_max = _TL_label_box(TL_label, label_height)
        if not (lbl_right > TL_line_left and lbl_left < TL_line_right):  # only check TL lines in column
            return False

        # require 75% label-height of clearance from the (next) TL line so a label placed
        # direct-above/below is visually unambiguous (couldn't be mis-read as belonging to nearby TL
        # above/below). `source_y` is the TL we're labelling, so it is excluded from this check (it sits
        # _DIRECT_LABEL_DY_FRAC*label_height from the anchor):
        clearance = _TL_LINE_CLEARANCE_FRAC * label_height
        # for TL labels above the TL, we don't worry about other TL lines below (which may still fall
        # within the 'clearance' window), and vice versa for TL lines below the source TL:
        y_min = y_min - clearance if TL_label.y < source_y else source_y
        y_max = y_max + clearance if TL_label.y > source_y else source_y
        # check collision with any other TL (except source TL):
        return any(y_min <= TL_y <= y_max for TL_y in [ly for ly in TL_y_positions if ly != source_y])

    def collides_with_placed(TL_label: TransitionLevelLabel) -> bool:
        """
        Whether a label overlaps an already-placed label.
        """
        y_buf = _LABEL_STACK_Y_BUFFER_FRAC * label_height  # small vertical buffer; ~30% label_height
        box = _TL_label_box(TL_label, label_height)
        return any(_boxes_overlap(box, _TL_label_box(p, label_height), y_buf=y_buf) for p in placed)

    def _side_candidates(TL_label: TransitionLevelLabel) -> list[TransitionLevelLabel]:
        """
        Return the off-column candidate positions for one TL: direct left/right
        (no connector) plus four diagonal positions (with connector).

        Ordered with the spacier side first.
        """
        # determine spacier side (side with greatest distance to other TLs/labels):
        closest_TL_dist_right_side: float = abs(xlim[1] - x_center)
        closest_TL_dist_left_side: float = abs(x_center - xlim[0])
        if neighbor_columns:
            for neighbour_x, _neighbour_half_w, ny_list in neighbor_columns:
                if not ny_list:
                    continue  # no TLs in this neighbouring column -> no constraint from it
                # Euclidean distance from this column's edge to the closest TL in the neighbouring column:
                min_TL_dist = np.hypot(
                    neighbour_x - x_center, min(abs(ny - TL_label.TL_eV) for ny in ny_list)
                )
                if neighbour_x > x_center:
                    closest_TL_dist_right_side = min(closest_TL_dist_right_side, min_TL_dist)
                else:
                    closest_TL_dist_left_side = min(closest_TL_dist_left_side, min_TL_dist)

        right_first = closest_TL_dist_right_side >= closest_TL_dist_left_side

        # connector source x (column edge) for right/left diagonal candidates, fixed by their x:
        conn_x_right = _connector_x0(side_x_right, x_center, TL_line_left, TL_line_right)
        conn_x_left = _connector_x0(side_x_left, x_center, TL_line_left, TL_line_right)
        direct_right = TL_label._replace(x=side_x_right, y=TL_label.TL_eV, ha="left", va="center")
        direct_left = direct_right._replace(x=side_x_left, ha="right", va="center")
        # diagonal candidates are the direct side labels nudged up/down by ``diag_dy``, with a connector
        # back to the source TL line (``conn_y``/``conn_x`` set):
        diag_right = direct_right._replace(conn_y=TL_label.TL_eV, conn_x=conn_x_right)
        diag_left = direct_left._replace(conn_y=TL_label.TL_eV, conn_x=conn_x_left)
        diag_dy = _DIAG_LABEL_DY_FRAC * label_height  # diagonal y-offset (+/- on either side)
        return [
            direct_right if right_first else direct_left,
            direct_left if right_first else direct_right,
            diag_right._replace(y=TL_label.TL_eV + diag_dy),
            diag_left._replace(y=TL_label.TL_eV + diag_dy),
            diag_right._replace(y=TL_label.TL_eV - diag_dy),
            diag_left._replace(y=TL_label.TL_eV - diag_dy),
        ]

    def _candidate_ok(TL_label: TransitionLevelLabel, source_y: float) -> bool:
        """
        Whether a candidate label position is acceptable.

        Rejects band/figure-edge overlaps as well as TL-line, placed-label and
        connector collisions.
        """
        return not (
            collides_with_band(TL_label)
            or collides_with_tl_line(TL_label, source_y=source_y)
            or collides_with_placed(TL_label)
        )

    # ----- Try direct above / below for each TL (inline, greedy, cheap) -----
    # Anything that doesn't fit cleanly inline becomes a "side-bound" TL, returned for the caller to
    # cluster and optimise globally (its off-column candidate positions are generated here).
    side_bound: list[_SideBoundTL] = []
    for i, tl in enumerate(tls):  # ``TransitionLevel`` objects
        # add ``None`` as placeholder; kept as ``None`` for unlabelled faded TLs and side-bound (the latter
        # of which are overwritten later by this function's caller), and overwritten for inline labels here
        results.append(None)
        if skip_faded and tl.faded:
            continue
        label = _format_TL_charge_label(tl.charges, pos_meta=tl.pos_meta, neg_meta=tl.neg_meta)
        label_w = per_char_label_width * len(label)
        above_TL_y = tl.TL_eV + _DIRECT_LABEL_DY_FRAC * label_height  # y-value for label directly above TL
        TL_label = TransitionLevelLabel(x_center, above_TL_y, "center", "bottom", label, label_w, tl.TL_eV)

        chosen = None
        for cand in (
            TL_label,
            TL_label._replace(y=tl.TL_eV - _DIRECT_LABEL_DY_FRAC * label_height, va="top"),
        ):  # direct above, then direct below
            if _candidate_ok(cand, source_y=tl.TL_eV):
                chosen = cand  # suitable inline candidate label
                break
        if chosen is not None:  # add to results and placed, then move to next TL in loop
            results[i] = chosen
            # add to ``placed`` to use for same-column above/below TL collision checks (later in this loop)
            placed.append(chosen)
            continue

        # no inline placement: build off-column candidate positions, dropping any that would straddle a
        # band edge or fall outside the figure. A TL left with no valid candidate is skipped entirely (set
        # to ``None``) rather than being forced into a band-/figure-edge-violating position,
        # but a warning is thrown to notify the user:
        if opts := [c for c in _side_candidates(TL_label) if not collides_with_band(c)]:
            side_bound.append(_SideBoundTL(-1, i, TL_label.TL_eV, opts))  # ``col`` set by the caller
        else:
            warnings.warn(
                f"Could not automatically find a suitable label position for {label}! It will be omitted, "
                f"and an appropriate labelling can be added manually."
            )

    return results, side_bound


def transition_level_diagram(
    defect_thermodynamics: "DefectThermodynamics",
    all: bool | str = "faded",
    defect_subset: list[str] | str | None = None,
    include_site_info: bool | None = None,
    ylim: tuple[float, float] | None = None,
    show_charge_labels: bool = True,
    show_band_labels: bool | None = None,
    label_fontsize: float | None = None,
    column_width: float = 0.4,
    figsize: tuple[float, float] | None = None,
    filename: PathLike | None = None,
):
    r"""
    Produce a vertical transition level diagram for a |DefectThermodynamics|
    object, with one column per defect and short horizontal lines marking each
    charge transition level position within the host band gap.

    The valence band maximum (``self.vbm``) is at 0 eV (blue shaded region) and
    the conduction band minimum (``self.vbm + self.band_gap``) is shown in the
    orange shaded region at the top of the plot. Within each defect column,
    each transition level is drawn as a short horizontal line, labelled with
    the charge state transition (e.g. ``(+1/0)``)(if ``show_charge_labels`` is
    ``True``; default). Metastable charge states are denoted with a ``*`` in
    the label, as in the |DefectThermodynamics| methods.

    Args:
        defect_thermodynamics (|DefectThermodynamics|):
            |DefectThermodynamics| object containing the defects to plot.
        all (bool, str):
            Controls inclusion of single-electron transition levels involving
            metastable defect charge states (denoted with ``*`` in the
            labels). Mostly equivalent to ``all`` in |get_TLs|. Allowed values:

            - ``"faded"`` (default): show all single-electron TLs, with
              metastable-containing TLs drawn as faded lines `without`
              labels (keeps the plot uncluttered).
            - ``"faded_labels"``: same as ``"faded"`` but `with` labels
              drawn for the faded metastable TLs too.
            - ``True``: show all single-electron TLs at full opacity.
            - ``False``: show only the thermodynamic ground-state transition
              levels (i.e. those visible on the standard defect formation
              energy diagram).
        defect_subset (list[str], str):
            If provided, only defects whose name contains at least one of the
            given substrings are plotted (e.g. ``["v_", "Te_Cd"]`` would keep
            all vacancies plus ``Te_Cd``). A bare string is treated as a
            single-element list. (Default: ``None`` -- all defects)
        include_site_info (bool, None):
            Whether to include site info in defect names in the column headers
            (e.g. ``$V_{Cd_{Td}}$`` rather than ``$V_{Cd}$``). If ``None``
            (default), site info is omitted unless needed to disambiguate
            non-grouped defects with the same name (i.e. inequivalent sites for
            the same defect type). If ``False``, site info is never included.
            If ``True``, site info is shown for all defect names. In all cases,
            if duplicate defect names remain, "-a", "-b", "-c" etc. are
            appended to the names to differentiate them.
        ylim (tuple):
            Energy axis limits in eV (relative to VBM at 0). Defaults to
            ``(-0.05 * band_gap, 1.05 * band_gap)``.
        show_charge_labels (bool):
            Whether to label each transition level with its charge states
            (e.g. ``"(+1/0)"``). Defaults to ``True``.
        show_band_labels (bool):
            Whether to draw the "VBM" and "CBM" labels in the blue/orange
            band-edge shaded zones. If ``None`` (default), they are shown only
            if they would not overlap any transition level label (with the
            right side of the plot tried first, then the left); if both sides
            would clash they are hidden. ``True`` forces them on the right;
            ``False`` hides them.
        label_fontsize (float):
            Font size for the charge transition level labels. Defaults to ~90%
            of the current ``font.size`` rcParam. Can be a useful parameter to
            tune for busy plots.
        column_width (float):
            Width (in axes units) of the horizontal line segments inside each
            defect column, on a scale where the column spacing is 1. Defaults
            to ``0.4``. Can be a useful parameter to tune for busy plots.
        figsize (tuple):
            ``(width, height)`` of the figure in inches. Defaults to a width
            that scales with the number of defects.
        filename (PathLike):
            If set, save the figure to this path. (Default: None)

    Returns:
        ``matplotlib`` ``Figure`` object.
    """
    if defect_thermodynamics.band_gap is None:
        raise ValueError("`DefectThermodynamics.band_gap` is not set, cannot plot transition levels.")

    if all not in (False, True, "faded", "faded_labels"):
        raise ValueError(f"`all` must be False, True, 'faded' or 'faded_labels', not {all!r}")

    # get TL data:
    tl_data = _get_transition_level_data(defect_thermodynamics, all=all)
    tl_data = _filter_by_defect_subset(tl_data, defect_subset)
    if not tl_data:
        defect_subset_info = f" (after `defect_subset={defect_subset!r}` filter)" if defect_subset else ""
        raise ValueError(f"No defects with transition levels to plot{defect_subset_info}.")

    # setup axis limits and figure/label sizing:
    if ylim is None:
        margin = max(0.05 * defect_thermodynamics.band_gap, 0.05)
        ylim = (-margin, defect_thermodynamics.band_gap + margin)

    n_defects = len(tl_data)
    half_w = column_width / 2.0
    styled_font_size = plt.rcParams["font.size"]
    label_fontsize = label_fontsize or (styled_font_size * 0.9)

    # estimate label horizontal extent (in data units = column spacing) so we can extend xlim to leave
    # room for labels at the sides of the outer columns. The data width is approximated by ``n_defects``
    # (columns are spaced 1 data-unit apart) as ``xlim`` is not yet known, assuming ~7-character labels:
    if figsize is None:
        styled_figsize = plt.rcParams["figure.figsize"]
        figsize_w = max(styled_figsize[0], n_defects + 1.0)
        figsize_h = styled_figsize[1] * 1.15
    else:
        figsize_w, figsize_h = figsize
    label_width_est = _estimate_label_width(label_fontsize, _LABEL_MAX_CHARS, float(n_defects), figsize_w)
    side_pad = half_w + label_width_est + 0.1

    # y-axis ticks point inward (when ytick.direction is "in"/"inout"; default in ``doped`` style) so the
    # left-most labels need some extra clearance to avoid overlapping with tick marks
    left_extra_pad = 0.15 if plt.rcParams["ytick.direction"] in ("in", "inout") else 0.0
    if figsize is None:
        # widen the figure to accommodate the side padding without squishing column spacing:
        figsize = (figsize_w + 0.8 * (2 * side_pad + left_extra_pad), figsize_h)

    fig, ax = plt.subplots(figsize=figsize)
    xlim = (-side_pad - left_extra_pad, n_defects - 1 + side_pad)

    shade_band_edges(ax, defect_thermodynamics.band_gap, xlim, ylim, orientation="vertical")

    # determine label widths and overlap offsets:
    label_height = max(  # minimum vertical spacing (in eV) between successive labels to avoid overlap
        (label_fontsize / 72.0) * (ylim[1] - ylim[0]) / max(figsize[1], 1.0) * 1.2,
        0.04,
    )  # scales with the height (in points) of the label text
    # horizontal extent of labels in (normalised) data units (column width = ``1``) per character count,
    # used for collision checks in ``_place_inline_labels_for_column`` (which doesn't have access to
    # x-limits, figsize, fontsize etc.), scaled there by each label's actual character count
    per_char_label_width = _estimate_label_width(label_fontsize, 1, xlim[1] - xlim[0], figsize[0])
    faded_alpha = 0.4

    # column headers sit a small distance above ylim[1]; labels for TLs near/inside the CB/VB band-edge
    # zones are allowed to extend a little past ylim[1] / below ylim[0] (symmetrically), so their labels
    # can be placed directly above/below the TL line even when the TL itself sits inside a band-edge zone:
    header_pad_frac = 0.08
    header_y = ylim[1] + header_pad_frac * (ylim[1] - ylim[0])
    label_buf = 0.5 * (header_y - ylim[1])
    label_y_max = ylim[1] + label_buf
    label_y_min = ylim[0] - label_buf

    # Build per-column TL lists, neighbour data and headers; then draw the TL lines, do implement
    # intelligent label placement algorithm. ``columns_data`` holds per-column
    # ``(x_center, half_w, [TL y-positions in range])`` so that for each column we can pass the `other`
    # columns as neighbour data to inform side-clearance picking:
    columns_data: list[tuple[float, float, list[float]]] = []
    column_in_range_tls: list[list[TransitionLevel]] = []
    for i, tls in enumerate(tl_data.values()):
        in_range_tls = [tl for tl in tls if ylim[0] <= tl.TL_eV <= ylim[1]]
        column_in_range_tls.append(in_range_tls)
        columns_data.append((float(i), half_w, [tl.TL_eV for tl in in_range_tls]))

        # draw TL lines (faded grey for metastable-containing TLs when all="faded"):
        x_center = float(i)
        for tl in in_range_tls:
            ax.plot(
                [x_center - half_w, x_center + half_w],
                [tl.TL_eV, tl.TL_eV],
                color="0.45" if tl.faded else "k",
                alpha=faded_alpha if tl.faded else 1.0,
                lw=plt.rcParams["lines.linewidth"] * 1.1,
                solid_capstyle="butt",
                zorder=3,
            )

    if show_charge_labels:  # label placement
        # ``column_positions[k]`` is column ``k``'s TL (& label) placement list (``TransitionLevelLabel``
        # or ``None`` per input TL). Placement is done globally in two stages: (1) place all inline
        # (direct above/below) labels per column which don't overlap with other labels/TL lines,
        # accumulating the leftover "side-bound" TLs; (2) cluster the side-bound TLs by locality and
        # (globally) optimise each cluster's side-label positions:
        column_positions: list[list | None] = [None] * n_defects
        side_bound: list[_SideBoundTL] = []
        for i in range(n_defects):
            column_positions[i], side_bound_i = _place_inline_labels_for_column(
                tls=column_in_range_tls[i],
                x_center=float(i),
                half_w=half_w,
                band_gap=defect_thermodynamics.band_gap,
                ylim=ylim,
                xlim=xlim,
                label_height=label_height,
                per_char_label_width=per_char_label_width,
                neighbor_columns=[c for k, c in enumerate(columns_data) if k != i],
                skip_faded=(all != "faded_labels"),
                label_y_max=label_y_max,
                label_y_min=label_y_min,
            )
            side_bound.extend(sb._replace(col_idx=i) for sb in side_bound_i)

        # cluster side-bound TLs by locality and optimise each cluster independently:
        cluster_y_threshold = 2.5 * _DIAG_LABEL_DY_FRAC * label_height
        for cluster in _cluster_side_bound_tls(side_bound, cluster_y_threshold):
            chosen_TL_labels = _optimise_side_placements(
                side_candidates_per_tl=[sb.candidates for sb in cluster],
                label_height=label_height,
            )
            for sb, TL_label in zip(cluster, chosen_TL_labels, strict=True):
                this_defect_col_positions = column_positions[sb.col_idx]
                assert isinstance(this_defect_col_positions, list)  # typing
                # set the optimised TL label for the corresponding entry in column_positions[sb.col_idx]:
                this_defect_col_positions[sb.idx_in_col] = TL_label
    else:
        column_positions = [None] * n_defects

    # draw labels (and their connectors) for each column:
    for i in range(n_defects):
        defect_column_positions = column_positions[i]
        if defect_column_positions is None:
            continue
        for TL_label, tl_tuple in zip(defect_column_positions, column_in_range_tls[i], strict=True):
            if TL_label is None:  # faded TL with skip_faded=True -- no label drawn
                continue
            ax.text(
                TL_label.x,
                TL_label.y,
                TL_label.label,
                ha=TL_label.ha,
                va=TL_label.va,
                fontsize=label_fontsize,
                color="0.55" if tl_tuple.faded else "0.2",
                alpha=faded_alpha if tl_tuple.faded else 1.0,
                zorder=4,
                clip_on=False,
            )
            if (
                TL_label.conn_y is not None
            ):  # draw a thin connector from TL to label (stopping 20% short, each side)
                conn_x0 = TL_label.conn_x  # column-edge source x, set when the label was placed
                ax.plot(
                    [
                        conn_x0 + 0.2 * (TL_label.x - conn_x0),
                        conn_x0 + 0.8 * (TL_label.x - conn_x0),
                    ],
                    [
                        TL_label.conn_y + 0.2 * (TL_label.y - TL_label.conn_y),
                        TL_label.conn_y + 0.8 * (TL_label.y - TL_label.conn_y),
                    ],
                    color="0.55" if tl_tuple.faded else "0.4",
                    alpha=faded_alpha if tl_tuple.faded else 1.0,
                    lw=plt.rcParams["lines.linewidth"] * 1.1 * 0.5,
                    zorder=2.5,
                )

    # add VBM / CBM labels in the shaded band-edge zones, avoiding overlap with TL labels.
    # If `show_band_labels` is None we try the right side first (preferred), then the left if the right
    # would overlap any placed TL label; if both sides clash we omit the labels.
    if show_band_labels is not False:
        force_right = show_band_labels is True
        all_placed_label_boxes: list[tuple[float, float, float, float]] = (
            [
                _TL_label_box(p, label_height=label_height)
                for positions in column_positions
                if positions
                for p in positions
                if p is not None
            ]
            if show_charge_labels
            else []
        )

        # estimated band-label box size (~3 chars wide, full font height) in data units:
        band_label_w = _estimate_label_width(styled_font_size, 3, xlim[1] - xlim[0], figsize[0])
        band_label_h = max((styled_font_size / 72.0) * (ylim[1] - ylim[0]) / max(figsize[1], 1.0), 0.05)

        def _band_label_overlaps(x: float, ha: str, y: float) -> bool:
            """
            Whether a VBM/CBM band label at ``x`` overlaps any placed label.
            """
            band_box = _label_box(x, y, ha, "center", band_label_w, band_label_h)
            return any(_boxes_overlap(band_box, b) for b in all_placed_label_boxes)

        vbm_y = ylim[0] + 0.5 * (0 - ylim[0])
        cbm_y = defect_thermodynamics.band_gap + 0.5 * (ylim[1] - defect_thermodynamics.band_gap)
        right_x = xlim[1] - 0.05
        left_x = xlim[0] + 0.05
        for text, y in (("VBM", vbm_y), ("CBM", cbm_y)):
            # decide side: right preferred; fall back to left; omit if both overlap (unless forced)
            if not _band_label_overlaps(right_x, "right", y) or force_right:
                x, ha = right_x, "right"
            elif not _band_label_overlaps(left_x, "left", y):
                x, ha = left_x, "left"
            else:
                continue  # both sides clash; omit
            ax.text(
                x,
                y,
                text,
                ha=ha,
                va="center",
                fontsize=styled_font_size,
                color="0.25",
                zorder=2,
            )

    # column headers (defect names) at the top; same order as TL data columns plotted above:
    for i, name in enumerate(format_defect_names(list(tl_data), include_site_info=include_site_info)):
        ax.annotate(
            name,
            xy=(i, header_y),
            ha="center",
            va="center",
            fontsize=styled_font_size * 1.15,
            annotation_clip=False,
            zorder=5,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_ylabel("Fermi Level (eV)")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    if filename is not None:
        fig.savefig(filename, dpi=600, bbox_inches="tight", transparent=True)

    return fig
