"""
Code for plotting defect formation energies and transition levels.
"""

import contextlib
import functools
import os
import re
import warnings
from collections.abc import Callable, Iterable
from itertools import product
from typing import TYPE_CHECKING

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, ticker
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.font_manager import FontProperties
from pymatgen.core.periodic_table import Element
from pymatgen.util.string import latexify
from pymatgen.util.typing import PathLike

from doped.utils.symmetry import sch_symbols  # point group symbols

if TYPE_CHECKING:
    from doped.thermodynamics import DefectThermodynamics

# Recognised vacancy/interstitial substrings in defect names, used by ``format_defect_name`` to infer
# the defect type. Sorted longest-first so the most specific match is found first. Note "V"/"_V" are
# treated as vacancies (not Vanadium) and "I"/"_I" are `not` treated as interstitials (could be iodine).
recognised_pre_vacancy_strings = sorted(
    ["v_", "v", "va_", "Va_", "va", "Va", "V_", "V", "Vac", "vac", "Vac_", "vac_"], key=len, reverse=True
)
recognised_post_vacancy_strings = sorted(
    ["_v", "v", "_vac", "_Vac", "vac", "Vac", "va", "Va", "_va", "_Va"], key=len, reverse=True
)
recognised_pre_interstitial_strings = sorted(
    ["i", "i_", "Int", "int", "Int_", "int_", "Inter", "inter", "Inter_", "inter_"], key=len, reverse=True
)
recognised_post_interstitial_strings = sorted(
    ["_i", "_int", "_Int", "int", "Int", "inter", "Inter", "_inter", "_Inter"], key=len, reverse=True
)


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
        orig = getattr(obj, method_name)

        @functools.wraps(orig)
        def _styled(*args, **kwargs):
            with mpl.rc_context(rc):
                return orig(*args, **kwargs)

        setattr(obj, method_name, _styled)

    _wrap(fig, "draw")
    _wrap(fig.canvas, "print_figure")
    fig._doped_styled_draw = True
    return fig


def _chempot_warning(dft_chempots: dict | None) -> None:
    """
    Issue a warning if DFT chemical potentials are not provided.

    Args:
        dft_chempots (dict | None):
            Dictionary of chemical potentials (to be used for computing
            formation energies for plotting). If ``None``, a warning is raised
            indicating that absolute formation energies will be inaccurate.
    """
    if dft_chempots is None:
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
                alpha = float(default.split("_alpha_")[-1])
                default = default.split("_alpha_")[0]

            warnings.warn(
                f"Colormap '{colormap}' not found in `cmcrameri` "
                f"(https://www.fabiocrameri.ch/colourmaps) or `matplotlib` "
                f"(https://matplotlib.org/stable/users/explain/colors/colormaps) colormaps. "
                f"Defaulting to '{default}' colormap."
            )
            cmap = cmc.cmaps.get(default, colormaps.get(default, cmc.batlow))

        colormap = cmap

    if alpha is not None and hasattr(colormap, "colors"):  # apply alpha to (listed) colormap colours
        colormap.colors = [(*color[:3], alpha) for color in colormap.colors]

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
) -> tuple[list[str | tuple[float, ...]], list[str]]:
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
    if isinstance(cmap, ListedColormap) and len(cmap.colors) < 150:  # cmcrameri returned with 256 colors
        # repeat the listed colours (cycling) until we have one per line:
        base = list(cmap.colors)
        colors = base * (num_lines // len(base)) + base[: num_lines % len(base)]
    else:
        colors = cmap(np.linspace(0, 1, num_lines))

    linestyles = get_linestyles(linestyles, num_lines)

    return colors, linestyles


def _plot_formation_energy_lines(
    xy: dict,
    colors: list[str | tuple[float, ...]],
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
    for cnt, (x_vals, y_vals) in enumerate(xy.values()):
        ax.plot(
            x_vals,
            y_vals,
            color=colors[cnt],
            linestyle=linestyles[cnt],
            markeredgecolor=colors[cnt],
            lw=styled_linewidth * 1.2,
            markersize=styled_markersize * (4 / 6),
            **kwargs,
        )


def _plot_transition_level_markers(
    ax: plt.Axes,
    defect_thermodynamics: "DefectThermodynamics",
    defect_names: Iterable[str],
    colors: list[str | tuple[float, ...]],
    dft_chempots: dict | None,
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
        dft_chempots (dict | None):
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
    for cnt, def_name in enumerate(defect_names):
        x_trans, y_trans, tl_labels, tl_label_type = [], [], [], []
        for x_val, chargeset in tl_map[def_name].items():
            x_trans.append(x_val)
            y_trans.append(
                next(
                    defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=x_val
                    )
                    for defect_entry in defect_thermodynamics.stable_entries[def_name]
                    if defect_entry.charge_state == chargeset[0]  # formation energy of first entry in TL
                )
            )
            tl_labels.append(rf"$\epsilon${_format_TL_charge_label((max(chargeset), min(chargeset)))}")
            tl_label_type.append("start_positive" if max(chargeset) > 0 else "end_negative")

        if not x_trans:
            continue

        color = "k" if all_entries is True else colors[cnt]
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


def _shade_band_edges(
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
    shared_kwargs = {
        "vmin": 0,
        "vmax": 3,
        "interpolation": "bicubic",
        "rasterized": True,
        "aspect": "auto",
        "zorder": 0,
    }
    if orientation == "horizontal":  # gradient along x, fixed (large) vertical extent
        ax.imshow([(0, 1), (0, 1)], cmap=plt.cm.Blues, extent=(min(xlim), 0, -50, 100), **shared_kwargs)
        ax.imshow(
            [(1, 0), (1, 0)], cmap=plt.cm.Oranges, extent=(band_gap, max(xlim), -50, 100), **shared_kwargs
        )
    else:  # vertical: gradient along y, spanning the full x-range
        ax.imshow([(1, 1), (0, 0)], cmap=plt.cm.Blues, extent=(*xlim, min(ylim), 0.0), **shared_kwargs)
        ax.imshow(
            [(0, 0), (1, 1)], cmap=plt.cm.Oranges, extent=(*xlim, band_gap, max(ylim)), **shared_kwargs
        )


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
    chempot_table: bool = True,
    filename: PathLike | None = None,
    styled_font_size: float = 9.0,
) -> None:
    """
    TODO.
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
        ax.figure.savefig(filename, dpi=600, bbox_inches="tight", transparent=True)


def format_defect_name(
    defect_species: str,
    include_site_info: bool = False,
    wout_charge: bool = False,
) -> str | None:
    r"""
    Format defect name using LaTeX styling, intended for plot labelling/titles.

    For example, converts ``"Cd_i_C3v_0"`` to ``"$Cd_{i}^{0}$"`` or
    ``"$Cd_{i_{C3v}}^{0}$"``, if ``include_site_info`` is ``True``). Note this
    assumes "V\_..." means vacancy not Vanadium.

    Args:
        defect_species (str):
            Name of defect including charge state (e.g. ``"Cd_i_C3v_0"``).
        include_site_info (bool):
            Whether to include site info in name (e.g. ``"$Cd_{i}^{0}$"``
            or ``"$Cd_{i_{C3v}}^{0}$"``). Defaults to ``False``.
        wout_charge (bool):
            Whether to exclude the charge state from the formatted
            ``defect_species`` name. Defaults to ``False``.

    Returns:
        str | None:
            Formatted defect name, or ``None`` if it could not be parsed.
    """
    if not isinstance(defect_species, str):  # check inputs
        raise TypeError(f"`defect_species` {defect_species} should be a string")

    if wout_charge:
        defect_species += "_99"  # add dummy charge for parsing; 99 red balloons go by...

    try:
        charge = int(defect_species.split("_")[-1])  # charge comes last
        charge_string = f"{charge:+}" if charge > 0 else f"{charge}"
    except ValueError as e:
        raise ValueError(
            f"Problem reading defect name {defect_species}, should end with charge state after underscore "
            f"(e.g. Te_i_Td_Te2.83_+1)"
        ) from e

    # TODO: Move subfunctions out of format_defect_name, and give short docstrings

    # Format defect name for title/axis labels:
    defect_name = None
    dummy_h = Element("H")

    # LaTeX name builders (``site`` = optional site-info subscript):
    def _vac_name(element: str, site: str | None = None) -> str:
        inner = f"{element}_{{{site}}}" if site else element
        return rf"$\it{{V}}\!$ $_{{{inner}}}^{{{charge_string}}}$"

    def _int_name(element: str, site: str | None = None) -> str:
        sub = f"_{{i_{{{site}}}}}" if site else "_i"  # note: bare "_i" (no braces) when no site info
        return f"{element}${sub}^{{{charge_string}}}$"

    def _sub_name(sub_element: str, orig_element: str, site: str | None = None) -> str:
        inner = f"{orig_element}_{{{site}}}" if site else orig_element
        return f"{sub_element}$_{{{inner}}}^{{{charge_string}}}$"

    def _valid_symbols(strings) -> list[str]:
        """
        Unique valid element symbols (order-preserving) from ``strings``.
        """
        seen: list[str] = []
        for s in strings:
            if dummy_h.is_valid_symbol(s) and s not in seen:
                seen.append(s)
        return seen

    pre_charge_name = defect_species.rsplit("_", 1)[0]  # defect name without charge state
    trimmed_pre_charge_name = pre_charge_name  # later trimmed of any pre/post vacancy/interstitial strings

    doped_site_info = None
    # check if name is doped format, having site info as point group symbol (and more) after 2nd "_":
    with contextlib.suppress(IndexError):
        # from 2nd underscore to last underscore (before charge state) is site info:
        point_group_symbol = defect_species.split("_")[2]
        if point_group_symbol in sch_symbols and all(  # recognised point group symbol?
            i not in pre_charge_name for i in ["int", "Int", "vac", "Vac", "sub", "Sub", "as_"]  # no As_
        ):
            # format point group symbol to (e.g. C1 -> C_1) (already in math mode here)
            doped_site_info = f"{point_group_symbol[0]}_{{{point_group_symbol[1:]}}}"
            if defect_species.split("_")[3:-1]:  # if there is more site info after point group symbol
                doped_site_info += "-" + "-".join(defect_species.split("_")[3:-1])
            trimmed_pre_charge_name = pre_charge_name.replace(
                f"_{'_'.join(defect_species.split('_')[2:-1])}", ""
            )

    def _check_matching_defect_format(
        element: str,
        name: str,
        pre_def_type_list: list[str],
        post_def_type_list: list[str],
    ) -> int:
        """
        Check if the given ``name`` matches expected defect naming formats,
        including the placement of the ``element`` and pre/post defect type
        information, without consideration of any site information (parsed
        separately).

        Args:
            element (str):
                The defect element being checked (e.g., "Ag", "Cd", "H").
            name (str):
                The string to check for a matching format.
            pre_def_type_list (list[str]):
                List of possible defect type strings (e.g., ["v_", "V_"]),
                occurring `before` the ``element`` position in ``name``.
            post_def_type_list (list[str]):
                List of possible defect type strings (e.g., ["_i", "_int"]),
                occurring `after` the ``element`` position in ``name``.

        Returns:
            int:
                Returns the length of the ``name`` string minus the match start
                character position (so matching the start of ``name`` returns
                ``len(name)``, a match after 3 characters returns
                ``len(name) - 3`` etc., in order to score potential matches and
                favour matching near the start of the string). As such, a
                return value of ``0`` indicates no match found.
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
        Checks if the given ``name`` matches expected defect naming formats,
        including the placement of the ``element`` and site information, for
        older defect naming formats.

        Args:
            element (str):
                The defect element being checked (e.g., "Ag", "Cd", "H").
            name (str):
                The string to check for matching format.
            pre_def_type_list (list[str]):
                List of possible defect type strings (e.g., ["v_", "V_"]),
                occurring `before` the ``element`` position in ``name``.
            post_def_type_list (list[str]):
                List of possible defect type strings (e.g., ["_i", "_int"]),
                occurring `after` the ``element`` position in ``name``.

        Returns:
            tuple[bool, str | None]:
                A tuple where the first element is a boolean indicating if
                the format matches, and the second element is the site
                information (if applicable) or ``None``.
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

    def _try_vacancy_interstitial_match(element, name, include_site_info):
        defect_name_without_site_info = defect_name_with_site_info = None

        match_found, site_info = _check_matching_defect_format_with_old_site_info(
            element, name, recognised_pre_vacancy_strings, recognised_post_vacancy_strings
        )
        if match_found:
            defect_name_with_site_info = _vac_name(element, site_info)
            defect_name_without_site_info = _vac_name(element)

        else:
            match_found, site_info = _check_matching_defect_format_with_old_site_info(
                element, name, recognised_pre_interstitial_strings, recognised_post_interstitial_strings
            )
            if match_found:
                defect_name_with_site_info = _int_name(element, site_info)
                defect_name_without_site_info = _int_name(element)

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
            return name_func(element, doped_site_info)

        return name_func(element)

    def _try_substitution_match(substituting_element, orig_site_element, name, include_site_info):
        defect_name = None
        if (
            f"{substituting_element}_{orig_site_element}" in name
            or f"{substituting_element}_on_{orig_site_element}" in name
        ):
            defect_name = _sub_name(
                substituting_element, orig_site_element, doped_site_info if include_site_info else None
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
                    defect_name = _sub_name(substituting_element, orig_site_element, site_info)
                    break

        return defect_name.replace("mult", "m") if defect_name is not None else None

    def _defect_name_from_matching_elements(
        element_matches: list[str],
        name: str = pre_charge_name,
        include_site_info: bool = include_site_info,
    ) -> str | None:
        if len(element_matches) == 1:  # vacancy or interstitial?
            defect_name = _try_vacancy_interstitial_match(element_matches[0], name, include_site_info)
        elif len(element_matches) == 2:
            # try substitution/antisite match, if not try vacancy/interstitial with first element
            defect_name = _try_substitution_match(
                element_matches[0], element_matches[1], name, include_site_info
            )
            if defect_name is None:
                defect_name = _try_vacancy_interstitial_match(element_matches[0], name, include_site_info)
        else:
            # try use first match and see if we match vacancy or interstitial format; if not, try first and
            # second matches and see if we match substitution format; otherwise fail
            defect_name = _try_vacancy_interstitial_match(element_matches[0], name, include_site_info)
            if defect_name is None:
                defect_name = _try_substitution_match(
                    element_matches[0],
                    element_matches[1],
                    name,
                    include_site_info,
                )

        return defect_name

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

    if possible_two_character_elements:
        defect_name = _defect_name_from_matching_elements(possible_two_character_elements)

        if defect_name is None and len(possible_two_character_elements) == 1:
            # possibly one single-character element and one two-character element
            possible_one_character_elements = _valid_symbols(
                trimmed_pre_charge_name.replace(possible_two_character_elements[0], "")
            )

            if possible_one_character_elements:
                # in this case, we don't know the order of the 1-character vs 2-character elements in the
                # name, so we try both orderings:
                defect_name = _defect_name_from_matching_elements(
                    possible_two_character_elements + possible_one_character_elements,
                )
                if defect_name is None:
                    defect_name = _defect_name_from_matching_elements(
                        possible_one_character_elements + possible_two_character_elements,
                    )

    if defect_name is None and (
        possible_one_character_elements := _valid_symbols(trimmed_pre_charge_name)
    ):
        defect_name = _defect_name_from_matching_elements(possible_one_character_elements)

    if defect_name is None:  # try matching to PyCDT/old-doped style:
        try:
            defect_type = defect_species.split("_")[0]  # vac, as or int
            if (
                defect_type.capitalize() == "Int"
            ):  # for interstitials, name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
                site_element = defect_species.split("_")[1]  # element then site for interstitials
                site: str | None = defect_species.split("_")[2]
                defect_name = _int_name(site_element, site if include_site_info else None)
            else:
                site = defect_species.split("_")[1]  # number indicating defect site (from doped)
                site_element = defect_species.split("_")[2]  # element at defect site

            site = site if include_site_info else None  # whether to include the site number
            if defect_type.lower() == "vac":
                defect_name = _vac_name(site_element, site)
            elif defect_type.lower() in ["as", "sub"]:
                subs_element = defect_species.split("_")[4]
                defect_name = _sub_name(site_element, subs_element, site)
            elif defect_type.capitalize() != "Int":
                raise ValueError(f"Defect type {defect_type} not recognized. Please check spelling.")
        except Exception:
            return None

    return f"{defect_name.rsplit('^', 1)[0]}$" if (defect_name and wout_charge) else defect_name


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


def _get_legend_txt(
    for_legend: list[str], all_entries: bool = False, include_site_info: bool = False
) -> list[str]:
    # get latex-like legend titles; don't include site info by default, unless duplicates
    def _get_defect_name(defect_entry_name, site_info):
        try:
            return format_defect_name(
                defect_species=defect_entry_name,
                include_site_info=site_info,
                wout_charge=not all_entries,  # defect names without charge
            )

        except Exception:  # if formatting fails, just use the defect_species name
            return defect_entry_name

    legend_txt = [
        _get_defect_name(defect_entry_name, include_site_info) for defect_entry_name in for_legend
    ]

    if len(legend_txt) == len(set(legend_txt)):  # no duplicates, good to go
        return legend_txt

    # duplicates in defect names; rename to avoid overwriting:
    if not include_site_info:  # first see if using site info with duplicates removes duplicate names
        site_info_entry_names = [
            _get_defect_name(defect_entry_name, True) for defect_entry_name in for_legend
        ]
        legend_txt = [
            (
                site_info_name
                if site_info_entry_names.count(site_info_name) < legend_txt.count(non_site_info_name)
                else non_site_info_name
            )
            for site_info_name, non_site_info_name in zip(site_info_entry_names, legend_txt, strict=False)
        ]

    if len(legend_txt) == len(set(legend_txt)):
        return legend_txt

    # duplicates in entry names and site info doesn't (fully) solve it, so append "a,b,c.." for different
    # defect species with the same name:
    final_legend_txt: list[str] = []
    for defect_name in legend_txt:
        final_legend_txt.append(
            _uniquified_name(
                defect_name,
                variant_exists=lambda base: any(base in i for i in final_legend_txt),
                exact_exists=lambda n: n in final_legend_txt,
                suffix=lambda base, i: f"{base}$_{{-{chr(96 + i)}}}$",
                rename_prev=lambda old, new: final_legend_txt.__setitem__(
                    final_legend_txt.index(old), new
                ),
            )
        )

    return final_legend_txt


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
        for single_output_dict in output_dicts:
            single_output_dict[new] = single_output_dict.pop(old)

    key = _uniquified_name(
        key,
        variant_exists=lambda base: base in output_dict
        or any(f"{base}_{chr(96 + i)}" in output_dict for i in range(1, 27)),
        exact_exists=lambda n: n in output_dict,
        suffix=lambda base, i: f"{base}_{chr(96 + i)}",
        rename_prev=_rename_prev,
    )
    return key, output_dicts


def _get_formation_energy_lines(
    defect_thermodynamics: "DefectThermodynamics",
    dft_chempots: dict | None,
    xlim: tuple[float, float],
    defect_subset: list[str] | str | None = None,
):
    if isinstance(defect_subset, str):
        defect_subset = [defect_subset]

    def _name_in_subset(name):
        return not defect_subset or any(s in name for s in defect_subset)

    def _form_en(defect_entry, fermi_level):  # formation energy at a given Fermi level
        return defect_thermodynamics.get_formation_energy(
            defect_entry, chempots=dft_chempots, fermi_level=fermi_level
        )

    def _entry_with_charge(entries, charge):  # first entry in ``entries`` with this charge state
        return next(e for e in entries if e.charge_state == charge)

    xy: dict = {}  # {defect_name: [[x_vals], [y_vals]]} for stable (ground-state) lines
    all_lines_xy: dict = {}  # as above, but for all entries (every charge state)
    y_range_vals: list[float] = []  # y-values at the x-limits, used to set the y-axis range
    all_entries_y_range_vals: list[float] = []  # y-values at the x-limits, used to set the y-axis range
    lower_cap, upper_cap = -100, 100  # arbitrary values to extend lines to
    ymin = 0

    for defect_name_wout_charge, defect_entry_list in defect_thermodynamics.all_entries.items():
        if not _name_in_subset(defect_name_wout_charge):
            continue
        for defect_entry in defect_entry_list:
            # all_lines name includes charge state; rename in case of duplicate entry names:
            defect_name_w_charge, [all_lines_xy] = _rename_key_and_dicts(defect_entry.name, [all_lines_xy])
            all_lines_xy[defect_name_w_charge] = [
                [lower_cap, upper_cap],
                [_form_en(defect_entry, lower_cap), _form_en(defect_entry, upper_cap)],
            ]
            all_entries_y_range_vals.extend(_form_en(defect_entry, x_window) for x_window in xlim)

    for def_name, def_tl in defect_thermodynamics.transition_level_map.items():
        if not _name_in_subset(def_name):
            continue
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
        yvals = _get_in_gap_yvals(xy[def_name][0], xy[def_name][1], (0, defect_thermodynamics.band_gap))
        if all(y < 0 for y in yvals):  # Check if all y-values are below zero
            warnings.warn(
                f"All formation energies for {def_name} are below zero across the "
                f"entire band gap range. This is typically unphysical (see docs), and likely due to "
                f"mis-specification of chemical potentials (see docstrings and/or tutorials)."
            )
            ymin = min(ymin, *yvals)

    if not y_range_vals:
        raise ValueError("No formation energy data available to plot.")

    return (xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin


def _get_ylim_from_y_range_vals(
    y_range_vals: list[float], ymin: float = 0, auto_labels: bool = False
) -> tuple[float, float]:
    window = max(y_range_vals) - min(*y_range_vals, ymin)
    spacer = 0.1 * window
    ylim = (ymin, max(y_range_vals) + spacer)
    if auto_labels:  # need to manually set xlim or ylim if labels cross axes!!
        # Increase y_limit to give space for transition level labels
        ylim = (ymin, max(y_range_vals) * 1.17) if spacer / ylim[1] < 0.145 else ylim

    return ylim


def _get_in_gap_yvals(
    x_coords: list[float], y_coords: list[float], x_range: tuple[float, float]
) -> np.ndarray:
    relevant_x = np.linspace(x_range[0], x_range[1], 100)  # x values in range
    return np.interp(relevant_x, x_coords, y_coords)  # y values in range


def formation_energy_plot(
    defect_thermodynamics: "DefectThermodynamics",
    dft_chempots: dict | None = None,
    el_refs: dict | None = None,
    all_entries: bool | str = False,
    include_site_info: bool = False,
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
        dft_chempots (dict):
            Dictionary of ``{Element: value}`` giving the chemical potential of
            each element.
        el_refs (dict):
            Dictionary of ``{Element: value}`` giving the reference energy of
            each element.
        all_entries (bool, str):
            Whether to plot the formation energy lines of `all` defect entries,
            rather than the default of showing only the equilibrium states at
            each Fermi level position (traditional). If instead set to "faded",
            will plot the equilibrium states in bold, and all unstable states
            in faded grey. (Default: False)
        include_site_info (bool):
            Whether to include site info in defect names in the plot legend
            (e.g. ``$Cd_{i_{C3v}}^{0}$`` rather than ``$Cd_{i}^{0}$``). Default
            is ``False``, where site info is not included unless we have
            inequivalent sites for the same defect type. If, even with site
            info added, there are duplicate defect names, then "-a", "-b", "-c"
            etc. are appended to the names to differentiate.
        chempot_table (bool):
            Whether to print the chemical potential table above the plot.
            (Default: True)
        defect_subset (list[str], str):
            If provided, only defects whose name contains at least one of the
            given substrings are plotted (e.g. ``["v_", "Te_Cd"]`` would keep
            all vacancies plus ``Te_Cd``). A bare string is treated as a
            single-element list. (Default: ``None`` -- all defects)
            # TODO: Test
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
    _chempot_warning(dft_chempots)
    if defect_thermodynamics.band_gap is None:
        raise ValueError(
            "`band_gap` is not set for `DefectThermodynamics`, cannot plot formation energies."
        )

    if xlim is None:
        xlim = (-0.3, defect_thermodynamics.band_gap + 0.3)

    (xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin = _get_formation_energy_lines(
        defect_thermodynamics, dft_chempots, xlim, defect_subset=defect_subset
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
        dft_chempots,
        styled_markersize=styled_markersize,
        styled_font_size=styled_font_size,
        all_entries=all_entries,
        auto_labels=auto_labels,
    )

    legend_txt = _get_legend_txt(
        list((plotting_xy).keys()),
        all_entries=all_entries is True,
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

    _shade_band_edges(ax, defect_thermodynamics.band_gap, xlim, ylim, orientation="horizontal")
    _set_TLD_axis_labels_limits_ticks(ax, xlim, ylim)

    # dashed line for E_formation = 0 (in case ymin < 0) and Fermi level if provided:
    ax.plot([xlim[0], xlim[1]], [0, 0], c="k", ls="--", alpha=0.7)
    if fermi_level is not None:
        ax.axvline(x=fermi_level, linestyle="-.", color="k")

    if chempot_table and dft_chempots:
        plot_chemical_potential_table(ax, dft_chempots, el_refs=el_refs)

    _set_title_and_save_figure(ax, title, chempot_table, filename, styled_font_size)

    return fig


def plot_chemical_potential_table(
    ax: plt.Axes,
    dft_chempots: dict[str, float],
    cellLoc: str = "left",
    el_refs: dict[str, float] | None = None,
) -> plt.table:
    """
    Plot a table of chemical potentials above the plot in ``ax``.

    Args:
        ax (plt.Axes):
            Axes object to plot the table in.
        dft_chempots (dict):
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
        dft_chempots = {el: energy - el_refs[el] for el, energy in dft_chempots.items()}
    labels = [rf"$\mathregular{{\mu_{{{s}}}}}$," for s in sorted(dft_chempots.keys())]
    labels[0] = f"({labels[0]}"
    labels[-1] = f"{labels[-1][:-1]})"  # [:-1] removes trailing comma
    labels = ["Chemical Potentials", *labels, " Units:"]

    text_list = [f"{dft_chempots[el]:.2f}," for el in sorted(dft_chempots.keys())]

    # add brackets to first and last entries:
    text_list[0] = f"({text_list[0]}"
    text_list[-1] = f"{text_list[-1][:-1]})"  # [:-1] removes trailing comma
    if el_refs is not None:
        text_list = ["(wrt Elemental refs)", *text_list, "  [eV]"]
    else:
        text_list = ["(from calculations)", *text_list, "  [eV]"]
    widths = [0.1] + [0.9 / len(dft_chempots)] * (len(dft_chempots) + 2)
    tab = ax.table(cellText=[text_list], colLabels=labels, colWidths=widths, loc="top", cellLoc=cellLoc)
    tab.auto_set_column_width(list(range(len(widths))))

    for cell in tab.get_celld().values():
        cell.set_linewidth(0)
        cell.set_facecolor("none")  # make transparent as with rest of plot

    return tab


# TODO: Reduce redundancy; use get_TLs functions from thermodynamics, same for TL naming etc?
# TODO: General code condensing, simplification, readability, review with Codex
# TODO: General code cleanup for this module if possible; typing etc. Review with Claude and Codex


def _get_transition_level_data(
    defect_thermodynamics: "DefectThermodynamics",
    all_TLs: bool | str = False,
):
    """
    Collect transition level data for ``transition_level_diagram``.

    Returns a dict
    ``{defect_name: [(TL_eV, charges, i_meta, j_meta, faded), ...]}`` sorted
    by TL energy, where ``charges = (q_upper, q_lower)`` (more positive then
    more negative charge state), ``i_meta``/``j_meta`` indicate whether the
    corresponding charge state is metastable, and ``faded`` is ``True`` if
    the TL should be drawn faded (only used when ``all_TLs == "faded"``).

    Args:
        defect_thermodynamics (|DefectThermodynamics|):
            Source of TL data.
        all_TLs (bool, str):

            - ``False``: only thermodynamic ground-state TLs (from
              ``transition_level_map``). ``faded`` is always ``False``.
            - ``True``: all single-electron TLs. ``faded`` is always ``False``.
            - ``"faded"`` / ``"faded_labels"``: ground-state TLs (solid) plus
              single-electron TLs that involve at least one metastable charge
              state (these latter are marked ``faded=True``). The two values
              return the same data; the renderer chooses whether to draw
              labels for the faded TLs.
    """
    # ground-state TLs (i.e. those visible on the formation energy diagram):
    gs_per_defect: dict[str, list[tuple]] = {}
    for defect_name, tl_dict in defect_thermodynamics.transition_level_map.items():
        gs_per_defect[defect_name] = [
            (float(TL), (max(chargeset), min(chargeset)), False, False, False)
            for TL, chargeset in tl_dict.items()
        ]

    if all_TLs is False:
        for tls in gs_per_defect.values():
            tls.sort(key=lambda x: x[0])
        return gs_per_defect

    # all single-electron TLs (consecutive charge pairs with diff=1):
    stable_entries_list = defect_thermodynamics.all_stable_entries
    se_per_defect: dict[str, list[tuple]] = {}
    for defect_name, grouped_defect_entries in defect_thermodynamics.all_entries.items():
        se_per_defect[defect_name] = []
        sorted_entries = sorted(grouped_defect_entries, key=lambda x: x.charge_state)
        for i, j in product(sorted_entries, repeat=2):
            if i.charge_state - j.charge_state == 1:
                mean_VBM = float(
                    np.mean([x.calculation_metadata.get("vbm", defect_thermodynamics.vbm) for x in [i, j]])
                )
                TL = j.get_ediff() - i.get_ediff() - mean_VBM
                i_meta = not any(i == y for y in stable_entries_list)
                j_meta = not any(j == y for y in stable_entries_list)
                se_per_defect[defect_name].append(
                    (float(TL), (i.charge_state, j.charge_state), i_meta, j_meta)
                )

    if all_TLs is True:
        out = {}
        # iterate in the order defined by transition_level_map (which respects defect
        # appearance order); add any defects only in all_entries afterwards.
        ordered_names = list(defect_thermodynamics.transition_level_map.keys())
        for name in se_per_defect:
            if name not in ordered_names:
                ordered_names.append(name)
        for name in ordered_names:
            tls = [(*tl, False) for tl in se_per_defect.get(name, [])]
            tls.sort(key=lambda x: x[0])
            out[name] = tls
        return out

    # all_TLs in {"faded", "faded_labels"}: GS TLs solid + metastable single-electron faded
    out = {}
    ordered_names = list(defect_thermodynamics.transition_level_map.keys())
    for name in se_per_defect:
        if name not in ordered_names:
            ordered_names.append(name)
    for name in ordered_names:
        merged = list(gs_per_defect.get(name, []))  # GS, not faded
        for tl_eV, charges, i_meta, j_meta in se_per_defect.get(name, []):
            if i_meta or j_meta:
                merged.append((tl_eV, charges, i_meta, j_meta, True))
        merged.sort(key=lambda x: x[0])
        out[name] = merged
    return out


def _format_TL_charge_label(charges, i_meta=False, j_meta=False):
    """
    Format a charge transition label like ``"(+1/0)"`` or ``"(+1*/0)"``.

    ``charges = (q_upper, q_lower)`` where ``q_upper`` is the more positive
    charge state. ``i_meta``/``j_meta`` denote whether the upper/lower charge
    state is metastable, in which case a ``*`` is appended to that charge.
    """
    q_i, q_j = charges
    return (
        f"({'+' if q_i > 0 else ''}{q_i}{'*' if i_meta else ''}"
        f"/{'+' if q_j > 0 else ''}{q_j}{'*' if j_meta else ''})"
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


def _label_y_extent(y: float, va: str, label_height: float) -> tuple[float, float]:
    """
    Vertical extent (``y_min``, ``y_max``) of a label centred or anchored at
    ``y`` with vertical alignment ``va`` and height ``label_height``.
    """
    if va == "bottom":
        return y, y + label_height
    if va == "top":
        return y - label_height, y
    # "center"
    return y - 0.5 * label_height, y + 0.5 * label_height


def _label_x_extent(x: float, ha: str, label_width: float) -> tuple[float, float]:
    """
    Horizontal extent (``x_min``, ``x_max``) of a label anchored at ``x`` with
    horizontal alignment ``ha`` and width ``label_width``.
    """
    if ha == "left":
        return x, x + label_width  # text extends to the right of the anchor
    if ha == "right":
        return x - label_width, x  # text extends to the left of the anchor
    # "center"
    return x - 0.5 * label_width, x + 0.5 * label_width


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
    Liang-Barsky test: does the line segment ``(x0,y0)-(x1,y1)`` intersect
    the axis-aligned rectangle ``[rx_min, rx_max] x [ry_min, ry_max]``?
    """
    dx, dy = x1 - x0, y1 - y0
    t_min, t_max = 0.0, 1.0
    for p, q in ((-dx, x0 - rx_min), (dx, rx_max - x0), (-dy, y0 - ry_min), (dy, ry_max - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False  # parallel and outside
            continue
        t = q / p
        if p < 0:
            if t > t_min:
                t_min = t
        elif t < t_max:
            t_max = t
        if t_min > t_max:
            return False
    return True


def _optimise_side_placements(
    side_candidates_per_tl: list[list[tuple]],
    placed_inline: list[tuple],
    line_y_positions: list[float],
    line_left: float,
    line_right: float,
    x_center: float,
    label_h: float,
    label_w: float,
    max_brute_force_combos: int = 200_000,
) -> list[tuple]:
    r"""
    Pick one position per side-bound TL so as to minimise total overlap cost.

    For each TL we have several candidate ``(x, y, ha, va, conn_y)`` positions.
    The cost of an assignment is the sum of pairwise overlap penalties between
    label boxes, label-vs-TL-line overlaps, and connector intersections (with
    other labels and with TL lines). The combination minimising the total cost
    is returned.

    Brute-force enumeration is used when the search space (product of options
    per TL) is small. For larger spaces we fall back to a greedy first-pick
    plus a few hill-climbing refinement passes.
    """
    n = len(side_candidates_per_tl)
    if n == 0:
        return []

    def lbl_box(pos: tuple, w: float = label_w) -> tuple[float, float, float, float]:
        x_pos, y_pos, ha, va, _conn = pos
        x_min, x_max = _label_x_extent(x_pos, ha, w)
        y_min, y_max = _label_y_extent(y_pos, va, label_h)
        return x_min, x_max, y_min, y_max

    # add a small y-buffer (~30% of a label height) so two side labels packed almost
    # touch-to-touch on the same side are treated as overlapping (the same buffer applied
    # to inline label-vs-label checks):
    y_buf = 0.3 * label_h

    def boxes_overlap(b1, b2) -> bool:
        return b1[1] > b2[0] and b1[0] < b2[1] and b1[3] + y_buf > b2[2] and b1[2] - y_buf < b2[3]

    def conn_endpoints(pos: tuple) -> tuple[float, float, float, float] | None:
        x_pos, y_pos, _ha, _va, conn_y = pos
        if conn_y is None:
            return None
        conn_x0 = x_center if x_pos == x_center else (line_right if x_pos > x_center else line_left)
        return conn_x0, conn_y, x_pos, y_pos

    line_eps = 0.05 * label_h
    inline_boxes = [lbl_box(p[:5], p[4]) for p in placed_inline]  # placed_inline label bounds
    inline_connectors = [
        (p[5], p[6], p[0], p[1]) for p in placed_inline if p[5] is not None and p[6] is not None
    ]

    # Precompute the box and connector geometry for every candidate, so the search below works
    # purely with cached coordinates and integer cost tables, rather than recomputing label /
    # connector geometry per combination (the previous bottleneck, dominated by millions of
    # ``_segment_intersects_rect`` calls). The cost decomposes into a per-candidate "unary" term
    # (overlaps with inline labels / TL lines, independent of the other side picks) and a pairwise
    # term between two side picks; both are tabulated once up front here.
    counts = [len(c) for c in side_candidates_per_tl]
    cand_boxes = [[lbl_box(c) for c in cands] for cands in side_candidates_per_tl]
    cand_conns = [[conn_endpoints(c) for c in cands] for cands in side_candidates_per_tl]

    def _unary_cost(i: int, a: int) -> int:
        cost = 0
        box = cand_boxes[i][a]
        for ibox in inline_boxes:  # overlap with inline labels
            if boxes_overlap(box, ibox):
                cost += 10
        if box[1] > line_left and box[0] < line_right:  # overlap with TL lines in the column
            for ly in line_y_positions:
                if box[2] - line_eps <= ly <= box[3] + line_eps:
                    cost += 5
                    break
        endpoints = cand_conns[i][a]
        if endpoints is not None:
            cx0, cy0, cx1, cy1 = endpoints
            for ibox in inline_boxes:  # connector through inline label boxes
                if _segment_intersects_rect(cx0, cy0, cx1, cy1, *ibox):
                    cost += 4
            for ly in line_y_positions:  # connector through TL lines (excluding source)
                if abs(ly - cy0) < line_eps:
                    continue
                if _segment_intersects_rect(
                    cx0, cy0, cx1, cy1, line_left, line_right, ly - line_eps, ly + line_eps
                ):
                    cost += 3
            for cox0, coy0, cox1, coy1 in inline_connectors:  # connector through inline connectors
                if _segment_intersects_rect(
                    cx0,
                    cy0,
                    cx1,
                    cy1,
                    min(cox0, cox1) - 0.001,
                    max(cox0, cox1) + 0.001,
                    min(coy0, coy1) - 0.001,
                    max(coy0, coy1) + 0.001,
                ):
                    cost += 2
        return cost

    unary = [[_unary_cost(i, a) for a in range(counts[i])] for i in range(n)]

    def _pair_cost(i: int, a: int, j: int, b: int) -> int:
        """
        Pairwise cost between side pick ``a`` of TL ``i`` and pick ``b`` of TL
        ``j`` (with ``i < j``): box-box overlap plus the higher-index pick's
        connector crossing the lower-index pick's box (matching the original
        ``total_cost`` accumulation order, where ``position_cost`` only checks
        the later label's connector against earlier labels' boxes).
        """
        box_i = cand_boxes[i][a]
        cost = 10 if boxes_overlap(box_i, cand_boxes[j][b]) else 0
        conn_j = cand_conns[j][b]
        if conn_j is not None and _segment_intersects_rect(*conn_j, *box_i):
            cost += 4
        return cost

    # tabulate pairwise costs for all i < j as ``pairwise[(i, j)][a][b]``:
    pair_items = [
        ((i, j), [[_pair_cost(i, a, j, b) for b in range(counts[j])] for a in range(counts[i])])
        for i in range(n)
        for j in range(i + 1, n)
    ]

    # search space size:
    space = 1
    for c in counts:
        space *= c
        if space > max_brute_force_combos:
            break

    if space <= max_brute_force_combos:
        # brute force: enumerate all index combinations, summing the precomputed cost tables.
        best_cost = float("inf")
        best_indices: tuple[int, ...] = tuple([0] * n)
        for indices in product(*(range(c) for c in counts)):
            cost = sum(unary[i][indices[i]] for i in range(n))
            for (i, j), mat in pair_items:
                cost += mat[indices[i]][indices[j]]
            if cost < best_cost:
                best_cost = cost
                best_indices = indices
                if cost == 0:
                    break
        return [side_candidates_per_tl[k][best_indices[k]] for k in range(n)]

    # greedy + hill-climb fallback for large spaces, using the precomputed geometry. The cost of
    # giving TL ``i`` pick ``a`` given the other chosen picks mirrors the original
    # ``position_cost``: its unary cost plus, for every other chosen TL ``j``, their box-box
    # overlap and this label's connector crossing the other's box.
    def _conditional_cost(i: int, a: int, picks: list[int]) -> int:
        cost = unary[i][a]
        box_i = cand_boxes[i][a]
        conn_i = cand_conns[i][a]
        for j, b in enumerate(picks):
            if b < 0 or j == i:
                continue
            if boxes_overlap(box_i, cand_boxes[j][b]):
                cost += 10
            if conn_i is not None and _segment_intersects_rect(*conn_i, *cand_boxes[j][b]):
                cost += 4
        return cost

    picks: list[int] = [-1] * n  # -1 == not yet chosen
    for i in range(n):  # greedy first-pick: lowest-cost candidate given prior picks
        best_idx, best_cost = 0, float("inf")
        for a in range(counts[i]):
            c = _conditional_cost(i, a, picks)
            if c < best_cost:
                best_cost, best_idx = c, a
        picks[i] = best_idx

    for _ in range(5):  # hill-climbing refinement: swap to a lower-cost alternative until stable
        improved = False
        for i in range(n):
            best_alt, best_alt_cost = None, _conditional_cost(i, picks[i], picks)
            for a in range(counts[i]):
                if a == picks[i]:
                    continue
                c = _conditional_cost(i, a, picks)
                if c < best_alt_cost:
                    best_alt_cost, best_alt = c, a
            if best_alt is not None:
                picks[i] = best_alt
                improved = True
        if not improved:
            break

    return [side_candidates_per_tl[k][picks[k]] for k in range(n)]


def _place_labels_for_column(
    tls: list[tuple],
    x_center: float,
    half_w: float,
    band_gap: float,
    ylim: tuple[float, float],
    xlim: tuple[float, float],
    label_offset_eV: float,
    label_width_eV: float,
    skip_faded: bool = True,
    header_y_min: float | None = None,
    label_y_min: float | None = None,
    neighbor_columns: list[tuple[float, float, list[float]]] | None = None,
    cross_column_placed: list[tuple] | None = None,
):
    r"""
    Decide where to place the label for each TL in a single defect column,
    independently of any clustering.

    For each TL (in input order), the label is tried first directly above the
    TL line, then directly below, then directly to the left (with a small
    horizontal connector to the column edge), then directly to the right,
    then diagonally up/down to the right and left, then stacked further above
    or below. The first candidate position that doesn't overlap with another
    TL line in the same column, an already-placed label, the connector route
    of an already-placed label, or one of the band edges (VBM at 0 eV, CBM
    at ``band_gap``) is used.

    Args:
        tls: list of TL 5-tuples ``(TL_eV, charges, i_meta, j_meta, faded)``.
        x_center: x-coordinate of the centre of the defect column.
        half_w: half-width of the TL lines in the column (in x-units).
        band_gap: band gap in eV (CBM position; VBM is at 0 eV), used as a
            collision boundary for label placement.
        ylim: ``(y_min, y_max)`` axis limits in eV.
        xlim: ``(x_min, x_max)`` axis limits in x-units.
        label_offset_eV: vertical offset/height of a label in y-units (eV),
            used for stacking and collision checks.
        label_width_eV: width of a label in x-units, used for collision checks.
        skip_faded: if ``True``, faded TLs (5th element ``True``) do not get
            labels (their lines are still part of collision checks). Their
            entry in the returned list is ``None``.
        header_y_min: if provided, labels are not allowed to extend above
            this y, so they cannot collide with the column header above
            ``ylim[1]``.
        label_y_min: if provided, labels are not allowed to extend below this
            y.
        neighbor_columns: list of ``(x_center, half_w, line_y_positions)``
            tuples for neighbouring columns, used to avoid placing labels over
            their TL lines.
        cross_column_placed: list of already-placed label boxes from other
            columns, used to avoid cross-column label overlaps.

    Returns a list of ``(x, y, label, ha, va, connector_from_y)`` tuples (one
    per TL, in input order) or ``None`` for TLs whose label was skipped.
    ``connector_from_y`` is ``None`` for inline labels with no connector, or
    the line y otherwise (so the caller can draw a connector from the TL
    line to the label).
    """
    line_y_positions = [tl[0] for tl in tls]
    line_left = x_center - half_w
    line_right = x_center + half_w
    side_x_right = line_right + 0.06
    side_x_left = line_left - 0.06
    # vertical "height" of a label in y-units (≈ the text height we computed for stacking):
    label_h = label_offset_eV
    # horizontal half-width of a label in x-units (estimated; used only for collision checks):
    label_hw = 0.5 * label_width_eV
    # labels may extend up to `header_y_min` (slightly past ylim[1], into the buffer below
    # the column header) and down to `label_y_min` (slightly past ylim[0]); this lets TLs
    # that sit in/near the CBM (orange) or VBM (blue) zones have their labels placed
    # directly above/below their line:
    y_max_allowed = header_y_min if header_y_min is not None else ylim[1]
    y_min_allowed = label_y_min if label_y_min is not None else ylim[0]

    # `placed` accumulates label boxes for collision checks. We seed it with any labels
    # already placed in earlier columns (cross_column_placed) so this column's labels avoid
    # cross-column overlaps in addition to in-column ones. New labels placed by this call are
    # appended onto this list. The caller can use the returned `column_placed` list to learn
    # which labels were added (to thread into the next column's call).
    placed: list[tuple[float, float, str, str, float, float | None, float | None]] = list(
        cross_column_placed or []
    )
    n_seeded = len(placed)  # boundary between cross-column placements and this column's
    # each placed entry: (x_pos, y_pos, ha, va, label_width, conn_from_x, conn_from_y)
    results: list[tuple[float, float, str, str, str, float | None] | None] = []
    label_w = 2 * label_hw

    def collides_with_band(y: float, va: str, x_pos: float = x_center, ha: str = "center") -> bool:
        y_min, y_max = _label_y_extent(y, va, label_h)
        # reject labels that straddle a band edge (CBM at band_gap, VBM at 0 eV); labels
        # entirely above CBM (in the orange zone) or entirely below VBM are OK -- they're
        # needed for TLs that happen to lie above CBM or below VBM.
        if y_min < band_gap < y_max or y_min < 0.0 < y_max:
            return True
        # always reject labels past the plot top/bottom (or into the header strip):
        if y_max > y_max_allowed or y_min < y_min_allowed:
            return True
        # also reject labels that would extend past the figure x-limits (y-axis or right edge):
        lbl_left, lbl_right = _label_x_extent(x_pos, ha, label_w)
        return lbl_left < xlim[0] or lbl_right > xlim[1]

    def collides_with_tl_line(
        y: float, va: str, x_pos: float, ha: str, source_y: float | None = None
    ) -> bool:
        y_min, y_max = _label_y_extent(y, va, label_h)
        lbl_left, lbl_right = _label_x_extent(x_pos, ha, label_w)
        # only check column lines (extending from line_left to line_right):
        if not (lbl_right > line_left and lbl_left < line_right):
            return False
        # require ~half a label-height of clearance from the next TL line so a label placed
        # direct-above/below is visually unambiguous (it doesn't read like it could belong
        # to a closely-spaced neighbouring TL). `source_y` is the TL we're labelling, so it
        # is excluded from this check (it sits 0.4*label_h from the anchor by construction):
        for ly in line_y_positions:
            if source_y is not None and abs(ly - source_y) < 0.05 * label_h:
                continue
            if y_min - 0.5 * label_h <= ly <= y_max + 0.5 * label_h:
                return True
        return False

    def collides_with_placed(y: float, va: str, x_pos: float, ha: str) -> bool:
        y_min, y_max = _label_y_extent(y, va, label_h)
        lbl_left, lbl_right = _label_x_extent(x_pos, ha, label_w)
        # require a small vertical buffer (~30% of a label height) between two labels even
        # when their bounding boxes wouldn't strictly intersect, so closely-stacked labels
        # (e.g. label-below-TL-A + label-above-TL-B with A just above B) don't read as one:
        y_buf = 0.3 * label_h
        for px, py, pha, pva, p_label_w, _pcx, _pcy in placed:
            py_min, py_max = _label_y_extent(py, pva, label_h)
            px_left, px_right = _label_x_extent(px, pha, p_label_w)
            if (
                y_max + y_buf > py_min
                and y_min - y_buf < py_max
                and lbl_right > px_left
                and lbl_left < px_right
            ):
                return True
        return False

    def connector_through_placed(conn_x0: float, conn_y0: float, conn_x1: float, conn_y1: float) -> bool:
        """
        Does the proposed connector pass through any already-placed label box?
        """
        for px, py, pha, pva, p_label_w, _pcx, _pcy in placed:
            py_min, py_max = _label_y_extent(py, pva, label_h)
            px_left, px_right = _label_x_extent(px, pha, p_label_w)
            if _segment_intersects_rect(
                conn_x0, conn_y0, conn_x1, conn_y1, px_left, px_right, py_min, py_max
            ):
                return True
        return False

    def label_through_placed_connector(x_pos: float, y_pos: float, va: str, ha: str) -> bool:
        """
        Would the new label box be crossed by an already-placed label's
        connector?
        """
        y_min, y_max = _label_y_extent(y_pos, va, label_h)
        lbl_left, lbl_right = _label_x_extent(x_pos, ha, label_w)
        for px, py, _pha, _pva, _p_label_w, pcx, pcy in placed:
            if pcx is None or pcy is None:
                continue
            if _segment_intersects_rect(pcx, pcy, px, py, lbl_left, lbl_right, y_min, y_max):
                return True
        return False

    def connector_crosses_tl_line(conn_x0: float, conn_y0: float, conn_x1: float, conn_y1: float) -> bool:
        """
        Does the connector cross any TL line in this column other than its own
        source?
        """
        line_eps = 0.05 * label_h  # treat TL lines as thin rectangles of this height
        for ly in line_y_positions:
            if abs(ly - conn_y0) < line_eps:  # the source TL we're connecting from
                continue
            if _segment_intersects_rect(
                conn_x0,
                conn_y0,
                conn_x1,
                conn_y1,
                line_left,
                line_right,
                ly - line_eps,
                ly + line_eps,
            ):
                return True
        return False

    def _side_candidates(tl_eV: float) -> list[tuple]:
        """
        Return the off-column candidate positions for one TL: direct left/right (no
        connector) plus the four diagonal positions (with connector). Ordered with the
        spacier side first.
        """
        y_tol = label_h
        right_obstacle_x = xlim[1]
        left_obstacle_x = xlim[0]
        if neighbor_columns:
            for nx, nhw, ny_list in neighbor_columns:
                if not any(abs(ny - tl_eV) <= y_tol for ny in ny_list):
                    continue
                if nx - nhw > line_right:
                    right_obstacle_x = min(right_obstacle_x, nx - nhw)
                elif nx + nhw < line_left:
                    left_obstacle_x = max(left_obstacle_x, nx + nhw)
        right_clearance = right_obstacle_x - (side_x_right + label_w)
        left_clearance = (side_x_left - label_w) - left_obstacle_x
        right_first = right_clearance >= left_clearance
        direct_side_first = (
            (side_x_right, tl_eV, "left", "center", None)
            if right_first
            else (side_x_left, tl_eV, "right", "center", None)
        )
        direct_side_second = (
            (side_x_left, tl_eV, "right", "center", None)
            if right_first
            else (side_x_right, tl_eV, "left", "center", None)
        )
        # direct sides first (no connector), then diagonals (off-column with diagonal
        # connector). The spacier side is tried first when both are valid. Diagonals are
        # offset by 1.6*label_h so a diagonal label is clearly distinct from a direct-side
        # one on the same side for a closely-spaced neighbouring TL:
        diag_dy = 1.6 * label_h
        return [
            direct_side_first,
            direct_side_second,
            (side_x_right, tl_eV + diag_dy, "left", "center", tl_eV),
            (side_x_left, tl_eV + diag_dy, "right", "center", tl_eV),
            (side_x_right, tl_eV - diag_dy, "left", "center", tl_eV),
            (side_x_left, tl_eV - diag_dy, "right", "center", tl_eV),
        ]

    def _candidate_ok(x_pos, y_pos, ha, va, conn_y, strict: bool, source_y: float | None = None) -> bool:
        if collides_with_band(y_pos, va, x_pos, ha):  # NEVER allow band/figure-edge overlap
            return False
        if strict and (
            collides_with_tl_line(y_pos, va, x_pos, ha, source_y=source_y)
            or collides_with_placed(y_pos, va, x_pos, ha)
            or label_through_placed_connector(x_pos, y_pos, va, ha)
        ):
            return False
        if conn_y is not None and strict:
            conn_x0 = x_center if x_pos == x_center else (line_right if x_pos > x_center else line_left)
            if connector_through_placed(conn_x0, conn_y, x_pos, y_pos):
                return False
            if connector_crosses_tl_line(conn_x0, conn_y, x_pos, y_pos):
                return False
        return True

    # ----- Phase 1: try direct above / below for each TL (greedy, cheap). -----
    # Anything that doesn't fit cleanly inline becomes a "side-bound" TL handled in Phase 2.
    side_bound: list[tuple[int, tuple]] = []  # (result_index, tl_tuple)
    pending_placements: list[tuple[int, str, tuple]] = []  # (result_index, label, position)
    for tl_tuple in tls:
        tl_eV, charges, i_meta, j_meta, faded = tl_tuple[:5]
        i = len(results)
        results.append(None)  # placeholder, filled in phase 3
        if skip_faded and faded:
            continue
        label = _format_TL_charge_label(charges, i_meta=i_meta, j_meta=j_meta)

        chosen = None
        for cand in (
            (x_center, tl_eV + 0.4 * label_h, "center", "bottom", None),  # direct above
            (x_center, tl_eV - 0.4 * label_h, "center", "top", None),  # direct below
        ):
            if _candidate_ok(*cand, strict=True, source_y=tl_eV):
                chosen = cand
                break
        if chosen is not None:
            pending_placements.append((i, label, chosen))
            # commit immediately so later phase-1 TLs see this label as placed:
            x_pos, y_pos, ha, va, conn_y = chosen
            placed.append((x_pos, y_pos, ha, va, label_w, None, None))
            continue
        side_bound.append((i, tl_tuple))

    # ----- Phase 2: optimise the side-bound TLs as a group. -----
    # For each side-bound TL we generate its (up to 6) off-column candidates and pick the
    # combination of positions that minimises a total overlap cost. Brute-force search for
    # tractable sizes; greedy fallback for larger groups.
    if side_bound:
        side_candidates_per_tl = [
            [c for c in _side_candidates(tl[0]) if not collides_with_band(c[1], c[3], c[0], c[2])]
            for _, tl in side_bound
        ]
        # if any TL has no valid candidates, fall back to its first (band-violating ignored):
        for k, opts in enumerate(side_candidates_per_tl):
            if not opts:
                side_candidates_per_tl[k] = _side_candidates(side_bound[k][1][0])

        chosen_positions = _optimise_side_placements(
            side_candidates_per_tl=side_candidates_per_tl,
            placed_inline=list(placed),
            line_y_positions=line_y_positions,
            line_left=line_left,
            line_right=line_right,
            x_center=x_center,
            label_h=label_h,
            label_w=label_w,
        )

        for (i, tl_tuple), pos in zip(side_bound, chosen_positions, strict=True):
            tl_eV, charges, i_meta, j_meta, _ = tl_tuple[:5]
            label = _format_TL_charge_label(charges, i_meta=i_meta, j_meta=j_meta)
            pending_placements.append((i, label, pos))
            x_pos, y_pos, ha, va, conn_y = pos
            conn_from_x: float | None
            conn_from_y: float | None
            if conn_y is not None:
                conn_from_x = (
                    x_center if x_pos == x_center else (line_right if x_pos > x_center else line_left)
                )
                conn_from_y = conn_y
            else:
                conn_from_x = conn_from_y = None
            placed.append((x_pos, y_pos, ha, va, label_w, conn_from_x, conn_from_y))

    # ----- Phase 3: assemble results in original TL order. -----
    for i, label, pos in pending_placements:
        x_pos, y_pos, ha, va, conn_y = pos
        results[i] = (x_pos, y_pos, label, ha, va, conn_y)

    # placed_in_this_column = the new entries appended beyond the cross-column seed
    placed_in_this_column = placed[n_seeded:]
    return results, placed_in_this_column


def transition_level_diagram(
    defect_thermodynamics: "DefectThermodynamics",
    all_TLs: bool | str = "faded",
    defect_subset: list[str] | str | None = None,
    include_site_info: bool = False,
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
    orange shaded region at the top. Within each defect column, each transition
    level is drawn as a short horizontal line, labelled with the charge state
    transition (e.g. ``(+1/0)``). Metastable charge states are denoted with a
    ``*`` in the label.

    Args:
        defect_thermodynamics (|DefectThermodynamics|):
            |DefectThermodynamics| object containing the defects to plot.
        all_TLs (bool, str):
            Controls inclusion of single-electron transition levels involving
            metastable defect charge states (denoted with ``*`` in the
            labels). Allowed values:

            - ``"faded"`` (default): show all single-electron TLs, with
              metastable-containing TLs drawn as faded lines `without`
              labels (keeps the plot uncluttered).
            - ``"faded_labels"``: same as ``"faded"`` but `with` labels
              drawn for the faded metastable TLs too.
            - ``True``: show all single-electron TLs at full strength.
            - ``False``: show only the thermodynamic ground-state transition
              levels (i.e. those visible on the defect formation energy
              diagram).
        defect_subset (list[str], str):
            If provided, only defects whose name contains at least one of the
            given substrings are plotted (e.g. ``["v_", "Te_Cd"]`` would keep
            all vacancies plus ``Te_Cd``). A bare string is treated as a
            single-element list. (Default: ``None`` -- all defects)
        include_site_info (bool):
            Whether to include site info in defect names in the column
            headers (e.g. ``$V_{Cd_{Td}}$`` rather than ``$V_{Cd}$``).
            Defaults to ``False``.
        ylim (tuple):
            Energy axis limits in eV (relative to VBM at 0). Defaults to
            ``(-0.05 * band_gap, 1.05 * band_gap)``.
        show_charge_labels (bool):
            Whether to label each transition level with its charge states
            (e.g. ``"(+1/0)"``). Defaults to ``True``.
        show_band_labels (bool):
            Whether to draw the "VBM" and "CBM" labels in the blue/orange
            band-edge shaded zones. If ``None`` (default), they are shown
            only if they would not overlap any transition level label
            (with the right side tried first, then the left); if both
            sides would clash they are hidden. ``True`` forces them on
            the right; ``False`` hides them.
        label_fontsize (float):
            Font size for the transition level charge labels. Defaults to
            ~70% of the current ``font.size`` rcParam.
        column_width (float):
            Width (in axes units) of the horizontal line segments inside each
            defect column, on a scale where the column spacing is 1. Defaults
            to ``0.4``.
        figsize (tuple):
            ``(width, height)`` of the figure in inches. Defaults to a width
            that scales with the number of defects.
        filename (PathLike):
            If set, save the figure to this path. (Default: None)

    Returns:
        ``matplotlib`` ``Figure`` object.
    """
    if defect_thermodynamics.band_gap is None:
        raise ValueError(
            "`band_gap` is not set for `DefectThermodynamics`, cannot plot transition levels."
        )

    if all_TLs not in (False, True, "faded", "faded_labels"):
        raise ValueError(f"`all_TLs` must be False, True, 'faded' or 'faded_labels', not {all_TLs!r}")

    faded_labels = all_TLs == "faded_labels"
    tl_data = _get_transition_level_data(defect_thermodynamics, all_TLs=all_TLs)
    tl_data = _filter_by_defect_subset(tl_data, defect_subset)
    if not tl_data:
        raise ValueError(
            "No defects with transition levels to plot"
            + (f" (after `defect_subset={defect_subset!r}` filter)" if defect_subset else "")
            + "."
        )

    if ylim is None:
        margin = max(0.05 * defect_thermodynamics.band_gap, 0.05)
        ylim = (-margin, defect_thermodynamics.band_gap + margin)

    n_defects = len(tl_data)
    half_w = column_width / 2.0
    styled_font_size = plt.rcParams["font.size"]
    if label_fontsize is None:
        label_fontsize = styled_font_size * 0.7

    # estimate label horizontal extent (in data units = column spacing) so we can extend xlim
    # to leave room for labels at the sides of the outer columns. ~7 characters at fontsize:
    approx_xrange = float(n_defects)
    if figsize is None:
        styled_figsize = plt.rcParams["figure.figsize"]
        figsize_w = max(styled_figsize[0], 0.8 * n_defects + 1.0)
        figsize_h = styled_figsize[1] * 1.15
    else:
        figsize_w, figsize_h = figsize
    label_width_eV_est = max(
        7 * (label_fontsize * 0.55 / 72.0) * approx_xrange / max(figsize_w, 1.0),
        0.15,
    )
    side_pad = half_w + label_width_eV_est + 0.1
    # the y-axis ticks point inward (xtick.direction=in) so the left-most labels need a bit
    # of extra clearance to avoid sitting on top of the tick marks:
    left_extra_pad = 0.15
    if figsize is None:
        # widen the figure to accommodate the side padding without squishing column spacing:
        figsize = (figsize_w + 0.8 * (2 * side_pad + left_extra_pad), figsize_h)

    fig, ax = plt.subplots(figsize=figsize)

    # shade band edge regions across the full x-range; extend xlim past the outer columns so
    # there is room to place direct-side / diagonal labels off the left- and right-most columns:
    xlim = (-side_pad - left_extra_pad, n_defects - 1 + side_pad)
    _shade_band_edges(ax, defect_thermodynamics.band_gap, xlim, ylim, orientation="vertical")

    # plot lines and labels for each defect:
    # minimum vertical spacing (in eV) between successive labels so they don't overlap;
    # scales with the height (in points) of the label text:
    label_offset_eV = max(
        (label_fontsize / 72.0) * (ylim[1] - ylim[0]) / max(figsize[1], 1.0) * 1.4,
        0.04,
    )
    # rough horizontal extent of a typical label in axes (data) units, for collision checks;
    # ~7 characters wide at the given font size:
    label_width_eV = max(
        7 * (label_fontsize * 0.55 / 72.0) * (xlim[1] - xlim[0]) / max(figsize[0], 1.0),
        0.15,
    )
    line_lw = plt.rcParams["lines.linewidth"] * 1.1
    faded_alpha = 0.4

    # column headers sit a small distance above ylim[1]; labels for TLs near/inside the CBM
    # (orange) or VBM (blue) band-edge zones are allowed to extend a little past ylim[1] /
    # below ylim[0] (symmetrically), so their labels can be placed directly above/below the
    # TL line even when the TL itself sits inside a band-edge zone:
    header_pad_frac = 0.08
    header_y = ylim[1] + header_pad_frac * (ylim[1] - ylim[0])
    label_buf = 0.35 * (header_y - ylim[1])
    label_y_max = ylim[1] + label_buf
    label_y_min = ylim[0] - label_buf

    # pre-build per-column (x_center, half_w, [TL y-positions in range]) so that for each column
    # we can pass the OTHER columns as neighbour data to inform side-clearance picking:
    columns_data: list[tuple[float, float, list[float]]] = []
    for cnt, (_dn, tls_for_col) in enumerate(tl_data.items()):
        in_range_y = [tl[0] for tl in tls_for_col if ylim[0] <= tl[0] <= ylim[1]]
        columns_data.append((float(cnt), half_w, in_range_y))

    # Build per-column TL lists and headers, draw the TL lines, then do label placement
    # globally with iterative cross-column refinement (so a label on the right side of column
    # A can be moved if it overlaps a label on the left side of column A+1, and vice versa).
    defect_items = list(tl_data.items())
    column_in_range_tls: list[list] = []
    formatted_names: list[str] = []
    for cnt, (defect_name, tls) in enumerate(defect_items):
        x_center = float(cnt)
        try:
            header = (
                format_defect_name(
                    defect_species=defect_name,
                    include_site_info=include_site_info,
                    wout_charge=True,
                )
                or defect_name
            )
        except Exception:
            header = defect_name
        formatted_names.append(header)

        in_range_tls = [tl for tl in tls if ylim[0] <= tl[0] <= ylim[1]]
        column_in_range_tls.append(in_range_tls)

        # draw TL lines (faded grey for metastable-containing TLs when all_TLs="faded"):
        for tl_eV, _charges, _i_meta, _j_meta, faded in in_range_tls:
            ax.plot(
                [x_center - half_w, x_center + half_w],
                [tl_eV, tl_eV],
                color="0.45" if faded else "k",
                alpha=faded_alpha if faded else 1.0,
                lw=line_lw,
                solid_capstyle="butt",
                zorder=3,
            )

    if show_charge_labels:
        column_placed: list[list] = [[] for _ in range(n_defects)]
        column_positions: list[list | None] = [None] * n_defects

        def _place_column(cnt: int):
            cross_column = [p for k in range(n_defects) if k != cnt for p in column_placed[k]]
            neighbor_cols = [c for i, c in enumerate(columns_data) if i != cnt]
            assert ylim is not None  # typing
            assert defect_thermodynamics.band_gap is not None  # typing
            return _place_labels_for_column(
                tls=column_in_range_tls[cnt],
                x_center=float(cnt),
                half_w=half_w,
                band_gap=defect_thermodynamics.band_gap,
                ylim=ylim,
                xlim=xlim,
                label_offset_eV=label_offset_eV,
                label_width_eV=label_width_eV,
                skip_faded=not faded_labels,
                header_y_min=label_y_max,
                label_y_min=label_y_min,
                neighbor_columns=neighbor_cols,
                cross_column_placed=cross_column,
            )

        # initial pass: each column sees only earlier columns' placements as obstacles
        for cnt in range(n_defects):
            positions, placed_in_column = _place_column(cnt)
            column_positions[cnt] = positions
            column_placed[cnt] = placed_in_column

        # global refinement: re-pick each column's labels with FULL cross-column context
        # (so a column can now see later-column placements too). Iterate until stable.
        for _ in range(3):
            changed = False
            for cnt in range(n_defects):
                positions, placed_in_column = _place_column(cnt)
                if placed_in_column != column_placed[cnt]:
                    column_positions[cnt] = positions
                    column_placed[cnt] = placed_in_column
                    changed = True
            if not changed:
                break
    else:
        column_positions = [None] * n_defects

    # draw labels (and their connectors) for each column:
    for cnt in range(n_defects):
        positions = column_positions[cnt]
        if not show_charge_labels or positions is None:
            continue
        x_center = float(cnt)
        in_range_tls = column_in_range_tls[cnt]
        for position, tl_tuple in zip(positions, in_range_tls, strict=True):
            if position is None:  # faded TL with skip_faded=True -- no label drawn
                continue
            x_pos, y_pos, label, ha, va, conn_y = position
            faded = tl_tuple[4]
            ax.text(
                x_pos,
                y_pos,
                label,
                ha=ha,
                va=va,
                fontsize=label_fontsize,
                color="0.55" if faded else "0.2",
                alpha=faded_alpha if faded else 1.0,
                zorder=4,
                clip_on=False,
            )
            if conn_y is not None:
                # draw a thin connector that stops a little short of both the TL line and the
                # label, so it doesn't visually touch either (10% gap at the line, 10% at the label):
                conn_x0 = (
                    x_center
                    if x_pos == x_center
                    else (x_center + half_w if x_pos > x_center else x_center - half_w)
                )
                conn_x_start = conn_x0 + 0.2 * (x_pos - conn_x0)
                conn_y_start = conn_y + 0.2 * (y_pos - conn_y)
                conn_x_end = conn_x0 + 0.8 * (x_pos - conn_x0)
                conn_y_end = conn_y + 0.8 * (y_pos - conn_y)
                ax.plot(
                    [conn_x_start, conn_x_end],
                    [conn_y_start, conn_y_end],
                    color="0.55" if faded else "0.4",
                    alpha=faded_alpha if faded else 1.0,
                    lw=line_lw * 0.5,
                    zorder=2.5,
                )

    # add VBM / CBM labels in the shaded band-edge zones, avoiding overlap with TL labels.
    # If `show_band_labels` is None we try the right side first (preferred), then the left
    # if the right would overlap any placed TL label; if both sides clash we omit the labels.
    if show_band_labels is not False:
        force_right = show_band_labels is True
        all_placed_label_boxes: list[tuple[float, float, float, float]] = []
        if show_charge_labels:
            for col_placed in column_placed:
                for px, py, pha, pva, p_label_w, _pcx, _pcy in col_placed:
                    px_left, px_right = _label_x_extent(px, pha, p_label_w)
                    py_min, py_max = _label_y_extent(py, pva, label_offset_eV)
                    all_placed_label_boxes.append((px_left, px_right, py_min, py_max))

        # estimated band-label box size (~3 chars wide, full font height) in data units:
        band_label_w = max(
            3 * (styled_font_size * 0.55 / 72.0) * (xlim[1] - xlim[0]) / max(figsize[0], 1.0), 0.15
        )
        band_label_h = max((styled_font_size / 72.0) * (ylim[1] - ylim[0]) / max(figsize[1], 1.0), 0.05)

        def _band_label_overlaps(x: float, ha: str, y: float) -> bool:
            lx_min, lx_max = _label_x_extent(x, ha, band_label_w)
            ly_min, ly_max = _label_y_extent(y, "center", band_label_h)
            return any(
                lx_max > b[0] and lx_min < b[1] and ly_max > b[2] and ly_min < b[3]
                for b in all_placed_label_boxes
            )

        vbm_y = ylim[0] + 0.5 * (0 - ylim[0])
        cbm_y = defect_thermodynamics.band_gap + 0.5 * (ylim[1] - defect_thermodynamics.band_gap)
        for text, y in (("VBM", vbm_y), ("CBM", cbm_y)):
            # decide side: right preferred; fall back to left; omit if both overlap (unless forced)
            right_x = xlim[1] - 0.05
            left_x = xlim[0] + 0.05
            right_ok = not _band_label_overlaps(right_x, "right", y)
            if right_ok or force_right:
                x, ha = right_x, "right"
            elif not _band_label_overlaps(left_x, "left", y):
                x, ha = left_x, "left"
            else:
                continue  # both sides clash; omit
            ax.text(x, y, text, ha=ha, va="center", fontsize=styled_font_size, color="0.25", zorder=2)

    # column headers (defect names) at the top:
    for cnt, name in enumerate(formatted_names):
        ax.annotate(
            name,
            xy=(cnt, header_y),
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
