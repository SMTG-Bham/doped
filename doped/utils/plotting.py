"""
Code to analyse VASP defect calculations.

These functions are built from a combination of useful modules from pymatgen
and AIDE (by Adam Jackson and Alex Ganose), alongside substantial modification,
in the efforts of making an efficient, user-friendly package for managing and
analysing defect calculations, with publication-quality outputs.
"""
import contextlib
import re
import warnings
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, colors, ticker
from pymatgen.core.periodic_table import Element
from pymatgen.util.string import latexify

from doped.utils.symmetry import sch_symbols  # point group symbols


def _get_backend(save_format: str) -> Optional[str]:
    """
    Try use pycairo as backend if installed, and save_format is pdf.
    """
    backend = None
    if "pdf" in save_format:
        try:
            import cairo  # noqa: F401

            backend = "cairo"
        except ImportError:
            warnings.warn(
                "Unable to import pycairo. Defaulting to matplotlib's pdf backend, so default doped fonts "
                "may not be used. Try setting `save_format` to 'png' or doing `conda remove pycairo; "
                "conda install pycairo` if you want doped's default font."
            )
    return backend


def _chempot_warning(dft_chempots):
    if dft_chempots is None:
        warnings.warn(
            "You have not specified chemical potentials (`chempots`), so chemical potentials are set to "
            "zero for each species. This will give large errors in the absolute values of formation "
            "energies, but the transition level positions will be unaffected."
        )


def _get_plot_setup(colormap, xy):
    if colormap is None:  # future updated colour handling (based on defect type etc) should remove
        # the need for this!
        colormap = "Dark2" if len(xy) <= 8 else "tab20"
    cmap = colormaps[colormap] if isinstance(colormap, str) else colormap
    colors = cmap(np.linspace(0, 1, len(xy)))

    # generate plot:
    styled_fig_size = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=((2.6 / 3.5) * styled_fig_size[0], (1.95 / 3.5) * styled_fig_size[1]))
    # Gives a final figure width matching styled_fig_size, with dimensions matching the doped default
    styled_font_size = plt.rcParams["font.size"]
    styled_linewidth = plt.rcParams["lines.linewidth"]
    styled_markersize = plt.rcParams["lines.markersize"]

    return cmap, colors, fig, ax, styled_fig_size, styled_font_size, styled_linewidth, styled_markersize


def _plot_formation_energy_lines(
    xy,
    colors,
    ax,
    styled_linewidth,
    styled_markersize,
    alpha=1.0,
):
    defect_names_for_legend = []
    for cnt, def_name in enumerate(xy.keys()):  # plot formation energy lines
        ax.plot(
            xy[def_name][0],
            xy[def_name][1],
            color=colors[cnt],
            markeredgecolor=colors[cnt],
            lw=styled_linewidth * 1.2,
            markersize=styled_markersize * (4 / 6),
            alpha=alpha,
        )
        defect_names_for_legend.append(def_name)

    return defect_names_for_legend


def _add_band_edges_and_axis_limits(ax, band_gap, xlim, ylim, fermi_level=None):
    ax.imshow(
        [(0, 1), (0, 1)],
        cmap=plt.cm.Blues,
        extent=(xlim[0], 0, ylim[0], ylim[1]),
        vmin=0,
        vmax=3,
        interpolation="bicubic",
        rasterized=True,
        aspect="auto",
    )

    ax.imshow(
        [(1, 0), (1, 0)],
        cmap=plt.cm.Oranges,
        extent=(band_gap, xlim[1], ylim[0], ylim[1]),
        vmin=0,
        vmax=3,
        interpolation="bicubic",
        rasterized=True,
        aspect="auto",
    )

    ax.set_xlim(xlim)
    # dashed line for E_formation = 0 in case ymin < 0
    ax.plot([xlim[0], xlim[1]], [0, 0], c="k", ls="--", alpha=0.7)
    ax.set_ylim(ylim)

    if fermi_level is not None:
        ax.axvline(x=fermi_level, linestyle="-.", color="k")
    ax.set_xlabel("Fermi Level (eV)")
    ax.set_ylabel("Formation Energy (eV)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def _set_title_and_save_figure(ax, fig, title, chempot_table, filename, styled_font_size):
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
        fig.savefig(
            filename, dpi=600, bbox_inches="tight", backend=_get_backend(filename), transparent=True
        )


def _format_defect_name(
    defect_species: str,
    include_site_info_in_name: bool,
    wout_charge: bool = False,
) -> Optional[str]:
    """
    Format defect name for plot titles. (i.e. from Cd_i_C3v_0 to $Cd_{i}^{0}$
    or $Cd_{i_{C3v}}^{0}$. Note this assumes "V_" means vacancy not Vanadium.

    Args:
        defect_species (:obj:`str`):
            Name of defect including charge state (e.g. Cd_i_C3v_0)
        include_site_info_in_name (:obj:`bool`):
            Whether to include site info in name (e.g. $Cd_{i}^{0}$ or
            $Cd_{i_{C3v}}^{0}$).
        wout_charge (:obj:`bool`, optional):
            Whether the charge state is included in the defect_species name.
            Defaults to False.

    Returns:
        :obj:`str`:
            formatted defect name
    """
    if wout_charge:
        defect_species += "_99"  # add dummy charge for parsing; 99 red balloons go by...

    if not isinstance(defect_species, str):  # Check inputs
        raise (TypeError(f"`defect_species` {defect_species} should be a string"))
    try:
        charge = int(defect_species.split("_")[-1])  # charge comes last
        charge_string = f"{charge:+}" if charge > 0 else f"{charge}"
    except ValueError as e:
        raise ValueError(
            f"Problem reading defect name {defect_species}, should end with charge state "
            f"after underscore (e.g. Te_i_Td_Te2.83_+1)"
        ) from e

    # Format defect name for title/axis labels:
    recognised_pre_vacancy_strings = sorted(
        [
            "v_",
            "v",
            "va_",
            "Va_",
            "va",
            "Va",
            "V_",
            "V",
            "Vac",
            "vac",
            "Vac_",
            "vac_",
        ],
        key=len,
        reverse=True,
    )
    recognised_post_vacancy_strings = sorted(
        [
            "_v",  # but not '_V' as could be vanadium
            "v",  # but not 'V' as could be vanadium
            "_vac",
            "_Vac",
            "vac",
            "Vac",
            "va",
            "Va",
            "_va",
            "_Va",
        ],
        key=len,
        reverse=True,
    )
    recognised_pre_interstitial_strings = sorted(
        [
            "i",  # but not 'I' as could be iodine
            "i_",  # but not 'I_' as could be iodine
            "Int",
            "int",
            "Int_",
            "int_",
            "Inter",
            "inter",
            "Inter_",
            "inter_",
        ],
        key=len,
        reverse=True,
    )
    recognised_post_interstitial_strings = sorted(
        [
            "_i",  # but not '_I' as could be iodine
            "_int",
            "_Int",
            "int",
            "Int",
            "inter",
            "Inter",
            "_inter",
            "_Inter",
        ],
        key=len,
        reverse=True,
    )

    defect_name = None
    dummy_h = Element("H")
    pre_charge_name = defect_species.rsplit("_", 1)[0]  # defect name without charge state
    trimmed_pre_charge_name = pre_charge_name  # later trimmed to remove any pre or post
    # vacancy/interstitial strings from name

    doped_site_info = None
    # check if name is doped format, having site info as point group symbol (and more) after 2nd "_":
    with contextlib.suppress(IndexError):
        point_group_symbol = defect_species.split("_")[2]
        if point_group_symbol in sch_symbols:  # recognised point group symbol?
            # from 2nd underscore to last underscore (before charge state) is site info
            # convert point group symbol to formatted version (e.g. C1 -> C_1):
            formatted_point_group_symbol = (
                f"{point_group_symbol[0]}_{{{point_group_symbol[1:]}}}"  # already in math mode here
            )
            doped_site_info = formatted_point_group_symbol
            if defect_species.split("_")[3:-1]:  # if there is more site info after point group symbol
                doped_site_info += "-" + "-".join(defect_species.split("_")[3:-1])
            trimmed_pre_charge_name = pre_charge_name.replace(
                f"_{'_'.join(defect_species.split('_')[2:-1])}", ""
            )

    def _check_matching_defect_format(element, name, pre_def_type_list, post_def_type_list):
        return any(f"{pre_def_type}{element}" in name for pre_def_type in pre_def_type_list) or any(
            f"{element}{post_def_type}" in name for post_def_type in post_def_type_list
        )

    def _check_matching_defect_format_with_site_info(element, name, pre_def_type_list, post_def_type_list):
        for site_preposition in ["s", "m", "mult", ""]:
            for site_postposition in [r"[a-z]", ""]:
                match = re.match(
                    f"([a-z_]+)({site_preposition}[0-9]+{site_postposition})",
                    name,
                    re.I,
                )

                if match:
                    items = match.groups()
                    for match_generator in [
                        (
                            fstring in name
                            for pre_def_type in pre_def_type_list
                            for fstring in [
                                f"{pre_def_type}{items[1]}{element}",
                                f"{pre_def_type}{element}{items[1]}",
                                f"{pre_def_type}{items[1]}_{element}",
                                f"{pre_def_type}{element}_{items[1]}",
                            ]
                        ),
                    ]:
                        if any(match_generator):
                            return True, items[1].replace("mult", "m")

                    for match_generator in [
                        (
                            fstring in name
                            for post_def_type in post_def_type_list
                            for fstring in [
                                f"{element}{items[1]}{post_def_type}",
                                f"{items[1]}{element}{post_def_type}",
                                f"{element}{items[1]}_{post_def_type}",
                                f"{items[1]}_{element}{post_def_type}",
                            ]
                        ),
                    ]:
                        if any(match_generator):
                            return True, items[1].replace("mult", "m")

        return False, None

    def _try_vacancy_interstitial_match(
        element,
        name,
        include_site_info_in_name,
        pre_vacancy_strings=None,
        post_vacancy_strings=None,
        pre_interstitial_strings=None,
        post_interstitial_strings=None,
    ):
        if pre_vacancy_strings is None:
            pre_vacancy_strings = recognised_pre_vacancy_strings
        if post_vacancy_strings is None:
            post_vacancy_strings = recognised_post_vacancy_strings
        if pre_interstitial_strings is None:
            pre_interstitial_strings = recognised_pre_interstitial_strings
        if post_interstitial_strings is None:
            post_interstitial_strings = recognised_post_interstitial_strings
        defect_name = None
        defect_name_without_site_info = None
        defect_name_with_site_info = None

        match_found, site_info = _check_matching_defect_format_with_site_info(
            element,
            name,
            pre_vacancy_strings,
            post_vacancy_strings,
        )
        if match_found:
            defect_name_with_site_info = (
                f"$\\it{{V}}\\!$ $_{{{element}_{{{site_info}}}}}^{{{charge_string}}}$"
            )
            defect_name_without_site_info = f"$\\it{{V}}\\!$ $_{{{element}}}^{{{charge_string}}}$"

        else:
            match_found, site_info = _check_matching_defect_format_with_site_info(
                element,
                name,
                pre_interstitial_strings,
                post_interstitial_strings,
            )
            if match_found:
                defect_name_with_site_info = f"{element}$_{{i_{{{site_info}}}}}^{{{charge_string}}}$"
                defect_name_without_site_info = f"{element}$_i^{{{charge_string}}}$"

        if include_site_info_in_name and defect_name_with_site_info is not None:
            return defect_name_with_site_info

        if (
            _check_matching_defect_format(element, name, pre_vacancy_strings, post_vacancy_strings)
            and defect_name is None
        ):
            if include_site_info_in_name and doped_site_info is not None:
                return f"$\\it{{V}}\\!$ $_{{{element}_{{{doped_site_info}}}}}^{{{charge_string}}}$"

            return f"$\\it{{V}}\\!$ $_{{{element}}}^{{{charge_string}}}$"

        if (
            _check_matching_defect_format(
                element,
                name,
                pre_interstitial_strings,
                post_interstitial_strings,
            )
            and defect_name is None
        ):
            if include_site_info_in_name and doped_site_info is not None:
                return f"{element}$_{{i_{{{doped_site_info}}}}}^{{{charge_string}}}$"

            return f"{element}$_i^{{{charge_string}}}$"

        if defect_name is None and defect_name_without_site_info is not None:
            return defect_name_without_site_info

        return defect_name

    def _try_substitution_match(substituting_element, orig_site_element, name, include_site_info_in_name):
        defect_name = None
        if (
            f"{substituting_element}_{orig_site_element}" in name
            or f"{substituting_element}_on_{orig_site_element}" in name
        ):
            if include_site_info_in_name and doped_site_info is not None:
                defect_name = (
                    f"{substituting_element}$_{{{orig_site_element}_{{{doped_site_info}}}}}^"
                    f"{{{charge_string}}}$"
                )

            else:
                defect_name = f"{substituting_element}$_{{{orig_site_element}}}^{{{charge_string}}}$"

        if (
            defect_name and include_site_info_in_name
        ):  # if we have a match, check if we can add the site number
            for site_preposition in ["s", "m", "mult", ""]:
                for site_postposition in [r"[a-z]", ""]:
                    match = re.match(
                        f"([a-z_]+)({site_preposition}[0-9]+{site_postposition})",
                        name,
                        re.I,
                    )

                    if match:
                        items = match.groups()
                        if any(
                            fstring in name
                            for fstring in [
                                f"{items[1]}_{substituting_element}_{orig_site_element}",
                                f"{substituting_element}_{orig_site_element}_{items[1]}",
                                f"{items[1]}_{substituting_element}_on_{orig_site_element}",
                                f"{substituting_element}_on_{orig_site_element}_{items[1]}",
                            ]
                        ):
                            defect_name = (
                                f"{substituting_element}$_{{{orig_site_element}_{{{items[1]}}}}}^"
                                f"{{{charge_string}}}$"
                            )
                            return defect_name.replace("mult", "m")

        if defect_name:
            defect_name = defect_name.replace("mult", "m")

        return defect_name

    def _defect_name_from_matching_elements(element_matches, name, include_site_info_in_name):
        if len(element_matches) == 1:  # vacancy or interstitial?
            defect_name = _try_vacancy_interstitial_match(
                element_matches[0], name, include_site_info_in_name
            )
        elif len(element_matches) == 2:
            # try substitution/antisite match, if not try vacancy/interstitial with first element
            defect_name = _try_substitution_match(
                element_matches[0], element_matches[1], name, include_site_info_in_name
            )
            if defect_name is None:
                defect_name = _try_vacancy_interstitial_match(
                    element_matches[0], name, include_site_info_in_name
                )
        else:
            # try use first match and see if we match vacancy or interstitial format
            # if not, try first and second matches and see if we match substitution format
            # otherwise fail
            defect_name = _try_vacancy_interstitial_match(
                element_matches[0], name, include_site_info_in_name
            )
            if defect_name is None:
                defect_name = _try_substitution_match(
                    element_matches[0],
                    element_matches[1],
                    name,
                    include_site_info_in_name,
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

    two_character_pairs_in_name = [
        trimmed_pre_charge_name[i : i + 2]  # trimmed_pre_charge_name name for finding elements,
        # pre_charge_name for matching defect format
        for i in range(0, len(trimmed_pre_charge_name), 1)
        if len(trimmed_pre_charge_name[i : i + 2]) == 2
    ]
    possible_two_character_elements = [
        two_char_string
        for two_char_string in two_character_pairs_in_name
        if dummy_h.is_valid_symbol(two_char_string)
    ]

    if possible_two_character_elements:
        defect_name = _defect_name_from_matching_elements(
            possible_two_character_elements,
            pre_charge_name,  # trimmed_pre_charge_name name for finding elements, pre_charge_name
            # for matching defect format
            include_site_info_in_name,
        )

        if defect_name is None and len(possible_two_character_elements) == 1:
            # possibly one single-character element and one two-character element
            possible_one_character_elements = [
                character
                for character in trimmed_pre_charge_name.replace(possible_two_character_elements[0], "")
                if dummy_h.is_valid_symbol(character)
            ]
            if possible_one_character_elements:
                # in this case, we don't know the order of the 1-character vs 2-character elements in
                # the name, so we try both orderings:
                defect_name = _defect_name_from_matching_elements(
                    possible_two_character_elements + possible_one_character_elements,
                    pre_charge_name,  # trimmed_pre_charge_name name for finding elements,
                    # pre_charge_name for matching defect format
                    include_site_info_in_name,
                )
                if defect_name is None:
                    defect_name = _defect_name_from_matching_elements(
                        possible_one_character_elements + possible_two_character_elements,
                        pre_charge_name,  # trimmed_pre_charge_name name for finding elements,
                        # pre_charge_name for matching defect format
                        include_site_info_in_name,
                    )

    if defect_name is None:
        # try single-character element match
        possible_one_character_elements = [
            character
            for character in trimmed_pre_charge_name  # trimmed_pre_charge_name name for finding
            # elements, pre_charge_name for matching defect format
            if dummy_h.is_valid_symbol(character)
        ]

        if possible_one_character_elements:
            defect_name = _defect_name_from_matching_elements(
                possible_one_character_elements,
                pre_charge_name,  # trimmed_pre_charge_name name for finding elements,
                # pre_charge_name for matching defect format
                include_site_info_in_name,
            )

    if defect_name is None:
        # try matching to PyCDT/old-doped style:
        try:
            defect_type = defect_species.split("_")[0]  # vac, as or int
            if (
                defect_type.capitalize() == "Int"
            ):  # for interstitials, name formatting is different (eg Int_Cd_1 vs vac_1_Cd)
                site_element = defect_species.split("_")[1]
                site = defect_species.split("_")[2]
                if include_site_info_in_name:
                    # by default include defect site in defect name for interstitials
                    defect_name = f"{site_element}$_{{i_{{{site}}}}}^{{{charge_string}}}$"
                else:
                    defect_name = f"{site_element}$_i^{{{charge_string}}}$"
            else:
                site = defect_species.split("_")[1]  # number indicating defect site (from doped)
                site_element = defect_species.split("_")[2]  # element at defect site

            if include_site_info_in_name:  # whether to include the site number in defect name
                if defect_type.lower() == "vac":
                    defect_name = f"$\\it{{V}}\\!$ $_{{{site_element}_{{{site}}}}}^{{{charge_string}}}$"
                    # double brackets to treat it literally (tex), then extra {} for
                    # python str formatting
                elif defect_type.lower() in ["as", "sub"]:
                    subs_element = defect_species.split("_")[4]
                    defect_name = f"{site_element}$_{{{subs_element}_{{{site}}}}}^{{{charge_string}}}$"
                elif defect_type.capitalize() != "Int":
                    raise ValueError("Defect type not recognized. Please check spelling.")
            else:
                if defect_type.lower() == "vac":
                    defect_name = f"$\\it{{V}}\\!$ $_{{{site_element}}}^{{{charge_string}}}$"
                elif defect_type.lower() in ["as", "sub"]:
                    subs_element = defect_species.split("_")[4]
                    defect_name = f"{site_element}$_{{{subs_element}}}^{{{charge_string}}}$"
                elif defect_type.capitalize() != "Int":
                    raise ValueError(f"Defect type {defect_type} not recognized. Please check spelling.")
        except Exception:
            return None

    return f"{defect_name.rsplit('^', 1)[0]}$" if wout_charge else defect_name


def _get_legends_txt(for_legend, all_entries=False):
    # get latex-like legend titles
    legends_txt = []
    for defect_entry_name in for_legend:
        include_site_info = not all(  # all PyCDT/old-doped format, don't include site num
            any(name.startswith(i) for i in ["Int_", "vac_", "as_", "sub_"]) for name in for_legend
        )
        try:
            defect_name = _format_defect_name(
                defect_species=defect_entry_name,
                include_site_info_in_name=include_site_info,
                wout_charge=not all_entries,  # defect names without charge
            )

        except Exception:  # if formatting fails, just use the defect_species name
            defect_name = defect_entry_name

        # append "a,b,c.." for different defect species with the same name
        if any(defect_name in i for i in legends_txt):
            i = 3

            if defect_name in legends_txt:  # first repeat, direct match, rename previous entry
                # find index of previous defect_name, and rename
                prev_idx = legends_txt.index(defect_name)
                legends_txt[prev_idx] = f"{defect_name}$_{{-{chr(96 + 1)}}}$"  # a
                defect_name = f"{defect_name}$_{{-{chr(96 + 2)}}}$"  # b

            else:
                defect_name = f"{defect_name}$_{{-{chr(96 + i)}}}$"  # c

            while defect_name in legends_txt:
                i += 1
                defect_name = f"{defect_name}$_{{-{chr(96 + i)}}}$"  # d, e, f etc

        legends_txt.append(defect_name)

    return legends_txt


def _get_formation_energy_lines(defect_thermodynamics, dft_chempots, xlim):
    xy, all_lines_xy = {}, {}  # dict of {defect_name: [[x_vals],[y_vals]]}
    y_range_vals, all_entries_y_range_vals = (
        [],
        [],
    )  # for finding max/min values on y-axis based on x-limits
    lower_cap, upper_cap = -100, 100  # arbitrary values to extend lines to
    ymin = 0

    for def_name, defect_entry_list in defect_thermodynamics.all_entries.items():
        for defect_entry in defect_entry_list:
            charge = defect_entry.charge_state
            # all_lines name includes charge state:
            defect_entry_name = f"{def_name}_{'+' if charge > 0 else ''}{charge}"
            all_lines_xy[defect_entry_name] = [[], []]
            for x_extrem in [lower_cap, upper_cap]:
                all_lines_xy[defect_entry_name][0].append(x_extrem)
                all_lines_xy[defect_entry_name][1].append(
                    defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=x_extrem
                    )
                )
                all_entries_y_range_vals.extend(
                    defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=x_window
                    )
                    for x_window in xlim
                )

    for def_name, def_tl in defect_thermodynamics.transition_level_map.items():
        xy[def_name] = [[], []]

        if def_tl:
            org_x = sorted(def_tl.keys())
            # establish lower x-bound
            first_charge = max(def_tl[org_x[0]])
            for defect_entry in defect_thermodynamics.stable_entries[def_name]:
                if defect_entry.charge_state == first_charge:
                    form_en = defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=lower_cap
                    )
                    fe_left = defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=xlim[0]
                    )
            xy[def_name][0].append(lower_cap)
            xy[def_name][1].append(form_en)
            y_range_vals.append(fe_left)

            # iterate over stable charge state transitions
            for fl in org_x:
                charge = max(def_tl[fl])
                for defect_entry in defect_thermodynamics.stable_entries[def_name]:
                    if defect_entry.charge_state == charge:
                        form_en = defect_thermodynamics.get_formation_energy(
                            defect_entry, chempots=dft_chempots, fermi_level=fl
                        )
                xy[def_name][0].append(fl)
                xy[def_name][1].append(form_en)
                y_range_vals.append(form_en)

            # establish upper x-bound
            last_charge = min(def_tl[org_x[-1]])
            for defect_entry in defect_thermodynamics.stable_entries[def_name]:
                if defect_entry.charge_state == last_charge:
                    form_en = defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=upper_cap
                    )
                    fe_right = defect_thermodynamics.get_formation_energy(
                        defect_entry, chempots=dft_chempots, fermi_level=xlim[1]
                    )
            xy[def_name][0].append(upper_cap)
            xy[def_name][1].append(form_en)
            y_range_vals.append(fe_right)

        else:  # no transition level -> only one stable charge state, add all_lines_xy and extend
            # y_range_vals; means this is only a 1-pump (chmp) loop
            def_name_w_charge = [i for i in all_lines_xy if i.startswith(f"{def_name}_")][0]
            xy[def_name] = all_lines_xy[def_name_w_charge]  # get xy from all_lines_xy, using name w/charge
            defect_entry = defect_thermodynamics.stable_entries[def_name][0]
            y_range_vals.extend(
                defect_thermodynamics.get_formation_energy(
                    defect_entry, chempots=dft_chempots, fermi_level=x_window
                )
                for x_window in xlim
            )

        # if xy corresponds to a line below 0 for all x in (0, band_gap), warn!
        yvals = _get_in_gap_yvals(xy[def_name][0], xy[def_name][1], (0, defect_thermodynamics.band_gap))
        if all(y < 0 for y in yvals):  # Check if all y-values are below zero
            warnings.warn(
                f"All formation energies for {def_name} are below zero across the "
                f"entire band gap range. This is typically unphysical (see docs), and likely due to "
                f"mis-specification of chemical potentials (see docstrings and/or tutorials)."
            )
            ymin = min(ymin, *yvals)  # TODO: Test this

    return (xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin


def _get_ylim_from_y_range_vals(y_range_vals, ymin=0, auto_labels=False):
    window = max(y_range_vals) - min(y_range_vals)
    spacer = 0.1 * window
    ylim = (ymin, max(y_range_vals) + spacer)
    if auto_labels:  # need to manually set xlim or ylim if labels cross axes!!
        ylim = (ymin, max(y_range_vals) * 1.17) if spacer / ylim[1] < 0.145 else ylim
        # Increase y_limit to give space for transition level labels

    return ylim


def _get_in_gap_yvals(x_coords, y_coords, x_range):
    relevant_x = np.linspace(x_range[0], x_range[1], 100)  # x values in range
    return np.interp(relevant_x, x_coords, y_coords)  # y values in range


def _TLD_plot(
    defect_thermodynamics,
    dft_chempots=None,
    el_refs=None,
    chempot_table=True,
    all_entries: Union[bool, str] = False,
    xlim=None,
    ylim=None,
    fermi_level=None,
    title=None,
    colormap: Optional[Union[str, colors.Colormap]] = None,
    auto_labels=False,
    filename=None,
):
    """
    Produce defect Formation energy vs Fermi energy plot
    Args:
        dft_chempots:
            a dictionary of {Element:value} giving the chemical
            potential of each element
        xlim:
            Tuple (min,max) giving the range of the x (fermi energy) axis. This may need to be
            set manually when including transition level labels, so that they don't cross the axes.
        ylim:
            Tuple (min,max) giving the range for the formation energy axis. This may need to be
            set manually when including transition level labels, so that they don't cross the axes.

    Returns:
        a matplotlib object.
    """
    _chempot_warning(dft_chempots)
    if xlim is None:
        xlim = (-0.3, defect_thermodynamics.band_gap + 0.3)

    (xy, y_range_vals), (all_lines_xy, all_entries_y_range_vals), ymin = _get_formation_energy_lines(
        defect_thermodynamics, dft_chempots, xlim
    )

    (
        cmap,
        colors,
        fig,
        ax,
        styled_fig_size,
        styled_font_size,
        styled_linewidth,
        styled_markersize,
    ) = _get_plot_setup(colormap, all_lines_xy if all_entries is True else xy)

    defect_names_for_legend = _plot_formation_energy_lines(  # plot formation energies and get legend names
        all_lines_xy if all_entries is True else xy,
        colors=colors,
        ax=ax,
        styled_linewidth=styled_linewidth,
        styled_markersize=styled_markersize,
    )

    if all_entries == "faded":  # Redo `for` loop so grey 'all_lines_xy' not included in legend
        _legend = _plot_formation_energy_lines(
            all_lines_xy,
            colors=[(0.8, 0.8, 0.8)] * len(all_lines_xy),
            ax=ax,
            styled_linewidth=styled_linewidth,
            styled_markersize=styled_markersize,
            alpha=0.5,
        )

    for cnt, def_name in enumerate(xy.keys()):  # plot transition levels
        x_trans: List[float] = []
        y_trans: List[float] = []
        tl_labels, tl_label_type = [], []
        for x_val, chargeset in defect_thermodynamics.transition_level_map[def_name].items():
            x_trans.append(x_val)
            y_trans.append(
                [
                    defect_thermodynamics.get_formation_energy(
                        defect_entry,
                        chempots=dft_chempots,
                        fermi_level=x_val,
                    )
                    for defect_entry in defect_thermodynamics.stable_entries[def_name]
                    if defect_entry.charge_state == chargeset[0]
                ][0]
            )
            tl_labels.append(
                rf"$\epsilon$({max(chargeset):{'+' if max(chargeset) else ''}}/"
                f"{min(chargeset):{'+' if min(chargeset) else ''}})"
            )
            tl_label_type.append("start_positive" if max(chargeset) > 0 else "end_negative")
        if x_trans:
            ax.plot(
                x_trans,
                y_trans,
                marker="o",
                color=colors[cnt],
                markeredgecolor=colors[cnt],
                lw=styled_linewidth * 1.2,
                markersize=styled_markersize * (4 / 6),
                fillstyle="full",
            )
            if auto_labels:
                for index, coords in enumerate(zip(x_trans, y_trans)):
                    text_alignment = "right" if tl_label_type[index] == "start_positive" else "left"
                    ax.annotate(
                        tl_labels[index],  # this is the text
                        coords,  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 5),  # distance from text to points (x,y)
                        ha=text_alignment,  # horizontal alignment of text
                        size=styled_font_size * 0.9,
                        annotation_clip=True,
                    )  # only show label if coords in current axes

    ax.legend(
        _get_legends_txt(
            [defect_entry.name for defect_entry in defect_thermodynamics.defect_entries]
            if all_entries is True
            else defect_names_for_legend,
            all_entries=all_entries,
        ),
        loc=2,
        bbox_to_anchor=(1, 1),
    )

    if ylim is None:
        ylim = _get_ylim_from_y_range_vals(
            all_entries_y_range_vals if all_entries is True else y_range_vals,
            ymin=ymin,
            auto_labels=auto_labels,
        )

    _add_band_edges_and_axis_limits(
        ax, defect_thermodynamics.band_gap, xlim, ylim, fermi_level=fermi_level
    )  # Show colourful band edges
    if chempot_table and dft_chempots:
        _plot_chemical_potential_table(ax, dft_chempots, loc="left", el_refs=el_refs)

    _set_title_and_save_figure(ax, fig, title, chempot_table, filename, styled_font_size)

    return fig


def _plot_chemical_potential_table(
    ax,
    dft_chempots,
    loc="left",
    el_refs=None,
):
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
    tab = ax.table(cellText=[text_list], colLabels=labels, colWidths=widths, loc="top", cellLoc=loc)
    tab.auto_set_column_width(list(range(len(widths))))

    for cell in tab.get_celld().values():
        cell.set_linewidth(0)

    return tab
