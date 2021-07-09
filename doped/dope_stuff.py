# coding: utf-8

"""
Code to analyse VASP defect calculation results, and other dope things.
These functions are built from a combination of extremely useful modules
from pymatgen and AIDE (by Adam Jackson and Alex Ganose), alongside
substantial modification, in the efforts of making an
efficient, user-friendly package for managing and analysing defect
calculations, with publication-quality outputs.
"""

__author__ = "Seán Kavanagh"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Seán Kavanagh"
__email__ = "sean.kavanagh.19@ucl.ac.uk"
__date__ = "June 18, 2020"


from operator import itemgetter
import pickle
from typing import Any
import warnings
import numpy as np
import copy
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc

from tabulate import tabulate
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram
from pymatgen.util.string import latexify, unicodeify
from doped import aide_murphy_correction

default_fonts = [
    "Whitney Book Extended",
    "Arial",
    "Whitney Book",
    "Helvetica",
    "Liberation Sans",
    "Andale Sans",
]


def dpd_from_parsed_defect_dict(parsed_defect_dict: dict) -> DefectPhaseDiagram:
    """Generates a DefectPhaseDiagram object from a dictionary of parsed defect calculations
    (presumably created using SingleDefectParser
    from doped.pycdt.utils.parse_calculations), which can then be plotted (using
    pretty_formation_energy_plot) to get formation energies, transition levels etc.
    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (presumably created using SingleDefectParser
            from doped.pycdt.utils.parse_calculations) (see example notebook)
            Must have 'vbm' and 'gap' in defect.parameters for each defect (from
            SingleDefectParser.get_bulk_gap_data())
    Returns:
        DefectPhaseDiagram object
    """
    vbm_vals = []
    bandgap_vals = []
    for defect in parsed_defect_dict.values():
        vbm_vals.append(defect.parameters["vbm"])
        bandgap_vals.append(defect.parameters["gap"])
    if len(set(vbm_vals)) > 1:  # Check if all defects give same vbm
        raise ValueError(
            f"VBM values don't match for defects in given defect dictionary, "
            f"the VBM values in the dictionary are: {vbm_vals}. "
            f"Are you sure the correct/same bulk files were used with "
            f"SingleDefectParser and/or get_bulk_gap_data()?"
        )
    if len(set(bandgap_vals)) > 1:  # Check if all defects give same bandgap
        raise ValueError(
            f"Bandgap values don't match for defects in given defect dictionary, "
            f"the bandgap values in the dictionary are: {bandgap_vals}. "
            f"Are you sure the correct/same bulk files were used with "
            f"SingleDefectParser and/or get_bulk_gap_data()?"
        )
    vbm = vbm_vals[0]
    bandgap = bandgap_vals[0]
    dpd = DefectPhaseDiagram(
        list(parsed_defect_dict.values()), vbm, bandgap, filter_compatible=False
    )

    return dpd


def suggest_larger_supercells(defect_phase_diagram: DefectPhaseDiagram, tolerance=0.1):
    """
    This is a pymatgen DefectPhaseDiagram method that has a bug in it, so I've rewritten the code
    without the bug here.
    Suggest larger supercells for different defect+chg combinations based on use of
    compatibility analysis. Does this for any charged defects which have is_compatible = False,
    and the defect+chg formation energy is stable at fermi levels within the band gap.
    NOTE: Requires self.filter_compatible = False
    Args:
        tolerance (float): tolerance with respect to the VBM and CBM for considering
                           larger supercells for a given charge
    """
    if defect_phase_diagram.filter_compatible:
        raise ValueError("Cannot suggest larger supercells if filter_compatible is True.")
    recommendations = {}
    for def_type in defect_phase_diagram.defect_types:
        template_entry = defect_phase_diagram.stable_entries[def_type][0].copy()
        defect_indices = [int(def_ind) for def_ind in def_type.split("@")[-1].split("-")]
        for charge in defect_phase_diagram.finished_charges[def_type]:
            chg_defect = template_entry.defect.copy()
            chg_defect.set_charge(charge)
            for entry_index in defect_indices:
                entry = defect_phase_diagram.entries[entry_index]
                if entry.charge == charge:
                    break
            if entry.parameters.get("is_compatible", True):
                continue
            else:
                # consider if transition level is within
                # tolerance of band edges
                suggest_bigger_supercell = True
                for tl, chgset in defect_phase_diagram.transition_level_map[def_type].items():
                    sorted_chgset = list(chgset)
                    sorted_chgset.sort(reverse=True)
                    if charge == sorted_chgset[0] and tl < tolerance:
                        suggest_bigger_supercell = False
                    elif charge == sorted_chgset[1] and tl > (
                        defect_phase_diagram.band_gap - tolerance
                    ):
                        suggest_bigger_supercell = False
            if suggest_bigger_supercell:
                if def_type not in recommendations:
                    recommendations[def_type] = []
                recommendations[def_type].append(charge)
    return recommendations


def dpd_transition_levels(defect_phase_diagram: DefectPhaseDiagram):
    for def_type, tl_info in defect_phase_diagram.transition_level_map.items():
        bold_print(f"\nDefect: {def_type.split('@')[0]}")
        for tl_efermi, chargeset in tl_info.items():
            print(
                f"Transition Level ({max(chargeset):{'+' if max(chargeset) else ''}}/"
                f"{min(chargeset):{'+' if min(chargeset) else ''}}) at {tl_efermi:.3f}"
                f" eV above the VBM"
            )


# def pretty_formation_energy_plot(
#     defect_phase_diagram: DefectPhaseDiagram,
#     chempots: dict = None,
#     xlim: float = None,
#     ylim: float = None,
#     ax_fontsize: float = 1.3,
#     lg_fontsize: float = 1.0,
#     lg_position: tuple = None,
#     fermi_level: float = None,
#     title: str = None,
#     saved: bool = False,
# ):
#     """
#         Produces pretty Defect Formation Energy vs Fermi Energy plots for the defects in
#         defect_phase_diagram, for each chemical potential limit in chempot_limits.
#         Args:
#             defect_phase_diagram (DefectPhaseDiagram):
#                 DefectPhaseDiagram object for which to plot Formation Energy vs Fermi Energy
#             chempots:
#                 A dictionary of {Element: Energy} giving the chemical potential of each element.
#                 If None, chemical potential is set to 0 for each element.
#                 (default: None}
#             xlim (tuple):
#                 Tuple (min,max) to set the range of the x (Fermi Energy) axis
#                 (default: None)
#             ylim (tuple):
#                 Tuple (min,max) to set the range for the Formation Energy axis
#                 (default: None)
#             ax_fontsize (float):
#                 Float  multiplier to change axis label fontsize
#                 (default: 1.3)
#             lg_fontsize (float):
#                 Float  multiplier to change legend label fontsize
#                 (default: 1.0)
#             lg_position (tuple):
#                 Tuple (horizontal-position, vertical-position) giving the fractional position
#                 to place the legend.
#                 Example: (0.5,-0.75) will likely put it below the x-axis.
#                 (default: None)
#             fermi_level (float):
#                 Plot the specified Fermi Level position as a vertical line.
#                 (default: None)
#             title (str):
#                 Title of plot.
#                 (default: None)
#             saved (bool):
#                 Whether to save the plot as an image file.
#                 (default: False)
#         Returns:
#             A matplotlib plot object
#         """
#     plot = defect_phase_diagram.plot(
#         mu_elts=chempots,
#         xlim=xlim,
#         ylim=ylim,
#         ax_fontsize=ax_fontsize,
#         lg_fontsize=lg_fontsize,
#         lg_position=lg_position,
#         fermi_level=fermi_level,
#         title=title,
#         saved=saved,
#     )
#     for i in defect_phase_diagram.transition_level_map.values():
#         for trans_level, charges in i.items():
#             plot.vlines(trans_level, *plot.ylim(), colors="cyan", linestyles="dashdot")
#             plot.annotate(
#                 f"({max(charges)}/{min(charges)})", (trans_level + 0.02, plot.ylim()[1] - 0.2)
#             )
#
#     return plot


def formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempot_limits: dict = None,
    fermi_level: float = 0,
    hide_cols: list = None,
    show_key: bool = True,
    pd_facets: list = None,
):
    """
        Prints the formation energy tables for either a single chemical potential limit (i.e. phase
        diagram facet) or each facet in the chempot_limits dict, depending on which version you
        provide. Returns the results as either a pandas dataframe or list of dataframes.
        """
    if "facets" in chempot_limits:
        list_of_dfs = []
        if not pd_facets:
            pd_facets = chempot_limits["facets"].keys()  # Phase diagram facets to use for chemical
            # potentials, to tabulate formation energies
        for facet in pd_facets:
            bold_print("Facet: " + unicodeify(facet))
            df = single_formation_energy_table(
                defect_phase_diagram,
                chempots=chempot_limits["facets"][facet],
                fermi_level=fermi_level,
                hide_cols=hide_cols,
                show_key=show_key,
            )
            list_of_dfs.append(df)
            print("\n")

        return list_of_dfs

    else:  # If you only want to give {Elt: Energy} dict for chempot_limits
        df = single_formation_energy_table(
            defect_phase_diagram,
            chempots=chempot_limits,
            fermi_level=fermi_level,
            hide_cols=hide_cols,
            show_key=show_key,
        )
        return df


def single_formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempots: dict = None,
    fermi_level: float = 0,
    hide_cols: list = None,
    show_key: bool = True,
):
    """
    Prints the formation energy table for a single chemical potential limit (i.e. phase diagram
    facet), and returns the results as a pandas dataframe.
    """
    header = ["Defect", "Charge", "Defect Path"]
    table = []
    if hide_cols is None:
        hide_cols = []
    for defect_entry in defect_phase_diagram.entries:
        row = [defect_entry.name, defect_entry.charge, defect_entry.parameters["defect_path"]]
        if "Uncorrected_E" not in hide_cols:
            header += ["Uncorrected_E"]
            row += [f"{defect_entry.uncorrected_energy:.2f} eV"]
        if "Corrected_E" not in hide_cols:
            header += ["Corrected_E"]
            row += [
                f"{defect_entry.energy:.2f} eV"
            ]  # With 0 chemical potentials, at the calculation
            # fermi level
        header += ["Formation_E"]
        row += [
            f"{defect_entry.formation_energy(chemical_potentials=chempots, fermi_level=fermi_level):.2f} eV"
        ]

        table.append(row)
    table = sorted(table, key=itemgetter(0, 1))
    print(
        tabulate(table, headers=header, tablefmt="fancy_grid", stralign="left", numalign="left"),
        "\n",
    )

    if show_key:
        bold_print("Table Key:")
        print(
            """'Defect' -> Defect Type and Multiplicity
'Charge' -> Defect Charge State
'Uncorrected_E' -> Defect Energy from calculation, without corrections
'Corrected_E' -> Defect Energy from calculation (E_defect - E_host + corrections)
(chemical potentials set to 0 and the fermi level at average electrostatic potential in the
supercell)
'Formation_E' -> Final Defect Formation Energy, with the specified chemical potentials (
chempot_limits)(default: all 0) and the chosen fermi_level (default: 0)(i.e. at the VBM)
        """
        )

    sorted_df = pd.DataFrame(
        table,
        columns = ['Defect', 'Charge', 'defect_path', 'Uncorrected_E', 'Corrected_E', 'Formation_E']
    )
    sorted_df = sorted_df.sort_values('Formation_E')
    return sorted_df


def bold_print(string: str) -> None:
    """Does what it says on the tin. Prints the input string in bold."""
    print("\033[1m" + string + "\033[0m")


def save_to_pickle(python_object: Any, filename: str) -> None:
    with open(filename, "wb") as fp:
        pickle.dump(python_object, fp)


def load_from_pickle(filepath: str) -> None:
    with open(filepath, "rb") as fp:
        return pickle.load(fp)


def formation_energy_plot(
    defect_phase_diagram,
    chempot_limits=None,
    ax=None,
    fonts=None,
    xlim=None,
    ylim=None,
    ax_fontsize=1.0,
    lg_fontsize=1.0,
    lg_position=None,
    fermi_level=None,
    title: str = None,
    saved=False,
    colormap="Dark2",
    minus_symbol="-",
    frameon=False,
    chem_pot_table=True,
    pd_facets: list = None,
    auto_labels: bool = False,
    filename: str = None,
    emphasis=False,
):
    if chempot_limits and "facets" in chempot_limits:
        if not pd_facets:
            pd_facets = chempot_limits["facets"].keys()  # Phase diagram facets to use for chemical
            # potentials, to calculate and plot formation energies
        for facet in pd_facets:
            mu_elts = chempot_limits["facets"][facet]
            elt_refs = chempot_limits["facets_wrt_elt_refs"][facet]
            if not title:
                title = facet
            if not filename:
                filename = title + "_" + facet + ".pdf"

            return _aide_pmg_plot(
                defect_phase_diagram,
                mu_elts=mu_elts,
                elt_refs=elt_refs,
                ax=ax,
                fonts=fonts,
                xlim=xlim,
                ylim=ylim,
                ax_fontsize=ax_fontsize,
                lg_fontsize=lg_fontsize,
                lg_position=lg_position,
                fermi_level=fermi_level,
                title=title,
                saved=saved,
                colormap=colormap,
                minus_symbol=minus_symbol,
                frameon=frameon,
                chem_pot_table=chem_pot_table,
                auto_labels=auto_labels,
                filename=filename,
                emphasis=emphasis,
            )
    else:  # If you only want to give {Elt: Energy} dict for chempot_limits, or no chempot_limits
        return _aide_pmg_plot(
            defect_phase_diagram,
            mu_elts=chempot_limits,
            elt_refs=None,
            ax=ax,
            fonts=fonts,
            xlim=xlim,
            ylim=ylim,
            ax_fontsize=ax_fontsize,
            lg_fontsize=lg_fontsize,
            lg_position=lg_position,
            fermi_level=fermi_level,
            title=title,
            saved=saved,
            colormap=colormap,
            minus_symbol=minus_symbol,
            frameon=frameon,
            chem_pot_table=chem_pot_table,
            auto_labels=auto_labels,
            filename=filename,
            emphasis=emphasis,
        )


def _aide_pmg_plot(
    defect_phase_diagram,
    mu_elts=None,
    elt_refs=None,
    ax=None,
    fonts=None,
    xlim=None,
    ylim=None,
    ax_fontsize=1.0,
    lg_fontsize=1.0,
    lg_position=None,
    fermi_level=None,
    title=None,
    saved=False,
    colormap="Dark2",
    minus_symbol="-",
    frameon=False,
    chem_pot_table=True,
    auto_labels=False,
    filename=None,
    emphasis=False,
):
    """
    Produce defect Formation energy vs Fermi energy plot
    Args:
        mu_elts:
            a dictionary of {Element:value} giving the chemical
            potential of each element
        xlim:
            Tuple (min,max) giving the range of the x (fermi energy) axis. This may need to be
            set manually when including transition level labels, so that they dont' cross the axes.
        ylim:
            Tuple (min,max) giving the range for the formation energy axis. This may need to be
            set manually when including transition level labels, so that they dont' cross the axes.
        ax_fontsize:
            float  multiplier to change axis label fontsize
        lg_fontsize:
            float  multiplier to change legend label fontsize
        lg_position:
            Tuple (horizontal-position, vertical-position) giving the position
            to place the legend.
            Example: (0.5,-0.75) will likely put it below the x-axis.
    Returns:
        a matplotlib object
    """

    if xlim is None:
        xlim = (-0.4, defect_phase_diagram.band_gap + 0.4)
    xy = {}
    all_lines_xy = {} # For emphasis plots with faded grey E_form lines for all charge states
    lower_cap = -100.0
    upper_cap = 100.0
    y_range_vals = []  # for finding max/min values on y-axis based on x-limits

    for defnom, def_tl in defect_phase_diagram.transition_level_map.items():
        xy[defnom] = [[], []]
        if emphasis:
            all_lines_xy[defnom] = [[], []]
            for chg_ent in defect_phase_diagram.stable_entries[defnom]:
                for x_extrem in [lower_cap, upper_cap]:
                    all_lines_xy[defnom][0].append(x_extrem)
                    all_lines_xy[defnom][1].append(
                        chg_ent.formation_energy(chemical_potentials=mu_elts, fermi_level=x_extrem)
                    )
                # for x_window in xlim:
                #    y_range_vals.append(
                #        chg_ent.formation_energy(chemical_potentials=mu_elts, fermi_level=x_window)
                #    )

        if def_tl:
            org_x = list(def_tl.keys())  # list of transition levels
            org_x.sort()  # sorted with lowest first

            # establish lower x-bound
            first_charge = max(def_tl[org_x[0]])
            for chg_ent in defect_phase_diagram.stable_entries[defnom]:
                if chg_ent.charge == first_charge:
                    form_en = chg_ent.formation_energy(
                        chemical_potentials=mu_elts, fermi_level=lower_cap
                    )
                    fe_left = chg_ent.formation_energy(
                        chemical_potentials=mu_elts, fermi_level=xlim[0]
                    )
            xy[defnom][0].append(lower_cap)
            xy[defnom][1].append(form_en)
            y_range_vals.append(fe_left)
            # iterate over stable charge state transitions
            for fl in org_x:
                charge = max(def_tl[fl])
                for chg_ent in defect_phase_diagram.stable_entries[defnom]:
                    if chg_ent.charge == charge:
                        form_en = chg_ent.formation_energy(
                            chemical_potentials=mu_elts, fermi_level=fl
                        )
                xy[defnom][0].append(fl)
                xy[defnom][1].append(form_en)
                y_range_vals.append(form_en)
            # establish upper x-bound
            last_charge = min(def_tl[org_x[-1]])
            for chg_ent in defect_phase_diagram.stable_entries[defnom]:
                if chg_ent.charge == last_charge:
                    form_en = chg_ent.formation_energy(
                        chemical_potentials=mu_elts, fermi_level=upper_cap
                    )
                    fe_right = chg_ent.formation_energy(
                        chemical_potentials=mu_elts, fermi_level=xlim[1]
                    )
            xy[defnom][0].append(upper_cap)
            xy[defnom][1].append(form_en)
            y_range_vals.append(fe_right)
        else:
            # no transition - just one stable charge
            chg_ent = defect_phase_diagram.stable_entries[defnom][0]
            for x_extrem in [lower_cap, upper_cap]:
                xy[defnom][0].append(x_extrem)
                xy[defnom][1].append(
                    chg_ent.formation_energy(chemical_potentials=mu_elts, fermi_level=x_extrem)
                )
            for x_window in xlim:
                y_range_vals.append(
                    chg_ent.formation_energy(chemical_potentials=mu_elts, fermi_level=x_window)
                )

    cmap = cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(xy)))
    if colormap == "Dark2" and len(xy) >= 8:
        warnings.warn(
            f"""
The chosen colormap is Dark2, which only has 8 colours, yet you have {len(xy)} defect species (so
some defects will have the same line colour). Recommended to change/set colormap to 'tab10' or
'tab20' (10 and 20 colours each)."""
        )
    plt.figure(dpi=600, figsize=(2.6, 1.95))  # Gives a final figure width of c. 3.5
    # inches, the standard single column width for publication (which is what we're about)
    plt.clf()
    width = 9
    ax = pretty_axis(ax=ax, fonts=fonts)
    # plot formation energy lines
    for_legend = []
    for cnt, defnom in enumerate(xy.keys()):
        ax.plot(
            xy[defnom][0],
            xy[defnom][1],
            color=colors[cnt],
            markeredgecolor=colors[cnt],
            lw=1.2,
            markersize=3.5,
        )
        for_legend.append(defect_phase_diagram.stable_entries[defnom][0].copy())
    # Redo for loop so grey 'all_lines_xy' not included in legend
    for cnt, defnom in enumerate(xy.keys()):
        if emphasis:
            ax.plot(
                all_lines_xy[defnom][0],
                all_lines_xy[defnom][1],
                color=(0.8, 0.8, 0.8),
                markeredgecolor=colors[cnt],
                lw=1.2,
                markersize=3.5,
                alpha=0.5,
            )
    # plot transition levels
    for cnt, defnom in enumerate(xy.keys()):
        x_trans, y_trans = [], []
        tl_labels = []
        tl_label_type = []
        for x_val, chargeset in defect_phase_diagram.transition_level_map[defnom].items():
            x_trans.append(x_val)
            for chg_ent in defect_phase_diagram.stable_entries[defnom]:
                if chg_ent.charge == chargeset[0]:
                    form_en = chg_ent.formation_energy(
                        chemical_potentials=mu_elts, fermi_level=x_val
                    )
            y_trans.append(form_en)
            tl_labels.append(
                f"$\epsilon$({max(chargeset):{'+' if max(chargeset) else ''}}/"
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
                lw=1.2,
                markersize=3.5,
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
                        size=ax_fontsize * width * 0.9,
                        annotation_clip=True,
                    )  # only show label if coords in current axes

    # get latex-like legend titles
    legends_txt = []
    for dfct in for_legend:
        flds = dfct.name.split("_")
        if flds[0] == "Vac":
            base = "$\mathrm{V"
            sub_str = "_{" + flds[1] + "}}$"
        elif flds[0] == "Sub":
            flds = dfct.name.split("_")
            base = "$\mathrm{" + flds[1]
            sub_str = "_{" + flds[3] + "}}$"
        elif flds[0] == "Int":
            base = "$\mathrm{" + flds[1]
            sub_str = "_{i}}$"
        else:
            base = dfct.name
            sub_str = ""
        def_name = base + sub_str
        # add subscript labels for different configurations of same defect species
        labelled_def_name = def_name + r"$_{, 1}$"
        if def_name in legends_txt:
            def_name = labelled_def_name
        if def_name in legends_txt:
            i = 1
            while def_name in legends_txt:
                i += 1
                def_name = def_name[:-3] + f"{i}" + def_name[-2:]
            legends_txt.append(def_name)
        else:
            legends_txt.append(def_name)

    if not lg_position:
        ax.legend(
            legends_txt,
            fontsize=lg_fontsize * width,
            loc=2,
            bbox_to_anchor=(1, 1),
            frameon=frameon,
            prop=fonts,
        )
    else:
        ax.legend(
            legends_txt,
            fontsize=lg_fontsize * width,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=lg_position,
        )

    if ylim is None:
        window = max(y_range_vals) - min(y_range_vals)
        spacer = 0.1 * window
        ylim = (0, max(y_range_vals) + spacer)
        if auto_labels:  # need to manually set xlim or ylim if labels cross axes!!
            ylim = (0, max(y_range_vals) * 1.17) if spacer / ylim[1] < 0.145 else ylim
            # Increase y_limit to give space for transition level labels

    # Show colourful band edges
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
        extent=(defect_phase_diagram.band_gap, xlim[1], ylim[0], ylim[1]),
        vmin=0,
        vmax=3,
        interpolation="bicubic",
        rasterized=True,
        aspect="auto",
    )

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.plot([xlim[0], xlim[1]], [0, 0], "k-")  # black dashed line for E_formation = 0

    if fermi_level is not None:
        plt.axvline(
            x=fermi_level, linestyle="-.", color="k", linewidth=1
        )  # smaller dashed lines for gap edges
    ax.set_xlabel("Fermi Level (eV)", size=ax_fontsize * width)
    ax.set_ylabel("Formation Energy (eV)", size=ax_fontsize * width)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(_CustomScalarFormatter(minus_symbol=minus_symbol))
    ax.yaxis.set_major_formatter(_CustomScalarFormatter(minus_symbol=minus_symbol))
    if chem_pot_table:
        if elt_refs:
            _plot_chemical_potential_table(
                plt,
                elt_refs,
                "",
                fontsize=ax_fontsize * width,
                minus_symbol=minus_symbol,
                wrt_elt_refs=True,
            )
        elif mu_elts:
            _plot_chemical_potential_table(
                plt,
                mu_elts,
                "",
                fontsize=ax_fontsize * width,
                minus_symbol=minus_symbol,
                wrt_elt_refs=False,
            )

    if title and chem_pot_table:
        ax.set_title(latexify(title), size=1.2 * ax_fontsize * width, pad=28, fontdict={
            "fontweight": "bold"})
    elif title:
        ax.set_title(latexify(title), size=ax_fontsize * width, fontdict={"fontweight": "bold"})
    if saved or filename:
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=600)
        else:
            plt.savefig(str(title) + "_doped_plot.pdf", bbox_inches="tight", dpi=600)
    return ax


def _plot_chemical_potential_table(
    plt,
    elt_refs,
    chem_pot_label="",
    fontsize=9,
    loc="left",
    ax=None,
    minus_symbol="−",
    wrt_elt_refs=False,
):
    if ax is None:
        ax = plt.gca()
    chemical_potentials = elt_refs

    labels = [""] + [
        "$\mathregular{{\mu_{{{}}}}}$,".format(s) for s in sorted(chemical_potentials.keys())
    ]
    # add if else here, to use 'facets' if no wrt_elts, and don't say wrt elt_refs etc.
    labels[1] = "(" + labels[1]
    labels[-1] = labels[-1][:-1] + ")"
    labels = ["Chemical Potentials"] + labels + [" Units:"]
    text = [[chem_pot_label]]

    for el in sorted(chemical_potentials.keys()):
        text[0].append("{:.2f},".format(chemical_potentials[el]).replace("-", minus_symbol))

    text[0][1] = "(" + text[0][1]
    text[0][-1] = text[0][-1][:-1] + ")"
    if wrt_elt_refs:
        text[0] = ["(wrt Elemental refs)"] + text[0] + ["  [eV]"]
    else:
        text[0] = ["(from calculations)"] + text[0] + ["  [eV]"]
    widths = [0.1] + [0.9 / len(chemical_potentials)] * (len(chemical_potentials) + 2)
    tab = ax.table(cellText=text, colLabels=labels, colWidths=widths, loc="top", cellLoc=loc)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)

    tab.auto_set_column_width(list(range(len(widths))))
    tab.scale(1.0, 1.0)  # Default spacing is based on fontsize, just bump it up
    for cell in tab.get_celld().values():
        cell.set_linewidth(0)

    return tab


class _CustomScalarFormatter(ticker.ScalarFormatter):
    """Derived matplotlib tick formatter for arbitrary minus signs

    Args:
        minus_symbol (str): Symbol used in place of hyphen"""

    def __init__(
        self, useOffset=None, useMathText=None, useLocale=None, minus_symbol="\N{MINUS SIGN}"
    ):
        self.minus_symbol = minus_symbol
        super(_CustomScalarFormatter, self).__init__(
            useOffset=useOffset, useMathText=useMathText, useLocale=useLocale
        )

    def fix_minus(self, s):
        return s.replace("-", self.minus_symbol)


def pretty_axis(ax=None, fonts=None):

    ticklabelsize = 9
    ticksize = 8
    linewidth = 1.0

    if ax is None:
        ax = plt.gca()

    ax.tick_params(width=linewidth, size=ticksize)
    ax.tick_params(which="major", size=ticksize, width=linewidth, labelsize=ticklabelsize, pad=3)
    ax.tick_params(which="minor", size=ticksize / 2, width=linewidth)

    ax.set_title(ax.get_title(), size=9.5)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(linewidth)

    labelsize = int(9)

    ax.set_xlabel(ax.get_xlabel(), size=labelsize)
    ax.set_ylabel(ax.get_ylabel(), size=labelsize)

    fonts = default_fonts if fonts is None else fonts + default_fonts

    rc("font", **{"family": "sans-serif", "sans-serif": fonts})
    rc("text", usetex=False)
    rc("pdf", fonttype=42)
    # rc('mathtext', fontset='stixsans')

    return ax


def lany_zunger_corrected_defect_dict_from_freysoldt(defect_dict: dict):
    """Convert input parsed defect dictionary (presumably created using SingleDefectParser
     from doped.pycdt.utils.parse_calculations) with Freysoldt charge corrections to
     the same parsed defect dictionary but with the Lany-Zunger charge correction (same potential
     alignment plus 0.65 * Makov-Payne image charge correction (same image charge correction as
     Freysoldt scheme)).
     Args:
         parsed_defect_dict (dict):
             Dictionary of parsed defect calculations (presumably created using SingleDefectParser
             from doped.pycdt.utils.parse_calculations) (see example notebook)
             Must have 'freysoldt_meta' in defect.parameters for each charged defect (from
             SingleDefectParser.freysoldt_loader())
     Returns:
         Parsed defect dictionary with Lany-Zunger charge corrections.
     """
    random_defect_entry = list(defect_dict.values())[0]  # Just need any DefectEntry from
    # defect_dict to get the lattice and dielectric matrix
    lattice = random_defect_entry.bulk_structure.lattice.matrix
    dielectric = random_defect_entry.parameters["dielectric"]
    lz_image_charge_corrections = aide_murphy_correction.get_image_charge_correction(
        lattice, dielectric
    )
    lz_corrected_defect_dict = copy.deepcopy(defect_dict)
    for defect_name, defect_entry in lz_corrected_defect_dict.items():
        if defect_entry.charge != 0:
            potalign = defect_entry.parameters["freysoldt_meta"][
                "freysoldt_potential_alignment_correction"
            ]
            makove_payne_pc_correction = lz_image_charge_corrections[abs(defect_entry.charge)]
            defect_entry.parameters.update(
                {
                    "Lany-Zunger_Corrections": {
                        "(Freysoldt)_Potential_Alignment_Correction": potalign,
                        "Makov-Payne_Image_Charge_Correction": makove_payne_pc_correction,
                        "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65
                        * makove_payne_pc_correction,
                        "Total_Lany-Zunger_Correction": potalign
                        + 0.65 * makove_payne_pc_correction,
                    }
                }
            )
            defect_entry.corrections["charge_correction"] = defect_entry.parameters[
                "Lany-Zunger_Corrections"
            ]["Total_Lany-Zunger_Correction"]

        lz_corrected_defect_dict.update({defect_name: defect_entry})
    return lz_corrected_defect_dict


def lany_zunger_corrected_defect_dict_from_kumagai(defect_dict: dict):
    """Convert input parsed defect dictionary (presumably created using SingleDefectParser
     from doped.pycdt.utils.parse_calculations) with Kumagai charge corrections to
     the same parsed defect dictionary but with the 'Lany-Zunger' charge correction (same potential
     alignment plus 0.65 * image charge correction.
     Args:
         parsed_defect_dict (dict):
             Dictionary of parsed defect calculations (presumably created using SingleDefectParser
             from doped.pycdt.utils.parse_calculations) (see example notebook)
             Must have 'kumagai_meta' in defect.parameters for each charged defect (from
             SingleDefectParser.kumagai_loader())
     Returns:
         Parsed defect dictionary with Lany-Zunger charge corrections.
     """
    random_defect_entry = list(defect_dict.values())[0]  # Just need any DefectEntry from
    # defect_dict to get the lattice and dielectric matrix
    lattice = random_defect_entry.bulk_structure.lattice.matrix
    dielectric = random_defect_entry.parameters["dielectric"]
    lz_image_charge_corrections = aide_murphy_correction.get_image_charge_correction(
        lattice, dielectric
    )
    lz_corrected_defect_dict = copy.deepcopy(defect_dict)
    for defect_name, defect_entry in lz_corrected_defect_dict.items():
        if defect_entry.charge != 0:
            potalign = defect_entry.parameters["kumagai_meta"][
                "kumagai_potential_alignment_correction"
            ]
            makove_payne_pc_correction = lz_image_charge_corrections[abs(defect_entry.charge)]
            defect_entry.parameters.update(
                {
                    "Lany-Zunger_Corrections": {
                        "(Kumagai)_Potential_Alignment_Correction": potalign,
                        "Makov-Payne_Image_Charge_Correction": makove_payne_pc_correction,
                        "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65
                        * makove_payne_pc_correction,
                        "Total_Lany-Zunger_Correction": potalign
                        + 0.65 * makove_payne_pc_correction,
                    }
                }
            )
            defect_entry.corrections["charge_correction"] = defect_entry.parameters[
                "Lany-Zunger_Corrections"
            ]["Total_Lany-Zunger_Correction"]

        lz_corrected_defect_dict.update({defect_name: defect_entry})
    return lz_corrected_defect_dict


def all_lines_formation_energy_plot(
    defect_phase_diagram,
    chempot_limits=None,
    ax=None,
    fonts=None,
    xlim=None,
    ylim=None,
    ax_fontsize=1.0,
    lg_fontsize=1.0,
    lg_position=None,
    fermi_level=None,
    title=None,
    saved=False,
    colormap="Dark2",
    minus_symbol="-",
    frameon=False,
    chem_pot_table=True,
    pd_facets: list = None,
    auto_labels: bool = False,
    filename: str = None,
):
    if chempot_limits and "facets" in chempot_limits:
        if not pd_facets:
            pd_facets = chempot_limits["facets"].keys()  # Phase diagram facets to use for chemical
            # potentials, to calculate and plot formation energies
        for facet in pd_facets:
            mu_elts = chempot_limits["facets"][facet]
            elt_refs = chempot_limits["facets_wrt_elt_refs"][facet]
            plot_filename = filename
            if title:
                plot_title = title
                if not filename:
                    plot_filename = plot_title + "_" + facet + ".pdf"
            else:
                plot_title = facet

            _all_lines_aide_pmg_plot(
                defect_phase_diagram,
                mu_elts=mu_elts,
                elt_refs=elt_refs,
                ax=ax,
                fonts=fonts,
                xlim=xlim,
                ylim=ylim,
                ax_fontsize=ax_fontsize,
                lg_fontsize=lg_fontsize,
                lg_position=lg_position,
                fermi_level=fermi_level,
                title=plot_title,
                saved=saved,
                colormap=colormap,
                minus_symbol=minus_symbol,
                frameon=frameon,
                chem_pot_table=chem_pot_table,
                auto_labels=auto_labels,
                filename=plot_filename,
            )
    else:  # If you only want to give {Elt: Energy} dict for chempot_limits, or no chempot_limits
        _all_lines_aide_pmg_plot(
            defect_phase_diagram,
            mu_elts=chempot_limits,
            elt_refs=None,
            ax=ax,
            fonts=fonts,
            xlim=xlim,
            ylim=ylim,
            ax_fontsize=ax_fontsize,
            lg_fontsize=lg_fontsize,
            lg_position=lg_position,
            fermi_level=fermi_level,
            title=title,
            saved=saved,
            colormap=colormap,
            minus_symbol=minus_symbol,
            frameon=frameon,
            chem_pot_table=chem_pot_table,
            auto_labels=auto_labels,
            filename=filename,
        )


def _all_lines_aide_pmg_plot(
    defect_phase_diagram,
    mu_elts=None,
    elt_refs=None,
    ax=None,
    fonts=None,
    xlim=None,
    ylim=None,
    ax_fontsize=1.0,
    lg_fontsize=1.0,
    lg_position=None,
    fermi_level=None,
    title=None,
    saved=False,
    colormap="Dark2",
    minus_symbol="-",
    frameon=False,
    chem_pot_table=True,
    auto_labels=False,
    filename=None,
):
    """
    Produce defect Formation energy vs Fermi energy plot
    Args:
        mu_elts:
            a dictionnary of {Element:value} giving the chemical
            potential of each element
        xlim:
            Tuple (min,max) giving the range of the x (fermi energy) axis. This may need to be
            set manually when including transition level labels, so that they dont' cross the axes.
        ylim:
            Tuple (min,max) giving the range for the formation energy axis. This may need to be
            set manually when including transition level labels, so that they dont' cross the axes.
        ax_fontsize:
            float  multiplier to change axis label fontsize
        lg_fontsize:
            float  multiplier to change legend label fontsize
        lg_position:
            Tuple (horizontal-position, vertical-position) giving the position
            to place the legend.
            Example: (0.5,-0.75) will likely put it below the x-axis.
    Returns:
        a matplotlib object
    """

    if xlim is None:
        xlim = (-0.4, defect_phase_diagram.band_gap + 0.4)
    xy = {}
    lower_cap = -100.0
    upper_cap = 100.0
    y_range_vals = []  # for finding max/min values on y-axis based on x-limits

    legends_txt = []
    for chg_ent in defect_phase_diagram.entries:
        defnom = chg_ent.name + f"_{chg_ent.charge}"
        flds = defnom.split("_")
        if flds[0] == "Vac":
            base = "$\mathrm{V"
            sub_str = "_{" + flds[1] + "}}$"
        elif flds[0] == "Sub":
            flds = defnom.split("_")
            base = "$\mathrm{" + flds[1]
            sub_str = "_{" + flds[3] + "}}$"
        elif flds[0] == "Int":
            base = "$\mathrm{" + flds[1]
            sub_str = "_{i}}$"
        else:
            base = defnom
            sub_str = ""
        def_name = (
            base + sub_str + r"$^{" + f"{int(flds[-1]):{'+' if int(flds[-1]) > 0 else ''}}" + r"}$"
        )

        # add subscript labels for different configurations of same defect species
        legends_txt = [
            def_name + r"$_{, 1}$" if def_name == orig_def_name else orig_def_name
            for orig_def_name in legends_txt
        ]
        labelled_def_name = def_name + r"$_{, 1}$"
        if labelled_def_name in legends_txt:
            i = 1
            while labelled_def_name in legends_txt:
                i += 1
                labelled_def_name = def_name + r"$_{, " + f"{i}" + r"}$"
            def_name = labelled_def_name
        legends_txt.append(def_name)

        xy[def_name] = [[], []]
        for x_extrem in [lower_cap, upper_cap]:
            xy[def_name][0].append(x_extrem)
            xy[def_name][1].append(
                chg_ent.formation_energy(chemical_potentials=mu_elts, fermi_level=x_extrem)
            )
        for x_window in xlim:
            y_range_vals.append(
                chg_ent.formation_energy(chemical_potentials=mu_elts, fermi_level=x_window)
            )

    cmap = cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(xy)))
    if colormap == "Dark2" and len(xy) >= 8:
        warnings.warn(
            f"""
The chosen colormap is Dark2, which only has 8 colours, yet you have {len(xy)} defect species (so
some defects will have the same line colour). Recommended to change/set colormap to 'tab10' or
'tab20' (10 and 20 colours each)."""
        )
    plt.figure(dpi=600, figsize=(2.6, 1.95))  # Gives a final figure width of c. 3.5
    # inches, the standard single column width for publication (which is what we're about)
    plt.clf()
    width = 9
    ax = pretty_axis(ax=ax, fonts=fonts)
    # plot formation energy lines

    for cnt, def_name in enumerate(xy.keys()):
        ax.plot(
            xy[def_name][0],
            xy[def_name][1],
            color=colors[cnt],
            markeredgecolor=colors[cnt],
            lw=1.2,
            markersize=3.5,
        )

    if not lg_position:
        ax.legend(
            legends_txt,
            fontsize=lg_fontsize * width,
            loc=2,
            bbox_to_anchor=(1, 1),
            frameon=frameon,
            prop=fonts,
        )
    else:
        ax.legend(
            legends_txt,
            fontsize=lg_fontsize * width,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=lg_position,
        )

    if ylim is None:
        window = max(y_range_vals) - min(y_range_vals)
        spacer = 0.1 * window
        ylim = (0, max(y_range_vals) + spacer)
        if auto_labels:  # need to manually set xlim or ylim if labels cross axes!!
            ylim = (0, max(y_range_vals) * 1.17) if spacer / ylim[1] < 0.145 else ylim
            # Increase y_limit to give space for transition level labels

    # Show colourful band edges
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
        extent=(defect_phase_diagram.band_gap, xlim[1], ylim[0], ylim[1]),
        vmin=0,
        vmax=3,
        interpolation="bicubic",
        rasterized=True,
        aspect="auto",
    )

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.plot([xlim[0], xlim[1]], [0, 0], "k-")  # black dashed line for E_formation = 0

    if fermi_level is not None:
        plt.axvline(
            x=fermi_level, linestyle="-.", color="k", linewidth=1
        )  # smaller dashed lines for gap edges
    ax.set_xlabel("Fermi Level (eV)", size=ax_fontsize * width)
    ax.set_ylabel("Formation Energy (eV)", size=ax_fontsize * width)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(_CustomScalarFormatter(minus_symbol=minus_symbol))
    ax.yaxis.set_major_formatter(_CustomScalarFormatter(minus_symbol=minus_symbol))
    if chem_pot_table:
        if elt_refs:
            _plot_chemical_potential_table(
                plt,
                elt_refs,
                "",
                fontsize=ax_fontsize * width,
                minus_symbol=minus_symbol,
                wrt_elt_refs=True,
            )
        elif mu_elts:
            _plot_chemical_potential_table(
                plt,
                mu_elts,
                "",
                fontsize=ax_fontsize * width,
                minus_symbol=minus_symbol,
                wrt_elt_refs=False,
            )

    if title and chem_pot_table:
        ax.set_title(latexify(title), size=1.2 * ax_fontsize * width, pad=28, fontdict={
            "fontweight":
                                                                                   "bold"})
    elif title:
        ax.set_title(latexify(title), size=ax_fontsize * width, fontdict={"fontweight": "bold"})
    if saved or filename:
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=600)
        else:
            plt.savefig(str(title) + "_doped_plot.pdf", bbox_inches="tight", dpi=600)
    else:
        return ax
