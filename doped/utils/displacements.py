"""
Code to analyse site displacements around a defect.
"""
import os
import warnings
from copy import deepcopy
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.util.coord import pbc_diff

from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    _get_defect_supercell_bulk_site_coords,
    _get_defect_supercell_site,
)

try:
    import plotly.express as px
    from plotly.graph_objects import Scatter
    from plotly.subplots import make_subplots

    plotly_installed = True
except ImportError:
    plotly_installed = False


def _calc_site_displacements(
    defect_entry,
    vector_to_project_on: Optional[list] = None,
    relative_to_defect: Optional[bool] = False,
) -> dict:
    """
    Calculates the site displacements in the defect supercell, relative to the
    bulk supercell. The signed displacements are stored in the
    calculation_metadata of the DefectEntry object under the
    "site_displacements" key.

    Args:
        defect_entry (DefectEntry): DefectEntry object
        vector_to_project_on (list): Direction to project the site
            displacements along (e.g. [0, 0, 1]). Defaults to None.
        relative_to_defect (bool): Whether to calculate the signed displacements
            along the line from the defect site to that atom. Negative values
            indicate the atom moves towards the defect (compressive strain),
            positive values indicate the atom moves away from the defect.
            Defaults to False. If True, the relative displacements are stored in
            the `Displacement wrt defect` key of the returned dictionary.

    Returns:
        Dictionary with site displacements (compared to pristine supercell).
    """
    from doped.utils.parsing import get_site_mapping_indices

    def _get_bulk_struct_with_defect(defect_entry) -> tuple:
        """
        Returns structures for bulk and defect supercells with the same number
        of sites and species, to be used for site matching. If Vacancy, adds
        (unrelaxed) site to defect structure. If Interstitial, adds relaxed
        site to bulk structure. If Substitution, replaces (unrelaxed) defect
        site in bulk structure.

        Returns tuple of (bulk_sc_with_defect, defect_sc_with_defect).
        """
        # TODO: Code from `check_atom_mapping_far_from_defect` might be more efficient and robust for this,
        #  should check.
        defect_type = defect_entry.defect.defect_type.name
        bulk_sc_with_defect = _get_bulk_supercell(defect_entry).copy()
        # Check position of relaxed defect has been parsed (it's an optional arg)
        sc_defect_frac_coords = _get_defect_supercell_bulk_site_coords(defect_entry)
        if sc_defect_frac_coords is None:
            raise ValueError(
                "The relaxed defect position (`DefectEntry.sc_defect_frac_coords`) has not been parsed. "
                "Please use `DefectsParser`/`DefectParser` to parse relaxed defect positions before "
                "calculating site displacements."
            )

        defect_sc_with_defect = _get_defect_supercell(defect_entry).copy()
        if defect_type == "Vacancy":
            # Add Vacancy atom to defect structure
            defect_sc_with_defect.append(
                defect_entry.defect.site.specie,
                defect_entry.defect.site.frac_coords,  # _unrelaxed_ defect site
                coords_are_cartesian=False,
            )
            defect_site_index = len(defect_sc_with_defect) - 1
        elif defect_type == "Interstitial":
            # If Interstitial, add interstitial site to bulk structure
            bulk_sc_with_defect.append(
                defect_entry.defect.site.specie,
                defect_entry.defect.site.frac_coords,  # _relaxed_ defect site for interstitials
                coords_are_cartesian=False,
            )
            # Ensure last site of defect structure is defect site. Needed to then calculate site
            # distances to defect
            # Get index of defect site in defect supercell
            if not np.allclose(
                defect_sc_with_defect[-1].frac_coords,
                sc_defect_frac_coords,  # _relaxed_ defect site
            ):
                # Get index of defect site in defect structure
                defect_site_index = defect_sc_with_defect.index(_get_defect_supercell_site(defect_entry))
                # Swap defect site with last site
                # defect_site = defect_sc_with_defect.pop(defect_site_index)
                # defect_sc_with_defect.append(
                #     defect_site.specie,
                #     defect_site.frac_coords,
                #     coords_are_cartesian=False,
                # )
            else:
                defect_site_index = len(defect_sc_with_defect) - 1
        elif defect_type == "Substitution":
            # If Substitution, replace site in bulk supercell
            bulk_sc_with_defect.replace(
                defect_entry.defect.defect_site_index,
                defect_entry.defect.site.specie,
                defect_entry.defect.site.frac_coords,  # _unrelaxed_ defect site
                coords_are_cartesian=False,
            )
            # Move defect site to last position of defect supercell
            # Get index of defect site in defect supercell
            defect_site_index = defect_sc_with_defect.index(
                _get_defect_supercell_site(defect_entry)  # _relaxed_ defect site
            )
            # defect_site = defect_sc_with_defect.pop(defect_site_index)
            # defect_sc_with_defect.append(
            #     defect_site.specie,
            #     defect_site.frac_coords,
            #     coords_are_cartesian=False,
            # )
        else:
            raise ValueError(f"Defect type {defect_type} not supported")
        return bulk_sc_with_defect, defect_sc_with_defect, defect_site_index

    bulk_sc, defect_sc_with_site, defect_site_index = _get_bulk_struct_with_defect(defect_entry)
    # Map sites in defect supercell to bulk supercell
    mappings = get_site_mapping_indices(defect_sc_with_site, bulk_sc)
    mappings_dict = {i[1]: i[2] for i in mappings}  # {defect_sc_index: bulk_sc_index}
    # Loop over sites in defect sc
    disp_dict = {  # mapping defect site index (in defect sc) to displacement
        "Index (defect)": [],
        "Species": [],
        "Species_with_index": [],
        "Abs. displacement": [],
        "Distance to defect": [],
        "Displacement wrt defect": [],
        "Displacement projected along vector": [],
    }  # type: dict
    # if relative_to_defect:
    #     disp_dict["Displacement wrt defect"] = []
    # if vector_to_project_on:
    #     disp_dict["Displacement projected along vector"] = []
    for i, site in enumerate(defect_sc_with_site):
        # print(i, site.specie, site.frac_coords)
        bulk_sc_index = mappings_dict[i]  # Map to bulk sc
        bulk_site = bulk_sc[bulk_sc_index]  # Get site in bulk sc
        # Calculate displacement (need to account for pbc!)
        frac_disp = pbc_diff(site.frac_coords, bulk_site.frac_coords)  # in fractional coords
        disp = bulk_sc.lattice.get_cartesian_coords(frac_disp)  # in Angstroms
        # Distance to defect site (last site in defect sc)
        distance = defect_sc_with_site.get_distance(i, defect_site_index)  # len(defect_sc_with_site) - 1)
        # print(i, displacement, np.linalg.norm(abs_disp), "Distance:", distance)
        disp_dict["Index (defect)"].append(i)
        disp_dict["Abs. displacement"].append(disp)
        disp_dict["Distance to defect"].append(distance)
        disp_dict["Species_with_index"].append(f"{site.specie.name}({i})")
        disp_dict["Species"].append(site.specie.name)
        if relative_to_defect:
            # Find vector from defect to site
            vector_to_project_on = np.array(
                defect_sc_with_site[defect_site_index].frac_coords - site.frac_coords
            )
            norm = np.linalg.norm(vector_to_project_on)
            if norm == 0:  # If defect site and site are the same
                disp_dict["Displacement wrt defect"].append(0)
            else:
                proj = np.dot(disp, vector_to_project_on / norm)
                disp_dict["Displacement wrt defect"].append(proj)
        if vector_to_project_on:
            # Normalize vector to project on
            norm = np.linalg.norm(vector_to_project_on)
            if norm == 0:
                raise ValueError(
                    "Norm of vector to project on is zero! Choose a non-zero vector to project on."
                )
            proj = np.dot(disp, vector_to_project_on / norm)
            disp_dict["Displacement projected along vector"].append(proj)

    # sort each list in disp dict by index of species in bulk element list, then by distance to defect:
    element_list = [
        el.symbol for el in defect_entry.defect.structure.composition.elements
    ]  # host elements
    element_list += sorted(
        [  # extrinsic elements, sorted alphabetically for deterministic ordering in output:
            el.symbol
            for el in defect_entry.defect.defect_structure.composition.elements
            if el.symbol not in element_list
        ]
    )

    # Combine the lists into a list of tuples, then sort, then unpack:
    combined = list(zip(*disp_dict.values()))
    combined.sort(
        key=lambda x: (element_list.index(x[1]), x[4], x[0])
    )  # Sort by species, then distance, then index
    (
        disp_dict["Index (defect)"],
        disp_dict["Species"],
        disp_dict["Species_with_index"],
        disp_dict["Abs. displacement"],
        disp_dict["Distance to defect"],
        disp_dict["Displacement wrt defect"],
        disp_dict["Displacement projected along vector"],
    ) = zip(*combined)

    # Store in DefectEntry.calculation_metadata
    # For vacancies, before storing displacements data, remove the last site
    # (defect site) as not present in input defect supercell
    # But leave it in disp_dict as clearer to include in the displacement plot?
    disp_list = deepcopy(disp_dict["Abs. displacement"])
    distance_list = deepcopy(disp_dict["Distance to defect"])
    if defect_entry.defect.defect_type.name == "Vacancy":
        disp_list.pop(defect_site_index)
        distance_list.pop(defect_site_index)
    # Store in DefectEntry.calculation_metadata
    defect_entry.calculation_metadata["site_displacements"] = {
        "displacements": disp_list,  # Ordered by site index in defect supercell
        "distances": distance_list,
    }
    return disp_dict


def _plot_site_displacements(
    defect_entry,
    separated_by_direction: Optional[bool] = False,
    relative_to_defect: Optional[bool] = False,
    vector_to_project_on: Optional[list] = None,
    use_plotly: Optional[bool] = True,
    style_file: Optional[str] = "",
):
    """
    Plots site displacements around a defect.

    Args:
        defect_entry: DefectEntry object
        separated_by_direction: Whether to plot site displacements separated by
            direction (x, y, z). Default is False.
        relative_to_defect (bool): Whether to plot the signed displacements
            along the line from the defect site to that atom. Negative values
            indicate the atom moves towards the defect (compressive strain),
            positive values indicate the atom moves away from the defect
            (tensile strain).
        vector_to_project_on: Direction to project the site displacements along
            (e.g. [0, 0, 1]). Defaults to None (e.g. the displacements are calculated
            in the cartesian basis x, y, z).
        use_plotly: Whether to use Plotly for plotting. Default is True.
        style_file: Path to matplotlib style file. Default is "", which will use
            the doped default style.

    Returns:
        Plotly or matplotlib figure.
    """

    def _mpl_plot_total_disp(
        disp_type_key,
        ylabel,
        disp_dict,
        color_dict,
        styled_fig_size,
        styled_font_size,
    ):
        """
        Function to plot absolute/total displacement.

        Depending on the disp_type_key specified, will plot either the
        normalised displacement (disp_type_key="Abs. displacement"), the
        displacement wrt the defect (disp_type_key="Displacement wrt defect"),
        or the displacmeent projected along a specified direction (
        disp_type_key="Displacement projected along vector").
        """
        fig, ax = plt.subplots(figsize=(styled_fig_size[0], styled_fig_size[1]))
        if disp_type_key == "Abs. displacement":
            y_data = [np.linalg.norm(i) for i in disp_dict["Abs. displacement"]]
        else:
            y_data = disp_dict[disp_type_key]
        ax.scatter(
            disp_dict["Distance to defect"],
            y_data,
            c=[color_dict[i] for i in disp_dict["Species"]],
            alpha=0.4,
            edgecolor="none",
        )
        ax.set_xlabel("Distance to defect ($\\AA$)", fontsize=styled_font_size)
        ax.set_ylabel("Displacement towards defect ($\\AA$)", fontsize=styled_font_size)
        # Add legend with species manually
        patches = [mpl.patches.Patch(color=color_dict[i], label=i) for i in unique_species]
        ax.legend(handles=patches)
        return fig

    def _plotly_plot_total_disp(
        disp_type_key,
        ylabel,
        disp_dict,
    ):
        if disp_type_key == "Abs. displacement":
            y_data = [np.linalg.norm(i) for i in disp_dict["Abs. displacement"]]
        else:
            y_data = disp_dict[disp_type_key]
        fig = px.scatter(
            x=disp_dict["Distance to defect"],
            y=y_data,
            hover_data={
                "Distance to defect": disp_dict["Distance to defect"],
                "Absolute displacement": y_data,
                "Species_with_index": disp_dict["Species_with_index"],
            },
            color=disp_dict["Species"],
            # trendline="ols"
        )
        # Round x and y in hover data
        fig.update_traces(
            hovertemplate=hovertemplate.replace("{x", "{customdata[0]")
            .replace("{y", "{customdata[1]")
            .replace("{z", "{customdata[2]")
        )
        # Add axis labels
        fig.update_layout(xaxis_title="Distance to defect (\u212B)", yaxis_title=f"{ylabel} (\u212B)")
        return fig

    disp_dict = _calc_site_displacements(
        defect_entry=defect_entry,
    )
    if use_plotly and not plotly_installed:
        warnings.warn("Plotly not installed, using matplotlib instead")
        use_plotly = False
    if use_plotly:
        hovertemplate = "Distance to defect: %{x:.2f}<br>Absolute displacement: %{y:.2f}<br>Species: %{z}"
        if relative_to_defect:
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement wrt defect",
                ylabel="Displacement towards defect",  # Angstrom symbol added in function
                disp_dict=disp_dict,
            )
        elif vector_to_project_on:
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement projected along vector",
                ylabel=f"Displacement along vector {tuple(vector_to_project_on)}",
                disp_dict=disp_dict,
            )
        elif not separated_by_direction:  # total displacement
            fig = _plotly_plot_total_disp(
                disp_type_key="Abs. displacement",
                ylabel="Absolute displacement",
                disp_dict=disp_dict,
            )
        else:  # separated by direction
            fig = make_subplots(
                rows=1, cols=3, subplot_titles=("x", "y", "z"), shared_xaxes=True, shared_yaxes=True
            )
            unique_species = list(set(disp_dict["Species"]))
            color_dict = dict(zip(unique_species, px.colors.qualitative.Plotly[: len(unique_species)]))
            for dir_index, _direction in enumerate(["x", "y", "z"]):
                fig.add_trace(
                    Scatter(
                        x=disp_dict["Distance to defect"],
                        y=[abs(i[dir_index]) for i in disp_dict["Abs. displacement"]],
                        hovertemplate=hovertemplate.replace("{z", "{text"),
                        text=disp_dict["Species_with_index"],
                        marker={"color": [color_dict[i] for i in disp_dict["Species"]]},
                        # Only scatter plot, no line
                        mode="markers",
                        showlegend=False,
                    ),
                    row=1,
                    col=dir_index + 1,
                )
            # Add legend for color used for each species
            for specie, color in color_dict.items():
                fig.add_trace(
                    Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker={"color": color},
                        showlegend=True,
                        legendgroup="1",
                        name=specie,
                    ),
                    row=1,
                    col=1,
                )
            # Add axis labels
            fig.update_layout(
                xaxis_title="Distance to defect (\u212B)", yaxis_title="Absolute displacement (\u212B)"
            )
    else:
        element_list = [
            el.symbol for el in defect_entry.defect.structure.composition.elements
        ]  # host elements
        element_list += sorted(
            [  # extrinsic elements, sorted alphabetically for deterministic ordering in output:
                el.symbol
                for el in defect_entry.defect.defect_structure.composition.elements
                if el.symbol not in element_list
            ]
        )

        style_file = style_file or f"{os.path.dirname(__file__)}/displacement.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        with plt.style.context(style_file):
            # Color by species
            unique_species = list(set(disp_dict["Species"]))
            unique_species.sort(key=lambda x: element_list.index(x))
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] or list(
                dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS).keys()
            )
            color_dict = {i: colors[index] for index, i in enumerate(unique_species)}
            styled_fig_size = plt.rcParams["figure.figsize"]
            # Gives a final figure width matching styled_fig_size,
            # with dimensions matching the doped default
            styled_font_size = plt.rcParams["font.size"]
            if relative_to_defect:
                return _mpl_plot_total_disp(
                    disp_type_key="Displacement wrt defect",
                    ylabel="Displacement towards defect ($\\AA$)",
                    disp_dict=disp_dict,
                    color_dict=color_dict,
                    styled_fig_size=styled_fig_size,
                    styled_font_size=styled_font_size,
                )
            if vector_to_project_on:
                return _mpl_plot_total_disp(
                    disp_type_key="Displacement projected along vector",
                    ylabel=f"Displacement along vector {tuple(vector_to_project_on)} ($\\AA$)",
                    disp_dict=disp_dict,
                    color_dict=color_dict,
                    styled_fig_size=styled_fig_size,
                    styled_font_size=styled_font_size,
                )
            if not separated_by_direction:
                return _mpl_plot_total_disp(
                    disp_type_key="Abs. displacement",
                    ylabel="Absolute displacement ($\\AA$)",
                    disp_dict=disp_dict,
                    color_dict=color_dict,
                    styled_fig_size=styled_fig_size,
                    styled_font_size=styled_font_size,
                )
            # Else, separated by direction
            fig, ax = plt.subplots(
                1,
                3,
                figsize=(2.0 * styled_fig_size[0], 0.6 * styled_fig_size[1]),  # (13, 4),
                sharey=True,
                sharex=True,
            )
            for index, i in enumerate(["x", "y", "z"]):
                ax[index].scatter(
                    disp_dict["Distance to defect"],
                    [abs(j[index]) for j in disp_dict["Abs. displacement"]],
                    c=[color_dict[i] for i in disp_dict["Species"]],
                    alpha=0.4,
                    edgecolor="none",
                )
                # Title with direction
                ax[index].set_title(f"{i}")
            ax[0].set_ylabel("Site displacements ($\\AA$)", fontsize=styled_font_size)
            ax[1].set_xlabel("Distance to defect ($\\AA$)", fontsize=styled_font_size)
            # Add legend with species manually
            patches = [mpl.patches.Patch(color=color_dict[i], label=i) for i in unique_species]
            ax[0].legend(handles=patches)
            # Set separation between subplots
            fig.subplots_adjust(wspace=0.07)
    return fig
