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
import pandas as pd
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry
from doped.generation import _get_element_list
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    _get_defect_supercell_bulk_site_coords,
    _get_defect_supercell_site,
    get_site_mapping_indices,
)
from doped.utils.symmetry import _round_floats

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_installed = True
except ImportError:
    plotly_installed = False


def calc_site_displacements(
    defect_entry: DefectEntry,
    relative_to_defect: bool = False,
    vector_to_project_on: Optional[list] = None,
) -> pd.DataFrame:
    """
    Calculates the site displacements in the defect supercell, relative to the
    bulk supercell, and returns a ``DataFrame`` of site displacement info.

    The signed displacements are stored in the calculation_metadata of the
    ``DefectEntry`` object under the ``"site_displacements"`` key.

    Args:
        defect_entry (DefectEntry):
            DefectEntry object
        relative_to_defect (bool):
            Whether to calculate the signed displacements along the line
            from the defect site to that atom. Negative values indicate
            the atom moves towards the defect (compressive strain),
            positive values indicate the atom moves away from the defect.
            Defaults to False. If True, the relative displacements are stored in
            the `Displacement wrt defect` key of the returned dictionary.
        vector_to_project_on (list):
            Direction to project the site displacements along
            (e.g. [0, 0, 1]). Defaults to None.

    Returns:
        ``pandas`` ``DataFrame`` with site displacements (compared to pristine supercell),
        and other displacement-related information.
    """
    bulk_sc, defect_sc_with_site, defect_site_index = _get_bulk_struct_with_defect(defect_entry)

    # Map sites in defect supercell to bulk supercell:
    mappings = get_site_mapping_indices(defect_sc_with_site, bulk_sc)
    mappings_dict = {i[1]: i[2] for i in mappings}  # {defect_sc_index: bulk_sc_index}

    disp_dict: dict[str, list] = {  # mapping defect site index (in defect sc) to displacement
        "Species": [],
        "Distance to defect": [],
        "Displacement": [],
        "Displacement vector": [],
        "Index (defect supercell)": [],
    }
    if relative_to_defect:
        disp_dict["Vector to site from defect"] = []
        disp_dict["Displacement wrt defect"] = []
    if vector_to_project_on is not None:
        disp_dict["Displacement projected along vector"] = []
        disp_dict["Displacement perpendicular to vector"] = []

    for i, site in enumerate(defect_sc_with_site):  # Loop over sites in defect sc
        bulk_sc_index = mappings_dict[i]  # Map to bulk sc
        bulk_site = bulk_sc[bulk_sc_index]  # Get site in bulk sc
        # Calculate displacement (need to account for pbc!)
        # First final point, then initial point
        disp = pbc_shortest_vectors(bulk_sc.lattice, bulk_site.frac_coords, site.frac_coords)[0, 0]

        # Distance to defect site (last site in defect sc)
        distance = defect_sc_with_site.get_distance(i, defect_site_index)  # len(defect_sc_with_site) - 1)
        disp_dict["Species"].append(site.specie.name)
        disp_dict["Distance to defect"].append(distance)
        disp_dict["Displacement"].append(np.linalg.norm(disp))
        disp_dict["Displacement vector"].append(disp)
        disp_dict["Index (defect supercell)"].append(i)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in scalar divide")
            if relative_to_defect:
                # Find vector from defect to site, accounting for periodic boundary conditions
                vector_defect_to_site = pbc_shortest_vectors(
                    bulk_sc.lattice, defect_sc_with_site[defect_site_index].frac_coords, site.frac_coords
                )[0, 0]
                norm = np.linalg.norm(vector_defect_to_site)
                disp_dict["Vector to site from defect"].append(vector_defect_to_site)
                if np.isclose(norm, 0, atol=1e-4):  # If defect site and site are the same
                    disp_dict["Displacement wrt defect"].append(0)
                else:
                    proj = np.dot(disp, vector_defect_to_site / norm)
                    disp_dict["Displacement wrt defect"].append(proj)

            if vector_to_project_on is not None:
                norm = np.linalg.norm(vector_to_project_on)  # Normalize vector to project on
                if norm == 0:
                    raise ValueError(
                        "Norm of vector to project on is zero! Choose a non-zero vector to project on."
                    )
                proj = np.dot(disp, vector_to_project_on / norm)
                angle = np.arccos(proj / np.linalg.norm(disp))
                rejection = np.linalg.norm(disp) * np.sin(angle)
                disp_dict["Displacement projected along vector"].append(proj)
                disp_dict["Displacement perpendicular to vector"].append(rejection)

    # sort each list in disp dict by index of species in bulk element list, then by distance to defect:
    element_list = _get_element_list(defect_entry)
    disp_df = pd.DataFrame(disp_dict)
    # Sort by species, then distance to defect, then index:
    disp_df = disp_df.sort_values(
        by=["Species", "Distance to defect", "Index (defect supercell)"],
        key=lambda col: col.map(element_list.index) if col.name == "Species" else col,
    )
    disp_df = _round_floats(disp_df, 3)  # round numerical values to 3 dp
    # reorder columns as species, distance, displacement, etc, then index last:
    initial_columns = ["Species", "Distance to defect", "Displacement", "Displacement vector"]
    disp_df = disp_df[
        initial_columns
        + [col for col in disp_df.columns if col not in initial_columns and "Index" not in col]
        + ["Index (defect supercell)"]
    ]

    # Store in DefectEntry.calculation_metadata
    # For vacancies, before storing displacements data, remove the last site
    # (defect site) as not present in input defect supercell
    # But leave it in disp_df as clearer to include in the displacement plot
    disp_vectors_list = deepcopy(list(disp_df["Displacement vector"]))
    distance_list = deepcopy(list(disp_df["Distance to defect"]))
    if defect_entry.defect.defect_type.name == "Vacancy":
        # get idx of value closest to zero:
        min_idx = min(range(len(distance_list)), key=lambda i: abs(distance_list[i]))
        if np.isclose(distance_list[min_idx], 0, atol=1e-2):  # just to be sure
            disp_vectors_list.pop(min_idx)
            distance_list.pop(min_idx)
    # Store in DefectEntry.calculation_metadata
    defect_entry.calculation_metadata["site_displacements"] = {
        "displacements": disp_vectors_list,  # Ordered by site index in defect supercell
        "distances": distance_list,
    }

    return disp_df


def plot_site_displacements(
    defect_entry: DefectEntry,
    separated_by_direction: bool = False,
    relative_to_defect: bool = False,
    vector_to_project_on: Optional[list] = None,
    use_plotly: bool = False,
    style_file: Optional[PathLike] = None,
):
    """
    Plots site displacements around a defect.

    Set ``use_plotly = True`` to get an interactive ``plotly``
    plot, useful for analysis!

    Args:
        defect_entry (DefectEntry): DefectEntry object
        separated_by_direction (bool):
            Whether to plot site displacements separated by
            direction (x, y, z). Default is False.
        relative_to_defect (bool):
            Whether to plot the signed displacements
            along the line from the defect site to that atom. Negative values
            indicate the atom moves towards the defect (compressive strain),
            positive values indicate the atom moves away from the defect
            (tensile strain). Uses the *relaxed* defect position as reference.
        vector_to_project_on (bool):
            Direction to project the site displacements along
            (e.g. [0, 0, 1]). Defaults to None (e.g. the displacements
            are calculated in the cartesian basis x, y, z).
        use_plotly (bool):
            Whether to use ``plotly`` for plotting. Default is ``False``.
            Set to ``True`` to get an interactive plot.
        style_file (PathLike):
            Path to ``matplotlib`` style file. if not set, will use the
            ``doped`` default displacements style.

    Returns:
        ``plotly`` or ``matplotlib`` ``Figure``.
    """

    def _mpl_plot_total_disp(
        disp_type_key,
        ylabel,
        disp_df,
        color_dict,
        styled_fig_size,
        styled_font_size,
    ):
        """
        Function to plot absolute/total displacement.

        Depending on the disp_type_key specified, will plot either the
        normalised displacement (disp_type_key="Displacement"), the
        displacement wrt the defect (disp_type_key="Displacement wrt defect"),
        or the displacmeent projected along a specified direction (
        disp_type_key="Displacement projected along vector").
        """
        fig, ax = plt.subplots(figsize=(styled_fig_size[0], styled_fig_size[1]))
        y_data = disp_df[disp_type_key]
        ax.scatter(
            disp_df["Distance to defect"],
            y_data,
            c=disp_df["Species"].map(color_dict),
            alpha=0.4,
            edgecolor="none",
        )
        ax.set_xlabel("Distance to defect ($\\AA$)", fontsize=styled_font_size)
        ax.set_ylabel(ylabel, fontsize=styled_font_size)
        # Add legend with species manually
        patches = [mpl.patches.Patch(color=color_dict[i], label=i) for i in unique_species]
        ax.legend(handles=patches)
        if disp_type_key in ("Displacement wrt defect", "Displacement projected along vector"):
            # Add horizontal line at 0
            ax.axhline(0, color="grey", alpha=0.3, linestyle="--")
        return fig

    def _plotly_plot_total_disp(
        disp_type_key,
        ylabel,
        disp_df,
    ):
        y_data = disp_df[disp_type_key]
        fig = px.scatter(
            x=disp_df["Distance to defect"],
            y=y_data,
            hover_data={
                "Distance to defect": disp_df["Distance to defect"],
                "Absolute displacement": y_data,
                "Species_with_index": [
                    f"{species} ({disp_df['Index (defect supercell)'][i]})"
                    for i, species in enumerate(disp_df["Species"])
                ],
            },
            color=disp_df["Species"],
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

    # Check user didn't set both relative_to_defect and vector_to_project_on
    if (
        separated_by_direction
        and (relative_to_defect or vector_to_project_on is not None)
        or (relative_to_defect and vector_to_project_on is not None)
    ):
        raise ValueError(
            "Cannot separate by direction and also plot relative displacements or displacements "
            "projected along a vector. Please only set one of these three options (e.g. to plot "
            "displacements relative to defect, rerun with relative_to_defect=True, "
            "separated_by_direction=False and vector_to_project_on=None)"
        )

    disp_df = calc_site_displacements(
        defect_entry=defect_entry,
        relative_to_defect=relative_to_defect,
        vector_to_project_on=vector_to_project_on,
    )
    if use_plotly and not plotly_installed:
        warnings.warn("Plotly not installed, using matplotlib instead")
        use_plotly = False
    if use_plotly:
        hovertemplate = "Distance to defect: %{x:.2f}<br>Absolute displacement: %{y:.2f}<br>Species: %{z}"
        if relative_to_defect:
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement wrt defect",
                ylabel="Displacement wrt defect",  # Angstrom symbol added in function
                disp_df=disp_df,
            )
        elif vector_to_project_on:
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement projected along vector",
                ylabel=f"Disp. along vector {tuple(vector_to_project_on)}",
                disp_df=disp_df,
            )
        elif not separated_by_direction:  # total displacement
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement",
                ylabel="Absolute displacement",
                disp_df=disp_df,
            )
        else:  # separated by direction
            fig = make_subplots(
                rows=1, cols=3, subplot_titles=("x", "y", "z"), shared_xaxes=True, shared_yaxes=True
            )
            unique_species = list(set(disp_df["Species"]))
            color_dict = dict(zip(unique_species, px.colors.qualitative.Plotly[: len(unique_species)]))
            for dir_index, _direction in enumerate(["x", "y", "z"]):
                fig.add_trace(
                    go.Scatter(
                        x=disp_df["Distance to defect"],
                        y=[abs(i[dir_index]) for i in disp_df["Displacement vector"]],
                        hovertemplate=hovertemplate.replace("{z", "{text"),
                        text=[
                            f"{species} ({disp_df['Index (defect supercell)'][i]})"
                            for i, species in enumerate(disp_df["Species"])
                        ],
                        marker={"color": disp_df["Species"].map(color_dict)},
                        mode="markers",
                        showlegend=False,
                    ),  # Only scatter plot, no line
                    row=1,
                    col=dir_index + 1,
                )
            # Add legend for color used for each species
            for specie, color in color_dict.items():
                fig.add_trace(
                    go.Scatter(
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
        element_list = _get_element_list(defect_entry)

        style_file = style_file or f"{os.path.dirname(__file__)}/displacement.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        with plt.style.context(style_file):
            # Color by species
            unique_species = list(set(disp_df["Species"]))
            unique_species.sort(key=lambda x: element_list.index(x))
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] or list(
                dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS).keys()
            )
            color_dict = {i: colors[index] for index, i in enumerate(unique_species)}

            # final figure width matching styled_fig_size, with dimensions matching the doped default:
            styled_fig_size = plt.rcParams["figure.figsize"]
            styled_font_size = plt.rcParams["font.size"]
            if relative_to_defect:
                return _mpl_plot_total_disp(
                    disp_type_key="Displacement wrt defect",
                    ylabel="Displacement wrt defect ($\\AA$)",
                    disp_df=disp_df,
                    color_dict=color_dict,
                    styled_fig_size=styled_fig_size,
                    styled_font_size=styled_font_size,
                )
            if not (vector_to_project_on or separated_by_direction):
                return _mpl_plot_total_disp(
                    disp_type_key="Displacement",
                    ylabel="Absolute displacement ($\\AA$)",
                    disp_df=disp_df,
                    color_dict=color_dict,
                    styled_fig_size=styled_fig_size,
                    styled_font_size=styled_font_size,
                )
            if vector_to_project_on:
                fig, ax = plt.subplots(
                    1,
                    2,
                    sharey=True,
                    sharex=True,
                    figsize=(1.5 * styled_fig_size[0], 0.6 * styled_fig_size[1]),  # (9.5, 4),
                )
                for index, i, title in zip(
                    [0, 1],
                    ["Displacement projected along vector", "Displacement perpendicular to vector"],
                    [
                        f"Parallel {tuple(vector_to_project_on)}",
                        f"Perpendicular {tuple(vector_to_project_on)}",
                    ],
                ):
                    ax[index].scatter(
                        disp_df["Distance to defect"],
                        disp_df[i],
                        c=disp_df["Species"].map(color_dict),
                        alpha=0.4,
                        edgecolor="none",
                    )
                    ax[index].axhline(0, color="grey", alpha=0.3, linestyle="--")
                    ax[index].set_title(f"{title}", fontsize=styled_font_size)  # Title with direction
                ax[0].set_ylabel("Displacements ($\\AA$)", fontsize=styled_font_size)

            else:  # else separated by direction
                fig, ax = plt.subplots(
                    1,
                    3,
                    figsize=(2.0 * styled_fig_size[0], 0.6 * styled_fig_size[1]),  # (13, 4),
                    sharey=True,
                    sharex=True,
                )
                for index, title in enumerate(["x", "y", "z"]):
                    ax[index].scatter(
                        disp_df["Distance to defect"],
                        [abs(j[index]) for j in disp_df["Displacement vector"]],
                        c=disp_df["Species"].map(color_dict),
                        alpha=0.4,
                        edgecolor="none",
                    )
                    ax[index].set_title(f"{title}")  # Title with direction
                ax[0].set_ylabel("Site displacements ($\\AA$)", fontsize=styled_font_size)
                fig.subplots_adjust(wspace=0.07)  # Set separation between subplots

            ax[1].set_xlabel("Distance to defect ($\\AA$)", fontsize=styled_font_size)
            patches = [mpl.patches.Patch(color=color_dict[i], label=i) for i in unique_species]
            ax[0].legend(handles=patches)  # Add legend with species manually

    return fig


def calc_displacements_ellipsoid(
    defect_entry: DefectEntry,
    quantile=0.8,
    return_extras=False,
) -> tuple:
    """
    Calculate displacements around a defect site and fit an ellipsoid to these
    displacements, returning a tuple of the ellipsoid's center, radii, rotation
    matrix and dataframe of anisotropy information.

    Args:
        defect_entry (DefectEntry): ``DefectEntry`` object.
        quantile (float):
            The quantile threshold for selecting significant displacements
            (between 0 and 1). Default is 0.8.
        return_extras (bool):
            Whether to also return the ``disp_df`` (output from
            ``calc_site_displacements(defect_entry, relative_to_defect=True)``)
            and the points used to fit the ellipsoid, corresponding to the
            Cartesian coordinates of the sites with displacements above the
            threshold, where the structure has been shifted to place the defect at
            the cell midpoint ([0.5, 0.5, 0.5]) in fractional coordinates.
            Default is ``False``.

    Returns:
    - (ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, aniostropy_df)
        A tuple containing the ellipsoid's center, radii, rotation matrix, and
        a dataframe of anisotropy information, or ``(None, None, None, None)`` if
        fitting was unsuccessful.
    - If ``return_extras=True``, also returns ``disp_df`` and the points used to
      fit the ellipsoid, appended to the return tuple.
    """

    def _get_minimum_volume_ellipsoid(P):
        """
        Find the minimum volume ellipsoid which holds all the points.

        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!

        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z], <-- one point per line
            [x,y,z],
            ...
            [x,y,z]]

        Returns:
        (center, radii, rotation)
        """
        tolerance = 0.01

        P = np.array(P)
        N, d = np.shape(P)

        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)])
        QT = Q.T

        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT, np.dot(np.linalg.inv(V), Q)))  # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse
        center = np.dot(P.T, u)

        # the A matrix for the ellipse
        A = (
            np.linalg.inv(
                np.dot(P.T, np.dot(np.diag(u), P)) - np.array([[a * b for b in center] for a in center])
            )
            / d
        )

        # Get the values we'd like to return
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0 / np.sqrt(s)

        return center, radii, rotation

    disp_df = calc_site_displacements(defect_entry, relative_to_defect=True)

    # Calculate the threshold for displacement norm, ensuring it's at least 0.05
    threshold = max(disp_df["Displacement"].quantile(quantile), 0.05)

    # Filter the DataFrame to get points where the displacement norm exceeds the threshold
    displacement_norm_over_threshold = disp_df[disp_df["Displacement"] > threshold]

    # Extract the Cartesian coordinates of the points that are over the threshold
    midpoint_coords = _get_defect_supercell(defect_entry).lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    points = np.array(
        list(
            displacement_norm_over_threshold["Vector to site from defect"].map(
                lambda x: x + midpoint_coords
            )
        )
    )
    ellipsoid_center = ellipsoid_radii = ellipsoid_rotation = anisotropy_df = None

    if points.shape[0] >= 10:  # only proceed if there are at least 10 points over the threshold
        try:
            # Try to fit a minimum volume ellipsoid to the points
            ellipsoid_center, ellipsoid_radii, ellipsoid_rotation = _get_minimum_volume_ellipsoid(points)
            anisotropy_df = pd.DataFrame(
                {
                    "Longest Radius": ellipsoid_radii[2],
                    "2nd_Longest/Longest": ellipsoid_radii[1] / ellipsoid_radii[2],
                    "3rd_Longest/Longest": ellipsoid_radii[0] / ellipsoid_radii[2],
                    "Threshold": threshold,
                },
                index=[0],
            )

        except np.linalg.LinAlgError:  # handle the case where the matrix is singular and fitting fails
            print("The matrix is singular and the system has no unique solution.")

    else:  # If there aren't enough points, suggest using a smaller quantile and return None values
        print("Not enough points for plotting, try using a smaller quantile!")

    if return_extras:
        return ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, anisotropy_df, disp_df, points
    return ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, anisotropy_df


def plot_displacements_ellipsoid(
    defect_entry: DefectEntry,
    plot_ellipsoid: bool = True,
    plot_anisotropy: bool = False,
    quantile=0.8,
    use_plotly: bool = False,
    show_supercell: bool = True,
    style_file: Optional[PathLike] = None,
) -> tuple:
    """
    Plot the displacement ellipsoid and/or anisotropy around a relaxed defect.

    Set ``use_plotly = True`` to get an interactive ``plotly``
    plot, useful for analysis!

    The supercell edges are also plotted if ``show_supercell = True``
    (default).

    Args:
        defect_entry (DefectEntry): ``DefectEntry`` object.
        plot_ellipsoid (bool):
            If True, plot the fitted ellipsoid in the crystal lattice.
        plot_anisotropy (bool):
            If True, plot the anisotropy of the ellipsoid radii.
        quantile (float):
            The quantile threshold for selecting significant displacements
            (between 0 and 1). Default is 0.8.
        use_plotly (bool):
            Whether to use ``plotly`` for plotting. Default is ``False``.
            Set to ``True`` to get an interactive plot.
        show_supercell (bool):
            Whether to show the supercell edges in the plot. Default is
            ``True``.
        style_file (PathLike):
            Path to ``matplotlib`` style file. if not set, will use the
            ``doped`` default displacements style.

    Returns:
        Either a single ``plotly`` or ``matplotlib`` ``Figure``, if only one
        of ``plot_ellipsoid`` or ``plot_anisotropy`` are ``True``, or a tuple
        of plots if both are ``True``.
    """
    if use_plotly and not plotly_installed:
        warnings.warn("Plotly not installed, using matplotlib instead")
        use_plotly = False

    if not (plot_ellipsoid or plot_anisotropy):
        raise ValueError("At least one of plot_ellipsoid or plot_anisotropy must be True!")

    def _get_ellipsoid_surface_xyz_and_axes(ellipsoid_center, ellipsoid_radii, ellipsoid_rotation):
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)

        # Cartesian coordinates corresponding to the spherical angles:
        x = ellipsoid_radii[0] * np.outer(np.cos(u), np.sin(v))
        y = ellipsoid_radii[1] * np.outer(np.sin(u), np.sin(v))
        z = ellipsoid_radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = (
                    np.dot([x[i, j], y[i, j], z[i, j]], ellipsoid_rotation) + ellipsoid_center
                )

        axes = np.array(
            [
                [ellipsoid_radii[0], 0.0, 0.0],
                [0.0, ellipsoid_radii[1], 0.0],
                [0.0, 0.0, ellipsoid_radii[2]],
            ]
        )
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], ellipsoid_rotation)

        return x, y, z, axes

    def _generate_lattice_points(lattice_matrix, scale=0.1, n_points=11):
        points = []

        for i in range(3):  # generate points for single lattice vectors
            points.append(
                {
                    "x": [lattice_matrix[i][0] * scale * n for n in range(n_points)],
                    "y": [lattice_matrix[i][1] * scale * n for n in range(n_points)],
                    "z": [lattice_matrix[i][2] * scale * n for n in range(n_points)],
                }
            )

        for i in range(3):  # generate points for combinations of two lattice vectors
            for j in range(3):
                if i != j:
                    points.append(
                        {
                            "x": [
                                lattice_matrix[i][0] * scale * n + lattice_matrix[j][0]
                                for n in range(n_points)
                            ],
                            "y": [
                                lattice_matrix[i][1] * scale * n + lattice_matrix[j][1]
                                for n in range(n_points)
                            ],
                            "z": [
                                lattice_matrix[i][2] * scale * n + lattice_matrix[j][2]
                                for n in range(n_points)
                            ],
                        }
                    )

        # generate points for combinations of three lattice vectors
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]]:
            points.append(
                {
                    "x": [
                        lattice_matrix[order[0]][0] * scale * n
                        + lattice_matrix[order[1]][0]
                        + lattice_matrix[order[2]][0]
                        for n in range(n_points)
                    ],
                    "y": [
                        lattice_matrix[order[0]][1] * scale * n
                        + lattice_matrix[order[1]][1]
                        + lattice_matrix[order[2]][1]
                        for n in range(n_points)
                    ],
                    "z": [
                        lattice_matrix[order[0]][2] * scale * n
                        + lattice_matrix[order[1]][2]
                        + lattice_matrix[order[2]][2]
                        for n in range(n_points)
                    ],
                }
            )

        return points

    def _mpl_plot_ellipsoid(
        ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix, style_file
    ):
        x, y, z, axes = _get_ellipsoid_surface_xyz_and_axes(
            ellipsoid_center, ellipsoid_radii, ellipsoid_rotation
        )

        # Create a 3D plot
        style_file = style_file or f"{os.path.dirname(__file__)}/displacement.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        with plt.style.context(style_file):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot the ellipsoid surface
            ax.plot_surface(x, y, z, color="blue", alpha=0.2, rstride=4, cstride=4)

            # Plot the points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="black", s=10)

            # Plot the ellipsoid axes
            for p in axes:
                ax.plot(
                    [ellipsoid_center[0], ellipsoid_center[0] + p[0]],
                    [ellipsoid_center[1], ellipsoid_center[1] + p[1]],
                    [ellipsoid_center[2], ellipsoid_center[2] + p[2]],
                    color="black",
                    linewidth=2,
                )

            if show_supercell:  # plot lattice vectors
                for point_set in _generate_lattice_points(lattice_matrix):
                    ax.plot(point_set["x"], point_set["y"], point_set["z"], color="black")

            ax.set_box_aspect([1, 1, 1], zoom=0.9)  # set the aspect ratio and limits, and zoom out a bit
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        return fig

    def _plotly_plot_ellipsoid(
        ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix
    ):
        x, y, z, axes = _get_ellipsoid_surface_xyz_and_axes(
            ellipsoid_center, ellipsoid_radii, ellipsoid_rotation
        )

        fig = go.Figure(
            data=[go.Surface(x=x, y=y, z=z, opacity=0.2, showscale=False, surfacecolor=np.zeros_like(x))]
        )

        # Add the points contained in P
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"color": "black", "size": 3},
            )
        )

        fig.update_layout(
            scene={"aspectmode": "data"},
            template="plotly_white",
            paper_bgcolor="white",
            showlegend=False,
            width=700,
            height=600,
        )

        # plot axes
        for p in axes:
            fig.add_trace(
                go.Scatter3d(
                    x=[ellipsoid_center[0], ellipsoid_center[0] + p[0]],
                    y=[ellipsoid_center[1], ellipsoid_center[1] + p[1]],
                    z=[ellipsoid_center[2], ellipsoid_center[2] + p[2]],
                    mode="lines",
                    line={"color": "black", "width": 2},
                )
            )

        if show_supercell:  # plot lattice vectors
            for point_set in _generate_lattice_points(lattice_matrix):
                fig.add_trace(
                    go.Scatter3d(
                        x=point_set["x"],
                        y=point_set["y"],
                        z=point_set["z"],
                        marker={"size": 0.5, "color": "black"},
                        mode="lines",
                    )
                )

        return fig

    def _mpl_plot_anisotropy(disp_df, anisotropy_df, style_file):
        style_file = style_file or f"{os.path.dirname(__file__)}/displacement.mplstyle"
        plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
        with plt.style.context(style_file):
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            # Part 1: Displacement Distribution Box Plot
            axs[0].boxplot(disp_df["Displacement"], vert=True, patch_artist=True)
            axs[0].set_title("Displacement Norm Distribution")
            axs[0].set_ylabel("Displacement Norm (Å)")
            axs[0].grid(False)
            axs[0].xaxis.set_visible(False)  # Hide x-axis labels for box plot

            # Part 2: Anisotropy Scatter Plot
            if anisotropy_df is not None:
                scatter = axs[1].scatter(  # Create scatter plot
                    anisotropy_df["2nd_Longest/Longest"],
                    anisotropy_df["3rd_Longest/Longest"],
                    c=anisotropy_df["Longest Radius"],
                    cmap="rainbow",
                    s=100,
                    alpha=1,
                )
                axs[1].plot([0, 1], [0, 1], "k--")  # Add y=x line

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axs[1])
                cbar.set_label("Ellipsoid Maximum Radius (Å)")

                # Set titles and labels
                axs[1].set_title(f"Anisotropy (Threshold = {anisotropy_df['Threshold'].iloc[0]:.3f} Å)")
                axs[1].set_xlabel("2nd-Largest Radius / Largest Radius")
                axs[1].set_ylabel("Shortest Radius / Largest Radius")
                axs[1].set_xlim([0, 1])
                axs[1].set_ylim([0, 1])
                axs[1].grid(False)

            fig.tight_layout()  # adjust layout
        return fig

    def _plotly_plot_anisotropy(disp_df, anisotropy_df):
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Displacement Norm Distribution",
                f"Anisotropy (Threshold = {anisotropy_df['Threshold'].iloc[0]:.3f} Å)",
            ],
            column_widths=[0.5, 0.5],
        )

        # Part 1: Displacement Distribution Box Plot
        fig.add_trace(go.Box(y=disp_df["Displacement"], boxpoints="all"), row=1, col=1)
        fig.update_yaxes(row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide x-axis labels for box plot

        # Add frame to Displacement Distribution plot
        fig.update_xaxes(linecolor="black", linewidth=1, row=1, col=1, mirror=True)
        fig.update_yaxes(
            title_text="Displacement Norm (Å)", linecolor="black", linewidth=1, row=1, col=1, mirror=True
        )

        # Part 2: Anisotropy Scatter Plot
        if anisotropy_df is not None:
            scatter = go.Scatter(
                x=anisotropy_df["2nd_Longest/Longest"],
                y=anisotropy_df["3rd_Longest/Longest"],
                mode="markers",
                marker={
                    "size": 10,
                    "opacity": 0.5,
                    "color": anisotropy_df["Longest Radius"],  # set color according to column "a"
                    "colorscale": "rainbow",
                    "colorbar": {"title": "Ellipsoid Maximum Radius (Å)", "titleside": "right"},
                },
                text=anisotropy_df["Longest Radius"],
                hoverinfo="text",
            )
            fig.add_trace(scatter, row=1, col=2)

            # Add y=x line to the anisotropy plot
            line = go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", line={"color": "black", "dash": "dash"}, name="y=x Line"
            )
            fig.add_trace(line, row=1, col=2)

            # Add frame to Anisotropy plot
            fig.update_xaxes(
                title_text="2nd-Largest Radius / Largest Radius",
                range=[0, 1],
                row=1,
                col=2,
                title_font={"size": 10},
                tickfont={"size": 12},
                linecolor="black",
                linewidth=1,
                mirror=True,
            )
            fig.update_yaxes(
                title_text="Shortest Radius / Largest Radius",
                range=[0, 1],
                row=1,
                col=2,
                title_font={"size": 10},
                tickfont={"size": 12},
                linecolor="black",
                linewidth=1,
                mirror=True,
                title_standoff=5,
            )

            # Layout adjustments
            fig.update_layout(
                width=1100,  # Double the width to accommodate both plots
                height=500,
                plot_bgcolor="white",
                showlegend=False,  # Disable legend
            )

        return fig

    ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, anisotropy_df, disp_df, points = (
        calc_displacements_ellipsoid(defect_entry, quantile, return_extras=True)
    )

    return_list = []
    # If ellipsoid plotting is enabled, plot the ellipsoid with the given lattice matrix
    if plot_ellipsoid:
        bulk_sc, defect_sc_with_site, defect_site_index = _get_bulk_struct_with_defect(defect_entry)
        lattice_matrix = bulk_sc.as_dict()["lattice"]["matrix"]
        func = _mpl_plot_ellipsoid if not use_plotly else _plotly_plot_ellipsoid
        args = [ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix]
        if not use_plotly:
            args.append(style_file)

        return_list.append(func(*args))  # type: ignore

    # If anisotropy plotting is enabled, plot the ellipsoid's radii anisotropy
    if plot_anisotropy:
        func = _mpl_plot_anisotropy if not use_plotly else _plotly_plot_anisotropy
        args = [disp_df, anisotropy_df]
        if not use_plotly:
            args.append(style_file)

        return_list.append(func(*args))  # type: ignore

    return next(iter(return_list)) if len(return_list) == 1 else tuple(return_list)


def _get_bulk_struct_with_defect(defect_entry) -> tuple:
    """
    Returns structures for bulk and defect supercells with the same number of
    sites and species, to be used for site matching.

    If Vacancy, adds (unrelaxed) site to defect structure. If Interstitial,
    adds relaxed site to bulk structure. If Substitution, replaces (unrelaxed)
    defect site in bulk structure.

    Returns tuple of ``(bulk_sc_with_defect, defect_sc_with_defect)``.
    """
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
    else:
        raise ValueError(f"Defect type {defect_type} not supported")
    return bulk_sc_with_defect, defect_sc_with_defect, defect_site_index
