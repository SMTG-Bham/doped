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
from pymatgen.util.coord import pbc_diff
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    _get_defect_supercell_bulk_site_coords,
    _get_defect_supercell_site,
    get_site_mapping_indices,
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_installed = True
except ImportError:
    plotly_installed = False


def calc_site_displacements(
    defect_entry: DefectEntry,
    vector_to_project_on: Optional[list] = None,
    relative_to_defect: bool = False,
) -> dict:
    """
    Calculates the site displacements in the defect supercell, relative to the
    bulk supercell. The signed displacements are stored in the
    calculation_metadata of the ``DefectEntry`` object under the
    "site_displacements" key.

    Args:
        defect_entry (DefectEntry):
            DefectEntry object
        vector_to_project_on (list):
            Direction to project the site displacements along
            (e.g. [0, 0, 1]). Defaults to None.
        relative_to_defect (bool):
            Whether to calculate the signed displacements along the line
            from the defect site to that atom. Negative values indicate
            the atom moves towards the defect (compressive strain),
            positive values indicate the atom moves away from the defect.
            Defaults to False. If True, the relative displacements are stored in
            the `Displacement wrt defect` key of the returned dictionary.

    Returns:
        Dictionary with site displacements (compared to pristine supercell).
    """
    bulk_sc, defect_sc_with_site, defect_site_index = _get_bulk_struct_with_defect(defect_entry)
    # Map sites in defect supercell to bulk supercell
    mappings = get_site_mapping_indices(defect_sc_with_site, bulk_sc)
    mappings_dict = {i[1]: i[2] for i in mappings}  # {defect_sc_index: bulk_sc_index}
    # Loop over sites in defect sc
    disp_dict = {  # mapping defect site index (in defect sc) to displacement
        "Index (defect)": [],
        "Species": [],
        "Species_with_index": [],
        "Displacement": [],
        "Distance to defect": [],
    }  # type: dict
    if relative_to_defect:
        disp_dict["Displacement wrt defect"] = []
    if vector_to_project_on is not None:
        disp_dict["Displacement projected along vector"] = []
        disp_dict["Displacement perpendicular to vector"] = []
    for i, site in enumerate(defect_sc_with_site):
        # print(i, site.specie, site.frac_coords)  # debugging
        bulk_sc_index = mappings_dict[i]  # Map to bulk sc
        bulk_site = bulk_sc[bulk_sc_index]  # Get site in bulk sc
        # Calculate displacement (need to account for pbc!)
        # First final point, then initial point
        frac_disp = pbc_diff(site.frac_coords, bulk_site.frac_coords)  # in fractional coords
        disp = bulk_sc.lattice.get_cartesian_coords(frac_disp)  # in Angstroms
        # Distance to defect site (last site in defect sc)
        distance = defect_sc_with_site.get_distance(i, defect_site_index)  # len(defect_sc_with_site) - 1)
        # print(i, displacement, np.linalg.norm(abs_disp), "Distance:", distance)  # debugging
        disp_dict["Index (defect)"].append(i)
        disp_dict["Displacement"].append(disp)
        disp_dict["Distance to defect"].append(distance)
        disp_dict["Species_with_index"].append(f"{site.specie.name}({i})")
        disp_dict["Species"].append(site.specie.name)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in scalar divide")
            if relative_to_defect:
                # Find vector from defect to site, accounting for periodic boundary conditions
                vector_defect_to_site = pbc_diff(
                    site.frac_coords, defect_sc_with_site[defect_site_index].frac_coords
                )
                norm = np.linalg.norm(vector_defect_to_site)
                if norm == 0:  # If defect site and site are the same
                    disp_dict["Displacement wrt defect"].append(0)
                else:
                    proj = np.dot(disp, vector_defect_to_site / norm)
                    disp_dict["Displacement wrt defect"].append(proj)
            if vector_to_project_on is not None:
                # Normalize vector to project on
                norm = np.linalg.norm(vector_to_project_on)
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
    if relative_to_defect and vector_to_project_on is not None:
        (
            disp_dict["Index (defect)"],
            disp_dict["Species"],
            disp_dict["Species_with_index"],
            disp_dict["Displacement"],
            disp_dict["Distance to defect"],
            disp_dict["Displacement wrt defect"],
            disp_dict["Displacement projected along vector"],
            disp_dict["Displacement perpendicular to vector"],
        ) = zip(*combined)
    if relative_to_defect and vector_to_project_on is None:
        (
            disp_dict["Index (defect)"],
            disp_dict["Species"],
            disp_dict["Species_with_index"],
            disp_dict["Displacement"],
            disp_dict["Distance to defect"],
            disp_dict["Displacement wrt defect"],
        ) = zip(*combined)
    elif vector_to_project_on is not None and not relative_to_defect:
        (
            disp_dict["Index (defect)"],
            disp_dict["Species"],
            disp_dict["Species_with_index"],
            disp_dict["Displacement"],
            disp_dict["Distance to defect"],
            disp_dict["Displacement projected along vector"],
            disp_dict["Displacement perpendicular to vector"],
        ) = zip(*combined)
    else:
        (
            disp_dict["Index (defect)"],
            disp_dict["Species"],
            disp_dict["Species_with_index"],
            disp_dict["Displacement"],
            disp_dict["Distance to defect"],
        ) = zip(*combined)

    # Store in DefectEntry.calculation_metadata
    # For vacancies, before storing displacements data, remove the last site
    # (defect site) as not present in input defect supercell
    # But leave it in disp_dict as clearer to include in the displacement plot?
    disp_list = list(deepcopy(disp_dict["Displacement"]))
    distance_list = list(deepcopy(disp_dict["Distance to defect"]))
    if defect_entry.defect.defect_type.name == "Vacancy":
        # get idx of value closest to zero:
        min_idx = min(range(len(distance_list)), key=lambda i: abs(distance_list[i]))
        if np.isclose(distance_list[min_idx], 0, atol=1e-2):  # just to be sure
            disp_list.pop(min_idx)
            distance_list.pop(min_idx)
    # Store in DefectEntry.calculation_metadata
    defect_entry.calculation_metadata["site_displacements"] = {
        "displacements": disp_list,  # Ordered by site index in defect supercell
        "distances": distance_list,
    }
    return disp_dict


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
            Path to matplotlib style file. if not set, will use the
            ``doped`` default style.

    Returns:
        ``plotly`` or ``matplotlib`` ``Figure``.
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
        normalised displacement (disp_type_key="Displacement"), the
        displacement wrt the defect (disp_type_key="Displacement wrt defect"),
        or the displacmeent projected along a specified direction (
        disp_type_key="Displacement projected along vector").
        """
        fig, ax = plt.subplots(figsize=(styled_fig_size[0], styled_fig_size[1]))
        if disp_type_key == "Displacement":
            y_data = [np.linalg.norm(i) for i in disp_dict["Displacement"]]
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
        disp_dict,
    ):
        if disp_type_key == "Displacement":
            y_data = [np.linalg.norm(i) for i in disp_dict["Displacement"]]
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

    # Check user didn't set both relative_to_defect and vector_to_project_on
    if (
        separated_by_direction
        and (relative_to_defect or vector_to_project_on is not None)
        or (relative_to_defect and vector_to_project_on is not None)
    ):
        raise ValueError(
            "Cannot separate by direction and also plot relative displacements"
            " or displacements projected along a vector. Please only set one"
            " of these three options (e.g. to plot displacements relative to defect,"
            " rerun with relative_to_defect=True, separated_by_direction=False"
            " and vector_to_project_on=None)"
        )

    disp_dict = calc_site_displacements(
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
                disp_dict=disp_dict,
            )
        elif vector_to_project_on:
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement projected along vector",
                ylabel=f"Disp. along vector {tuple(vector_to_project_on)}",
                disp_dict=disp_dict,
            )
        elif not separated_by_direction:  # total displacement
            fig = _plotly_plot_total_disp(
                disp_type_key="Displacement",
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
                    go.Scatter(
                        x=disp_dict["Distance to defect"],
                        y=[abs(i[dir_index]) for i in disp_dict["Displacement"]],
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
                    ylabel="Displacement wrt defect ($\\AA$)",
                    disp_dict=disp_dict,
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
                        disp_dict["Distance to defect"],
                        disp_dict[i],
                        c=[color_dict[i] for i in disp_dict["Species"]],
                        alpha=0.4,
                        edgecolor="none",
                    )
                    ax[index].axhline(0, color="grey", alpha=0.3, linestyle="--")
                    # Title with direction
                    ax[index].set_title(f"{title}", fontsize=styled_font_size)
                ax[0].set_ylabel("Displacements ($\\AA$)", fontsize=styled_font_size)
                ax[1].set_xlabel("Distance to defect ($\\AA$)", fontsize=styled_font_size)
                # Add legend with species manually
                patches = [mpl.patches.Patch(color=color_dict[i], label=i) for i in unique_species]
                ax[0].legend(handles=patches)
                return fig
            if not separated_by_direction:
                return _mpl_plot_total_disp(
                    disp_type_key="Displacement",
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
                    [abs(j[index]) for j in disp_dict["Displacement"]],
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


def calc_displacements_ellipsoid(
    defect_entry: DefectEntry,
    plot_ellipsoid: bool = False,
    plot_anisotropy: bool = False,
    use_plotly: bool = False,
    quantile=0.8,
):
    """
    Calculate displacements around a defect site and fit an ellipsoid to these
    displacements.

    Set ``use_plotly = True`` to get an interactive ``plotly``
    plot, useful for analysis!

    Args:
        defect_entry (DefectEntry): ``DefectEntry`` object.
        plot_ellipsoid (bool):
            If True, plot the fitted ellipsoid in the crystal lattice.
        plot_anisotropy (bool):
            If True, plot the anisotropy of the ellipsoid radii.
        use_plotly (bool):
            Whether to use ``plotly`` for plotting. Default is ``False``.
            Set to ``True`` to get an interactive plot.
        quantile (float):
            The quantile threshold for selecting significant displacements
            (between 0 and 1). Default is 0.8.

    Returns:
    - (ellipsoid_center, ellipsoid_radii, ellipsoid_rotation):
        A tuple containing the ellipsoid's center, radii, and rotation matrix,
        or ``(None, None, None)`` if fitting was unsuccessful.
    """
    if use_plotly and not plotly_installed:
        warnings.warn("Plotly not installed, using matplotlib instead")
        use_plotly = False

    def _get_minimum_volume_ellipsoid(P):
        """
        Find the minimum volume ellipsoid which holds all the points.

        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!

        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
            [x,y,z,...],
            [x,y,z,...]]

        Returns:
        (center, radii, rotation)
        """
        tolerance = 0.01

        (N, d) = np.shape(P)
        d = float(d)

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

        return (center, radii, rotation)

    def _mpl_plot_ellipsoid(ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix):
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

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the ellipsoid surface
        ax.plot_surface(x, y, z, color="blue", alpha=0.2, rstride=4, cstride=4)

        # Plot the points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="black", s=10)

        # Plot the ellipsoid axes
        axes = np.array(
            [
                [ellipsoid_radii[0], 0.0, 0.0],
                [0.0, ellipsoid_radii[1], 0.0],
                [0.0, 0.0, ellipsoid_radii[2]],
            ]
        )
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], ellipsoid_rotation)

        for p in axes:
            ax.plot(
                [ellipsoid_center[0], ellipsoid_center[0] + p[0]],
                [ellipsoid_center[1], ellipsoid_center[1] + p[1]],
                [ellipsoid_center[2], ellipsoid_center[2] + p[2]],
                color="black",
                linewidth=2,
            )

        def _plot_lattice(lattice_matrix, ax):

            # Scale factor for the lattice lines
            scale = 0.1

            # Create lines along each lattice vector
            for i in range(3):
                x = [lattice_matrix[i][0] * scale * n for n in range(11)]
                y = [lattice_matrix[i][1] * scale * n for n in range(11)]
                z = [lattice_matrix[i][2] * scale * n for n in range(11)]
                ax.plot(x, y, z, color="black", linewidth=0.5)

                # Create lines for combinations of lattice vectors
                for j in range(3):
                    if i != j:
                        x_comb = [
                            lattice_matrix[i][0] * scale * n + lattice_matrix[j][0] for n in range(11)
                        ]
                        y_comb = [
                            lattice_matrix[i][1] * scale * n + lattice_matrix[j][1] for n in range(11)
                        ]
                        z_comb = [
                            lattice_matrix[i][2] * scale * n + lattice_matrix[j][2] for n in range(11)
                        ]
                        ax.plot(x_comb, y_comb, z_comb, color="black", linewidth=0.5)

                        for k in range(3):
                            if i != k and j != k:
                                x_comb3 = [
                                    lattice_matrix[i][0] * scale * n
                                    + lattice_matrix[j][0]
                                    + lattice_matrix[k][0]
                                    for n in range(11)
                                ]
                                y_comb3 = [
                                    lattice_matrix[i][1] * scale * n
                                    + lattice_matrix[j][1]
                                    + lattice_matrix[k][1]
                                    for n in range(11)
                                ]
                                z_comb3 = [
                                    lattice_matrix[i][2] * scale * n
                                    + lattice_matrix[j][2]
                                    + lattice_matrix[k][2]
                                    for n in range(11)
                                ]
                                ax.plot(x_comb3, y_comb3, z_comb3, color="black", linewidth=0.5)

        _plot_lattice(lattice_matrix, ax)

        # Set the aspect ratio and limits
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()

    def _plotly_plot_ellipsoid(
        ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix
    ):
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)

        # cartesian coordinates that correspond to the spherical angles:
        x = ellipsoid_radii[0] * np.outer(np.cos(u), np.sin(v))
        y = ellipsoid_radii[1] * np.outer(np.sin(u), np.sin(v))
        z = ellipsoid_radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = (
                    np.dot([x[i, j], y[i, j], z[i, j]], ellipsoid_rotation) + ellipsoid_center
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

        # show supercell
        def _plot_lattice(lattice_matrix, fig):
            for i in range(3):  # lattice vectors
                fig.add_trace(
                    go.Scatter3d(
                        x=[lattice_matrix[i][0] * 0.1 * n for n in range(11)],
                        y=[lattice_matrix[i][1] * 0.1 * n for n in range(11)],
                        z=[lattice_matrix[i][2] * 0.1 * n for n in range(11)],
                        marker={"size": 0.5, "color": "black"},
                        mode="lines",
                    )
                )
                for j in range(i + 1, 3):  # add other two lattice vectors
                    fig.add_trace(  # add one of the other lattice vectors
                        go.Scatter3d(
                            x=[lattice_matrix[i][0] * 0.1 * n + lattice_matrix[j][0] for n in range(11)],
                            y=[lattice_matrix[i][1] * 0.1 * n + lattice_matrix[j][1] for n in range(11)],
                            z=[lattice_matrix[i][2] * 0.1 * n + lattice_matrix[j][2] for n in range(11)],
                            marker={"size": 0.5, "color": "black"},
                            mode="lines",
                        )
                    )

                    fig.add_trace(  # add both other lattice vectors
                        go.Scatter3d(
                            x=[
                                lattice_matrix[i][0] * 0.1 * n
                                + lattice_matrix[j][0]
                                + lattice_matrix[(j + 1) % 3][0]
                                for n in range(11)
                            ],
                            y=[
                                lattice_matrix[i][1] * 0.1 * n
                                + lattice_matrix[j][1]
                                + lattice_matrix[(j + 1) % 3][1]
                                for n in range(11)
                            ],
                            z=[
                                lattice_matrix[i][2] * 0.1 * n
                                + lattice_matrix[j][2]
                                + lattice_matrix[(j + 1) % 3][2]
                                for n in range(11)
                            ],
                            marker={"size": 0.5, "color": "black"},
                            mode="lines",
                        )
                    )

        _plot_lattice(lattice_matrix, fig)
        fig.show()

    def _mpl_plot_anisotropy(ellipsoid_radii, disp_df, threshold):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Part 1: Displacement Distribution Box Plot
        axs[0].boxplot(disp_df["Displacement Norm"], vert=True, patch_artist=True)
        axs[0].set_title("Displacement Norm Distribution")
        axs[0].set_ylabel("Displacement Norm (Å)")
        axs[0].grid(False)
        axs[0].xaxis.set_visible(False)  # Hide x-axis labels for box plot

        # Part 2: Anisotropy Scatter Plot
        if ellipsoid_radii is not None:
            the_longest_radius = ellipsoid_radii[2]
            the_second_longest_radius = ellipsoid_radii[1]
            the_third_longest_radius = ellipsoid_radii[0]
            ratio_of_second_to_the_longest = the_second_longest_radius / the_longest_radius
            ratio_of_third_to_the_longest = the_third_longest_radius / the_longest_radius

            # Create scatter plot
            scatter = axs[1].scatter(
                ratio_of_second_to_the_longest,
                ratio_of_third_to_the_longest,
                c=the_longest_radius,
                cmap="rainbow",
                s=100,
                alpha=1,
            )
            axs[1].plot([0, 1], [0, 1], "k--")  # Add y=x line

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axs[1])
            cbar.set_label("Ellipsoid Maximum Radius (Å)")

            # Set titles and labels
            axs[1].set_title(f"Anisotropy (Threshold = {threshold:.3f} Å)")
            axs[1].set_xlabel("2nd-Largest Radius / Largest Radius")
            axs[1].set_ylabel("Shortest Radius / Largest Radius")
            axs[1].set_xlim([0, 1])
            axs[1].set_ylim([0, 1])
            axs[1].grid(False)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def _plotly_plot_anisotropy(ellipsoid_radii, disp_df, threshold):
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Displacement Norm Distribution",
                f"Anisotropy (Threshold = {threshold:.3f} Å)",
            ],
            column_widths=[0.5, 0.5],
        )

        # Part 1: Displacement Distribution Box Plot
        fig.add_trace(go.Box(y=disp_df["Displacement Norm"], boxpoints="all"), row=1, col=1)
        fig.update_yaxes(row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide x-axis labels for box plot

        # Add frame to Displacement Distribution plot
        fig.update_xaxes(linecolor="black", linewidth=1, row=1, col=1, mirror=True)
        fig.update_yaxes(
            title_text="Displacement Norm (Å)", linecolor="black", linewidth=1, row=1, col=1, mirror=True
        )

        # Part 2: Anisotropy Scatter Plot
        if ellipsoid_radii is not None:
            the_longest_radius = ellipsoid_radii[2]
            the_second_longest_radius = ellipsoid_radii[1]
            the_third_longest_radius = ellipsoid_radii[0]
            ratio_of_second_to_the_longest = the_second_longest_radius / the_longest_radius
            ratio_of_third_to_the_longest = the_third_longest_radius / the_longest_radius
            anisotropy_info_list = [
                [
                    threshold,
                    the_longest_radius,
                    ratio_of_second_to_the_longest,
                    ratio_of_third_to_the_longest,
                ]
            ]
            anisotropy_info_df = pd.DataFrame(
                anisotropy_info_list,
                columns=[
                    "threshold",
                    "the_longest_radius",
                    "ratio_of_second_to_the_longest",
                    "ratio_of_third_to_the_longest",
                ],
            )

            scatter = go.Scatter(
                x=anisotropy_info_df["ratio_of_second_to_the_longest"],
                y=anisotropy_info_df["ratio_of_third_to_the_longest"],
                mode="markers",
                marker={
                    "size": 10,
                    "opacity": 0.5,
                    "color": anisotropy_info_df["the_longest_radius"],  # Set color according to column "a"
                    "colorscale": "rainbow",
                    "colorbar": {"title": "Ellipsoid Maximum Radius (Å)", "titleside": "right"},
                },
                text=anisotropy_info_df["the_longest_radius"],
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

            # Show the combined plot
            fig.show()

    def _shift_defect_site_to_center_of_the_supercell(sites_frac_coords, defect_frac_coords, bulk_sc):
        """
        Shifts the fractional coordinates of a site so that the defect site is
        at the center of the supercell.

        Parameters:
        - sites_frac_coords (array-like): Fractional coordinates of the site to be shifted.
        - defect_frac_coords (array-like): Fractional coordinates of the defect site.
        - bulk_sc: Structure object of bulk supercell.

        Returns:
        - shifted_sites_cart_coords (np.array): Cartesian coordinates of the shifted site.
        """
        # Initialize the shifted fractional coordinates as a zero vector
        shifted_sites_frac_coords = np.zeros(3)

        # Define the fractional coordinates for the center of the supercell
        center_frac_coors = np.array([0.5, 0.5, 0.5])

        # Calculate the difference between the center of the supercell and the defect site
        diff_frac_coords = center_frac_coors - defect_frac_coords

        # Shift the site coordinates by the difference, bringing the defect site to the center
        tmp_sites = sites_frac_coords + diff_frac_coords

        # Adjust the fractional coordinates to ensure they stay within the unit cell [0, 1)
        for i, tmp_site in enumerate(tmp_sites):
            if tmp_site > 1:
                shifted_sites_frac_coords[i] = tmp_site - 1
            elif tmp_site < 0:
                shifted_sites_frac_coords[i] = tmp_site + 1
            else:
                shifted_sites_frac_coords[i] = tmp_site

        # Convert the shifted fractional coordinates to Cartesian coordinates using the lattice
        return bulk_sc.lattice.get_cartesian_coords(shifted_sites_frac_coords)

    bulk_sc, defect_sc_with_site, defect_site_index = _get_bulk_struct_with_defect(defect_entry)

    # Map sites in defect supercell to bulk supercell
    mappings = get_site_mapping_indices(defect_sc_with_site, bulk_sc)
    mappings_dict = {i[1]: i[2] for i in mappings}  # {defect_sc_index: bulk_sc_index}
    # Loop over sites in defect sc
    disp_dict = {  # mapping defect site index (in defect sc) to displacement
        "Index (defect)": [],
        "Species": [],
        "Species_with_index": [],
        "Displacement": [],
        "Displacement Norm": [],
        "Distance to defect": [],
        "X sites in cartesian coordinate (defect)": [],
        "Y sites in cartesian coordinate (defect)": [],
        "Z sites in cartesian coordinate (defect)": [],
    }  # type: dict

    sc_defect_frac_coords = _get_defect_supercell_bulk_site_coords(defect_entry)
    if sc_defect_frac_coords is None:
        raise ValueError(
            "The relaxed defect position (`DefectEntry.sc_defect_frac_coords`) has not been parsed. "
            "Please use `DefectsParser`/`DefectParser` to parse relaxed defect positions before "
            "calculating site displacements."
        )

    for i, site in enumerate(defect_sc_with_site):
        bulk_sc_index = mappings_dict[i]  # Map to bulk sc
        bulk_site = bulk_sc[bulk_sc_index]  # Get site in bulk sc
        # Calculate displacement (need to account for pbc!)
        # First final point, then initial point
        frac_disp = pbc_diff(site.frac_coords, bulk_site.frac_coords)  # in fractional coords
        disp = bulk_sc.lattice.get_cartesian_coords(frac_disp)  # in Angstroms
        # Distance to defect site (last site in defect sc)
        distance = defect_sc_with_site.get_distance(i, defect_site_index)  # len(defect_sc_with_site) - 1)

        disp_dict["Index (defect)"].append(i)
        disp_dict["Displacement"].append(disp)
        disp_dict["Displacement Norm"].append(np.linalg.norm(disp, ord=2))
        disp_dict["Distance to defect"].append(distance)
        disp_dict["Species_with_index"].append(f"{site.specie.name}({i})")
        disp_dict["Species"].append(site.specie.name)
        disp_dict["X sites in cartesian coordinate (defect)"].append(
            _shift_defect_site_to_center_of_the_supercell(
                site.frac_coords, sc_defect_frac_coords, bulk_sc
            )[0]
        )
        disp_dict["Y sites in cartesian coordinate (defect)"].append(
            _shift_defect_site_to_center_of_the_supercell(
                site.frac_coords, sc_defect_frac_coords, bulk_sc
            )[1]
        )
        disp_dict["Z sites in cartesian coordinate (defect)"].append(
            _shift_defect_site_to_center_of_the_supercell(
                site.frac_coords, sc_defect_frac_coords, bulk_sc
            )[2]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in scalar divide")

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
        disp_dict["Displacement"],
        disp_dict["Displacement Norm"],
        disp_dict["Distance to defect"],
        disp_dict["X sites in cartesian coordinate (defect)"],
        disp_dict["Y sites in cartesian coordinate (defect)"],
        disp_dict["Z sites in cartesian coordinate (defect)"],
    ) = zip(*combined)

    # Convert the displacement dictionary into a pandas DataFrame
    disp_df = pd.DataFrame(disp_dict)

    # Calculate the threshold for displacement norm, ensuring it's at least 0.05
    threshold = max(disp_df["Displacement Norm"].quantile(quantile), 0.05)

    # Filter the DataFrame to get points where the displacement norm exceeds the threshold
    displacement_norm_over_threshold = disp_df[disp_df["Displacement Norm"] > threshold]

    # Extract the Cartesian coordinates of the points that are over the threshold
    points = displacement_norm_over_threshold[
        [
            "X sites in cartesian coordinate (defect)",
            "Y sites in cartesian coordinate (defect)",
            "Z sites in cartesian coordinate (defect)",
        ]
    ].to_numpy()

    # Only proceed if there are at least 10 points over the threshold
    if points.shape[0] >= 10:
        try:
            # Try to fit a minimum volume ellipsoid to the points
            (ellipsoid_center, ellipsoid_radii, ellipsoid_rotation) = _get_minimum_volume_ellipsoid(points)

            # If ellipsoid plotting is enabled, plot the ellipsoid with the given lattice matrix
            if plot_ellipsoid:
                lattice_matrix = bulk_sc.as_dict()["lattice"]["matrix"]
                if use_plotly:
                    _plotly_plot_ellipsoid(
                        ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix
                    )
                else:
                    _mpl_plot_ellipsoid(
                        ellipsoid_center, ellipsoid_radii, ellipsoid_rotation, points, lattice_matrix
                    )

            # If anisotropy plotting is enabled, plot the ellipsoid's radii anisotropy
            if plot_anisotropy:
                if use_plotly:
                    _plotly_plot_anisotropy(ellipsoid_radii, disp_df, threshold)
                else:
                    _mpl_plot_anisotropy(ellipsoid_radii, disp_df, threshold)

            # Return the ellipsoid's center, radii, and rotation matrix
            return (ellipsoid_center, ellipsoid_radii, ellipsoid_rotation)

        except np.linalg.LinAlgError:
            # Handle the case where the matrix is singular and fitting fails
            print("The matrix is singular and the system has no unique solution.")
            ellipsoid_center = None
            ellipsoid_radii = None
            ellipsoid_rotation = None
            return (ellipsoid_center, ellipsoid_radii, ellipsoid_rotation)
    else:
        # If there aren't enough points, suggest using a smaller quantile and return None values
        print("Use smaller quantile.")
        ellipsoid_center = None
        ellipsoid_radii = None
        ellipsoid_rotation = None
        return (ellipsoid_center, ellipsoid_radii, ellipsoid_rotation)


def _get_bulk_struct_with_defect(defect_entry) -> tuple:
    """
    Returns structures for bulk and defect supercells with the same number of
    sites and species, to be used for site matching. If Vacancy, adds
    (unrelaxed) site to defect structure. If Interstitial, adds relaxed site to
    bulk structure. If Substitution, replaces (unrelaxed) defect site in bulk
    structure.

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
