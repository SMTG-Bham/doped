"""
Helper functions for setting up PHS analysis.

Contains modified versions of functions from pydefect
https://github.com/kumagai-group/pydefect
and vise
https://github.com/kumagai-group/vise, to avoid the user requiring additional files i.e. PROCAR.
"""
import logging
import os
import warnings
from importlib.metadata import version
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from vise import user_settings
# suppress pydefect INFO messages
user_settings.logger.setLevel(logging.CRITICAL)

from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, OrbitalInfo, PerfectBandEdgeState
from pydefect.analyzer.eigenvalue_plotter import EigenvalueMplPlotter
from pydefect.analyzer.make_band_edge_states import make_band_edge_states
from pydefect.analyzer.make_defect_structure_info import MakeDefectStructureInfo
from pydefect.cli.vasp.make_band_edge_orbital_infos import calc_orbital_character, calc_participation_ratio
from pydefect.cli.vasp.make_perfect_band_edge_state import get_edge_info
from pydefect.defaults import defaults
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun
from vise.analyzer.vasp.band_edge_properties import VaspBandEdgeProperties

from doped.utils.plotting import _get_backend, format_defect_name

# TODO: Update linting, update tests, and update tips and tricks

def make_band_edge_orbital_infos(
    vasprun: Vasprun, vbm: float, cbm: float, str_info, eigval_shift: float = 0.0
):
    """
    Make BandEdgeOrbitalInfos from vasprun.xml.

    Args:
        vasprun: Vasprun object
        vbm: vbm_info
        cbm: cbm_info
        str_info: DefectStructureInfo
        eigval_shift (float): shift eigenvalues by a value

    Returns:
        BandEdgeOrbitalInfos
    """
    eigval_range = defaults.eigval_range
    kpt_coords = [tuple(coord) for coord in vasprun.actual_kpoints]
    max_energy_by_spin, min_energy_by_spin = [], []
    neighbors = str_info.neighbor_atom_indices

    for e in vasprun.eigenvalues.values():
        max_energy_by_spin.append(np.amax(e[:, :, 0], axis=0))
        min_energy_by_spin.append(np.amin(e[:, :, 0], axis=0))

    max_energy_by_band = np.amax(np.vstack(max_energy_by_spin), axis=0)
    min_energy_by_band = np.amin(np.vstack(min_energy_by_spin), axis=0)

    lower_idx = np.argwhere(max_energy_by_band > vbm - eigval_range)[0][0]
    upper_idx = np.argwhere(min_energy_by_band < cbm + eigval_range)[-1][-1]

    orbs, s = vasprun.projected_eigenvalues, vasprun.final_structure
    orb_infos: List[Any] = []
    for spin, eigvals in vasprun.eigenvalues.items():
        orb_infos.append([])
        for k_idx in range(len(kpt_coords)):
            orb_infos[-1].append([])
            for b_idx in range(lower_idx, upper_idx + 1):
                e, occ = eigvals[k_idx, b_idx, :]
                orbitals = calc_orbital_character(orbs, s, spin, k_idx, b_idx)
                if neighbors:
                    p_ratio = calc_participation_ratio(orbs, spin, k_idx, b_idx, neighbors)
                else:
                    p_ratio = None
                orb_infos[-1][-1].append(OrbitalInfo(e, orbitals, occ, p_ratio))

    return BandEdgeOrbitalInfos(
        orbital_infos=orb_infos,
        kpt_coords=kpt_coords,
        kpt_weights=vasprun.actual_kpoints_weights,
        lowest_band_index=int(lower_idx),
        fermi_level=vasprun.efermi,
        eigval_shift=eigval_shift,
    )


def get_band_edge_info(DefectParser, bulk_vr, bulk_outcar, defect_vr):
    """
    Load metadata required for performing phs identification.

    Args:
        DefectParser: DefectParser object
        bulk_vr: Vasprun object for bulk
        bulk_outcar: Outcar object for bulk
        defect_vr: Vasprun object for defect
    Returns:
        pydefect EdgeInfo class
    """
    # Change this to a warning...
    try:
        band_edge_prop = VaspBandEdgeProperties(bulk_vr, bulk_outcar)
        if defect_vr.parameters.get("LNONCOLLINEAR") is True:
            assert band_edge_prop._ho_band_index(Spin.up) == int(bulk_vr.parameters.get("NELECT")) - 1
    except AssertionError:
        v_vise = version("vise")
        if v_vise <= "0.8.1":
            warnings.warn(
                f"You have version {v_vise} of the package `vise`,"
                f" which does not allow the parsing of non-collinear calculations."
                f" You can install the updated version of `vise` from the GitHub repo for this"
                f" functionality. Attempting to load the PHS data has been automatically skipped"
            )
            return None, None, None

    orbs, s = bulk_vr.projected_eigenvalues, bulk_vr.final_structure
    vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, bulk_vr)
    cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, bulk_vr)

    # Using default values pydefect
    dsinfo = MakeDefectStructureInfo(
        DefectParser.defect_entry.bulk_supercell,
        DefectParser.defect_entry.defect_supercell,
        DefectParser.defect_entry.defect_supercell,
        symprec=0.1,
        dist_tol=1.0,
        neighbor_cutoff_factor=1.3,
    )

    band_orb = make_band_edge_orbital_infos(
        defect_vr,
        vbm_info.orbital_info.energy,
        cbm_info.orbital_info.energy,
        eigval_shift=-vbm_info.orbital_info.energy,
        str_info=dsinfo.defect_structure_info,
    )

    return band_orb, vbm_info, cbm_info


def get_phs_and_eigenvalue(DefectEntry, filename: Optional[str] = None, ks_labels: bool = False):
    """
    Get PHS info and eigenvalue plot for a given DefectEntry.

    Args:
        DefectEntry: DefectEntry object
        filename (str):
            Filename to save the Kumagai site potential plots to.
            If None, plots are not saved.
        ks_labels (bool):
            Add the band index to the KS levels.

    Returns:
        pydefect PerfectBandEdgeState class
    """
    band_orb = DefectEntry.calculation_metadata["phs_data"]["band_orb"]
    vbm_info = DefectEntry.calculation_metadata["phs_data"]["vbm_info"]
    cbm_info = DefectEntry.calculation_metadata["phs_data"]["cbm_info"]

    perfect = PerfectBandEdgeState(vbm_info, cbm_info)
    bes = make_band_edge_states(band_orb, perfect)

    vbm = vbm_info.orbital_info.energy + band_orb.eigval_shift
    cbm = cbm_info.orbital_info.energy + band_orb.eigval_shift

    style_file = f"{os.path.dirname(__file__)}/displacement.mplstyle"
    plt.style.use(style_file)

    emp = EigenvalueMplPlotter(
        title="Eigenvalues",
        band_edge_orb_infos=band_orb,
        supercell_vbm=vbm,
        supercell_cbm=cbm,
        y_range=[vbm - 3, cbm + 3],
    )
    f"{format_defect_name(DefectEntry.name, False)}+ eigenvalues"

    with plt.style.context(style_file):
        plt.rcParams["axes.titlesize"] = 12
        plt.rc('axes', unicode_minus=False)

        # plt.close("all")  # close any previous figures
        emp.construct_plot()
        partial = None

        # Change colors to match sumo and doped conventions
        for a in range(len(emp.axs)):
            for i in range(len(emp.axs[a].get_children())):
                if hasattr(emp.axs[a].get_children()[i], "get_facecolor"):
                    if np.array_equal(
                        emp.axs[a].get_children()[i].get_facecolor(), [[1, 0, 0, 1]]
                    ):  # Check for red color
                        emp.axs[a].get_children()[i].set_facecolor((0.22, 0.325, 0.643))
                        emp.axs[a].get_children()[i].set_edgecolor((0.22, 0.325, 0.643))
                    elif np.array_equal(emp.axs[a].get_children()[i].get_facecolor(), [[0, 0, 1, 1]]):
                        emp.axs[a].get_children()[i].set_facecolor((0.98, 0.639, 0.086))
                        emp.axs[a].get_children()[i].set_edgecolor((0.98, 0.639, 0.086))
                    elif np.array_equal(emp.axs[a].get_children()[i].get_facecolor(), [[0, 0.5, 0, 1]]):
                        partial = True

        if ks_labels is True:
            for axes in emp.axs:
                annotations = [child for child in axes.get_children() if isinstance(child, plt.Annotation)]
                for annotation in annotations:
                    if annotation.get_position()[0] > 1:
                        annotation.remove()
        else:
            for axes in emp.axs:
                annotations = [child for child in axes.get_children() if isinstance(child, plt.Annotation)]
                for annotation in annotations:
                    if annotation:
                        annotation.remove()

        if len(emp.axs) > 1:
            emp.axs[0].set_title("spin down")
            emp.axs[1].set_title("spin up")
        else:
            emp.axs[0].set_title("KS levels")

        ymin, ymax = 0, 0
        for spin in emp._energies_and_occupations:
            for kpoint in spin:
                if ymin > min(x[0] for x in kpoint):
                    ymin = min(x[0] for x in kpoint)
                if ymax < max(x[0] for x in kpoint):
                    ymax = max(x[0] for x in kpoint)

        gamma_check = "\N{GREEK CAPITAL LETTER GAMMA}"
        labels = emp.axs[0].get_xticklabels()
        # Replace the label containing 'gamma' (γ) with the word 'gamma'
        labels = [label.get_text() for label in labels]
        # Replace the label containing 'gamma' (γ) with the word 'gamma'
        for i, label in enumerate(labels):
            if gamma_check in label:  # Check if the label contains 'gamma'
                labels[i] = r'$\Gamma$'  # Replace the label
        emp.axs[0].set_xticklabels(labels)

        if len(emp.axs) > 1:
            labels = emp.axs[1].get_xticklabels()
            # Replace the label containing 'gamma' (γ) with the word 'gamma'
            labels = [label.get_text() for label in labels]
            # Replace the label containing 'gamma' (γ) with the word 'gamma'
            for i, label in enumerate(labels):
                if gamma_check in label:  # Check if the label contains 'gamma'
                    labels[i] = r'$\Gamma$'  # Replace the label
            emp.axs[1].set_xticklabels(labels)

        fig = emp.plt.gcf()
        ax = fig.gca()

        ax.set_ylim([ymin - 0.25, ymax + 0.75])
        # add a point at 0,-5 with the color range and label unoccopied states
        ax.scatter(0, -5, label="Occupied", color=(0.98, 0.639, 0.086))
        ax.scatter(0, -5, label="Unoccupied", color=(0.22, 0.325, 0.643))
        if partial:
            ax.scatter(0, -5, label="Partially Occupied", color=(0, 0.5, 0))
        ax.axhline(-5, 0, 1, color="black", linewidth=0.5, linestyle="-.", label="Band edges")
        ax.legend(loc="upper right", fontsize=7)

    if filename:
        emp.plt.savefig(filename, bbox_inches="tight", transparent=True, backend=_get_backend(filename))

    return bes, fig
