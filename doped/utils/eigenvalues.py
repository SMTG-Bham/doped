"""
Helper functions for setting up PHS analysis.

Contains modified versions of functions from pydefect (https://github.com/kumagai-group/pydefect)
and vise (https://github.com/kumagai-group/vise), to avoid requiring additional files (i.e. ``PROCAR``s).
"""

import logging
import os
from collections import defaultdict
from importlib.metadata import version
from itertools import zip_longest
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# ruff: noqa: E402
from vise import user_settings

user_settings.logger.setLevel(logging.CRITICAL)

import pydefect.analyzer.make_band_edge_states
import pydefect.cli.vasp.make_band_edge_orbital_infos as make_bes
from easyunfold.procar import Procar as ProcarEasyunfold
from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, OrbitalInfo, PerfectBandEdgeState
from pydefect.analyzer.eigenvalue_plotter import EigenvalueMplPlotter
from pydefect.analyzer.make_band_edge_states import make_band_edge_states
from pydefect.analyzer.make_defect_structure_info import MakeDefectStructureInfo
from pydefect.cli.vasp.make_perfect_band_edge_state import (
    get_edge_info,
    make_perfect_band_edge_state_from_vasp,
)
from pydefect.defaults import defaults
from pydefect.util.structure_tools import Coordination, Distances
from pymatgen.core import Element, Species
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from shakenbreak.plotting import _install_custom_font
from vise.analyzer.vasp.band_edge_properties import VaspBandEdgeProperties

from doped.utils.plotting import _get_backend


def _coordination(self, include_on_site=True, cutoff_factor=None) -> "Coordination":
    cutoff_factor = cutoff_factor or defaults.cutoff_distance_factor
    cutoff = self.shortest_distance * cutoff_factor
    elements = [element.specie.name for element in self.structure]
    e_d = zip(elements, self.distances(remove_self=False))

    unsorted_distances = defaultdict(list)
    neighboring_atom_indices = []
    for i, (element, distance) in enumerate(e_d):
        if distance < cutoff and include_on_site:
            unsorted_distances[element].append(round(distance, 2))
            neighboring_atom_indices.append(i)

    distance_dict = {element: sorted(distances) for element, distances in unsorted_distances.items()}
    return Coordination(distance_dict, round(cutoff, 3), neighboring_atom_indices)


def _distances(self, remove_self=True, specie=None) -> list[float]:
    result = []
    lattice = self.structure.lattice
    if isinstance(specie, Element):
        el = specie.symbol
    elif specie is None:
        el = None
    elif isinstance(specie, Species):
        el = specie.element
    for site in self.structure:
        site_specie = site.specie.element if isinstance(site.specie, Species) else site.specie
        if el and Element(el) != site_specie:
            result.append(float("inf"))
            continue
        distance, _ = lattice.get_distance_and_image(site.frac_coords, self.coord)
        if remove_self and distance < 1e-5:
            continue
        result.append(distance)
    return result


def _make_band_edge_orbital_infos_vr(
    defect_vr: Vasprun, vbm: float, cbm: float, str_info, eigval_shift: float = 0.0
):
    """
    Make ``BandEdgeOrbitalInfos`` from ``vasprun.xml``.

    Modified from ``pydefect`` to use projected
    orbitals stored in ``vasprun.xml``.

    Args:
        defect_vr: defect ``Vasprun`` object
        vbm: VBM eigenvalue in eV
        cbm: CBM eigenvalue in eV
        str_info: ``pydefect`` ``DefectStructureInfo``
        eigval_shift (float): Shift eigenvalues by E-E_VBM to set VBM at 0 eV

    Returns:
        ``BandEdgeOrbitalInfos `` object
    """
    eigval_range = defaults.eigval_range
    kpt_coords = [tuple(coord) for coord in defect_vr.actual_kpoints]
    max_energy_by_spin, min_energy_by_spin = [], []
    neighbors = str_info.neighbor_atom_indices

    for e in defect_vr.eigenvalues.values():
        max_energy_by_spin.append(np.amax(e[:, :, 0], axis=0))
        min_energy_by_spin.append(np.amin(e[:, :, 0], axis=0))

    max_energy_by_band = np.amax(np.vstack(max_energy_by_spin), axis=0)
    min_energy_by_band = np.amin(np.vstack(min_energy_by_spin), axis=0)

    lower_idx = np.argwhere(max_energy_by_band > vbm - eigval_range)[0][0]
    upper_idx = np.argwhere(min_energy_by_band < cbm + eigval_range)[-1][-1]

    orbs, s = defect_vr.projected_eigenvalues, defect_vr.final_structure
    orb_infos: list[Any] = []
    for spin, eigvals in defect_vr.eigenvalues.items():
        orb_infos.append([])
        for k_idx in range(len(kpt_coords)):
            orb_infos[-1].append([])
            for b_idx in range(lower_idx, upper_idx + 1):
                e, occ = eigvals[k_idx, b_idx, :]
                orbitals = make_bes.calc_orbital_character(orbs, s, spin, k_idx, b_idx)
                if neighbors:
                    p_ratio = make_bes.calc_participation_ratio(orbs, spin, k_idx, b_idx, neighbors)
                else:
                    p_ratio = None
                orb_infos[-1][-1].append(OrbitalInfo(e, orbitals, occ, p_ratio))

    return BandEdgeOrbitalInfos(
        orbital_infos=orb_infos,
        kpt_coords=kpt_coords,
        kpt_weights=defect_vr.actual_kpoints_weights,
        lowest_band_index=int(lower_idx),
        fermi_level=defect_vr.efermi,
        eigval_shift=eigval_shift,
    )


def get_band_edge_info(
    bulk_vr: Vasprun,
    bulk_outcar: Outcar,
    defect_vr: Vasprun,
    bulk_procar: Optional[ProcarEasyunfold] = None,
    defect_procar: Optional[ProcarEasyunfold] = None,
):
    """
    Load metadata required for performing eigenvalue & orbital analysis.

    See https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states

    Args:
        bulk_vr (Vasprun):
            ``Vasprun`` object of the bulk supercell calculation
        bulk_outcar (Outcar):
            ``Outcar`` object of
        defect_vr (Vasprun):
            ``Vasprun`` object of the defect supercell calculation
        bulk_procar (Procar):
            ``Procar`` object of the bulk supercell calculation
        defect_procar (Procar):
            ``Procar`` object of the defect supercell calculation
    Returns:
        ``pydefect`` ``EdgeInfo`` class
    """
    # Check in the correct version of vise installed if a non-collinear calculation is parsed.
    # TODO: Remove this check when ``vise 0.8.2`` is released on PyPi.
    try:
        band_edge_prop = VaspBandEdgeProperties(bulk_vr, bulk_outcar)
        if bulk_vr.parameters.get("LNONCOLLINEAR") is True:
            assert band_edge_prop._ho_band_index(Spin.up) == int(bulk_vr.parameters.get("NELECT")) - 1
        if bulk_procar is not None:
            pbes = make_perfect_band_edge_state_from_vasp(bulk_procar, bulk_vr, bulk_outcar)
    except AssertionError as exc:
        v_vise = version("vise")
        if v_vise <= "0.8.1":
            raise RuntimeError(
                f"You have version {v_vise} of the package `vise`, which does not allow the parsing of "
                f"non-collinear (SOC) calculations. You can install the updated version of `vise` from "
                f"the GitHub repo for this functionality."
            ) from exc

        raise exc

    _orig_method_coor = Distances.coordination
    Distances.coordination = _coordination
    _orig_method_dist = Distances.distances
    Distances.distances = _distances

    dsinfo = MakeDefectStructureInfo(  # Using default values suggested by pydefect
        bulk_vr.final_structure,
        defect_vr.final_structure,
        defect_vr.final_structure,
        symprec=0.1,
        dist_tol=1.0,
        neighbor_cutoff_factor=1.3,  # Neighbors are sites within min_dist * neighbor_cutoff_factor
    )

    # Undo monkey patch in case used in other parts of the code:
    Distances.coordination = _orig_method_coor
    Distances.distances = _orig_method_dist

    if bulk_procar:
        vbm_info, cbm_info = pbes.vbm_info, pbes.cbm_info

        band_orb = make_bes.make_band_edge_orbital_infos(
            defect_procar,
            defect_vr,
            vbm_info.orbital_info.energy,
            cbm_info.orbital_info.energy,
            eigval_shift=-vbm_info.orbital_info.energy,
            str_info=dsinfo.defect_structure_info,
        )

    else:
        orbs, s = bulk_vr.projected_eigenvalues, bulk_vr.final_structure
        vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, bulk_vr)
        cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, bulk_vr)

        band_orb = _make_band_edge_orbital_infos_vr(
            defect_vr,
            vbm_info.orbital_info.energy,
            cbm_info.orbital_info.energy,
            eigval_shift=-vbm_info.orbital_info.energy,
            str_info=dsinfo.defect_structure_info,
        )

    return band_orb, vbm_info, cbm_info


def get_eigenvalue_analysis(
    DefectEntry,
    filename: Optional[str] = None,
    plot: bool = True,
    ks_labels: bool = False,
    style_file: Optional[str] = None,
):
    """
    Get eigenvalue & orbital info (with automated classification of PHS states)
    with corresponding single-particle eigenvalues plot (if ``plot = True``;
    default) for a given ``DefectEntry``.

    Args:
        DefectEntry: ``DefectEntry`` object
        filename (str):
            Filename to save the Kumagai site potential plots to.
            If None (default), plots are not saved.
        ks_labels (bool):
            Add the band index to the KS levels.
            (Default: False)
        plot (bool):
            Whether to plot the single-particle eigenvalues.
            (Default: True)
        style_file (str):
            Path to a ``mplstyle`` file to use for the plot. If None
            (default), uses the ``doped`` displacement plot style
            (``doped/utils/displacement.mplstyle``).

    Returns:
        ``pydefect`` ``PerfectBandEdgeState`` class
    """
    band_orb = DefectEntry.calculation_metadata["eigenvalue_data"]["band_orb"]
    vbm_info = DefectEntry.calculation_metadata["eigenvalue_data"]["vbm_info"]
    cbm_info = DefectEntry.calculation_metadata["eigenvalue_data"]["cbm_info"]

    # Ensures consistent number of sig. fig. so test work 100% of the time
    def _orbital_diff(orbital_1: dict, orbital_2: dict) -> float:
        element_set = set(list(orbital_1.keys()) + list(orbital_2.keys()))
        orb_1, orb_2 = defaultdict(list, orbital_1), defaultdict(list, orbital_2)
        result = 0
        for e in element_set:
            result += sum(abs(i - j) for i, j in zip_longest(orb_1[e], orb_2[e], fillvalue=0))
        return round(result, 3)

    pydefect.analyzer.make_band_edge_states.orbital_diff = _orbital_diff

    perfect = PerfectBandEdgeState(vbm_info, cbm_info)
    bes = make_band_edge_states(band_orb, perfect)

    if not plot:
        return bes

    vbm = vbm_info.orbital_info.energy + band_orb.eigval_shift
    cbm = cbm_info.orbital_info.energy + band_orb.eigval_shift

    _install_custom_font()  # in case not installed already
    style_file = style_file or f"{os.path.dirname(__file__)}/displacement.mplstyle"
    plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter

    emp = EigenvalueMplPlotter(
        title="Eigenvalues",
        band_edge_orb_infos=band_orb,
        supercell_vbm=vbm,
        supercell_cbm=cbm,
        y_range=[vbm - 3, cbm + 3],
    )

    with plt.style.context(style_file):
        plt.rcParams["axes.titlesize"] = 12
        plt.rc("axes", unicode_minus=False)

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

        for axes in emp.axs:
            annotations = [child for child in axes.get_children() if isinstance(child, plt.Annotation)]
            for annotation in annotations:
                if (ks_labels and annotation.get_position()[0] > 1) or not ks_labels:
                    annotation.remove()

        if len(emp.axs) > 1:
            emp.axs[0].set_title("Spin Down")
            emp.axs[1].set_title("Spin Up")
        else:
            emp.axs[0].set_title("KS levels")

        ymin, ymax = 0, 0
        for spin in emp._energies_and_occupations:
            for kpoint in spin:
                ymin = min(ymin, *(x[0] for x in kpoint))
                ymax = max(ymax, *(x[0] for x in kpoint))

        gamma_check = "\N{GREEK CAPITAL LETTER GAMMA}"
        for ax in emp.axs:
            labels = ax.get_xticklabels()
            labels = [label.get_text() for label in labels]
            for i, label in enumerate(labels):
                if gamma_check in label:
                    labels[i] = r"$\Gamma$"
            ax.set_xticklabels(labels)

        fig = emp.plt.gcf()
        ax = fig.gca()

        ax.set_ylim([ymin - 0.25, ymax + 0.75])
        # add a point at 0,-5 with the color range and label unoccopied states
        ax.scatter(0, -5, label="Unoccupied", color=(0.98, 0.639, 0.086))
        ax.scatter(0, -5, label="Occupied", color=(0.22, 0.325, 0.643))
        if partial:
            ax.scatter(0, -5, label="Partially Occupied", color=(0, 0.5, 0))
        ax.axhline(-5, 0, 1, color="black", linewidth=0.5, linestyle="-.", label="Band Edges")
        ax.legend(loc="upper right", fontsize=7)

    if filename:
        emp.plt.savefig(filename, bbox_inches="tight", transparent=True, backend=_get_backend(filename))

    return bes, fig
