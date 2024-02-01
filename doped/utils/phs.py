"""
Helper functions for setting up PHS analysis.
"""
from typing import Any, List

import numpy as np
from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, OrbitalInfo, PerfectBandEdgeState
from pydefect.analyzer.eigenvalue_plotter import EigenvalueMplPlotter
from pydefect.analyzer.make_band_edge_states import make_band_edge_states
from pydefect.analyzer.make_defect_structure_info import MakeDefectStructureInfo
from pydefect.cli.vasp.make_band_edge_orbital_infos import calc_orbital_character, calc_participation_ratio
from pydefect.cli.vasp.make_perfect_band_edge_state import get_edge_info
from pydefect.defaults import defaults
from pymatgen.io.vasp.outputs import Vasprun
from vise.analyzer.vasp.band_edge_properties import VaspBandEdgeProperties


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
    band_edge_prop = VaspBandEdgeProperties(bulk_vr, bulk_outcar)
    orbs, s = bulk_vr.projected_eigenvalues, bulk_vr.final_structure
    vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, bulk_vr)
    cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, bulk_vr)

    dsinfo = MakeDefectStructureInfo(
        DefectParser.defect_entry.bulk_supercell,
        DefectParser.defect_entry.defect_supercell,
        DefectParser.defect_entry.defect_supercell,
        symprec=0.1,
        dist_tol=0.05,
        neighbor_cutoff_factor=2,
    )

    band_orb = make_band_edge_orbital_infos(
        defect_vr,
        vbm_info.orbital_info.energy,
        cbm_info.orbital_info.energy,
        eigval_shift=-vbm_info.orbital_info.energy,
        str_info=dsinfo.defect_structure_info,
    )

    return band_orb, vbm_info, cbm_info


def get_phs_and_eigenvalue(DefectEntry, save_plot: bool = False):
    """
    Get PHS info and eigenvalue plot for a given DefectEntry.

    Args:
        DefectEntry: DefectEntry object
        save_plot (bool):
            Whether to save the eigenvalue plot.
            Default is False.

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

    plotter = EigenvalueMplPlotter(
        title="title",
        band_edge_orb_infos=band_orb,
        supercell_vbm=vbm,
        supercell_cbm=cbm,
        y_range=[vbm - 3, cbm + 3],
    )

    plotter.construct_plot()

    if save_plot:
        plotter.plt.savefig(DefectEntry.name + "eigenvalue.png", bbox_inches="tight")

    return bes
