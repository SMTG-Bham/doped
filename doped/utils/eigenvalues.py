r"""
Helper functions for setting up PHS analysis.

Contains modified versions of functions from pydefect (https://github.com/kumagai-group/pydefect)
and vise (https://github.com/kumagai-group/vise), to avoid requiring additional files (i.e. ``PROCAR``\s).
"""

# suppress pydefect INFO messages
import contextlib
import logging
import os
import warnings
from collections import defaultdict
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.structure import PeriodicSite
from pymatgen.electronic_structure.core import Spin
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Procar, Vasprun
from pymatgen.util.typing import PathLike

from doped.analysis import defect_from_structures
from doped.core import DefectEntry
from doped.utils.parsing import get_magnetization_from_vasprun, get_nelect_from_vasprun, get_procar
from doped.utils.plotting import _get_backend

orig_simplefilter = warnings.simplefilter
warnings.simplefilter = lambda *args, **kwargs: None  # monkey-patch to avoid vise warning suppression

if TYPE_CHECKING:
    from easyunfold.procar import Procar as EasyunfoldProcar

try:
    from vise import user_settings

    user_settings.logger.setLevel(logging.CRITICAL)
    import pydefect.analyzer.make_band_edge_states
    import pydefect.cli.vasp.make_band_edge_orbital_infos as make_bes
    from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, OrbitalInfo, PerfectBandEdgeState
    from pydefect.analyzer.eigenvalue_plotter import EigenvalueMplPlotter
    from pydefect.analyzer.make_band_edge_states import make_band_edge_states
    from pydefect.cli.vasp.make_perfect_band_edge_state import get_edge_info
    from pydefect.defaults import defaults
    from vise.analyzer.vasp.band_edge_properties import BandEdgeProperties, eigenvalues_from_vasprun

except ImportError as exc:
    raise ImportError(
        "To perform eigenvalue & orbital analysis, you need to install pydefect. "
        "You can do this by running `pip install pydefect`."
    ) from exc

warnings.simplefilter = orig_simplefilter  # reset to original


def band_edge_properties_from_vasprun(
    vasprun: Vasprun, integer_criterion: float = 0.1
) -> BandEdgeProperties:
    """
    Create a ``pydefect`` ``BandEdgeProperties`` object from a ``Vasprun``
    object.

    Args:
        vasprun (Vasprun): ``Vasprun`` object.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or
            fully occupied (> 1 - ``integer_criterion``).
            Default is 0.1.

    Returns:
        ``BandEdgeProperties`` object.
    """
    band_edge_prop = BandEdgeProperties(
        eigenvalues=eigenvalues_from_vasprun(vasprun),
        nelect=get_nelect_from_vasprun(vasprun),
        magnetization=get_magnetization_from_vasprun(vasprun),
        kpoint_coords=vasprun.actual_kpoints,
        integer_criterion=integer_criterion,
        is_non_collinear=vasprun.parameters.get("LNONCOLLINEAR", False),
    )
    band_edge_prop.structure = vasprun.final_structure
    return band_edge_prop


def make_perfect_band_edge_state_from_vasp(
    vasprun: Vasprun, procar: Procar, integer_criterion: float = 0.1
) -> PerfectBandEdgeState:
    """
    Create a ``pydefect`` ``PerfectBandEdgeState`` object from just a
    ``Vasprun`` and ``Procar`` object, without the need for the ``Outcar``
    input (as in ``pydefect``).

    Args:
        vasprun (Vasprun): ``Vasprun`` object.
        procar (Procar): ``Procar`` object.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or
            fully occupied (> 1 - ``integer_criterion``).
            Default is 0.1.

    Returns:
        ``PerfectBandEdgeState`` object.
    """
    band_edge_prop = band_edge_properties_from_vasprun(vasprun, integer_criterion)
    orbs, s = procar.data, vasprun.final_structure
    vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, vasprun)
    cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, vasprun)
    return PerfectBandEdgeState(vbm_info, cbm_info)


def make_band_edge_orbital_infos(
    defect_vr: Vasprun,
    vbm: float,
    cbm: float,
    eigval_shift: float = 0.0,
    neighbor_indices: Optional[list[int]] = None,
    defect_procar: Optional[Union["EasyunfoldProcar", Procar]] = None,
):
    r"""
    Make ``BandEdgeOrbitalInfos`` from a ``Vasprun`` object.

    Modified from ``pydefect`` to use projected orbitals
    stored in the ``Vasprun`` object.

    Args:
        defect_vr (Vasprun): Defect ``Vasprun`` object.
        vbm (float): VBM eigenvalue in eV.
        cbm (float): CBM eigenvalue in eV.
        eigval_shift (float):
            Shift eigenvalues down by this value (to set VBM at 0 eV).
            Default is 0.0.
        neighbor_indices (list[int]):
            Indices of neighboring atoms to the defect site, for localisation analysis.
            Default is ``None``\.
        defect_procar (EasyunfoldProcar, Procar):
            ``EasyunfoldProcar`` or ``Procar`` object, for the defect supercell,
            if projected eigenvalue/orbitals data is not provided in ``defect_vr``\.

    Returns:
        ``BandEdgeOrbitalInfos`` object.
    """
    eigval_range = defaults.eigval_range
    kpt_coords = [tuple(coord) for coord in defect_vr.actual_kpoints]
    max_energy_by_spin, min_energy_by_spin = [], []

    for e in defect_vr.eigenvalues.values():
        max_energy_by_spin.append(np.amax(e[:, :, 0], axis=0))
        min_energy_by_spin.append(np.amin(e[:, :, 0], axis=0))

    max_energy_by_band = np.amax(np.vstack(max_energy_by_spin), axis=0)
    min_energy_by_band = np.amin(np.vstack(min_energy_by_spin), axis=0)

    lower_idx = np.argwhere(max_energy_by_band > vbm - eigval_range)[0][0]
    upper_idx = np.argwhere(min_energy_by_band < cbm + eigval_range)[-1][-1]

    orbs = defect_vr.projected_eigenvalues if defect_procar is None else defect_procar.data
    s = defect_vr.final_structure
    orb_infos: list[Any] = []
    for spin, eigvals in defect_vr.eigenvalues.items():
        orb_infos.append([])
        for k_idx in range(len(kpt_coords)):
            orb_infos[-1].append([])
            for b_idx in range(lower_idx, upper_idx + 1):
                e, occ = eigvals[k_idx, b_idx, :]
                orbitals = make_bes.calc_orbital_character(orbs, s, spin, k_idx, b_idx)
                if neighbor_indices:
                    p_ratio = make_bes.calc_participation_ratio(orbs, spin, k_idx, b_idx, neighbor_indices)
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


def _parse_procar(procar: Optional[Union[PathLike, "EasyunfoldProcar", Procar]] = None):
    """
    Parse a ``procar`` input to a ``Procar`` object in the correct format.

    Args:
        procar (PathLike, EasyunfoldProcar, Procar):
            Either a path to the ``VASP`` ``PROCAR``` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/``pymatgen``
            ``Procar`` object.
    """
    if not hasattr(procar, "data"):  # not a parsed Procar object
        if procar and hasattr(procar, "proj_data") and not isinstance(procar, (PathLike, Procar)):
            if procar._is_soc:
                procar.data = {Spin.up: procar.proj_data[0]}
            else:
                procar.data = {Spin.up: procar.proj_data[0], Spin.down: procar.proj_data[1]}
            del procar.proj_data

        elif isinstance(procar, PathLike):  # path to PROCAR file
            procar = get_procar(procar)

    return procar


def get_band_edge_info(
    bulk_vr: Vasprun,
    defect_vr: Vasprun,
    bulk_procar: Optional[Union[PathLike, "EasyunfoldProcar", Procar]] = None,
    defect_procar: Optional[Union[PathLike, "EasyunfoldProcar", Procar]] = None,
    defect_supercell_site: Optional[PeriodicSite] = None,
    neighbor_cutoff_factor: float = 1.3,
):
    """
    Generate metadata required for performing eigenvalue & orbital analysis,
    specifically ``pydefect`` ``BandEdgeOrbitalInfos``, and ``EdgeInfo``
    objects for the bulk VBM and CBM.

    See https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states.

    Args:
        bulk_vr (Vasprun):
            ``Vasprun`` object of the bulk supercell calculation.
            If ``bulk_procar`` is not provided, then this must have the
            ``projected_eigenvalues`` attribute (i.e. from a calculation
            with ``LORBIT > 10`` in the ``INCAR`` and parsed with
            ``parse_projected_eigen = True``).
        defect_vr (Vasprun):
            ``Vasprun`` object of the defect supercell calculation.
            If ``defect_procar`` is not provided, then this must have the
            ``projected_eigenvalues`` attribute (i.e. from a calculation
            with ``LORBIT > 10`` in the ``INCAR`` and parsed with
            ``parse_projected_eigen = True``).
        bulk_procar (PathLike, EasyunfoldProcar, Procar):
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the bulk supercell
            calculation. Not required if the supplied ``bulk_vr`` was
            parsed with ``parse_projected_eigen = True``.
            Default is ``None``.
        defect_procar (PathLike, EasyunfoldProcar, Procar):
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the defect supercell
            calculation. Not required if the supplied ``bulk_vr`` was
            parsed with ``parse_projected_eigen = True``.
            Default is ``None``.
        defect_supercell_site (PeriodicSite):
            ``PeriodicSite`` object of the defect site in the defect
            supercell, from which the defect neighbours are determined
            for localisation analysis. If ``None`` (default), then the
            defect site is determined automatically from the defect
            and bulk supercell structures.
        neighbor_cutoff_factor (float):
            Sites within ``min_distance * neighbor_cutoff_factor`` of
            the defect site in the `relaxed` defect supercell are
            considered neighbors for localisation analysis, where
            ``min_distance`` is the minimum distance between sites in
            the defect supercell. Default is 1.3 (matching the ``pydefect``
            default).

    Returns:
        ``pydefect`` ``BandEdgeOrbitalInfos``, and ``EdgeInfo`` objects
        for the bulk VBM and CBM.
    """
    band_edge_prop = band_edge_properties_from_vasprun(bulk_vr)

    if bulk_procar is not None:
        bulk_procar = _parse_procar(bulk_procar)
        pbes = make_perfect_band_edge_state_from_vasp(vasprun=bulk_vr, procar=bulk_procar)

    # get defect neighbour indices
    sorted_distances = np.sort(defect_vr.final_structure.distance_matrix.flatten())
    min_distance = sorted_distances[sorted_distances > 0.5][0]

    if defect_supercell_site is None:
        (
            _defect,
            defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
            defect_site_in_bulk,  # vacancy site
            _defect_site_index,
            _bulk_site_index,
            _guessed_initial_defect_structure,
            _unrelaxed_defect_structure,
            _bulk_voronoi_node_dict,
        ) = defect_from_structures(
            bulk_vr.final_structure,
            defect_vr.final_structure.copy(),
            return_all_info=True,
            oxi_state="Undetermined",
        )
        defect_supercell_site = defect_site or defect_site_in_bulk

    neighbor_indices = [
        i
        for i, site in enumerate(defect_vr.final_structure.sites)
        if defect_supercell_site.distance(site) <= min_distance * neighbor_cutoff_factor
    ]

    from pydefect.analyzer.band_edge_states import logger

    logger.setLevel(logging.CRITICAL)  # quieten unnecessary eigenvalue shift INFO message

    if bulk_procar is not None:
        vbm_info, cbm_info = pbes.vbm_info, pbes.cbm_info
    else:
        orbs, s = bulk_vr.projected_eigenvalues, bulk_vr.final_structure
        vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, bulk_vr)
        cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, bulk_vr)

    band_orb = make_band_edge_orbital_infos(
        defect_vr,
        vbm_info.orbital_info.energy,
        cbm_info.orbital_info.energy,
        eigval_shift=-vbm_info.orbital_info.energy,
        neighbor_indices=neighbor_indices,
        defect_procar=_parse_procar(defect_procar),
    )

    return band_orb, vbm_info, cbm_info


def _add_eigenvalues(
    self,
    occupied_color=(0.22, 0.325, 0.643),
    unoccupied_color=(0.98, 0.639, 0.086),
    partial_color=(0.0, 0.5, 0.0),
):
    """
    Add eigenvalues to plot.

    Refactored from implementation in ``pydefect`` to avoid calling
    ``ax.scatter`` individually many times when we have many kpoints
    and bands, which can make the plotting quite slow (>10 seconds),
    and allow setting custom colors for occupied, unoccupied, and
    partially occupied states.
    """
    for _spin_idx, (eo_by_spin, ax) in enumerate(zip(self._energies_and_occupations, self.axs)):
        kpt_indices = []
        energies = []
        color_list = []
        annotations = []
        for kpt_idx, eo_by_k_idx in enumerate(eo_by_spin):
            for band_idx, eo_by_band in enumerate(eo_by_k_idx):
                energy, occup = eo_by_band
                color_list.append(
                    occupied_color if occup > 0.9 else unoccupied_color if occup < 0.1 else partial_color
                )
                kpt_indices.append(kpt_idx)
                energies.append(energy)

                try:
                    higher_band_e = eo_by_k_idx[band_idx + 1][0]
                    lower_band_e = eo_by_k_idx[band_idx - 1][0]
                except IndexError:
                    continue

                if self._add_band_idx(energy, higher_band_e, lower_band_e):
                    annotations.append((kpt_idx + 0.05, energy, band_idx + self._lowest_band_idx + 1))

        ax.scatter(kpt_indices, energies, c=color_list, s=self._mpl_defaults.circle_size)
        for annotation in annotations:
            ax.annotate(
                annotation[2],
                (annotation[0], annotation[1]),
                va="center",
                fontsize=self._mpl_defaults.tick_label_size,
            )


EigenvalueMplPlotter._add_eigenvalues = _add_eigenvalues


def get_eigenvalue_analysis(
    defect_entry: Optional[DefectEntry] = None,
    plot: bool = True,
    filename: Optional[str] = None,
    ks_labels: bool = False,
    style_file: Optional[str] = None,
    bulk_vr: Optional[Union[PathLike, Vasprun]] = None,
    bulk_procar: Optional[Union[PathLike, "EasyunfoldProcar", Procar]] = None,
    defect_vr: Optional[Union[PathLike, Vasprun]] = None,
    defect_procar: Optional[Union[PathLike, "EasyunfoldProcar", Procar]] = None,
    force_reparse: bool = False,
    ylims: Optional[tuple[float, float]] = None,
    legend_kwargs: Optional[dict] = None,
    similar_orb_criterion: Optional[float] = None,
    similar_energy_criterion: Optional[float] = None,
):
    r"""
    Get eigenvalue & orbital info (with automated classification of PHS states)
    for the band edge and in-gap electronic states for the input defect entry /
    calculation outputs, as well as a plot of the single-particle electronic
    eigenvalues and their occupation (if ``plot=True``).

    Can be used to determine if a defect is adopting a perturbed host
    state (PHS / shallow state), see
    https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states.
    Note that the classification of electronic states as band edges or localized
    orbitals is based on the similarity of orbital projections and eigenvalues
    between the defect and bulk cell calculations (see
    ``similar_orb/energy_criterion`` argument descriptions below for more details).
    You may want to adjust the default values of these keyword arguments, as the
    defaults may not be appropriate in all cases. In particular, the P-ratio values
    can give useful insight, revealing the level of (de)localisation of the states.

    Either a ``doped`` ``DefectEntry`` object can be provided, or the required
    VASP output files/objects for the bulk and defect supercell calculations
    (``Vasprun``\s, or ``Vasprun``\s and ``Procar``\s).
    If a ``DefectEntry`` is provided but eigenvalue data has not already been
    parsed (default in ``doped`` is to parse this data with ``DefectsParser``/
    ``DefectParser``, as controlled by the ``parse_projected_eigen`` flag),
    then this function will attempt to load the eigenvalue data from either
    the input ``Vasprun``/``PROCAR`` objects or files, or from the
    ``bulk/defect_path``\s in ``defect_entry.calculation_metadata``.
    If so, will initially try to load orbital projections from ``vasprun.xml(.gz)``
    files (slightly slower but more accurate), or failing that from ``PROCAR(.gz)``
    files if present.

    This function uses code from ``pydefect``, so please cite the ``pydefect`` paper:
    "Insights into oxygen vacancies from high-throughput first-principles calculations"
    Yu Kumagai, Naoki Tsunoda, Akira Takahashi, and Fumiyasu Oba
    Phys. Rev. Materials 5, 123803 (2021) -- 10.1103/PhysRevMaterials.5.123803

    Args:
        defect_entry (DefectEntry):
            ``doped`` ``DefectEntry`` object. Default is ``None``.
        plot (bool):
            Whether to plot the single-particle eigenvalues.
            (Default: True)
        filename (str):
            Filename to save the eigenvalue plot to (if ``plot = True``).
            If ``None`` (default), plots are not saved.
        ks_labels (bool):
            Whether to add band index labels to the KS levels.
            (Default: False)
        style_file (str):
            Path to a ``mplstyle`` file to use for the plot. If None
            (default), uses the ``doped`` displacement plot style
            (``doped/utils/displacement.mplstyle``).
        bulk_vr (PathLike, Vasprun):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``vasprun.xml(.gz)`` output file
            or a ``pymatgen`` ``Vasprun`` object, for the reference bulk
            supercell calculation. If ``None`` (default), tries to load
            the ``Vasprun`` object from
            ``defect_entry.calculation_metadata["run_metadata"]["bulk_vasprun_dict"]``,
            or, failing that, from a ``vasprun.xml(.gz)`` file at
            ``defect_entry.calculation_metadata["bulk_path"]``.
        bulk_procar (PathLike, EasyunfoldProcar, Procar):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``),
            or if ``bulk_vr`` was parsed with ``parse_projected_eigen = True``.
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the reference bulk supercell
            calculation.
            If ``None`` (default), tries to load from a ``PROCAR(.gz)``
            file at ``defect_entry.calculation_metadata["bulk_path"]``.
        defect_vr (PathLike, Vasprun):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``vasprun.xml(.gz)`` output file
            or a ``pymatgen`` ``Vasprun`` object, for the defect supercell
            calculation. If ``None`` (default), tries to load the ``Vasprun``
            object from
            ``defect_entry.calculation_metadata["run_metadata"]["defect_vasprun_dict"]``,
            or, failing that, from a ``vasprun.xml(.gz)`` file at
            ``defect_entry.calculation_metadata["defect_path"]``.
        defect_procar (PathLike, EasyunfoldProcar, Procar):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``),
            or if ``defect_vr`` was parsed with ``parse_projected_eigen = True``.
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the defect supercell calculation.
            If ``None`` (default), tries to load from a ``PROCAR(.gz)``
            file at ``defect_entry.calculation_metadata["defect_path"]``.
        force_reparse (bool):
            Whether to force re-parsing of the eigenvalue data, even if
            already present in the ``calculation_metadata``.
        ylims (tuple[float, float]):
            Custom y-axis limits for the eigenvalue plot. If ``None`` (default),
            the y-axis limits are automatically set to +/-5% of the eigenvalue
            range.
        legend_kwargs (dict):
            Custom keyword arguments to pass to the ``ax.legend`` call in the
            eigenvalue plot (e.g. "loc", "fontsize", "framealpha" etc.). If set
            to ``False``, then no legend is shown. Default is ``None``.
        similar_orb_criterion (float):
            Threshold criterion for determining if the orbitals of two eigenstates
            are similar (for identifying band-edge and defect states). If the
            summed orbital projection differences, normalised by the total orbital
            projection coefficients,  are less than this value, then the orbitals
            are considered similar. Default is to try with 0.2 (``pydefect`` default),
            then if this fails increase to 0.35.
        similar_energy_criterion (float):
            Threshold criterion for considering two eigenstates similar in energy,
            used for identifying band-edge (and defect states). Bands within this
            energy difference from the VBM/CBM of the bulk are considered potential
            band-edge states. Default is to try with the larger of either 0.25 eV
            or 0.1 eV + the potential alignment from defect to bulk cells as
            determined by the charge correction in ``defect_entry.corrections_metadata``
            if present. If this fails, then it is increased to the ``pydefect`` default
            of 0.5 eV.

    Returns:
        ``pydefect`` ``PerfectBandEdgeState`` class
    """
    if defect_entry is None:
        if not all([bulk_vr, defect_vr]):
            raise ValueError(
                "If `defect_entry` is not provided, then both `bulk_vr` and `defect_vr` at a minimum "
                "must be provided!"
            )
        from doped.analysis import defect_from_structures

        bulk_vr = bulk_vr if isinstance(bulk_vr, Vasprun) else Vasprun(bulk_vr)
        defect_vr = defect_vr if isinstance(defect_vr, Vasprun) else Vasprun(defect_vr)

        (
            defect,
            defect_site,
            defect_site_in_bulk,  # bulk site for vac/sub, relaxed defect site w/interstitials
            defect_site_index,  # in this initial_defect_structure
            bulk_site_index,
            guessed_initial_defect_structure,
            unrelaxed_defect_structure,
            _bulk_voronoi_node_dict,
        ) = defect_from_structures(
            bulk_vr.final_structure,
            defect_vr.final_structure,
            oxi_state="Undetermined",
            return_all_info=True,
        )
        defect_entry = DefectEntry(
            # pmg attributes:
            defect=defect,  # this corresponds to _unrelaxed_ defect
            charge_state=0,
            sc_entry=ComputedStructureEntry(
                structure=defect_vr.final_structure,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
            bulk_entry=None,
            # doped attributes:
            defect_supercell_site=defect_site,  # _relaxed_ defect site
            defect_supercell=defect_vr.final_structure,
            bulk_supercell=bulk_vr.final_structure,
        )

    defect_entry._load_and_parse_eigenvalue_data(
        bulk_vr=bulk_vr,
        defect_vr=defect_vr,
        bulk_procar=bulk_procar,
        defect_procar=defect_procar,
        force_reparse=force_reparse,
    )

    band_orb = defect_entry.calculation_metadata["eigenvalue_data"]["band_orb"]
    vbm_info = defect_entry.calculation_metadata["eigenvalue_data"]["vbm_info"]
    cbm_info = defect_entry.calculation_metadata["eigenvalue_data"]["cbm_info"]

    # Ensures consistent number of significant figures
    def _orbital_diff(orbital_1: dict, orbital_2: dict) -> float:
        element_set = set(list(orbital_1.keys()) + list(orbital_2.keys()))
        orb_1, orb_2 = defaultdict(list, orbital_1), defaultdict(list, orbital_2)
        result = 0
        for e in element_set:
            result += sum(abs(i - j) for i, j in zip_longest(orb_1[e], orb_2[e], fillvalue=0))
        return round(result, 3) / sum(sum(orb_list) for orb_list in orb_2.values())

    pydefect.analyzer.make_band_edge_states.orbital_diff = _orbital_diff

    perfect = PerfectBandEdgeState(vbm_info, cbm_info)

    dynamic_criterion_warning = any([similar_orb_criterion, similar_energy_criterion])
    defaults._similar_orb_criterion = similar_orb_criterion or 0.2

    # similar energy criterion should be based on the charge correction potential alignment, as this is
    # what will potentially be shifting the band edge:
    def _get_pot_diff_from_entry(defect_entry: DefectEntry):
        pot_diff = 0
        if defect_entry.corrections_metadata:
            for _charge_corr_type, subdict in defect_entry.corrections_metadata.items():
                if isinstance(subdict, dict) and "pydefect_ExtendedFnvCorrection" in subdict:
                    efnv = subdict["pydefect_ExtendedFnvCorrection"]
                    if isinstance(efnv, dict):
                        pot_diff = np.mean(
                            [
                                s["potential"] - s["pc_potential"]
                                for s in efnv["sites"]
                                if s["distance"] > efnv["defect_region_radius"]
                            ]
                        )
                    else:
                        pot_diff = efnv.average_potential_diff

                elif isinstance(subdict, dict) and "mean_alignments" in subdict:
                    pot_diff = subdict["mean_alignments"]
        return pot_diff

    pot_diff = _get_pot_diff_from_entry(defect_entry)
    defaults._similar_energy_criterion = similar_energy_criterion or max(0.25, abs(pot_diff) + 0.1)

    try:
        bes = make_band_edge_states(band_orb, perfect)
    except ValueError:  # increase to pydefect defaults:
        if dynamic_criterion_warning:  # only warn if user has set custom criteria
            warnings.warn(
                f"Band-edge state identification failed with the current criteria: "
                f"similar_orb_criterion={defaults._similar_orb_criterion}, "
                f"similar_energy_criterion={defaults._similar_energy_criterion} eV. "
                f"Trying with values of 0.35 and 0.5 eV."
            )
        defaults._similar_orb_criterion = 0.35
        defaults._similar_energy_criterion = 0.5
        bes = make_band_edge_states(band_orb, perfect)  # if 2nd round fails, let it raise pydefect error

    if not plot:
        return bes

    vbm = vbm_info.orbital_info.energy + band_orb.eigval_shift
    cbm = cbm_info.orbital_info.energy + band_orb.eigval_shift

    with contextlib.suppress(Exception):
        from shakenbreak.plotting import _install_custom_font

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*glyph.*")
            emp.construct_plot()

        partial = False
        for axes in emp.axs:
            children = axes.get_children()
            annotations = [child for child in children if isinstance(child, plt.Annotation)]
            for annotation in annotations:
                if (ks_labels and annotation.get_position()[0] > 1) or not ks_labels:
                    annotation.remove()

            for child in children:
                if hasattr(child, "get_facecolor"):
                    partial = partial or any(
                        np.array_equal(child.get_facecolor()[i], [0, 0.5, 0, 1])
                        for i in range(len(child.get_facecolor()))
                    )

        if len(emp.axs) > 1:
            emp.axs[0].set_title("Spin Up")
            emp.axs[1].set_title("Spin Down")
        else:
            emp.axs[0].set_title("KS levels")

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
        if ylims is None:
            ymin, ymax = 0, 0
            for spin in emp._energies_and_occupations:
                for kpoint in spin:
                    ymin = min(ymin, *(x[0] for x in kpoint))
                    ymax = max(ymax, *(x[0] for x in kpoint))
            y_range = ymax - ymin
            ax.set_ylim([ymin - 0.05 * y_range, ymax + 0.05 * y_range])  # match default mpl +/-5% y-range
        else:
            ax.set_ylim(ylims)

        # add a point at 0,-25 with the color range and label unoccupied states
        ax.scatter(0, -25, label="Unoccupied", color=(0.98, 0.639, 0.086))
        ax.scatter(0, -25, label="Occupied", color=(0.22, 0.325, 0.643))
        if partial:
            ax.scatter(0, -25, label="Partially Occupied", color=(0, 0.5, 0))
        ax.axhline(-25, 0, 1, color="black", linewidth=0.5, linestyle="-.", label="Band Edges")

        if legend_kwargs is not False:  # otherwise no legend
            legend_kwargs = legend_kwargs or {}
            legend_kwargs["fontsize"] = legend_kwargs.get("fontsize", 7)
            legend_kwargs["framealpha"] = legend_kwargs.get("framealpha", 0.5)
            ax.legend(**legend_kwargs)

        for text_obj in emp.fig.texts:  # fix x-label alignment
            if text_obj.get_text() == "K-point coords":
                text_obj.remove()

        sub_ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis:
        sub_ax.tick_params(
            labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
        )
        bbox = sub_ax.get_position()
        x_center = bbox.x0 + bbox.width / 2  # Calculate the x position for the center of the subplot
        fig.text(x_center, 0, "$k$-point coords", ha="center", size=12)

    if filename:
        emp.plt.savefig(filename, bbox_inches="tight", transparent=True, backend=_get_backend(filename))

    return bes, fig
