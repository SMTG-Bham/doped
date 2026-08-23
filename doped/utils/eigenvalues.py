r"""
Helper functions for setting up PHS analysis.

Contains modified versions of functions from ``pydefect`` and ``vise``
(https://github.com/kumagai-group/pydefect / vise).
"""

import warnings
from collections import defaultdict
from itertools import zip_longest
from types import MethodType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.structure import PeriodicSite
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Procar, Vasprun
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry, template_defect_entry_from_structures
from doped.io.outputs import CalculationOutputs
from doped.parsing import defect_site_from_structures
from doped.utils import vise_handling
from doped.utils.plotting import doped_plot_style

with vise_handling():  # avoid vise issues (warning suppression, logging, Windows bug)
    import pydefect.analyzer.make_band_edge_states
    import pydefect.cli.vasp.make_band_edge_orbital_infos as make_bes
    from pydefect.analyzer.band_edge_states import (
        BandEdgeOrbitalInfos,
        BandEdgeStates,
        EdgeInfo,
        OrbitalInfo,
        PerfectBandEdgeState,
    )
    from pydefect.analyzer.eigenvalue_plotter import EigenvalueMplPlotter
    from pydefect.defaults import defaults
    from vise.analyzer.vasp.band_edge_properties import BandEdgeProperties


def _as_calculation_outputs(
    outputs_or_vr: CalculationOutputs | Vasprun,
    procar: PathLike | Procar | None = None,
) -> CalculationOutputs:
    """
    Normalise a ``CalculationOutputs`` / |Vasprun| (+ possible |Procar|) input
    to a (calculator-agnostic) ``CalculationOutputs`` object.

    An explicitly-supplied ``procar`` takes precedence for the orbital
    projections (``projected_eigenvalues``), matching previous ``doped``
    behaviour with VASP objects.
    """
    if isinstance(outputs_or_vr, CalculationOutputs):
        return outputs_or_vr

    from doped.io.vasp.outputs import _parse_procar, calculation_outputs_from_vasprun

    outputs = calculation_outputs_from_vasprun(outputs_or_vr, procar=procar)
    if procar is not None:  # explicitly-supplied PROCAR takes precedence for orbital projections
        outputs.projected_eigenvalues = _parse_procar(procar).data
    return outputs


def _weighted_occ_sum(outputs: CalculationOutputs, spin: Spin = Spin.up) -> float:
    """
    Get the `k`-point-weighted sum of the band occupancies for the given spin
    channel -- i.e. the number of electrons in that spin channel.
    """
    assert outputs.eigenvalues is not None  # checked by callers (with ``require()``)
    band_occs = outputs.eigenvalues[spin][:, :, 1].sum(axis=1)  # summed over bands, for each k-point
    return float(np.sum(band_occs * np.asarray(outputs.kpoint_weights)))


def _is_noncollinear(outputs: CalculationOutputs) -> bool:
    """
    Determine whether the ``outputs`` calculation is a non-collinear (NCL) spin
    calculation (i.e. with spin-orbit coupling (SOC) and/or non-collinear
    magnetism), from the eigenvalues, occupancies, `k`-point weights and
    electron count.

    Non-collinear calculations have a single set of (spinor) bands, each
    holding one electron, while non-spin-polarised calculations have a single
    set of doubly-occupied bands (and collinear-spin-polarised calculations
    have have two sets of singly-occupied bands) -- so the `k`-point-weighted
    sum of band occupancies matches ``nelect`` for NCL calculations, but
    ``nelect/2`` for non-spin-polarised (and ``nelect`` again, over both spin
    channels, for collinear spin-polarised calculations).
    """
    eigenvalues_and_occs = outputs.eigenvalues
    assert eigenvalues_and_occs is not None  # checked by callers (with ``require()``)
    assert outputs.nelect is not None  # checked by callers (with ``require()``)
    if len(eigenvalues_and_occs) > 1:
        return False  # spin-polarised (collinear) calculation; two spin channels

    summed_occs = _weighted_occ_sum(outputs)
    # matches nelect for NCL (1 electron per spinor band), or nelect/2 for non-spin-polarised:
    return abs(summed_occs - outputs.nelect) < abs(2 * summed_occs - outputs.nelect)


def band_edge_properties_from_outputs(
    outputs: CalculationOutputs | Vasprun, integer_criterion: float = 0.1
) -> BandEdgeProperties:
    """
    Create a ``pydefect`` ``BandEdgeProperties`` object from a
    :class:`~doped.io.outputs.CalculationOutputs` (or |Vasprun|) object.

    Args:
        outputs (CalculationOutputs or |Vasprun|):
            ``CalculationOutputs`` (or |Vasprun|) object for the calculation,
            with ``eigenvalues``, ``kpoint_coords``, ``kpoint_weights`` and
            ``nelect``.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or fully
            occupied (> 1 - ``integer_criterion``). Default is 0.1.

    Returns:
        ``BandEdgeProperties`` object.
    """
    outputs = _as_calculation_outputs(outputs)
    outputs.require(
        "eigenvalues",
        "kpoint_coords",
        "kpoint_weights",
        "nelect",
        task="Band edge properties determination",
    )

    collinear_magnetization: float | np.ndarray = 0
    assert outputs.eigenvalues is not None  # typing (require() ensures this)
    if len(outputs.eigenvalues) > 1:  # collinear spin-polarised calculation
        collinear_magnetization = (
            outputs.magnetization
            if outputs.magnetization is not None
            else _weighted_occ_sum(outputs, Spin.up) - _weighted_occ_sum(outputs, Spin.down)
        )

    band_edge_prop = BandEdgeProperties(
        eigenvalues={spin: e[:, :, 0] for spin, e in outputs.eigenvalues.items()},
        nelect=outputs.nelect,
        magnetization=collinear_magnetization,  # used by ``pydefect``/``vise`` w/collinear spin-polarised
        kpoint_coords=outputs.kpoint_coords,
        integer_criterion=integer_criterion,
        is_non_collinear=_is_noncollinear(outputs),
    )
    band_edge_prop.structure = outputs.structure
    return band_edge_prop


def band_edge_properties_from_vasprun(
    vasprun: Vasprun, integer_criterion: float = 0.1
) -> BandEdgeProperties:
    """
    Create a ``pydefect`` ``BandEdgeProperties`` object from a |Vasprun|
    object.

    Convenience (VASP) wrapper for :func:`band_edge_properties_from_outputs`.

    Args:
        vasprun (|Vasprun|): |Vasprun| object.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or fully
            occupied (> 1 - ``integer_criterion``). Default is 0.1.

    Returns:
        ``BandEdgeProperties`` object.
    """
    return band_edge_properties_from_outputs(vasprun, integer_criterion)


def _get_edge_info(band_edge, orbs, outputs: CalculationOutputs) -> EdgeInfo:
    """
    Get the ``pydefect`` ``EdgeInfo`` object for a band edge, using the
    calculation outputs (reimplementation of the ``pydefect`` ``get_edge_info``
    function, working off calculator-agnostic ``CalculationOutputs`` rather
    than |Vasprun| objects).
    """
    orbitals = make_bes.calc_orbital_character(
        orbs, outputs.structure, Spin.up, band_edge.kpoint_index, band_edge.band_index
    )
    assert outputs.eigenvalues is not None
    energy, occupation = outputs.eigenvalues[Spin.up][band_edge.kpoint_index, band_edge.band_index, :]
    return EdgeInfo(
        band_edge.band_index,
        tuple(band_edge.kpoint_coords),
        OrbitalInfo(energy=energy, occupation=occupation, orbitals=orbitals),
    )


def make_perfect_band_edge_state_from_outputs(
    outputs: CalculationOutputs | Vasprun, integer_criterion: float = 0.1
) -> PerfectBandEdgeState:
    """
    Create a ``pydefect`` ``PerfectBandEdgeState`` object from a
    :class:`~doped.io.outputs.CalculationOutputs` (or |Vasprun|) object,
    without the need for the |Outcar| input (as in ``pydefect``).

    Args:
        outputs (CalculationOutputs or |Vasprun|):
            ``CalculationOutputs`` (or |Vasprun|) object for the bulk cell
            calculation, with eigenvalue data and orbital projections.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or fully
            occupied (> 1 - ``integer_criterion``). Default is 0.1.

    Returns:
        ``PerfectBandEdgeState`` object.
    """
    outputs = _as_calculation_outputs(outputs)
    outputs.require("projected_eigenvalues", task="Band edge state determination")
    band_edge_prop = band_edge_properties_from_outputs(outputs, integer_criterion)
    orbs = outputs.projected_eigenvalues
    vbm_info = _get_edge_info(band_edge_prop.vbm_info, orbs, outputs)
    cbm_info = _get_edge_info(band_edge_prop.cbm_info, orbs, outputs)
    return PerfectBandEdgeState(vbm_info, cbm_info)


def make_perfect_band_edge_state_from_vasp(
    vasprun: Vasprun, procar: Procar, integer_criterion: float = 0.1
) -> PerfectBandEdgeState:
    """
    Create a ``pydefect`` ``PerfectBandEdgeState`` object from just a |Vasprun|
    and |Procar| object, without the need for the |Outcar| input (as in
    ``pydefect``).

    Convenience (VASP) wrapper for
    :func:`make_perfect_band_edge_state_from_outputs`.

    Args:
        vasprun (|Vasprun|): |Vasprun| object.
        procar (|Procar|): |Procar| object.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or fully
            occupied (> 1 - ``integer_criterion``). Default is 0.1.

    Returns:
        ``PerfectBandEdgeState`` object.
    """
    return make_perfect_band_edge_state_from_outputs(
        _as_calculation_outputs(vasprun, procar=procar), integer_criterion
    )


def make_band_edge_orbital_infos(
    defect_outputs: CalculationOutputs | Vasprun,
    vbm: float,
    cbm: float,
    eigval_shift: float = 0.0,
    neighbor_indices: list[int] | None = None,
    defect_procar: Procar | None = None,
):
    r"""
    Make ``BandEdgeOrbitalInfos`` from a
    :class:`~doped.io.outputs.CalculationOutputs` (or |Vasprun|) object.

    Modified from ``pydefect`` to use the projected orbitals stored in the
    calculation outputs.

    Args:
        defect_outputs (CalculationOutputs or |Vasprun|):
            ``CalculationOutputs`` (or |Vasprun|) object for the defect
            supercell calculation.
        vbm (float): VBM eigenvalue in eV.
        cbm (float): CBM eigenvalue in eV.
        eigval_shift (float):
            Shift eigenvalues by this value in eV. Default is 0.0.
        neighbor_indices (list[int]):
            Indices of neighboring atoms to the defect site, for localisation
            analysis. Default is ``None``.
        defect_procar (|Procar|):
            ``pymatgen`` |Procar| object, for the defect supercell, if
            projected eigenvalue/orbitals data is not provided in
            ``defect_outputs``.

    Returns:
        ``BandEdgeOrbitalInfos`` object.
    """
    outputs = _as_calculation_outputs(defect_outputs, procar=defect_procar)
    outputs.require(
        "eigenvalues", "projected_eigenvalues", "kpoint_coords", task="Eigenvalue & orbital analysis"
    )
    assert outputs.eigenvalues is not None  # typing (require() ensures these)
    assert outputs.kpoint_coords is not None
    eigval_range = defaults.eigval_range
    kpt_coords = [tuple(coord) for coord in outputs.kpoint_coords]
    max_energy_by_spin, min_energy_by_spin = [], []

    for e in outputs.eigenvalues.values():
        max_energy_by_spin.append(np.amax(e[:, :, 0], axis=0))
        min_energy_by_spin.append(np.amin(e[:, :, 0], axis=0))

    max_energy_by_band = np.amax(np.vstack(max_energy_by_spin), axis=0)
    min_energy_by_band = np.amin(np.vstack(min_energy_by_spin), axis=0)

    lower_idx = np.argwhere(max_energy_by_band > vbm - eigval_range)[0][0]
    upper_idx = np.argwhere(min_energy_by_band < cbm + eigval_range)[-1][-1]

    orbs = outputs.projected_eigenvalues
    s = outputs.structure
    orb_infos: list[Any] = []
    for spin, eigvals in outputs.eigenvalues.items():
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
        kpt_weights=np.asarray(outputs.kpoint_weights).tolist(),
        lowest_band_index=int(lower_idx),
        fermi_level=outputs.efermi,
        eigval_shift=eigval_shift,
    )


def get_band_edge_info(
    defect_outputs: CalculationOutputs | Vasprun,
    bulk_outputs: CalculationOutputs | Vasprun,
    defect_procar: PathLike | Procar | None = None,
    bulk_procar: PathLike | Procar | None = None,
    defect_supercell_site: PeriodicSite | None = None,
    neighbor_cutoff_factor: float = 1.3,
) -> tuple[BandEdgeOrbitalInfos, EdgeInfo, EdgeInfo]:
    """
    Generate metadata required for performing eigenvalue & orbital analysis,
    specifically ``pydefect`` ``BandEdgeOrbitalInfos``, and ``EdgeInfo``
    objects for the bulk VBM and CBM.

    See the :ref:`Tips:Perturbed Host States (Shallow Defects)` tips section.

    Args:
        defect_outputs (CalculationOutputs or |Vasprun|):
            :class:`~doped.io.outputs.CalculationOutputs` or |Vasprun| object
            of the defect supercell calculation. If ``defect_procar`` is not
            provided, then this must have orbital projection data
            (``projected_eigenvalues``; i.e. from a calculation with
            ``LORBIT > 10`` in the ``INCAR`` and parsed with
            ``parse_projected_eigen = True`` (default), with VASP).
        bulk_outputs (CalculationOutputs or |Vasprun|):
            :class:`~doped.io.outputs.CalculationOutputs` or |Vasprun| object
            of the bulk supercell calculation. If ``bulk_procar`` is not
            provided, then this must have orbital projection data
            (``projected_eigenvalues``; i.e. from a calculation with
            ``LORBIT > 10`` in the ``INCAR`` and parsed with
            ``parse_projected_eigen = True`` (default), with VASP).
        defect_procar (PathLike, |Procar|):
            Either a path to the ``VASP`` ``PROCAR(.gz)`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or a ``pymatgen`` |Procar|
            object, for the defect supercell calculation. Not required if the
            supplied ``defect_outputs`` has orbital projection data. Default
            is ``None``.
        bulk_procar (PathLike, |Procar|):
            Either a path to the ``VASP`` ``PROCAR(.gz)`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or a ``pymatgen`` |Procar|
            object, for the reference bulk supercell calculation. Not required
            if the supplied ``bulk_outputs`` has orbital projection data.
            Default is ``None``.
        defect_supercell_site (|PeriodicSite|):
            |PeriodicSite| object of the defect site in the defect supercell,
            from which the defect neighbours are determined for localisation
            analysis. If ``None`` (default), then the defect site is determined
            automatically from the defect and bulk supercell structures.
        neighbor_cutoff_factor (float):
            Sites within ``min_distance * neighbor_cutoff_factor`` of the
            defect site in the `relaxed` defect supercell are considered
            neighbours for localisation analysis, where ``min_distance`` is the
            minimum distance between sites in the defect supercell. Default is
            1.3 (matching the ``pydefect`` default).

    Returns:
        ``pydefect`` ``BandEdgeOrbitalInfos``, and ``EdgeInfo`` objects for the
        bulk VBM and CBM.
    """
    bulk_outputs = _as_calculation_outputs(bulk_outputs, procar=bulk_procar)
    defect_outputs = _as_calculation_outputs(defect_outputs, procar=defect_procar)
    band_edge_prop = band_edge_properties_from_outputs(bulk_outputs)

    # get defect neighbour indices
    sorted_distances = np.sort(defect_outputs.structure.distance_matrix.flatten())
    min_distance = sorted_distances[sorted_distances > 0.5][0]

    if defect_supercell_site is None:
        defect_supercell_site = defect_site_from_structures(
            defect_outputs.structure, bulk_outputs.structure, _parameter_order_warn=False
        )
        assert isinstance(defect_supercell_site, PeriodicSite)  # typing

    neighbor_indices = [
        i
        for i, site in enumerate(defect_outputs.structure.sites)
        if defect_supercell_site.distance(site) <= min_distance * neighbor_cutoff_factor
    ]

    with vise_handling():  # avoid vise issues (warning suppression, logging, Windows bug)
        orbs = bulk_outputs.projected_eigenvalues
        vbm_info = _get_edge_info(band_edge_prop.vbm_info, orbs, bulk_outputs)
        cbm_info = _get_edge_info(band_edge_prop.cbm_info, orbs, bulk_outputs)

        band_orb = make_band_edge_orbital_infos(
            defect_outputs,
            vbm_info.orbital_info.energy,
            cbm_info.orbital_info.energy,
            neighbor_indices=neighbor_indices,
        )

    return band_orb, vbm_info, cbm_info


def get_eigenvalue_analysis(
    defect_entry: DefectEntry | None = None,
    plot: bool = True,
    filename: str | None = None,
    ks_labels: bool = False,
    style_file: str | None = None,
    bulk_outputs: PathLike | Vasprun | CalculationOutputs | None = None,
    bulk_procar: PathLike | Procar | None = None,
    defect_outputs: PathLike | Vasprun | CalculationOutputs | None = None,
    defect_procar: PathLike | Procar | None = None,
    force_reparse: bool = False,
    ylims: tuple[float, float] | None = None,
    legend_kwargs: dict | None = None,
    similar_orb_criterion: float | None = None,
    similar_energy_criterion: float | None = None,
) -> BandEdgeStates | tuple[BandEdgeStates, plt.Figure]:
    r"""
    Get eigenvalue & orbital info (with automated classification of PHS states)
    for the band edge and in-gap electronic states for the input defect entry /
    calculation outputs, as well as a plot of the single-particle electronic
    eigenvalues and their occupation (if ``plot=True``).

    Can be used to determine if a defect is adopting a perturbed host state
    (PHS / shallow state), see the
    :ref:`Tips:Perturbed Host States (Shallow Defects)` tips section.

    Note that the classification of electronic states as band edges or
    localised orbitals is based on the similarity of orbital projections and
    eigenvalues between the defect and bulk cell calculations (see
    ``similar_orb/energy_criterion`` argument descriptions below for more
    details). You may want to adjust the default values of these keyword
    arguments, as the defaults may not be appropriate in all cases. In
    particular, the P-ratio values can give useful insight, revealing the level
    of (de)localisation of the states.

    Either a ``doped`` |DefectEntry| object can be provided, or the required
    VASP output files/objects for the bulk and defect supercell calculations
    (|Vasprun|\s, or |Vasprun|\s and |Procar|\s). If a |DefectEntry| is
    provided but eigenvalue data has not already been parsed (default in
    ``doped`` is to parse this data with |DefectsParser|/``DefectParser``, as
    controlled by the ``parse_projected_eigen`` flag), then this function will
    attempt to load the eigenvalue data from either the input |Vasprun| /
    |Procar| objects or files, or from the ``bulk/defect_path``\s in
    ``defect_entry.calculation_metadata``. If so, will initially try to load
    orbital projections from ``vasprun.xml(.gz)`` files (more accurate due to
    less rounding errors), or failing that from ``PROCAR(.gz)`` files if
    present.

    This function uses code from ``pydefect``, so please cite the ``pydefect``
    paper: https://doi.org/10.1103/PhysRevMaterials.5.123803

    Args:
        defect_entry (|DefectEntry|):
            ``doped`` |DefectEntry| object. Default is ``None``.
        plot (bool):
            Whether to plot the single-particle eigenvalues. (Default: True)
        filename (str):
            Filename to save the eigenvalue plot to (if ``plot = True``). If
            ``None`` (default), plots are not saved.
        ks_labels (bool):
            Whether to add band index labels to the KS levels. (Default: False)
        style_file (str):
            Path to a ``mplstyle`` file to use for the plot. If ``None``
            (default), uses the ``doped`` displacement plot style
            (``doped/utils/displacement.mplstyle``).
        bulk_outputs (PathLike, |Vasprun| or CalculationOutputs):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``, data
            in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``vasprun.xml(.gz)`` output file, a
            ``pymatgen`` |Vasprun| object or a
            :class:`~doped.io.outputs.CalculationOutputs` object, for the
            reference bulk supercell calculation. If ``None`` (default), tries
            to load the |Vasprun| object from
            ``defect_entry.calculation_metadata["run_metadata"]["bulk_vasprun_dict"]``
            or, failing that, from a ``vasprun.xml(.gz)`` file at
            ``defect_entry.calculation_metadata["bulk_path"]``.
        bulk_procar (PathLike, |Procar|):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``, data
            in ``defect_entry.calculation_metadata["eigenvalue_data"]``), or if
            ``bulk_outputs`` was parsed with ``parse_projected_eigen = True``
            (default). Either a path to the ``VASP`` ``PROCAR`` output file
            (with ``LORBIT > 10`` in the ``INCAR``) or a ``pymatgen``
            |Procar| object, for the reference bulk supercell calculation. If
            ``None`` (default), tries to load from a ``PROCAR(.gz)`` file at
            ``defect_entry.calculation_metadata["bulk_path"]``.
        defect_outputs (PathLike, |Vasprun| or CalculationOutputs):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``, data
            in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``vasprun.xml(.gz)`` output file, a
            ``pymatgen`` |Vasprun| object or a
            :class:`~doped.io.outputs.CalculationOutputs` object, for the
            defect supercell calculation. If ``None`` (default), tries to load
            the |Vasprun| object from
            ``defect_entry.calculation_metadata["run_metadata"]["defect_vasprun_dict"]``
            or, failing that, from a ``vasprun.xml(.gz)`` file at
            ``defect_entry.calculation_metadata["defect_path"]``.
        defect_procar (PathLike, |Procar|):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``, data
            in ``defect_entry.calculation_metadata["eigenvalue_data"]``), or if
            ``defect_outputs`` was parsed with ``parse_projected_eigen = True``
            (default). Either a path to the ``VASP`` ``PROCAR`` output file
            (with ``LORBIT > 10`` in the ``INCAR``) or a ``pymatgen``
            |Procar| object, for the defect supercell calculation. If
            ``None`` (default), tries to load from a ``PROCAR(.gz)`` file at
            ``defect_entry.calculation_metadata["defect_path"]``.
        force_reparse (bool):
            Whether to force re-parsing of the eigenvalue data, even if already
            present in the ``calculation_metadata`` dict.
        ylims (tuple[float, float]):
            Custom y-axis limits for the eigenvalue plot. If ``None``
            (default), the y-axis limits are automatically set to +/-5% of the
            eigenvalue range.
        legend_kwargs (dict):
            Custom keyword arguments to pass to the ``ax.legend`` call in the
            eigenvalue plot (e.g. "loc", "fontsize", "framealpha" etc.). If set
            to ``False``, then no legend is shown. Default is ``None``.
        similar_orb_criterion (float):
            Threshold criterion for determining if the orbitals of two
            eigenstates are similar (for identifying band-edge and defect
            states). If the summed orbital projection differences, normalised
            by the total orbital projection coefficients, are less than this
            value, then the orbitals are considered similar. Default is to try
            with 0.2 (``pydefect`` default), then if this fails increase to
            0.35, and lastly 0.5.
        similar_energy_criterion (float):
            Threshold criterion for considering two eigenstates similar in
            energy, used for identifying band-edge (and defect states). Bands
            within this energy difference from the VBM/CBM of the bulk are
            considered potential band-edge states. Default is to try with the
            larger of either 0.25 eV or 0.1 eV + the potential alignment from
            defect to bulk cells as determined by the charge correction in
            ``defect_entry.corrections_metadata`` if present. If this fails,
            then it is increased to the ``pydefect`` default of 0.5 eV.

    Returns:
        ``pydefect`` ``BandEdgeStates`` object, containing the band-edge and
        defect eigenvalue information, and the eigenvalue plot (if
        ``plot=True``).
    """
    if defect_entry is None:
        if not all([bulk_outputs, defect_outputs]):
            raise ValueError(
                "If `defect_entry` is not provided, then both `bulk_outputs` and `defect_outputs` at a "
                "minimum must be provided!"
            )

        if not isinstance(bulk_outputs, Vasprun | CalculationOutputs):
            bulk_outputs = Vasprun(bulk_outputs)
        if not isinstance(defect_outputs, Vasprun | CalculationOutputs):
            defect_outputs = Vasprun(defect_outputs)

        def _structure_of(outputs_or_vr):
            if isinstance(outputs_or_vr, CalculationOutputs):
                return outputs_or_vr.structure
            return outputs_or_vr.final_structure

        defect_entry = template_defect_entry_from_structures(
            _structure_of(defect_outputs),
            _structure_of(bulk_outputs),
            oxi_state="Undetermined",
            multiplicity=1,
        )

    # TODO: Allow just bulk and 'defect_outputs' to be passed directly for this function, so it can be
    #  used with e.g. polarons etc
    defect_entry._load_and_parse_eigenvalue_data(
        bulk_outputs=bulk_outputs,
        defect_outputs=defect_outputs,
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
        bes = pydefect.analyzer.make_band_edge_states.make_band_edge_states(band_orb, perfect)
    except ValueError:  # increase to pydefect defaults:
        defaults._similar_orb_criterion = 0.35
        defaults._similar_energy_criterion = 0.5
        try:
            bes = pydefect.analyzer.make_band_edge_states.make_band_edge_states(band_orb, perfect)
        except ValueError:
            defaults._similar_orb_criterion = 0.5
            # if fails, let it raise pydefect error:
            bes = pydefect.analyzer.make_band_edge_states.make_band_edge_states(band_orb, perfect)

        if dynamic_criterion_warning:  # only warn if user has set custom criteria
            warnings.warn(
                f"Band-edge state identification failed with the current criteria: "
                f"similar_orb_criterion={similar_orb_criterion}, "
                f"similar_energy_criterion={similar_energy_criterion} eV, but succeeded with "
                f"similar_orb_criterion={defaults._similar_orb_criterion}, "
                f"similar_energy_criterion={defaults._similar_energy_criterion} eV. "
            )

    if not plot:
        return bes

    vbm = vbm_info.orbital_info.energy + band_orb.eigval_shift
    cbm = cbm_info.orbital_info.energy + band_orb.eigval_shift

    with vise_handling():  # avoid vise issues (warning suppression, logging, Windows bug)
        # style the figure created during plotter construction:
        with doped_plot_style(style_file, style="displacement"):
            emp = EigenvalueMplPlotter(
                title="Eigenvalues",
                band_edge_orb_infos=band_orb,
                supercell_vbm=vbm,
                supercell_cbm=cbm,
                y_range=[vbm - 3, cbm + 3],
            )

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
            for _spin_idx, (eo_by_spin, ax) in enumerate(
                zip(self._energies_and_occupations, self.axs, strict=False)
            ):
                kpt_indices = []
                energies = []
                color_list = []
                annotations = []
                for kpt_idx, eo_by_k_idx in enumerate(eo_by_spin):
                    for band_idx, eo_by_band in enumerate(eo_by_k_idx):
                        energy, occup = eo_by_band
                        color_list.append(
                            occupied_color
                            if occup > 0.9
                            else unoccupied_color
                            if occup < 0.1
                            else partial_color
                        )
                        kpt_indices.append(kpt_idx)
                        energies.append(energy)

                        try:
                            higher_band_e = eo_by_k_idx[band_idx + 1][0]
                            lower_band_e = eo_by_k_idx[band_idx - 1][0]
                        except IndexError:
                            continue

                        if self._add_band_idx(energy, higher_band_e, lower_band_e):
                            annotations.append(
                                (kpt_idx + 0.05, energy, band_idx + self._lowest_band_idx + 1)
                            )

                ax.scatter(kpt_indices, energies, c=color_list, s=self._mpl_defaults.circle_size)
                for annotation in annotations:
                    ax.annotate(
                        annotation[2],
                        (annotation[0], annotation[1]),
                        va="center",
                        fontsize=self._mpl_defaults.tick_label_size,
                    )

        emp._add_eigenvalues = MethodType(_add_eigenvalues, emp)  # faster monkey-patch for eigenvalues

    with doped_plot_style(style_file, style="displacement"):
        plt.rcParams["axes.titlesize"] = 12
        plt.rc("axes", unicode_minus=False)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*glyph.*")
            emp.construct_plot()  # calls ``self._add_eigenvalues()``

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

        emp.axs[0].set_ylabel("Eigenvalue (eV)")

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
            ymin, ymax = vbm, cbm
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
        emp.plt.savefig(filename, bbox_inches="tight", transparent=True)

    return bes, fig
