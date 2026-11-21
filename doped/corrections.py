"""
Code to compute finite-size charge corrections for charged defects in periodic
systems.

The charge-correction methods implemented are:

1. Extended FNV (eFNV) / Kumagai correction for isotropic and anisotropic
   systems. This is the recommended default as it has a more efficient
   implementation and tends to be more robust in cases of small supercells.
   Includes:

       a) anisotropic point-charge (PC) energy (optionally multiple PCs).
       b) potential alignment by averaging atomic site potential differences,
          in a sampling region away from the defect.

2. Freysoldt (FNV) correction for isotropic and anisotropic systems (with
   anisotropic screening treated as in ``sxdefectalign``). Only recommended if
   eFNV correction is not possible. Includes:

       a) anisotropic point-charge energy.
       b) potential alignment by planar averaging the volumetric potential
          difference, in planar regions furthest from the defect.

If you use the corrections implemented in this module, please cite:

    Kumagai and Oba, Phys. Rev. B. 2014 for the eFNV correction
    or
    Freysoldt, Neugebauer, and Van de Walle, Phys. Rev. Lett. 2009 for FNV.

**Note**: Ideally, the "defect site" used for all charge corrections should
actually be the centre of the localised charge in the defect supercell. Usually
this coincides with the defect site, but not always (e.g. vacancy where the
charge localises as a polaron on a neighbouring atom, complex defects etc.).
For sufficiently large supercells this is usually fine, as the defect and
centre-of-charge sites are close enough that any resulting quantitative error
in the correction is negligible. However, in cases where we have large
finite-size correction values, this can be significant. If some efficient
methods for determining the centroid of charge difference between bulk/defect
LOCPOTs (for FNV) or bulk/defect site potentials (for eFNV) were developed,
they should be used here.
"""

import contextlib
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from monty.json import MontyDecoder
from pymatgen.analysis.defects.corrections import freysoldt
from pymatgen.analysis.defects.utils import CorrectionResult
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.util.typing import PathLike

from doped.analysis import _custom_formatwarning
from doped.utils import vise_handling
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_core_potentials_from_outcar_obj,
    _get_defect_supercell,
    _get_defect_supercell_frac_coords,
    get_core_potentials_from_outcar,
    get_half_max_face_distance,
    get_locpot,
    get_site_mappings,
)
from doped.utils.plotting import doped_plot_style, format_defect_name
from doped.utils.supercells import _get_min_image_distance_from_matrix, _get_unwrapped_complex_frac_coords

warnings.formatwarning = _custom_formatwarning


def _monty_decode_nested_dicts(d):
    """
    Decode any MSON dicts in ``d`` (in place), which may be nested in dicts or
    lists, tolerating per-key decoding failures.

    Used to ensure ``DefectEntry.calculation_metadata`` is decoded (after
    reloading from JSON).
    """
    for key, value in d.items():
        try:
            d[key] = MontyDecoder().process_decoded(value)
        except Exception as exc:
            print(f"Failed to decode {key} with error {exc!r}")


def _check_if_None_and_raise_error_if_so(var, var_name, display_name):
    if var is None:
        raise ValueError(
            f"{var_name} must be provided as an argument or be present in the `defect_entry` "
            f"`calculation_metadata` (as `{display_name}`), neither of which is the case here!"
        )


def _get_and_check_metadata(entry, key, display_name):
    value = entry.calculation_metadata.get(key, None)
    _check_if_None_and_raise_error_if_so(value, display_name, key)
    return value


def _check_if_pathlike_and_get_locpot_or_core_pots(
    locpot_or_outcar: Locpot | Outcar | PathLike | dict,
    obj_type: str = "locpot",
    dir_type: str = "",
    total_energy: list | float | None = None,
):
    if isinstance(locpot_or_outcar, PathLike):
        if obj_type == "locpot":
            return get_locpot(locpot_or_outcar)
        return get_core_potentials_from_outcar(  # otherwise OUTCAR
            locpot_or_outcar, dir_type=dir_type, total_energy=total_energy
        )

    if isinstance(locpot_or_outcar, Outcar):
        return _get_core_potentials_from_outcar_obj(
            locpot_or_outcar, dir_type=dir_type, total_energy=total_energy
        )

    if not isinstance(locpot_or_outcar, Locpot | Outcar | dict):
        raise TypeError(
            f"`{obj_type}` input must be either a path to a {obj_type.upper()} file or a pymatgen "
            f"{obj_type.upper()[0] + obj_type[1:]} object, but got {type(locpot_or_outcar)} instead."
        )

    return locpot_or_outcar


def _check_and_unwrap_point_charges(
    lattice, charge, point_charges, correction_type: str = "eFNV"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check the input ``point_charges`` for a multi-site charge correction, and
    unwrap them to a single (minimum image) configuration, i.e. the isolated
    complex geometry.

    Args:
        lattice (|Lattice|):
            Lattice of the defect supercell.
        charge (float):
            Total defect charge state, which the point charges must sum to.
        point_charges (list[tuple[float, ArrayLike]]):
            List of ``(charge, frac_coords)`` pairs of the model charges.
        correction_type (str):
            Name of the charge correction, for the warning message. Default is
            "eFNV".

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Array of the site charges, and array of their unwrapped fractional
            coordinates.
    """
    site_charges = np.array([q for q, _fc in point_charges], dtype=float)
    if not np.isclose(site_charges.sum(), charge):
        raise ValueError(
            f"The sum of the supplied point charges ({site_charges.sum():+.3f}) does not match the total "
            f"defect charge state ({charge:+.3f})."
        )
    charge_coords = np.array(_get_unwrapped_complex_frac_coords(lattice, [fc for _q, fc in point_charges]))
    half_min_image_dist = _get_min_image_distance_from_matrix(lattice.matrix) / 2
    for i, j in combinations(range(len(site_charges)), 2):
        pair_dist = np.linalg.norm(lattice.get_cartesian_coords(charge_coords[j] - charge_coords[i]))
        if pair_dist < 0.1:  # pair term is finite as pair_dist -> 0, but likely an input error
            raise ValueError(
                f"Point charges {i} and {j} are (near-)coincident ({pair_dist:.2f} Å apart), so should "
                f"likely be combined into a single point charge."
            )
        if pair_dist > half_min_image_dist:
            warnings.warn(
                f"Point charges {i} and {j} are {pair_dist:.2f} Å apart, which is more than half the "
                f"minimum image distance of the supercell ({half_min_image_dist:.2f} Å), so the multiple "
                f"point charge model (and {correction_type} correction) may be unreliable."
            )
    return site_charges, charge_coords


def _get_multi_site_freysoldt_correction(
    site_charges: np.ndarray,
    charge_coords: np.ndarray,
    dielectric: float | np.ndarray | list,
    defect_locpot: Locpot | dict,
    bulk_locpot: Locpot | dict,
    lattice,
    q_model=None,
    energy_cutoff: float = 1500,
    mad_tol: float = 1e-6,
) -> CorrectionResult:
    r"""
    Freysoldt (FNV) correction for a model charge of one or more Gaussians, as
    with multiple charges in ``sxdefectalign``.

    This generalises the ``pymatgen-analysis-defects``
    ``get_freysoldt_correction`` function, with the same output format and
    treatment of anisotropic dielectric screening, and reproduces it for a
    single charge. The potential alignment along each axis is sampled furthest
    from all charges (half a cell from the grid plane nearest the middle of the
    charges, i.e. opposite the largest gap between them along that axis).

    Args:
        site_charges (np.ndarray):
            Charges of the model Gaussians.
        charge_coords (np.ndarray):
            Unwrapped fractional coordinates of the model Gaussians (see
            ``_check_and_unwrap_point_charges``).
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constant (ionic + static contributions).
        defect_locpot (Locpot | dict):
            ``Locpot`` object, or dictionary of the planar-averaged potential
            along each axis, for the defect supercell.
        bulk_locpot (Locpot | dict):
            ``Locpot`` object, or dictionary of the planar-averaged potential
            along each axis, for the bulk supercell.
        lattice (|Lattice|):
            Lattice of the defect supercell.
        q_model (QModel):
            ``pymatgen-analysis-defects`` ``QModel`` object, setting the width
            (``beta``) of each model charge. Non-Gaussian model charges
            (``expnorm > 0``) are only supported for a single charge. Default
            is ``None`` (``QModel()``; Gaussians with ``beta`` = 1 bohr).
        energy_cutoff (float):
            Maximum reciprocal space energy cutoff (in eV) to try in converging
            the periodic energy of the model charge. Default is 1500.
        mad_tol (float):
            Convergence tolerance (in Ha) for the periodic energy of the model
            charge. Default is 1e-6 (tighter than the 1e-4 default in
            ``pymatgen-analysis-defects``, at negligible cost here).

    Returns:
        CorrectionResult:
            Correction summary object, with the same ``metadata`` as from the
            ``pymatgen-analysis-defects`` ``get_freysoldt_correction``
            function.
    """
    from pymatgen.analysis.defects.utils import QModel, ang_to_bohr, converge, genrecip, hart_to_ev
    from scipy.integrate import quad

    q_model = q_model or QModel()
    if q_model.expnorm and len(site_charges) > 1:  # pair interaction below is for Gaussians
        raise ValueError("Only Gaussian model charges (``expnorm = 0``) are supported for multi-site FNV.")

    def get_planar_averages(locpot):
        if not isinstance(locpot, dict):
            return [locpot.get_average_along_axis(i) for i in range(3)]
        locpot = {int(k): v for k, v in locpot.items()}  # keys are strings if reloaded from ``json``
        return [np.asarray(locpot[i], dtype=float) for i in range(3)]

    defect_avgs, bulk_avgs = get_planar_averages(defect_locpot), get_planar_averages(bulk_locpot)
    total_charge = site_charges.sum()
    eps = freysoldt._get_dielectric_tensor(dielectric)
    avg_eps = freysoldt.get_spherically_averaged_dielectric(eps)
    eps_eigvals, eps_eigvecs = np.linalg.eigh((eps + eps.T) / 2)  # symmetric part, as enters G·ε·G
    cell = ang_to_bohr * lattice.matrix  # atomic units
    volume = abs(np.linalg.det(cell))
    cart_coords = charge_coords @ cell
    # G = 0 term of the long-range model potential (per unit charge), which is included in both the
    # periodic energy and the model potential (so cancels in the total correction), as in
    # ``pymatgen-analysis-defects``:
    g0_term = 4 * np.pi * q_model.rho_rec_limit0 / (avg_eps * volume)

    def e_periodic(encut):  # (2π/V) Σ_G |Σᵢ qᵢ exp(-iG·rᵢ)|² rho(G)² / (G·ε·G), for G ≠ 0
        G = np.array(list(genrecip(*cell, encut)))
        if not len(G):
            return 0.0
        structure_factor = np.exp(-1j * G @ cart_coords.T) @ site_charges
        rho_sq = q_model.rho_rec(np.einsum("ij,ij->i", G, G)) ** 2
        screened_G_sq = np.einsum("ij,jk,ik->i", G, eps, G)
        return 2 * np.pi / volume * np.sum(np.abs(structure_factor) ** 2 * rho_sq / screened_G_sq)

    def e_isolated_pair(r_ij):  # interaction of two unit Gaussians separated by r_ij, in the dielectric
        u = (eps_eigvecs.T @ r_ij) / np.sqrt(eps_eigvals)
        widths = q_model.beta**2 / eps_eigvals

        def integrand(t):
            denom = 1 + 2 * t * t * widths
            return np.exp(-t * t * np.sum(u * u / denom)) / np.sqrt(np.prod(denom))

        integral = quad(integrand, 0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
        return 2 / np.sqrt(np.pi) * integral / np.sqrt(np.prod(eps_eigvals))

    # isolated energy; model charge self-energies, screened by the spherically-averaged dielectric
    # constant as in ``pymatgen-analysis-defects``, plus pair interactions:
    e_self = quad(lambda g: q_model.rho_rec(g * g) ** 2, 0, np.inf, epsabs=1e-13, epsrel=1e-11)[0] / np.pi
    e_isolated = np.sum(site_charges**2) * e_self / avg_eps + sum(
        site_charges[i] * site_charges[j] * e_isolated_pair(cart_coords[j] - cart_coords[i])
        for i, j in combinations(range(len(site_charges)), 2)
    )
    e_periodic_converged = converge(e_periodic, 5, mad_tol, energy_cutoff) + total_charge**2 * g0_term
    es_corr = (e_isolated - e_periodic_converged) * hart_to_ev

    alignment_corrs, plot_data = {}, {}
    for axis in range(3):
        n = len(defect_avgs[axis])
        # planar-averaged long-range model potential, as in ``perform_pot_corr``:
        b_vec = lattice.reciprocal_lattice.matrix[axis]
        k = np.arange(1, (n + 1) // 2)
        g2 = (k * lattice.reciprocal_lattice.abc[axis] / ang_to_bohr) ** 2
        coeffs = 8 * np.pi * q_model.rho_rec(g2) / (b_vec @ eps @ b_vec / (b_vec @ b_vec) * g2) / volume
        phases = np.arange(n)[:, None] / n - charge_coords[None, :, axis]  # (n, N_charges)
        model_pot = -hart_to_ev * (
            np.cos(2 * np.pi * phases[:, :, None] * k) @ coeffs @ site_charges + total_charge * g0_term
        )

        # sample alignment furthest from all charges along this axis; shifting the grid plane nearest the
        # centre of the charges (i.e. opposite their largest gap) to the origin, with ``perform_pot_corr``
        # (for q = 0) then sampling half a cell from it:
        sorted_coords = np.sort(charge_coords[:, axis] % 1)
        gaps = np.diff([*sorted_coords, sorted_coords[0] + 1])
        shift = -int(np.floor((sorted_coords[np.argmax(gaps)] + gaps.max() / 2 - 0.5) * n + 0.5))
        defect_avg, bulk_avg, model_pot = (
            np.roll(avg, shift) for avg in (defect_avgs[axis], bulk_avgs[axis], model_pot)
        )
        alignment_corrs[axis], plot_data[axis] = freysoldt.perform_pot_corr(
            np.arange(n) / n * lattice.abc[axis], pureavg=bulk_avg, defavg=defect_avg - model_pot,
            lattice=lattice, q=0, defect_frac_coords=[0, 0, 0], axis=axis, dielectric=eps, q_model=q_model,
        )  # fmt: skip
        plot_data[axis]["pot_plot_data"].update({"Vr": model_pot, "dft_diff": defect_avg - bulk_avg})

    mean_alignment = np.mean(list(alignment_corrs.values()))
    pot_corr = mean_alignment * total_charge
    return CorrectionResult(
        correction_energy=es_corr + pot_corr,
        metadata={
            "plot_data": plot_data,
            "electrostatic": es_corr,
            "alignments": alignment_corrs,
            "mean_alignments": mean_alignment,
            "potential": pot_corr,
        },
    )


def get_freysoldt_correction(
    defect_entry,
    dielectric: float | np.ndarray | list | None = None,
    defect_locpot: PathLike | Locpot | dict | None = None,
    bulk_locpot: PathLike | Locpot | dict | None = None,
    plot: bool = False,
    filename: PathLike | None = None,
    axis: int | None = None,
    verbose: bool = True,
    style_file: PathLike | None = None,
    ax: plt.Axes | list[plt.Axes] | np.ndarray | None = None,
    point_charges: list[tuple[float, np.ndarray | list]] | None = None,
    **kwargs,
) -> CorrectionResult:
    """
    Function to compute the Freysoldt (FNV) correction for the input
    defect_entry.

    This function `does not` add the correction to ``defect_entry.corrections``
    (but the ``defect_entry.get_freysoldt_correction`` method does). If this
    correction is used, please cite Freysoldt's original paper;
    10.1103/PhysRevLett.102.016402.

    Anisotropic dielectric screening is treated as in ``sxdefectalign``, using
    the full dielectric tensor for the periodic (lattice) energy of the model
    charge and its planar-averaged long-range potential, and the spherical
    average of the dielectric tensor for the isolated energy and the G = 0
    (alignment) term.

    The defect coordinates are taken as the relaxed defect site by default
    (``DefectEntry.defect_supercell_site``) -- which is the bulk site for vacancies,
    but this can be overridden with the ``defect_frac_coords`` keyword argument.

    As a general rule of thumb, the charge correction terms should follow
    relatively consistent trends in terms of magnitudes. A large outlier
    (easily scanned with |get_formation_energies|) often indicates something
    unusual/unexpected. See the FNV/eFNV and other finite-size charge
    correction papers for further details.

    Args:
        defect_entry:
            |DefectEntry| object for which to compute the FNV finite-size
            charge correction.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constants (ionic + static contributions), in the
            same xyz Cartesian basis as the supercell calculations (likely but
            not necessarily the same as the raw output of a VASP dielectric
            calculation, if an oddly-defined primitive cell is used). If
            ``None``, then the dielectric constant is taken from the
            ``defect_entry`` ``calculation_metadata`` if available. See the
            |Dielectric Constant| tutorial section for information on
            calculating and converging the dielectric constant.
        defect_locpot:
            Path to the output VASP LOCPOT file from the defect supercell
            calculation, or the corresponding ``pymatgen`` ``Locpot`` object,
            or a dictionary of the planar-averaged potential in the form:
            ``{i: Locpot.get_average_along_axis(i) for i in [0,1,2]}``.
            If ``None``, will try to use ``defect_locpot_dict`` from the
            ``defect_entry`` ``calculation_metadata`` if available.
        bulk_locpot:
            Path to the output ``VASP`` ``LOCPOT`` file from the bulk supercell
            calculation, or the corresponding ``pymatgen`` ``Locpot`` object,
            or a dictionary of the planar-averaged potential in the form:
            ``{i: Locpot.get_average_along_axis(i) for i in [0,1,2]}``.
            If ``None``, will try to use ``bulk_locpot_dict`` from the
            ``defect_entry`` ``calculation_metadata`` if available.
        plot (bool):
            Whether to plot the FNV electrostatic potential plots (for
            manually checking the behaviour of the charge correction here).
        filename (PathLike):
            Filename to save the FNV electrostatic potential plots to.
            If None, plots are not saved.
        axis (int or None):
            If int, then the FNV electrostatic potential plot along the
            specified axis (0, 1, 2 for a, b, c) will be plotted. Note that the
            output charge correction is still that for `all` axes. If ``None``,
            then all three axes are plotted.
        verbose (bool):
            Whether to print the correction energy (default = True).
        style_file (PathLike):
            Path to a ``.mplstyle`` file to use for the plot. If ``None``
            (default), uses the default ``doped`` style
            (from ``doped/utils/doped.mplstyle``).
        ax (plt.Axes, list[plt.Axes]):
            ``matplotlib`` ``Axes`` to plot onto (e.g. for multi-panel
            figures), rather than creating a new figure, in which case the
            parent ``Figure`` of the ``Axes`` is returned. If ``axis`` is
            ``None`` (all three axes plotted), a list/array of three ``Axes``
            (for the a, b and c axes, in order) should be given (note that the
            shared y-axis of the default three-panel figure is not applied to
            user-provided ``Axes``), otherwise a single ``Axes``.
            Default: None.
        point_charges (list[tuple[float, ArrayLike]]):
            List of ``(charge, frac_coords)`` pairs specifying multiple
            (Gaussian) model charges for the defect (e.g. for the constituents
            of a defect complex), used in place of the default single charge
            (total defect charge at the defect site), as with multiple charges
            in ``sxdefectalign``. The charges must sum to the defect charge
            state, and are unwrapped to a single (minimum image) configuration.
            The potential alignment along each axis is then sampled furthest
            from all charges. Default is ``None`` (single charge).
        **kwargs:
            Additional keyword arguments for the FNV correction; ``q_model``
            (``pymatgen-analysis-defects`` ``QModel`` object for the model
            charge(s); Gaussian(s) with ``beta`` = 1 bohr by default),
            ``energy_cutoff`` and ``mad_tol`` (maximum reciprocal space energy
            cutoff, in eV, and convergence tolerance, in Ha, for the periodic
            energy of the model charge; 1500 and 1e-6 by default) and
            ``defect_frac_coords``.

    Returns:
        ``CorrectionResults`` (summary of the corrections applied and
        metadata), and the ``matplotlib`` ``Figure`` object (or axis object if
        axis specified) if ``plot`` is ``True``.
    """
    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    if hasattr(defect_entry, "calculation_metadata"):
        _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    if dielectric is None:
        dielectric = _get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")

    defect_locpot = defect_locpot or _get_and_check_metadata(
        defect_entry, "defect_locpot_dict", "Defect LOCPOT"
    )
    bulk_locpot = bulk_locpot or _get_and_check_metadata(defect_entry, "bulk_locpot_dict", "Bulk LOCPOT")

    defect_locpot = _check_if_pathlike_and_get_locpot_or_core_pots(defect_locpot, obj_type="locpot")
    bulk_locpot = _check_if_pathlike_and_get_locpot_or_core_pots(bulk_locpot, obj_type="locpot")

    lattice = (
        _get_defect_supercell(defect_entry).lattice
        if isinstance(defect_locpot, dict)
        else defect_locpot.structure.lattice
    )
    defect_frac_coords = kwargs.pop(  # _relaxed_ defect coords (except for vacancies) by default
        "defect_frac_coords", _get_defect_supercell_frac_coords(defect_entry)
    )
    site_charges, charge_coords = _check_and_unwrap_point_charges(
        lattice,
        defect_entry.charge_state,
        point_charges or [(defect_entry.charge_state, defect_frac_coords)],
        "FNV",
    )
    fnv_correction = _get_multi_site_freysoldt_correction(
        site_charges, charge_coords, dielectric, defect_locpot, bulk_locpot, lattice, **kwargs
    )
    if point_charges:  # store multiple point charge model, for reproducibility
        fnv_correction.metadata["point_charges"] = [
            (float(q), fc.tolist()) for q, fc in zip(site_charges, charge_coords, strict=True)
        ]

    if verbose:
        print(f"Calculated Freysoldt (FNV) correction is {fnv_correction.correction_energy:.3f} eV")

    if not plot and filename is None:
        return fnv_correction

    axis_label_dict = {0: r"$a$-axis", 1: r"$b$-axis", 2: r"$c$-axis"}
    axes = None if ax is None else list(np.asarray(ax, dtype=object).ravel())  # Axes, or list/array
    if axes is not None and len(axes) != (num_axes := 3 if axis is None else 1):
        raise ValueError(
            f"Got {len(axes)} `Axes` with `ax`, but {num_axes} are required "
            f"({'one per a/b/c axis, as `axis` is None' if axis is None else 'as `axis` is set'})."
        )

    directions = range(3) if axis is None else [axis]
    with doped_plot_style(style_file):
        if axes is None and axis is None:
            _fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 3.5), dpi=600)
        for i, direction in enumerate(directions):
            plotted_ax = plot_FNV(
                fnv_correction.metadata["plot_data"][direction],
                ax=None if axes is None else axes[i],
                title=axis_label_dict[direction],
                style_file=style_file,
            )
    fig = plotted_ax.figure.figure  # root parent figure (of the user-provided ``ax``, if given)

    if filename:
        fig.savefig(filename, bbox_inches="tight", transparent=True)

    return fnv_correction, fig


def plot_FNV(plot_data, title=None, ax=None, style_file=None):
    """
    Plots the planar-averaged electrostatic potential against the long range
    and short range models from the FNV correction method.

    Code templated from the original PyCDT and ``pymatgen.analysis.defects``
    implementations.

    Args:
         plot_data (dict):
            Dictionary of Freysoldt correction metadata to plot
            (i.e. ``defect_entry.corrections_metadata["plot_data"][axis]``
            where ``axis`` is one of [0, 1, 2] specifying which axis to plot
            along (a, b, c)).
         title (str):
            Title for the plot. Default is no title.
         ax (matplotlib.axes.Axes):
            Axes object to plot on. If ``None``, makes new ``Figure``.
         style_file (PathLike):
            Path to a mplstyle file to use for the plot. If ``None`` (default),
            uses the default ``doped`` style (from doped/utils/doped.mplstyle).
    """
    if not plot_data["pot_plot_data"]:
        raise ValueError(
            "Input `plot_data` has no `pot_plot_data` entry. Cannot plot FNV potential "
            "alignment before running correction!"
        )

    x = plot_data["pot_plot_data"]["x"]
    v_R = plot_data["pot_plot_data"]["Vr"]
    diff = plot_data["pot_plot_data"]["dft_diff"]
    short_range = plot_data["pot_plot_data"]["short_range"]
    check = plot_data["pot_plot_data"]["check"]
    C = plot_data["pot_plot_data"]["shift"]

    with doped_plot_style(style_file):
        if ax is None:
            _fig, ax = plt.subplots()
        (line1,) = ax.plot(x, v_R, c="black", zorder=1, label="FNV long-range model ($V_{lr}$)")
        (line2,) = ax.plot(x, diff, c="red", label=r"$\Delta$(Locpot)")
        (line3,) = ax.plot(
            x,
            short_range,
            c="green",
            label=r"$V_{sr}$ = $\Delta$(Locpot) - $V_{lr}$" + "\n(pre-alignment)",
        )
        leg1 = ax.legend(handles=[line1, line2, line3], loc=9)  # middle top legend
        ax.add_artist(leg1)  # so isn't overwritten with later legend call

        line4 = ax.axhline(C, color="k", linestyle="--", label=f"Alignment constant C = {C:.3f} V")
        tmpx = [x[i] for i in range(check[0], check[1])]
        poly_coll = ax.fill_between(tmpx, -100, 100, facecolor="red", alpha=0.15, label="Sampling region")
        ax.legend(handles=[line4, poly_coll], loc=8)  # bottom middle legend

        ax.set_xlim(round(x[0]), round(x[-1]))
        ymin = min(*v_R, *diff, *short_range)
        ymax = max(*v_R, *diff, *short_range)
        ax.set_ylim(-0.2 + ymin, 0.2 + ymax)
        ax.set_xlabel(r"Distance along axis ($\AA$)")
        ax.set_ylabel("Potential (V)")
        ax.axhline(y=0, linewidth=0.2, color="black")
        if title is not None:
            ax.set_title(str(title))
        ax.set_xlim(0, max(x))

        return ax


MIN_EFNV_SAMPLED_GROUPS = 8  # minimum number of inequivalent sites in the default eFNV sampling region


def _get_equivalent_site_groups(
    species, distances, potentials, dist_tol: float = 0.02, pot_tol: float = 2e-3
) -> np.ndarray:
    """
    Group atomic sites which are (near-)equivalent for the eFNV potential
    sampling.

    Sites of the same species, at the same distance from the (nearest) defect
    charge (within ``dist_tol``, in Å) and with the same site potential
    difference (within ``pot_tol``, in V) are taken to be equivalent, and so
    give only one `independent` sample of the far-field potential.

    Args:
        species (Sequence[str]):
            Species (strings) of the sites.
        distances (Sequence[float]):
            Distances (in Å) of the sites to the (nearest) defect charge.
        potentials (Sequence[float]):
            Site potential differences (defect - bulk; in V).
        dist_tol (float):
            Distance tolerance (in Å). Default is 0.02.
        pot_tol (float):
            Potential tolerance (in V). Default is 2e-3.

    Returns:
        np.ndarray:
            Integer group label for each site.
    """
    keys = [
        (str(specie), round(float(d) / dist_tol), round(float(pot) / pot_tol))
        for specie, d, pot in zip(species, distances, potentials, strict=True)
    ]
    labels: dict = {}
    return np.array([labels.setdefault(key, len(labels)) for key in keys], dtype=int)


def get_efnv_sampled_group_means(efnv_correction) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the mean sampled potential difference (``diff_pot``; in V) and
    multiplicity (number of sites) of each group of equivalent sites in the
    sampling region of a ``pydefect`` ``ExtendedFnvCorrection`` object (see
    ``_get_equivalent_site_groups``).

    The scatter of these values gives the sampling noise in the eFNV potential
    alignment (see ``get_efnv_alignment_error``), which is used to estimate a
    lower bound on the error in the charge correction (itself a lower bound on
    the remnant finite-size error).

    Args:
        efnv_correction (ExtendedFnvCorrection):
            ``pydefect`` ``ExtendedFnvCorrection`` object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Mean ``diff_pot`` (in V) and multiplicity for each group of
            equivalent sites.
    """
    sampled = [s for s in efnv_correction.sites if s.distance > efnv_correction.defect_region_radius]
    if not sampled:
        return np.array([]), np.array([], dtype=int)
    groups = _get_equivalent_site_groups(
        [s.specie for s in sampled], [s.distance for s in sampled], [s.potential for s in sampled]
    )
    diff_pots = np.array([s.diff_pot for s in sampled])
    unique_groups, multiplicities = np.unique(groups, return_counts=True)
    return np.array([diff_pots[groups == g].mean() for g in unique_groups]), multiplicities


def get_efnv_alignment_error(efnv_correction) -> float:
    """
    Estimate the sampling error (in V) in the eFNV potential alignment of a
    ``pydefect`` ``ExtendedFnvCorrection`` object, as the standard error of the
    mean sampled potential difference (``diff_pot``).

    Equivalent sites (see ``_get_equivalent_site_groups``) are not independent
    samples of the far-field potential, so the standard error is evaluated over
    the groups of equivalent sites, weighted by their multiplicities ``w`` (as
    the alignment is the mean over all sampled sites); the weighted standard
    deviation of the group means, divided by ``sqrt(n_eff - 1)`` where
    ``n_eff = sum(w)^2 / sum(w^2)`` is the effective sample size. This reduces
    to the usual standard error of the mean when no sites are equivalent, which
    is also used if all sampled sites are equivalent.

    Multiplied by the defect charge, this gives a lower bound on the error in
    the charge correction (itself a lower bound on the remnant finite-size
    error).

    Args:
        efnv_correction (ExtendedFnvCorrection):
            ``pydefect`` ``ExtendedFnvCorrection`` object.

    Returns:
        float:
            Estimated sampling error in the potential alignment (in V); ``nan``
            if fewer than two sites are sampled.
    """
    means, weights = get_efnv_sampled_group_means(efnv_correction)
    if len(means) < 2:  # all sampled sites are equivalent, so can only use the scatter between them
        sites = efnv_correction.sites
        means = np.array([s.diff_pot for s in sites if s.distance > efnv_correction.defect_region_radius])
        weights = np.ones(len(means))
    if len(means) < 2:
        return np.nan
    n_eff = weights.sum() ** 2 / np.sum(weights**2)
    variance = np.average((means - np.average(means, weights=weights)) ** 2, weights=weights)
    return float(np.sqrt(variance / (n_eff - 1)))


def get_kumagai_correction(
    defect_entry,
    dielectric: float | np.ndarray | list | None = None,
    point_charges: list[tuple[float, np.ndarray | list]] | None = None,
    defect_region_radius: float | None = None,
    excluded_indices: list[int] | None = None,
    defect_outcar: PathLike | Outcar | None = None,
    bulk_outcar: PathLike | Outcar | None = None,
    plot: bool = False,
    filename: PathLike | None = None,
    verbose: bool = True,
    style_file: PathLike | None = None,
    **kwargs,
) -> CorrectionResult:
    """
    Function to compute the Kumagai (eFNV) finite-size charge correction for
    the input defect_entry. Compatible with both isotropic/cubic and
    anisotropic systems.

    This function `does not` add the correction to ``defect_entry.corrections``
    (but the defect_entry.get_kumagai_correction method does). If this
    correction is used, please cite the Kumagai & Oba (eFNV) paper:
    10.1103/PhysRevB.89.195205 and the ``pydefect`` paper: "Insights into
    oxygen vacancies from high-throughput first-principles calculations" Yu
    Kumagai, Naoki Tsunoda, Akira Takahashi, and Fumiyasu Oba Phys. Rev.
    Materials 5, 123803 (2021) -- 10.1103/PhysRevMaterials.5.123803

    Typically for reasonably well-converged supercell sizes, the default
    ``defect_region_radius`` works perfectly well. However, for certain
    materials at small/intermediate supercell sizes, you may want to adjust
    this (and/or ``excluded_indices``) to ensure the best sampling of the
    plateau region away from the defect position -- ``doped`` should throw a
    warning in these cases (about the correction error being above the default
    tolerance (25 meV)). For example, with layered materials, the defect charge
    is often localised to one layer, so we may want to adjust
    ``defect_region_radius`` and/or ``excluded_indices`` to ensure that only
    sites in other layers are used for the sampling region (plateau) -- see
    example on doped docs Tips page.

    The defect coordinates are taken as the relaxed defect site by default
    (``DefectEntry.defect_supercell_site``) -- which is the bulk site for
    vacancies, but this can be overridden with the ``defect_coords`` keyword
    argument.

    As a general rule of thumb, the charge correction terms should follow
    relatively consistent trends in terms of magnitudes. A large outlier
    (easily scanned with |get_formation_energies|) often indicates something
    unusual/unexpected. See the FNV/eFNV and other finite-size charge
    correction papers for further details.

    Args:
        defect_entry (|DefectEntry|):
            |DefectEntry| object for which to compute the Kumagai finite-size
            charge correction.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constants (ionic + static contributions), in the
            same xyz Cartesian basis as the supercell calculations (likely but
            not necessarily the same as the raw output of a VASP dielectric
            calculation, if an oddly-defined primitive cell is used). If
            ``None``, then the dielectric constant is taken from the
            ``defect_entry`` ``calculation_metadata`` if available. See the
            |Dielectric Constant| tutorial section for information on
            calculating and converging the dielectric constant.
        point_charges (list[tuple[float, ArrayLike]]):
            List of ``(charge, frac_coords)`` pairs specifying multiple point
            charges for the defect (e.g. for the constituents of a defect
            complex), used in place of the default single point charge (total
            defect charge at ``DefectEntry.defect_supercell_site`` /
            ``defect_coords``). The charges must sum to the defect charge
            state. The point charges are unwrapped to a single (minimum image)
            configuration, and the potential sampling region is then defined by
            the distance to the nearest point charge (so
            ``defect_region_radius`` applies around each). Default is ``None``
            (single point charge).
        defect_region_radius (float):
            Radius of the defect region (in Å). Sites outside the defect region
            are used for sampling the electrostatic potential far from the
            defect (to obtain the potential alignment). If ``None`` (default),
            uses half the largest distance between parallel cell faces (as in
            ``pydefect``), reduced as necessary to include at least 8
            inequivalent sites in the sampling region (or a quarter of all
            inequivalent sites if fewer, with a minimum of 3 sites). Sites are
            taken as equivalent if they have the same species, distance to the
            (nearest) defect charge and site potential difference.
        excluded_indices (list):
            List of site indices (in the defect supercell) to exclude from
            the site potential sampling in the correction calculation/plot.
            If ``None`` (default), no sites are excluded.
        defect_outcar (PathLike or |Outcar|):
            Path to the output ``VASP`` ``OUTCAR`` file from the defect
            supercell calculation, or the corresponding ``pymatgen`` |Outcar|
            object. If ``None``, will try to use the ``defect_site_potentials``
            from the ``defect_entry`` ``calculation_metadata`` if available.
        bulk_outcar (PathLike or |Outcar|):
            Path to the output ``VASP`` ``OUTCAR`` file from the bulk supercell
            calculation, or the corresponding ``pymatgen`` |Outcar| object.
            If ``None``, will try to use the ``bulk_site_potentials``
            from the ``defect_entry`` ``calculation_metadata`` if available.
        plot (bool):
            Whether to plot the Kumagai site potential plots (for
            manually checking the behaviour of the charge correction here).
        filename (PathLike):
            Filename to save the Kumagai site potential plots to.
            If None, plots are not saved.
        verbose (bool):
            Whether to print the correction energy (default = True).
        style_file (PathLike):
            Path to a mplstyle file to use for the plot. If ``None`` (default),
            uses the default ``doped`` style (from doped/utils/doped.mplstyle).
        **kwargs:
            Additional kwargs for the eFNV correction calculation (e.g.
            ``defect_coords``, ``accuracy``).

    Returns:
        ``CorrectionResults`` (summary of the corrections applied and
        metadata), and the ``matplotlib`` ``Figure`` object if ``plot`` is
        ``True``.
    """
    with vise_handling():  # avoid vise issues (warning suppression, logging, Windows bug)
        from pydefect.analyzer.calc_results import CalcResults
        from pydefect.corrections.efnv_correction import ExtendedFnvCorrection, PotentialSite
        from pydefect.corrections.ewald import Ewald
        from pydefect.corrections.site_potential_plotter import SitePotentialMplPlotter
        from pydefect.defaults import defaults
        from pydefect.util.error_classes import SupercellError

    def doped_make_efnv_correction(
        charge: float,
        calc_results: CalcResults,
        perfect_calc_results: CalcResults,
        dielectric_tensor: np.ndarray,
        defect_coords: np.ndarray | list,
        defect_region_radius: float | None = None,
        accuracy: float = defaults.ewald_accuracy,
        unit_conversion: float = 180.95128169876497,
        excluded_indices: list | None = None,
        point_charges: list[tuple[float, np.ndarray | list]] | None = None,
    ):
        r"""
        This is a modified version of ``pydefect``\'s ``make_efnv_correction``
        function (in ``pydefect.cli.vasp.make_efnv_correction``) to allow the
        defect region radius to be adjusted (e.g. in cases of layered
        materials, where often the defect charge is localised to one layer, so
        we likely want to adjust the defect region radius to ensure that only
        `other` layers are used for the sampling (plateau) region), and to
        allow the model charge to be split over multiple point charges (e.g.
        over the constituent point defects of a defect complex), with the
        sampling region then defined by the distance to the nearest point
        charge.

        If ``defect_region_radius`` is not specified, then the ``pydefect``
        default (half the largest distance between parallel cell faces) is
        used, reduced as necessary to include at least 8 inequivalent sites in
        the sampling region (or a quarter of all inequivalent sites if fewer,
        with a minimum of 3 sites).
        """
        if calc_results.structure.lattice != perfect_calc_results.structure.lattice:
            raise SupercellError("The lattice constants for defect and perfect models are different")
        lattice = calc_results.structure.lattice
        sites, rel_coords = [], []

        site_charges, charge_coords = _check_and_unwrap_point_charges(
            lattice, charge, point_charges or [(charge, defect_coords)]
        )

        excluded_indices = [] if excluded_indices is None else [int(i) for i in excluded_indices]

        mapping = get_site_mappings(
            calc_results.structure, perfect_calc_results.structure, threshold=np.inf
        )

        for _d_to_p_dist, d, p in mapping:
            if d not in excluded_indices and not any(i is None for i in [d, p]):
                specie = str(calc_results.structure[d].specie)
                frac_coords = calc_results.structure[d].frac_coords
                # sampling distance is to the nearest point charge (i.e. the defect site, by default):
                distance = min(lattice.get_distance_and_image(c, frac_coords)[0] for c in charge_coords)
                pot = calc_results.potentials[d] - perfect_calc_results.potentials[p]
                sites.append(PotentialSite(specie, distance, pot, None))
                rel_coords.append((frac_coords - charge_coords).tolist())  # (N_charges, 3)

        # evaluate the Ewald sums in the eps^-1/2-scaled frame, where the dielectric is the identity (with
        # potentials scaled by 1/sqrt(det(eps))), for efficiency and accuracy (w/pydefect Ewald defaults)
        eigvals, eigvecs = np.linalg.eigh((dielectric_tensor + dielectric_tensor.T) / 2)
        ewald = Ewald(lattice.matrix @ (eigvecs * eigvals**-0.5) @ eigvecs.T, np.eye(3), accuracy=accuracy)
        unit_conversion /= np.sqrt(np.prod(eigvals))

        # Monkey-patch pydefect Ewald functions to vectorise the calculations (otherwise can be slow):
        from scipy.special import erfc

        def ewald_real(self, include_self: bool, shift: list[float]) -> float:
            R = np.asarray(list(self.r_lattice_set(include_self, shift)))  # (N, 3)
            rMr = np.einsum("ij,jk,ik->i", R, self.epsilon_inv, R)
            root = np.sqrt(rMr)
            return np.sum(erfc(self.mod_ewald_param * root) / root) / (4 * np.pi * self.root_epsilon)

        def ewald_rec(self, coord) -> float:
            cart_coord = coord @ self.lattice
            G = np.asarray(list(self.g_lattice_set()))  # (N, 3)
            gEg = np.einsum("ij,jk,ik->i", G, self.dielectric_tensor, G)
            phase = np.cos(G @ cart_coord)
            return np.sum(np.exp(-gEg / (4 * self.mod_ewald_param**2)) / gEg * phase) / self.volume

        # (multi-site) PC correction: -E_latt·Σqᵢ² - Σ_{i<j} qᵢqⱼ[φ_per(r_ij) - φ_iso(r_ij)]
        Ewald.ewald_real = ewald_real
        Ewald.ewald_rec = ewald_rec
        point_charge_correction = (  # -E_latt·Σqᵢ²
            -ewald.lattice_energy * sum(q**2 for q in site_charges) if any(site_charges) else 0.0
        )
        for i, j in combinations(range(len(site_charges)), 2):  # multiple charges: pair interactions
            rel_ij = (charge_coords[j] - charge_coords[i]).tolist()
            # direct (isolated) Coulomb interaction of the pair, subtracted from the periodic interaction:
            isolated_pot = 1 / (4 * np.pi * np.linalg.norm(np.dot(rel_ij, ewald.lattice)))  # eps-scaled
            point_charge_correction -= (
                site_charges[i] * site_charges[j] * (ewald.atomic_site_potential(rel_ij) - isolated_pot)
            )  # qᵢqⱼ[φ_per(r_ij) - φ_iso(r_ij)]

        site_distances = np.array([site.distance for site in sites])
        if defect_region_radius is None:
            defect_region_radius = get_half_max_face_distance(lattice)
            # ensure a minimum number of _inequivalent_ sites in the sampling region, as the default radius
            # can leave just a few (often equivalent) sites for e.g. elongated supercells, extended
            # (multi-site) defects or slabs with vacuum, giving a noisy potential alignment and unreliable
            # error estimate. The 8 furthest groups (or a quarter of all groups, for small/high-symmetry
            # cells) and the 3 furthest sites must be sampled:
            groups = _get_equivalent_site_groups(
                [site.specie for site in sites], site_distances, [site.potential for site in sites]
            )
            group_distances = np.sort([site_distances[groups == g].max() for g in np.unique(groups)])
            min_groups = min(MIN_EFNV_SAMPLED_GROUPS, int(np.ceil(len(group_distances) / 4)))
            furthest_sites = np.sort(site_distances)[-3:]
            min_distance = min([*group_distances[-min_groups:], *furthest_sites], default=np.inf)
            if min_distance <= defect_region_radius:  # reduce to just below it:
                if furthest_sites[0] <= defect_region_radius:  # unusual; e.g. slabs with vacuum
                    warnings.warn(
                        f"Fewer than {len(furthest_sites)} atomic sites lie outside the default eFNV "
                        f"defect region radius, so it has been reduced to {min_distance - 1e-3:.2f} Å to "
                        f"include at least {len(furthest_sites)} sites in the potential sampling region."
                    )
                defect_region_radius = min_distance - 1e-3  # (with tolerance for equal distances)

        elif (num_sampled := np.count_nonzero(site_distances > defect_region_radius)) == 0:
            raise ValueError(
                f"No atomic sites lie outside the supplied eFNV `defect_region_radius` "
                f"({defect_region_radius:.2f} Å), so the potential alignment cannot be computed. Use a "
                f"smaller `defect_region_radius`, or the default (None)."
            )
        elif num_sampled < 3:
            warnings.warn(
                f"Only {num_sampled} atomic site{'s lie' if num_sampled > 1 else ' lies'} outside the "
                f"supplied eFNV `defect_region_radius` ({defect_region_radius:.2f} Å), so the potential "
                f"alignment may be unreliable."
            )

        for site, rel_coord in zip(sites, rel_coords, strict=False):
            if site.distance > defect_region_radius:
                if not any(site_charges):
                    site.pc_potential = 0
                else:
                    site.pc_potential = (
                        sum(
                            q * ewald.atomic_site_potential(rc)
                            for q, rc in zip(site_charges, rel_coord, strict=True)
                        )
                        * unit_conversion
                    )

        return ExtendedFnvCorrection(
            charge=charge,
            point_charge_correction=point_charge_correction * unit_conversion,
            defect_region_radius=defect_region_radius,
            sites=sites,
            defect_coords=tuple(charge_coords.mean(axis=0)),  # centroid (defect site, by default)
        )

    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    if hasattr(defect_entry, "calculation_metadata"):
        _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    if dielectric is None:
        dielectric = _get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")

    core_potentials_dict = {}
    for key, outcar in zip(["defect", "bulk"], [defect_outcar, bulk_outcar], strict=False):
        total_energy = []
        with contextlib.suppress(Exception):
            total_energy.append(
                defect_entry.sc_entry.energy if key == "defect" else defect_entry.bulk_entry.energy
            )
            total_energy.append(
                defect_entry.calculation_metadata["run_metadata"][f"{key}_vasprun_dict"]["output"][
                    "ionic_steps"
                ][-1]["electronic_steps"][-1]["e_0_energy"]
            )

        if outcar is not None:
            core_potentials_dict[key] = _check_if_pathlike_and_get_locpot_or_core_pots(
                outcar, obj_type="outcar", dir_type=key, total_energy=total_energy
            )
        else:
            core_potentials_dict[key] = _get_and_check_metadata(
                defect_entry,
                f"{key}_site_potentials",
                f"{key.capitalize()} OUTCAR (for atomic site potentials)",
            )

    defect_supercell = _get_defect_supercell(defect_entry).copy()
    defect_supercell.remove_oxidation_states()  # pydefect needs structure without oxidation states
    defect_calc_results_for_eFNV = CalcResults(
        structure=defect_supercell,
        energy=np.inf,
        magnetization=np.inf,
        potentials=core_potentials_dict["defect"],
    )

    bulk_supercell = _get_bulk_supercell(defect_entry).copy()
    bulk_supercell.remove_oxidation_states()  # pydefect needs structure without oxidation states
    if bulk_supercell.lattice != defect_supercell.lattice:  # pydefect will crash
        # check if the difference is tolerable (< 0.01 Å)
        if not np.allclose(bulk_supercell.lattice.matrix, defect_supercell.lattice.matrix, atol=1e-2):
            warnings.warn(
                f"Bulk and defect supercells have different lattices, and so the eFNV (Kumagai) "
                f"correction may be unreliable!"
                f"\nBulk lattice:\n{bulk_supercell.lattice}\nDefect lattice:\n{defect_supercell.lattice}"
            )
        # scale bulk lattice to match defect lattice, so pydefect doesn't crash:
        bulk_supercell.lattice = defect_supercell.lattice

    bulk_calc_results_for_eFNV = CalcResults(
        structure=bulk_supercell,
        energy=np.inf,
        magnetization=np.inf,
        potentials=core_potentials_dict["bulk"],
    )

    # _relaxed_ defect coords (except for vacancies) by default:
    defect_coords = kwargs.pop("defect_coords", _get_defect_supercell_frac_coords(defect_entry))
    efnv_correction = doped_make_efnv_correction(
        charge=defect_entry.charge_state,
        calc_results=defect_calc_results_for_eFNV,
        perfect_calc_results=bulk_calc_results_for_eFNV,
        dielectric_tensor=freysoldt._get_dielectric_tensor(dielectric),
        defect_coords=defect_coords,
        defect_region_radius=defect_region_radius,
        excluded_indices=excluded_indices,
        point_charges=point_charges,
        **kwargs,
    )
    kumagai_correction_result = CorrectionResult(
        correction_energy=efnv_correction.correction_energy,
        metadata={"pydefect_ExtendedFnvCorrection": efnv_correction},
    )
    if point_charges:  # store multiple point charge model, for reproducibility
        kumagai_correction_result.metadata["point_charges"] = [
            (float(q), np.asarray(fc, dtype=float).tolist()) for q, fc in point_charges
        ]

    if verbose:
        print(
            f"Calculated Kumagai (eFNV) correction is {kumagai_correction_result.correction_energy:.3f} eV"
        )

    if not plot and filename is None:
        return kumagai_correction_result

    spp = SitePotentialMplPlotter.from_efnv_corr(
        title=f"{format_defect_name(defect_entry.name, False)} -- eFNV Site Potentials",
        efnv_correction=efnv_correction,
    )
    with doped_plot_style(style_file):
        plt.figure()  # new current figure for pydefect's pyplot-based plotting
        spp.construct_plot()
        fig = spp.plt.gcf()
        ax = fig.gca()

        # reformat plot slightly:
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        # shade in sampling region:
        spp.plt.fill_between(
            [spp.defect_region_radius, x_lims[1]],
            *y_lims,
            color="purple",
            alpha=0.15,
            label="Sampling Region",
        )
        spp.plt.ylim(*y_lims)  # reset y-lims after fill_between

        # update legend:
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            label.replace("point charge", "Point Charge (PC)").replace(
                "potential difference", r"$\Delta V$"
            )
            for label in labels
        ]
        labels = [
            label + r" ($V_{defect} - V_{bulk}$)" if Element.is_valid_symbol(label) else label
            for label in labels
        ]

        # add entry for dashed red line:
        handles += [Line2D([0], [0], **spp._mpl_defaults.hline)]
        labels += [
            rf"Avg. $\Delta V$ = {spp.ave_pot_diff:.3f} V",
        ]
        ax.legend_.remove()
        ax.legend(handles, labels, loc="best", borderaxespad=0, fontsize=8)
        x_label = "Distance from nearest PC" if point_charges else "Distance from defect"
        ax.set_xlabel(f"{x_label} ({spp._x_unit})", size=spp._mpl_defaults.label_font_size)

    if filename:
        fig.savefig(filename, bbox_inches="tight", transparent=True)

    return kumagai_correction_result, fig
