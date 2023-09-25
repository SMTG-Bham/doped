"""
Code to compute finite-size charge corrections for charged defects in periodic
systems.

The charge-correction methods implemented are:
1) Extended FNV (eFNV) / Kumagai correction for isotropic and anistropic systems.
   This is the recommended default as it works regardless of system (an)isotropy, has a more efficient
   implementation and tends to be more robust in cases of small supercells.
   Includes:
       a) anisotropic PC energy
       b) potential alignment by atomic site averaging outside Wigner Seitz radius

2) Freysoldt (FNV) correction for isotropic systems. Only recommended if eFNV correction fails for some
   reason. Includes:
       a) point-charge (PC) energy
       b) potential alignment by planar averaging

If you use the corrections implemented in this module, cite:
    Kumagai and Oba, Phys. Rev. B. 89, 195205 (2014) for the eFNV correction
    or
    Freysoldt, Neugebauer, and Van de Walle, Phys. Status Solidi B. 248, 1067-1076 (2011) for FNV
"""

import logging
import os
import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from monty.json import MontyDecoder
from pymatgen.analysis.defects.corrections import freysoldt
from pymatgen.analysis.defects.utils import CorrectionResult
from pymatgen.io.vasp.outputs import Locpot, Outcar
from shakenbreak.plotting import _install_custom_font

from doped.analysis import _convert_dielectric_to_tensor
from doped.plotting import _format_defect_name, _get_backend
from doped.utils.parsing import get_locpot, get_outcar

warnings.simplefilter("default")
# `message` only needs to match start of message:
warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`")
warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")


def _monty_decode_nested_dicts(d):
    """
    Recursively find any dictionaries in defect_entry.calculation_metadata,
    which may be nested in dicts or in lists of dicts, and decode them.
    """
    for key, value in d.items():
        if isinstance(value, dict) and all(k not in value for k in ["@module", "@class"]):
            _monty_decode_nested_dicts(value)
        elif (
            isinstance(value, list)
            and all(isinstance(i, dict) for i in value)
            and all(k in i for k in ["@module", "@class"] for i in value)
        ):
            try:
                d[key] = [MontyDecoder().process_decoded(i) for i in value]
            except Exception as exc:
                print(f"Failed to decode {key} with error {exc}")
        if isinstance(value, dict) and all(k in value for k in ["@module", "@class"]):
            try:
                d[key] = MontyDecoder().process_decoded(value)
            except Exception as exc:
                print(f"Failed to decode {key} with error {exc}")


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


def _check_if_str_and_get_pmg_obj(locpot_or_outcar, obj_type="locpot"):
    if isinstance(locpot_or_outcar, str):
        if obj_type == "locpot":
            return get_locpot(locpot_or_outcar)
        return get_outcar(locpot_or_outcar)

    if not isinstance(locpot_or_outcar, [Locpot, Outcar]):
        raise TypeError(
            f"`{obj_type}` input must be either a path to a {obj_type.upper()} file or a pymatgen "
            f"{obj_type.upper()[0]+obj_type[1:]} object, object, but got {type(locpot_or_outcar)} instead."
        )

    return locpot_or_outcar


def get_freysoldt_correction(
    defect_entry,
    dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
    defect_locpot: Optional[Union[str, Locpot, dict]] = None,
    bulk_locpot: Optional[Union[str, Locpot, dict]] = None,
    plot: bool = False,
    filename: Optional[str] = None,
    axis: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> CorrectionResult:
    """
    Function to compute the _isotropic_ Freysoldt (FNV) correction for the
    input defect_entry.

    This function _does not_ add the correction to `defect_entry.corrections`
    (but the defect_entry.get_freysoldt_correction method does).
    If this correction is used, please cite Freysoldt's
    original paper; 10.1103/PhysRevLett.102.016402.

    Args:
        defect_entry:
            DefectEntry object with the following for which to compute the
            FNV finite-size charge correction.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constant of the host compound (including both
            ionic and (high-frequency) electronic contributions). If None,
            then the dielectric constant is taken from the `defect_entry`
            `calculation_metadata` if available.
        defect_locpot:
            Path to the output VASP LOCPOT file from the defect supercell
            calculation, or the corresponding pymatgen Locpot object, or
            a dictionary of the planar-averaged potential.
            If None, will try to use `defect_locpot` from the
            `defect_entry` `calculation_metadata` if available.
        bulk_locpot:
            Path to the output VASP LOCPOT file from the bulk supercell
            calculation, or the corresponding pymatgen Locpot object, or
            a dictionary of the planar-averaged potential.
            If None, will try to use `bulk_locpot` from the
            `defect_entry` `calculation_metadata` if available.
        plot (bool):
            Whether to plot the FNV electrostatic potential plots (for
            manually checking the behaviour of the charge correction here).
        filename (str):
            Filename to save the FNV electrostatic potential plots to.
            If None, plots are not saved.
        axis (int or None):
            If int, then the FNV electrostatic potential plot along the
            specified axis (0, 1, 2 for a, b, c) will be plotted. Note that
            the output charge correction is still that for _all_ axes.
            If None, then all three axes are plotted.
        verbose (bool):
            Whether to print the correction energy (default = True).
        **kwargs:
            Additional kwargs to pass to
            pymatgen.analysis.defects.corrections.freysoldt.get_freysoldt_correction
            (e.g. energy_cutoff, mad_tol, q_model, step).

    Returns:
        CorrectionResults (summary of the corrections applied and metadata), and
        the matplotlib figure object (or axis object if axis specified) if `plot`
        or `saved` is True.
    """
    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    dielectric = dielectric or _get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")
    dielectric = _convert_dielectric_to_tensor(dielectric)

    defect_locpot = defect_locpot or _get_and_check_metadata(
        defect_entry, "defect_locpot", "Defect LOCPOT"
    )
    bulk_locpot = bulk_locpot or _get_and_check_metadata(defect_entry, "bulk_locpot", "Bulk LOCPOT")

    defect_locpot = _check_if_str_and_get_pmg_obj(defect_locpot, obj_type="locpot")
    bulk_locpot = _check_if_str_and_get_pmg_obj(bulk_locpot, obj_type="locpot")

    fnv_correction = freysoldt.get_freysoldt_correction(
        q=defect_entry.charge_state,
        dielectric=dielectric,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        defect_frac_coords=defect_entry.sc_defect_frac_coords,
        **kwargs,
    )

    if not plot and filename is None:
        return fnv_correction

    _install_custom_font()

    axis_label_dict = {0: r"$a$-axis", 1: r"$b$-axis", 2: r"$c$-axis"}
    if axis is None:
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 3.5), dpi=600)
        for direction in range(3):
            plot_FNV(
                fnv_correction.metadata["plot_data"][direction],
                ax=axs[direction],
                title=axis_label_dict[direction],
            )
    else:
        fig = plot_FNV(fnv_correction.metadata["plot_data"][axis], title=axis_label_dict[axis])
        # actually an axis object

    if filename:
        plt.savefig(filename, bbox_inches="tight", transparent=True, backend=_get_backend(filename))
    else:
        plt.show()

    if verbose:
        print(f"Calculated Freysoldt (FNV) correction is {fnv_correction.correction_energy:.3f} eV")

    return fnv_correction, fig


def plot_FNV(plot_data, title=None, ax=None):
    """
    Plots the planar-averaged electrostatic potential against the long range
    and short range models from the FNV correction method.

    Code templated from the original PyCDT and new pymatgen.analysis.defects
    implementations.

    Args:
         plot_data (dict):
            Dictionary of Freysoldt correction metadata to plot
            (i.e. defect_entry.corrections_metadata["plot_data"][axis] where
            axis is one of [0, 1, 2] specifying which axis to plot along (a, b, c)).
         title (str): Title for the plot. Default is no title.
         ax (matplotlib.axes.Axes): Axes object to plot on. If None, makes new figure.
    """
    if not plot_data["pot_plot_data"]:
        raise ValueError(
            "Input `plot_data` has no `pot_plot_data` entry. Cannot plot FNV potential "
            "alignment before running correction!"
        )

    x = plot_data["pot_plot_data"]["x"]
    v_R = plot_data["pot_plot_data"]["Vr"]
    dft_diff = plot_data["pot_plot_data"]["dft_diff"]
    short_range = plot_data["pot_plot_data"]["short_range"]
    check = plot_data["pot_plot_data"]["check"]
    C = plot_data["pot_plot_data"]["shift"]

    with plt.style.context(f"{os.path.dirname(__file__)}/doped.mplstyle"):
        if ax is None:
            fig, ax = plt.subplots()
        (line1,) = ax.plot(x, v_R, c="black", zorder=1, label="FNV long-range model ($V_{lr}$)")
        (line2,) = ax.plot(x, dft_diff, c="red", label=r"$\Delta$(Locpot)")
        (line3,) = ax.plot(
            x,
            short_range,
            c="green",
            label=r"$V_{sr}$ = $\Delta$(Locpot) - $V_{lr}$" + "\n(pre-alignment)",  # noqa: ISC003
        )
        leg1 = ax.legend(handles=[line1, line2, line3], loc=9)  # middle top legend
        ax.add_artist(leg1)  # so isn't overwritten with later legend call

        line4 = ax.axhline(C, color="k", linestyle="--", label=f"Alignment constant C = {C:.3f} V")
        tmpx = [x[i] for i in range(check[0], check[1])]
        poly_coll = ax.fill_between(tmpx, -100, 100, facecolor="red", alpha=0.15, label="Sampling region")
        ax.legend(handles=[line4, poly_coll], loc=8)  # bottom middle legend

        ax.set_xlim(round(x[0]), round(x[-1]))
        ymin = min(*v_R, *dft_diff, *short_range)
        ymax = max(*v_R, *dft_diff, *short_range)
        ax.set_ylim(-0.2 + ymin, 0.2 + ymax)
        ax.set_xlabel(r"Distance along axis ($\AA$)")
        ax.set_ylabel("Potential (V)")
        ax.axhline(y=0, linewidth=0.2, color="black")
        if title is not None:
            ax.set_title(str(title))
        ax.set_xlim(0, max(x))

        return ax


def get_kumagai_correction(
    defect_entry,
    dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
    defect_outcar: Optional[Union[str, Outcar]] = None,
    bulk_outcar: Optional[Union[str, Outcar]] = None,
    plot: bool = False,
    filename: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> CorrectionResult:
    """
    Function to compute the Kumagai (eFNV) finite-size charge correction for
    the input defect_entry. Compatible with both isotropic/cubic and
    anisotropic systems.

    This function _does not_ add the correction to `defect_entry.corrections`
    (but the defect_entry.get_kumagai_correction method does).
    If this correction is used, please cite the Kumagai & Oba paper:
    10.1103/PhysRevB.89.195205

    Args:
        defect_entry:
            DefectEntry object with the following for which to compute the
            Kumagai finite-size charge correction.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constant of the host compound (including both
            ionic and (high-frequency) electronic contributions). If None,
            then the dielectric constant is taken from the `defect_entry`
            `calculation_metadata` if available.
        defect_outcar:
            Path to the output VASP OUTCAR file from the defect supercell
            calculation, or the corresponding pymatgen Outcar object.
            If None, will try to use the `defect_supercell_site_potentials`
            from the `defect_entry` `calculation_metadata` if available.
        bulk_outcar:
            Path to the output VASP OUTCAR file from the bulk supercell
            calculation, or the corresponding pymatgen Outcar object.
            If None, will try to use the `bulk_supercell_site_potentials`
            from the `defect_entry` `calculation_metadata` if available.
        plot (bool):
            Whether to plot the Kumagai site potential plots (for
            manually checking the behaviour of the charge correction here).
        filename (str):
            Filename to save the Kumagai site potential plots to.
            If None, plots are not saved.
        verbose (bool):
            Whether to print the correction energy (default = True).
        **kwargs:
            Additional kwargs to pass to
            pydefect.corrections.efnv_correction.ExtendedFnvCorrection
            (e.g. charge, defect_region_radius, defect_coords).

    Returns:
        CorrectionResults (summary of the corrections applied and metadata), and
        the matplotlib figure object if `plot` or `saved` is True.
    """
    # suppress pydefect INFO messages
    from vise import user_settings  #

    user_settings.logger.setLevel(logging.CRITICAL)
    from pydefect.analyzer.calc_results import CalcResults
    from pydefect.cli.vasp.make_efnv_correction import make_efnv_correction
    from pydefect.corrections.site_potential_plotter import SitePotentialMplPlotter

    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    dielectric = dielectric or _get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")
    dielectric = _convert_dielectric_to_tensor(dielectric)

    def _raise_incomplete_outcar_error(outcar, dir_type="bulk"):
        """
        Raise error about supplied OUTCAR not having atomic core potential
        info.

        Input outcar is either a path or a pymatgen Outcar object
        """
        outcar_info = f"OUTCAR at {outcar}" if isinstance(outcar, str) else "OUTCAR object"
        raise ValueError(
            f"Unable to parse atomic core potentials from {dir_type} {outcar_info}. This can happen if "
            f"`ICORELEVEL` was not set to 0 (= default) in the `INCAR`, or if the calculation was "
            f"finished prematurely with a `STOPCAR`. The Kumagai charge correction cannot be computed "
            f"without this data!"
        )

    if defect_outcar is not None:
        defect_outcar = _check_if_str_and_get_pmg_obj(defect_outcar, obj_type="outcar")
        if defect_outcar.electrostatic_potential is None:
            _raise_incomplete_outcar_error(defect_outcar, dir_type="defect")
        defect_site_potentials = -1 * np.array(defect_outcar.electrostatic_potential)
    else:
        defect_site_potentials = _get_and_check_metadata(
            defect_entry, "defect_supercell_site_potentials", "Defect OUTCAR (for atomic site potentials)"
        )

    if bulk_outcar is not None:
        bulk_outcar = _check_if_str_and_get_pmg_obj(bulk_outcar, obj_type="outcar")
        if bulk_outcar.electrostatic_potential is None:
            _raise_incomplete_outcar_error(bulk_outcar, dir_type="bulk")
        bulk_site_potentials = -1 * np.array(bulk_outcar.site_potentials)
    else:
        bulk_site_potentials = _get_and_check_metadata(
            defect_entry, "bulk_supercell_site_potentials", "Bulk OUTCAR (for atomic site potentials)"
        )

    defect_calc_results_for_eFNV = CalcResults(
        structure=defect_entry.calculation_metadata.get(
            "unrelaxed_defect_structure", defect_entry.sc_entry.structure
        ),
        energy=np.inf,
        magnetization=np.inf,
        potentials=defect_site_potentials,
    )
    bulk_calc_results_for_eFNV = CalcResults(
        structure=defect_entry.bulk_entry.structure,
        energy=np.inf,
        magnetization=np.inf,
        potentials=bulk_site_potentials,
    )

    efnv_correction = make_efnv_correction(
        charge=defect_entry.charge_state,
        calc_results=defect_calc_results_for_eFNV,
        perfect_calc_results=bulk_calc_results_for_eFNV,
        dielectric_tensor=dielectric,
        defect_coords=defect_entry.sc_defect_frac_coords,
        **kwargs,
    )
    kumagai_correction_result = CorrectionResult(
        correction_energy=efnv_correction.correction_energy,
        metadata={"pydefect_ExtendedFnvCorrection": efnv_correction},
    )

    if not plot and filename is None:
        return kumagai_correction_result

    _install_custom_font()

    spp = SitePotentialMplPlotter.from_efnv_corr(
        title=f"{_format_defect_name(defect_entry.name, False)} - eFNV Site Potentials",
        efnv_correction=efnv_correction,
    )
    with plt.style.context("../doped/utils/doped.mplstyle"):
        spp.construct_plot()
        fig = spp.plt.gcf()
        ax = fig.gca()

        # reformat plot slightly
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            label.replace("point charge", "Point Charge (PC)").replace(
                "potential difference", r"$\Delta V$"
            )
            for label in labels
        ]
        # add entry for dashed red line

        handles += [Line2D([0], [0], **spp._mpl_defaults.hline)]
        labels += [
            rf"Avg. $\Delta V$ = {spp.ave_pot_diff:.3f} V",
        ]
        ax.legend(handles, labels, loc="best", borderaxespad=0, fontsize=8)
        ax.set_xlabel(f"Distance from defect ({spp._x_unit})", size=spp._mpl_defaults.label_font_size)

    if filename:
        spp.plt.savefig(filename, bbox_inches="tight", transparent=True, backend=_get_backend(filename))
    else:
        spp.plt.show()

    if verbose:
        print(
            f"Calculated Kumagai (eFNV) correction is {kumagai_correction_result.correction_energy:.3f} eV"
        )

    return kumagai_correction_result, fig
