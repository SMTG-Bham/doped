"""
Code to compute finite-size charge corrections for charged defects in periodic
systems. These functions are built from a combination of useful modules from
pymatgen, pycdt and AIDE (by Adam Jackson and Alex Ganose), alongside
substantial modification, in the efforts of making an efficient, user-friendly
package for managing and analysing defect calculations, with publication-
quality outputs.

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

import copy
import itertools
import os
import warnings
from math import erfc, exp
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from monty.json import MontyDecoder
from pymatgen.analysis.defects.corrections import freysoldt
from pymatgen.io.vasp.outputs import Locpot
from shakenbreak.plotting import _install_custom_font

from doped.analysis import DefectParser, _convert_dielectric_to_tensor
from doped.plotting import _get_backend
from doped.utils.legacy_pmg.corrections import KumagaiCorrection

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


def get_freysoldt_correction(
    defect_entry,
    dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
    defect_locpot: Optional[Union[str, Locpot, dict]] = None,
    bulk_locpot: Optional[Union[str, Locpot, dict]] = None,
    plot: bool = False,
    filename: Optional[str] = None,
    axis=None,
    **kwargs,
):
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
            FNV finite-size charge correction. The correction will be added
            to the `corrections` dictionary attribute (and thus used in any
            following formation energy calculations).
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

    def _check_if_None_and_raise_error_if_so(var, var_name, display_name):
        if var is None:
            raise ValueError(
                f"{var_name} (`{display_name}`) must be provided as an argument or be present in the "
                "`defect_entry` `calculation_metadata`, neither of which is the case here!"
            )

    def get_and_check_metadata(entry, key, display_name):
        value = entry.calculation_metadata.get(key, None)
        _check_if_None_and_raise_error_if_so(value, display_name, key)
        return value

    dielectric = dielectric or get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")
    dielectric = _convert_dielectric_to_tensor(dielectric)

    defect_locpot = defect_locpot or get_and_check_metadata(defect_entry, "defect_locpot", "Defect LOCPOT")
    bulk_locpot = bulk_locpot or get_and_check_metadata(defect_entry, "bulk_locpot", "Bulk LOCPOT")

    def _check_if_str_and_get_Locpot(locpot):
        if isinstance(locpot, str):
            return Locpot.from_file(locpot)

        if not isinstance(locpot, Locpot):
            raise TypeError(
                f"`locpot` input must be either a path to a LOCPOT file or a pymatgen Locpot object, "
                f"but got {type(locpot)} instead."
            )

        return locpot

    defect_locpot = _check_if_str_and_get_Locpot(defect_locpot)
    bulk_locpot = _check_if_str_and_get_Locpot(bulk_locpot)

    fnv_correction = freysoldt.get_freysoldt_correction(
        q=defect_entry.charge_state,
        dielectric=dielectric,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        defect_frac_coords=defect_entry.sc_defect_frac_coords,
        **kwargs,
    )

    if all(x is None for x in [plot, filename]):
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

    # TODO: Remove or change this if this function is used in our parsing loops
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

        line4 = ax.axhline(C, color="k", linestyle="--", label=f"Alignment constant C = {C:.3f}")
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


def get_correction_kumagai(
    defect_entry, dielectric, plot: bool = False, filename: Optional[str] = None, partflag="All", **kwargs
):
    """
    Function to compute the Kumagai correction for each defect (modified freysoldt for
    anisotropic dielectric).
    NOTE that bulk_init class must be pre-instantiated to use this function
    Args:
        defect_entry: DefectEntry object with the following
            keys stored in defect.calculation_metadata:
                required:
                    bulk_atomic_site_averages (list):  list of bulk structure"s atomic site
                    averaged ESPs * charge, in same order as indices of bulk structure note this
                    is list given by VASP's OUTCAR (so it is multiplied by a test charge of -1).

                    defect_atomic_site_averages (list):  list of defect structure"s atomic site
                    averaged ESPs * charge, in same order as indices of defect structure note
                    this is list given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    site_matching_indices (list):  list of corresponding site index values for
                    bulk and defect site structures EXCLUDING the defect site itself (ex. [[bulk
                    structure site index, defect structure"s corresponding site index], ... ]

                    initial_defect_structure (Structure): Pymatgen Structure object representing
                    un-relaxed defect structure

                    defect_frac_sc_coords (array): Defect Position in fractional coordinates of
                    the supercell given in bulk_structure
                optional:
                    gamma (float): Ewald parameter, Default is to determine it based on
                        convergence of brute summation tolerance
                    sampling_radius (float): radius (in Angstrom) which sites must be outside of
                        to be included in the correction. Publication by Kumagai advises to use
                        Wigner-Seitz radius of defect supercell, so this is default value.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
        plot (bool): decides whether to plot electrostatic potential plots or not.
        filename (str): if None, plots are not saved, if a string, then the plot will be saved as
            '{filename}.pdf'
        partflag: four options for correction output:
               'pc' for just point charge correction, or
               'potalign' for just potalign correction, or
               'All' for both (added together), or
               'AllSplit' for individual parts split up (form is [PC, potterm, full])
    """
    # TODO: If updating to use pydefect, need to add this to dependencies!

    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    dielectric = _convert_dielectric_to_tensor(dielectric)

    if partflag not in ["All", "AllSplit", "pc", "potalign"]:
        raise ValueError(
            f'{partflag} is incorrect potalign type. Must be "All", "AllSplit", "pc", or "potalign".'
        )
    sampling_radius = defect_entry.calculation_metadata.get("sampling_radius", None)
    gamma = defect_entry.calculation_metadata.get("gamma", None)

    if not defect_entry.charge_state:
        print("Charge is zero so charge correction is zero.")
        return 0.0

    template_defect = copy.deepcopy(defect_entry)
    corr_class = KumagaiCorrection(dielectric, sampling_radius=sampling_radius, gamma=gamma)
    k_corr_summ = corr_class.get_correction(template_defect)

    if plot:
        _install_custom_font()
        p = corr_class.plot(title="Kumagai", saved=False)
        if filename:
            p.savefig(
                f"{filename}.pdf", bbox_inches="tight", transparent=True, backend=_get_backend("pdf")
            )
        if kwargs.get("return_fig", False):  # for testing
            return p
        plt.show()

    if partflag in ["AllSplit", "All"]:
        kumagai_val = np.sum(list(k_corr_summ.values()))
    elif partflag == "pc":
        kumagai_val = k_corr_summ["kumagai_electrostatic"]
    elif partflag == "potalign":
        kumagai_val = k_corr_summ["kumagai_potential_alignment"]

    print(f"\nFinal Kumagai correction is {kumagai_val:.3f} eV")

    if partflag == "AllSplit":
        kumagai_val = [
            k_corr_summ["kumagai_electrostatic"],
            k_corr_summ["kumagai_potential_alignment"],
            kumagai_val,
        ]
    return kumagai_val


def kumagai_correction_from_paths(
    defect_file_path,
    bulk_file_path,
    dielectric,
    defect_charge,
    plot=False,
    filename: Optional[str] = None,
    **kwargs,
):
    """
    A function for performing the Kumagai correction with a set of file paths.
    If this correction is used, please reference Kumagai and Oba's original
    paper (doi: 10.1103/PhysRevB.89.195205) as well as Freysoldt's original
    paper (doi: 10.1103/PhysRevLett.102.016402.

    :param defect_file_path (str): file path to defect folder of interest
    :param bulk_file_path (str)
    : file path to bulk folder of interest : param     dielectric (float or int
        or 3x1 matrix or 3x3 matrix): ionic +     static contributions to
        dielectric constant :param charge (int): charge of     defect structure
        of interest :param plot (bool): decides whether to plot
        electrostatic potential plots or not. :param filename (str): if None,
        plots     are not saved, if a string, then the plot will be saved as
        '{filename}.pdf'
    :return: Dictionary of Kumagai Correction for defect
    """
    dp = DefectParser.from_paths(defect_file_path, bulk_file_path, dielectric, defect_charge)
    _ = dp.kumagai_loader()
    if plot:
        print(dp.defect_entry.name)

    return get_correction_kumagai(dp.defect_entry, dielectric, plot=plot, filename=filename, **kwargs)


# The following functions are taken from the deprecated AIDE package developed by the dynamic duo
# Adam Jackson and Alex Ganose (https://github.com/SMTG-Bham/aide)


def get_murphy_image_charge_correction(
    lattice,
    dielectric_matrix,
    conv=0.3,
    factor=30,
    verbose=False,
):
    """
    Calculates the anisotropic image charge correction by Sam Murphy in eV.

    This a rewrite of the code 'madelung.pl' written by Sam Murphy (see [1]).
    The default convergence parameter of conv = 0.3 seems to work perfectly
    well. However, it may be worth testing convergence of defect energies with
    respect to the factor (i.e. cut-off radius).

    References:
        [1] S. T. Murphy and N. D. H. Hine, Phys. Rev. B 87, 094111 (2013).

    Args:
        lattice (list): The defect cell lattice as a 3x3 matrix.
        dielectric_matrix (list): The dielectric tensor as 3x3 matrix.
        conv (float): A value between 0.1 and 0.9 which adjusts how much real
                      space vs reciprocal space contribution there is.
        factor: The cut-off radius, defined as a multiple of the longest cell
            parameter.
        verbose (bool): If True details of the correction will be printed.

    Returns:
        The image charge correction as {charge: correction}
    """
    inv_diel = np.linalg.inv(dielectric_matrix)
    det_diel = np.linalg.det(dielectric_matrix)
    latt = np.sqrt(np.sum(lattice**2, axis=1))

    # calc real space cutoff
    longest = max(latt)
    r_c = factor * longest

    # Estimate the number of boxes required in each direction to ensure
    # r_c is contained (the tens are added to ensure the number of cells
    # contains r_c). This defines the size of the supercell in which
    # the real space section is performed, however only atoms within rc
    # will be conunted.
    axis = np.array([int(r_c / a + 10) for a in latt])

    # Calculate supercell parallelpiped and dimensions
    sup_latt = np.dot(np.diag(axis), lattice)

    # Determine which of the lattice calculation_metadata is the largest and determine
    # reciprocal space supercell
    recip_axis = np.array([int(x) for x in factor * max(latt) / latt])
    recip_volume = abs(np.dot(np.cross(lattice[0], lattice[1]), lattice[2]))

    # Calculatate the reciprocal lattice vectors (need factor of 2 pi)
    recip_latt = np.linalg.inv(lattice).T * 2 * np.pi

    real_space = _get_real_space(conv, inv_diel, det_diel, r_c, axis, sup_latt)
    reciprocal = _get_recip(
        conv,
        recip_axis,
        recip_volume,
        recip_latt,
        dielectric_matrix,
    )

    # calculate the other terms and the final Madelung potential
    third_term = -2 * conv / np.sqrt(np.pi * det_diel)
    fourth_term = -3.141592654 / (recip_volume * conv**2)
    madelung = -(real_space + reciprocal + third_term + fourth_term)

    # convert to atomic units
    conversion = 14.39942
    real_ev = real_space * conversion / 2
    recip_ev = reciprocal * conversion / 2
    third_ev = third_term * conversion / 2
    fourth_ev = fourth_term * conversion / 2
    madelung_ev = madelung * conversion / 2

    correction = {}
    for q in range(1, 8):
        makov = 0.5 * madelung * q**2 * conversion
        lany = 0.65 * makov
        correction[q] = makov

    if verbose:
        print(
            """
    Results                      v_M^scr    dE(q=1) /eV
    -----------------------------------------------------
    Real space contribution    =  {:.6f}     {:.6f}
    Reciprocal space component =  {:.6f}     {:.6f}
    Third term                 = {:.6f}    {:.6f}
    Neutralising background    = {:.6f}    {:.6f}
    -----------------------------------------------------
    Final Madelung potential   = {:.6f}     {:.6f}
    -----------------------------------------------------""".format(
                real_space,
                real_ev,
                reciprocal,
                recip_ev,
                third_term,
                third_ev,
                fourth_term,
                fourth_ev,
                madelung,
                madelung_ev,
            )
        )

        print(
            """
    Here are your final corrections:
    +--------+------------------+-----------------+
    | Charge | Point charge /eV | Lany-Zunger /eV |
    +--------+------------------+-----------------+"""
        )
        for q in range(1, 8):
            makov = 0.5 * madelung * q**2 * conversion
            lany = 0.65 * makov
            correction[q] = makov
            print(f"|   {q}    |     {makov:10f}   |    {lany:10f}   |")
        print("+--------+------------------+-----------------+")

    return correction


def _get_real_space(conv, inv_diel, det_diel, r_c, axis, sup_latt):
    # Calculate real space component
    axis_ranges = [range(-a, a) for a in axis]

    # Pre-compute square of cutoff distance for cheaper comparison than
    # separation < r_c
    r_c_sq = r_c**2

    def _real_loop_function(mno):
        # Calculate the defect's fractional position in extended supercell
        d_super = np.array(mno, dtype=float) / axis
        d_super_cart = np.dot(d_super, sup_latt)

        # Test if the new atom coordinates fall within r_c, then solve
        separation_sq = np.sum(np.square(d_super_cart))
        # Take all cases within r_c except m,n,o != 0,0,0
        if separation_sq < r_c_sq and any(mno):
            mod = np.dot(d_super_cart, inv_diel)
            dot_prod = np.dot(mod, d_super_cart)
            N = np.sqrt(dot_prod)
            return 1 / np.sqrt(det_diel) * erfc(conv * N) / N

        return 0.0

    return sum(_real_loop_function(mno) for mno in itertools.product(*axis_ranges))


def _get_recip(
    conv,
    recip_axis,
    recip_volume,
    recip_latt,
    dielectric_matrix,
):
    # convert factional motif to reciprocal space and
    # calculate reciprocal space supercell parallelpiped
    recip_sup_latt = np.dot(np.diag(recip_axis), recip_latt)

    # Calculate reciprocal space component
    axis_ranges = [range(-a, a) for a in recip_axis]

    def _recip_loop_function(mno):
        # Calculate the defect's fractional position in extended supercell
        d_super = np.array(mno, dtype=float) / recip_axis
        d_super_cart = np.dot(d_super, recip_sup_latt)

        if any(mno):
            mod = np.dot(d_super_cart, dielectric_matrix)
            dot_prod = np.dot(mod, d_super_cart)
            return exp(-dot_prod / (4 * conv**2)) / dot_prod

        return 0.0

    reciprocal = sum(_recip_loop_function(mno) for mno in itertools.product(*axis_ranges))
    scale_factor = 4 * np.pi / recip_volume
    return reciprocal * scale_factor
