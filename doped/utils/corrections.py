"""
Code to compute finite-size charge corrections for charged defects in periodic
systems. These functions are built from a combination of useful modules from
pymatgen, pycdt and AIDE (by Adam Jackson and Alex Ganose), alongside
substantial modification, in the efforts of making an efficient, user-friendly
package for managing and analysing defect calculations, with publication-
quality outputs.

The charge-correction methods implemented are:
1) Freysoldt correction for isotropic systems. Includes:
       a) PC energy
       b) potential alignment by planar averaging.
2) Extended Freysoldt or Kumagai correction for anistropic systems. Includes:
       a) anisotropic PC energy
       b) potential alignment by atomic site averaging outside Wigner Seitz radius

If you use the corrections implemented in this module, cite
   Freysoldt, Neugebauer, and Van de Walle,
    Phys. Status Solidi B. 248, 1067-1076 (2011) for isotropic correction
   Kumagai and Oba, Phys. Rev. B. 89, 195205 (2014) for anisotropic correction
"""

import copy
import itertools
import warnings
from math import erfc, exp
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from monty.json import MontyDecoder
from shakenbreak.plotting import _install_custom_font

from doped.analysis import DefectParser, _convert_dielectric_to_tensor
from doped.plotting import _get_backend
from doped.utils.legacy_pmg.corrections import FreysoldtCorrection, KumagaiCorrection

warnings.simplefilter("default")
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


def get_correction_freysoldt(
    defect_entry,
    dielectric,
    plot: bool = False,
    filename: Optional[str] = None,
    partflag="All",
    axis=None,
    **kwargs,
):
    """
    Function to compute the isotropic Freysoldt correction for each defect.
    If this correction is used, please reference Freysoldt's original paper.
    doi: 10.1103/PhysRevLett.102.016402
    Args:
        defect_entry: DefectEntry object with the following
            keys stored in defect.calculation_metadata:
                required:
                    axis_grid (3 x NGX where NGX is the length of the NGX grid
                    in the x,y and z axis directions. Same length as planar
                    average lists):
                        A list of 3 numpy arrays which contain the cartesian axis
                        values (in angstroms) that correspond to each planar avg
                        potential supplied.

                    bulk_planar_averages (3 x NGX where NGX is the length of
                    the NGX grid in the x,y and z axis directions.):
                        A list of 3 numpy arrays which contain the planar averaged
                        electrostatic potential for the bulk supercell.

                    defect_planar_averages (3 x NGX where NGX is the length of
                    the NGX grid in the x,y and z axis directions.):
                        A list of 3 numpy arrays which contain the planar averaged
                        electrostatic potential for the defective supercell.

                    bulk_sc_structure (Structure) bulk structure corresponding to
                        defect supercell structure (uses Lattice for charge correction)

                    defect_frac_sc_coords (3 x 1 array) Fracitional coordinates of
                        defect location in supercell structure
                optional:
                    'encut' : energy cutoff desired for Freysoldt correction
                    'madetol' : madelung tolerance for Freysoldt correction
                    'q_model' : Charge Model for Freysoldt correction
                    'q_model' : Charge Model for Freysoldt correction
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
        plot (bool): decides whether to plot electrostatic potential plots or not.
        filename (str): if None, plots are not saved, if a string,
            then the plot will be saved as '{filename}_{axis}.pdf'
        partflag: four options for correction output:
               'pc' for just point charge correction, or
               'potalign' for just potalign correction, or
               'All' for both (added together), or
               'AllSplit' for individual parts split up (form is [PC, potterm, full])
        axis (int or None):
            if integer, then freysoldt correction is performed on the single axis specified.
            If it is None, then averaging of the corrections for the three axes is used for the
            correction.

    Returns Correction
    """
    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    dielectric = _convert_dielectric_to_tensor(dielectric)

    if partflag not in ["All", "AllSplit", "pc", "potalign"]:
        raise ValueError(
            f'{partflag} is incorrect potalign type. Must be "All", "AllSplit", "pc", or "potalign".'
        )

    q_model = defect_entry.calculation_metadata.get("q_model", None)
    encut = defect_entry.calculation_metadata.get("encut", 520)
    madetol = defect_entry.calculation_metadata.get("madetol", 0.0001)

    if not defect_entry.charge_state:
        print("Charge is zero so charge correction is zero.")
        return 0.0

    template_defect = copy.deepcopy(defect_entry)
    corr_class = FreysoldtCorrection(
        dielectric, q_model=q_model, energy_cutoff=encut, madetol=madetol, axis=axis
    )
    f_corr_summ = corr_class.get_correction(template_defect)

    if plot:
        axis_labels = ["x", "y", "z"]
        if axis is None:
            ax_list = [[k, f"${axis_labels[k]}$-axis"] for k in corr_class.metadata["pot_plot_data"]]
        else:
            ax_list = [[axis, f"${axis_labels[axis]}$-axis"]]

        for ax_key, ax_title in ax_list:
            _install_custom_font()
            p = corr_class.plot(ax_key, title=ax_title, saved=False)
            if filename:
                p.savefig(
                    f"{filename}_{ax_title.replace('$','')}.pdf",
                    bbox_inches="tight",
                    transparent=True,
                    backend=_get_backend("pdf"),
                )
            if kwargs.get("return_fig", False):  # for testing
                return p
            plt.show()

    if partflag in ["AllSplit", "All"]:
        freyval = np.sum(list(f_corr_summ.values()))
    elif partflag == "pc":
        freyval = f_corr_summ["freysoldt_electrostatic"]
    elif partflag == "potalign":
        freyval = f_corr_summ["freysoldt_potential_alignment"]

    print(f"Final Freysoldt correction is {freyval:.3f} eV")

    if partflag == "AllSplit":
        freyval = [
            f_corr_summ["freysoldt_electrostatic"],
            f_corr_summ["freysoldt_potential_alignment"],
            freyval,
        ]

    return freyval


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


def freysoldt_correction_from_paths(
    defect_file_path,
    bulk_file_path,
    dielectric,
    defect_charge,
    plot=False,
    filename: Optional[str] = None,
    **kwargs,
):
    """
    A function for performing the Freysoldt correction with a set of file paths.
    If this correction is used, please reference Freysoldt's original paper.
    doi: 10.1103/PhysRevLett.102.016402.

    :param defect_file_path (str): file path to defect folder of interest
    :param bulk_file_path (str): file path to bulk folder of interest
    :param dielectric (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
    :param charge (int): charge of defect structure of interest
    :param plot (bool): decides whether to plot electrostatic potential plots or not.
    :param filename (str): if None, plots are not saved, if a string,
            then the plot will be saved as '{filename}_{axis}.pdf'
    :return:
        Dictionary of Freysoldt Correction for defect
    """
    dp = DefectParser.from_paths(defect_file_path, bulk_file_path, dielectric, defect_charge)
    _ = dp.freysoldt_loader()
    if plot:
        print(dp.defect_entry.name)

    return get_correction_freysoldt(dp.defect_entry, dielectric, plot=plot, filename=filename, **kwargs)


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
# Adam Jackson and Alex Ganose (https://github.com/SMTG-UCL/aide)


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
