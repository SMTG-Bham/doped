import itertools
import logging
from math import erfc, exp

import numpy as np

# These functions are taken from the AIDE package developed by
# Adam Jackson and Alex Ganose (https://github.com/SMTG-UCL/aide)


def get_image_charge_correction(
    lattice,
    dielectric_matrix,
    conv=0.3,
    factor=30,
    motif=[0.0, 0.0, 0.0],
    verbose=False,
):
    """Calculates the anisotropic image charge correction by Sam Murphy in eV.

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
        motif: The defect motif (doesn't matter for single point defects, but
            included in case we include the extended code for defect clusters).
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

    # Determine which of the lattice parameters is the largest and determine
    # reciprocal space supercell
    recip_axis = np.array([int(x) for x in factor * max(latt) / latt])
    recip_volume = abs(np.dot(np.cross(lattice[0], lattice[1]), lattice[2]))

    # Calculatate the reciprocal lattice vectors (need factor of 2 pi)
    recip_latt = np.linalg.inv(lattice).T * 2 * np.pi

    real_space = _get_real_space(
        conv, inv_diel, det_diel, latt, longest, r_c, axis, sup_latt
    )
    reciprocal = _get_recip(
        conv,
        inv_diel,
        det_diel,
        latt,
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

    # TODO: Use tabulate
    if verbose:
        logging.info(
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

        logging.info(
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
            logging.info(
                "|   {}    |     {:10f}   |    {:10f}   |".format(q, makov, lany)
            )
        logging.info("+--------+------------------+-----------------+")
    return correction


def _get_real_space(conv, inv_diel, det_diel, latt, longest, r_c, axis, sup_latt):
    # Calculate real space component
    real_space = 0.0
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
            contribution = 1 / np.sqrt(det_diel) * erfc(conv * N) / N
            return contribution
        else:
            return 0.0

    real_space = sum(
        _real_loop_function(mno) for mno in itertools.product(*axis_ranges)
    )
    return real_space


def _get_recip(
    conv,
    inv_diel,
    det_diel,
    latt,
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
            contribution = exp(-dot_prod / (4 * conv**2)) / dot_prod
            return contribution
        else:
            return 0.0

    reciprocal = sum(
        _recip_loop_function(mno) for mno in itertools.product(*axis_ranges)
    )
    scale_factor = 4 * np.pi / recip_volume
    return reciprocal * scale_factor
