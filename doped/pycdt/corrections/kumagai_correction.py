"""
This module computes finite size supercell charge corrections for
defects in anistropic systems using extended Freysoldt (or Kumagai) method 
developed by Kumagai and Oba.
Kumagai method includes
   a) anisotropic PC energy
   b) potential alignment by atomic site averaging at Wigner Seitz cell
      edge
If you use the corrections implemented in this module, cite
 a) Kumagai and Oba, Phys. Rev. B. 89, 195205 (2014) and
 b) Freysoldt, Neugebauer, and Van de Walle,
    Phys. Status Solidi B. 248, 1067-1076 (2011)  and
in addition to the pycdt paper
"""

import logging
import math
import warnings

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.outputs import Locpot, Outcar

from doped.pycdt.corrections.utils import *
from doped.pycdt.utils.units import hart_to_ev

norm = np.linalg.norm


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def kumagai_init(structure, dieltens):
    angset = structure.lattice.get_cartesian_coords(1)

    dieltens = np.array(dieltens)
    if not len(dieltens.shape):
        dieltens = dieltens * np.identity(3)
    elif len(dieltens.shape) == 1:
        dieltens = np.diagflat(dieltens)

    logging.getLogger(__name__).debug(
        "Lattice constants (in Angs): " + str(cleanlat(angset))
    )
    [a1, a2, a3] = ang_to_bohr * angset  # convert to bohr
    bohrset = [a1, a2, a3]
    vol = np.dot(a1, np.cross(a2, a3))

    logging.getLogger(__name__).debug(
        "Lattice constants (in Bohr): " + str(cleanlat([a1, a2, a3]))
    )
    determ = np.linalg.det(dieltens)
    invdiel = np.linalg.inv(dieltens)
    logging.getLogger(__name__).debug("inv dielectric tensor: " + str(invdiel))

    return angset, bohrset, vol, determ, invdiel


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def real_sum(a1, a2, a3, r, q, dieltens, gamma, tolerance):
    invdiel = np.linalg.inv(dieltens)
    determ = np.linalg.det(dieltens)
    realpre = q / np.sqrt(determ)
    tolerance /= hart_to_ev

    # Real space sum by converging with respect to real space vectors
    # create list of real space vectors that satisfy |i*a1+j*a2+k*a3|<=N
    Nmaxlength = 40  # tolerance for stopping real space sum convergence
    N = 2
    r_sums = []
    while N < Nmaxlength:
        r_sum = 0.0
        if norm(r):
            for i in range(-N, N + 1):
                for j in range(-N, N + 1):
                    for k in range(-N, N + 1):
                        r_vec = i * a1 + j * a2 + k * a3 - r
                        loc_res = np.dot(r_vec, np.dot(invdiel, r_vec))
                        nmr = math.erfc(gamma * np.sqrt(loc_res))
                        dmr = np.sqrt(determ * loc_res)
                        r_sum += nmr / dmr
        else:
            for i in range(-N, N + 1):
                for j in range(-N, N + 1):
                    for k in range(-N, N + 1):
                        if i == j == k == 0:
                            continue
                        else:
                            r_vec = i * a1 + j * a2 + k * a3
                            loc_res = np.dot(r_vec, np.dot(invdiel, r_vec))
                            nmr = math.erfc(gamma * np.sqrt(loc_res))
                            dmr = np.sqrt(determ * loc_res)
                            r_sum += nmr / dmr
        r_sums.append([N, realpre * r_sum])

        if N == Nmaxlength - 1:
            logging.getLogger(__name__).warning(
                "Direct part could not converge with real space translation "
                "tolerance of {} for gamma {}".format(Nmaxlength - 1, gamma)
            )
            return
        elif len(r_sums) > 3:
            if abs(abs(r_sums[-1][1]) - abs(r_sums[-2][1])) < tolerance:
                r_sum = r_sums[-1][1]
                logging.debug("gamma is {}".format(gamma))
                logging.getLogger(__name__).debug(
                    "convergence for real summatin term occurs at step {} "
                    "where real sum is {}".format(N, r_sum * hart_to_ev)
                )
                break

        N += 1
    return r_sum


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def get_g_sum_at_r(g_sum, structure, dim, r):
    """
    Args:
        g_sum: Reciprocal summation calculated from reciprocal_sum method
        structure: Bulk structure pymatgen object
        dim : ngxf dimension
        r: Position relative to defect (in cartesian coords)
    Returns:
        reciprocal summ value at g_sum[i_rx,j_ry,k_rz]
    """

    fraccoord = structure.lattice.get_fractional_coords(r)
    i, j, k = getgridind(structure, dim, fraccoord)

    return g_sum[i, j, k]


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def anisotropic_madelung_potential(
    structure, dim, g_sum, r, dieltens, q, gamma, tolerance
):
    """
    Compute the anisotropic Madelung potential at r not equal to 0.
    For r=(0,0,0) use anisotropic_pc_energy function
    Args:
        structure: Bulk pymatgen structure type
        dim : ngxf dimension
        g_sum: Precomputed reciprocal sum for all r_vectors
        r: r vector (in cartesian coordinates) relative to defect position.
           Non zero r is expected
        dieltens: dielectric tensor
        q: Point charge (in units of e+)
        tolerance: Tolerance parameter for numerical convergence
        gamma (float): Convergence parameter
        silence (bool): Verbosity flag. If False, messages are printed.
    """
    angset, [a1, a2, a3], vol, determ, invdiel = kumagai_init(structure, dieltens)

    recippartreal = q * get_g_sum_at_r(g_sum, structure, dim, r)
    directpart = real_sum(a1, a2, a3, r, q, dieltens, gamma, tolerance)

    # now add up total madelung potential part with two extra parts:
    # self interaction term
    selfint = q * np.pi / (vol * (gamma**2))
    logging.getLogger(__name__).debug(
        "self interaction piece is {}".format(selfint * hart_to_ev)
    )

    pot = hart_to_ev * (directpart + recippartreal - selfint)
    return pot


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def anisotropic_pc_energy(structure, g_sum, dieltens, q, gamma, tolerance):
    """
    Compute the anistropic periodic point charge interaction energy.
    Args:
        structure: Bulk pymatgen structure type
        g_sum : comes from KumagaiBulkInit class
        dieltens: dielectric tensor
        q: Point charge (in units of e+)
        gamma : convergence parameter optimized in KumagaiBulkInit class
        silence (bool): Verbosity flag. If False, messages are printed.
    """
    angset, [a1, a2, a3], vol, determ, invdiel = kumagai_init(structure, dieltens)

    g_part = q * g_sum[0, 0, 0]
    r_part = real_sum(a1, a2, a3, [0, 0, 0], q, dieltens, gamma, tolerance)
    selfint = q * np.pi / (vol * (gamma**2))  # self interaction term
    # surface term (only for r not at origin)
    surfterm = 2 * gamma * q / np.sqrt(np.pi * determ)

    logger = logging.getLogger(__name__)
    logger.debug("reciprocal part: {}".format(g_part * hart_to_ev))
    logger.debug("real part: {}".format(r_part * hart_to_ev))
    logger.debug("self interaction part: {}".format(selfint * hart_to_ev))
    logger.debug("surface term: {}".format(surfterm * hart_to_ev))

    pc_energy = -(q * 0.5 * hart_to_ev) * (r_part + g_part - selfint - surfterm)
    logging.debug("Final PC Energy term: {} eV".format(pc_energy))

    return pc_energy


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def getgridind(structure, dim, r, gridavg=0.0):
    """
    Computes the index of a point, r, in the locpot grid
    Args:
        structure:
            Pymatgen structure object
        dim:
            dimension of FFT grid (NGXF dimension list in VASP)
        r:
            Relative co-ordinates with respect to abc lattice vectors
        gridavg:
            If you want to do atomic site averaging, set gridavg to
            the radius of the atom at r
    Returns:
        [i,j,k]: Indices as list
    TODO: Once final, remove the getgridind inside disttrans function
    """
    abc = structure.lattice.abc
    grdind = []

    if gridavg:
        radvals = []  # radius in terms of indices
        dxvals = []

    for i in range(3):
        if r[i] < 0:
            while r[i] < 0:
                r[i] += 1
        elif r[i] >= 1:
            while r[i] >= 1:
                r[i] -= 1
        r[i] *= abc[i]
        num_pts = dim[i]
        x = [now_num / float(num_pts) * abc[i] for now_num in range(num_pts)]
        dx = x[1] - x[0]
        x_rprojection_delta_abs = np.absolute(x - r[i])
        ind = np.argmin(x_rprojection_delta_abs)
        if x_rprojection_delta_abs[ind] > dx * 1.1:  # to avoid numerical errors
            logger = logging.getLogger(__name__)
            logger.error("Input position not within the locpot grid")
            logger.error("%d, %d, %f", i, ind, r)
            logger.error("%f", x_rprojection_delta_abs)
            raise ValueError("Input position is not within the locpot grid")
        grdind.append(ind)
        if gridavg:
            radvals.append(int(np.ceil(gridavg / dx)))
            dxvals.append(dx)

    if gridavg:
        grdindfull = []
        for i in range(-radvals[0], radvals[0] + 1):
            for j in range(-radvals[1], radvals[1] + 1):
                for k in range(-radvals[2], radvals[2] + 1):
                    dtoc = [i * dxvals[0], j * dxvals[1], k * dxvals[2]]
                    if norm(dtoc) < gridavg:
                        ival = (i + grdind[0]) % dim[0]
                        jval = (j + grdind[1]) % dim[1]
                        kval = (k + grdind[2]) % dim[2]
                        grdindfull.append((ival, jval, kval))
        grdind = grdindfull

    return grdind


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def disttrans(struct, defstruct, defpos=None):
    """
    To calculate distance from defect to each atom and finding NGX grid
    pts at each atom.
    Args:
        struct: Bulk structure object
        defstruct: Defect structure object
        defpos: (if known) defect position as a pymatgen Site object within bulk supercell
    """

    # Find defect location in bulk and defect cells
    blksite, defsite = find_defect_pos(struct, defstruct, defpos=defpos)
    logger = logging.getLogger(__name__)
    if blksite is None and defsite is None:
        logger.error("Not able to determine defect site")
        return
    if blksite is None:
        logger.debug("Found defect to be Interstitial type at %s", repr(defsite))
    elif defsite is None:
        logger.debug("Found defect to be Vacancy type at %s", repr(blksite))
    else:
        logger.debug(
            "Found defect to be antisite/subsitution type at %s "
            " in bulk, and %s in defect cell",
            repr(blksite),
            repr(defsite),
        )

    if blksite is None:
        blksite = defsite
    elif defsite is None:
        defsite = blksite

    def_ccoord = blksite[:]
    defcell_def_ccoord = defsite[:]

    if len(struct.sites) >= len(defstruct.sites):
        sitelist = struct.sites[:]
    else:  # for interstitial list
        sitelist = defstruct.sites[:]

    # better image getter since pymatgen wasnt working well for this
    def returnclosestr(vec):
        from operator import itemgetter

        listvals = []
        abclats = defstruct.lattice.matrix
        trylist = [-1, 0, 1]
        for i in trylist:
            for j in trylist:
                for k in trylist:
                    transvec = i * abclats[0] + j * abclats[1] + k * abclats[2]
                    rnew = vec - (defcell_def_ccoord + transvec)
                    listvals.append([norm(rnew), rnew, transvec])
        listvals.sort(key=itemgetter(0))
        return listvals[0]  # will return [dist,r to defect, and transvec for defect]

    grid_sites = {}  # dictionary with indices keys in order of structure list
    for i in sitelist:
        if np.array_equal(i.coords, def_ccoord):
            logging.debug("Site {} is defect! Skipping ".format(i))
            continue

        blksite, defsite = closestsites(struct, defstruct, i.coords)
        blkindex = blksite[-1]
        defindex = defsite[-1]

        dcart_coord = defsite[0].coords
        closeimage = returnclosestr(dcart_coord)
        cart_reldef = closeimage[1]
        defdist = closeimage[0]

        if abs(norm(cart_reldef) - defdist) > 0.1:
            logger.warning("Image locater issue encountered for site = %d", blkindex)
            logger.warning("In defect supercell")
            logger.warning("Distance should be %f", defdist)
            logger.warning("But, calculated distance is %f", norm(cart_reldef))

        if blkindex in grid_sites:
            logger.warning("Index %d already exists in potinddict!", blkindex)
            logger.warning("Overwriting information.")

        grid_sites[blkindex] = {
            "dist": defdist,
            "cart": dcart_coord,
            "cart_reldef": cart_reldef,
            "siteobj": [i.coords, i.frac_coords, i.species_string],
            "bulk_site_index": blkindex,
            "def_site_index": defindex,
        }

    return grid_sites


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def wigner_seitz_radius(structure):
    """
    Calculate the Wigner Seitz radius for the given structure.
    Args:
        structure: pymatgen Structure object
    """
    wz = structure.lattice.get_wigner_seitz_cell()

    dist = []
    for facet in wz:
        midpt = np.mean(np.array(facet), axis=0)
        dist.append(norm(midpt))
    wsrad = min(dist)

    return wsrad


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


def read_ES_avg_fromlocpot(locpot):
    """
    Reads Electrostatic potential at each atomic
    site from Locpot Pymatgen object
    """
    structure = locpot.structure
    radii = {specie: 1.0 for specie in set(structure.species)}
    # TODO: The above radii could be smarter (related to ENAUG?)
    # but turns out you get a similar result to Outcar differences
    # when taking locpot avgd differences

    ES_data = {"sampling_radii": radii, "ngxf_dims": locpot.dim}
    pot = []
    for site in structure.sites:
        indexlist = getgridind(
            structure, locpot.dim, site.frac_coords, gridavg=radii[site.specie]
        )
        samplevals = []
        for u, v, w in indexlist:
            samplevals.append(locpot.data["total"][u][v][w])
        pot.append(np.mean(samplevals))

    ES_data.update({"potential": pot})

    return ES_data


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


class KumagaiBulkInit(object):
    """
    Compute the anisotropic madelung potential array from the bulk
    locpot. This helps in evaluating the bulk supercell related part
    once to speed up the calculations.
    """

    def __init__(
        self, structure, dim, epsilon, encut=520, tolerance=0.0001, optgamma=False
    ):
        """
        Args
            structure:
                Pymatgen structure object of bulk cell
            dim:
                Fine FFT grid dimensions as a list
                For vasp this is NGXF grid dimensions
            epsilon:
                Dielectric tensor
            encut (float):
                Energy cutoff for optimal gamma
            tolerance (float):
                Accuracy parameter
            optgamma:
                if you know optimized gamma, give its value.
                Otherwise it will be computed.
        """
        self.structure = structure
        self.dim = dim
        self.epsilon = epsilon
        self.encut = encut
        self.tolerance = tolerance
        # self.silence = silence
        if not optgamma:
            self.gamma = self.find_optimal_gamma()
        else:
            self.gamma = optgamma
        self.g_sum = self.reciprocal_sum()
        logging.getLogger(__name__).info("optimized gamma: %f", self.gamma)

    def find_optimal_gamma(self):
        """
        Find optimal gamma by evaluating the brute force reciprocal
        summation and seeing when the values are on the order of 1,
        This calculation is the anisotropic Madelung potential at r = (0,0,0).
        Note this only requires the STRUCTURE not the LOCPOT object.
        """
        angset, [a1, a2, a3], vol, determ, invdiel = kumagai_init(
            self.structure, self.epsilon
        )
        optgam = None

        # do brute force recip summation
        def get_recippart(encut, gamma):
            recippart = 0.0
            for rec in genrecip(a1, a2, a3, encut):
                Gdotdiel = np.dot(rec, np.dot(self.epsilon, rec))
                summand = math.exp(-Gdotdiel / (4 * (gamma**2))) / Gdotdiel
                recippart += summand
            recippart *= 4 * np.pi / vol
            return recippart, 0.0

        def do_summation(gamma):
            # Do recip sum until it is bigger than 1eV
            # First do Recip space sum convergence with respect to encut for
            # this gamma
            encut = 20  # start with small encut for expediency
            recippartreal1, recippartimag1 = get_recippart(encut, gamma)
            encut += 10
            recippartreal, recippartimag = get_recippart(encut, gamma)
            converge = [recippartreal1, recippartreal]

            logger = logging.getLogger(__name__)
            while (
                abs(abs(converge[0]) - abs(converge[1])) * hart_to_ev > self.tolerance
            ):
                encut += 10
                recippartreal, recippartimag = get_recippart(encut, gamma)
                converge.reverse()
                converge[1] = recippartreal
                if encut > self.encut:
                    msg = "Optimal gamma not found at {} eV cutoff".format(self.encut)
                    logger.error(msg)
                    raise ValueError(msg)

            if abs(recippartimag) * hart_to_ev > self.tolerance:
                logger.error("Imaginary part of reciprocal sum not converged.")
                logger.error(
                    "Imaginary sum value is {} (eV)".format(recippartimag * hart_to_ev)
                )
                return None, None
            logger.debug(
                "Reciprocal sum converged to %f eV", recippartreal * hart_to_ev
            )
            logger.debug("Convergin encut = %d eV", encut)

            if abs(converge[1]) * hart_to_ev < 1 and not optgam:
                logger.warning("Reciprocal summation value is less than 1 eV.")
                logger.warning("Might lead to errors")
                logger.warning("Change gamma.")
                return None, "Try Again"

            return recippartreal, gamma

        logger = logging.getLogger(__name__)
        # start with gamma s.t. gamma*L=5 (this is optimal)
        # optimizing gamma for the reciprocal sum to improve convergence
        gamma = 5.0 / (vol ** (1 / 3.0))
        optimal_gamma_found = False

        while not optimal_gamma_found:
            recippartreal, optgamma = do_summation(gamma)
            if optgamma == gamma:
                logger.debug("optimized gamma found to be %f", optgamma)
                optimal_gamma_found = True
            elif "Try Again" in optgamma:
                gamma *= 1.5
            else:
                logger.error("Had problem in gamma optimization process.")
                return None

            if gamma > 50:
                logger.error("Could not optimize gamma before gamma = %d", 50)
                return None

        return optgamma

    def reciprocal_sum(self):
        """
        Compute the reciprocal summation in the anisotropic Madelung
        potential.

        TODO: Get the input to fft cut by half by using rfft instead of fft
        """
        logger = logging.getLogger(__name__)
        logger.debug("Reciprocal summation in Madeling potential")
        over_atob = 1.0 / ang_to_bohr
        atob3 = ang_to_bohr**3

        latt = self.structure.lattice
        vol = latt.volume * atob3  # in Bohr^3

        reci_latt = latt.reciprocal_lattice
        [b1, b2, b3] = reci_latt.get_cartesian_coords(1)
        b1 = np.array(b1) * over_atob  # In 1/Bohr
        b2 = np.array(b2) * over_atob
        b3 = np.array(b3) * over_atob

        nx, ny, nz = self.dim
        logging.debug("nx: %d, ny: %d, nz: %d", nx, ny, nz)
        ind1 = np.arange(nx)
        for i in range(int(nx / 2), nx):
            ind1[i] = i - nx
        ind2 = np.arange(ny)
        for i in range(int(ny / 2), ny):
            ind2[i] = i - ny
        ind3 = np.arange(nz)
        for i in range(int(nz / 2), nz):
            ind3[i] = i - nz

        g_array = np.zeros(self.dim, np.dtype("c16"))
        gamm2 = 4 * (self.gamma**2)
        for i in ind1:
            for j in ind2:
                for k in ind3:
                    g = i * b1 + j * b2 + k * b3
                    g_eps_g = np.dot(g, np.dot(self.epsilon, g))
                    if i == j == k == 0:
                        continue
                    else:
                        g_array[i, j, k] = math.exp(-g_eps_g / gamm2) / g_eps_g

        r_array = np.fft.fftn(g_array)
        over_vol = 4 * np.pi / vol  # Multiply with q later
        r_array *= over_vol
        r_arr_real = np.real(r_array)
        r_arr_imag = np.imag(r_array)

        max_imag = r_arr_imag.max()
        logger.debug("Max imaginary part found to be %f", max_imag)

        return r_arr_real


warnings.warn(
    "Replacing PyCDT usage of Kumagai base classes and plotting with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All core Kumagai code will be removed with Version 2.5 of PyCDT."
    " (note these functions all exist in pymatgen)",
    DeprecationWarning,
)


class KumagaiCorrection(object):
    """
    Extended freysoldt correction developed by Kumagai and Oba.
    """

    def __init__(
        self,
        dielectric_tensor,
        q,
        gamma,
        g_sum,
        bulk_structure,
        defect_structure,
        energy_cutoff=520,
        madetol=0.0001,
        lengths=None,
        **kw
    ):
        """
        Args:
            dielectric_tensor:
                Macroscopic dielectric tensor
                Include ionic also if defect is relaxed, othewise ion clamped.
                Can be a matrix array or scalar.
            q:
                Charge associated with the defect. Typically integer
            gamma:
                Convergence parameter. Obtained from KumagaiBulkPart
            g_sum:
                value that is dependent on the Bulk only.
                Obtained from KumagaiBulkPart
            bulk_structure:
                bulk Pymatgen structure object. Need to specify this if
                using Outcar method for atomic site avg.
                (If you specify outcar files for bulk_file_path but dont
                specify structure then code will break)
                (TO DO: resolve this dumb dependency by being smarter
                about where structure comes from?)
            defect_structure:
                defect structure. Needed if using Outcar method
            energy_cutoff:
                Energy for plane wave cutoff (in eV).
                If not given, Materials Project default 520 eV is used.
            madetol:
                Tolerance for convergence of energy terms in eV
            lengths:
                Lengths of axes, for speeding up plotting slightly
            keywords:
                1) bulk_locpot: Bulk Locpot file path OR Bulk Locpot
                   defect_locpot: Defect Locpot file path or defect Locpot
                2) (Or) bulk_outcar:   Bulk Outcar file path
                   defect_outcar: Defect outcar file path
                3) defect_position: Defect position as a pymatgen Site object in the bulk supercell structure
                    NOTE: this is optional but recommended, if not provided then analysis is done to find
                    the defect position; this analysis has been rigorously tested, but has broken in an example with
                    severe long range relaxation
                    (at which point you probably should not be including the defect in your analysis...)
        """
        if isinstance(dielectric_tensor, int) or isinstance(dielectric_tensor, float):
            self.dieltens = np.identity(3) * dielectric_tensor
        else:
            self.dieltens = np.array(dielectric_tensor)

        if "bulk_locpot" in kw:
            if isinstance(kw["bulk_locpot"], Locpot):
                self.locpot_blk = kw["bulk_locpot"]
            else:
                self.locpot_blk = Locpot.from_file(kw["bulk_locpot"])
            if isinstance(kw["defect_locpot"], Locpot):
                self.locpot_def = kw["defect_locpot"]
            else:
                self.locpot_def = Locpot.from_file(kw["defect_locpot"])
            self.dim = self.locpot_blk.dim

            self.outcar_blk = None
            self.outcar_def = None
            self.do_outcar_method = False

        if "bulk_outcar" in kw:
            self.outcar_blk = Outcar(str(kw["bulk_outcar"]))
            self.outcar_def = Outcar(str(kw["defect_outcar"]))
            self.do_outcar_method = True
            self.locpot_blk = None
            self.locpot_def = None
            self.dim = self.outcar_blk.ngf

        if "defect_position" in kw:
            self._defpos = kw["defect_position"]
        else:
            self._defpos = None

        self.madetol = madetol
        self.q = q
        self.encut = energy_cutoff
        self.structure = bulk_structure
        self.defstructure = defect_structure
        self.gamma = gamma
        self.g_sum = g_sum

        self.lengths = lengths

    def correction(self, title=None, partflag="All"):
        """
        Computes the extended Freysoldt correction for anistropic systems
        developed by Y. Kumagai and F. Oba (Ref: PRB 89, 195205 (2014)
        Args:
            title:
                If plot of potential averaging process is wanted set title
            partflag:
                Specifies the part of correction computed
                'pc': periodic interaction of defect charges (point charge) only
                'potalign': potential alignmnet correction only,
                'All' (default): pc and potalign combined into one value,
                'AllSplit' for correction in form [PC, potterm, full]
        """
        logger = logging.getLogger(__name__)
        logger.info("This is Kumagai Correction.")

        if not self.q:
            if partflag == "AllSplit":
                return [0.0, 0.0, 0.0]
            else:
                return 0.0

        if partflag != "potalign":
            energy_pc = self.pc()

        if partflag != "pc":
            potalign = self.potalign(title=title)

        # logger.info('Kumagai Correction details:')
        # if partflag != 'potalign':
        #    logger.info('PCenergy (E_lat) = %f', round(energy_pc, 5))
        # if partflag != 'pc':
        #    logger.info('potential alignment (-q*delta V) = %f',
        #                 round(potalign, 5))
        if partflag in ["All", "AllSplit"]:
            logger.info("Total Kumagai correction = %f", round(energy_pc + potalign, 5))

        if partflag == "pc":
            return round(energy_pc, 5)
        elif partflag == "potalign":
            return round(potalign, 5)
        elif partflag == "All":
            return round(energy_pc + potalign, 5)
        else:
            return map(
                lambda x: round(x, 5), [energy_pc, potalign, energy_pc + potalign]
            )

    def pc(self):
        energy_pc = anisotropic_pc_energy(
            self.structure, self.g_sum, self.dieltens, self.q, self.gamma, self.madetol
        )

        logger = logging.getLogger(__name__)
        logger.info(
            "PC energy determined to be %f eV (%f Hartree)",
            energy_pc,
            energy_pc / hart_to_ev,
        )

        return energy_pc

    def potalign(self, title=None, output_sr=False):
        """
        Potential alignment for Kumagai method
        Args:
            title: Title for the plot. None will not generate the plot
            output_sr allows for output of the short range potential
                (Good for delocalization analysis)
        """
        logger = logging.getLogger(__name__)
        logger.info("\nRunning potential alignment (atomic site averaging)")

        angset, [a1, a2, a3], vol, determ, invdiel = kumagai_init(
            self.structure, self.dieltens
        )

        potinddict = disttrans(self.structure, self.defstructure, defpos=self._defpos)

        minlat = min(norm(a1), norm(a2), norm(a3))
        lat_perc_diffs = [100 * abs(norm(a1) - norm(lat)) / minlat for lat in [a2, a3]]
        lat_perc_diffs.append(100 * abs(norm(a2) - norm(a3)) / minlat)
        if not all(i < 45 for i in lat_perc_diffs):
            logger.warning("Detected that cell was not very cubic.")
            logger.warning(
                "Sampling atoms outside wigner-seitz cell may " "not be optimal"
            )
        wsrad = wigner_seitz_radius(self.structure)
        logger.debug("wsrad %f", wsrad)

        for i in potinddict.keys():
            logger.debug("Atom %d, distance: %f", i, potinddict[i]["dist"])
            if potinddict[i]["dist"] > wsrad:
                potinddict[i]["OutsideWS"] = True
            else:
                potinddict[i]["OutsideWS"] = False

        if not self.do_outcar_method:
            puredat = read_ES_avg_fromlocpot(self.locpot_blk)
            defdat = read_ES_avg_fromlocpot(self.locpot_def)
        else:
            puredat = {"potential": self.outcar_blk.electrostatic_potential}
            defdat = {"potential": self.outcar_def.electrostatic_potential}

        jup = 0
        for i in potinddict.keys():
            jup += 1
            if not title and not potinddict[i]["OutsideWS"]:
                # dont need to calculate inside WS if not printing plot
                continue

            j = potinddict[i]["def_site_index"]  # assuming zero defined
            k = potinddict[i]["bulk_site_index"]
            v_qb = defdat["potential"][j] - puredat["potential"][k]

            cart_reldef = potinddict[i]["cart_reldef"]
            v_pc = anisotropic_madelung_potential(
                self.structure,
                self.dim,
                self.g_sum,
                cart_reldef,
                self.dieltens,
                self.q,
                self.gamma,
                self.madetol,
            )
            v_qb *= -1  # change charge sign convention

            potinddict[i]["Vpc"] = v_pc
            potinddict[i]["Vqb"] = v_qb

            logger.debug("Atom: %d, anisotropic madelung potential: %f", i, v_pc)
            logger.debug("Atom: %d, bulk/defect difference = %f", i, v_qb)

        if title:
            fullspecset = self.structure.species
            specset = list(set(fullspecset))
            shade, forplot = {}, {}
            for i in specset:
                shade[i.symbol] = {"r": [], "Vpc": [], "Vqb": []}
                forplot[i.symbol] = {"r": [], "Vpc": [], "Vqb": [], "sites": []}

        forcorrection = []
        for i in potinddict.keys():
            if not title and not potinddict[i]["OutsideWS"]:
                continue
            if potinddict[i]["OutsideWS"]:
                forcorrection.append(potinddict[i]["Vqb"] - potinddict[i]["Vpc"])
                if title:
                    elt = fullspecset[i].symbol
                    shade[elt]["r"].append(potinddict[i]["dist"])
                    shade[elt]["Vpc"].append(potinddict[i]["Vpc"])
                    shade[elt]["Vqb"].append(potinddict[i]["Vqb"])
            if title:
                elt = fullspecset[i].symbol
                forplot[elt]["r"].append(potinddict[i]["dist"])
                forplot[elt]["Vpc"].append(potinddict[i]["Vpc"])
                forplot[elt]["Vqb"].append(potinddict[i]["Vqb"])
                forplot[elt]["sites"].append(potinddict[i]["siteobj"])

        potalign = np.mean(forcorrection)

        if title:
            forplot["EXTRA"] = {"wsrad": wsrad, "potalign": potalign}
            try:
                forplot["EXTRA"]["lengths"] = self.structure.lattice.abc
            except:
                forplot["EXTRA"]["lengths"] = self.lengths

            if title != "written":
                KumagaiCorrection.plot(forplot, title=title)
            else:
                # TODO: use a more descriptive fname that describes the defect
                from monty.json import MontyEncoder
                from monty.serialization import dumpfn

                fname = "KumagaiData.json"
                dumpfn(forplot, fname, cls=MontyEncoder)

        logger.info("potential alignment (site averaging): %f", np.mean(forcorrection))
        logger.info(
            "Potential correction energy: %f eV", -self.q * np.mean(forcorrection)
        )

        if output_sr:
            outpot = {"sampled": forcorrection, "alldata": potinddict}
            return (
                (-self.q * np.mean(forcorrection)),
                outpot,
            )  # pot align energy correction (eV)
        else:
            return -self.q * np.mean(forcorrection)  # pot align energy correction (eV)

    @classmethod
    def plot(cls, forplot, title):
        """
        Plotting of locpot data
        TODO: Rename forplot to a more descriptive name
        """
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        plt.figure()
        plt.clf()
        collis = ["b", "g", "c", "m", "y", "w", "k"]
        ylis = []
        rlis = []
        for i in range(len(forplot.keys())):
            inkey = list(forplot.keys())[i]
            if inkey == "EXTRA":
                continue
            for k in forplot[inkey]["r"]:
                rlis.append(k)
            for k in ["Vqb", "Vpc"]:
                for u in forplot[inkey][k]:
                    ylis.append(u)
            plt.plot(
                forplot[inkey]["r"],
                forplot[inkey]["Vqb"],
                color=collis[i],
                marker="^",
                linestyle="None",
                label=str(inkey) + ": $V_{q/b}$",
            )
            plt.plot(
                forplot[inkey]["r"],
                forplot[inkey]["Vpc"],
                color=collis[i],
                marker="o",
                linestyle="None",
                label=str(inkey) + ": $V_{pc}$",
            )
        full = []
        for i in forplot.keys():
            if i == "EXTRA":
                continue
            for k in range(len(forplot[i]["Vpc"])):
                full.append(
                    [forplot[i]["r"][k], forplot[i]["Vqb"][k] - forplot[i]["Vpc"][k]]
                )
        realfull = sorted(full, key=lambda x: x[0])
        r, y = [], []
        for i in realfull:
            r.append(i[0])
            y.append(i[1])
        wsrad = forplot["EXTRA"]["wsrad"]
        potalign = forplot["EXTRA"]["potalign"]
        plt.plot(
            r,
            y,
            color=collis[-1],
            marker="x",
            linestyle="None",
            label="$V_{q/b}$ - $V_{pc}$",
        )
        plt.xlabel("Distance from defect ($\AA$)", fontsize=20)
        plt.ylabel("Potential (V)", fontsize=20)

        x = np.arange(wsrad, max(forplot["EXTRA"]["lengths"]), 0.01)
        plt.fill_between(
            x,
            min(ylis) - 1,
            max(ylis) + 1,
            facecolor="red",
            alpha=0.15,
            label="sampling region",
        )
        plt.axhline(y=potalign, linewidth=0.5, color="red", label="pot. align. / q")

        fontP = FontProperties()
        fontP.set_size("small")
        plt.legend(bbox_to_anchor=(1.05, 0.5), prop=fontP)
        plt.axhline(y=0, linewidth=0.2, color="black")
        plt.ylim([min(ylis) - 0.5, max(ylis) + 0.5])
        plt.xlim([0, max(rlis) + 3])

        plt.title("%s atomic site potential plot" % title)
        plt.savefig("%s_kumagaisiteavgPlot.pdf" % title)

    @classmethod
    def plot_from_datfile(cls, name="KumagaiData.json", title="default"):
        """
        Takes data file called 'name' and does plotting.
        Good for later plotting of locpot data after running run_correction()

        """
        from monty.json import MontyDecoder
        from monty.serialization import loadfn

        forplot = loadfn(name, cls=MontyDecoder)
        cls.plot(forplot, title=title)
