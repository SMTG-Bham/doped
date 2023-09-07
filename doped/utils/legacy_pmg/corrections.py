# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
"""
This code has been copied over from pymatgen==2022.7.25, as it was deleted in
later versions. This is a temporary measure while refactoring to use the new
pymatgen-analysis-defects package takes place.

Implementation of defect correction methods.
"""

import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from pymatgen.analysis.defects.utils import (
    QModel,
    ang_to_bohr,
    converge,
    eV_to_k,
    generate_reciprocal_vectors_squared,
    hart_to_ev,
    kumagai_to_V,
)
from scipy import stats

from doped.plotting import _get_backend
from doped.utils.legacy_pmg import DefectCorrection

logger = logging.getLogger(__name__)


def _get_defect_structure_from_calc_metadata(calculation_metadata):
    keys = [
        "unrelaxed_defect_structure",
        "guessed_initial_defect_structure",
        "defect_structure",
        "final_defect_structure",
    ]

    for key in keys:
        defect_structure = calculation_metadata.get(key)
        if defect_structure is not None:
            return defect_structure

    raise ValueError(
        "No defect structure found in calculation_metadata, so cannot compute finite-size charge "
        "correction!"
    )


class FreysoldtCorrection(DefectCorrection):
    """
    A class for FreysoldtCorrection class. Largely adapted from PyCDT code.

    If this correction is used, please reference Freysoldt's original paper.
    doi: 10.1103/PhysRevLett.102.016402
    """

    def __init__(
        self,
        dielectric_const,
        q_model=None,
        energy_cutoff=520,
        madetol=0.0001,
        axis=None,
    ):
        """
        Initializes the FreysoldtCorrection class
        Args:
            dielectric_const (float or 3x3 matrix): Dielectric constant for the structure
            q_model (QModel): instantiated QModel object or None.
                Uses default calculation_metadata to instantiate QModel if None supplied
            energy_cutoff (int): Maximum energy in eV in reciprocal space to perform
                integration for potential correction.
            madeltol(float): Convergence criteria for the Madelung energy for potential correction
            axis (int): Axis to calculate correction.
                If axis is None, then averages over all three axes is performed.
        """
        self.q_model = q_model if q_model else QModel()
        self.energy_cutoff = energy_cutoff
        self.madetol = madetol
        self.dielectric_const = dielectric_const

        if isinstance(dielectric_const, (int, float)):
            self.dielectric = float(dielectric_const)
        else:
            self.dielectric = float(np.mean(np.diag(dielectric_const)))

        self.axis = axis

        self.metadata = {"pot_plot_data": {}, "pot_corr_uncertainty_md": {}}

    def get_correction(self, entry):
        """
        Gets the Freysoldt correction for a defect entry
        Args:
            entry (DefectEntry): defect entry to compute Freysoldt correction on.

                Requires following keys to exist in DefectEntry.calculation_metadata dict:

                    axis_grid (3 x NGX where NGX is the length of the NGX grid
                    in the x,y and z axis directions. Same length as planar
                    average lists):
                        A list of 3 numpy arrays which contain the Cartesian axis
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

                    defect_structure (Structure) structure corresponding to
                        defect supercell structure (uses Lattice for charge correction)

                    defect_frac_sc_coords (3 x 1 array) Fractional coordinates of
                        defect location in supercell structure
        Returns:
            FreysoldtCorrection values as a dictionary
        """
        if self.axis is None:
            list_axis_grid = np.array(entry.calculation_metadata["axis_grid"], dtype=object)
            list_bulk_plnr_avg_esp = np.array(
                entry.calculation_metadata["bulk_planar_averages"], dtype=object
            )
            list_defect_plnr_avg_esp = np.array(
                entry.calculation_metadata["defect_planar_averages"], dtype=object
            )
            list_axes = range(len(list_axis_grid))
        else:
            list_axes = np.array(self.axis)
            list_axis_grid, list_bulk_plnr_avg_esp, list_defect_plnr_avg_esp = (
                [],
                [],
                [],
            )
            for ax in list_axes:
                list_axis_grid.append(np.array(entry.calculation_metadata["axis_grid"][ax]))
                list_bulk_plnr_avg_esp.append(
                    np.array(entry.calculation_metadata["bulk_planar_averages"][ax])
                )
                list_defect_plnr_avg_esp.append(
                    np.array(entry.calculation_metadata["defect_planar_averages"][ax])
                )

        defect_structure = _get_defect_structure_from_calc_metadata(entry.calculation_metadata)
        lattice = defect_structure.lattice.copy()
        defect_frac_sc_coords = entry.sc_defect_frac_coords

        es_corr = self.perform_es_corr(lattice, entry.charge_state)

        pot_corr_tracker = []

        for x, pureavg, defavg, axis in zip(
            list_axis_grid, list_bulk_plnr_avg_esp, list_defect_plnr_avg_esp, list_axes
        ):
            tmp_pot_corr = self.perform_pot_corr(
                x,
                pureavg,
                defavg,
                lattice,
                entry.charge_state,
                defect_frac_sc_coords,
                axis,
                widthsample=1.0,
            )
            pot_corr_tracker.append(tmp_pot_corr)

        pot_corr = np.mean(pot_corr_tracker)

        metadata = entry.calculation_metadata.setdefault("freysoldt_meta", {})
        metadata.update(self.metadata)  # updates bandfilling_metadata (as dictionaries are mutable) if
        # it already exists, otherwise creates it

        entry.calculation_metadata["potalign"] = (
            pot_corr / (-entry.charge_state) if entry.charge_state else 0.0
        )

        return {
            "freysoldt_electrostatic": es_corr,
            "freysoldt_potential_alignment": pot_corr,
        }

    def perform_es_corr(self, lattice, q, step=1e-4):
        """
        Perform Electrostatic Freysoldt Correction
        Args:
            lattice: Pymatgen lattice object
            q (int): Charge of defect
            step (float): step size for numerical integration
        Return:
            Electrostatic Point Charge contribution to Freysoldt Correction (float).
        """
        logger.info("Running Freysoldt 2011 PC calculation (should be equivalent to sxdefectalign)")
        logger.debug("defect lattice constants are (in angstroms)" + str(lattice.abc))

        [a1, a2, a3] = ang_to_bohr * np.array(lattice.get_cartesian_coords(1))
        logging.debug("In atomic units, lat consts are (in bohr):" + str([a1, a2, a3]))
        vol = np.dot(a1, np.cross(a2, a3))  # vol in bohr^3

        def e_iso(encut):
            gcut = eV_to_k(encut)  # gcut is in units of 1/A
            return (
                scipy.integrate.quad(lambda g: self.q_model.rho_rec(g * g) ** 2, step, gcut)[0]
                * (q**2)
                / np.pi
            )

        def e_per(encut):
            eper = 0
            for g2 in generate_reciprocal_vectors_squared(a1, a2, a3, encut):
                eper += (self.q_model.rho_rec(g2) ** 2) / g2
            eper *= (q**2) * 2 * round(np.pi, 6) / vol
            eper += (q**2) * 4 * round(np.pi, 6) * self.q_model.rho_rec_limit0 / vol
            return eper

        eiso = converge(e_iso, 5, self.madetol, self.energy_cutoff)
        logger.debug("Eisolated : %f", round(eiso, 5))

        eper = converge(e_per, 5, self.madetol, self.energy_cutoff)

        logger.info("Eperiodic : %f hartree", round(eper, 5))
        logger.info("difference (periodic-iso) is %f hartree", round(eper - eiso, 6))
        logger.info("difference in (eV) is %f", round((eper - eiso) * hart_to_ev, 4))

        es_corr = round((eiso - eper) / self.dielectric * hart_to_ev, 6)
        logger.info("Defect Correction without alignment %f (eV): ", es_corr)
        return es_corr

    def perform_pot_corr(
        self,
        axis_grid,
        pureavg,
        defavg,
        lattice,
        q,
        defect_frac_position,
        axis,
        widthsample=1.0,
    ):
        """
        For performing planar averaging potential alignment
        Args:
             axis_grid (1 x NGX where NGX is the length of the NGX grid
                    in the axis direction. Same length as pureavg list):
                        A numpy array which contain the Cartesian axis
                        values (in angstroms) that correspond to each planar avg
                        potential supplied.
             pureavg (1 x NGX where NGX is the length of the NGX grid in
                    the axis direction.):
                        A numpy array for the planar averaged
                        electrostatic potential of the bulk supercell.
             defavg (1 x NGX where NGX is the length of the NGX grid in
                    the axis direction.):
                        A numpy array for the planar averaged
                        electrostatic potential of the defect supercell.
             lattice: Pymatgen Lattice object of the defect supercell
             q (float or int): charge of the defect
             defect_frac_position: Fracitional Coordinates of the defect in the supercell
             axis (int): axis for performing the freysoldt correction on
             widthsample (float): width (in Angstroms) of the region in between defects
                where the potential alignment correction is averaged. Default is 1 Angstrom.

        Returns:
            Potential Alignment contribution to Freysoldt Correction (float).
        """
        logging.debug("run Freysoldt potential alignment method for axis " + str(axis))
        nx = len(axis_grid)

        # shift these planar averages to have defect at origin
        axfracval = defect_frac_position[axis]
        axbulkval = axfracval * lattice.abc[axis]
        if axbulkval < 0:
            axbulkval += lattice.abc[axis]
        elif axbulkval > lattice.abc[axis]:
            axbulkval -= lattice.abc[axis]

        if axbulkval:
            for i in range(nx):
                if axbulkval < axis_grid[i]:
                    break
            rollind = len(axis_grid) - i
            pureavg = np.roll(pureavg, rollind)
            defavg = np.roll(defavg, rollind)

        # if not self._silence:
        logger.debug("calculating lr part along planar avg axis")
        reci_latt = lattice.reciprocal_lattice
        dg = reci_latt.abc[axis]
        dg /= ang_to_bohr  # convert to bohr to do calculation in atomic units

        # Build background charge potential with defect at origin
        v_G = np.empty(len(axis_grid), np.dtype("c16"))
        v_G[0] = 4 * np.pi * -q / self.dielectric * self.q_model.rho_rec_limit0
        g = np.roll(np.arange(-nx / 2, nx / 2, 1, dtype=int), nx // 2) * dg
        g2 = np.multiply(g, g)[1:]
        v_G[1:] = 4 * np.pi / (self.dielectric * g2) * -q * self.q_model.rho_rec(g2)
        v_G[nx // 2] = 0 if not (nx % 2) else v_G[nx // 2]

        # Get the real space potential by performing a  fft and grabbing the imaginary portion
        v_R = np.fft.fft(v_G)

        if abs(np.imag(v_R).max()) > self.madetol:
            raise Exception("imaginary part found to be %s", repr(np.imag(v_R).max()))
        v_R /= lattice.volume * ang_to_bohr**3
        v_R = np.real(v_R) * hart_to_ev

        # get correction
        short = np.array(defavg) - np.array(pureavg) - np.array(v_R)
        checkdis = int((widthsample / 2) / (axis_grid[1] - axis_grid[0]))
        mid = int(len(short) / 2)

        tmppot = [short[i] for i in range(mid - checkdis, mid + checkdis + 1)]
        logger.debug("shifted defect position on axis (%s) to origin", repr(axbulkval))
        logger.debug(
            "means sampling region is (%f,%f)",
            axis_grid[mid - checkdis],
            axis_grid[mid + checkdis],
        )

        C = -np.mean(tmppot)
        logger.debug("C = %f", C)
        final_shift = [short[j] + C for j in range(len(v_R))]
        v_R = [elmnt - C for elmnt in v_R]

        logger.info("C value is averaged to be %f eV ", C)
        logger.info("Potentital alignment energy correction (-q*delta V):  %f (eV)", -q * C)
        self.pot_corr = -q * C

        # log plotting data:
        self.metadata["pot_plot_data"][axis] = {
            "Vr": v_R,
            "x": axis_grid,
            "dft_diff": np.array(defavg) - np.array(pureavg),
            "final_shift": final_shift,
            "check": [mid - checkdis, mid + checkdis + 1],
        }

        # log uncertainty:
        self.metadata["pot_corr_uncertainty_md"][axis] = {
            "stats": stats.describe(tmppot)._asdict(),
            "potcorr": -q * C,
        }

        return self.pot_corr

    def plot(self, axis, title=None, saved=False):
        """
        Plots the planar average electrostatic potential against the Long range
        and short range models from Freysoldt. Must run perform_pot_corr or
        get_correction (to load metadata) before this can be used.

        Args:
             axis (int): axis to plot
             title (str): Title to be given to plot. Default is no title.
             saved (bool): Whether to save file or not. If False then returns plot
                object. If True then saves plot as   str(title) + "FreyplnravgPlot.pdf".
        """
        if not self.metadata["pot_plot_data"]:
            raise ValueError("Cannot plot potential alignment before running correction!")

        with plt.style.context(f"{os.path.dirname(__file__)}/../doped.mplstyle"):
            x = self.metadata["pot_plot_data"][axis]["x"]
            v_R = self.metadata["pot_plot_data"][axis]["Vr"]
            dft_diff = self.metadata["pot_plot_data"][axis]["dft_diff"]
            final_shift = self.metadata["pot_plot_data"][axis]["final_shift"]
            check = self.metadata["pot_plot_data"][axis]["check"]

            plt.figure()
            plt.clf()
            plt.plot(x, v_R, c="green", zorder=1, label="Long range from model")
            plt.plot(x, dft_diff, c="red", label="DFT locpot diff")
            plt.plot(x, final_shift, c="blue", label="Short range (aligned)")

            tmpx = [x[i] for i in range(check[0], check[1])]
            plt.fill_between(tmpx, -100, 100, facecolor="red", alpha=0.15, label="Sampling region")

            plt.xlim(round(x[0]), round(x[-1]))
            ymin = min(*v_R, *dft_diff, *final_shift)
            ymax = max(*v_R, *dft_diff, *final_shift)
            plt.ylim(-0.2 + ymin, 0.2 + ymax)
            plt.xlabel(r"Distance along axis ($\AA$)")
            plt.ylabel("Potential (V)")
            plt.legend(loc=9)
            plt.axhline(y=0, linewidth=0.2, color="black")
            plt.title(f"{title!s} Defect Potential")
            plt.xlim(0, max(x))
            if saved:
                plt.savefig(
                    f"{title!s}FreyplnravgPlot.pdf",
                    bbox_inches="tight",
                    backend=_get_backend("pdf"),
                    transparent=True,
                )
                return None
            return plt


class KumagaiCorrection(DefectCorrection):
    """
    A class for KumagaiCorrection class. Largely adapted from PyCDT code.

    If this correction is used, please reference Kumagai and Oba's original
    paper (doi: 10.1103/PhysRevB.89.195205) as well as Freysoldt's original
    paper (doi: 10.1103/PhysRevLett.102.016402)

    NOTE that equations 8 and 9 from Kumagai et al. reference are divided by (4
    pi) to get SI units
    """

    def __init__(self, dielectric_tensor, sampling_radius=None, gamma=None):
        """
        Initializes the Kumagai Correction
        Args:
            dielectric_tensor (float or 3x3 matrix): Dielectric constant for the structure.

            optional data that can be tuned:
                sampling_radius (float): radius (in Angstrom) which sites must be outside
                    of to be included in the correction. Publication by Kumagai advises to
                    use Wigner-Seitz radius of defect supercell, so this is default value.
                gamma (float): convergence parameter for gamma function.
                    Code will automatically determine this if set to None.
        """
        self.metadata = {
            "gamma": gamma,
            "sampling_radius": sampling_radius,
            "potalign": None,
        }

        if isinstance(dielectric_tensor, (int, float)):
            self.dielectric = np.identity(3) * dielectric_tensor
        else:
            self.dielectric = np.array(dielectric_tensor)

    def get_correction(self, entry):
        """
        Gets the Kumagai correction for a defect entry
        Args:
            entry (DefectEntry): defect entry to compute Kumagai correction on.

                Requires following calculation_metadata in the DefectEntry to exist:

                    bulk_atomic_site_averages (list):  list of bulk structure"s atomic site averaged
                        ESPs * charge, in same order as indices of bulk structure note this is list
                        given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    defect_atomic_site_averages (list):  list of defect structure"s atomic site averaged
                        ESPs * charge, in same order as indices of defect structure note this is list
                        given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    site_matching_indices (list):  list of corresponding site index values for
                        bulk and defect site structures EXCLUDING the defect site itself
                        (e.g. [[bulk structure site index, defect structure"s corresponding site index],
                        ... ])

                    initial_defect_structure (Structure): Pymatgen Structure object representing
                        un-relaxed defect structure

                    defect_frac_sc_coords (array): Defect position in fractional coordinates of the
                        supercell given in bulk_structure
        Returns:
            KumagaiCorrection values as a dictionary

        """
        bulk_atomic_site_averages = entry.calculation_metadata["bulk_atomic_site_averages"]
        defect_atomic_site_averages = entry.calculation_metadata["defect_atomic_site_averages"]
        site_matching_indices = entry.calculation_metadata["site_matching_indices"]
        defect_structure = _get_defect_structure_from_calc_metadata(entry.calculation_metadata)
        defect_frac_sc_coords = entry.sc_defect_frac_coords

        lattice = defect_structure.lattice
        volume = lattice.volume

        if not self.metadata["gamma"]:
            self.metadata["gamma"] = tune_for_gamma(lattice, self.dielectric)

        prec_set = [25, 28]
        g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
            self.metadata["gamma"], prec_set, lattice, self.dielectric
        )

        pot_shift = self.get_potential_shift(self.metadata["gamma"], volume)
        si = self.get_self_interaction(self.metadata["gamma"])
        es_corr = [(real_summation[ind] + recip_summation[ind] + pot_shift + si) for ind in range(2)]

        # increase precision if correction is not converged yet
        # TODO: allow for larger prec_set to be tried if this fails
        if abs(es_corr[0] - es_corr[1]) > 0.0001:
            logger.debug(
                f"Es_corr summation not converged! ({es_corr[0]} vs. {es_corr[1]})\nTrying a larger "
                f"prec_set..."
            )
            prec_set = [30, 35]
            g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
                self.metadata["gamma"], prec_set, lattice, self.dielectric
            )
            es_corr = [(real_summation[ind] + recip_summation[ind] + pot_shift + si) for ind in range(2)]
            if abs(es_corr[0] - es_corr[1]) < 0.0001:
                raise ValueError(
                    "Correction still not converged after trying prec_sets up to 35... serious error."
                )

        es_corr = es_corr[0] * -(entry.charge_state**2.0) * kumagai_to_V / 2.0  # [eV]

        # if no sampling radius specified for pot align, then assuming Wigner-Seitz radius:
        if not self.metadata["sampling_radius"]:
            wz = lattice.get_wigner_seitz_cell()
            dist = []
            for facet in wz:
                midpt = np.mean(np.array(facet), axis=0)
                dist.append(np.linalg.norm(midpt))
            self.metadata["sampling_radius"] = min(dist)

        # assemble site_list based on matching indices
        # [[defect_site object, Vqb for site], .. repeat for all non defective sites]
        site_list = []
        for bs_ind, ds_ind in site_matching_indices:
            Vqb = -(defect_atomic_site_averages[int(ds_ind)] - bulk_atomic_site_averages[int(bs_ind)])
            site_list.append([defect_structure[int(ds_ind)], Vqb])

        pot_corr = self.perform_pot_corr(
            defect_structure,
            defect_frac_sc_coords,
            site_list,
            self.metadata["sampling_radius"],
            entry.charge_state,
            r_vecs[0],
            g_vecs[0],
            self.metadata["gamma"],
        )

        metadata = entry.calculation_metadata.setdefault("kumagai_meta", {})
        metadata.update(self.metadata)  # updates bandfilling_metadata (as dictionaries are mutable) if
        # it already exists, otherwise creates it

        entry.calculation_metadata["potalign"] = (
            pot_corr / (-entry.charge_state) if entry.charge_state else 0.0
        )

        return {
            "kumagai_electrostatic": es_corr,
            "kumagai_potential_alignment": pot_corr,
        }

    def perform_es_corr(self, gamma, prec, lattice, charge):
        """
        Perform Electrostatic Kumagai Correction
        Args:
            gamma (float): Ewald parameter
            prec (int): Precision parameter for reciprical/real lattice vector generation
            lattice: Pymatgen Lattice object corresponding to defect supercell
            charge (int): Defect charge
        Return:
            Electrostatic Point Charge contribution to Kumagai Correction (float).
        """
        volume = lattice.volume

        g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
            gamma, [prec], lattice, self.dielectric
        )
        recip_summation = recip_summation[0]
        real_summation = real_summation[0]

        es_corr = (
            recip_summation
            + real_summation
            + self.get_potential_shift(gamma, volume)
            + self.get_self_interaction(gamma)
        )

        es_corr *= -(charge**2.0) * kumagai_to_V / 2.0  # [eV]

        return es_corr

    def perform_pot_corr(
        self,
        defect_structure,
        defect_frac_coords,
        site_list,
        sampling_radius,
        q,
        r_vecs,
        g_vecs,
        gamma,
    ):
        """
        For performing potential alignment in manner described by Kumagai et
        al.

        Args:
            defect_structure: Pymatgen Structure object corresponding to the defect supercell.

            defect_frac_coords (array): Defect Position in fractional coordinates of the supercell
                given in bulk_structure

            site_list: list of corresponding site index values for
                bulk and defect site structures EXCLUDING the defect site itself
                (ex. [[bulk structure site index, defect structure"s corresponding site index], ... ]

            sampling_radius (float): radius (in Angstrom) which sites must be outside
                of to be included in the correction. Publication by Kumagai advises to
                use Wigner-Seitz radius of defect supercell, so this is default value.

            q (int): Defect charge

            r_vecs: List of real lattice vectors to use in summation

            g_vecs: List of reciprocal lattice vectors to use in summation

            gamma (float): Ewald parameter

        Return:
            Potential alignment contribution to Kumagai Correction (float)
        """
        volume = defect_structure.lattice.volume
        potential_shift = self.get_potential_shift(gamma, volume)

        pot_dict = {}  # keys will be site index in the defect structure
        for_correction = []  # region to sample for correction

        # for each atom, do the following:
        # (a) get relative_vector from defect_site to site in defect_supercell structure
        # (b) recalculate the recip and real summation values based on this r_vec
        # (c) get information needed for pot align
        for site, Vqb in site_list:
            dist, jimage = site.distance_and_image_from_frac_coords(defect_frac_coords)
            vec_defect_to_site = defect_structure.lattice.get_cartesian_coords(
                site.frac_coords - jimage - defect_frac_coords
            )
            dist_to_defect = np.linalg.norm(vec_defect_to_site)
            if abs(dist_to_defect - dist) > 0.001:
                raise ValueError("Error in computing vector to defect")

            relative_real_vectors = [r_vec - vec_defect_to_site for r_vec in r_vecs[:]]

            real_sum = self.get_real_summation(gamma, relative_real_vectors)
            recip_sum = self.get_recip_summation(gamma, g_vecs, volume, r=vec_defect_to_site[:])

            Vpc = (real_sum + recip_sum + potential_shift) * kumagai_to_V * q

            defect_struct_index = defect_structure.index(site)
            pot_dict[defect_struct_index] = {
                "Vpc": Vpc,
                "Vqb": Vqb,
                "dist_to_defect": dist_to_defect,
            }

            logger.debug(f"For atom {defect_struct_index}\n\tbulk/defect DFT potential difference = {Vqb}")
            logger.debug(f"\tanisotropic model charge: {Vpc}")
            logger.debug(f"\t\treciprocal part: {recip_sum * kumagai_to_V * q}")
            logger.debug(f"\t\treal part: {real_sum * kumagai_to_V * q}")
            logger.debug(f"\t\tself interaction part: {potential_shift * kumagai_to_V * q}")
            logger.debug(f"\trelative_vector to defect: {vec_defect_to_site}")

            if dist_to_defect > sampling_radius:
                logger.debug(
                    f"\tdistance to defect is {dist_to_defect} which is outside minimum sampling "
                    f"radius {sampling_radius}"
                )
                for_correction.append(Vqb - Vpc)
            else:
                logger.debug(
                    f"\tdistance to defect is {dist_to_defect} which is inside minimum sampling "
                    f"radius {sampling_radius} (so will not include for correction)"
                )

        if for_correction:
            pot_alignment = np.mean(for_correction)
        else:
            logger.info(
                "No atoms sampled for_correction radius! Assigning potential alignment value of 0."
            )
            pot_alignment = 0.0

        self.metadata["potalign"] = pot_alignment
        pot_corr = -q * pot_alignment

        # log uncertainty stats:
        self.metadata["pot_corr_uncertainty_md"] = {
            "stats": stats.describe(for_correction)._asdict(),
            "number_sampled": len(for_correction),
        }
        self.metadata["pot_plot_data"] = pot_dict

        logger.info("Kumagai potential alignment (site averaging): %f", pot_alignment)
        logger.info("Kumagai potential alignment correction energy: %f eV", pot_corr)

        return pot_corr

    def get_real_summation(self, gamma, real_vectors):
        """
        Get real summation term from list of real-space vectors.
        """
        real_part = 0
        invepsilon = np.linalg.inv(self.dielectric)
        rd_epsilon = np.sqrt(np.linalg.det(self.dielectric))

        for r_vec in real_vectors:
            if np.linalg.norm(r_vec) > 1e-8:
                loc_res = np.sqrt(np.dot(r_vec, np.dot(invepsilon, r_vec)))
                nmr = scipy.special.erfc(gamma * loc_res)  # pylint: disable=E1101
                real_part += nmr / loc_res

        real_part /= 4 * np.pi * rd_epsilon

        return real_part

    def get_recip_summation(self, gamma, recip_vectors, volume, r=None):
        """
        Get Reciprocal summation term from list of reciprocal-space vectors.
        """
        if r is None:
            r = np.array([0, 0, 0])
        recip_part = 0

        for g_vec in recip_vectors:
            # dont need to avoid G=0, because it will not be
            # in recip list (if generate_R_and_G_vecs is used)
            Gdotdiel = np.dot(g_vec, np.dot(self.dielectric, g_vec))
            summand = np.exp(-Gdotdiel / (4 * (gamma**2))) * np.cos(np.dot(g_vec, r)) / Gdotdiel
            recip_part += summand

        recip_part /= volume

        return recip_part

    def get_self_interaction(self, gamma):
        """
        Returns the self-interaction energy of defect.
        """
        determ = np.linalg.det(self.dielectric)
        return -gamma / (2.0 * np.pi * np.sqrt(np.pi * determ))

    @staticmethod
    def get_potential_shift(gamma, volume):
        """
        Args:
            gamma (float): Gamma
            volume (float): Volume.

        Returns:
            Potential shift for defect.
        """
        return -0.25 / (volume * gamma**2.0)

    def plot(self, title=None, saved=False):
        """
        Plots the AtomicSite electrostatic potential against the long and short
        range models from Kumagai and Oba (doi: 10.1103/PhysRevB.89.195205).
        """
        if "pot_plot_data" not in self.metadata.keys():
            raise ValueError("Cannot plot potential alignment before running correction!")

        sampling_radius = self.metadata["sampling_radius"]
        site_dict = self.metadata["pot_plot_data"]
        potalign = self.metadata["potalign"]

        plt.figure()
        plt.clf()

        distances, sample_region = [], []
        Vqb_list, Vpc_list, diff_list = [], [], []
        for _site_ind, _site_dict in site_dict.items():
            dist = _site_dict["dist_to_defect"]
            distances.append(dist)

            Vqb = _site_dict["Vqb"]
            Vpc = _site_dict["Vpc"]

            Vqb_list.append(Vqb)
            Vpc_list.append(Vpc)
            diff_list.append(Vqb - Vpc)

            if dist > sampling_radius:
                sample_region.append(Vqb - Vpc)

        with plt.style.context(f"{os.path.dirname(__file__)}/../doped.mplstyle"):
            plt.plot(
                distances,
                Vqb_list,
                color="r",
                marker="^",
                linestyle="None",
                label="$V_{q/b}$",
            )

            plt.plot(
                distances,
                Vpc_list,
                color="g",
                marker="o",
                linestyle="None",
                label="$V_{pc}$",
            )

            plt.plot(
                distances,
                diff_list,
                color="b",
                marker="x",
                linestyle="None",
                label="$V_{q/b}$ - $V_{pc}$",
            )

            x = np.arange(sampling_radius, max(distances) * 1.05, 0.01)
            y_max = max(*Vqb_list, *Vpc_list, *diff_list) + 0.1
            y_min = min(*Vqb_list, *Vpc_list, *diff_list) - 0.1
            plt.fill_between(x, y_min, y_max, facecolor="red", alpha=0.15, label="Sampling region")
            plt.axhline(y=potalign, linewidth=0.5, color="red", label="Pot. align. / -q")

            plt.legend(loc=0)
            plt.axhline(y=0, linewidth=0.2, color="black")

            plt.ylim([y_min, y_max])
            plt.xlim([0, max(distances) * 1.1])

            plt.xlabel(r"Distance from defect ($\AA$)")
            plt.ylabel("Potential (V)")
            plt.title(f"{title!s} Atomic Site Potential")

            if saved:
                plt.savefig(
                    f"{title!s}KumagaiESPavgPlot.pdf",
                    bbox_inches="tight",
                    backend=_get_backend("pdf"),
                    transparent=True,
                )
                return None
            return plt


class BandFillingCorrection(DefectCorrection):
    """
    A class for BandFillingCorrection class.

    Largely adapted from PyCDT code.
    """

    def __init__(self, resolution=0.01):
        """
        Initializes the Bandfilling correction.

        Args:
            resolution (float): energy resolution to maintain for gap states
        """
        self.resolution = resolution
        self.metadata = {"num_hole_vbm": None, "num_elec_cbm": None, "potalign": None}

    def get_correction(self, entry):
        """
        Gets the BandFilling correction for a defect entry
        Args:
            entry (DefectEntry): defect entry to compute BandFilling correction on.
                Requires following calculation_metadata in the DefectEntry to exist:
                    eigenvalues
                        dictionary of defect eigenvalues, as stored in a Vasprun object.

                    kpoint_weights (list of floats)
                        kpoint weights corresponding to the dictionary of eigenvalues,
                        as stored in a Vasprun object

                    potalign (float)
                        potential alignment for the defect calculation
                        Only applies to non-zero charge,
                        When using potential alignment correction (freysoldt or kumagai),
                        need to divide by -q

                    cbm (float)
                        CBM of bulk calculation (or band structure calculation of bulk);
                        calculated on same level of theory as the defect
                        (ex. GGA defects -> requires GGA cbm)

                    vbm (float)
                        VBM of bulk calculation (or band structure calculation of bulk);
                        calculated on same level of theory as the defect
                        (ex. GGA defects -> requires GGA vbm)

                    run_metadata["defect_incar"] (dict)
                        Dictionary of INCAR settings for the defect calculation,
                        required to check if the calculation included spin-orbit coupling
                        (to determine the spin factor for occupancies of the electron bands)

        Returns:
            Bandfilling Correction value as a dictionary

        """
        eigenvalues = entry.calculation_metadata["eigenvalues"]
        kpoint_weights = entry.calculation_metadata["kpoint_weights"]
        potalign = entry.calculation_metadata["potalign"]
        vbm = entry.calculation_metadata["vbm"]
        cbm = entry.calculation_metadata["cbm"]
        soc_calc = entry.calculation_metadata["run_metadata"]["defect_incar"].get("LSORBIT")

        bf_corr = self.perform_bandfill_corr(eigenvalues, kpoint_weights, potalign, vbm, cbm, soc_calc)

        metadata = entry.calculation_metadata.setdefault("bandfilling_meta", {})
        metadata.update(self.metadata)  # updates bandfilling_metadata (as dictionaries are mutable) if
        # it already exists, otherwise creates it

        return {"bandfilling_correction": bf_corr}

    def perform_bandfill_corr(self, eigenvalues, kpoint_weights, potalign, vbm, cbm, soc_calc=False):
        """
        This calculates the band filling correction based on excess of
        electrons/holes in CB/VB...

        Note that the total free holes and electrons may also be used for a "shallow donor/acceptor"
               correction with specified band shifts:
                +num_elec_cbm * Delta E_CBM (or -num_hole_vbm * Delta E_VBM)
        """
        bf_corr = 0.0

        self.metadata["potalign"] = potalign
        self.metadata["num_hole_vbm"] = 0.0
        self.metadata["num_elec_cbm"] = 0.0

        core_occupation_value = list(eigenvalues.values())[0][0][0][
            1
        ]  # get occupation of a core eigenvalue
        if len(eigenvalues.keys()) == 1:
            # needed because occupation of non-spin calcs is sometimes still 1... should be 2
            spinfctr = 2.0 if core_occupation_value == 1.0 and not soc_calc else 1.0
        elif len(eigenvalues.keys()) == 2:
            spinfctr = 1.0
        else:
            raise ValueError("Eigenvalue keys greater than 2")

        # for tracking mid gap states...
        shifted_cbm = cbm - potalign  # shift cbm with potential alignment
        shifted_vbm = vbm - potalign  # shift vbm with potential alignment

        for spinset in eigenvalues.values():
            for kptset, weight in zip(spinset, kpoint_weights):
                for eig, occu in kptset:  # eig is eigenvalue and occu is occupation
                    if occu and (eig > shifted_cbm - self.resolution):  # donor MB correction
                        bf_corr += (
                            weight * spinfctr * occu * (eig - shifted_cbm)
                        )  # "move the electrons down"
                        self.metadata["num_elec_cbm"] += weight * spinfctr * occu
                    elif (occu != core_occupation_value) and (
                        eig <= shifted_vbm + self.resolution
                    ):  # acceptor MB correction
                        bf_corr += (
                            weight * spinfctr * (core_occupation_value - occu) * (shifted_vbm - eig)
                        )  # "move the holes up"
                        self.metadata["num_hole_vbm"] += weight * spinfctr * (core_occupation_value - occu)

        bf_corr *= -1  # need to take negative of this shift for energetic correction

        return bf_corr


def generate_R_and_G_vecs(gamma, prec_set, lattice, epsilon):
    """
    This returns a set of real and reciprocal lattice vectors (and real/recip
    summation values) based on a list of precision values (prec_set).

    gamma (float): Ewald parameter
    prec_set (list or number): for prec values to consider (20, 25, 30 are sensible numbers)
    lattice: Lattice object of supercell in question
    """
    if not isinstance(prec_set, list):
        prec_set = [prec_set]

    [a1, a2, a3] = lattice.matrix  # Angstrom
    volume = lattice.volume
    [b1, b2, b3] = lattice.reciprocal_lattice.matrix  # 1/ Angstrom
    invepsilon = np.linalg.inv(epsilon)
    rd_epsilon = np.sqrt(np.linalg.det(epsilon))

    # generate reciprocal vector set (for each prec_set)
    recip_set = [[] for _ in prec_set]
    recip_summation_values = [0.0 for _ in prec_set]
    recip_cut_set = [(2 * gamma * prec) for prec in prec_set]

    i_max = int(math.ceil(max(recip_cut_set) / np.linalg.norm(b1)))
    j_max = int(math.ceil(max(recip_cut_set) / np.linalg.norm(b2)))
    k_max = int(math.ceil(max(recip_cut_set) / np.linalg.norm(b3)))
    for i in np.arange(-i_max, i_max + 1):
        for j in np.arange(-j_max, j_max + 1):
            for k in np.arange(-k_max, k_max + 1):
                if not i and not j and not k:
                    continue
                gvec = i * b1 + j * b2 + k * b3
                normgvec = np.linalg.norm(gvec)
                for recip_cut_ind, recip_cut in enumerate(recip_cut_set):
                    if normgvec <= recip_cut:
                        recip_set[recip_cut_ind].append(gvec)

                        Gdotdiel = np.dot(gvec, np.dot(epsilon, gvec))
                        summand = math.exp(-Gdotdiel / (4 * (gamma**2))) / Gdotdiel
                        recip_summation_values[recip_cut_ind] += summand

    recip_summation_values = np.array(recip_summation_values)
    recip_summation_values /= volume

    # generate real vector set (for each prec_set)
    real_set = [[] for prec in prec_set]
    real_summation_values = [0.0 for prec in prec_set]
    real_cut_set = [(prec / gamma) for prec in prec_set]

    i_max = int(math.ceil(max(real_cut_set) / np.linalg.norm(a1)))
    j_max = int(math.ceil(max(real_cut_set) / np.linalg.norm(a2)))
    k_max = int(math.ceil(max(real_cut_set) / np.linalg.norm(a3)))
    for i in np.arange(-i_max, i_max + 1):
        for j in np.arange(-j_max, j_max + 1):
            for k in np.arange(-k_max, k_max + 1):
                rvec = i * a1 + j * a2 + k * a3
                normrvec = np.linalg.norm(rvec)
                for real_cut_ind, real_cut in enumerate(real_cut_set):
                    if normrvec <= real_cut:
                        real_set[real_cut_ind].append(rvec)
                        if normrvec > 1e-8:
                            sqrt_loc_res = np.sqrt(np.dot(rvec, np.dot(invepsilon, rvec)))
                            nmr = math.erfc(gamma * sqrt_loc_res)
                            real_summation_values[real_cut_ind] += nmr / sqrt_loc_res

    real_summation_values = np.array(real_summation_values)
    real_summation_values /= 4 * np.pi * rd_epsilon

    return recip_set, recip_summation_values, real_set, real_summation_values


def tune_for_gamma(lattice, epsilon):
    """
    This tunes the gamma parameter for Kumagai anisotropic Ewald calculation.

    Method is to find a gamma parameter which generates a similar number of
    reciprocal and real lattice vectors, given the suggested cut off radii by
    Kumagai and Oba.
    """
    logger.debug("Converging for Ewald parameter...")
    prec = 25  # a reasonable precision to tune gamma for

    gamma = (2 * np.average(lattice.abc)) ** (-1 / 2.0)
    recip_set, _, real_set, _ = generate_R_and_G_vecs(gamma, prec, lattice, epsilon)
    recip_set = recip_set[0]
    real_set = real_set[0]

    logger.debug(
        f"First approach with gamma ={gamma}\nProduced {len(real_set)} real vecs and {len(recip_set)} "
        f"recip vecs."
    )

    while float(len(real_set)) / len(recip_set) > 1.05 or float(len(recip_set)) / len(real_set) > 1.05:
        gamma *= (float(len(real_set)) / float(len(recip_set))) ** 0.17
        logger.debug(f"\tNot converged...Try modifying gamma to {gamma}.")
        recip_set, _, real_set, _ = generate_R_and_G_vecs(gamma, prec, lattice, epsilon)
        recip_set = recip_set[0]
        real_set = real_set[0]
        logger.debug(f"Now have {len(real_set)} real vecs and {len(recip_set)} recip vecs.")

    logger.debug(f"Converged with gamma = {gamma}")

    return gamma
