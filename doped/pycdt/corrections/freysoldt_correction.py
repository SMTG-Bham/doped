"""
This module is Freysoldt correction for isotropic systems
1) Freysoldt correction for isotropic systems.
includes
   a) PC energy
   b) potential alignment by planar averaging.
If you use the corrections implemented in this module, cite
   Freysoldt, Neugebauer, and Van de Walle, Phys. Rev. Lett. 102, 016402 (2009)
   [Optionally Phys. Status Solidi B. 248, 1067-1076 (2011) ]
   in addition to the pycdt paper
"""
__author__ = 'Danny Broberg, Bharat Medasani'
__email__ = 'dbroberg@gmail.com, mbkumar@gmail.com'

import sys
import math
import logging

import numpy as np

from pymatgen.io.vasp.outputs import Locpot
from pymatgen.core.structure import Structure

from doped.pycdt.corrections.utils import *
from doped.pycdt.utils.units import hart_to_ev

norm = np.linalg.norm

import warnings

warnings.warn("Replacing PyCDT usage of Freysoldt base classes with calls to "
              "corresponding objects in pymatgen.analysis.defects.corrections\n"
              "Will remove QModel with Version 2.5 of PyCDT.",
              DeprecationWarning)
class QModel():
    """
    Model for the defect charge distribution.
    A combination of exponential tail and gaussian distribution is used
    (see Freysoldt (2011), DOI: 10.1002/pssb.201046289 )
    q_model(r) = q [x exp(-r/gamma) + (1-x) exp(-r^2/beta^2)]
            without normalization constants
    By default, gaussian distribution with 1 Bohr width is assumed.
    If defect charge is more delocalized, exponential tail is suggested.
    """
    def __init__(self, beta=1.0, expnorm=0.0, gamma=1.0):
        """
        Args:
            beta: Gaussian decay constant. Default value is 1 Bohr.
                  When delocalized (eg. diamond), 2 Bohr is more appropriate.
            expnorm: Weight for the exponential tail in the range of [0-1].
                     Default is 0.0 indicating no tail .
                     For delocalized charges ideal value is around 0.54-0.6.
            gamma: Exponential decay constant
        """
        self.beta2 = beta * beta
        self.x = expnorm
        self.gamma2 = gamma * gamma
        if expnorm and not gamma:
            raise ValueError("Please supply exponential decay constant.")

    def rho_rec(self, g2):
        """
        Reciprocal space model charge value
        for input squared reciprocal vector.
        Args:
            g2: Square of reciprocal vector

        Returns:
            Charge density at the reciprocal vector magnitude
        """
        return (self.x / np.sqrt(1+self.gamma2*g2)
                + (1-self.x) * np.exp(-0.25*self.beta2*g2))

    def rho_rec_limit0(self):
        """
        Reciprocal space model charge value
        close to reciprocal vector 0 .
        rho_rec(g->0) -> 1 + rho_rec_limit0 * g^2
        """
        return -2*self.gamma2*self.x - 0.25*self.beta2*(1-self.x)


warnings.warn("Replacing PyCDT usage of Freysoldt base classes with calls to "
              "corresponding objects in pymatgen.analysis.defects.corrections\n"
              "All correction plotting functionalities exist within pymatgen v2019.5.1."
              "Version 2.5 of PyCDT will remove pycdt.corrections.freysoldt_correction.FreysoldtCorrPlotter.",
              DeprecationWarning)
class FreysoldtCorrPlotter(object):
    def __init__(self, x, v_R, dft_diff, final_shift, check):
        self.x = x
        self.v_R = v_R
        self.dft_diff = dft_diff
        self.final_shift = final_shift
        self.check = check

    def plot(self, title='default'):
        """
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure()
        plt.clf()
        plt.plot(self.x, self.v_R, c="green", zorder=1,
                 label="long range from model")
        plt.plot(self.x, self.dft_diff, c="red", label="DFT locpot diff")
        plt.plot(self.x, self.final_shift, c="blue", 
                 label="short range (aligned)")
        tmpx = [self.x[i] for i in range(self.check[0], self.check[1])]
        plt.fill_between(tmpx, -100, 100, facecolor='red', alpha=0.15, 
                         label='sampling region')

        plt.xlim(round(self.x[0]), round(self.x[-1]))
        ymin = min(min(self.v_R), min(self.dft_diff), min(self.final_shift))
        ymax = max(max(self.v_R), max(self.dft_diff), max(self.final_shift))
        plt.ylim(-0.2+ymin, 0.2+ymax)
        plt.xlabel('distance along axis ' + str(1) + ' ($\AA$)', fontsize=20)
        plt.ylabel('Potential (V)', fontsize=20)
        plt.legend(loc=9)
        plt.axhline(y=0, linewidth=0.2, color='black')
        plt.title(str(title) + ' defect potential')
        plt.xlim(0, max(self.x))

        plt.savefig(str(title) + 'FreyplnravgPlot.pdf')

    def to_datafile(self, file_name='FreyAxisData'):

        np.savez(file_name, x=self.x, v_R=self.v_R, 
                 dft_diff=self.dft_diff, #defavg-pureavg,
                 final_shift=self.final_shift, #finalshift,
                 check_range=self.check) #np.array([mid-checkdis, mid+checkdis]))

    @classmethod
    def plot_from_datfile(cls, file_name='FreyAxisData.npz', title='default'):
        """
        Takes data file called 'name' and does plotting.
        Good for later plotting of locpot data after running run_correction()
        """
        with open(file_name) as f:
            plotvals = np.load(f)

            x = plotvals['x']
            v_R = plotvals['v_R']
            dft_diff = plotvals['dft_diff']
            final_shift = plotvals['final_shift']
            check = plotvals['check_range']

            plotter = cls(x, v_R, dft_diff, final_shift, check)
            plotter.plot(title)


warnings.warn("Replacing PyCDT usage of Freysoldt base classes with calls to "
              "corresponding objects in pymatgen.analysis.defects.corrections\n"
              "Will remove pycdt.corrections.freysoldt_correction.FreysoldtCorrection "
              "with Version 2.5 of PyCDT. (Corrections will all come from pymatgen for "
              "longer term maintenance).",
              DeprecationWarning)
class FreysoldtCorrection(object):
    def __init__(self, axis, dielectricconst, pure_locpot_path,
                 defect_locpot_path, q, energy_cutoff=520,
                 madetol=0.0001, q_model=None, **kw):
        """
        Args:
            axis:
                axis to do Freysoldt averaging over (zero-defined).
            dielectric_tensor:
                Macroscopic dielectric tensor. Include ionic also if
                defect is relaxed, otherwise use ion clamped.
                Can be a matrix array or scalar.
            pure_locpot_path:
                Bulk Locpot file path or locpot object
            defect_locpot_path:
                Defect Locpot file path or locpot object
            q (int or float):
                Charge associated with the defect (not of the homogeneous
                background). Typically integer
            energy_cutoff:
                Energy for plane wave cutoff (in eV).
                If not given, Materials Project default 520 eV is used.
            madetol (float):
                Tolerance for convergence of energy terms (in eV)
            q_model (QModel object):
                User defined charge for correction.
                If not given, highly localized charge is assumed.
            keywords:
                1) defect_position: Defect position as a pymatgen Site object in the bulk supercell structure
                    NOTE: this is optional but recommended, if not provided then analysis is done to find
                    the defect position; this analysis has been rigorously tested, but has broken in an example with
                    severe long range relaxation
                    (at which point you probably should not be including the defect in your analysis...)
        """
        self._axis = axis
        if isinstance(dielectricconst, int) or \
                isinstance(dielectricconst, float):
            self._dielectricconst = float(dielectricconst)
        else:
            self._dielectricconst = float(np.mean(np.diag(dielectricconst)))
        self._purelocpot = pure_locpot_path
        self._deflocpot = defect_locpot_path
        self._madetol = madetol
        self._q = q
        self._encut = energy_cutoff
        if 'defect_position' in kw:
            self._defpos = kw['defect_position']
        else:
            self._defpos = None #code will determine defect position in defect cell
        if not q_model:
            self._q_model = QModel()

    def correction(self, title=None, partflag='All'):
        """
        Args:
            title: set if you want to plot the planar averaged potential
            partflag: four options
                'pc' for just point charge correction, or
               'potalign' for just potalign correction, or
               'All' for both, or
               'AllSplit' for individual parts split up (form [PC,potterm,full])
        """
        logger = logging.getLogger(__name__)
        logger.info('This is Freysoldt Correction.')
        if not self._q:
            if partflag == 'AllSplit':
                return [0.0,0.0,0.0]
            else:
                return 0.0

        if not type(self._purelocpot) is Locpot:
            logger.debug('Load bulk locpot')
            self._purelocpot = Locpot.from_file(self._purelocpot)

        logger.debug('\nRun PC energy')
        if partflag != 'potalign':
            energy_pc = self.pc()
            logger.debug('PC calc done, correction = %f', round(energy_pc, 4))
            logger.debug('Now run potenttial alignment script')

        if partflag != 'pc':
            if not type(self._deflocpot) is Locpot:
                logger.debug('Load defect locpot')
                self._deflocpot = Locpot.from_file(self._deflocpot)
            potalign = self.potalign(title=title)

        logger.info('\n\nFreysoldt Correction details:')
        if partflag != 'potalign':
            logger.info('PCenergy (E_lat) = %f', round(energy_pc, 5))
        if partflag != 'pc':
            logger.info('potential alignment (-q*delta V) = %f',
                         round(potalign, 5))
        if partflag in ['All','AllSplit']:
            logger.info('TOTAL Freysoldt correction = %f',
                         round(energy_pc + potalign, 5))

        if partflag == 'pc':
            return round(energy_pc, 5)
        elif partflag == 'potalign':
            return round(potalign, 5)
        elif partflag == 'All':
            return round(energy_pc + potalign, 5)
        else:
            return map(lambda x: round(x, 5), 
                       [energy_pc, potalign, energy_pc + potalign])

    def pc(self, struct=None):
        """
        Peform Electrostatic Correction
        note this ony needs structural info
        so struct input object speeds this calculation up
        equivalently fast if input Locpot is a locpot object
        """
        logger = logging.getLogger(__name__)
        if type(struct) is Structure:
            s1 = struct
        else:
            if not type(self._purelocpot) is Locpot:
                logging.info('load Pure locpot')
                self._purelocpot = Locpot.from_file(self._purelocpot)
            s1 = self._purelocpot.structure

        ap = s1.lattice.get_cartesian_coords(1)
        logger.info('Running Freysoldt 2011 PC calculation (should be '\
                     'equivalent to sxdefectalign)')
        logger.debug('defect lattice constants are (in angstroms)' \
                      + str(cleanlat(ap)))
        [a1, a2, a3] = ang_to_bohr * ap
        logging.debug( 'In atomic units, lat consts are (in bohr):' \
                      + str(cleanlat([a1, a2, a3])))
        vol = np.dot(a1, np.cross(a2, a3))  #vol in bohr^3

        #compute isolated energy
        step = 1e-4
        encut1 = 20  #converge to some smaller encut first [eV]
        flag = 0
        converge = []
        while (flag != 1):
            eiso = 1.
            gcut = eV_to_k(encut1)  #gcut is in units of 1/A
            g = step  #initalize
            while g < (gcut + step):
                #simpson integration
                eiso += 4*(self._q_model.rho_rec(g*g) ** 2)
                eiso += 2*(self._q_model.rho_rec((g+step) ** 2) ** 2)
                g += 2 * step
            eiso -= self._q_model.rho_rec(gcut ** 2) ** 2
            eiso *= (self._q ** 2) * step / (3 * round(np.pi, 6))
            converge.append(eiso)
            if len(converge) > 2:
                if abs(converge[-1] - converge[-2]) < self._madetol:
                    flag = 1
                elif encut1 > self._encut:
                    logger.error('Eiso did not converge before ' \
                                  + str(self._encut) + ' eV')
                    raise
            encut1 += 20
        eiso = converge[-1]
        logger.debug('Eisolated : %f, converged at encut: %d',
                      round(eiso, 5), encut1-20)

        #compute periodic energy;
        encut1 = 20  #converge to some smaller encut
        flag = 0
        converge = []
        while flag != 1:
            eper = 0.0
            for g2 in generate_reciprocal_vectors_squared(a1, a2, a3, encut1):
                eper += (self._q_model.rho_rec(g2) ** 2) / g2
            eper *= (self._q**2) *2* round(np.pi, 6) / vol
            eper += (self._q**2) *4* round(np.pi, 6) \
                    * self._q_model.rho_rec_limit0() / vol
            converge.append(eper)
            if len(converge) > 2:
                if abs(converge[-1] - converge[-2]) < self._madetol:
                    flag = 1
                elif encut1 > self._encut:
                    logger.error('Eper did not converge before %d eV',
                                  self._encut)
                    return
            encut1 += 20
        eper = converge[-1]

        logger.info('Eperiodic : %f hartree, converged at encut %d eV',
                     round(eper, 5), encut1-20)
        logger.info('difference (periodic-iso) is %f hartree',
                     round(eper - eiso, 6))
        logger.info( 'difference in (eV) is %f',
                     round((eper-eiso) * hart_to_ev, 4))

        PCfreycorr = round((eiso-eper) / self._dielectricconst * hart_to_ev, 6)
        logger.info('Defect Correction without alignment %f (eV): ', PCfreycorr)

        return PCfreycorr

    def potalign(self, title=None, widthsample=1.0, axis=None, output_sr=False):
        """
        For performing planar averaging potential alignment

        Accounts for defects in arbitrary positions
        title is for name of plot, if you dont want a plot then leave it as None
        widthsample is the width of the region in between defects where the potential alignment correction is averaged
        axis allows you to override the axis setting of class
                (good for quickly plotting multiple axes without having to reload Locpot)
        output_sr allows for output of the short range potential in the middle (sampled) region.
                (Good for delocalization analysis)
        """
        logger = logging.getLogger(__name__)
        if axis is None:
            axis = self._axis
        else:
            axis = axis

        if not type(self._purelocpot) is Locpot:
            logger.debug('load pure locpot object')
            self._purelocpot = Locpot.from_file(self._purelocpot)
        if not type(self._deflocpot) is Locpot:
            logger.debug('load defect locpot object')
            self._deflocpot = Locpot.from_file(self._deflocpot)

        #determine location of defects
        blksite, defsite = find_defect_pos(self._purelocpot.structure, 
                                           self._deflocpot.structure,
                                           defpos = self._defpos)
        if blksite is None and defsite is None:
            logger.error('Not able to determine defect site')
            return

        if blksite is None:
            logger.debug('Found defect to be Interstitial type at %s',
                          repr(defsite))
        elif defsite is None:
            logger.debug('Found defect to be Vacancy type at %s',
                          repr(blksite))
        else:
            logger.debug('Found defect to be antisite/substitution type at '
                          '%s in bulk, and %s in defect cell', 
                          repr(blksite), repr(defsite))

        #It is important to do planar averaging at same position, otherwise
        #you can get rigid shifts due to atomic changes at far away from defect
        #note these are cartesian co-ordinate sites...
        if defsite is None: #vacancies
            self._pos=blksite
        else: #all else, do w.r.t defect site
            self._pos=defsite

        x = np.array(self._purelocpot.get_axis_grid(axis))  #angstrom
        nx = len(x)
        logging.debug('run Freysoldt potential alignment method')

        #perform potential alignment part
        pureavg = self._purelocpot.get_average_along_axis(axis)  #eV
        defavg = self._deflocpot.get_average_along_axis(axis)  #eV

        #now shift these planar averages to have defect at origin
        blklat=self._purelocpot.structure.lattice
        axfracval=blklat.get_fractional_coords(self._pos)[axis]
        axbulkval=axfracval*blklat.abc[axis]
        if axbulkval<0:
            axbulkval += blklat.abc[axis]
        elif axbulkval > blklat.abc[axis]:
            axbulkval -= blklat.abc[axis]

        if axbulkval:
            for i in range(len(x)):
                if axbulkval < x[i]:
                    break
            rollind = len(x) - i
            pureavg = np.roll(pureavg,rollind)
            defavg = np.roll(defavg,rollind)

        #if not self._silence:
        logger.debug('calculating lr part along planar avg axis')
        latt = self._purelocpot.structure.lattice
        reci_latt = latt.reciprocal_lattice
        dg = reci_latt.abc[axis]
        dg /= ang_to_bohr #convert to bohr to do calculation in atomic units

        v_G = np.empty(len(x), np.dtype('c16'))
        epsilon = self._dielectricconst
        # q needs to be that of the back ground
        v_G[0] = 4*np.pi * -self._q / epsilon * self._q_model.rho_rec_limit0()
        for i in range(1,nx):
            if (2*i < nx):
                g = i * dg
            else:
                g = (i-nx) * dg
            g2 = g * g
            v_G[i] = 4*np.pi/(epsilon*g2) * -self._q * self._q_model.rho_rec(g2)
        if not (nx % 2):
            v_G[nx//2] = 0
        v_R = np.fft.fft(v_G)
        v_R_imag = np.imag(v_R)
        v_R /= (latt.volume * ang_to_bohr**3)
        v_R = np.real(v_R) * hart_to_ev

        max_imag_vr = v_R_imag.max()
        if abs(max_imag_vr) > self._madetol:
            logging.error('imaginary part found to be %s', repr(max_imag_vr))
            sys.exit()

        #now get correction and do plots
        short = (defavg - pureavg - v_R)
        checkdis = int((widthsample/2) / (x[1]-x[0]))
        mid = int(len(short) / 2)

        tmppot = [short[i] for i in range(mid-checkdis, mid+checkdis)]
        logger.debug('shifted defect position on axis (%s) to origin',
                      repr(axbulkval))
        logger.debug('means sampling region is (%f,%f)',
                      x[mid-checkdis], x[mid+checkdis])

        C = -np.mean(tmppot)
        logger.debug('C = %f', C)
        final_shift = [short[j] + C for j in range(len(v_R))]
        v_R = [elmnt - C for elmnt in v_R]

        logger.info('C value is averaged to be %f eV ', C)
        logger.info('Potentital alignment (-q*delta V):  %f (eV)', -self._q*C)

        if title: # TODO: Make title  optional and use a flag for plotting
            plotter = FreysoldtCorrPlotter(x, v_R, defavg-pureavg, final_shift,
                      np.array([mid-checkdis, mid+checkdis]))

            if title != 'written':
                plotter.plot(title=title)
            else:
                # TODO: Make this default fname more defect specific so it doesnt
                # over write previous defect data written
                fname = 'FreyAxisData' # Extension is npz
                plotter.to_datafile(fname)

        if output_sr:
            return ((-self._q * C), tmppot)  #pot align energy correction (eV)
        else:
            return (-self._q * C)  #pot align energy correction (eV)


