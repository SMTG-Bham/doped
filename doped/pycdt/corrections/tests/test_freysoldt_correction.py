# coding: utf-8

from __future__ import division

__author__ = "Danny Broberg, Bharat Medasani"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Danny Broberg, Bharat Medasani"
__email__ = "mbkumar@gmail.com"
__status__ = "Development"
__date__ = "June 23, 2016"

import os
import unittest

from pymatgen.io.vasp.outputs import Locpot
from doped.pycdt.corrections.freysoldt_correction import *

#Paths to files we are testing on
bl_path = os.path.abspath(os.path.join(
    __file__, '..', '..', '..', '..', 'test_files', 'bLOCPOT.gz'))
dl_path = os.path.abspath(os.path.join(
    __file__, '..', '..', '..', '..', 'test_files', 'dLOCPOT.gz'))
fad_path = os.path.abspath(os.path.join(
    __file__, '..', '..', '..', '..', 'test_files', 'testFreyAxisData.npz'))

class FreysoldtCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.fc = FreysoldtCorrection(0, 15, bl_path, dl_path, -3)

    def test_pc(self):
        self.assertAlmostEqual(self.fc.pc(), 2.131583)

    def test_potalign(self):
        self.assertAlmostEqual(self.fc.potalign(), 1.8596805562556484)

    def test_correction(self):
        self.assertAlmostEqual(self.fc.correction(), 3.99126)


class FreysoldtCorrPlotterTest(unittest.TestCase):
    def setUp(self):
        x = [0, 1, 2, 3]
        v_R = [1, 0.5, 0.5, 1]
        dft_diff = [0.5, 0.2, 0.2, 0.5]
        final_shift = [0.3, 0, 0, 0.3]
        check = [1, 2]
        self.fcp = FreysoldtCorrPlotter(x, v_R, dft_diff, final_shift, check)

    def test_plot(self):
        self.fcp.plot(title='TMPplot')
        self.assertTrue(os.path.exists('TMPplotFreyplnravgPlot.pdf'))
        os.system('rm TMPplotFreyplnravgPlot.pdf')

    def test_to_datafile(self):
        self.fcp.to_datafile(file_name='TMPFreyAxisData')
        self.assertTrue(os.path.exists('TMPFreyAxisData.npz'))
        os.system('rm TMPFreyAxisData.npz')

    def test_plot_from_datafile(self):
        self.fcp.plot_from_datfile(file_name=fad_path, title='TMPplot')
        self.assertTrue(os.path.exists('TMPplotFreyplnravgPlot.pdf'))
        os.system('rm TMPplotFreyplnravgPlot.pdf')


class QModelTest(unittest.TestCase):
    def setUp(self):
        self.qm = QModel()
        self.modqm = QModel(beta=2., expnorm=0.5, gamma=0.1)

    def test_rho_rec(self):
        self.assertEqual(self.qm.rho_rec(1.), 0.77880078307140488)
        self.assertEqual(self.modqm.rho_rec(1.), 0.6814583156907158)

    def test_rho_rec_limit0(self):
        self.assertEqual(self.qm.rho_rec_limit0(), -0.25)
        self.assertEqual(self.modqm.rho_rec_limit0(), -0.51)


import unittest
if __name__ == '__main__':
    unittest.main()
