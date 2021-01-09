# coding: utf-8

from __future__ import division

__author__ = "Nils E. R. Zimmermann, Danny Broberg"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Nils E. R. Zimmermann"
__email__ = "nils.e.r.zimmermann@gmail.com"
__status__ = "Development"
__date__ = "October 9, 2017"

import unittest
from dope.pycdt.utils.units import k_to_eV, eV_to_k

class EnergyFunctionsTest(unittest.TestCase):
    def test_k_to_eV(self):
        g = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(k_to_eV(g), 0.5333804)

    def test_eV_to_k(self):
        self.assertAlmostEqual(eV_to_k(1.), 0.9681404248678961)


import unittest
if __name__ == '__main__':
    unittest.main()
