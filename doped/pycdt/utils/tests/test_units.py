# coding: utf-8

from __future__ import division

__status__ = "Development"

import unittest

from doped.pycdt.utils.units import eV_to_k, k_to_eV


class EnergyFunctionsTest(unittest.TestCase):
    def test_k_to_eV(self):
        g = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(k_to_eV(g), 0.5333804)

    def test_eV_to_k(self):
        self.assertAlmostEqual(eV_to_k(1.0), 0.9681404248678961)


import unittest

if __name__ == "__main__":
    unittest.main()
