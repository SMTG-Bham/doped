# coding: utf-8

from __future__ import division

__status__ = "Development"

import os
import unittest

from doped.pycdt.corrections.ldau_correction import *

test_file_loc = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "test_files")
)


class LDAUCorrectionTest(unittest.TestCase):
    def setUp(self):
        exp_gap, ggau_gap, gga_gap = (3, 2, 1)
        self.corrector = LDAUCorrection(exp_gap, ggau_gap, gga_gap)

    def test_transition_correction_with_identical_levels(self):
        self.assertAlmostEqual(0, self.corrector.get_transition_correction(0, 0))
        self.assertAlmostEqual(0, self.corrector.get_transition_correction(0.1, 0.1))

    def test_transition_correction_with_greater_lda_transition(self):
        self.assertAlmostEqual(-0.1, self.corrector.get_transition_correction(0.1, 0.2))

    def test_transition_correction_with_greater_ldau_transition(self):
        self.assertAlmostEqual(0.1, self.corrector.get_transition_correction(0.2, 0.1))

    def test_transition_correction_with_lda_transition_greater_than_bg(self):
        self.assertAlmostEqual(-1.1, self.corrector.get_transition_correction(0.4, 1.5))


import unittest

if __name__ == "__main__":
    unittest.main()
