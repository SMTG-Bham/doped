# coding: utf-8

from __future__ import division

__status__ = "Development"

import os
import unittest

import numpy as np
from pymatgen.core.structure import Structure

from doped.pycdt.corrections.utils import *

bs_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "test_files", "POSCAR_Ga4As4")
)
ds_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "test_files", "POSCAR_Ga3As4")
)


class StructureFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.bs = Structure.from_file(bs_path)
        self.ds = Structure.from_file(ds_path)
        self.a = self.bs.lattice.matrix[0]
        self.b = self.bs.lattice.matrix[1]
        self.c = self.bs.lattice.matrix[2]

    def test_cleanlat(self):
        self.assertAlmostEqual(
            cleanlat(self.bs.lattice.matrix), [5.750183, 5.750183, 5.750183]
        )

    def test_genrecip(self):
        brecip = map(
            np.array,
            [
                [-1.0926931033637688, 0.0, 0.0],
                [0.0, -1.0926931033637688, 0.0],
                [0.0, 0.0, -1.0926931033637688],
                [0.0, 0.0, 1.0926931033637688],
                [0.0, 1.0926931033637688, 0.0],
                [1.0926931033637688, 0.0, 0.0],
            ],
        )
        recip = genrecip(self.a, self.b, self.c, 1.3)
        for vec in brecip:
            self.assertTrue(np.isclose(next(recip), vec).all())

    def test_generate_reciprocal_vectors_squared(self):
        brecip = [1.1939782181387439 for i in range(6)]
        self.assertAlmostEqual(
            list(generate_reciprocal_vectors_squared(self.a, self.b, self.c, 1.3)),
            brecip,
        )

    def test_closestsites(self):
        pos = [0.0000, 2.8751, 2.8751]
        bsite, dsite = closestsites(self.bs, self.ds, pos)
        self.assertAlmostEqual(list(bsite[0].coords), list(self.bs.sites[1].coords))
        self.assertAlmostEqual(list(dsite[0].coords), list(self.ds.sites[0].coords))

    def test_find_defect_pos(self):
        """
        TODO: might be good to include more defect poscars to test all
        types of defect finding.
        """
        bdefect, ddefect = find_defect_pos(self.bs, self.ds)
        self.assertAlmostEqual(list(bdefect), [0, 0, 0])
        self.assertIsNone(ddefect)


import unittest

if __name__ == "__main__":
    unittest.main()
