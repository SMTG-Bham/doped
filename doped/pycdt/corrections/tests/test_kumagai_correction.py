# coding: utf-8

from __future__ import division

__status__ = "Development"

import os
import unittest

import numpy as np
from pymatgen.io.vasp.outputs import Locpot
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.corrections.kumagai_correction import *

# Paths to files we are testing on
bl_path = os.path.abspath(os.path.join(
    __file__, '..', '..', '..', '..', 'test_files', 'bLOCPOT.gz'))
dl_path = os.path.abspath(os.path.join(
    __file__, '..', '..', '..', '..', 'test_files', 'dLOCPOT.gz'))
kad_path = os.path.abspath(os.path.join(
    __file__, '..', '..', '..', '..', 'test_files', 'testKumagaiData.json'))

class KumagaiBulkInitANDCorrectionTest(PymatgenTest):
    #TODO: add test for outcar Kumagai method...
    def setUp(self):
        self.bl = Locpot.from_file(bl_path)
        self.dl = Locpot.from_file(dl_path)
        self.bs = self.bl.structure
        self.ds = self.dl.structure
        self.kbi = KumagaiBulkInit(self.bs, self.bl.dim, 15,
                                   optgamma=3.49423226983)
        self.kc = KumagaiCorrection(15, -3, 3.49423226983, self.kbi.g_sum,
                                    self.bs, self.ds, bulk_locpot=self.bl,
                                    defect_locpot=self.dl)

    def test_find_optimal_gamma(self):
        self.assertEqual(self.kbi.find_optimal_gamma(), 3.4942322698305639)

    def test_reciprocal_sum(self):
        self.assertEqual(self.kbi.g_sum.size, 884736)
        self.assertAlmostEqual(self.kbi.g_sum[0][0][0], 0.050661706751775192)

    def test_pc(self):
        self.assertAlmostEqual(self.kc.pc(), 2.1315841582145407)

    def test_potalign(self):
        self.assertAlmostEqual(self.kc.potalign(), 2.1091426308966001)

    def test_correction(self):
        self.assertAlmostEqual(self.kc.correction(), 4.24073)

    def test_plot(self):
        tmpforplot = {'C': {'r': [1, 2, 3], 'Vqb': [0.1, 0.2, 0.3],
                            'Vpc': [-0.05, -0.15, -0.25]},
                      'EXTRA': {'wsrad': 1, 'potalign': 0.05,
                                'lengths': (3, 3, 3)}}
        KumagaiCorrection.plot(tmpforplot, 'TMP')
        self.assertTrue(os.path.exists('TMP_kumagaisiteavgPlot.pdf'))
        os.system('rm TMP_kumagaisiteavgPlot.pdf')

    def test_plot_from_datfile(self):
        KumagaiCorrection.plot_from_datfile(name=kad_path, title='TMP')
        self.assertTrue(os.path.exists('TMP_kumagaisiteavgPlot.pdf'))
        os.system('rm TMP_kumagaisiteavgPlot.pdf')

    #NOTE there are here because g_sum is pre computed for it
    def test_get_sum_at_r(self):
        val = get_g_sum_at_r(self.kbi.g_sum, self.bs, self.bl.dim,
                             [0.1, 0.1, 0.1])
        self.assertAlmostEqual(val, 0.04795055159361078)

    def test_anisotropic_madelung_potential(self):
        val = anisotropic_madelung_potential(
                self.bs, self.bl.dim, self.kbi.g_sum, [0.1, 0.1, 0.1],
                [[15, 0.1, -0.1], [0.1, 13, 0], [-0.1, 0, 20]], -3,
                self.kbi.gamma, self.kbi.tolerance)
        self.assertAlmostEqual(val, -4.2923511216202419)

    def test_anisotropic_pc_energy(self):
        val = anisotropic_pc_energy(
                self.bs, self.kbi.g_sum,
                [[15, 0.1, -0.1], [0.1, 13, 0], [-0.1, 0, 20]], -3,
                self.kbi.gamma, self.kbi.tolerance)
        self.assertAlmostEqual(val, 1.5523329679084736)


class KumagaiSetupFunctionsTest(PymatgenTest):
    def setUp(self):
        self.bl = Locpot.from_file(bl_path)
        self.dl = Locpot.from_file(dl_path)
        self.bs = self.bl.structure
        self.ds = self.dl.structure

    def test_kumagai_init(self):
        angset, bohrset, vol, determ, invdiel = kumagai_init(
                self.bs, [[15, 0.1, -0.1], [0.1, 13, 0], [-0.1, 0, 20]])
        newangset = [list(row) for row in angset]
        newbohrset = [list(row) for row in bohrset]
        newinvdiel = [list(row) for row in invdiel]
        self.assertEqual(newangset, [[5.750183, 0, 0], [0, 5.750183, 0],
                                  [0, 0, 5.750183]])
        self.assertEqual(newbohrset, [[10.866120815099999, 0, 0],
                                      [0, 10.866120815099999, 0],
                                      [0, 0, 10.866120815099999]])
        self.assertAlmostEqual(vol, 1282.9909362724345)
        self.assertAlmostEqual(determ, 3899.6699999999969)
        tmpinvdiel = [[0.066672308169665628, -0.00051286390899742801, 0.00033336154084832821],
                      [-0.00051286390899742801, 0.076927022030069209, -0.0000025643195449871406],
                      [0.00033336154084832826, -0.0000025643195449871406, 0.050001666807704244]]
        np.testing.assert_array_almost_equal(newinvdiel, tmpinvdiel)

    def test_real_sum(self):
        a = self.bs.lattice.matrix[0]
        b = self.bs.lattice.matrix[1]
        c = self.bs.lattice.matrix[2]
        tmpdiel = [[15, 0.1, -0.1], [0.1, 13, 0], [-0.1, 0, 20]]
        val = real_sum(a, b, c, np.array([0.1, 0.1, 0.1]), -1, tmpdiel, 3, 1)
        self.assertAlmostEqual(val, -0.0049704211394050414)

    def test_getgridind(self):
        triv_ans = getgridind(self.bs, (96,96,96), [0,0,0])
        self.assertArrayEqual(triv_ans, [0,0,0])
        diff_ans = getgridind(self.bs, (96,96,96), [0.2,0.3,0.4])
        self.assertArrayEqual(diff_ans, [19,29,38])
        #test atomic site averaging approach
        asa_ans = getgridind(self.bs, (96,96,96), [0,0,0], gridavg=0.08)
        correct_avg = [(95, 0, 0), (0, 95, 0), (0, 0, 95), (0, 0, 0),
                       (0, 0, 1), (0, 1, 0), (1, 0, 0)]
        self.assertArrayEqual(asa_ans, correct_avg)

    def test_disttrans(self):
        nodefpos = disttrans( self.bs, self.ds)
        self.assertArrayEqual(list(nodefpos.keys()), [1, 2, 3, 4, 5, 6, 7])
        self.assertArrayEqual(nodefpos[3]['cart_reldef'], [ 2.8750915, 2.8750915, 0.])
        self.assertEqual(nodefpos[3]['bulk_site_index'], 3)
        self.assertEqual(nodefpos[3]['dist'], 4.0659933923636054)
        self.assertEqual(nodefpos[3]['def_site_index'], 2)
        self.assertArrayEqual(nodefpos[3]['cart'], [ 2.8750915, 2.8750915, 0.])
        self.assertArrayEqual(nodefpos[3]['siteobj'][0], [ 2.8750915, 2.8750915, 0.])
        self.assertArrayEqual(nodefpos[3]['siteobj'][1], [ 0.5, 0.5, 0.])
        self.assertEqual(nodefpos[3]['siteobj'][2], 'Ga')

    def test_wigner_seitz_radius(self):
        self.assertAlmostEqual(wigner_seitz_radius(self.bs), 2.8750914999999999)

    def test_read_ES_avg_fromlocpot(self):
        potdict = read_ES_avg_fromlocpot(self.bl)
        correct_potential = [-12.275644645892712, -12.275648494671156, -12.275640119791278,
                             -12.275643801037788, -24.350725695796118, -24.350725266618685,
                             -24.350723873595474, -24.350723145563087]
        self.assertArrayEqual(potdict['potential'], correct_potential)
        self.assertArrayEqual(potdict['ngxf_dims'], (96, 96, 96))


import unittest

if __name__ == '__main__':
    unittest.main()
