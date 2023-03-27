# coding: utf-8

from __future__ import division

__status__ = "Development"

import os

from pymatgen.analysis.defects.core import DefectEntry, Vacancy
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram
from pymatgen.core import Element
from pymatgen.core.structure import Lattice, PeriodicSite, Structure

# from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.utils.plotter import DefectPlotter


class DefectPlotterTest(PymatgenTest):
    def setUp(self):
        l = Lattice([[3.52, 0.0, 2.033], [1.174, 3.32, 2.033], [0.0, 0.0, 4.066]])
        s_bulk = Structure(
            l, ["Ga", "As"], [[0.0000, 0.0000, 0.0000], [0.2500, 0.2500, 0.2500]]
        )
        defect_site = PeriodicSite("As", [0.25, 0.25, 0.25], l)
        defect = Vacancy(s_bulk, defect_site, charge=1.0)
        defect_entry = DefectEntry(defect, 0.0)

        entries = [defect_entry]
        vbm = 0.2
        band_gap = 1.0
        dpd = DefectPhaseDiagram(entries, vbm, band_gap)
        self.dp = DefectPlotter(dpd)

    def test_get_plot_form_energy(self):
        mu_elts = {Element("As"): 0, Element("Ga"): 0}
        self.dp.get_plot_form_energy(mu_elts).savefig("test.pdf")
        self.assertTrue(os.path.exists("test.pdf"))
        os.system("rm test.pdf")

    # def test_plot_conc_temp(self):
    #     self.dp.plot_conc_temp().savefig('test.pdf')
    #     self.assertTrue(os.path.exists('test.pdf'))
    #     os.system('rm test.pdf')
    #
    # def test_plot_carriers_ef(self):
    #     self.dp.plot_carriers_ef().savefig('test.pdf')
    #     self.assertTrue(os.path.exists('test.pdf'))
    #     os.system('rm test.pdf')

    # def tearDown(self):
    #     self.da


import unittest

if __name__ == "__main__":
    unittest.main()
