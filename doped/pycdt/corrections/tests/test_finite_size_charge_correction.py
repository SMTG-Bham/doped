# coding: utf-8

from __future__ import division

__status__ = "Development"

import unittest

import numpy as np
from pymatgen.analysis.defects.core import DefectEntry, Vacancy
from pymatgen.core.sites import PeriodicSite
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.corrections.finite_size_charge_correction import (
    get_correction_freysoldt,
    get_correction_kumagai,
)


class FiniteSizeChargeCorrectionTest(PymatgenTest):
    """
    Test functions for getting freysoldt and kumagai corrections
    """

    def setUp(self):
        self.epsilon = 15.0

        struc = PymatgenTest.get_structure("VO2")
        struc.make_supercell(3)
        vac = Vacancy(struc, struc.sites[0], charge=-3)

        # load neccessary parameters for defect_entry to make use
        # of Freysoldt and Kumagai corrections
        p = {}
        ids = vac.generate_defect_structure(1)
        abc = struc.lattice.abc
        axisdata = [np.arange(0.0, lattval, 0.2) for lattval in abc]
        bldata = [
            np.array([1.0 for u in np.arange(0.0, lattval, 0.2)]) for lattval in abc
        ]
        dldata = [
            np.array(
                [
                    (-1 - np.cos(2 * np.pi * u / lattval))
                    for u in np.arange(0.0, lattval, 0.2)
                ]
            )
            for lattval in abc
        ]
        p.update(
            {
                "axis_grid": axisdata,
                "bulk_planar_averages": bldata,
                "defect_planar_averages": dldata,
                "initial_defect_structure": ids,
                "defect_frac_sc_coords": struc.sites[0].frac_coords,
                "bulk_sc_structure": struc,
            }
        )

        bulk_atomic_site_averages, defect_atomic_site_averages = [], []
        defect_site_with_sc_lattice = PeriodicSite(
            struc.sites[0].specie,
            struc.sites[0].coords,
            struc.lattice,
            coords_are_cartesian=True,
        )
        max_dist = 9.6
        pert_amnt = 1.0
        for site_ind, site in enumerate(struc.sites):
            if site.specie.symbol == "O":
                Oval = -30.6825
                bulk_atomic_site_averages.append(Oval)
                if site_ind:
                    dist_to_defect = site.distance_and_image(
                        defect_site_with_sc_lattice
                    )[0]
                    defect_site_val = (
                        Oval
                        - 0.3
                        + pert_amnt * ((max_dist - dist_to_defect) / max_dist) ** 2
                    )
                    defect_atomic_site_averages.append(defect_site_val)
            else:
                Vval = -51.6833
                bulk_atomic_site_averages.append(Vval)
                if site_ind:
                    dist_to_defect = site.distance_and_image(
                        defect_site_with_sc_lattice
                    )[0]
                    defect_site_val = (
                        Vval
                        - 0.3
                        + pert_amnt * ((max_dist - dist_to_defect) / max_dist) ** 2
                    )
                    defect_atomic_site_averages.append(defect_site_val)

        site_matching_indices = [
            [ind, ind - 1] for ind in range(len(struc.sites)) if ind != 0
        ]

        p.update(
            {
                "bulk_atomic_site_averages": bulk_atomic_site_averages,
                "defect_atomic_site_averages": defect_atomic_site_averages,
                "site_matching_indices": site_matching_indices,
            }
        )
        self.defect_entry = DefectEntry(vac, 0.0, parameters=p)

    def test_get_correction_freysoldt(self):
        freyout = get_correction_freysoldt(
            self.defect_entry, self.epsilon, title=None, partflag="All", axis=None
        )
        self.assertEqual(freyout, 5.445950368792991)

        freyout = get_correction_freysoldt(
            self.defect_entry, self.epsilon, title=None, partflag="AllSplit", axis=None
        )
        self.assertEqual(freyout[0], 0.975893)
        self.assertEqual(freyout[1], 4.4700573687929905)
        self.assertEqual(freyout[2], 5.445950368792991)

    def test_get_correction_kumagai(self):
        kumagaiout = get_correction_kumagai(
            self.defect_entry, self.epsilon, title=None, partflag="AllSplit"
        )
        self.assertEqual(kumagaiout[0], 0.9763991294314076)
        self.assertEqual(kumagaiout[1], 0.2579750033409367)
        self.assertEqual(kumagaiout[2], 1.2343741327723443)


if __name__ == "__main__":
    unittest.main()
