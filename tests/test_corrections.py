"""
Test functions for getting freysoldt and kumagai corrections.

These tests are templated off those originally written by the PyCDT (
https://doi.org/10.1016/j.cpc.2018.01.004)
developers.
"""
import matplotlib as mpl
import numpy as np
from pymatgen.core.sites import PeriodicSite
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.util.testing import PymatgenTest

from doped.core import DefectEntry, Vacancy
from doped.corrections import get_freysoldt_correction, get_kumagai_correction

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally

# TODO: Test verbose options here


class FiniteSizeChargeCorrectionTest(PymatgenTest):
    """
    Test functions for getting freysoldt and kumagai corrections.
    """

    def setUp(self):
        self.dielectric = 15.0

        struct = PymatgenTest.get_structure("VO2")
        struct.make_supercell(3)
        vac = Vacancy(struct, struct.sites[0], charge=-3)

        ids = vac.defect_structure
        abc = struct.lattice.abc
        bldata = {
            i: np.array([1.0 for _ in np.arange(0.0, lattval, 0.2)]) for i, lattval in enumerate(abc)
        }
        dldata = {
            i: np.array([(-1 - np.cos(2 * np.pi * u / lattval)) for u in np.arange(0.0, lattval, 0.2)])
            for i, lattval in enumerate(abc)
        }
        # load necessary parameters for defect_entry to make use of Freysoldt and Kumagai corrections
        metadata = {
            "bulk_locpot_dict": bldata,
            "defect_locpot_dict": dldata,
            "defect_frac_sc_coords": struct.sites[0].frac_coords,
        }
        bulk_atomic_site_averages, defect_atomic_site_averages = [], []
        defect_site_with_sc_lattice = PeriodicSite(
            struct.sites[0].specie,
            struct.sites[0].coords,
            struct.lattice,
            coords_are_cartesian=True,
        )
        max_dist = 9.6
        pert_amnt = 1.0
        for site_ind, site in enumerate(struct.sites):
            if site.specie.symbol == "O":
                Oval = -30.6825
                bulk_atomic_site_averages.append(Oval)
                if site_ind:
                    dist_to_defect = site.distance_and_image(defect_site_with_sc_lattice)[0]
                    defect_site_val = (
                        Oval - 0.3 + pert_amnt * ((max_dist - dist_to_defect) / max_dist) ** 2
                    )
                    defect_atomic_site_averages.append(defect_site_val)
            else:
                Vval = -51.6833
                bulk_atomic_site_averages.append(Vval)
                if site_ind:
                    dist_to_defect = site.distance_and_image(defect_site_with_sc_lattice)[0]
                    defect_site_val = (
                        Vval - 0.3 + pert_amnt * ((max_dist - dist_to_defect) / max_dist) ** 2
                    )
                    defect_atomic_site_averages.append(defect_site_val)

        metadata.update(
            {
                "bulk_site_potentials": -1 * np.array(bulk_atomic_site_averages),
                "defect_site_potentials": -1 * np.array(defect_atomic_site_averages),
            }
        )

        self.defect_entry = DefectEntry(
            vac,
            charge_state=-3,
            sc_entry=ComputedStructureEntry(
                structure=ids,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            bulk_entry=ComputedStructureEntry(
                structure=struct,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            sc_defect_frac_coords=struct.sites[0].frac_coords,
            calculation_metadata=metadata,
        )

    def test_get_correction_freysoldt(self):
        fnv_corr_list = [get_freysoldt_correction(self.defect_entry, self.dielectric)]
        fnv_corr_list.append(get_freysoldt_correction(self.defect_entry, self.dielectric, axis=0))
        fnv_corr_list.append(get_freysoldt_correction(self.defect_entry, self.dielectric, axis=2))
        fnv_corr_list.append(
            get_freysoldt_correction(self.defect_entry, self.dielectric, axis=2, plot=True)[0]
        )

        for fnv_corr in fnv_corr_list:
            assert np.isclose(fnv_corr.correction_energy, 5.445950368792991)

    def test_get_correction_kumagai(self):
        efnv_corr_list = [get_kumagai_correction(self.defect_entry, self.dielectric)]
        efnv_corr_list.append(get_kumagai_correction(self.defect_entry, self.dielectric, plot=True)[0])
        for efnv_corr in efnv_corr_list:
            assert np.isclose(efnv_corr.correction_energy, 1.2651776920778381)
