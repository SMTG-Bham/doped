"""
Test functions for getting freysoldt and kumagai corrections.

This uses tests originally written by the PyCDT (
https://doi.org/10.1016/j.cpc.2018.01.004)
developers,
with some updates.
"""
import os
import tarfile
from shutil import copyfile

import matplotlib as mpl
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core.sites import PeriodicSite
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp import Locpot
from pymatgen.util.testing import PymatgenTest

from doped.core import DefectEntry, Vacancy
from doped.utils.corrections import (
    freysoldt_correction_from_paths,
    get_correction_freysoldt,
    get_correction_kumagai,
    kumagai_correction_from_paths,
)

test_files_dir = os.path.join(os.path.dirname(__file__), "data/charge_correction_tests")
mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


class FilePathCorrectionsTest(PymatgenTest):
    def test_freysoldt_and_kumagai(self):
        # create scratch directory with files....
        # having to do it all at once to minimize amount of time copying over to Scratch Directory
        with ScratchDir("."):
            # setup with fake Locpot object copied over
            copyfile(
                os.path.join(test_files_dir, "test_path_files.tar.gz"),
                "./test_path_files.tar.gz",
            )
            tar = tarfile.open("test_path_files.tar.gz")
            tar.extractall()
            tar.close()
            blocpot = Locpot.from_file(os.path.join(test_files_dir, "bulk_LOCPOT.gz"))
            blocpot.write_file("test_path_files/bulk/LOCPOT")
            dlocpot = Locpot.from_file(os.path.join(test_files_dir, "defect_LOCPOT.gz"))
            dlocpot.write_file("test_path_files/sub_1_Sb_on_Ga/charge_2/LOCPOT")

            fcc = freysoldt_correction_from_paths(
                "test_path_files/sub_1_Sb_on_Ga/charge_2/",
                "test_path_files/bulk/",
                18.12,
                2,
                plot=True,
                filename="test_freysoldt_correction",
            )
            assert np.isclose(fcc, -1.4954476868106865, rtol=1e-5)  # note this has been updated from the
            # pycdt version, because there they used a `transformation.json` that gave an
            # incorrect `initial_defect_structure` (corresponding to primitive rather than bulk)
            assert os.path.exists("test_freysoldt_correction_x-axis.pdf")

            kcc = kumagai_correction_from_paths(
                "test_path_files/sub_1_Sb_on_Ga/charge_2/",
                "test_path_files/bulk/",
                18.12,
                2,
                plot=True,
                filename="test_kumagai_correction",
            )
            assert np.isclose(kcc, 0.638776853061614, rtol=1e-5)
            assert os.path.exists("test_kumagai_correction.pdf")


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
        axisdata = [np.arange(0.0, lattval, 0.2) for lattval in abc]
        bldata = [np.array([1.0 for _ in np.arange(0.0, lattval, 0.2)]) for lattval in abc]
        dldata = [
            np.array([(-1 - np.cos(2 * np.pi * u / lattval)) for u in np.arange(0.0, lattval, 0.2)])
            for lattval in abc
        ]
        # load necessary parameters for defect_entry to make use of Freysoldt and Kumagai corrections
        metadata = {
            "axis_grid": axisdata,
            "bulk_planar_averages": bldata,
            "defect_planar_averages": dldata,
            "defect_structure": ids,
            "defect_frac_sc_coords": struct.sites[0].frac_coords,
            "bulk_sc_structure": struct,
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

        site_matching_indices = [[ind, ind - 1] for ind in range(len(struct.sites)) if ind != 0]

        metadata.update(
            {
                "bulk_atomic_site_averages": bulk_atomic_site_averages,
                "defect_atomic_site_averages": defect_atomic_site_averages,
                "site_matching_indices": site_matching_indices,
            }
        )
        self.defect_entry = DefectEntry(
            vac,
            charge_state=-3,
            sc_entry=ComputedStructureEntry(
                structure=ids,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            sc_defect_frac_coords=struct.sites[0].frac_coords,
            calculation_metadata=metadata,
        )

    def test_get_correction_freysoldt(self):
        freyout = get_correction_freysoldt(self.defect_entry, self.dielectric, partflag="All", axis=None)
        assert freyout == 5.445950368792991

        freyout = get_correction_freysoldt(
            self.defect_entry, self.dielectric, partflag="AllSplit", axis=None
        )
        assert np.isclose(freyout[0], 0.975893, rtol=1e-5)
        assert np.isclose(freyout[1], 4.4700573687929905, rtol=1e-5)
        assert np.isclose(freyout[2], 5.445950368792991, rtol=1e-5)

    def test_get_correction_kumagai(self):
        kumagaiout = get_correction_kumagai(self.defect_entry, self.dielectric, partflag="AllSplit")
        assert np.isclose(kumagaiout[0], 0.9763991294314076, rtol=1e-5)
        assert np.isclose(kumagaiout[1], 0.2579750033409367, rtol=1e-5)
        assert np.isclose(kumagaiout[2], 1.2343741327723443, rtol=1e-5)
