import os
import tarfile
from shutil import copyfile
import numpy as np

from monty.tempfile import ScratchDir
from pymatgen.io.vasp import Locpot
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.defects.core import DefectEntry, Vacancy
from pymatgen.core.sites import PeriodicSite

from doped.corrections import (
    freysoldt_correction_from_paths,
    kumagai_correction_from_paths,
    get_correction_freysoldt,
    get_correction_kumagai,
)

test_files_dir = os.path.join(os.path.dirname(__file__), "../doped/pycdt/test_files")


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
            blocpot = Locpot.from_file(os.path.join(test_files_dir, "bLOCPOT.gz"))
            blocpot.write_file("test_path_files/bulk/LOCPOT")
            dlocpot = Locpot.from_file(os.path.join(test_files_dir, "dLOCPOT.gz"))
            dlocpot.write_file("test_path_files/sub_1_Sb_on_Ga/charge_2/LOCPOT")

            fcc = freysoldt_correction_from_paths(
                "test_path_files/sub_1_Sb_on_Ga/charge_2/",
                "test_path_files/bulk/",
                18.12,
                2,
                plot=True,
                filename="test_freysoldt_correction",
            )
            self.assertAlmostEqual(
                fcc, -1.4954476868106865
            )  # note this has been updated from the
            # pycdt version, because there they used a `transformation.json` that gave an
            # incorrect `initial_defect_structure` (corresponding to primitive rather than bulk)
            self.assertTrue(os.path.exists("test_freysoldt_correction_axis1.pdf"))

            kcc = kumagai_correction_from_paths(
                "test_path_files/sub_1_Sb_on_Ga/charge_2/",
                "test_path_files/bulk/",
                18.12,
                2,
                plot=True,
                filename="test_kumagai_correction",
            )
            self.assertAlmostEqual(kcc, 0.638776853061614)
            self.assertTrue(os.path.exists("test_kumagai_correction.pdf"))


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
            self.defect_entry, self.epsilon, partflag="All", axis=None
        )
        self.assertEqual(freyout, 5.445950368792991)

        freyout = get_correction_freysoldt(
            self.defect_entry, self.epsilon, partflag="AllSplit", axis=None
        )
        self.assertAlmostEqual(freyout[0], 0.975893)
        self.assertAlmostEqual(freyout[1], 4.4700573687929905)
        self.assertAlmostEqual(freyout[2], 5.445950368792991)

    def test_get_correction_kumagai(self):
        kumagaiout = get_correction_kumagai(
            self.defect_entry, self.epsilon, partflag="AllSplit"
        )
        self.assertAlmostEqual(kumagaiout[0], 0.9763991294314076)
        self.assertAlmostEqual(kumagaiout[1], 0.2579750033409367)
        self.assertAlmostEqual(kumagaiout[2], 1.2343741327723443)
