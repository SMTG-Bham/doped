# coding: utf-8

from __future__ import division

__status__ = "Development"

import os
import tarfile
import unittest
from shutil import copyfile

from monty.tempfile import ScratchDir
from pymatgen.analysis.defects.core import DefectEntry, Substitution, Vacancy
from pymatgen.core import Element, PeriodicSite
from pymatgen.io.vasp import Locpot, Vasprun
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.core.defects_analyzer import ComputedDefect
from doped.pycdt.utils.parse_calculations import (
    PostProcess,
    SingleDefectParser,
    _convert_cd_to_de,
)

file_loc = os.path.abspath(os.path.join(__file__, "..", "..", "..", "test_files"))


class SingleDefectParserTest(PymatgenTest):
    def test_all_methods(self):
        # create scratch directory with files....
        # having to do it all at once to minimize amount of time copying over to Scratch Directory
        with ScratchDir("."):
            # setup with fake Locpot object copied over
            copyfile(
                os.path.join(file_loc, "test_path_files.tar.gz"),
                "./test_path_files.tar.gz",
            )
            tar = tarfile.open("test_path_files.tar.gz")
            tar.extractall()
            tar.close()
            blocpot = Locpot.from_file(os.path.join(file_loc, "bLOCPOT.gz"))
            blocpot.write_file("test_path_files/bulk/LOCPOT")
            dlocpot = Locpot.from_file(os.path.join(file_loc, "dLOCPOT.gz"))
            dlocpot.write_file("test_path_files/sub_1_Sb_on_Ga/charge_2/LOCPOT")

            # test_from_path
            sdp = SingleDefectParser.from_paths(
                "test_path_files/sub_1_Sb_on_Ga/charge_2/",
                "test_path_files/bulk/",
                18.12,
                2,
            )
            self.assertIsInstance(sdp, SingleDefectParser)
            self.assertIsInstance(sdp.defect_entry.defect, Substitution)
            self.assertEqual(
                sdp.defect_entry.parameters["bulk_path"], "test_path_files/bulk/"
            )
            self.assertEqual(
                sdp.defect_entry.parameters["defect_path"],
                "test_path_files/sub_1_Sb_on_Ga/charge_2/",
            )

            # test_freysoldt_loader
            for param_key in [
                "axis_grid",
                "bulk_planar_averages",
                "defect_planar_averages",
                "initial_defect_structure",
                "defect_frac_sc_coords",
            ]:
                self.assertFalse(param_key in sdp.defect_entry.parameters.keys())
            bl = sdp.freysoldt_loader()
            self.assertIsInstance(bl, Locpot)
            for param_key in [
                "axis_grid",
                "bulk_planar_averages",
                "defect_planar_averages",
                "initial_defect_structure",
                "defect_frac_sc_coords",
            ]:
                self.assertTrue(param_key in sdp.defect_entry.parameters.keys())

            # test_kumagai_loader
            for param_key in [
                "bulk_atomic_site_averages",
                "defect_atomic_site_averages",
                "site_matching_indices",
                "sampling_radius",
                "defect_index_sc_coords",
            ]:
                self.assertFalse(param_key in sdp.defect_entry.parameters.keys())
            bo = sdp.kumagai_loader()
            for param_key in [
                "bulk_atomic_site_averages",
                "defect_atomic_site_averages",
                "site_matching_indices",
                "sampling_radius",
                "defect_index_sc_coords",
            ]:
                self.assertTrue(param_key in sdp.defect_entry.parameters.keys())

            # test_get_stdrd_metadata
            sdp.get_stdrd_metadata()
            for param_key in [
                "eigenvalues",
                "kpoint_weights",
                "bulk_energy",
                "final_defect_structure",
                "defect_energy",
                "run_metadata",
            ]:
                self.assertTrue(param_key in sdp.defect_entry.parameters.keys())

            # test_get_bulk_gap_data
            sdp.get_bulk_gap_data()
            self.assertEqual(sdp.defect_entry.parameters["mpid"], "mp-2534")
            self.assertAlmostEqual(sdp.defect_entry.parameters["gap"], 0.1887)

            # test_run_compatibility
            self.assertFalse("is_compatible" in sdp.defect_entry.parameters)
            sdp.run_compatibility()
            self.assertTrue("is_compatible" in sdp.defect_entry.parameters)


class PostProcessTest(PymatgenTest):
    def test_get_chempot_limits(self):
        with ScratchDir("."):
            # make a fake file structure to parse vaspruns
            os.mkdir("bulk")
            copyfile(os.path.join(file_loc, "vasprun.xml_GaAs"), "bulk/vasprun.xml")
            pp = PostProcess(".")
            gaas_cp = pp.get_chempot_limits()
            self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(gaas_cp.keys()))
            self.assertEqual(
                [-4.6580705550000001, -4.9884807425],
                [gaas_cp["As-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            )
            self.assertEqual(
                [-6.609317857500001, -3.03723344],
                [gaas_cp["Ga-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            )


if __name__ == "__main__":
    unittest.main()
