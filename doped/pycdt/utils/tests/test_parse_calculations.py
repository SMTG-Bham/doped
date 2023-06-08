import os
import tarfile
import unittest
from shutil import copyfile

import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.analysis.defects.core import Substitution
from pymatgen.core import Element
from pymatgen.io.vasp import Locpot
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.utils.parse_calculations import PostProcess, SingleDefectParser

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
                charge=2,
            )
            assert isinstance(sdp, SingleDefectParser)
            assert isinstance(sdp.defect_entry.defect, Substitution)
            assert sdp.defect_entry.parameters["bulk_path"] == "test_path_files/bulk/"
            assert sdp.defect_entry.parameters["defect_path"] == "test_path_files/sub_1_Sb_on_Ga/charge_2/"

            # test_freysoldt_loader
            for param_key in [
                "axis_grid",
                "bulk_planar_averages",
                "defect_planar_averages",
                "defect_frac_sc_coords",
            ]:
                assert param_key not in sdp.defect_entry.parameters.keys()
            sdp.freysoldt_loader()
            for param_key in [
                "axis_grid",
                "bulk_planar_averages",
                "defect_planar_averages",
                "initial_defect_structure",
                "defect_frac_sc_coords",
            ]:
                assert param_key in sdp.defect_entry.parameters

            # test_kumagai_loader
            for param_key in [
                "bulk_atomic_site_averages",
                "defect_atomic_site_averages",
                "site_matching_indices",
                "sampling_radius",
            ]:
                assert param_key not in sdp.defect_entry.parameters.keys()
            sdp.kumagai_loader()
            for param_key in [
                "bulk_atomic_site_averages",
                "defect_atomic_site_averages",
                "site_matching_indices",
                "defect_frac_sc_coords",
            ]:
                assert param_key in sdp.defect_entry.parameters

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
                assert param_key in sdp.defect_entry.parameters

            # test_get_bulk_gap_data
            sdp.get_bulk_gap_data()
            self.assertAlmostEqual(sdp.defect_entry.parameters["gap"], 1.2167)

            # test_run_compatibility
            assert "is_compatible" not in sdp.defect_entry.parameters
            sdp.run_compatibility()
            assert "is_compatible" in sdp.defect_entry.parameters


class PostProcessTest(PymatgenTest):
    def test_get_chempot_limits(self):
        with ScratchDir("."):
            # make a fake file structure to parse vaspruns
            os.mkdir("bulk")
            copyfile(os.path.join(file_loc, "vasprun.xml_GaAs"), "bulk/vasprun.xml")
            pp = PostProcess(".", mapi_key="c2LiJRMiBeaN5iXsH")  # SK MP Imperial email A/C API key
            gaas_cp = pp.get_chempot_limits()
            assert {"As-GaAs", "Ga-GaAs"} == set(gaas_cp.keys())
            np.testing.assert_almost_equal(
                [-4.65807055, -4.9884807425],
                [gaas_cp["As-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
                decimal=3,
            )
            np.testing.assert_almost_equal(
                [-6.618, -3.028], [gaas_cp["Ga-GaAs"][Element(elt)] for elt in ["As", "Ga"]], decimal=3
            )


if __name__ == "__main__":
    unittest.main()
