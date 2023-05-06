# coding: utf-8

from __future__ import division

__status__ = "Development"

import copy
import inspect
import os
import unittest
from shutil import copyfile

from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.core._chemical_potentials import (
    ChemPotAnalyzer,
    MPChemPotAnalyzer,
    UserChemPotAnalyzer,
    UserChemPotInputGenerator,
    get_mp_chempots_from_dpd,
)

TEST_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "test_files"))


class UserChemPotAnalyzerTest(PymatgenTest):
    def setUp(self):
        with MPRester(api_key="c2LiJRMiBeaN5iXsH") as mp:
            self.bulk_ce = mp.get_entry_by_material_id("mp-2534")
        self.UCPA = UserChemPotAnalyzer(
            bulk_ce=self.bulk_ce, mapi_key="c2LiJRMiBeaN5iXsH"
        )
        # SK MP Imperial email A/C API key
        self.UCPA_sub = UserChemPotAnalyzer(
            bulk_ce=self.bulk_ce, sub_species=["In"], mapi_key="c2LiJRMiBeaN5iXsH"
        )

    def test_read_phase_diagram_and_chempots(self):
        # set up a local phase diagram object...
        # test non mp case,
        with ScratchDir("."):
            # os.mkdir('PhaseDiagram')
            os.makedirs(os.path.join("PhaseDiagram", "Ga"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_Ga"),
                os.path.join("PhaseDiagram", "Ga", "vasprun.xml"),
            )
            os.mkdir(os.path.join("PhaseDiagram", "As"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_As"),
                os.path.join("PhaseDiagram", "As", "vasprun.xml"),
            )
            os.mkdir(os.path.join("PhaseDiagram", "GaAs"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_GaAs"),
                os.path.join("PhaseDiagram", "GaAs", "vasprun.xml"),
            )
            cp = self.UCPA.read_phase_diagram_and_chempots(
                full_sub_approach=False, include_mp_entries=False
            )
            self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(cp["facets"].keys()))
            self.assertEqual(
                [-5.36, -4.29],
                [
                    round(cp["facets"]["As-GaAs"][Element(elt)], 2)
                    for elt in ["As", "Ga"]
                ],
            )
            self.assertEqual(
                [-6.04, -3.61],
                [
                    round(cp["facets"]["Ga-GaAs"][Element(elt)], 2)
                    for elt in ["As", "Ga"]
                ],
            )

        # followed by an case where MP needs to supplement...
        with ScratchDir("."):
            os.mkdir("PhaseDiagram")
            # NO Ga entry included this time
            os.mkdir(os.path.join("PhaseDiagram", "As"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_As"),
                os.path.join("PhaseDiagram", "As", "vasprun.xml"),
            )
            os.mkdir(os.path.join("PhaseDiagram", "GaAs"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_GaAs"),
                os.path.join("PhaseDiagram", "GaAs", "vasprun.xml"),
            )
            cp = self.UCPA.read_phase_diagram_and_chempots(
                full_sub_approach=False, include_mp_entries=True
            )
            self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(cp["facets"].keys()))
            self.assertEqual(
                [-5.36, -4.29],
                [
                    round(cp["facets"]["As-GaAs"][Element(elt)], 2)
                    for elt in ["As", "Ga"]
                ],
            )
            self.assertEqual(
                [-6.62, -3.03],
                [
                    round(cp["facets"]["Ga-GaAs"][Element(elt)], 2)
                    for elt in ["As", "Ga"]
                ],
            )

        # quick and dirty test for finding extrinsic defects...
        with ScratchDir("."):
            os.mkdir("PhaseDiagram")
            # NO Ga entry or In entry this time
            os.mkdir(os.path.join("PhaseDiagram", "As"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_As"),
                os.path.join("PhaseDiagram", "As", "vasprun.xml"),
            )
            os.mkdir(os.path.join("PhaseDiagram", "GaAs"))
            copyfile(
                os.path.join(TEST_DIR, "vasprun.xml_GaAs"),
                os.path.join("PhaseDiagram", "GaAs", "vasprun.xml"),
            )
            cp = self.UCPA_sub.read_phase_diagram_and_chempots(
                full_sub_approach=False, include_mp_entries=True)
            self.assertEqual({'As-GaAs-In', 'Ga-GaAs-In'}, set(cp["facets"].keys()))


if __name__ == "__main__":
    unittest.main()
