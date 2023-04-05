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

file_loc = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "test_files"))


class DpdAnalyzerTest(PymatgenTest):
    def setUp(self):
        dpd_path = os.path.split(inspect.getfile(DefectPhaseDiagram))[0]
        pymatgen_test_files = os.path.abspath(os.path.join(dpd_path, "tests"))
        vbm_val = 2.6682
        gap = 1.5
        entries = list(
            loadfn(
                os.path.join(pymatgen_test_files, "GaAs_test_defentries.json")
            ).values()
        )
        for entry in entries:
            entry.parameters.update({"vbm": vbm_val})

        self.dpd = DefectPhaseDiagram(entries, vbm_val, gap)

    def test_get_mp_chempots_from_dpd(self):
        cps = get_mp_chempots_from_dpd(self.dpd)
        self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(cps.keys()))
        self.assertEqual(
            [-4.6580705550000001, -3.7317319750000006],
            [cps["As-GaAs"][Element("As")], cps["As-GaAs"][Element("Ga")]],
        )
        self.assertEqual(
            [-5.352569090000001, -3.03723344],
            [cps["Ga-GaAs"][Element("As")], cps["Ga-GaAs"][Element("Ga")]],
        )


class ChemPotAnalyzerTest(PymatgenTest):
    def setUp(self):
        self.CPA = ChemPotAnalyzer()

    def test_get_chempots_from_pda(self):
        with MPRester() as mp:
            bulk_ce = mp.get_entry_by_material_id("mp-2534")
            bulk_species_symbol = [s.symbol for s in bulk_ce.composition.elements]
            entries = mp.get_entries_in_chemsys(bulk_species_symbol)

        # intrinsic chempot test
        pd = PhaseDiagram(entries)
        self.CPA.bulk_ce = bulk_ce
        gaas_cp = self.CPA.get_chempots_from_pd(pd)
        self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(gaas_cp.keys()))
        self.assertEqual(
            [-4.6580705550000001, -3.7317319750000006],
            [gaas_cp["As-GaAs"][Element("As")], gaas_cp["As-GaAs"][Element("Ga")]],
        )
        self.assertEqual(
            [-5.352569090000001, -3.03723344],
            [gaas_cp["Ga-GaAs"][Element("As")], gaas_cp["Ga-GaAs"][Element("Ga")]],
        )

    def test_diff_bulk_sub_phases(self):
        fl = ["GaAs", "Sb", "GaSb", "Ga"]
        blk, blknom, subnom = self.CPA.diff_bulk_sub_phases(fl, "Sb")
        self.assertEqual(["Ga", "GaAs"], blk)
        self.assertEqual("Ga-GaAs", blknom)
        self.assertEqual("GaSb-Sb", subnom)


class MPChemPotAnalyzerTest(PymatgenTest):
    def setUp(self):
        self.MPCPA = MPChemPotAnalyzer()

    def test_analyze_GGA_chempots(self):
        """
        Test-cases to cover:
            (0) (stable + mpid given = tested in the CPA.test_get_chempots_from_pda unit test)
            (i) user's computed entry is stable w.r.t MP phase diagram; full_sub_approach = False
            (ii) user's computed entry is stable w.r.t MP phase diagram (and not currently in phase diagram);
                full_sub_approach = False
            (iii) not stable, composition exists in list of stable comps of PD; full_sub_approach = False
            (iv) not stable, composition DOESNT exists in list of stable comps of PD; full_sub_approach = False
            (v) one example of full_sub_approach = True... (just do for case with user's entry being stable)
        """
        with MPRester() as mp:
            bulk_ce = mp.get_entry_by_material_id(
                "mp-2534"
            )  # simulates an entry to play with for this unit test

        # test (i) user's computed entry is stable w.r.t MP phase diagram; full_sub_approach = False
        bce = copy.copy(bulk_ce)
        self.MPCPA = MPChemPotAnalyzer(bulk_ce=bce, sub_species=set(["Sb", "In"]))
        cp_fsaf = self.MPCPA.analyze_GGA_chempots(full_sub_approach=False)
        self.assertEqual(
            set(["Ga-GaAs-GaSb-In", "As-GaAs-InAs-SbAs"]), set(cp_fsaf.keys())
        )
        true_answer = [
            cp_fsaf["Ga-GaAs-GaSb-In"][Element(elt)] for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [-4.44626405, -5.352569090000001, -3.03723344, -2.7518583366666665],
            true_answer,
        )
        # true_answer = [cp_fsaf['InAs-GaSb-In-GaAs'][Element(elt)] for elt in ['Sb', 'As', 'Ga', 'In']]
        # self.assertEqual([-4.243837989999999, -5.15014303, -3.239659500000001, -2.72488125],
        #                        true_answer)
        # true_answer = [cp_fsaf['InAs-GaSb-Sb-GaAs'][Element(elt)] for elt in ['Sb', 'As', 'Ga', 'In']]
        # self.assertEqual([-4.127761275, -5.0340663150000005, -3.3557362150000003, -2.8409579649999994],
        #                        true_answer)
        true_answer = [
            cp_fsaf["As-GaAs-InAs-SbAs"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [
                -4.1687393749999995,
                -4.658070555,
                -3.7317319750000006,
                -3.2169537249999998,
            ],
            true_answer,
        )

        # test (v) one example of full_sub_approach = True... (just doing for case with user's entry being stable)
        cp_fsat = self.MPCPA.analyze_GGA_chempots(full_sub_approach=True)
        self.assertEqual(
            set(
                [
                    "Ga-GaAs-GaSb-In",
                    "GaAs-InAs-InSb-Sb",
                    "GaAs-In-InAs-InSb",
                    "As-GaAs-InAs-SbAs",
                    "GaAs-GaSb-In-InSb",
                    "GaAs-InAs-Sb-SbAs",
                    "GaAs-GaSb-InSb-Sb",
                ]
            ),
            set(cp_fsat.keys()),
        )
        true_answer = [
            cp_fsat["Ga-GaAs-GaSb-In"][Element(elt)] for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [-4.44626405, -5.352569090000001, -3.03723344, -2.7518583366666665],
            true_answer,
        )
        true_answer = [
            cp_fsat["GaAs-InAs-InSb-Sb"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [-4.127761275, -4.895951335, -3.4938511950000004, -2.9790729449999995],
            true_answer,
        )
        true_answer = [
            cp_fsat["GaAs-In-InAs-InSb"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [
                -4.354975883333333,
                -5.123165943333333,
                -3.2666365866666673,
                -2.7518583366666665,
            ],
            true_answer,
        )
        true_answer = [
            cp_fsat["As-GaAs-InAs-SbAs"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [
                -4.1687393749999995,
                -4.658070555,
                -3.7317319750000006,
                -3.2169537249999998,
            ],
            true_answer,
        )
        true_answer = [
            cp_fsat["GaAs-GaSb-In-InSb"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [
                -4.354975883333333,
                -5.2612809233333335,
                -3.1285216066666672,
                -2.7518583366666665,
            ],
            true_answer,
        )
        true_answer = [
            cp_fsat["GaAs-InAs-Sb-SbAs"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [
                -4.127761275,
                -4.6990486549999995,
                -3.6907538750000013,
                -3.1759756250000004,
            ],
            true_answer,
        )
        true_answer = [
            cp_fsat["GaAs-GaSb-InSb-Sb"][Element(elt)]
            for elt in ["Sb", "As", "Ga", "In"]
        ]
        self.assertEqual(
            [
                -4.127761275,
                -5.0340663150000005,
                -3.3557362150000003,
                -2.9790729449999995,
            ],
            true_answer,
        )

        # test (iii) not stable, composition exists in list of stable comps of PD; full_sub_approach = False
        us_bce = ComputedEntry(Composition({"Ga": 1, "As": 1}), -8)
        self.MPCPA = MPChemPotAnalyzer(bulk_ce=us_bce)
        cp_fsaf_us_ce = self.MPCPA.analyze_GGA_chempots(full_sub_approach=False)
        self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(cp_fsaf_us_ce.keys()))
        self.assertEqual(
            [-4.6580705550000001, -3.7317319750000006],
            [cp_fsaf_us_ce["As-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
        )
        self.assertEqual(
            [-5.352569090000001, -3.03723344],
            [cp_fsaf_us_ce["Ga-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
        )

        # test (ii) user's computed entry is stable w.r.t MP phase diagram
        #       AND  composition DOESNT exists in list of stable comps of PD; full_sub_approach = False
        s_cdne_bce = ComputedEntry(Composition({"Ga": 2, "As": 3}), -21.5)
        self.MPCPA = MPChemPotAnalyzer(bulk_ce=s_cdne_bce)
        cp_fsaf_s_cdne = self.MPCPA.analyze_GGA_chempots(full_sub_approach=False)
        self.assertEqual(set(["Ga2As3-GaAs", "As-Ga2As3"]), set(cp_fsaf_s_cdne.keys()))
        for tested, answer in zip(
            [cp_fsaf_s_cdne["Ga2As3-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            [-4.720394939999998, -3.669407590000002],
        ):
            self.assertAlmostEqual(answer, tested)
        self.assertEqual(
            [-4.6580705550000001, -3.7628941674999994],
            [cp_fsaf_s_cdne["As-Ga2As3"][Element(elt)] for elt in ["As", "Ga"]],
        )

        # test (iv) not stable, composition DOESNT exists in list of stable comps of PD; full_sub_approach = False
        #       case a) simple 2D phase diagram
        us_cdne_bce_a = ComputedEntry(Composition({"Ga": 2, "As": 3}), -20.0)
        self.MPCPA = MPChemPotAnalyzer(bulk_ce=us_cdne_bce_a)
        cp_fsaf_us_cdne_a = self.MPCPA.analyze_GGA_chempots(full_sub_approach=False)
        self.assertEqual(set(["As-GaAs"]), set(cp_fsaf_us_cdne_a.keys()))
        self.assertEqual(
            [-4.658070555, -3.7317319750000006],
            [cp_fsaf_us_cdne_a["As-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
        )

        #       case b) larger phase diagram
        us_cdne_bce_b = ComputedEntry(Composition({"Ga": 2, "As": 3, "Sb": 2}), -20.0)
        self.MPCPA = MPChemPotAnalyzer(bulk_ce=us_cdne_bce_b)
        cp_fsaf_us_cdne_b = self.MPCPA.analyze_GGA_chempots(full_sub_approach=False)
        self.assertEqual(set(["GaAs-Sb-SbAs"]), set(cp_fsaf_us_cdne_b.keys()))
        self.assertEqual(
            [-4.127761275, -4.6990486549999995, -3.6907538750000013],
            [
                cp_fsaf_us_cdne_b["GaAs-Sb-SbAs"][Element(elt)]
                for elt in ["Sb", "As", "Ga"]
            ],
        )

    def test_get_chempots_from_composition(self):
        bulk_comp = Composition("Cr2O3")
        self.MPCPA = MPChemPotAnalyzer()  # reinitalize
        cro_cp = self.MPCPA.get_chempots_from_composition(bulk_comp)
        self.assertEqual(set(["Cr2O3-CrO2", "Cr-Cr2O3"]), set(cro_cp.keys()))
        self.assertAlmostEqual(-14.635303979999982, cro_cp["Cr2O3-CrO2"][Element("Cr")])
        self.assertAlmostEqual(-5.51908629500001, cro_cp["Cr2O3-CrO2"][Element("O")])
        self.assertAlmostEqual(-9.63670386, cro_cp["Cr-Cr2O3"][Element("Cr")])
        self.assertAlmostEqual(-8.851486374999999, cro_cp["Cr-Cr2O3"][Element("O")])

    def test_get_mp_entries(self):
        # name mp-id of GaAs system and get mp-entries...
        self.MPCPA = MPChemPotAnalyzer(mpid="mp-2534")
        self.MPCPA.get_mp_entries()
        ents = self.MPCPA.entries
        self.assertEqual(set(["bulk_derived", "subs_set"]), set(ents.keys()))
        self.assertTrue(len(ents["bulk_derived"]))


class UserChemPotAnalyzerTest(PymatgenTest):
    def setUp(self):
        with MPRester() as mp:
            self.bulk_ce = mp.get_entry_by_material_id("mp-2534")
        self.UCPA = UserChemPotAnalyzer(bulk_ce=self.bulk_ce)

    def test_read_phase_diagram_and_chempots(self):
        # set up a local phase diagram object...
        # test non mp case,
        with ScratchDir("."):
            os.mkdir("PhaseDiagram")
            os.mkdir("PhaseDiagram/Ga")
            copyfile(
                os.path.join(file_loc, "vasprun.xml_Ga"), "PhaseDiagram/Ga/vasprun.xml"
            )
            os.mkdir("PhaseDiagram/As")
            copyfile(
                os.path.join(file_loc, "vasprun.xml_As"), "PhaseDiagram/As/vasprun.xml"
            )
            os.mkdir("PhaseDiagram/GaAs")
            copyfile(
                os.path.join(file_loc, "vasprun.xml_GaAs"),
                "PhaseDiagram/GaAs/vasprun.xml",
            )
            cp = self.UCPA.read_phase_diagram_and_chempots(
                full_sub_approach=False, include_mp_entries=False
            )
            self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(cp.keys()))
            self.assertEqual(
                [-5.3585191833333328, -4.2880321141666675],
                [cp["As-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            )
            self.assertEqual(
                [-6.0386246900000007, -3.6079266075],
                [cp["Ga-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            )

        # followed by an case where MP needs to supplement...
        with ScratchDir("."):
            os.mkdir("PhaseDiagram")
            # NO Ga entry included this time
            os.mkdir("PhaseDiagram/As")
            copyfile(
                os.path.join(file_loc, "vasprun.xml_As"), "PhaseDiagram/As/vasprun.xml"
            )
            os.mkdir("PhaseDiagram/GaAs")
            copyfile(
                os.path.join(file_loc, "vasprun.xml_GaAs"),
                "PhaseDiagram/GaAs/vasprun.xml",
            )
            cp = self.UCPA.read_phase_diagram_and_chempots(
                full_sub_approach=False, include_mp_entries=True
            )
            self.assertEqual(set(["As-GaAs", "Ga-GaAs"]), set(cp.keys()))
            self.assertEqual(
                [-5.3585191833333328, -4.2880321141666675],
                [cp["As-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            )
            self.assertEqual(
                [-6.609317857500001, -3.03723344],
                [cp["Ga-GaAs"][Element(elt)] for elt in ["As", "Ga"]],
            )


class UserChemPotInputGeneratorTest(PymatgenTest):
    def setUp(self):
        self.UCPIGT = UserChemPotInputGenerator(Composition({"Ga": 1, "As": 1}))

    def test_setup_phase_diagram_calculations(self):
        # test that files get craeted for some file of interest...
        with ScratchDir("."):
            self.UCPIGT.setup_phase_diagram_calculations(full_phase_diagram=True)
            self.assertTrue(os.path.exists("PhaseDiagram/"))
            self.assertTrue(os.path.exists("PhaseDiagram/mp-11_As"))
            self.assertTrue(os.path.exists("PhaseDiagram/mp-11_As/POSCAR"))
            self.assertTrue(os.path.exists("PhaseDiagram/mp-142_Ga"))
            self.assertTrue(os.path.exists("PhaseDiagram/mp-142_Ga/POSCAR"))
            self.assertTrue(os.path.exists("PhaseDiagram/mp-2534_GaAs/"))
            self.assertTrue(os.path.exists("PhaseDiagram/mp-2534_GaAs/POSCAR"))


if __name__ == "__main__":
    unittest.main()
