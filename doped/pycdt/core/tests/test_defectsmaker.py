# coding: utf-8

from __future__ import division

__status__ = "Development"

import os

from pymatgen.core import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from doped.pycdt.core.defectsmaker import *

file_loc = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "test_files"))


class GetOptimizedScScaleTest(PymatgenTest):
    def setUp(self):
        self.gaas_prim_struct = Structure.from_file(
            os.path.join(file_loc, "POSCAR_GaAs")
        )

    def test_biggercell_wanted(self):
        lattchange = get_optimized_sc_scale(self.gaas_prim_struct, 300)
        self.assertEqual([5, 5, 5], lattchange)
        lattchange = get_optimized_sc_scale(self.gaas_prim_struct, 100)
        self.assertEqual([3, 3, 3], lattchange)


class DefectChargerSemiconductorTest(PymatgenTest):
    def setUp(self):
        self.gaas_struct = Structure.from_file(os.path.join(file_loc, "POSCAR_GaAs"))
        self.def_charger = DefectChargerSemiconductor(self.gaas_struct)

    def test_vacancy_charges(self):
        """
        Reference: PRB 71, 125207 (2005)
        """
        ga_vac_qs = self.def_charger.get_charges("vacancy", "Ga")
        as_vac_qs = self.def_charger.get_charges("vacancy", "As")
        self.assertIn(0, ga_vac_qs)
        self.assertIn(-3, ga_vac_qs)
        self.assertIn(1, as_vac_qs)
        self.assertIn(-3, as_vac_qs)

    def test_antisite_charges(self):
        ga_on_as_qs = self.def_charger.get_charges("antisite", "Ga", "As")
        as_on_ga_qs = self.def_charger.get_charges("antisite", "As", "Ga")
        self.assertIn(0, ga_on_as_qs)
        self.assertIn(2, ga_on_as_qs)
        self.assertIn(0, as_on_ga_qs)
        self.assertIn(-3, as_on_ga_qs)

    def test_substitution_charges(self):
        s_impurity_qs = self.def_charger.get_charges("substitution", "As", "S")
        se_impurity_qs = self.def_charger.get_charges("substitution", "As", "Se")
        mg_impurity_qs = self.def_charger.get_charges("substitution", "Ga", "Mg")
        zn_impurity_qs = self.def_charger.get_charges("substitution", "Ga", "Zn")
        self.assertArrayEqual(s_impurity_qs, [-1, 0, 1, 2, 3, 4, 5, 6])
        self.assertArrayEqual(se_impurity_qs, [-1, 0, 1, 2, 3, 4, 5, 6])
        self.assertArrayEqual(mg_impurity_qs, [-2, -1, 0, 1, 2])
        self.assertArrayEqual(zn_impurity_qs, [-2, -1, 0, 1, 2])

    def test_interstitial_charges(self):
        """
        References:
        N interstitial: +1 to -3 [J. Phys.: Condens. Matter 20 (2008) 235231]
        As Self interstital: arxiv.org/pdf/1101.1413.pdf
        """
        n_qs = self.def_charger.get_charges("interstitial", "N")
        self.assertIn(1, n_qs)
        self.assertIn(-3, n_qs)
        self_qs = self.def_charger.get_charges("interstitial", "As")
        self.assertIn(-1, self_qs)
        self.assertIn(1, self_qs)


class DefectChargerInsulatorTest(PymatgenTest):
    def setUp(self):
        cr2o3_struct = Structure.from_file(os.path.join(file_loc, "POSCAR_Cr2O3"))
        self.def_charger = DefectChargerInsulator(cr2o3_struct)

    def test_vacancy_charges(self):
        """
        For insulators, the range of defect charges expected is
        -A to 0 or 0 to B, where A is cation charge in its native
        oxidation state in the compound, and B is the corresponding
        anion charge.
        """
        cr_vac_qs = self.def_charger.get_charges("vacancy", "Cr")
        o_vac_qs = self.def_charger.get_charges("vacancy", "O")
        self.assertIn(0, cr_vac_qs)
        self.assertIn(-3, cr_vac_qs)
        self.assertNotIn(-4, cr_vac_qs)
        self.assertNotIn(1, cr_vac_qs)
        self.assertIn(0, o_vac_qs)
        self.assertIn(2, o_vac_qs)
        self.assertNotIn(3, o_vac_qs)
        self.assertNotIn(-1, o_vac_qs)

    def test_antisite_charges(self):
        """
        Anitisites are not expected for insulators.
        Skipping this.
        """
        pass

    def test_substitution_charges(self):
        ti_on_cr_qs = self.def_charger.get_charges("substitution", "Cr", "Ti")
        self.assertIn(0, ti_on_cr_qs)
        self.assertIn(1, ti_on_cr_qs)
        self.assertNotIn(-1, ti_on_cr_qs)
        self.assertNotIn(2, ti_on_cr_qs)
        mg_on_cr_qs = self.def_charger.get_charges("substitution", "Cr", "Mg")
        self.assertIn(-1, mg_on_cr_qs)
        self.assertNotIn(0, mg_on_cr_qs)
        self.assertNotIn(-2, mg_on_cr_qs)
        self.assertNotIn(1, mg_on_cr_qs)

    def test_interstitial_charges(self):
        ti_inter_qs = self.def_charger.get_charges("interstitial", "Ti")
        self.assertIn(0, ti_inter_qs)
        self.assertIn(4, ti_inter_qs)
        self.assertNotIn(-1, ti_inter_qs)
        self.assertNotIn(5, ti_inter_qs)


class ChargedDefectsStructuresTest(PymatgenTest):
    def setUp(self):
        self.gaas_struct = Structure.from_file(os.path.join(file_loc, "POSCAR_GaAs"))
        self.ga_site = self.gaas_struct.sites[0]
        self.as_site = self.gaas_struct.sites[1]

    def test_simple_initialization(self):
        # test simple attributes
        CDS = ChargedDefectsStructures(self.gaas_struct)
        self.assertIsInstance(CDS.struct, Structure)
        self.assertIsInstance(CDS.defect_charger, DefectChargerSemiconductor)
        self.assertFalse(CDS.substitutions)
        superstruct = CDS.defects["bulk"]["supercell"]["structure"]
        self.assertIsInstance(superstruct, Structure)
        cellsize = CDS.defects["bulk"]["supercell"]["size"]
        self.assertEqual([4, 4, 4], cellsize)

        # test native (intrinsic) defect generation
        self.assertEqual("vac_1_Ga", CDS.defects["vacancies"][0]["name"])
        self.assertEqual(self.ga_site, CDS.defects["vacancies"][0]["unique_site"])
        self.assertEqual("vac_2_As", CDS.defects["vacancies"][1]["name"])
        self.assertEqual(self.as_site, CDS.defects["vacancies"][1]["unique_site"])

        self.assertEqual("as_1_As_on_Ga", CDS.defects["substitutions"][0]["name"])
        self.assertEqual(self.ga_site, CDS.defects["substitutions"][0]["unique_site"])
        self.assertEqual("as_1_Ga_on_As", CDS.defects["substitutions"][1]["name"])
        self.assertEqual(self.as_site, CDS.defects["substitutions"][1]["unique_site"])

    def test_extra_initialization(self):
        CDS = ChargedDefectsStructures(
            self.gaas_struct, cellmax=513, struct_type="insulator", antisites_flag=False
        )
        self.assertIsInstance(CDS.defect_charger, DefectChargerInsulator)
        cellsize = CDS.defects["bulk"]["supercell"]["size"]
        self.assertEqual([5, 5, 5], cellsize)
        self.assertFalse(len(CDS.defects["substitutions"]))  # testing antisite flag

    def test_subs_and_interstits(self):
        # test manual subtitution specification
        CDS = ChargedDefectsStructures(
            self.gaas_struct,
            antisites_flag=False,
            substitutions={"Ga": ["Si", "In"], "As": ["Sb"]},
        )
        check_subs = {
            sub["name"]: sub["unique_site"] for sub in CDS.defects["substitutions"]
        }
        self.assertEqual(3, len(check_subs))
        self.assertEqual(self.ga_site, check_subs["sub_1_Si_on_Ga"])
        self.assertEqual(self.ga_site, check_subs["sub_1_In_on_Ga"])
        self.assertEqual(self.as_site, check_subs["sub_2_Sb_on_As"])

        # Test automatic interstitial finding.
        CDS = ChargedDefectsStructures(
            self.gaas_struct, include_interstitials=True, interstitial_elements=["Mn"]
        )
        self.assertEqual(CDS.get_n_defects_of_type("interstitials"), 2)
        fnames = [
            i["name"][i["name"].index("M") :] for i in CDS.defects["interstitials"]
        ]
        self.assertEqual(sorted(fnames), sorted(["Mn_InFiT1_mult6", "Mn_InFiT2_mult6"]))
        nsites = len(CDS.defects["interstitials"][0]["supercell"]["structure"].sites)
        self.assertEqual(
            len(CDS.get_ith_supercell_of_defect_type(0, "interstitials").sites), nsites
        )
        self.assertEqual(
            CDS.defects["interstitials"][0]["charges"], [0, 1, 2, 3, 4, 5, 6, 7]
        )
        self.assertEqual(
            CDS.defects["interstitials"][1]["charges"], [0, 1, 2, 3, 4, 5, 6, 7]
        )

        # Test manual interstitial specification.
        isite = PeriodicSite(
            "Mn",
            CDS.defects["interstitials"][0]["supercell"]["structure"][
                nsites - 1
            ].coords,
            self.gaas_struct.lattice,
        )
        cds2 = ChargedDefectsStructures(
            self.gaas_struct,
            include_interstitials=True,
            standardized=False,
            # standardized = False is required when manually specifying interstitial,
            # because dont want base structure lattice to change
            intersites=(isite,),
        )
        self.assertEqual(cds2.get_n_defects_of_type("interstitials"), 2)
        fnames = [
            i["name"][i["name"].index("_") + 3 :] for i in cds2.defects["interstitials"]
        ]
        self.assertEqual(sorted(fnames), sorted(["As", "Ga"]))
        nsites = len(cds2.defects["interstitials"][0]["supercell"]["structure"].sites)
        self.assertEqual(
            len(cds2.get_ith_supercell_of_defect_type(0, "interstitials").sites), nsites
        )
        cds3 = ChargedDefectsStructures(
            self.gaas_struct,
            include_interstitials=True,
            standardized=False,
            # standardized = False is required when manually specifying interstitial,
            # because dont want base structure lattice to change
            interstitial_elements=["Mn"],
            intersites=(isite,),
        )
        self.assertEqual(cds3.get_n_defects_of_type("interstitials"), 1)
        fnames = [
            i["name"][i["name"].index("_") + 3 :] for i in cds3.defects["interstitials"]
        ]
        self.assertEqual(sorted(fnames), sorted(["Mn"]))
        nsites = len(cds3.defects["interstitials"][0]["supercell"]["structure"].sites)
        self.assertEqual(
            len(cds3.get_ith_supercell_of_defect_type(0, "interstitials").sites), nsites
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
