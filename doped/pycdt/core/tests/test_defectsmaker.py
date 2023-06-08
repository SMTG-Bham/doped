import os

from pymatgen.util.testing import PymatgenTest

from doped.pycdt.core.defectsmaker import *

file_loc = os.path.abspath(os.path.join(__file__, "..", "..", "..", "test_files"))


class GetOptimizedScScaleTest(PymatgenTest):
    def setUp(self):
        self.gaas_prim_struct = Structure.from_file(os.path.join(file_loc, "POSCAR_GaAs"))

    def test_biggercell_wanted(self):
        lattchange = get_optimized_sc_scale(self.gaas_prim_struct, 300)
        assert [5, 5, 5] == lattchange
        lattchange = get_optimized_sc_scale(self.gaas_prim_struct, 100)
        assert [3, 3, 3] == lattchange


class DefectChargerSemiconductorTest(PymatgenTest):
    def setUp(self):
        self.gaas_struct = Structure.from_file(os.path.join(file_loc, "POSCAR_GaAs"))
        self.def_charger = DefectChargerSemiconductor(self.gaas_struct)

    def test_vacancy_charges(self):
        """
        Reference: PRB 71, 125207 (2005).
        """
        ga_vac_qs = self.def_charger.get_charges("vacancy", "Ga")
        as_vac_qs = self.def_charger.get_charges("vacancy", "As")
        assert 0 in ga_vac_qs
        assert -3 in ga_vac_qs
        assert 1 in as_vac_qs
        assert -3 in as_vac_qs

    def test_antisite_charges(self):
        ga_on_as_qs = self.def_charger.get_charges("antisite", "Ga", "As")
        as_on_ga_qs = self.def_charger.get_charges("antisite", "As", "Ga")
        assert 0 in ga_on_as_qs
        assert 2 in ga_on_as_qs
        assert 0 in as_on_ga_qs
        assert -3 in as_on_ga_qs

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
        As Self interstital: arxiv.org/pdf/1101.1413.pdf.
        """
        n_qs = self.def_charger.get_charges("interstitial", "N")
        assert 1 in n_qs
        assert -3 in n_qs
        self_qs = self.def_charger.get_charges("interstitial", "As")
        assert -1 in self_qs
        assert 1 in self_qs


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
        assert 0 in cr_vac_qs
        assert -3 in cr_vac_qs
        assert -4 not in cr_vac_qs
        assert 1 not in cr_vac_qs
        assert 0 in o_vac_qs
        assert 2 in o_vac_qs
        assert 3 not in o_vac_qs
        assert -1 not in o_vac_qs

    def test_antisite_charges(self):
        """
        Anitisites are not expected for insulators.

        Skipping this.
        """

    def test_substitution_charges(self):
        ti_on_cr_qs = self.def_charger.get_charges("substitution", "Cr", "Ti")
        assert 0 in ti_on_cr_qs
        assert 1 in ti_on_cr_qs
        assert -1 not in ti_on_cr_qs
        assert 2 not in ti_on_cr_qs
        mg_on_cr_qs = self.def_charger.get_charges("substitution", "Cr", "Mg")
        assert -1 in mg_on_cr_qs
        assert 0 not in mg_on_cr_qs
        assert -2 not in mg_on_cr_qs
        assert 1 not in mg_on_cr_qs

    def test_interstitial_charges(self):
        ti_inter_qs = self.def_charger.get_charges("interstitial", "Ti")
        assert 0 in ti_inter_qs
        assert 4 in ti_inter_qs
        assert -1 not in ti_inter_qs
        assert 5 not in ti_inter_qs


class ChargedDefectsStructuresTest(PymatgenTest):
    def setUp(self):
        self.gaas_struct = Structure.from_file(os.path.join(file_loc, "POSCAR_GaAs"))
        self.ga_site = self.gaas_struct.sites[0]
        self.as_site = self.gaas_struct.sites[1]

    def test_simple_initialization(self):
        # test simple attributes
        CDS = ChargedDefectsStructures(self.gaas_struct)
        assert isinstance(CDS.struct, Structure)
        assert isinstance(CDS.defect_charger, DefectChargerSemiconductor)
        assert not CDS.substitutions
        superstruct = CDS.defects["bulk"]["supercell"]["structure"]
        assert isinstance(superstruct, Structure)
        cellsize = CDS.defects["bulk"]["supercell"]["size"]
        assert [4, 4, 4] == cellsize

        # test native (intrinsic) defect generation
        assert CDS.defects["vacancies"][0]["name"] == "vac_1_Ga"
        assert self.ga_site == CDS.defects["vacancies"][0]["unique_site"]
        assert CDS.defects["vacancies"][1]["name"] == "vac_2_As"
        assert self.as_site == CDS.defects["vacancies"][1]["unique_site"]

        assert CDS.defects["substitutions"][0]["name"] == "as_1_As_on_Ga"
        assert self.ga_site == CDS.defects["substitutions"][0]["unique_site"]
        assert CDS.defects["substitutions"][1]["name"] == "as_1_Ga_on_As"
        assert self.as_site == CDS.defects["substitutions"][1]["unique_site"]

    def test_extra_initialization(self):
        CDS = ChargedDefectsStructures(
            self.gaas_struct, cellmax=513, struct_type="insulator", antisites_flag=False
        )
        assert isinstance(CDS.defect_charger, DefectChargerInsulator)
        cellsize = CDS.defects["bulk"]["supercell"]["size"]
        assert [5, 5, 5] == cellsize
        assert not len(CDS.defects["substitutions"])  # testing antisite flag

    def test_subs_and_interstits(self):
        # test manual substitution specification
        CDS = ChargedDefectsStructures(
            self.gaas_struct,
            antisites_flag=False,
            substitutions={"Ga": ["Si", "In"], "As": ["Sb"]},
        )
        check_subs = {sub["name"]: sub["unique_site"] for sub in CDS.defects["substitutions"]}
        assert len(check_subs) == 3
        assert self.ga_site == check_subs["sub_1_Si_on_Ga"]
        assert self.ga_site == check_subs["sub_1_In_on_Ga"]
        assert self.as_site == check_subs["sub_2_Sb_on_As"]

        # Test automatic interstitial finding.
        CDS = ChargedDefectsStructures(
            self.gaas_struct, include_interstitials=True, interstitial_elements=["Mn"]
        )
        assert CDS.get_n_defects_of_type("interstitials") == 3
        fnames = [i["name"][i["name"].index("M") :] for i in CDS.defects["interstitials"]]
        nsites = len(CDS.defects["interstitials"][0]["supercell"]["structure"].sites)
        assert len(CDS.get_ith_supercell_of_defect_type(0, "interstitials").sites) == nsites
        assert CDS.defects["interstitials"][0]["charges"] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert CDS.defects["interstitials"][1]["charges"] == [0, 1, 2, 3, 4, 5, 6, 7]

        # Test manual interstitial specification.
        isite = PeriodicSite(
            "Mn",
            CDS.defects["interstitials"][0]["supercell"]["structure"][nsites - 1].coords,
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
        assert cds2.get_n_defects_of_type("interstitials") == 2
        fnames = [i["name"][i["name"].index("_") + 3 :] for i in cds2.defects["interstitials"]]
        assert sorted(fnames) == sorted(["As", "Ga"])
        nsites = len(cds2.defects["interstitials"][0]["supercell"]["structure"].sites)
        assert len(cds2.get_ith_supercell_of_defect_type(0, "interstitials").sites) == nsites
        cds3 = ChargedDefectsStructures(
            self.gaas_struct,
            include_interstitials=True,
            standardized=False,
            # standardized = False is required when manually specifying interstitial,
            # because dont want base structure lattice to change
            interstitial_elements=["Mn"],
            intersites=(isite,),
        )
        assert cds3.get_n_defects_of_type("interstitials") == 1
        fnames = [i["name"][i["name"].index("_") + 3 :] for i in cds3.defects["interstitials"]]
        assert sorted(fnames) == sorted(["Mn"])
        nsites = len(cds3.defects["interstitials"][0]["supercell"]["structure"].sites)
        assert len(cds3.get_ith_supercell_of_defect_type(0, "interstitials").sites) == nsites


if __name__ == "__main__":
    import unittest

    unittest.main()
