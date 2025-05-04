"""
Tests for the ``doped.complexes`` module and associated complex
generation/analysis functionality.
"""

import os
import shutil
import unittest

from pymatgen.core.structure import Structure
from test_generation import _check_defect_entry

from doped.complexes import classify_vacancy_geometry, generate_complex_from_defect_sites
from doped.generation import DefectsGenerator
from doped.utils.parsing import get_matching_site
from doped.utils.symmetry import get_all_equiv_sites, get_sga


def if_present_rm(path):
    """
    Remove the file/folder if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class ComplexDefectTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.R3c_Ga2O3 = Structure.from_file(os.path.join(self.data_dir, "Ga2O3_R3c_POSCAR"))
        self.Ga2O3_defect_gen = DefectsGenerator(self.R3c_Ga2O3, extrinsic="Se")
        self.Ga2O3_candidate_split_vacs = self.Ga2O3_defect_gen.get_candidate_split_vacancies_for_element(
            "Ga"
        )
        # self.Ga2O3_C3i_split_vac_dict = next(iter(  # TODO
        #     [vac_dict for vac_dict in self.Ga2O3_candidate_split_vacs.values() if
        #      vac_dict["VIV_dist_key"] == '4.63_3.60' and vac_dict["VIV_bond_angle_degrees"] == 73.627]))

    def test_Ga2O3_periodicity_breaking_supercell_multiplicities(self):
        """
        Test that the correct multiplicities of interstitials, vacancies and
        substitutions are returned, despite the periodicity-breaking supercell
        here for R-3c Ga2O3.

        Previously, incorrect multiplicities were returned with this
        periodicity-breaking supercell.
        """
        # incorrect symmetrized structure with periodicity-breaking cell:
        bulk_supercell_symm_struct = get_sga(
            self.Ga2O3_defect_gen.bulk_supercell
        ).get_symmetrized_structure()
        assert len(bulk_supercell_symm_struct.equivalent_sites) == 5
        # correct symmetrized structure with primitive cell:
        assert (
            len(
                get_sga(self.Ga2O3_defect_gen.primitive_structure)
                .get_symmetrized_structure()
                .equivalent_sites
            )
            == 2
        )

        # this tests that defect multiplicities are as expected
        for defect_name, defect_entry in self.Ga2O3_defect_gen.defect_entries.items():
            _check_defect_entry(defect_entry, defect_name, self.Ga2O3_defect_gen)

        # check that the wrong result is obtained with the standard pymatgen-analysis-defects approach:
        for defect_entry_name in ["v_Ga_0", "Ga_O_0"]:
            assert len(
                bulk_supercell_symm_struct.find_equivalent_sites(
                    get_matching_site(
                        self.Ga2O3_defect_gen[defect_entry_name].defect_supercell_site,
                        self.Ga2O3_defect_gen.bulk_supercell,
                        anonymous=True,
                    )
                )
            ) != (
                self.Ga2O3_defect_gen[defect_entry_name].defect.multiplicity
                * (
                    len(self.Ga2O3_defect_gen.primitive_structure)
                    / len(self.Ga2O3_defect_gen.bulk_supercell)
                )
            )
        assert len(
            get_all_equiv_sites(
                self.Ga2O3_defect_gen["Ga_i_C2_0"].defect_supercell_site.frac_coords,
                self.Ga2O3_defect_gen.bulk_supercell,
            )
        ) != (
            self.Ga2O3_defect_gen["Ga_i_C2_0"].defect.multiplicity
            * (len(self.Ga2O3_defect_gen.primitive_structure) / len(self.Ga2O3_defect_gen.bulk_supercell))
        )

    def test_Ga2O3_candidate_split_vac_generation(self):
        """
        Test the generation of candidate split vacancies in Ga2O3.
        """
        assert len(self.Ga2O3_candidate_split_vacs) == 269
        unique_VIV_split_vacs = {
            vac_dict["VIV_dist_key"]: vac_dict for vac_dict in self.Ga2O3_candidate_split_vacs.values()
        }
        assert len(unique_VIV_split_vacs) == 126

        unique_VIV_angle_split_vacs = {
            frozenset(
                (
                    round(vac_dict["vacancy_1_interstitial_distance"], 2),
                    round(vac_dict["vacancy_2_interstitial_distance"], 2),
                    round(vac_dict["VIV_bond_angle_degrees"], 1),
                )
            ): vac_dict
            for vac_dict in self.Ga2O3_candidate_split_vacs.values()
        }
        # this is the de-duplication strategy so should be same length:
        assert len(unique_VIV_angle_split_vacs) == len(self.Ga2O3_candidate_split_vacs)

        # check they are all classified as split vacancies as expected:
        for split_vac_dict in list(self.Ga2O3_candidate_split_vacs.values()):
            structure = generate_complex_from_defect_sites(
                self.Ga2O3_defect_gen.bulk_supercell,
                vacancy_sites=[split_vac_dict["vacancy_1_site"], split_vac_dict["vacancy_2_site"]],
                interstitial_sites=split_vac_dict["interstitial_site"],
            )
            split_vac_dict["structure"] = structure
            assert (
                classify_vacancy_geometry(structure, self.Ga2O3_defect_gen.bulk_supercell)
                == "Split Vacancy"
            )

    # def test_Ga2O3_generate_complex_from_defect_sites(self):
    #     """
    #     Test the ``generate_complex_from_defect_sites`` function, using R-3c
    #     Ga2O3 as an example case.
    #     """
    #     # TODO
    #     self.Ga2O3_defect_gen
