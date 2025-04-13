"""
Tests for the ``doped.complexes`` module and associated complex
generation/analysis functionality.
"""

import os
import shutil
import unittest

from pymatgen.core.structure import Structure

from doped.complexes import classify_vacancy_geometry, generate_complex_from_defect_sites
from doped.generation import DefectsGenerator


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
