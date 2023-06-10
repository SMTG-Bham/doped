"""
Tests for the `doped.generation` module.
"""
import os
import sys
import unittest
import warnings
from io import StringIO
from unittest.mock import patch

import numpy as np
from pymatgen.analysis.defects.core import Defect, DefectType
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

from doped.generation import DefectsGenerator


class DefectsGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.cdte_bulk = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        self.ytos_bulk_supercell = Structure.from_file(f"{self.example_dir}/YTOS/Bulk/POSCAR")
        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")

    def test_defects_generator_cdte(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            cdte_defect_gen = DefectsGenerator(self.cdte_bulk)
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        cdte_defect_gen_info = (
            """Vacancies    Charge States    Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
-----------  ---------------  ------------------  --------  ---------
v_Cd         [-2,-1,0,+1]     [0.00,0.00,0.00]    1         4a
v_Te         [-1,0,+1,+2]     [0.75,0.75,0.75]    1         4d

Substitutions    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------  ------------------  --------  ---------
Cd_Te            [-1,0,+1,+2,+3,+4]  [0.75,0.75,0.75]    1         4d
Te_Cd            [-4,-3,-2,-1,0,+1]  [0.00,0.00,0.00]    1         4a

Interstitials    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------  ------------------  --------  ---------
Cd_i_C3v         [-1,0,+1,+2]        [0.38,0.38,0.37]    4         16e
Cd_i_Td_Cd2.83   [-1,0,+1,+2]        [0.25,0.25,0.25]    1         4c
Cd_i_Td_Te2.83   [-1,0,+1,+2]        [0.50,0.50,0.50]    1         4b
Te_i_C3v         [-1,0,+1,+2,+3,+4]  [0.38,0.38,0.37]    4         16e
Te_i_Td_Cd2.83   [-1,0,+1,+2,+3,+4]  [0.25,0.25,0.25]    1         4c
Te_i_Td_Te2.83   [-1,0,+1,+2,+3,+4]  [0.50,0.50,0.50]    1         4b

\x1B[3mg\x1B[0m_site = Site Multiplicity (in Primitive Unit Cell)\n"""
            "Note that Wyckoff letters can depend on the ordering of elements in the primitive standard "
            "structure (returned by spglib)\n\n"
        )

        assert cdte_defect_gen_info in output

        # test attributes:
        structure_matcher = StructureMatcher()
        prim_struc_wout_oxi = cdte_defect_gen.primitive_structure.copy()
        prim_struc_wout_oxi.remove_oxidation_states()
        assert structure_matcher.fit(prim_struc_wout_oxi, self.cdte_bulk)
        np.testing.assert_array_equal(cdte_defect_gen.supercell_matrix, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

        # test defects
        assert len(cdte_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cdte_defect_gen.defects["vacancies"]) == 2
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in cdte_defect_gen.defects["vacancies"]
        )
        assert len(cdte_defect_gen.defects["substitutions"]) == 2
        assert all(
            defect.defect_type == DefectType.Substitution
            for defect in cdte_defect_gen.defects["substitutions"]
        )
        assert len(cdte_defect_gen.defects["interstitials"]) == 6
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in cdte_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in cdte_defect_gen.defects.values()
            for defect in defect_list
        )

        # test defect attributes
        assert cdte_defect_gen.defects["vacancies"][0].name == "v_Cd"
        assert cdte_defect_gen.defects["vacancies"][0].charge == 0
        assert cdte_defect_gen.defects["vacancies"][0].multiplicity == 1
        assert cdte_defect_gen.defects["vacancies"][0].defect_type == DefectType.Vacancy
        assert cdte_defect_gen.defects["vacancies"][0].structure == cdte_defect_gen.primitive_structure
        assert (
            cdte_defect_gen.defects["vacancies"][0].supercell_structure
            == cdte_defect_gen.supercell_structure
        )
        # assert cdte_defect_gen.defects["vacancies"][0].defect_site == PeriodicSite(
        #     "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        # )
        assert cdte_defect_gen.defects["vacancies"][0].defect_site_coords == [0, 0, 0]
        assert cdte_defect_gen.defects["vacancies"][0].defect_site_coords_frac == [0, 0, 0]
        assert cdte_defect_gen.defects["vacancies"][0].defect_site_coords_are_cartesian is False

        # test defect entries
        assert len(cdte_defect_gen.defect_entries) == 50
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in cdte_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].name == "Cd_i_C3v_0"
        assert cdte_defect_gen.defect_entries["v_Cd"].charge == 0
        assert cdte_defect_gen.defect_entries["v_Cd"].multiplicity == 1
        assert cdte_defect_gen.defect_entries["v_Cd"].defect_type == "vacancies"
        # neutral_defect_entry.wyckoff = wyckoff_label

        # test charge states

        # test other structures
        # test wyckoff generation for all space groups
        # test as_dict etc methods
        # test saving to and loading from json

    @patch("sys.stdout", new_callable=StringIO)
    def test_generator_tqdm(self, mock_stdout):
        with patch("doped.generation.tqdm") as mocked_tqdm:
            mocked_instance = mocked_tqdm.return_value
            DefectsGenerator(self.cdte_bulk)
            mocked_tqdm.assert_called_once()
            mocked_tqdm.assert_called_with(total=100)
            mocked_instance.set_description.assert_any_call("Getting primitive structure")
            mocked_instance.set_description.assert_any_call("Generating vacancies")
            mocked_instance.set_description.assert_any_call("Generating substitutions")
            mocked_instance.set_description.assert_any_call("Generating interstitials")
            mocked_instance.set_description.assert_any_call("Generating simulation supercell")
            mocked_instance.set_description.assert_any_call("Determining Wyckoff sites")
            mocked_instance.set_description.assert_any_call("Generating DefectEntry objects")

    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DefectsGenerator(self.cdte_bulk)
            assert len(w) == 0