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
from pymatgen.core.structure import PeriodicSite, Structure

from doped.generation import DefectsGenerator, get_wyckoff_dict_from_sgn


class DefectsGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
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
v_Te         [-1,0,+1,+2]     [0.25,0.25,0.25]    1         4c

Substitutions    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------  ------------------  --------  ---------
Cd_Te            [-1,0,+1,+2,+3,+4]  [0.25,0.25,0.25]    1         4c
Te_Cd            [-4,-3,-2,-1,0,+1]  [0.00,0.00,0.00]    1         4a

Interstitials    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------  ------------------  --------  ---------
Cd_i_C3v         [-1,0,+1,+2]        [0.12,0.62,0.62]    4         16e
Cd_i_Td_Cd2.83   [-1,0,+1,+2]        [0.75,0.75,0.75]    1         4d
Cd_i_Td_Te2.83   [-1,0,+1,+2]        [0.50,0.50,0.50]    1         4b
Te_i_C3v         [-1,0,+1,+2,+3,+4]  [0.12,0.62,0.62]    4         16e
Te_i_Td_Cd2.83   [-1,0,+1,+2,+3,+4]  [0.75,0.75,0.75]    1         4d
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
        np.testing.assert_array_equal(
            cdte_defect_gen.supercell_matrix, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
        )
        assert structure_matcher.fit(
            prim_struc_wout_oxi * cdte_defect_gen.supercell_matrix, cdte_defect_gen.bulk_supercell
        )

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

        # test some relevant defect attributes
        assert cdte_defect_gen.defects["vacancies"][0].name == "v_Cd"
        assert cdte_defect_gen.defects["vacancies"][0].oxi_state == -2
        assert cdte_defect_gen.defects["vacancies"][0].multiplicity == 1
        assert cdte_defect_gen.defects["vacancies"][0].defect_type == DefectType.Vacancy
        assert cdte_defect_gen.defects["vacancies"][0].structure == cdte_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            cdte_defect_gen.defects["vacancies"][0].defect_structure.lattice.matrix,
            cdte_defect_gen.primitive_structure.lattice.matrix,
        )
        assert cdte_defect_gen.defects["vacancies"][0].defect_site == PeriodicSite(
            "Cd2+", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )
        assert cdte_defect_gen.defects["vacancies"][0].site == PeriodicSite(
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )

        # test defect entries
        assert len(cdte_defect_gen.defect_entries) == 50
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in cdte_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].name == "Cd_i_C3v_0"
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].charge_state == 0
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.defect_type == DefectType.Interstitial
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].wyckoff == "16e"
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.multiplicity == 4
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].sc_defect_frac_coords,
            np.array([0.3125, 0.1875, 0.1875]),
        )
        for defect_name, defect_entry in cdte_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                cdte_defect_gen.bulk_supercell.lattice.matrix,
            )

        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.name == "v_Cd"
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.oxi_state == -2
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.multiplicity == 1
        assert cdte_defect_gen.defect_entries["v_Cd_0"].wyckoff == "4a"
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.defect_type == DefectType.Vacancy
        assert (
            cdte_defect_gen.defect_entries["v_Cd_0"].defect.structure
            == cdte_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            cdte_defect_gen.defect_entries["v_Cd_0"].defect.defect_structure.lattice.matrix,
            cdte_defect_gen.primitive_structure.lattice.matrix,
        )
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.defect_site == PeriodicSite(
            "Cd2+", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.site == PeriodicSite(
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )

        # TODO: test charge states (when charge state algorithm is implemented)
        # test other input structures
        # test as_dict etc methods
        # test saving to and loading from json (and that attributes remain)
        # test that inputting a structure with lattice vectors >10 A doesn't generate a new supercell
        # structure, unless it has less atoms than the input structure
        # test supercell transformation stuff!!

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


class WyckoffTest(unittest.TestCase):
    def test_wyckoff_dict_from_sgn(self):
        for sgn in range(1, 230):  # TODO: Space group 230 doesn't work, will PR!
            wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)
            assert isinstance(wyckoff_dict, dict)
            assert all(isinstance(k, str) for k in wyckoff_dict)
            assert all(isinstance(v, list) for v in wyckoff_dict.values())
