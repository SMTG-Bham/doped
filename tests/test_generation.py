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
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
        assert np.allclose(prim_struc_wout_oxi.lattice.matrix, self.cdte_bulk.lattice.matrix)
        np.testing.assert_allclose(
            cdte_defect_gen.supercell_matrix, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
        )
        assert structure_matcher.fit(
            prim_struc_wout_oxi * cdte_defect_gen.supercell_matrix, cdte_defect_gen.bulk_supercell
        )
        sga = SpacegroupAnalyzer(self.cdte_bulk)
        conv_cdte = sga.get_conventional_standard_structure()
        assert np.allclose(
            (prim_struc_wout_oxi * cdte_defect_gen.supercell_matrix).lattice.matrix,
            (conv_cdte * 2 * np.eye(3)).lattice.matrix,
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

    def test_ytos_supercell_input(self):
        # note that this tests the case of an input structure which is >10 Å in each direction and has
        # more atoms (198) than the pmg supercell (99), so the pmg supercell is used
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            ytos_defect_gen = DefectsGenerator(self.ytos_bulk_supercell)  # Y2Ti2S2O5 supercell
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        ytos_defect_gen_info = (
            """Vacancies    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
-----------  ------------------  ------------------  --------  ---------
v_Y          [-3,-2,-1,0,+1]     [0.67,0.67,0.00]    2         8h
v_Ti         [-4,-3,-2,-1,0,+1]  [0.92,0.92,0.00]    2         8h
v_S          [-1,0,+1,+2]        [0.80,0.80,0.00]    2         8h
v_O_C1       [-1,0,+1,+2]        [0.90,0.40,0.50]    4         16l
v_O_C2h      [-1,0,+1,+2]        [0.00,0.00,0.00]    1         2a

Substitutions    Charge States             Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------------  ------------------  --------  ---------
Y_Ti             [-1,0,+1]                 [0.92,0.92,0.00]    2         8h
Y_S              [-1,0,+1,+2,+3,+4,+5]     [0.80,0.80,0.00]    2         8h
Y_O_C1           [-1,0,+1,+2,+3,+4,+5]     [0.90,0.40,0.50]    4         16l
Y_O_C2h          [-1,0,+1,+2,+3,+4,+5]     [0.00,0.00,0.00]    1         2a
Ti_Y             [-1,0,+1]                 [0.67,0.67,0.00]    2         8h
Ti_S             [-1,0,+1,+2,+3,+4,+5,+6]  [0.80,0.80,0.00]    2         8h
Ti_O_C1          [-1,0,+1,+2,+3,+4,+5,+6]  [0.90,0.40,0.50]    4         16l
Ti_O_C2h         [-1,0,+1,+2,+3,+4,+5,+6]  [0.00,0.00,0.00]    1         2a
S_Y              [-5,-4,-3,-2,-1,0,+1]     [0.67,0.67,0.00]    2         8h
S_Ti             [-6,-5,-4,-3,-2,-1,0,+1]  [0.92,0.92,0.00]    2         8h
S_O_C1           [-1,0,+1]                 [0.90,0.40,0.50]    4         16l
S_O_C2h          [-1,0,+1]                 [0.00,0.00,0.00]    1         2a
O_Y              [-5,-4,-3,-2,-1,0,+1]     [0.67,0.67,0.00]    2         8h
O_Ti             [-6,-5,-4,-3,-2,-1,0,+1]  [0.92,0.92,0.00]    2         8h
O_S              [-1,0,+1]                 [0.80,0.80,0.00]    2         8h

Interstitials    Charge States    Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ---------------  ------------------  --------  ---------
Y_i_C1_S2.03     [-1,0,+1,+2,+3]  [0.36,0.73,0.63]    10        32o
Y_i_C1_Y1.92     [-1,0,+1,+2,+3]  [0.18,0.68,0.50]    4         16l
Y_i_Cs_O1.71     [-1,0,+1,+2,+3]  [0.68,0.04,0.00]    8         16l
Y_i_Cs_O1.92     [-1,0,+1,+2,+3]  [0.42,0.42,0.00]    2         8h
Y_i_Cs_O1.95     [-1,0,+1,+2,+3]  [0.29,0.64,0.00]    8         16l
Y_i_Cs_O2.68     [-1,0,+1,+2,+3]  [0.52,0.52,0.00]    2         8h
Ti_i_C1_S2.03    [-1,0,+1,+2,+3]  [0.36,0.73,0.63]    10        32o
Ti_i_C1_Y1.92    [-1,0,+1,+2,+3]  [0.18,0.68,0.50]    4         16l
Ti_i_Cs_O1.71    [-1,0,+1,+2,+3]  [0.68,0.04,0.00]    8         16l
Ti_i_Cs_O1.92    [-1,0,+1,+2,+3]  [0.42,0.42,0.00]    2         8h
Ti_i_Cs_O1.95    [-1,0,+1,+2,+3]  [0.29,0.64,0.00]    8         16l
Ti_i_Cs_O2.68    [-1,0,+1,+2,+3]  [0.52,0.52,0.00]    2         8h
S_i_C1_S2.03     [-1,0,+1,+2]     [0.36,0.73,0.63]    10        32o
S_i_C1_Y1.92     [-1,0,+1,+2]     [0.18,0.68,0.50]    4         16l
S_i_Cs_O1.71     [-1,0,+1,+2]     [0.68,0.04,0.00]    8         16l
S_i_Cs_O1.92     [-1,0,+1,+2]     [0.42,0.42,0.00]    2         8h
S_i_Cs_O1.95     [-1,0,+1,+2]     [0.29,0.64,0.00]    8         16l
S_i_Cs_O2.68     [-1,0,+1,+2]     [0.52,0.52,0.00]    2         8h
O_i_C1_S2.03     [-2,-1,0,+1]     [0.36,0.73,0.63]    10        32o
O_i_C1_Y1.92     [-2,-1,0,+1]     [0.18,0.68,0.50]    4         16l
O_i_Cs_O1.71     [-2,-1,0,+1]     [0.68,0.04,0.00]    8         16l
O_i_Cs_O1.92     [-2,-1,0,+1]     [0.42,0.42,0.00]    2         8h
O_i_Cs_O1.95     [-2,-1,0,+1]     [0.29,0.64,0.00]    8         16l
O_i_Cs_O2.68     [-2,-1,0,+1]     [0.52,0.52,0.00]    2         8h

\x1B[3mg\x1B[0m_site = Site Multiplicity (in Primitive Unit Cell)\n"""
            "Note that Wyckoff letters can depend on the ordering of elements in the primitive standard "
            "structure (returned by spglib)\n\n"
        )

        assert ytos_defect_gen_info in output

        # test attributes:
        structure_matcher = StructureMatcher()
        prim_struc_wout_oxi = ytos_defect_gen.primitive_structure.copy()
        prim_struc_wout_oxi.remove_oxidation_states()
        assert structure_matcher.fit(  # reduces to primitive, but StructureMatcher still matches
            prim_struc_wout_oxi, self.ytos_bulk_supercell
        )
        assert structure_matcher.fit(
            prim_struc_wout_oxi, self.ytos_bulk_supercell.get_primitive_structure()
        )  # reduces to primitive, but StructureMatcher still matches
        assert not np.allclose(prim_struc_wout_oxi.lattice.matrix, self.ytos_bulk_supercell.lattice.matrix)
        assert not np.allclose(
            ytos_defect_gen.bulk_supercell.lattice.matrix, self.ytos_bulk_supercell.lattice.matrix
        )  # different supercell because Kat YTOS one has 198 atoms but min >10Å one is 99 atoms
        assert structure_matcher.fit(
            ytos_defect_gen.bulk_supercell, self.ytos_bulk_supercell.get_primitive_structure()
        )  # reduces to primitive, but StructureMatcher still matches

        try:
            np.testing.assert_allclose(
                ytos_defect_gen.supercell_matrix, np.array([[0, 3, 3], [3, 0, 3], [0, 1, 0]])
            )
        except AssertionError:  # symmetry equivalent matrices (a, b equivalent for primitive YTOS)
            np.testing.assert_allclose(
                ytos_defect_gen.supercell_matrix, np.array([[0, 3, 3], [3, 0, 3], [1, 0, 0]])
            )

        assert structure_matcher.fit(
            prim_struc_wout_oxi * ytos_defect_gen.supercell_matrix, ytos_defect_gen.bulk_supercell
        )

        # test defects
        assert len(ytos_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(ytos_defect_gen.defects["vacancies"]) == 5
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in ytos_defect_gen.defects["vacancies"]
        )
        assert len(ytos_defect_gen.defects["substitutions"]) == 15
        assert all(
            defect.defect_type == DefectType.Substitution
            for defect in ytos_defect_gen.defects["substitutions"]
        )
        assert len(ytos_defect_gen.defects["interstitials"]) == 24
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in ytos_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in ytos_defect_gen.defects.values()
            for defect in defect_list
        )

        # test some relevant defect attributes
        assert ytos_defect_gen.defects["vacancies"][0].name == "v_Y"
        assert ytos_defect_gen.defects["vacancies"][0].oxi_state == -3
        assert ytos_defect_gen.defects["vacancies"][0].multiplicity == 2
        assert ytos_defect_gen.defects["vacancies"][0].defect_type == DefectType.Vacancy
        assert ytos_defect_gen.defects["vacancies"][0].structure == ytos_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            ytos_defect_gen.defects["vacancies"][0].defect_structure.lattice.matrix,
            ytos_defect_gen.primitive_structure.lattice.matrix,
        )

        # test defect entries
        assert len(ytos_defect_gen.defect_entries) == 221
        assert len(ytos_defect_gen) == 221
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in ytos_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert ytos_defect_gen.defect_entries["O_i_Cs_O2.68_0"].name == "O_i_Cs_O2.68_0"
        assert ytos_defect_gen.defect_entries["O_i_Cs_O2.68_0"].charge_state == 0
        assert ytos_defect_gen.defect_entries["O_i_Cs_O2.68_-2"].charge_state == -2
        assert (
            ytos_defect_gen.defect_entries["O_i_Cs_O2.68_-1"].defect.defect_type == DefectType.Interstitial
        )
        assert ytos_defect_gen.defect_entries["O_i_Cs_O2.68_-1"].wyckoff == "8h"
        assert ytos_defect_gen.defect_entries["O_i_Cs_O2.68_-1"].defect.multiplicity == 2
        try:
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["O_i_Cs_O2.68_0"].sc_defect_frac_coords,
                np.array([0.828227, 0.171773, 0.030638]),
                rtol=1e-2,
            )
        except AssertionError:  # symmetry equivalent matrices (a, b equivalent for primitive YTOS)
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["O_i_Cs_O2.68_0"].sc_defect_frac_coords,
                np.array([0.171773, 0.828227, 0.030638]),
                rtol=1e-2,
            )

        for defect_name, defect_entry in ytos_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                ytos_defect_gen.bulk_supercell.lattice.matrix,
            )

        assert ytos_defect_gen.defect_entries["v_Y_0"].defect.name == "v_Y"
        assert ytos_defect_gen.defect_entries["v_Y_0"].defect.oxi_state == -3
        assert ytos_defect_gen.defect_entries["v_Y_-2"].defect.multiplicity == 2
        assert ytos_defect_gen.defect_entries["v_Y_-2"].wyckoff == "8h"
        assert ytos_defect_gen.defect_entries["v_Y_-2"].defect.defect_type == DefectType.Vacancy
        assert (
            ytos_defect_gen.defect_entries["v_Y_0"].defect.structure == ytos_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            ytos_defect_gen.defect_entries["v_Y_0"].defect.defect_structure.lattice.matrix,
            ytos_defect_gen.primitive_structure.lattice.matrix,
        )

    def test_lmno(self):
        # battery material with a variety of important Wyckoff sites (and the terminology mainly
        # used in this field). Tough to find suitable supercell, goes to 448-atom supercell.
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            lmno_defect_gen = DefectsGenerator(self.lmno_primitive)  # Li2Mn3NiO8 unit cell
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        lmno_defect_gen_info = (
            """Vacancies    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
-----------  ------------------  ------------------  --------  ---------
v_Li         [-1,0,+1]           [0.00,0.00,0.00]    8         8c
v_Mn         [-4,-3,-2,-1,0,+1]  [0.12,0.13,0.62]    12        12d
v_Ni         [-2,-1,0,+1]        [0.12,0.88,0.38]    4         4b
v_O_C1       [-1,0,+1,+2]        [0.10,0.12,0.39]    24        24e
v_O_C3       [-1,0,+1,+2]        [0.12,0.62,0.88]    8         8c

Substitutions    Charge States             Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------------  ------------------  --------  ---------
Li_Mn            [-3,-2,-1,0,+1]           [0.12,0.13,0.62]    12        12d
Li_Ni            [-1,0,+1]                 [0.12,0.88,0.38]    4         4b
Li_O_C1          [-1,0,+1,+2,+3]           [0.10,0.12,0.39]    24        24e
Li_O_C3          [-1,0,+1,+2,+3]           [0.12,0.62,0.88]    8         8c
Mn_Li            [-1,0,+1]                 [0.00,0.00,0.00]    8         8c
Mn_Ni            [-1,0,+1]                 [0.12,0.88,0.38]    4         4b
Mn_O_C1          [-1,0,+1,+2,+3,+4]        [0.10,0.12,0.39]    24        24e
Mn_O_C3          [-1,0,+1,+2,+3,+4]        [0.12,0.62,0.88]    8         8c
Ni_Li            [-1,0,+1]                 [0.00,0.00,0.00]    8         8c
Ni_Mn            [-2,-1,0,+1]              [0.12,0.13,0.62]    12        12d
Ni_O_C1          [-1,0,+1,+2,+3,+4]        [0.10,0.12,0.39]    24        24e
Ni_O_C3          [-1,0,+1,+2,+3,+4]        [0.12,0.62,0.88]    8         8c
O_Li             [-3,-2,-1,0,+1]           [0.00,0.00,0.00]    8         8c
O_Mn             [-6,-5,-4,-3,-2,-1,0,+1]  [0.12,0.13,0.62]    12        12d
O_Ni             [-4,-3,-2,-1,0,+1]        [0.12,0.88,0.38]    4         4b

Interstitials    Charge States    Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ---------------  ------------------  --------  ---------
Li_i_C1_Li1.75   [-1,0,+1]        [0.69,0.05,0.55]    24        24e
Li_i_C1_O1.72    [-1,0,+1]        [0.73,1.00,0.50]    24        24e
Li_i_C1_O1.78    [-1,0,+1]        [0.50,0.01,0.73]    24        24e
Li_i_C2_Li1.83   [-1,0,+1]        [0.58,0.38,0.83]    24        12d
Li_i_C2_Li1.84   [-1,0,+1]        [0.65,0.13,0.60]    12        12d
Li_i_C2_Li1.86   [-1,0,+1]        [0.59,0.12,0.66]    12        12d
Li_i_C3          [-1,0,+1]        [1.00,0.00,0.50]    8         8c
Mn_i_C1_Li1.75   [-1,0,+1,+2,+3]  [0.69,0.05,0.55]    24        24e
Mn_i_C1_O1.72    [-1,0,+1,+2,+3]  [0.73,1.00,0.50]    24        24e
Mn_i_C1_O1.78    [-1,0,+1,+2,+3]  [0.50,0.01,0.73]    24        24e
Mn_i_C2_Li1.83   [-1,0,+1,+2,+3]  [0.58,0.38,0.83]    24        12d
Mn_i_C2_Li1.84   [-1,0,+1,+2,+3]  [0.65,0.13,0.60]    12        12d
Mn_i_C2_Li1.86   [-1,0,+1,+2,+3]  [0.59,0.12,0.66]    12        12d
Mn_i_C3          [-1,0,+1,+2,+3]  [1.00,0.00,0.50]    8         8c
Ni_i_C1_Li1.75   [-1,0,+1,+2]     [0.69,0.05,0.55]    24        24e
Ni_i_C1_O1.72    [-1,0,+1,+2]     [0.73,1.00,0.50]    24        24e
Ni_i_C1_O1.78    [-1,0,+1,+2]     [0.50,0.01,0.73]    24        24e
Ni_i_C2_Li1.83   [-1,0,+1,+2]     [0.58,0.38,0.83]    24        12d
Ni_i_C2_Li1.84   [-1,0,+1,+2]     [0.65,0.13,0.60]    12        12d
Ni_i_C2_Li1.86   [-1,0,+1,+2]     [0.59,0.12,0.66]    12        12d
Ni_i_C3          [-1,0,+1,+2]     [1.00,0.00,0.50]    8         8c
O_i_C1_Li1.75    [-2,-1,0,+1]     [0.69,0.05,0.55]    24        24e
O_i_C1_O1.72     [-2,-1,0,+1]     [0.73,1.00,0.50]    24        24e
O_i_C1_O1.78     [-2,-1,0,+1]     [0.50,0.01,0.73]    24        24e
O_i_C2_Li1.83    [-2,-1,0,+1]     [0.58,0.38,0.83]    24        12d
O_i_C2_Li1.84    [-2,-1,0,+1]     [0.65,0.13,0.60]    12        12d
O_i_C2_Li1.86    [-2,-1,0,+1]     [0.59,0.12,0.66]    12        12d
O_i_C3           [-2,-1,0,+1]     [1.00,0.00,0.50]    8         8c

\x1B[3mg\x1B[0m_site = Site Multiplicity (in Primitive Unit Cell)\n"""
            "Note that Wyckoff letters can depend on the ordering of elements in the primitive standard "
            "structure (returned by spglib)\n\n"
        )

        assert lmno_defect_gen_info in output

        # test attributes:
        structure_matcher = StructureMatcher()
        prim_struc_wout_oxi = lmno_defect_gen.primitive_structure.copy()
        prim_struc_wout_oxi.remove_oxidation_states()
        assert structure_matcher.fit(prim_struc_wout_oxi, self.lmno_primitive)
        assert structure_matcher.fit(
            prim_struc_wout_oxi, lmno_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches
        assert np.allclose(prim_struc_wout_oxi.lattice.matrix, self.lmno_primitive.lattice.matrix)

        np.testing.assert_allclose(
            lmno_defect_gen.supercell_matrix, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        )

        assert structure_matcher.fit(
            prim_struc_wout_oxi * lmno_defect_gen.supercell_matrix, lmno_defect_gen.bulk_supercell
        )

        # test defects
        assert len(lmno_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(lmno_defect_gen.defects["vacancies"]) == 5
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in lmno_defect_gen.defects["vacancies"]
        )
        assert len(lmno_defect_gen.defects["substitutions"]) == 15
        assert all(
            defect.defect_type == DefectType.Substitution
            for defect in lmno_defect_gen.defects["substitutions"]
        )
        assert len(lmno_defect_gen.defects["interstitials"]) == 28
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in lmno_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in lmno_defect_gen.defects.values()
            for defect in defect_list
        )

        # test defect entries
        assert len(lmno_defect_gen.defect_entries) == 207
        assert len(lmno_defect_gen) == 207
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in lmno_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].name == "Ni_i_C2_Li1.84_+2"
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].charge_state == +2
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect.defect_type
            == DefectType.Interstitial
        )
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].wyckoff == "12d"
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect.multiplicity == 12
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].sc_defect_frac_coords,
            np.array([0.32536836, 0.0625, 0.29963164]),
            rtol=1e-2,
        )

        for defect_name, defect_entry in lmno_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                lmno_defect_gen.bulk_supercell.lattice.matrix,
            )

    def test_zns_non_diagonal_supercell(self):
        # test inputting a non-diagonal supercell structure -> correct primitive structure
        # determined and reasonable supercell generated
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            zns_defect_gen = DefectsGenerator(self.non_diagonal_ZnS)  # ZnS non-diagonal supercell
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        zns_defect_gen_info = (
            """Vacancies    Charge States    Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
-----------  ---------------  ------------------  --------  ---------
v_Zn         [-2,-1,0,+1]     [0.00,0.00,0.00]    1         4a
v_S          [-1,0,+1,+2]     [0.25,0.25,0.25]    1         4c

Substitutions    Charge States       Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ------------------  ------------------  --------  ---------
Zn_S             [-1,0,+1,+2,+3,+4]  [0.25,0.25,0.25]    1         4c
S_Zn             [-4,-3,-2,-1,0,+1]  [0.00,0.00,0.00]    1         4a

Interstitials    Charge States    Unit Cell Coords    \x1B[3mg\x1B[0m_site    Wyckoff
---------------  ---------------  ------------------  --------  ---------
Zn_i_C3v         [-1,0,+1,+2]     [0.63,0.63,0.63]    4         16e
Zn_i_Td_S2.35    [-1,0,+1,+2]     [0.50,0.50,0.50]    1         4b
Zn_i_Td_Zn2.35   [-1,0,+1,+2]     [0.75,0.75,0.75]    1         4d
S_i_C3v          [-1,0,+1,+2]     [0.63,0.63,0.63]    4         16e
S_i_Td_S2.35     [-1,0,+1,+2]     [0.50,0.50,0.50]    1         4b
S_i_Td_Zn2.35    [-1,0,+1,+2]     [0.75,0.75,0.75]    1         4d

\x1B[3mg\x1B[0m_site = Site Multiplicity (in Primitive Unit Cell)\n"""
            "Note that Wyckoff letters can depend on the ordering of elements in the primitive standard "
            "structure (returned by spglib)\n\n"
        )

        assert zns_defect_gen_info in output

        # test attributes:
        structure_matcher = StructureMatcher()
        prim_struc_wout_oxi = zns_defect_gen.primitive_structure.copy()
        prim_struc_wout_oxi.remove_oxidation_states()
        assert structure_matcher.fit(prim_struc_wout_oxi, self.non_diagonal_ZnS)
        assert structure_matcher.fit(
            prim_struc_wout_oxi, zns_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches (but below lattice doesn't match)
        assert not np.allclose(prim_struc_wout_oxi.lattice.matrix, self.non_diagonal_ZnS.lattice.matrix)

        np.testing.assert_allclose(
            zns_defect_gen.supercell_matrix, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
        )

        assert structure_matcher.fit(
            prim_struc_wout_oxi * zns_defect_gen.supercell_matrix, zns_defect_gen.bulk_supercell
        )

        # test defects
        assert len(zns_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(zns_defect_gen.defects["vacancies"]) == 2
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in zns_defect_gen.defects["vacancies"]
        )
        assert len(zns_defect_gen.defects["substitutions"]) == 2
        assert all(
            defect.defect_type == DefectType.Substitution
            for defect in zns_defect_gen.defects["substitutions"]
        )
        assert len(zns_defect_gen.defects["interstitials"]) == 6
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in zns_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in zns_defect_gen.defects.values()
            for defect in defect_list
        )

        # test defect entries
        assert len(zns_defect_gen.defect_entries) == 44
        assert len(zns_defect_gen) == 44
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in zns_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].name == "S_i_Td_S2.35_+2"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].charge_state == +2
        assert (
            zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect.defect_type == DefectType.Interstitial
        )
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].wyckoff == "4b"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect.multiplicity == 1
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].sc_defect_frac_coords,
            np.array([0.25, 0.25, 0.25]),
            rtol=1e-2,
        )

        for defect_name, defect_entry in zns_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                zns_defect_gen.bulk_supercell.lattice.matrix,
            )


class WyckoffTest(unittest.TestCase):
    def test_wyckoff_dict_from_sgn(self):
        for sgn in range(1, 231):
            wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)
            assert isinstance(wyckoff_dict, dict)
            assert all(isinstance(k, str) for k in wyckoff_dict)
            assert all(isinstance(v, list) for v in wyckoff_dict.values())
