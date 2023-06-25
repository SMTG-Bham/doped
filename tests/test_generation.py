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
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from doped.generation import DefectsGenerator, get_wyckoff_dict_from_sgn


class DefectsGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.prim_cdte = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        sga = SpacegroupAnalyzer(self.prim_cdte)
        self.conv_cdte = sga.get_conventional_standard_structure()
        self.cdte_bulk_supercell = self.conv_cdte * 2 * np.eye(3)
        self.cdte_defect_gen_info = (
            """Vacancies    Charge States    Conv. Cell Coords    Wyckoff
-----------  ---------------  -------------------  ---------
v_Cd         [-2,-1,0,+1]     [0.000,0.000,0.000]  4a
v_Te         [-1,0,+1,+2]     [0.750,0.750,0.750]  4d

Substitutions    Charge States       Conv. Cell Coords    Wyckoff
---------------  ------------------  -------------------  ---------
Cd_Te            [-1,0,+1,+2,+3,+4]  [0.750,0.750,0.750]  4d
Te_Cd            [-4,-3,-2,-1,0,+1]  [0.000,0.000,0.000]  4a

Interstitials    Charge States       Conv. Cell Coords    Wyckoff
---------------  ------------------  -------------------  ---------
Cd_i_C3v         [-1,0,+1,+2]        [0.375,0.375,0.375]  16e
Cd_i_Td_Cd2.83   [-1,0,+1,+2]        [0.250,0.250,0.250]  4c
Cd_i_Td_Te2.83   [-1,0,+1,+2]        [0.500,0.500,0.500]  4b
Te_i_C3v         [-1,0,+1,+2,+3,+4]  [0.375,0.375,0.375]  16e
Te_i_Td_Cd2.83   [-1,0,+1,+2,+3,+4]  [0.250,0.250,0.250]  4c
Te_i_Td_Te2.83   [-1,0,+1,+2,+3,+4]  [0.500,0.500,0.500]  4b
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of CdTe.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.ytos_bulk_supercell = Structure.from_file(f"{self.example_dir}/YTOS/Bulk/POSCAR")

        self.ytos_defect_gen_info = (
            """Vacancies    Charge States       Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Y          [-3,-2,-1,0,+1]     [0.000,0.000,0.334]  4e
v_Ti         [-4,-3,-2,-1,0,+1]  [0.000,0.000,0.078]  4e
v_S          [-1,0,+1,+2]        [0.000,0.000,0.205]  4e
v_O_C2v      [-1,0,+1,+2]        [0.000,0.500,0.099]  8g
v_O_D4h      [-1,0,+1,+2]        [0.000,0.000,0.000]  2a

Substitutions    Charge States             Conv. Cell Coords    Wyckoff
---------------  ------------------------  -------------------  ---------
Y_Ti             [-1,0,+1]                 [0.000,0.000,0.078]  4e
Y_S              [-1,0,+1,+2,+3,+4,+5]     [0.000,0.000,0.205]  4e
Y_O_C2v          [-1,0,+1,+2,+3,+4,+5]     [0.000,0.500,0.099]  8g
Y_O_D4h          [-1,0,+1,+2,+3,+4,+5]     [0.000,0.000,0.000]  2a
Ti_Y             [-1,0,+1]                 [0.000,0.000,0.334]  4e
Ti_S             [-1,0,+1,+2,+3,+4,+5,+6]  [0.000,0.000,0.205]  4e
Ti_O_C2v         [-1,0,+1,+2,+3,+4,+5,+6]  [0.000,0.500,0.099]  8g
Ti_O_D4h         [-1,0,+1,+2,+3,+4,+5,+6]  [0.000,0.000,0.000]  2a
S_Y              [-5,-4,-3,-2,-1,0,+1]     [0.000,0.000,0.334]  4e
S_Ti             [-6,-5,-4,-3,-2,-1,0,+1]  [0.000,0.000,0.078]  4e
S_O_C2v          [-1,0,+1]                 [0.000,0.500,0.099]  8g
S_O_D4h          [-1,0,+1]                 [0.000,0.000,0.000]  2a
O_Y              [-5,-4,-3,-2,-1,0,+1]     [0.000,0.000,0.334]  4e
O_Ti             [-6,-5,-4,-3,-2,-1,0,+1]  [0.000,0.000,0.078]  4e
O_S              [-1,0,+1]                 [0.000,0.000,0.205]  4e

Interstitials    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Y_i_C2v          [-1,0,+1,+2,+3]  [0.000,0.500,0.184]  8g
Y_i_C4v_O1.92    [-1,0,+1,+2,+3]  [0.000,0.000,0.418]  4e
Y_i_C4v_O2.68    [-1,0,+1,+2,+3]  [0.000,0.000,0.485]  4e
Y_i_Cs_O1.71     [-1,0,+1,+2,+3]  [0.181,0.181,0.143]  16m
Y_i_Cs_O1.95     [-1,0,+1,+2,+3]  [0.325,0.325,0.039]  16m
Y_i_D2d          [-1,0,+1,+2,+3]  [0.000,0.500,0.250]  4d
Ti_i_C2v         [-1,0,+1,+2,+3]  [0.000,0.500,0.184]  8g
Ti_i_C4v_O1.92   [-1,0,+1,+2,+3]  [0.000,0.000,0.418]  4e
Ti_i_C4v_O2.68   [-1,0,+1,+2,+3]  [0.000,0.000,0.485]  4e
Ti_i_Cs_O1.71    [-1,0,+1,+2,+3]  [0.181,0.181,0.143]  16m
Ti_i_Cs_O1.95    [-1,0,+1,+2,+3]  [0.325,0.325,0.039]  16m
Ti_i_D2d         [-1,0,+1,+2,+3]  [0.000,0.500,0.250]  4d
S_i_C2v          [-1,0,+1,+2]     [0.000,0.500,0.184]  8g
S_i_C4v_O1.92    [-1,0,+1,+2]     [0.000,0.000,0.418]  4e
S_i_C4v_O2.68    [-1,0,+1,+2]     [0.000,0.000,0.485]  4e
S_i_Cs_O1.71     [-1,0,+1,+2]     [0.181,0.181,0.143]  16m
S_i_Cs_O1.95     [-1,0,+1,+2]     [0.325,0.325,0.039]  16m
S_i_D2d          [-1,0,+1,+2]     [0.000,0.500,0.250]  4d
O_i_C2v          [-2,-1,0,+1]     [0.000,0.500,0.184]  8g
O_i_C4v_O1.92    [-2,-1,0,+1]     [0.000,0.000,0.418]  4e
O_i_C4v_O2.68    [-2,-1,0,+1]     [0.000,0.000,0.485]  4e
O_i_Cs_O1.71     [-2,-1,0,+1]     [0.181,0.181,0.143]  16m
O_i_Cs_O1.95     [-2,-1,0,+1]     [0.325,0.325,0.039]  16m
O_i_D2d          [-2,-1,0,+1]     [0.000,0.500,0.250]  4d
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 2 formula unit(s) of Y2Ti2S2O5.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.lmno_defect_gen_info = (
            """Vacancies    Charge States       Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Li         [-1,0,+1]           [0.004,0.004,0.004]  8c
v_Mn         [-4,-3,-2,-1,0,+1]  [0.121,0.129,0.625]  12d
v_Ni         [-2,-1,0,+1]        [0.625,0.625,0.625]  4b
v_O_C1       [-1,0,+1,+2]        [0.101,0.124,0.392]  24e
v_O_C3       [-1,0,+1,+2]        [0.384,0.384,0.384]  8c

Substitutions    Charge States             Conv. Cell Coords    Wyckoff
---------------  ------------------------  -------------------  ---------
Li_Mn            [-3,-2,-1,0,+1]           [0.121,0.129,0.625]  12d
Li_Ni            [-1,0,+1]                 [0.625,0.625,0.625]  4b
Li_O_C1          [-1,0,+1,+2,+3]           [0.101,0.124,0.392]  24e
Li_O_C3          [-1,0,+1,+2,+3]           [0.384,0.384,0.384]  8c
Mn_Li            [-1,0,+1]                 [0.004,0.004,0.004]  8c
Mn_Ni            [-1,0,+1]                 [0.625,0.625,0.625]  4b
Mn_O_C1          [-1,0,+1,+2,+3,+4]        [0.101,0.124,0.392]  24e
Mn_O_C3          [-1,0,+1,+2,+3,+4]        [0.384,0.384,0.384]  8c
Ni_Li            [-1,0,+1]                 [0.004,0.004,0.004]  8c
Ni_Mn            [-2,-1,0,+1]              [0.121,0.129,0.625]  12d
Ni_O_C1          [-1,0,+1,+2,+3,+4]        [0.101,0.124,0.392]  24e
Ni_O_C3          [-1,0,+1,+2,+3,+4]        [0.384,0.384,0.384]  8c
O_Li             [-3,-2,-1,0,+1]           [0.004,0.004,0.004]  8c
O_Mn             [-6,-5,-4,-3,-2,-1,0,+1]  [0.121,0.129,0.625]  12d
O_Ni             [-4,-3,-2,-1,0,+1]        [0.625,0.625,0.625]  4b

Interstitials    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Li_i_C1_Li1.75   [-1,0,+1]        [0.199,0.303,0.444]  24e
Li_i_C1_O1.72    [-1,0,+1]        [0.001,0.770,0.002]  24e
Li_i_C1_O1.78    [-1,0,+1]        [0.017,0.261,0.250]  24e
Li_i_C2_Li1.83   [-1,0,+1]        [0.077,0.125,0.173]  12d
Li_i_C2_Li1.84   [-1,0,+1]        [0.151,0.375,0.401]  12d
Li_i_C2_Li1.86   [-1,0,+1]        [0.085,0.375,0.335]  12d
Li_i_C3          [-1,0,+1]        [0.497,0.497,0.497]  8c
Mn_i_C1_Li1.75   [-1,0,+1,+2,+3]  [0.199,0.303,0.444]  24e
Mn_i_C1_O1.72    [-1,0,+1,+2,+3]  [0.001,0.770,0.002]  24e
Mn_i_C1_O1.78    [-1,0,+1,+2,+3]  [0.017,0.261,0.250]  24e
Mn_i_C2_Li1.83   [-1,0,+1,+2,+3]  [0.077,0.125,0.173]  12d
Mn_i_C2_Li1.84   [-1,0,+1,+2,+3]  [0.151,0.375,0.401]  12d
Mn_i_C2_Li1.86   [-1,0,+1,+2,+3]  [0.085,0.375,0.335]  12d
Mn_i_C3          [-1,0,+1,+2,+3]  [0.497,0.497,0.497]  8c
Ni_i_C1_Li1.75   [-1,0,+1,+2]     [0.199,0.303,0.444]  24e
Ni_i_C1_O1.72    [-1,0,+1,+2]     [0.001,0.770,0.002]  24e
Ni_i_C1_O1.78    [-1,0,+1,+2]     [0.017,0.261,0.250]  24e
Ni_i_C2_Li1.83   [-1,0,+1,+2]     [0.077,0.125,0.173]  12d
Ni_i_C2_Li1.84   [-1,0,+1,+2]     [0.151,0.375,0.401]  12d
Ni_i_C2_Li1.86   [-1,0,+1,+2]     [0.085,0.375,0.335]  12d
Ni_i_C3          [-1,0,+1,+2]     [0.497,0.497,0.497]  8c
O_i_C1_Li1.75    [-2,-1,0,+1]     [0.199,0.303,0.444]  24e
O_i_C1_O1.72     [-2,-1,0,+1]     [0.001,0.770,0.002]  24e
O_i_C1_O1.78     [-2,-1,0,+1]     [0.017,0.261,0.250]  24e
O_i_C2_Li1.83    [-2,-1,0,+1]     [0.077,0.125,0.173]  12d
O_i_C2_Li1.84    [-2,-1,0,+1]     [0.151,0.375,0.401]  12d
O_i_C2_Li1.86    [-2,-1,0,+1]     [0.085,0.375,0.335]  12d
O_i_C3           [-2,-1,0,+1]     [0.497,0.497,0.497]  8c
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of Li2Mn3NiO8.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")
        self.zns_defect_gen_info = (
            """Vacancies    Charge States    Conv. Cell Coords    Wyckoff
-----------  ---------------  -------------------  ---------
v_Zn         [-2,-1,0,+1]     [0.000,0.000,0.000]  4a
v_S          [-1,0,+1,+2]     [0.250,0.250,0.250]  4c

Substitutions    Charge States       Conv. Cell Coords    Wyckoff
---------------  ------------------  -------------------  ---------
Zn_S             [-1,0,+1,+2,+3,+4]  [0.250,0.250,0.250]  4c
S_Zn             [-4,-3,-2,-1,0,+1]  [0.000,0.000,0.000]  4a

Interstitials    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Zn_i_C3v         [-1,0,+1,+2]     [0.625,0.625,0.625]  16e
Zn_i_Td_S2.35    [-1,0,+1,+2]     [0.500,0.500,0.500]  4b
Zn_i_Td_Zn2.35   [-1,0,+1,+2]     [0.750,0.750,0.750]  4d
S_i_C3v          [-1,0,+1,+2]     [0.625,0.625,0.625]  16e
S_i_Td_S2.35     [-1,0,+1,+2]     [0.500,0.500,0.500]  4b
S_i_Td_Zn2.35    [-1,0,+1,+2]     [0.750,0.750,0.750]  4d
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of ZnS.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        # TODO: test charge states (when charge state algorithm is implemented)
        # test other input structures (defective CdTe supercell, primitive one-atom Cu, CuAg)
        # test as_dict etc methods
        # test saving to and loading from json (and that _all_ attributes remain)
        # test all input parameters; extrinsic, interstitial_coords, interstitial/supercell gen kwargs,
        # target_frac_coords setting...
        # test input parameters used as attributes

    def cdte_defect_gen_check(self, cdte_defect_gen):
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(cdte_defect_gen.primitive_structure, self.prim_cdte)
        assert np.allclose(
            cdte_defect_gen.primitive_structure.lattice.matrix, self.prim_cdte.lattice.matrix
        )  # same lattice
        np.testing.assert_allclose(
            cdte_defect_gen.supercell_matrix, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
        )
        assert structure_matcher.fit(
            cdte_defect_gen.primitive_structure * cdte_defect_gen.supercell_matrix,
            cdte_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (cdte_defect_gen.primitive_structure * cdte_defect_gen.supercell_matrix).lattice.matrix,
            self.cdte_bulk_supercell.lattice.matrix,
        )
        assert structure_matcher.fit(cdte_defect_gen.conventional_structure, self.prim_cdte)
        assert np.allclose(
            cdte_defect_gen.conventional_structure.lattice.matrix,
            self.conv_cdte.lattice.matrix,
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
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.wyckoff == "16e"
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].conv_cell_frac_coords,
            np.array([0.375, 0.375, 0.375]),
        )
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.conv_cell_frac_coords,
            np.array([0.375, 0.375, 0.375]),
        )
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.multiplicity == 4
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].sc_defect_frac_coords,
            np.array([0.3125, 0.4375, 0.4375]),
        )
        np.testing.assert_allclose(
            cdte_defect_gen["Cd_i_C3v_0"].defect_supercell_site.frac_coords,
            np.array([0.3125, 0.4375, 0.4375]),  # closest to middle of supercell
        )
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect_supercell_site.specie.symbol == "Cd"
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.site.frac_coords,
            np.array([0.625, 0.625, 0.625]),
        )

        for defect_name, defect_entry in cdte_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                cdte_defect_gen.bulk_supercell.lattice.matrix,
            )
            assert np.allclose(
                defect_entry.conventional_structure.lattice.matrix,
                self.conv_cdte.lattice.matrix,
            )
            assert np.allclose(
                defect_entry.defect.conventional_structure.lattice.matrix,
                self.conv_cdte.lattice.matrix,
            )
            # get minimum distance of defect_entry.conv_cell_frac_coords to any site in
            # defect_entry.conventional_structure
            distances = []
            for site in defect_entry.conventional_structure:
                distances.append(
                    site.distance_and_image_from_frac_coords(defect_entry.conv_cell_frac_coords)[0]
                )
            assert min(np.array(distances)[np.array(distances) > 0.001]) > 0.9  # default min_dist = 0.9
            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, cdte_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 4 == int(defect_entry.wyckoff[:-1])
            assert defect_entry.defect_supercell_site

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
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["v_Cd_0"].conv_cell_frac_coords,
            np.array([0, 0, 0]),
        )
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["v_Cd_0"].defect.conv_cell_frac_coords,
            np.array([0, 0, 0]),
        )
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.multiplicity == 1
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["v_Cd_0"].sc_defect_frac_coords,
            np.array([0.5, 0.5, 0.5]),
        )
        np.testing.assert_allclose(
            cdte_defect_gen["v_Cd_0"].defect_supercell_site.frac_coords,
            np.array([0.5, 0.5, 0.5]),  # closest to middle of supercell
        )

    def test_defects_generator_cdte(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cdte_defect_gen = DefectsGenerator(self.prim_cdte)
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cdte_defect_gen_info in output  # matches expected 4b & 4d Wyckoff letters for Td
        # interstitials (https://doi.org/10.1016/j.solener.2013.12.017)

        self.cdte_defect_gen_check(cdte_defect_gen)

    def test_defects_generator_cdte_supercell_input(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cdte_defect_gen = DefectsGenerator(self.cdte_bulk_supercell)
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cdte_defect_gen_info in output

        self.cdte_defect_gen_check(cdte_defect_gen)

    def test_cdte_no_generate_supercell_supercell_input(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cdte_defect_gen = DefectsGenerator(self.cdte_bulk_supercell, generate_supercell=False)
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cdte_defect_gen_info in output

        self.cdte_defect_gen_check(cdte_defect_gen)

    @patch("sys.stdout", new_callable=StringIO)
    def test_generator_tqdm(self, mock_stdout):
        with patch("doped.generation.tqdm") as mocked_tqdm:
            mocked_instance = mocked_tqdm.return_value
            DefectsGenerator(self.prim_cdte)
            mocked_tqdm.assert_called_once()
            mocked_tqdm.assert_called_with(total=100)
            mocked_instance.set_description.assert_any_call("Getting primitive structure")
            mocked_instance.set_description.assert_any_call("Generating vacancies")
            mocked_instance.set_description.assert_any_call("Generating substitutions")
            mocked_instance.set_description.assert_any_call("Generating interstitials")
            mocked_instance.set_description.assert_any_call("Generating simulation supercell")
            mocked_instance.set_description.assert_any_call("Determining Wyckoff sites")
            mocked_instance.set_description.assert_any_call("Generating DefectEntry objects")

    def ytos_defect_gen_check(self, ytos_defect_gen, generate_supercell=True):
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(  # reduces to primitive, but StructureMatcher still matches
            ytos_defect_gen.primitive_structure, self.ytos_bulk_supercell
        )
        assert structure_matcher.fit(
            ytos_defect_gen.primitive_structure, self.ytos_bulk_supercell.get_primitive_structure()
        )  # reduces to primitive, but StructureMatcher still matches
        assert not np.allclose(
            ytos_defect_gen.primitive_structure.lattice.matrix, self.ytos_bulk_supercell.lattice.matrix
        )
        assert np.allclose(
            ytos_defect_gen.primitive_structure.volume,
            self.ytos_bulk_supercell.get_primitive_structure().volume,
        )
        assert structure_matcher.fit(
            ytos_defect_gen.bulk_supercell, self.ytos_bulk_supercell.get_primitive_structure()
        )  # reduces to primitive, but StructureMatcher still matches

        if generate_supercell:
            try:
                np.testing.assert_allclose(
                    ytos_defect_gen.supercell_matrix, np.array([[0, 3, 3], [3, 0, 3], [0, 1, 0]])
                )
            except AssertionError:  # symmetry equivalent matrices (a, b equivalent for primitive YTOS)
                np.testing.assert_allclose(
                    ytos_defect_gen.supercell_matrix, np.array([[0, 3, 3], [3, 0, 3], [1, 0, 0]])
                )
            assert not np.allclose(
                ytos_defect_gen.bulk_supercell.lattice.matrix, self.ytos_bulk_supercell.lattice.matrix
            )  # different supercell because Kat YTOS one has 198 atoms but min >10Å one is 99 atoms
        else:
            np.testing.assert_allclose(
                ytos_defect_gen.supercell_matrix, np.array([[0, 3, 3], [3, 0, 3], [1, 1, 0]])
            )
            assert np.allclose(
                ytos_defect_gen.bulk_supercell.lattice.matrix, self.ytos_bulk_supercell.lattice.matrix
            )

        assert structure_matcher.fit(
            ytos_defect_gen.primitive_structure * ytos_defect_gen.supercell_matrix,
            ytos_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (ytos_defect_gen.primitive_structure * ytos_defect_gen.supercell_matrix).lattice.matrix,
            ytos_defect_gen.bulk_supercell.lattice.matrix,
        )

        assert structure_matcher.fit(ytos_defect_gen.conventional_structure, self.ytos_bulk_supercell)
        sga = SpacegroupAnalyzer(self.ytos_bulk_supercell)
        assert np.allclose(
            ytos_defect_gen.conventional_structure.lattice.matrix,
            sga.get_conventional_standard_structure().lattice.matrix,
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
        assert np.allclose(
            ytos_defect_gen.defects["vacancies"][0].site.frac_coords,
            np.array([0.6661, 0.6661, 0.0]),
            atol=1e-3,
        )

        # test defect entries
        assert len(ytos_defect_gen.defect_entries) == 221
        assert len(ytos_defect_gen) == 221
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in ytos_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].name == "O_i_D2d_-1"
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].charge_state == -1
        assert ytos_defect_gen.defect_entries["O_i_D2d_-2"].charge_state == -2
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.defect_type == DefectType.Interstitial
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].wyckoff == "4d"
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.wyckoff == "4d"
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.multiplicity == 2
        if generate_supercell:
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["O_i_D2d_0"].sc_defect_frac_coords,
                np.array([0.41667, 0.41667, 0.5]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["O_i_D2d_0"].defect_supercell_site.frac_coords,
                np.array([0.41667, 0.41667, 0.5]),
                rtol=1e-3,
            )
        else:
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["O_i_D2d_0"].sc_defect_frac_coords,
                np.array([0.3333, 0.5, 0.25]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["O_i_D2d_0"].defect_supercell_site.frac_coords,
                np.array([0.3333, 0.5, 0.25]),
                rtol=1e-3,
            )
        assert ytos_defect_gen.defect_entries["O_i_D2d_0"].defect_supercell_site.specie.symbol == "O"

        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["O_i_D2d_-1"].conv_cell_frac_coords,
            np.array([0.000, 0.500, 0.25]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.conv_cell_frac_coords,
            np.array([0.000, 0.500, 0.25]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.site.frac_coords,
            np.array([0.25, 0.75, 0.5]),
            atol=1e-3,
        )

        for defect_name, defect_entry in ytos_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                ytos_defect_gen.bulk_supercell.lattice.matrix,
            )
            assert np.allclose(
                defect_entry.conventional_structure.lattice.matrix,
                sga.get_conventional_standard_structure().lattice.matrix,
            )
            assert np.allclose(
                defect_entry.defect.conventional_structure.lattice.matrix,
                sga.get_conventional_standard_structure().lattice.matrix,
            )
            # get minimum distance of defect_entry.conv_cell_frac_coords to any site in
            # defect_entry.conventional_structure
            distances = []
            for site in defect_entry.conventional_structure:
                distances.append(
                    site.distance_and_image_from_frac_coords(defect_entry.conv_cell_frac_coords)[0]
                )
            assert min(np.array(distances)[np.array(distances) > 0.001]) > 0.9  # default min_dist = 0.9
            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, ytos_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 2 == int(defect_entry.wyckoff[:-1])
            assert defect_entry.defect_supercell_site

        assert ytos_defect_gen.defect_entries["v_Y_0"].defect.name == "v_Y"
        assert ytos_defect_gen.defect_entries["v_Y_0"].defect.oxi_state == -3
        assert ytos_defect_gen.defect_entries["v_Y_-2"].defect.multiplicity == 2
        assert ytos_defect_gen.defect_entries["v_Y_-2"].wyckoff == "4e"
        assert ytos_defect_gen.defect_entries["v_Y_-2"].defect.defect_type == DefectType.Vacancy
        assert (
            ytos_defect_gen.defect_entries["v_Y_0"].defect.structure == ytos_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            ytos_defect_gen.defect_entries["v_Y_0"].defect.defect_structure.lattice.matrix,
            ytos_defect_gen.primitive_structure.lattice.matrix,
        )

        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["v_Y_0"].conv_cell_frac_coords,
            np.array([0, 0, 0.334]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["v_Y_0"].defect.conv_cell_frac_coords,
            np.array([0, 0, 0.334]),
            atol=1e-3,
        )
        if generate_supercell:
            x = 0.4446
            y = 0.5554
            try:
                np.testing.assert_allclose(
                    ytos_defect_gen.defect_entries["v_Y_0"].sc_defect_frac_coords,
                    np.array([x, y, 0.3322]),
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    ytos_defect_gen["v_Y_0"].defect_supercell_site.frac_coords,
                    np.array([x, y, 0.3322]),  # closest to middle of supercell
                    atol=1e-4,
                )
            except AssertionError:
                np.testing.assert_allclose(
                    ytos_defect_gen.defect_entries["v_Y_0"].sc_defect_frac_coords,
                    np.array([y, x, 0.3322]),  # closest to middle of supercell
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    ytos_defect_gen["v_Y_0"].defect_supercell_site.frac_coords,
                    np.array([y, x, 0.3322]),  # closest to middle of supercell
                    atol=1e-4,
                )
        else:
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["v_Y_0"].sc_defect_frac_coords,
                np.array([0.3333, 0.3333, 0.3339]),
                atol=1e-4,
            )
            np.testing.assert_allclose(
                ytos_defect_gen["v_Y_0"].defect_supercell_site.frac_coords,
                np.array([0.3333, 0.3333, 0.3339]),  # closest to middle of supercell
                atol=1e-4,
            )
        assert np.allclose(
            ytos_defect_gen["v_Y_0"].defect.site.frac_coords,
            np.array([0.6661, 0.6661, 0.0]),
            atol=1e-3,
        )

    def test_ytos_supercell_input(self):
        # note that this tests the case of an input structure which is >10 Å in each direction and has
        # more atoms (198) than the pmg supercell (99), so the pmg supercell is used
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ytos_defect_gen = DefectsGenerator(self.ytos_bulk_supercell)  # Y2Ti2S2O5 supercell
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.ytos_defect_gen_info in output

        self.ytos_defect_gen_check(ytos_defect_gen)

    def test_ytos_no_generate_supercell(self):
        # tests the case of an input structure which is >10 Å in each direction, has
        # more atoms (198) than the pmg supercell (99), but generate_supercell = False,
        # so the _input_ supercell is used
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ytos_defect_gen = DefectsGenerator(self.ytos_bulk_supercell, generate_supercell=False)
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.ytos_defect_gen_info in output

        self.ytos_defect_gen_check(ytos_defect_gen, generate_supercell=False)

    def lmno_defect_gen_check(self, lmno_defect_gen, generate_supercell=True):
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(  # reduces to primitive, but StructureMatcher still matches
            lmno_defect_gen.primitive_structure, self.lmno_primitive
        )
        assert np.allclose(
            lmno_defect_gen.primitive_structure.lattice.matrix, self.lmno_primitive.lattice.matrix
        )
        if generate_supercell:
            np.testing.assert_allclose(
                lmno_defect_gen.supercell_matrix, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
            )
        else:
            np.testing.assert_allclose(
                lmno_defect_gen.supercell_matrix, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            )
        assert structure_matcher.fit(
            lmno_defect_gen.primitive_structure * lmno_defect_gen.supercell_matrix,
            lmno_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (lmno_defect_gen.primitive_structure * lmno_defect_gen.supercell_matrix).lattice.matrix,
            lmno_defect_gen.bulk_supercell.lattice.matrix,
        )

        assert structure_matcher.fit(lmno_defect_gen.conventional_structure, self.lmno_primitive)
        sga = SpacegroupAnalyzer(self.lmno_primitive)
        assert np.allclose(
            lmno_defect_gen.conventional_structure.lattice.matrix,
            sga.get_conventional_standard_structure().lattice.matrix,
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

        # test some relevant defect attributes
        assert lmno_defect_gen.defects["vacancies"][0].name == "v_Li"
        assert lmno_defect_gen.defects["vacancies"][0].oxi_state == -1
        assert lmno_defect_gen.defects["vacancies"][0].multiplicity == 8  # prim = conv structure in LMNO
        assert lmno_defect_gen.defects["vacancies"][0].defect_type == DefectType.Vacancy
        assert lmno_defect_gen.defects["vacancies"][0].structure == lmno_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            lmno_defect_gen.defects["vacancies"][0].defect_structure.lattice.matrix,
            lmno_defect_gen.primitive_structure.lattice.matrix,
        )
        assert np.allclose(
            lmno_defect_gen.defects["vacancies"][0].site.frac_coords,
            np.array([0.0037, 0.0037, 0.0037]),
            atol=1e-4,
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
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect.wyckoff == "12d"
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect.multiplicity == 12
        )  # prim = conv structure in LMNO
        if generate_supercell:
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].sc_defect_frac_coords,
                np.array([0.4246, 0.4375, 0.5496]),  # closes to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect_supercell_site.frac_coords,
                np.array([0.4246, 0.4375, 0.5496]),  # closes to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        else:
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].sc_defect_frac_coords,
                np.array([0.15074, 0.375, 0.40074]),  # closes to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect_supercell_site.frac_coords,
                np.array([0.15074, 0.375, 0.40074]),  # closes to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect_supercell_site.specie.symbol == "Ni"
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].conv_cell_frac_coords,
            np.array([0.151, 0.375, 0.401]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect.conv_cell_frac_coords,
            np.array([0.151, 0.375, 0.401]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84_+2"].defect.site.frac_coords,
            np.array([0.1250, 0.5993, 0.6507]),
            atol=1e-3,
        )

        for defect_name, defect_entry in lmno_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                lmno_defect_gen.bulk_supercell.lattice.matrix,
            )
            assert np.allclose(
                defect_entry.conventional_structure.lattice.matrix,
                sga.get_conventional_standard_structure().lattice.matrix,
            )
            assert np.allclose(
                defect_entry.defect.conventional_structure.lattice.matrix,
                sga.get_conventional_standard_structure().lattice.matrix,
            )
            # get minimum distance of defect_entry.conv_cell_frac_coords to any site in
            # defect_entry.conventional_structure
            distances = []
            for site in defect_entry.conventional_structure:
                distances.append(
                    site.distance_and_image_from_frac_coords(defect_entry.conv_cell_frac_coords)[0]
                )
            assert min(np.array(distances)[np.array(distances) > 0.001]) > 0.9  # default min_dist = 0.9
            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, lmno_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity == int(
                defect_entry.wyckoff[:-1]
            )  # prim = conv in LMNO
            assert defect_entry.defect_supercell_site

        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.name == "Li_O"
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.oxi_state == +3
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.multiplicity == 8
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].wyckoff == "8c"
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.defect_type == DefectType.Substitution
        assert (
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.structure
            == lmno_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.defect_structure.lattice.matrix,
            lmno_defect_gen.primitive_structure.lattice.matrix,
        )

        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].conv_cell_frac_coords,
            np.array([0.384, 0.384, 0.384]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.conv_cell_frac_coords,
            np.array([0.384, 0.384, 0.384]),
            atol=1e-3,
        )
        if generate_supercell:
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Li_O_C3_+3"].sc_defect_frac_coords,
                np.array([0.4328, 0.4328, 0.4328]),  # closest to middle of supercell
                atol=1e-4,
            )
            np.testing.assert_allclose(
                lmno_defect_gen["Li_O_C3_+3"].defect_supercell_site.frac_coords,
                np.array([0.4328, 0.4328, 0.4328]),  # closest to middle of supercell
                atol=1e-4,
            )
        else:
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Li_O_C3_+3"].sc_defect_frac_coords,
                np.array([0.38447, 0.38447, 0.38447]),  # closest to middle of supercell
                atol=1e-4,
            )
            np.testing.assert_allclose(
                lmno_defect_gen["Li_O_C3_+3"].defect_supercell_site.frac_coords,
                np.array([0.38447, 0.38447, 0.38447]),  # closest to middle of supercell
                atol=1e-4,
            )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.site.frac_coords,
            np.array([0.1155, 0.6155, 0.8845]),
            atol=1e-3,
        )

    def test_lmno(self):
        # battery material with a variety of important Wyckoff sites (and the terminology mainly
        # used in this field). Tough to find suitable supercell, goes to 448-atom supercell.
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                lmno_defect_gen = DefectsGenerator(self.lmno_primitive)  # Li2Mn3NiO8 unit cell
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value

        assert self.lmno_defect_gen_info in output

        self.lmno_defect_gen_check(lmno_defect_gen)

    def test_lmno_no_generate_supercell(self):
        # test inputting a non-diagonal supercell structure with a lattice vector <10 Å with
        # generate_supercell = False
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                lmno_defect_gen = DefectsGenerator(self.lmno_primitive, generate_supercell=False)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                "Input structure is <10 Å in at least one direction (minimum image distance = 8.28 Å, "
                "which is usually too small for accurate defect calculations, but "
                "generate_supercell = False, so using input structure as defect & bulk supercells. "
                "Caution advised!" in str(w[-1].message)
            )
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.lmno_defect_gen_info in output

        self.lmno_defect_gen_check(lmno_defect_gen, generate_supercell=False)

    def zns_defect_gen_check(self, zns_defect_gen, generate_supercell=True):
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(zns_defect_gen.primitive_structure, self.non_diagonal_ZnS)
        assert structure_matcher.fit(
            zns_defect_gen.primitive_structure, zns_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches (but below lattice doesn't match)
        assert not np.allclose(
            zns_defect_gen.primitive_structure.lattice.matrix, self.non_diagonal_ZnS.lattice.matrix
        )

        if generate_supercell:
            np.testing.assert_allclose(
                zns_defect_gen.supercell_matrix, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
            )
        else:
            np.testing.assert_allclose(
                zns_defect_gen.supercell_matrix, np.array([[0, 0, -2], [0, -4, 2], [-4, 1, 2]])
            )
        assert structure_matcher.fit(
            zns_defect_gen.primitive_structure * zns_defect_gen.supercell_matrix,
            zns_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (zns_defect_gen.primitive_structure * zns_defect_gen.supercell_matrix).lattice.matrix,
            zns_defect_gen.bulk_supercell.lattice.matrix,
        )
        assert structure_matcher.fit(zns_defect_gen.conventional_structure, self.non_diagonal_ZnS)
        sga = SpacegroupAnalyzer(self.non_diagonal_ZnS)
        assert np.allclose(
            zns_defect_gen.conventional_structure.lattice.matrix,
            sga.get_conventional_standard_structure().lattice.matrix,
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

        # test some relevant defect attributes
        assert zns_defect_gen.defects["vacancies"][1].name == "v_S"
        assert zns_defect_gen.defects["vacancies"][1].oxi_state == +2
        assert zns_defect_gen.defects["vacancies"][1].multiplicity == 1
        assert zns_defect_gen.defects["vacancies"][1].defect_type == DefectType.Vacancy
        assert zns_defect_gen.defects["vacancies"][1].structure == zns_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            zns_defect_gen.defects["vacancies"][1].defect_structure.lattice.matrix,
            zns_defect_gen.primitive_structure.lattice.matrix,
        )
        assert np.allclose(
            zns_defect_gen.defects["vacancies"][1].site.frac_coords, np.array([0.25, 0.25, 0.25])
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
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect.wyckoff == "4b"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect.multiplicity == 1
        if generate_supercell:
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].sc_defect_frac_coords,
                np.array([0.25, 0.5, 0.5]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect_supercell_site.frac_coords,
                np.array([0.25, 0.5, 0.5]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        else:
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].sc_defect_frac_coords,
                np.array([0.59375, 0.46875, 0.375]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect_supercell_site.frac_coords,
                np.array([0.59375, 0.46875, 0.375]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect_supercell_site.specie.symbol == "S"
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect.conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_+2"].defect.site.frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )

        for defect_name, defect_entry in zns_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                zns_defect_gen.bulk_supercell.lattice.matrix,
            )
            assert np.allclose(
                defect_entry.conventional_structure.lattice.matrix,
                sga.get_conventional_standard_structure().lattice.matrix,
            )
            assert np.allclose(
                defect_entry.defect.conventional_structure.lattice.matrix,
                sga.get_conventional_standard_structure().lattice.matrix,
            )
            # get minimum distance of defect_entry.conv_cell_frac_coords to any site in
            # defect_entry.conventional_structure
            distances = []
            for site in defect_entry.conventional_structure:
                distances.append(
                    site.distance_and_image_from_frac_coords(defect_entry.conv_cell_frac_coords)[0]
                )
            assert min(np.array(distances)[np.array(distances) > 0.001]) > 0.9  # default min_dist = 0.9
            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, zns_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 4 == int(
                defect_entry.wyckoff[:-1]
            )  # 4 prim cells in conv cell in Zinc Blende (ZnS, CdTe)
            assert defect_entry.defect_supercell_site

        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.name == "Zn_S"
        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.oxi_state == +4
        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.multiplicity == 1
        assert zns_defect_gen.defect_entries["Zn_S_+2"].wyckoff == "4c"
        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.defect_type == DefectType.Substitution
        assert (
            zns_defect_gen.defect_entries["Zn_S_+2"].defect.structure == zns_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            zns_defect_gen.defect_entries["Zn_S_+2"].defect.defect_structure.lattice.matrix,
            zns_defect_gen.primitive_structure.lattice.matrix,
        )

        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["Zn_S_+2"].conv_cell_frac_coords,
            np.array([0.25, 0.25, 0.25]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["Zn_S_+2"].defect.conv_cell_frac_coords,
            np.array([0.25, 0.25, 0.25]),
            atol=1e-3,
        )
        if generate_supercell:
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["Zn_S_+2"].sc_defect_frac_coords,
                np.array([0.375, 0.375, 0.625]),  # closest to middle of supercell
                atol=1e-4,
            )
            np.testing.assert_allclose(
                zns_defect_gen["Zn_S_+2"].defect_supercell_site.frac_coords,
                np.array([0.375, 0.375, 0.625]),  # closest to middle of supercell
                atol=1e-4,
            )
        else:
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["Zn_S_+2"].sc_defect_frac_coords,
                np.array([0.359375, 0.546875, 0.4375]),  # closest to middle of supercell
                atol=1e-4,
            )
            np.testing.assert_allclose(
                zns_defect_gen["Zn_S_+2"].defect_supercell_site.frac_coords,
                np.array([0.359375, 0.546875, 0.4375]),  # closest to middle of supercell
                atol=1e-4,
            )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["Zn_S_+2"].defect.site.frac_coords,
            np.array([0.25, 0.25, 0.25]),
            atol=1e-3,
        )

    def test_zns_non_diagonal_supercell(self):
        # test inputting a non-diagonal supercell structure -> correct primitive structure
        # determined and reasonable supercell generated
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                zns_defect_gen = DefectsGenerator(self.non_diagonal_ZnS)  # ZnS non-diagonal supercell
                assert len(w) == 0
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.zns_defect_gen_info in output

        self.zns_defect_gen_check(zns_defect_gen)

    def test_zns_no_generate_supercell(self):
        # test inputting a non-diagonal supercell structure with a lattice vector <10 Å with
        # generate_supercell = False
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                zns_defect_gen = DefectsGenerator(
                    self.non_diagonal_ZnS, generate_supercell=False
                )  # ZnS non-diagonal supercell
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                "Input structure is <10 Å in at least one direction (minimum image distance = 7.59 Å, "
                "which is usually too small for accurate defect calculations, but "
                "generate_supercell = False, so using input structure as defect & bulk supercells. "
                "Caution advised!" in str(w[-1].message)
            )
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.zns_defect_gen_info in output

        self.zns_defect_gen_check(zns_defect_gen, generate_supercell=False)


class WyckoffTest(unittest.TestCase):
    def test_wyckoff_dict_from_sgn(self):
        for sgn in range(1, 231):
            wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)
            assert isinstance(wyckoff_dict, dict)
            assert all(isinstance(k, str) for k in wyckoff_dict)
            assert all(isinstance(v, list) for v in wyckoff_dict.values())
