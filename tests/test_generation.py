"""
Tests for the `doped.generation` module.
"""
import copy
import filecmp
import json
import os
import shutil
import sys
import unittest
import warnings
from io import StringIO
from unittest.mock import patch

import numpy as np
from ase.build import bulk, make_supercell
from monty.json import MontyEncoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects.core import Defect, DefectType
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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


class DefectsGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.prim_cdte = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        sga = SpacegroupAnalyzer(self.prim_cdte)
        self.conv_cdte = sga.get_conventional_standard_structure()
        self.cdte_bulk_supercell = self.conv_cdte * 2 * np.eye(3)
        self.cdte_defect_gen_string = (
            "DefectsGenerator for input composition CdTe, space group F-43m with 50 defect entries "
            "created."
        )
        self.cdte_defect_gen_info = (
            """Vacancies    Charge States    Conv. Cell Coords    Wyckoff
-----------  ---------------  -------------------  ---------
v_Cd         [-2,-1,0,+1]     [0.000,0.000,0.000]  4a
v_Te         [-1,0,+1,+2]     [0.250,0.250,0.250]  4c

Substitutions    Charge States          Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Cd_Te            [0,+1,+2,+3,+4]        [0.250,0.250,0.250]  4c
Te_Cd            [-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.000]  4a

Interstitials    Charge States          Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Cd_i_C3v         [0,+1,+2]              [0.625,0.625,0.625]  16e
Cd_i_Td_Cd2.83   [0,+1,+2]              [0.750,0.750,0.750]  4d
Cd_i_Td_Te2.83   [0,+1,+2]              [0.500,0.500,0.500]  4b
Te_i_C3v         [-2,-1,0,+1,+2,+3,+4]  [0.625,0.625,0.625]  16e
Te_i_Td_Cd2.83   [-2,-1,0,+1,+2,+3,+4]  [0.750,0.750,0.750]  4d
Te_i_Td_Te2.83   [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.500]  4b
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of CdTe.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.ytos_bulk_supercell = Structure.from_file(f"{self.example_dir}/YTOS/Bulk/POSCAR")
        self.ytos_defect_gen_string = (
            "DefectsGenerator for input composition Y2Ti2S2O5, space group I4/mmm with 221 defect "
            "entries created."
        )
        self.ytos_defect_gen_info = (
            """Vacancies    Charge States       Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Y          [-3,-2,-1,0,+1]     [0.000,0.000,0.334]  4e
v_Ti         [-4,-3,-2,-1,0,+1]  [0.000,0.000,0.078]  4e
v_S          [-1,0,+1,+2]        [0.000,0.000,0.205]  4e
v_O_C2v      [-1,0,+1,+2]        [0.000,0.500,0.099]  8g
v_O_D4h      [-1,0,+1,+2]        [0.000,0.000,0.000]  2a

Substitutions    Charge States                Conv. Cell Coords    Wyckoff
---------------  ---------------------------  -------------------  ---------
Y_Ti             [-1,0]                       [0.000,0.000,0.078]  4e
Y_S              [0,+1,+2,+3,+4,+5]           [0.000,0.000,0.205]  4e
Y_O_C2v          [0,+1,+2,+3,+4,+5]           [0.000,0.500,0.099]  8g
Y_O_D4h          [0,+1,+2,+3,+4,+5]           [0.000,0.000,0.000]  2a
Ti_Y             [-1,0,+1]                    [0.000,0.000,0.334]  4e
Ti_S             [0,+1,+2,+3,+4,+5,+6]        [0.000,0.000,0.205]  4e
Ti_O_C2v         [0,+1,+2,+3,+4,+5,+6]        [0.000,0.500,0.099]  8g
Ti_O_D4h         [0,+1,+2,+3,+4,+5,+6]        [0.000,0.000,0.000]  2a
S_Y              [-5,-4,-3,-2,-1,0,+1,+2,+3]  [0.000,0.000,0.334]  4e
S_Ti             [-6,-5,-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.078]  4e
S_O_C2v          [-1,0,+1]                    [0.000,0.500,0.099]  8g
S_O_D4h          [-1,0,+1]                    [0.000,0.000,0.000]  2a
O_Y              [-5,-4,-3,-2,-1,0]           [0.000,0.000,0.334]  4e
O_Ti             [-6,-5,-4,-3,-2,-1,0]        [0.000,0.000,0.078]  4e
O_S              [-1,0,+1]                    [0.000,0.000,0.205]  4e

Interstitials    Charge States          Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Y_i_C2v          [0,+1,+2,+3]           [0.000,0.500,0.184]  8g
Y_i_C4v_O1.92    [0,+1,+2,+3]           [0.000,0.000,0.418]  4e
Y_i_C4v_O2.68    [0,+1,+2,+3]           [0.000,0.000,0.485]  4e
Y_i_Cs_O1.71     [0,+1,+2,+3]           [0.181,0.181,0.143]  16m
Y_i_Cs_O1.95     [0,+1,+2,+3]           [0.325,0.325,0.039]  16m
Y_i_D2d          [0,+1,+2,+3]           [0.000,0.500,0.250]  4d
Ti_i_C2v         [0,+1,+2,+3,+4]        [0.000,0.500,0.184]  8g
Ti_i_C4v_O1.92   [0,+1,+2,+3,+4]        [0.000,0.000,0.418]  4e
Ti_i_C4v_O2.68   [0,+1,+2,+3,+4]        [0.000,0.000,0.485]  4e
Ti_i_Cs_O1.71    [0,+1,+2,+3,+4]        [0.181,0.181,0.143]  16m
Ti_i_Cs_O1.95    [0,+1,+2,+3,+4]        [0.325,0.325,0.039]  16m
Ti_i_D2d         [0,+1,+2,+3,+4]        [0.000,0.500,0.250]  4d
S_i_C2v          [-2,-1,0,+1,+2,+3,+4]  [0.000,0.500,0.184]  8g
S_i_C4v_O1.92    [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.418]  4e
S_i_C4v_O2.68    [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.485]  4e
S_i_Cs_O1.71     [-2,-1,0,+1,+2,+3,+4]  [0.181,0.181,0.143]  16m
S_i_Cs_O1.95     [-2,-1,0,+1,+2,+3,+4]  [0.325,0.325,0.039]  16m
S_i_D2d          [-2,-1,0,+1,+2,+3,+4]  [0.000,0.500,0.250]  4d
O_i_C2v          [-2,-1,0]              [0.000,0.500,0.184]  8g
O_i_C4v_O1.92    [-2,-1,0]              [0.000,0.000,0.418]  4e
O_i_C4v_O2.68    [-2,-1,0]              [0.000,0.000,0.485]  4e
O_i_Cs_O1.71     [-2,-1,0]              [0.181,0.181,0.143]  16m
O_i_Cs_O1.95     [-2,-1,0]              [0.325,0.325,0.039]  16m
O_i_D2d          [-2,-1,0]              [0.000,0.500,0.250]  4d
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 2 formula unit(s) of Y2Ti2S2O5.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.lmno_defect_gen_string = (
            "DefectsGenerator for input composition Li2Mn3NiO8, space group P4_332 with 197 defect "
            "entries created."
        )
        self.lmno_defect_gen_info = (
            """Vacancies    Charge States       Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Li         [-1,0,+1]           [0.004,0.004,0.004]  8c
v_Mn         [-4,-3,-2,-1,0,+1]  [0.121,0.129,0.625]  12d
v_Ni         [-2,-1,0,+1]        [0.625,0.625,0.625]  4b
v_O_C1       [-1,0,+1,+2]        [0.101,0.124,0.392]  24e
v_O_C3       [-1,0,+1,+2]        [0.384,0.384,0.384]  8c

Substitutions    Charge States          Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Li_Mn            [-3,-2,-1,0]           [0.121,0.129,0.625]  12d
Li_Ni            [-1,0]                 [0.625,0.625,0.625]  4b
Li_O_C1          [0,+1,+2,+3]           [0.101,0.124,0.392]  24e
Li_O_C3          [0,+1,+2,+3]           [0.384,0.384,0.384]  8c
Mn_Li            [0,+1,+2,+3]           [0.004,0.004,0.004]  8c
Mn_Ni            [0,+1,+2]              [0.625,0.625,0.625]  4b
Mn_O_C1          [0,+1,+2,+3,+4,+5,+6]  [0.101,0.124,0.392]  24e
Mn_O_C3          [0,+1,+2,+3,+4,+5,+6]  [0.384,0.384,0.384]  8c
Ni_Li            [0,+1,+2,+3]           [0.004,0.004,0.004]  8c
Ni_Mn            [-3,-2,-1,0]           [0.121,0.129,0.625]  12d
Ni_O_C1          [0,+1,+2,+3,+4,+5]     [0.101,0.124,0.392]  24e
Ni_O_C3          [0,+1,+2,+3,+4,+5]     [0.384,0.384,0.384]  8c
O_Li             [-3,-2,-1,0]           [0.004,0.004,0.004]  8c
O_Mn             [-6,-5,-4,-3,-2,-1,0]  [0.121,0.129,0.625]  12d
O_Ni             [-4,-3,-2,-1,0]        [0.625,0.625,0.625]  4b

Interstitials        Charge States    Conv. Cell Coords    Wyckoff
-------------------  ---------------  -------------------  ---------
Li_i_C1_Li1.75       [0,+1]           [0.199,0.303,0.444]  24e
Li_i_C1_O1.72        [0,+1]           [0.001,0.770,0.002]  24e
Li_i_C1_O1.78        [0,+1]           [0.017,0.261,0.250]  24e
Li_i_C2_Li1.84O1.84  [0,+1]           [0.073,0.177,0.125]  12d
Li_i_C2_Li1.84O1.94  [0,+1]           [0.151,0.375,0.401]  12d
Li_i_C2_Li1.86       [0,+1]           [0.085,0.375,0.335]  12d
Li_i_C3              [0,+1]           [0.497,0.497,0.497]  8c
Mn_i_C1_Li1.75       [0,+1,+2,+3,+4]  [0.199,0.303,0.444]  24e
Mn_i_C1_O1.72        [0,+1,+2,+3,+4]  [0.001,0.770,0.002]  24e
Mn_i_C1_O1.78        [0,+1,+2,+3,+4]  [0.017,0.261,0.250]  24e
Mn_i_C2_Li1.84O1.84  [0,+1,+2,+3,+4]  [0.073,0.177,0.125]  12d
Mn_i_C2_Li1.84O1.94  [0,+1,+2,+3,+4]  [0.151,0.375,0.401]  12d
Mn_i_C2_Li1.86       [0,+1,+2,+3,+4]  [0.085,0.375,0.335]  12d
Mn_i_C3              [0,+1,+2,+3,+4]  [0.497,0.497,0.497]  8c
Ni_i_C1_Li1.75       [0,+1,+2,+3,+4]  [0.199,0.303,0.444]  24e
Ni_i_C1_O1.72        [0,+1,+2,+3,+4]  [0.001,0.770,0.002]  24e
Ni_i_C1_O1.78        [0,+1,+2,+3,+4]  [0.017,0.261,0.250]  24e
Ni_i_C2_Li1.84O1.84  [0,+1,+2,+3,+4]  [0.073,0.177,0.125]  12d
Ni_i_C2_Li1.84O1.94  [0,+1,+2,+3,+4]  [0.151,0.375,0.401]  12d
Ni_i_C2_Li1.86       [0,+1,+2,+3,+4]  [0.085,0.375,0.335]  12d
Ni_i_C3              [0,+1,+2,+3,+4]  [0.497,0.497,0.497]  8c
O_i_C1_Li1.75        [-2,-1,0]        [0.199,0.303,0.444]  24e
O_i_C1_O1.72         [-2,-1,0]        [0.001,0.770,0.002]  24e
O_i_C1_O1.78         [-2,-1,0]        [0.017,0.261,0.250]  24e
O_i_C2_Li1.84O1.84   [-2,-1,0]        [0.073,0.177,0.125]  12d
O_i_C2_Li1.84O1.94   [-2,-1,0]        [0.151,0.375,0.401]  12d
O_i_C2_Li1.86        [-2,-1,0]        [0.085,0.375,0.335]  12d
O_i_C3               [-2,-1,0]        [0.497,0.497,0.497]  8c
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of Li2Mn3NiO8.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")
        self.zns_defect_gen_string = (
            "DefectsGenerator for input composition ZnS, space group F-43m with 36 defect entries "
            "created."
        )
        self.zns_defect_gen_info = (
            """Vacancies    Charge States    Conv. Cell Coords    Wyckoff
-----------  ---------------  -------------------  ---------
v_Zn         [-2,-1,0,+1]     [0.000,0.000,0.000]  4a
v_S          [-1,0,+1,+2]     [0.250,0.250,0.250]  4c

Substitutions    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Zn_S             [0,+1,+2,+3,+4]  [0.250,0.250,0.250]  4c
S_Zn             [-4,-3,-2,-1,0]  [0.000,0.000,0.000]  4a

Interstitials    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Zn_i_C3v         [0,+1,+2]        [0.625,0.625,0.625]  16e
Zn_i_Td_S2.35    [0,+1,+2]        [0.500,0.500,0.500]  4b
Zn_i_Td_Zn2.35   [0,+1,+2]        [0.750,0.750,0.750]  4d
S_i_C3v          [-2,-1,0]        [0.625,0.625,0.625]  16e
S_i_Td_S2.35     [-2,-1,0]        [0.500,0.500,0.500]  4b
S_i_Td_Zn2.35    [-2,-1,0]        [0.750,0.750,0.750]  4d
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of ZnS.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.prim_cu = Structure.from_file(f"{self.data_dir}/Cu_prim_POSCAR")
        self.cu_defect_gen_string = (
            "DefectsGenerator for input composition Cu, space group Fm-3m with 9 defect entries created."
        )
        self.cu_defect_gen_info = (
            """Vacancies    Charge States    Conv. Cell Coords    Wyckoff
-----------  ---------------  -------------------  ---------
v_Cu         [-1,0,+1]        [0.000,0.000,0.000]  4a

Interstitials    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Cu_i_Oh          [0,+1,+2]        [0.500,0.500,0.500]  4b
Cu_i_Td          [0,+1,+2]        [0.250,0.250,0.250]  8c
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of Cu.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        # AgCu:
        atoms = bulk("Cu")
        atoms = make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        atoms.set_chemical_symbols(["Cu", "Ag"] * 4)
        aaa = AseAtomsAdaptor()
        self.agcu = aaa.get_structure(atoms)
        self.agcu_defect_gen_string = (
            "DefectsGenerator for input composition AgCu, space group R-3m with 28 defect entries created."
        )
        self.agcu_defect_gen_info = (
            """Vacancies    Charge States    Conv. Cell Coords    Wyckoff
-----------  ---------------  -------------------  ---------
v_Cu         [-1,0,+1]        [0.000,0.000,0.000]  3a
v_Ag         [-1,0,+1]        [0.000,0.000,0.500]  3b

Substitutions    Charge States    Conv. Cell Coords    Wyckoff
---------------  ---------------  -------------------  ---------
Cu_Ag            [-1,0,+1,+2]     [0.000,0.000,0.500]  3b
Ag_Cu            [-1,0,+1]        [0.000,0.000,0.000]  3a

Interstitials                 Charge States    Conv. Cell Coords    Wyckoff
----------------------------  ---------------  -------------------  ---------
Cu_i_C3v_Ag1.56Cu1.56Ag2.99a  [0,+1,+2]        [0.000,0.000,0.125]  6c
Cu_i_C3v_Ag1.56Cu1.56Ag2.99b  [0,+1,+2]        [0.000,0.000,0.375]  6c
Cu_i_C3v_Ag1.80               [0,+1,+2]        [0.000,0.000,0.250]  6c
Ag_i_C3v_Ag1.56Cu1.56Ag2.99a  [0,+1]           [0.000,0.000,0.125]  6c
Ag_i_C3v_Ag1.56Cu1.56Ag2.99b  [0,+1]           [0.000,0.000,0.375]  6c
Ag_i_C3v_Ag1.80               [0,+1]           [0.000,0.000,0.250]  6c
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 3 formula unit(s) of AgCu.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.cd_i_cdte_supercell_defect_gen_info = (
            """Vacancies                    Charge States    Conv. Cell Coords    Wyckoff
---------------------------  ---------------  -------------------  ---------
v_Cd_C1_Te2.83Cd4.62Te5.42a  [-1,0,+1]        [0.000,0.250,0.000]  18c
v_Cd_C1_Te2.83Cd4.62Te5.42b  [-1,0,+1]        [0.250,0.250,0.500]  18c
v_Cd_C3v_Cd2.71              [-1,0,+1]        [0.000,0.000,0.688]  3a
v_Cd_C3v_Te2.83Cd4.25        [-1,0,+1]        [0.000,0.000,0.500]  3a
v_Cd_C3v_Te2.83Cd4.62        [-1,0,+1]        [0.000,0.000,0.000]  3a
v_Cd_Cs_Cd2.71               [-1,0,+1]        [0.500,0.250,0.000]  9b
v_Cd_Cs_Te2.83Cd4.25         [-1,0,+1]        [0.583,0.167,0.167]  9b
v_Cd_Cs_Te2.83Cd4.62Cd5.36   [-1,0,+1]        [0.000,0.500,0.000]  9b
v_Cd_Cs_Te2.83Cd4.62Te5.42a  [-1,0,+1]        [0.500,0.500,0.500]  9b
v_Cd_Cs_Te2.83Cd4.62Te5.42b  [-1,0,+1]        [0.083,0.167,0.167]  9b
v_Cd_Cs_Te2.83Cd4.62Te5.42c  [-1,0,+1]        [0.167,0.083,0.333]  9b
v_Te_C1_Cd2.83Te4.62Cd5.42a  [-1,0,+1]        [0.250,0.250,0.375]  18c
v_Te_C1_Cd2.83Te4.62Cd5.42b  [-1,0,+1]        [0.250,0.250,0.875]  18c
v_Te_C3v_Cd2.83Cd4.25        [-1,0,+1]        [0.000,0.000,0.875]  3a
v_Te_C3v_Cd2.83Te4.62        [-1,0,+1]        [0.000,0.000,0.375]  3a
v_Te_Cs_Cd2.71               [-1,0,+1]        [0.583,0.167,0.042]  9b
v_Te_Cs_Cd2.83Cd4.25         [-1,0,+1]        [0.083,0.167,0.542]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.36   [-1,0,+1]        [0.500,0.500,0.375]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.42a  [-1,0,+1]        [0.500,0.500,0.875]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.42b  [-1,0,+1]        [0.083,0.167,0.042]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.42c  [-1,0,+1]        [0.167,0.083,0.208]  9b

Substitutions                 Charge States    Conv. Cell Coords    Wyckoff
----------------------------  ---------------  -------------------  ---------
Cd_Te_C1_Cd2.83Te4.62Cd5.42a  [-1,0,+1,+2]     [0.250,0.250,0.375]  18c
Cd_Te_C1_Cd2.83Te4.62Cd5.42b  [-1,0,+1,+2]     [0.250,0.250,0.875]  18c
Cd_Te_C3v_Cd2.83Cd4.25        [-1,0,+1,+2]     [0.000,0.000,0.875]  3a
Cd_Te_C3v_Cd2.83Te4.62        [-1,0,+1,+2]     [0.000,0.000,0.375]  3a
Cd_Te_Cs_Cd2.71               [-1,0,+1,+2]     [0.583,0.167,0.042]  9b
Cd_Te_Cs_Cd2.83Cd4.25         [-1,0,+1,+2]     [0.083,0.167,0.542]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.36   [-1,0,+1,+2]     [0.500,0.500,0.375]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.42a  [-1,0,+1,+2]     [0.500,0.500,0.875]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.42b  [-1,0,+1,+2]     [0.083,0.167,0.042]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.42c  [-1,0,+1,+2]     [0.167,0.083,0.208]  9b
Te_Cd_C1_Te2.83Cd4.62Te5.42a  [-2,-1,0,+1]     [0.000,0.250,0.000]  18c
Te_Cd_C1_Te2.83Cd4.62Te5.42b  [-2,-1,0,+1]     [0.250,0.250,0.500]  18c
Te_Cd_C3v_Cd2.71              [-2,-1,0,+1]     [0.000,0.000,0.688]  3a
Te_Cd_C3v_Te2.83Cd4.25        [-2,-1,0,+1]     [0.000,0.000,0.500]  3a
Te_Cd_C3v_Te2.83Cd4.62        [-2,-1,0,+1]     [0.000,0.000,0.000]  3a
Te_Cd_Cs_Cd2.71               [-2,-1,0,+1]     [0.500,0.250,0.000]  9b
Te_Cd_Cs_Te2.83Cd4.25         [-2,-1,0,+1]     [0.583,0.167,0.167]  9b
Te_Cd_Cs_Te2.83Cd4.62Cd5.36   [-2,-1,0,+1]     [0.000,0.500,0.000]  9b
Te_Cd_Cs_Te2.83Cd4.62Te5.42a  [-2,-1,0,+1]     [0.500,0.500,0.500]  9b
Te_Cd_Cs_Te2.83Cd4.62Te5.42b  [-2,-1,0,+1]     [0.083,0.167,0.167]  9b
Te_Cd_Cs_Te2.83Cd4.62Te5.42c  [-2,-1,0,+1]     [0.167,0.083,0.333]  9b

Interstitials                Charge States    Conv. Cell Coords    Wyckoff
---------------------------  ---------------  -------------------  ---------
Cd_i_C1_Cd2.71Te2.71Cd4.01a  [0,+1,+2]        [0.458,0.167,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.01b  [0,+1,+2]        [0.208,0.042,0.604]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25a  [0,+1,+2]        [0.125,0.125,0.438]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25b  [0,+1,+2]        [0.250,0.250,0.188]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25c  [0,+1,+2]        [0.125,0.125,0.938]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25d  [0,+1,+2]        [0.250,0.250,0.688]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25e  [0,+1,+2]        [0.375,0.375,0.438]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25f  [0,+1,+2]        [0.375,0.375,0.938]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25g  [0,+1,+2]        [0.208,0.042,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25h  [0,+1,+2]        [0.083,0.292,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25i  [0,+1,+2]        [0.042,0.208,0.271]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25j  [0,+1,+2]        [0.458,0.042,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25k  [0,+1,+2]        [0.292,0.083,0.271]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25l  [0,+1,+2]        [0.042,0.458,0.271]  18c
Cd_i_C1_Cd2.83Te3.27Cd4.84   [0,+1,+2]        [0.250,0.250,0.625]  18c
Cd_i_C1_Cd2.83Te3.27Cd5.42   [0,+1,+2]        [0.250,0.250,0.125]  18c
Cd_i_C1_Te2.83Cd3.27Cd4.84   [0,+1,+2]        [0.750,0.750,0.750]  18c
Cd_i_C1_Te2.83Cd3.27Te5.42   [0,+1,+2]        [0.250,0.250,0.250]  18c
Cd_i_C3v_Cd2.71              [0,+1,+2]        [0.000,0.000,0.188]  3a
Cd_i_C3v_Cd2.83              [0,+1,+2]        [0.000,0.000,0.125]  3a
Cd_i_C3v_Te2.83              [0,+1,+2]        [0.000,0.000,0.250]  3a
Cd_i_Cs_Cd2.59Cd2.65         [0,+1,+2]        [0.052,0.104,0.775]  9b
Cd_i_Cs_Cd2.59Te2.65         [0,+1,+2]        [0.104,0.052,0.600]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25a  [0,+1,+2]        [0.500,0.500,0.188]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25b  [0,+1,+2]        [0.500,0.500,0.688]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25c  [0,+1,+2]        [0.083,0.042,0.104]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25d  [0,+1,+2]        [0.167,0.083,0.021]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25e  [0,+1,+2]        [0.042,0.083,0.271]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25f  [0,+1,+2]        [0.083,0.167,0.354]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25g  [0,+1,+2]        [0.083,0.542,0.104]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25h  [0,+1,+2]        [0.208,0.417,0.104]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25i  [0,+1,+2]        [0.167,0.083,0.521]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25j  [0,+1,+2]        [0.417,0.208,0.271]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25k  [0,+1,+2]        [0.542,0.083,0.271]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25l  [0,+1,+2]        [0.500,0.250,0.188]  9b
Cd_i_Cs_Cd2.83Te3.27Cd3.56   [0,+1,+2]        [0.500,0.250,0.125]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42a  [0,+1,+2]        [0.500,0.500,0.125]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42b  [0,+1,+2]        [0.500,0.500,0.625]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42c  [0,+1,+2]        [0.083,0.167,0.292]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42d  [0,+1,+2]        [0.167,0.083,0.458]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42e  [0,+1,+2]        [0.583,0.167,0.292]  9b
Cd_i_Cs_Te2.83Cd3.27Cd3.56   [0,+1,+2]        [0.250,0.500,0.250]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42a  [0,+1,+2]        [0.167,0.083,0.083]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42b  [0,+1,+2]        [0.500,0.250,0.250]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42c  [0,+1,+2]        [0.500,0.500,0.250]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42d  [0,+1,+2]        [0.500,0.500,0.750]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42e  [0,+1,+2]        [0.750,0.250,0.750]  9b
Te_i_C1_Cd2.71Te2.71Cd4.01a  [-2,-1,0]        [0.458,0.167,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.01b  [-2,-1,0]        [0.208,0.042,0.604]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25a  [-2,-1,0]        [0.125,0.125,0.438]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25b  [-2,-1,0]        [0.250,0.250,0.188]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25c  [-2,-1,0]        [0.125,0.125,0.938]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25d  [-2,-1,0]        [0.250,0.250,0.688]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25e  [-2,-1,0]        [0.375,0.375,0.438]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25f  [-2,-1,0]        [0.375,0.375,0.938]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25g  [-2,-1,0]        [0.208,0.042,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25h  [-2,-1,0]        [0.083,0.292,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25i  [-2,-1,0]        [0.042,0.208,0.271]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25j  [-2,-1,0]        [0.458,0.042,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25k  [-2,-1,0]        [0.292,0.083,0.271]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25l  [-2,-1,0]        [0.042,0.458,0.271]  18c
Te_i_C1_Cd2.83Te3.27Cd4.84   [-2,-1,0]        [0.250,0.250,0.625]  18c
Te_i_C1_Cd2.83Te3.27Cd5.42   [-2,-1,0]        [0.250,0.250,0.125]  18c
Te_i_C1_Te2.83Cd3.27Cd4.84   [-2,-1,0]        [0.750,0.750,0.750]  18c
Te_i_C1_Te2.83Cd3.27Te5.42   [-2,-1,0]        [0.250,0.250,0.250]  18c
Te_i_C3v_Cd2.71              [-2,-1,0]        [0.000,0.000,0.188]  3a
Te_i_C3v_Cd2.83              [-2,-1,0]        [0.000,0.000,0.125]  3a
Te_i_C3v_Te2.83              [-2,-1,0]        [0.000,0.000,0.250]  3a
Te_i_Cs_Cd2.59Cd2.65         [-2,-1,0]        [0.052,0.104,0.775]  9b
Te_i_Cs_Cd2.59Te2.65         [-2,-1,0]        [0.104,0.052,0.600]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25a  [-2,-1,0]        [0.500,0.500,0.188]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25b  [-2,-1,0]        [0.500,0.500,0.688]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25c  [-2,-1,0]        [0.083,0.042,0.104]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25d  [-2,-1,0]        [0.167,0.083,0.021]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25e  [-2,-1,0]        [0.042,0.083,0.271]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25f  [-2,-1,0]        [0.083,0.167,0.354]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25g  [-2,-1,0]        [0.083,0.542,0.104]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25h  [-2,-1,0]        [0.208,0.417,0.104]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25i  [-2,-1,0]        [0.167,0.083,0.521]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25j  [-2,-1,0]        [0.417,0.208,0.271]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25k  [-2,-1,0]        [0.542,0.083,0.271]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25l  [-2,-1,0]        [0.500,0.250,0.188]  9b
Te_i_Cs_Cd2.83Te3.27Cd3.56   [-2,-1,0]        [0.500,0.250,0.125]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42a  [-2,-1,0]        [0.500,0.500,0.125]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42b  [-2,-1,0]        [0.500,0.500,0.625]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42c  [-2,-1,0]        [0.083,0.167,0.292]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42d  [-2,-1,0]        [0.167,0.083,0.458]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42e  [-2,-1,0]        [0.583,0.167,0.292]  9b
Te_i_Cs_Te2.83Cd3.27Cd3.56   [-2,-1,0]        [0.250,0.500,0.250]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42a  [-2,-1,0]        [0.167,0.083,0.083]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42b  [-2,-1,0]        [0.500,0.250,0.250]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42c  [-2,-1,0]        [0.500,0.500,0.250]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42d  [-2,-1,0]        [0.500,0.500,0.750]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42e  [-2,-1,0]        [0.750,0.250,0.750]  9b
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 3 formula unit(s) of Cd33Te32.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        # TODO: test all input parameters; extrinsic, interstitial_coords, interstitial/supercell gen
        #  kwargs, target_frac_coords, charge_state_gen_kwargs setting...
        # test input parameters used as attributes
        # TODO: Test equiv coords list (supercell) for defect_entries, defects)
        # TODO: test Zn3P2 (and Sb2Se3)? Important test case(s) for charge state setting and Wyckoff
        #  handling (once charge state setting algorithm finalised a bit more)

    def _save_defect_gen_jsons(self, defect_gen):
        defect_gen.to_json("test.json")
        dumpfn(defect_gen, "test_defect_gen.json")
        defect_gen.to_json()  # test default

        formula, _fu = defect_gen.primitive_structure.composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )
        default_json_filename = f"{formula}_defects_generator.json"

        # assert these saved files are the exact same:
        assert filecmp.cmp("test.json", "test_defect_gen.json")
        assert filecmp.cmp("test.json", default_json_filename)
        if_present_rm("test.json")
        if_present_rm("test_defect_gen.json")

    def _load_and_test_defect_gen_jsons(self, defect_gen):  # , gen_check): - shouldn't need this as we
        # test that the jsons are identical (except for ordering)
        formula, _fu = defect_gen.primitive_structure.composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )
        default_json_filename = f"{formula}_defects_generator.json"
        defect_gen_from_json = DefectsGenerator.from_json(default_json_filename)
        defect_gen_from_json_loadfn = loadfn(default_json_filename)

        # gen_check(defect_gen_from_json)

        # test saving to json again gives same object:
        defect_gen_from_json.to_json("test.json")
        defect_gen_from_json_loadfn.to_json("test_loadfn.json")
        assert filecmp.cmp("test.json", "test_loadfn.json")

        # test it's the same as the original:
        # here we compare using json dumps because the ordering can change slightly when saving to json
        assert json.dumps(defect_gen_from_json, sort_keys=True, cls=MontyEncoder) == json.dumps(
            defect_gen, sort_keys=True, cls=MontyEncoder
        )
        if_present_rm("test.json")
        if_present_rm("test_loadfn.json")
        if_present_rm(default_json_filename)

    def cdte_defect_gen_check(self, cdte_defect_gen):
        # test attributes:
        assert self.cdte_defect_gen_info in cdte_defect_gen._defect_generator_info()
        assert cdte_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
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
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
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
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )
        assert cdte_defect_gen.defects["vacancies"][0].site == PeriodicSite(
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )
        assert (
            len(cdte_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4
        )  # 4x conv cell
        assert sum(vacancy.multiplicity for vacancy in cdte_defect_gen.defects["vacancies"]) == len(
            cdte_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(cdte_defect_gen.defect_entries) == 50
        assert len(cdte_defect_gen) == 50  # __len__()
        assert dict(cdte_defect_gen.items()) == cdte_defect_gen.defect_entries  # __iter__()
        assert str(cdte_defect_gen) == self.cdte_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(cdte_defect_gen)
            == self.cdte_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.cdte_defect_gen_info
        )
        assert all(
            defect_entry_name in cdte_defect_gen
            for defect_entry_name in cdte_defect_gen.defect_entries  # __contains__()
        )
        assert (
            cdte_defect_gen["Cd_i_C3v_0"] == cdte_defect_gen.defect_entries["Cd_i_C3v_0"]
        )  # __getitem__()
        assert cdte_defect_gen.get("Cd_i_C3v_0") == cdte_defect_gen.defect_entries["Cd_i_C3v_0"]  # get()
        cd_i_defect_entry = cdte_defect_gen.defect_entries["Cd_i_C3v_0"]
        del cdte_defect_gen["Cd_i_C3v_0"]  # __delitem__()
        assert "Cd_i_C3v_0" not in cdte_defect_gen
        cdte_defect_gen["Cd_i_C3v_0"] = cd_i_defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            cdte_defect_gen["Cd_i_C3v_0"] = cd_i_defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            fd_up_cd_i_defect_entry = copy.deepcopy(cdte_defect_gen.defect_entries["Cd_i_C3v_0"])
            fd_up_cd_i_defect_entry.defect.structure = self.cdte_bulk_supercell
            cdte_defect_gen["Cd_i_C3v_0"] = fd_up_cd_i_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )
        with self.assertRaises(ValueError) as e:
            fd_up_cd_i_defect_entry = copy.deepcopy(cdte_defect_gen.defect_entries["Cd_i_C3v_0"])
            fd_up_cd_i_defect_entry.sc_entry = copy.deepcopy(
                cdte_defect_gen.defect_entries["Cd_i_Td_Cd2.83_0"]
            )
            cdte_defect_gen["Cd_i_C3v_0"] = fd_up_cd_i_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in cdte_defect_gen.defect_entries.values()
        )
        # test defect entry charge state log:
        assert cdte_defect_gen.defect_entries["v_Cd_-1"].charge_state_guessing_log == [
            {
                "input_parameters": {"charge_state": -2},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.01,
                "padding": 1,
            },
            {
                "input_parameters": {"charge_state": -1},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.01,
                "padding": 1,
            },
            {
                "input_parameters": {"charge_state": 0},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.01,
                "padding": 1,
            },
            {
                "input_parameters": {"charge_state": 1},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.01,
                "padding": 1,
            },
        ]
        assert cdte_defect_gen.defect_entries["Cd_Te_0"].charge_state_guessing_log == [
            {
                "input_parameters": {
                    "charge_state": 4.0,
                    "oxi_state": 2.0,
                    "oxi_probability": 1.0,
                    "max_host_oxi_magnitude": 2.0,
                },
                "probability_factors": {
                    "oxi_probability": 1.0,
                    "charge_state_magnitude": 0.3968502629920499,
                    "charge_state_vs_max_host_charge": 0.3968502629920499,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability": 0.15749013123685918,
                "probability_threshold": 0.01,
            }
        ]
        assert cdte_defect_gen.defect_entries["Te_i_C3v_-1"].charge_state_guessing_log == [
            {
                "input_parameters": {
                    "charge_state": -2.0,
                    "oxi_state": -2.0,
                    "oxi_probability": 0.446,
                    "max_host_oxi_magnitude": 2.0,
                },
                "probability_factors": {
                    "oxi_probability": 0.446,
                    "charge_state_magnitude": 0.6299605249474366,
                    "charge_state_vs_max_host_charge": 1.0,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability": 0.2809623941265567,
                "probability_threshold": 0.01,
            },
            {
                "input_parameters": {
                    "charge_state": -1,
                    "oxi_state": -1,
                    "oxi_probability": 0.082,
                    "max_host_oxi_magnitude": 2.0,
                },
                "probability_factors": {
                    "oxi_probability": 0.082,
                    "charge_state_magnitude": 1.0,
                    "charge_state_vs_max_host_charge": 1.0,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability": 0.082,
                "probability_threshold": 0.01,
            },
            {
                "input_parameters": {
                    "charge_state": 4.0,
                    "oxi_state": 4.0,
                    "oxi_probability": 0.347,
                    "max_host_oxi_magnitude": 2.0,
                },
                "probability_factors": {
                    "oxi_probability": 0.347,
                    "charge_state_magnitude": 0.3968502629920499,
                    "charge_state_vs_max_host_charge": 0.3968502629920499,
                    "oxi_state_vs_max_host_charge": 0.3968502629920499,
                },
                "probability": 0.021687500000000002,
                "probability_threshold": 0.01,
            },
            {
                "input_parameters": {
                    "charge_state": 6.0,
                    "oxi_state": 6.0,
                    "oxi_probability": 0.111,
                    "max_host_oxi_magnitude": 2.0,
                },
                "probability_factors": {
                    "oxi_probability": 0.111,
                    "charge_state_magnitude": 0.3028534321386899,
                    "charge_state_vs_max_host_charge": 0.25,
                    "oxi_state_vs_max_host_charge": 0.25,
                },
                "probability": 0.0021010456854621616,
                "probability_threshold": 0.01,
            },
        ]

        # test defect entry attributes
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].name == "Cd_i_C3v_0"
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].charge_state == 0
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.defect_type == DefectType.Interstitial
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].wyckoff == "16e"
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.wyckoff == "16e"
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].conv_cell_frac_coords,
            np.array([0.625, 0.625, 0.625]),
        )
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.conv_cell_frac_coords,
            np.array([0.625, 0.625, 0.625]),
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
            np.array([0.625, 0.125, 0.625]),
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance)  # nn_distance the same for each equiv
                # site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, cdte_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 4 == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                assert np.isclose(
                    np.product(list(charge_state_dict["probability_factors"].values())),
                    charge_state_dict["probability"],
                )
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in cdte_defect_gen.defect_entries
                        for defect_name in cdte_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in cdte_defect_gen.defect_entries
                        for defect_name in cdte_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
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
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
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
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cdte_defect_gen_info in output  # matches expected 4b & 4d Wyckoff letters for Td
        # interstitials (https://doi.org/10.1016/j.solener.2013.12.017)

        # defect_gen_check changes defect_entries ordering, so save to json first:
        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

    def test_defects_generator_cdte_supercell_input(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cdte_defect_gen = DefectsGenerator(self.cdte_bulk_supercell)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cdte_defect_gen_info in output

        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

    def test_cdte_no_generate_supercell_supercell_input(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cdte_defect_gen = DefectsGenerator(self.cdte_bulk_supercell, generate_supercell=False)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

    @patch("sys.stdout", new_callable=StringIO)
    def test_generator_tqdm(self, mock_stdout):
        with patch("doped.generation.tqdm") as mocked_tqdm:
            mocked_instance = mocked_tqdm.return_value
            DefectsGenerator(self.prim_cdte)
            mocked_tqdm.assert_called_once()
            mocked_tqdm.assert_called_with(
                total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
            )
            mocked_instance.set_description.assert_any_call("Getting primitive structure")
            mocked_instance.set_description.assert_any_call("Generating vacancies")
            mocked_instance.set_description.assert_any_call("Generating substitutions")
            mocked_instance.set_description.assert_any_call("Generating interstitials")
            mocked_instance.set_description.assert_any_call("Generating simulation supercell")
            mocked_instance.set_description.assert_any_call("Determining Wyckoff sites")
            mocked_instance.set_description.assert_any_call("Generating DefectEntry objects")

    def ytos_defect_gen_check(self, ytos_defect_gen, generate_supercell=True):
        assert self.ytos_defect_gen_info in ytos_defect_gen._defect_generator_info()
        assert ytos_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
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
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
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
        assert (
            len(ytos_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4
        )  # 2x conv cell
        assert sum(vacancy.multiplicity for vacancy in ytos_defect_gen.defects["vacancies"]) == len(
            ytos_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(ytos_defect_gen.defect_entries) == 221
        assert len(ytos_defect_gen) == 221  # __len__()
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in ytos_defect_gen.defect_entries.values()
        )
        assert dict(ytos_defect_gen.items()) == ytos_defect_gen.defect_entries  # __iter__()
        assert str(ytos_defect_gen) == self.ytos_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(ytos_defect_gen)
            == self.ytos_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.ytos_defect_gen_info
        )
        assert all(
            defect_entry_name in ytos_defect_gen
            for defect_entry_name in ytos_defect_gen.defect_entries  # __contains__()
        )
        assert (
            ytos_defect_gen["O_i_D2d_-1"] == ytos_defect_gen.defect_entries["O_i_D2d_-1"]
        )  # __getitem__()
        assert ytos_defect_gen.get("O_i_D2d_-1") == ytos_defect_gen.defect_entries["O_i_D2d_-1"]  # get()
        defect_entry = ytos_defect_gen.defect_entries["O_i_D2d_-1"]
        del ytos_defect_gen["O_i_D2d_-1"]  # __delitem__()
        assert "O_i_D2d_-1" not in ytos_defect_gen
        ytos_defect_gen["O_i_D2d_-1"] = defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            ytos_defect_gen["O_i_D2d_-1"] = defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(ytos_defect_gen.defect_entries["O_i_D2d_-1"])
            fd_up_defect_entry.defect.structure = self.cdte_bulk_supercell
            ytos_defect_gen["O_i_D2d_-1"] = fd_up_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(ytos_defect_gen.defect_entries["O_i_D2d_-1"])
            fd_up_defect_entry.sc_entry = copy.deepcopy(ytos_defect_gen.defect_entries["v_Y_0"])
            ytos_defect_gen["O_i_D2d_-1"] = fd_up_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)

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
            np.array([0.75, 0.25, 0.5]),
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance)  # nn_distance the same for each equiv
                # site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, ytos_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 2 == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                assert np.isclose(
                    np.product(list(charge_state_dict["probability_factors"].values())),
                    charge_state_dict["probability"],
                )
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in ytos_defect_gen.defect_entries
                        for defect_name in ytos_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in ytos_defect_gen.defect_entries
                        for defect_name in ytos_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )

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
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.ytos_defect_gen_info in output

        self._save_defect_gen_jsons(ytos_defect_gen)
        self.ytos_defect_gen_check(ytos_defect_gen)
        self._load_and_test_defect_gen_jsons(ytos_defect_gen)

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
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.ytos_defect_gen_info in output

        self._save_defect_gen_jsons(ytos_defect_gen)
        self.ytos_defect_gen_check(ytos_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(ytos_defect_gen)

    def lmno_defect_gen_check(self, lmno_defect_gen, generate_supercell=True):
        assert self.lmno_defect_gen_info in lmno_defect_gen._defect_generator_info()
        assert lmno_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
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
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
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
        assert (
            len(lmno_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 8
        )  # prim = conv cell in LMNO
        assert sum(vacancy.multiplicity for vacancy in lmno_defect_gen.defects["vacancies"]) == len(
            lmno_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(lmno_defect_gen.defect_entries) == 197
        assert len(lmno_defect_gen) == 197  # __len__()
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in lmno_defect_gen.defect_entries.values()
        )
        assert dict(lmno_defect_gen.items()) == lmno_defect_gen.defect_entries  # __iter__()
        assert str(lmno_defect_gen) == self.lmno_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(lmno_defect_gen)
            == self.lmno_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.lmno_defect_gen_info
        )
        assert all(
            defect_entry_name in lmno_defect_gen
            for defect_entry_name in lmno_defect_gen.defect_entries  # __contains__()
        )
        assert (
            lmno_defect_gen["Ni_i_C2_Li1.84O1.94_+2"]
            == lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"]
        )  # __getitem__()
        assert (
            lmno_defect_gen.get("Ni_i_C2_Li1.84O1.94_+2")
            == lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"]
        )  # get()
        defect_entry = lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"]
        del lmno_defect_gen["Ni_i_C2_Li1.84O1.94_+2"]  # __delitem__()
        assert "Ni_i_C2_Li1.84O1.94_+2" not in lmno_defect_gen
        lmno_defect_gen["Ni_i_C2_Li1.84O1.94_+2"] = defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            lmno_defect_gen["Ni_i_C2_Li1.84O1.94_+2"] = defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"])
            fd_up_defect_entry.defect.structure = self.cdte_bulk_supercell
            lmno_defect_gen["Ni_i_C2_Li1.84O1.94_+2"] = fd_up_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"])
            fd_up_defect_entry.sc_entry = copy.deepcopy(lmno_defect_gen.defect_entries["Li_O_C3_+3"])
            lmno_defect_gen["Ni_i_C2_Li1.84O1.94_+2"] = fd_up_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)

        # test defect entry attributes
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].name == "Ni_i_C2_Li1.84O1.94_+2"
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].charge_state == +2
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect.defect_type
            == DefectType.Interstitial
        )
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].wyckoff == "12d"
        assert lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect.wyckoff == "12d"
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect.multiplicity == 12
        )  # prim = conv structure in LMNO
        if generate_supercell:
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].sc_defect_frac_coords,
                np.array([0.4246, 0.4375, 0.5496]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect_supercell_site.frac_coords,
                np.array([0.4246, 0.4375, 0.5496]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        else:
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].sc_defect_frac_coords,
                np.array([0.15074, 0.375, 0.40074]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect_supercell_site.frac_coords,
                np.array([0.15074, 0.375, 0.40074]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect_supercell_site.specie.symbol
            == "Ni"
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].conv_cell_frac_coords,
            np.array([0.151, 0.375, 0.401]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect.conv_cell_frac_coords,
            np.array([0.151, 0.375, 0.401]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C2_Li1.84O1.94_+2"].defect.site.frac_coords,
            np.array([0.65074, 0.125, 0.59926]),
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance)  # nn_distance the same for each equiv
                # site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, lmno_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity == int(
                defect_entry.wyckoff[:-1]
            )  # prim = conv in LMNO
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                try:
                    assert np.isclose(
                        np.product(list(charge_state_dict["probability_factors"].values())),
                        charge_state_dict["probability"],
                    )
                except AssertionError as e:
                    struc_w_oxi = defect_entry.defect.structure.copy()
                    struc_w_oxi.add_oxidation_state_by_guess()
                    defect_elt_sites_in_struct = [
                        site
                        for site in struc_w_oxi
                        if site.specie.symbol == defect_entry.defect.site.specie.symbol
                    ]
                    defect_elt_oxi_in_struct = (
                        int(np.mean([site.specie.oxi_state for site in defect_elt_sites_in_struct]))
                        if defect_elt_sites_in_struct
                        else None
                    )
                    if (
                        defect_entry.defect.defect_type != DefectType.Substitution
                        or charge_state not in [-1, 0, 1]
                        or defect_elt_oxi_in_struct is None
                    ):
                        raise e

                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in lmno_defect_gen.defect_entries
                        for defect_name in lmno_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in lmno_defect_gen.defect_entries
                        for defect_name in lmno_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )

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
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value

        assert self.lmno_defect_gen_info in output

        self._save_defect_gen_jsons(lmno_defect_gen)
        self.lmno_defect_gen_check(lmno_defect_gen)
        self._load_and_test_defect_gen_jsons(lmno_defect_gen)

    def test_lmno_no_generate_supercell(self):
        # test inputting a non-diagonal supercell structure with a lattice vector <10 Å with
        # generate_supercell = False
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                lmno_defect_gen = DefectsGenerator(self.lmno_primitive, generate_supercell=False)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                assert len(non_ignored_warnings) == 1
                assert issubclass(non_ignored_warnings[-1].category, UserWarning)
                assert (
                    "Input structure is <10 Å in at least one direction (minimum image distance = 8.28 Å, "
                    "which is usually too small for accurate defect calculations, but "
                    "generate_supercell = False, so using input structure as defect & bulk supercells. "
                    "Caution advised!" in str(non_ignored_warnings[-1].message)
                )
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.lmno_defect_gen_info in output

        self._save_defect_gen_jsons(lmno_defect_gen)
        self.lmno_defect_gen_check(lmno_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(lmno_defect_gen)

    def zns_defect_gen_check(self, zns_defect_gen, generate_supercell=True):
        assert self.zns_defect_gen_info in zns_defect_gen._defect_generator_info()
        assert zns_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
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
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
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
        assert len(zns_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4  # 4x conv cell
        assert sum(vacancy.multiplicity for vacancy in zns_defect_gen.defects["vacancies"]) == len(
            zns_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(zns_defect_gen.defect_entries) == 36
        assert len(zns_defect_gen) == 36  # __len__()
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in zns_defect_gen.defect_entries.values()
        )
        assert dict(zns_defect_gen.items()) == zns_defect_gen.defect_entries  # __iter__()
        assert str(zns_defect_gen) == self.zns_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(zns_defect_gen)
            == self.zns_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.zns_defect_gen_info
        )
        assert all(
            defect_entry_name in zns_defect_gen
            for defect_entry_name in zns_defect_gen.defect_entries  # __contains__()
        )
        assert (
            zns_defect_gen["S_i_Td_S2.35_-2"] == zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"]
        )  # __getitem__()
        assert (
            zns_defect_gen.get("S_i_Td_S2.35_-2") == zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"]
        )  # get()
        defect_entry = zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"]
        del zns_defect_gen["S_i_Td_S2.35_-2"]  # __delitem__()
        assert "S_i_Td_S2.35_-2" not in zns_defect_gen
        zns_defect_gen["S_i_Td_S2.35_-2"] = defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            zns_defect_gen["S_i_Td_S2.35_-2"] = defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"])
            fd_up_defect_entry.defect.structure = self.cdte_bulk_supercell
            zns_defect_gen["S_i_Td_S2.35_-2"] = fd_up_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"])
            fd_up_defect_entry.sc_entry = copy.deepcopy(zns_defect_gen.defect_entries["v_Zn_0"])
            zns_defect_gen["S_i_Td_S2.35_-2"] = fd_up_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)

        # test defect entry attributes
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].name == "S_i_Td_S2.35_-2"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].charge_state == -2
        assert (
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.defect_type == DefectType.Interstitial
        )
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].wyckoff == "4b"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.wyckoff == "4b"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.multiplicity == 1
        if generate_supercell:
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].sc_defect_frac_coords,
                np.array([0.25, 0.5, 0.5]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect_supercell_site.frac_coords,
                np.array([0.25, 0.5, 0.5]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        else:
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].sc_defect_frac_coords,
                np.array([0.59375, 0.46875, 0.375]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect_supercell_site.frac_coords,
                np.array([0.59375, 0.46875, 0.375]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect_supercell_site.specie.symbol == "S"
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.site.frac_coords,
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance)  # nn_distance the same for each equiv
                # site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, zns_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 4 == int(
                defect_entry.wyckoff[:-1]
            )  # 4 prim cells in conv cell in Zinc Blende (ZnS, CdTe)
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                assert np.isclose(
                    np.product(list(charge_state_dict["probability_factors"].values())),
                    charge_state_dict["probability"],
                )
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in zns_defect_gen.defect_entries
                        for defect_name in zns_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in zns_defect_gen.defect_entries
                        for defect_name in zns_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )

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
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.zns_defect_gen_info in output

        self._save_defect_gen_jsons(zns_defect_gen)
        self.zns_defect_gen_check(zns_defect_gen)
        self._load_and_test_defect_gen_jsons(zns_defect_gen)

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
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert len(non_ignored_warnings) == 1
                assert issubclass(non_ignored_warnings[-1].category, UserWarning)
                assert (
                    "Input structure is <10 Å in at least one direction (minimum image distance = 7.59 Å, "
                    "which is usually too small for accurate defect calculations, but "
                    "generate_supercell = False, so using input structure as defect & bulk supercells. "
                    "Caution advised!" in str(non_ignored_warnings[-1].message)
                )
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.zns_defect_gen_info in output

        self._save_defect_gen_jsons(zns_defect_gen)
        self.zns_defect_gen_check(zns_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(zns_defect_gen)

    def cu_defect_gen_check(self, cu_defect_gen):
        assert self.cu_defect_gen_info in cu_defect_gen._defect_generator_info()
        assert cu_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(cu_defect_gen.primitive_structure, self.prim_cu)
        assert structure_matcher.fit(
            cu_defect_gen.primitive_structure, cu_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches (but below lattice doesn't match)
        assert np.allclose(cu_defect_gen.primitive_structure.lattice.matrix, self.prim_cu.lattice.matrix)

        np.testing.assert_allclose(
            cu_defect_gen.supercell_matrix, np.array([[-3, 3, 3], [3, -3, 3], [3, 3, -3]])
        )
        assert structure_matcher.fit(
            cu_defect_gen.primitive_structure * cu_defect_gen.supercell_matrix,
            cu_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (cu_defect_gen.primitive_structure * cu_defect_gen.supercell_matrix).lattice.matrix,
            cu_defect_gen.bulk_supercell.lattice.matrix,
        )
        assert structure_matcher.fit(cu_defect_gen.conventional_structure, self.prim_cu)
        sga = SpacegroupAnalyzer(self.prim_cu)
        assert np.allclose(
            cu_defect_gen.conventional_structure.lattice.matrix,
            sga.get_conventional_standard_structure().lattice.matrix,
        )

        # test defects
        assert len(cu_defect_gen.defects) == 2  # vacancies, NO substitutions, interstitials
        assert len(cu_defect_gen.defects["vacancies"]) == 1
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in cu_defect_gen.defects["vacancies"]
        )
        assert cu_defect_gen.defects.get("substitutions") is None
        assert len(cu_defect_gen.defects["interstitials"]) == 2
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in cu_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in cu_defect_gen.defects.values()
            for defect in defect_list
        )
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
            for defect_list in cu_defect_gen.defects.values()
            for defect in defect_list
        )

        # test some relevant defect attributes
        assert cu_defect_gen.defects["vacancies"][0].name == "v_Cu"
        assert cu_defect_gen.defects["vacancies"][0].oxi_state == 0
        assert cu_defect_gen.defects["vacancies"][0].multiplicity == 1
        assert cu_defect_gen.defects["vacancies"][0].defect_type == DefectType.Vacancy
        assert cu_defect_gen.defects["vacancies"][0].structure == cu_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            cu_defect_gen.defects["vacancies"][0].defect_structure.lattice.matrix,
            cu_defect_gen.primitive_structure.lattice.matrix,
        )
        assert np.allclose(
            cu_defect_gen.defects["vacancies"][0].site.frac_coords, np.array([0.0, 0.0, 0.0])
        )
        assert len(cu_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4  # 4x conv cell
        assert sum(vacancy.multiplicity for vacancy in cu_defect_gen.defects["vacancies"]) == len(
            cu_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(cu_defect_gen.defect_entries) == 9
        assert len(cu_defect_gen) == 9  # __len__()
        assert all(
            isinstance(defect_entry, DefectEntry) for defect_entry in cu_defect_gen.defect_entries.values()
        )
        assert dict(cu_defect_gen.items()) == cu_defect_gen.defect_entries  # __iter__()
        assert str(cu_defect_gen) == self.cu_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(cu_defect_gen)
            == self.cu_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.cu_defect_gen_info
        )
        assert all(
            defect_entry_name in cu_defect_gen
            for defect_entry_name in cu_defect_gen.defect_entries  # __contains__()
        )
        assert cu_defect_gen["Cu_i_Oh_+1"] == cu_defect_gen.defect_entries["Cu_i_Oh_+1"]  # __getitem__()
        assert cu_defect_gen.get("Cu_i_Oh_+1") == cu_defect_gen.defect_entries["Cu_i_Oh_+1"]  # get()
        defect_entry = cu_defect_gen.defect_entries["Cu_i_Oh_+1"]
        del cu_defect_gen["Cu_i_Oh_+1"]  # __delitem__()
        assert "Cu_i_Oh_+1" not in cu_defect_gen
        cu_defect_gen["Cu_i_Oh_+1"] = defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            cu_defect_gen["Cu_i_Oh_+1"] = defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(cu_defect_gen.defect_entries["Cu_i_Oh_+1"])
            fd_up_defect_entry.defect.structure = self.cdte_bulk_supercell
            cu_defect_gen["Cu_i_Oh_+1"] = fd_up_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(cu_defect_gen.defect_entries["Cu_i_Oh_+1"])
            fd_up_defect_entry.sc_entry = copy.deepcopy(cu_defect_gen.defect_entries["v_Cu_0"])
            cu_defect_gen["Cu_i_Oh_+1"] = fd_up_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)

        # test defect entry attributes
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].name == "Cu_i_Oh_+1"
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].charge_state == +1
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.defect_type == DefectType.Interstitial
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].wyckoff == "4b"
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.wyckoff == "4b"
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.multiplicity == 1
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].sc_defect_frac_coords,
            np.array([0.5, 0.5, 0.5]),  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect_supercell_site.frac_coords,
            np.array([0.5, 0.5, 0.5]),  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect_supercell_site.specie.symbol == "Cu"
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.site.frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )

        for defect_name, defect_entry in cu_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                cu_defect_gen.bulk_supercell.lattice.matrix,
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance)  # nn_distance the same for each equiv
                # site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, cu_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 4 == int(
                defect_entry.wyckoff[:-1]
            )  # 4 prim cells in conv cell in Cu
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                assert np.isclose(
                    np.product(list(charge_state_dict["probability_factors"].values())),
                    charge_state_dict["probability"],
                )
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in cu_defect_gen.defect_entries
                        for defect_name in cu_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in cu_defect_gen.defect_entries
                        for defect_name in cu_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )

        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.name == "v_Cu"
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.oxi_state == 0
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.multiplicity == 1
        assert cu_defect_gen.defect_entries["v_Cu_0"].wyckoff == "4a"
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.defect_type == DefectType.Vacancy
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.structure == cu_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            cu_defect_gen.defect_entries["v_Cu_0"].defect.defect_structure.lattice.matrix,
            cu_defect_gen.primitive_structure.lattice.matrix,
        )

        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["v_Cu_0"].conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["v_Cu_0"].defect.conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["v_Cu_0"].sc_defect_frac_coords,
            np.array([0.3333, 0.5, 0.5]),  # closest to middle of supercell
            atol=1e-4,
        )
        np.testing.assert_allclose(
            cu_defect_gen["v_Cu_0"].defect_supercell_site.frac_coords,
            np.array([0.3333, 0.5, 0.5]),  # closest to middle of supercell
            atol=1e-4,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["v_Cu_0"].defect.site.frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )

    def test_cu(self):
        # test inputting a single-element single-atom primitive cell -> zero oxidation states
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cu_defect_gen = DefectsGenerator(self.prim_cu)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cu_defect_gen_info in output

        self._save_defect_gen_jsons(cu_defect_gen)
        self.cu_defect_gen_check(cu_defect_gen)
        self._load_and_test_defect_gen_jsons(cu_defect_gen)

    def test_cu_no_generate_supercell(self):
        # test inputting a single-element single-atom primitive cell -> zero oxidation states
        with self.assertRaises(ValueError) as e:
            single_site_no_supercell_error = ValueError(
                "Input structure has only one site, so cannot generate defects without supercell (i.e. "
                "with generate_supercell=False)! Vacancy defect will give empty cell!"
            )
            DefectsGenerator(self.prim_cu, generate_supercell=False)
            assert single_site_no_supercell_error in e.exception

    def agcu_defect_gen_check(self, agcu_defect_gen, generate_supercell=True):
        assert self.agcu_defect_gen_info in agcu_defect_gen._defect_generator_info()
        assert agcu_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(agcu_defect_gen.primitive_structure, self.agcu)
        assert structure_matcher.fit(
            agcu_defect_gen.primitive_structure, agcu_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches (but below lattice doesn't match)
        assert not np.allclose(  # reduces from input supercell
            agcu_defect_gen.primitive_structure.lattice.matrix, self.agcu.lattice.matrix
        )

        if generate_supercell:
            np.testing.assert_allclose(
                agcu_defect_gen.supercell_matrix, np.array([[2, 2, 0], [-5, 5, 0], [-3, -3, 6]])
            )
        else:
            np.testing.assert_allclose(
                agcu_defect_gen.supercell_matrix, np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
            )
        assert structure_matcher.fit(
            agcu_defect_gen.primitive_structure * agcu_defect_gen.supercell_matrix,
            agcu_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (agcu_defect_gen.primitive_structure * agcu_defect_gen.supercell_matrix).lattice.matrix,
            agcu_defect_gen.bulk_supercell.lattice.matrix,
        )
        assert structure_matcher.fit(agcu_defect_gen.conventional_structure, self.agcu)
        sga = SpacegroupAnalyzer(self.agcu)
        assert np.allclose(
            agcu_defect_gen.conventional_structure.lattice.matrix,
            sga.get_conventional_standard_structure().lattice.matrix,
        )

        # test defects
        assert len(agcu_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(agcu_defect_gen.defects["vacancies"]) == 2
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in agcu_defect_gen.defects["vacancies"]
        )
        assert len(agcu_defect_gen.defects["substitutions"]) == 2
        assert all(
            defect.defect_type == DefectType.Substitution
            for defect in agcu_defect_gen.defects["substitutions"]
        )
        assert len(agcu_defect_gen.defects["interstitials"]) == 6
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in agcu_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in agcu_defect_gen.defects.values()
            for defect in defect_list
        )
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
            for defect_list in agcu_defect_gen.defects.values()
            for defect in defect_list
        )

        # test some relevant defect attributes
        assert agcu_defect_gen.defects["vacancies"][1].name == "v_Ag"
        assert agcu_defect_gen.defects["vacancies"][1].oxi_state == 0
        assert agcu_defect_gen.defects["vacancies"][1].multiplicity == 1
        assert agcu_defect_gen.defects["vacancies"][1].defect_type == DefectType.Vacancy
        assert agcu_defect_gen.defects["vacancies"][1].structure == agcu_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            agcu_defect_gen.defects["vacancies"][1].defect_structure.lattice.matrix,
            agcu_defect_gen.primitive_structure.lattice.matrix,
        )
        assert np.allclose(
            agcu_defect_gen.defects["vacancies"][1].site.frac_coords, np.array([0.5, 0.5, 0.5])
        )
        assert (
            len(agcu_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 3
        )  # 3x conv cell
        assert sum(vacancy.multiplicity for vacancy in agcu_defect_gen.defects["vacancies"]) == len(
            agcu_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(agcu_defect_gen.defect_entries) == 28
        assert len(agcu_defect_gen) == 28  # __len__()
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in agcu_defect_gen.defect_entries.values()
        )
        assert dict(agcu_defect_gen.items()) == agcu_defect_gen.defect_entries  # __iter__()
        assert str(agcu_defect_gen) == self.agcu_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(agcu_defect_gen)
            == self.agcu_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.agcu_defect_gen_info
        )
        assert all(
            defect_entry_name in agcu_defect_gen
            for defect_entry_name in agcu_defect_gen.defect_entries  # __contains__()
        )
        assert (
            agcu_defect_gen["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]
            == agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]
        )  # __getitem__()
        assert (
            agcu_defect_gen.get("Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1")
            == agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]
        )  # get()
        defect_entry = agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]
        del agcu_defect_gen["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]  # __delitem__()
        assert "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1" not in agcu_defect_gen
        agcu_defect_gen["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"] = defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            agcu_defect_gen["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"] = defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(
                agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]
            )
            fd_up_defect_entry.defect.structure = self.cdte_bulk_supercell
            agcu_defect_gen["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"] = fd_up_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )
        with self.assertRaises(ValueError) as e:
            fd_up_defect_entry = copy.deepcopy(
                agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"]
            )
            fd_up_defect_entry.sc_entry = copy.deepcopy(agcu_defect_gen.defect_entries["Ag_Cu_-1"])
            agcu_defect_gen["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"] = fd_up_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)

        # test defect entry attributes
        assert (
            agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].name
            == "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"
        )
        assert agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].charge_state == +1
        assert (
            agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].defect.defect_type
            == DefectType.Interstitial
        )
        assert agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].wyckoff == "6c"
        assert agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].defect.wyckoff == "6c"
        assert agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].defect.multiplicity == 2
        if generate_supercell:
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].sc_defect_frac_coords,
                np.array([0.5312, 0.5, 0.3958]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries[
                    "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"
                ].defect_supercell_site.frac_coords,
                np.array([0.5312, 0.5, 0.3958]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        else:
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].sc_defect_frac_coords,
                np.array([0.375, 0.375, 0.375]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries[
                    "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"
                ].defect_supercell_site.frac_coords,
                np.array([0.375, 0.375, 0.375]),  # closest to [0.5, 0.5, 0.5]
                rtol=1e-2,
            )
        assert (
            agcu_defect_gen.defect_entries[
                "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"
            ].defect_supercell_site.specie.symbol
            == "Cu"
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.375]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].defect.conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.375]),
            rtol=1e-2,
        )
        if generate_supercell:
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].defect.site.frac_coords,
                np.array([0.375, 0.375, 0.375]),
                rtol=1e-2,
            )
        else:  # rotated primitive structure to match input supercell
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries["Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_+1"].defect.site.frac_coords,
                np.array([0.625, 0.625, 0.625]),
                rtol=1e-2,
            )

        for defect_name, defect_entry in agcu_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                agcu_defect_gen.bulk_supercell.lattice.matrix,
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance, atol=0.01)  # nn_distance the same for
                # each equiv site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, agcu_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 3 == int(
                defect_entry.wyckoff[:-1]
            )  # 3 prim cells in conv cell
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                try:
                    assert np.isclose(
                        np.product(list(charge_state_dict["probability_factors"].values())),
                        charge_state_dict["probability"],
                    )
                except AssertionError as e:
                    struc_w_oxi = defect_entry.defect.structure.copy()
                    struc_w_oxi.add_oxidation_state_by_guess()
                    defect_elt_sites_in_struct = [
                        site
                        for site in struc_w_oxi
                        if site.specie.symbol == defect_entry.defect.site.specie.symbol
                    ]
                    defect_elt_oxi_in_struct = (
                        int(np.mean([site.specie.oxi_state for site in defect_elt_sites_in_struct]))
                        if defect_elt_sites_in_struct
                        else None
                    )
                    if (
                        defect_entry.defect.defect_type != DefectType.Substitution
                        or charge_state not in [-1, 0, 1]
                        or defect_elt_oxi_in_struct is None
                    ):
                        raise e

                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in agcu_defect_gen.defect_entries
                        for defect_name in agcu_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in agcu_defect_gen.defect_entries
                        for defect_name in agcu_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )

        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.name == "Ag_Cu"
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.oxi_state == 0
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.multiplicity == 1
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].wyckoff == "3a"
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.defect_type == DefectType.Substitution
        assert (
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.structure
            == agcu_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.defect_structure.lattice.matrix,
            agcu_defect_gen.primitive_structure.lattice.matrix,
        )

        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )
        if generate_supercell:
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries["Ag_Cu_-1"].sc_defect_frac_coords,
                np.array([0.5, 0.5, 0.5]),  # closest to middle of supercell
                atol=1e-4,
            )
            np.testing.assert_allclose(
                agcu_defect_gen["Ag_Cu_-1"].defect_supercell_site.frac_coords,
                np.array([0.5, 0.5, 0.5]),  # closest to middle of supercell
                atol=1e-4,
            )
        else:
            np.testing.assert_allclose(
                agcu_defect_gen.defect_entries["Ag_Cu_-1"].sc_defect_frac_coords,
                np.array([0.5, 0.0, 0.5]),  # closest to middle of supercell
                atol=1e-4,
            )
            np.testing.assert_allclose(
                agcu_defect_gen["Ag_Cu_-1"].defect_supercell_site.frac_coords,
                np.array([0.5, 0.0, 0.5]),  # closest to middle of supercell
                atol=1e-4,
            )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.site.frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )

    def test_agcu(self):
        # test initialising with an intermetallic (where pymatgen oxidation state guessing fails)
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                agcu_defect_gen = DefectsGenerator(self.agcu)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                # warnings.simplefilter("always")
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.agcu_defect_gen_info in output

        self._save_defect_gen_jsons(agcu_defect_gen)
        self.agcu_defect_gen_check(agcu_defect_gen)
        self._load_and_test_defect_gen_jsons(agcu_defect_gen)

    def test_agcu_no_generate_supercell(self):
        # test high-symmetry intermetallic with generate_supercell = False
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                agcu_defect_gen = DefectsGenerator(self.agcu, generate_supercell=False)
            non_ignored_warnings = [
                warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
            ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
            # warnings.simplefilter("always")
            assert len(non_ignored_warnings) == 1
            assert issubclass(non_ignored_warnings[-1].category, UserWarning)
            assert (
                "Input structure is <10 Å in at least one direction (minimum image distance = 4.42 Å, "
                "which is usually too small for accurate defect calculations, but "
                "generate_supercell = False, so using input structure as defect & bulk supercells. "
                "Caution advised!" in str(non_ignored_warnings[-1].message)
            )
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.agcu_defect_gen_info in output

        self._save_defect_gen_jsons(agcu_defect_gen)
        self.agcu_defect_gen_check(agcu_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(agcu_defect_gen)

    def cd_i_cdte_supercell_defect_gen_check(self, cd_i_defect_gen, generate_supercell=True):
        assert self.cd_i_cdte_supercell_defect_gen_info in cd_i_defect_gen._defect_generator_info()
        assert cd_i_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        structure_matcher = StructureMatcher(comparator=ElementComparator())  # ignore oxidation states
        assert structure_matcher.fit(
            cd_i_defect_gen.primitive_structure, cd_i_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches (but below lattice doesn't match)
        assert not structure_matcher.fit(cd_i_defect_gen.primitive_structure, self.prim_cdte)
        assert not structure_matcher.fit(cd_i_defect_gen.primitive_structure, self.cdte_bulk_supercell)
        assert np.allclose(  # primitive cell of defect supercell here is same as bulk supercell
            cd_i_defect_gen.primitive_structure.lattice.matrix, self.cdte_bulk_supercell.lattice.matrix
        )

        np.testing.assert_allclose(cd_i_defect_gen.supercell_matrix, np.eye(3), atol=1e-3)
        assert structure_matcher.fit(
            cd_i_defect_gen.primitive_structure * cd_i_defect_gen.supercell_matrix,
            cd_i_defect_gen.bulk_supercell,
        )
        assert np.allclose(
            (cd_i_defect_gen.primitive_structure * cd_i_defect_gen.supercell_matrix).lattice.matrix,
            cd_i_defect_gen.bulk_supercell.lattice.matrix,
        )

        # test defects
        assert len(cd_i_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cd_i_defect_gen.defects["vacancies"]) == 21
        assert all(
            defect.defect_type == DefectType.Vacancy for defect in cd_i_defect_gen.defects["vacancies"]
        )
        assert len(cd_i_defect_gen.defects["substitutions"]) == 21
        assert all(
            defect.defect_type == DefectType.Substitution
            for defect in cd_i_defect_gen.defects["substitutions"]
        )
        assert len(cd_i_defect_gen.defects["interstitials"]) == 94
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in cd_i_defect_gen.defects["interstitials"]
        )
        assert all(
            isinstance(defect, Defect)
            for defect_list in cd_i_defect_gen.defects.values()
            for defect in defect_list
        )
        assert all(
            defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
            for defect_list in cd_i_defect_gen.defects.values()
            for defect in defect_list
        )

        # test some relevant defect attributes
        assert cd_i_defect_gen.defects["vacancies"][1].name == "v_Cd"
        assert cd_i_defect_gen.defects["vacancies"][1].oxi_state == 0  # pmg fails oxi guessing with
        # defective supercell
        assert cd_i_defect_gen.defects["vacancies"][1].multiplicity == 3
        assert cd_i_defect_gen.defects["vacancies"][1].defect_type == DefectType.Vacancy
        assert cd_i_defect_gen.defects["vacancies"][1].structure == cd_i_defect_gen.primitive_structure
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            cd_i_defect_gen.defects["vacancies"][1].defect_structure.lattice.matrix,
            cd_i_defect_gen.primitive_structure.lattice.matrix,
        )
        assert np.allclose(
            cd_i_defect_gen.defects["vacancies"][1].site.frac_coords, np.array([0.0, 0.75, 0.25])
        )
        assert (
            len(cd_i_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 9
        )  # 3x conv cell
        assert sum(vacancy.multiplicity for vacancy in cd_i_defect_gen.defects["vacancies"]) == len(
            cd_i_defect_gen.primitive_structure
        )

        # test defect entries
        assert len(cd_i_defect_gen.defect_entries) == 429
        assert len(cd_i_defect_gen) == 429
        assert all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in cd_i_defect_gen.defect_entries.values()
        )

        # test defect entry attributes
        assert (
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].name
            == "Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"
        )
        assert cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].charge_state == 0
        assert (
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.defect_type
            == DefectType.Interstitial
        )
        assert cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].wyckoff == "9b"
        assert cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.wyckoff == "9b"
        assert cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.multiplicity == 3
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].sc_defect_frac_coords,
            np.array([0.875, 0.625, 0.625]),  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries[
                "Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"
            ].defect_supercell_site.frac_coords,
            np.array([0.875, 0.625, 0.625]),  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        assert (
            cd_i_defect_gen.defect_entries[
                "Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"
            ].defect_supercell_site.specie.symbol
            == "Te"
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].conv_cell_frac_coords,
            np.array([0.583, 0.167, 0.292]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.conv_cell_frac_coords,
            np.array([0.583, 0.167, 0.292]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.site.frac_coords,
            np.array([0.875, 0.625, 0.625]),
            rtol=1e-2,
        )

        for defect_name, defect_entry in cd_i_defect_gen.defect_entries.items():
            assert defect_entry.name == defect_name
            assert defect_entry.charge_state == int(defect_name.split("_")[-1])
            assert defect_entry.wyckoff  # wyckoff label is not None
            assert defect_entry.defect
            assert defect_entry.defect.wyckoff
            assert isinstance(defect_entry.conv_cell_frac_coords, np.ndarray)
            assert isinstance(defect_entry.defect.conv_cell_frac_coords, np.ndarray)
            np.testing.assert_allclose(
                defect_entry.sc_entry.structure.lattice.matrix,
                cd_i_defect_gen.bulk_supercell.lattice.matrix,
            )
            sga = SpacegroupAnalyzer(cd_i_defect_gen.structure)
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
            conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                defect_entry.conv_cell_frac_coords
            )
            nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                conv_cell_cart_coords,
                5,
            )
            nn_distances = np.array(
                [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
            )
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            assert nn_distance > 0.9  # default min_dist = 0.9
            for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                conv_cell_cart_coords = defect_entry.conventional_structure.lattice.get_cartesian_coords(
                    conv_cell_frac_coords
                )
                nearest_atoms = defect_entry.conventional_structure.get_sites_in_sphere(
                    conv_cell_cart_coords,
                    5,
                )
                nn_distances = np.array(
                    [nn.distance_from_point(conv_cell_cart_coords) for nn in nearest_atoms]
                )
                equiv_nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
                assert np.isclose(equiv_nn_distance, nn_distance)  # nn_distance the same for each equiv
                # site

            assert np.allclose(
                defect_entry.bulk_supercell.lattice.matrix, cd_i_defect_gen.bulk_supercell.lattice.matrix
            )
            assert defect_entry.defect.multiplicity * 3 == int(
                defect_entry.wyckoff[:-1]
            )  # 3 prim cells in conv cell
            assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
            assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
            for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
                assert equiv_conv_cell_frac_coords in defect_entry.defect.equiv_conv_cell_frac_coords
            assert defect_entry.defect_supercell_site
            assert defect_entry.bulk_entry is None
            assert defect_entry._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
            assert (
                defect_entry._BilbaoCS_conv_cell_vector_mapping
                == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
            )
            assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
            assert not defect_entry.corrections
            with self.assertRaises(KeyError):
                print(defect_entry.corrected_energy)

            # test charge state guessing:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                assert np.isclose(
                    np.product(list(charge_state_dict["probability_factors"].values())),
                    charge_state_dict["probability"],
                )
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in cd_i_defect_gen.defect_entries
                        for defect_name in cd_i_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    assert all(
                        defect_name not in cd_i_defect_gen.defect_entries
                        for defect_name in cd_i_defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )

        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.name == "Cd_Te"
        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.oxi_state == 0
        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.multiplicity == 3
        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].wyckoff == "9b"
        assert (
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.defect_type
            == DefectType.Substitution
        )
        assert (
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.structure
            == cd_i_defect_gen.primitive_structure
        )
        np.testing.assert_array_equal(  # test that defect structure uses primitive structure
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.defect_structure.lattice.matrix,
            cd_i_defect_gen.primitive_structure.lattice.matrix,
        )

        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].conv_cell_frac_coords,
            np.array([0.58333, 0.16666, 0.04167]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.conv_cell_frac_coords,
            np.array([0.58333, 0.16666, 0.04167]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].sc_defect_frac_coords,
            np.array([0.375, 0.375, 0.625]),  # closest to middle of supercell
            atol=1e-4,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen["Cd_Te_Cs_Cd2.71_-1"].defect_supercell_site.frac_coords,
            np.array([0.375, 0.375, 0.625]),  # closest to middle of supercell
            atol=1e-4,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.site.frac_coords,
            np.array([0.125, 0.375, 0.375]),
            atol=1e-3,
        )

    def test_supercell_w_defect_cd_i_cdte(self):
        # test inputting a defective supercell
        cdte_defect_gen = DefectsGenerator(self.prim_cdte)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cd_i_defect_gen = DefectsGenerator(cdte_defect_gen["Cd_i_C3v_0"].sc_entry.structure)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting

            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert not non_ignored_warnings
        assert self.cd_i_cdte_supercell_defect_gen_info in output

        self._save_defect_gen_jsons(cd_i_defect_gen)
        self.cd_i_cdte_supercell_defect_gen_check(cd_i_defect_gen)
        self._load_and_test_defect_gen_jsons(cd_i_defect_gen)

    def test_supercell_w_defect_cd_i_cdte_no_generate_supercell(self):
        # test inputting a defective supercell; input supercell is good here so same output
        cdte_defect_gen = DefectsGenerator(self.prim_cdte)

        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cd_i_defect_gen = DefectsGenerator(
                    cdte_defect_gen["Cd_i_C3v_0"].sc_entry.structure, generate_supercell=False
                )
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                assert not non_ignored_warnings
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.cd_i_cdte_supercell_defect_gen_info in output

        self._save_defect_gen_jsons(cd_i_defect_gen)
        self.cd_i_cdte_supercell_defect_gen_check(cd_i_defect_gen)
        self._load_and_test_defect_gen_jsons(cd_i_defect_gen)
