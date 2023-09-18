"""
Tests for the `doped.generation` module.

Implicitly tests the `doped.utils.wyckoff` module as well.
"""
import copy
import filecmp
import json
import os
import random
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
from pymatgen.analysis.defects.core import DefectType
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff

from doped.core import Defect, DefectEntry
from doped.generation import DefectsGenerator
from doped.utils.wyckoff import get_BCS_conventional_structure, swap_axes


def if_present_rm(path):
    """
    Remove the file/folder if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _potcars_available() -> bool:
    """
    Check if the POTCARs are available for the tests (i.e. testing locally).
    """
    from doped.vasp import _test_potcar_functional_choice

    try:
        _test_potcar_functional_choice("PBE")
        return True
    except ValueError:
        return False


class DefectsGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.prim_cdte = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        sga = SpacegroupAnalyzer(self.prim_cdte)
        self.conv_cdte = sga.get_conventional_standard_structure()
        self.fd_up_sc_entry = ComputedStructureEntry(self.conv_cdte, 420, correction=0.0)  # for testing
        # in _check_editing_defect_gen() later
        self.structure_matcher = StructureMatcher(
            comparator=ElementComparator()
        )  # ignore oxidation states
        self.cdte_bulk_supercell = self.conv_cdte * 2 * np.eye(3)
        self.cdte_defect_gen_string = (
            "DefectsGenerator for input composition CdTe, space group F-43m with 50 defect entries "
            "created."
        )
        self.cdte_defect_gen_info = (
            """Vacancies    Guessed Charges    Conv. Cell Coords    Wyckoff
-----------  -----------------  -------------------  ---------
v_Cd         [-2,-1,0,+1]       [0.000,0.000,0.000]  4a
v_Te         [-1,0,+1,+2]       [0.250,0.250,0.250]  4c

Substitutions    Guessed Charges        Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Cd_Te            [0,+1,+2,+3,+4]        [0.250,0.250,0.250]  4c
Te_Cd            [-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.000]  4a

Interstitials    Guessed Charges        Conv. Cell Coords    Wyckoff
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
            """Vacancies    Guessed Charges     Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Y          [-3,-2,-1,0,+1]     [0.000,0.000,0.334]  4e
v_Ti         [-4,-3,-2,-1,0,+1]  [0.000,0.000,0.079]  4e
v_S          [-1,0,+1,+2]        [0.000,0.000,0.205]  4e
v_O_C2v      [-1,0,+1,+2]        [0.000,0.500,0.099]  8g
v_O_D4h      [-1,0,+1,+2]        [0.000,0.000,0.000]  2a

Substitutions    Guessed Charges              Conv. Cell Coords    Wyckoff
---------------  ---------------------------  -------------------  ---------
Y_Ti             [-1,0]                       [0.000,0.000,0.079]  4e
Y_S              [0,+1,+2,+3,+4,+5]           [0.000,0.000,0.205]  4e
Y_O_C2v          [0,+1,+2,+3,+4,+5]           [0.000,0.500,0.099]  8g
Y_O_D4h          [0,+1,+2,+3,+4,+5]           [0.000,0.000,0.000]  2a
Ti_Y             [-1,0,+1]                    [0.000,0.000,0.334]  4e
Ti_S             [0,+1,+2,+3,+4,+5,+6]        [0.000,0.000,0.205]  4e
Ti_O_C2v         [0,+1,+2,+3,+4,+5,+6]        [0.000,0.500,0.099]  8g
Ti_O_D4h         [0,+1,+2,+3,+4,+5,+6]        [0.000,0.000,0.000]  2a
S_Y              [-5,-4,-3,-2,-1,0,+1,+2,+3]  [0.000,0.000,0.334]  4e
S_Ti             [-6,-5,-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.079]  4e
S_O_C2v          [-1,0,+1]                    [0.000,0.500,0.099]  8g
S_O_D4h          [-1,0,+1]                    [0.000,0.000,0.000]  2a
O_Y              [-5,-4,-3,-2,-1,0]           [0.000,0.000,0.334]  4e
O_Ti             [-6,-5,-4,-3,-2,-1,0]        [0.000,0.000,0.079]  4e
O_S              [-1,0,+1]                    [0.000,0.000,0.205]  4e

Interstitials    Guessed Charges        Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Y_i_C2v          [0,+1,+2,+3]           [0.000,0.500,0.184]  8g
Y_i_C4v_O2.68    [0,+1,+2,+3]           [0.000,0.000,0.485]  4e
Y_i_C4v_Y1.92    [0,+1,+2,+3]           [0.000,0.000,0.418]  4e
Y_i_Cs_S1.71     [0,+1,+2,+3]           [0.180,0.180,0.143]  16m
Y_i_Cs_Ti1.95    [0,+1,+2,+3]           [0.325,0.325,0.039]  16m
Y_i_D2d          [0,+1,+2,+3]           [0.000,0.500,0.250]  4d
Ti_i_C2v         [0,+1,+2,+3,+4]        [0.000,0.500,0.184]  8g
Ti_i_C4v_O2.68   [0,+1,+2,+3,+4]        [0.000,0.000,0.485]  4e
Ti_i_C4v_Y1.92   [0,+1,+2,+3,+4]        [0.000,0.000,0.418]  4e
Ti_i_Cs_S1.71    [0,+1,+2,+3,+4]        [0.180,0.180,0.143]  16m
Ti_i_Cs_Ti1.95   [0,+1,+2,+3,+4]        [0.325,0.325,0.039]  16m
Ti_i_D2d         [0,+1,+2,+3,+4]        [0.000,0.500,0.250]  4d
S_i_C2v          [-2,-1,0,+1,+2,+3,+4]  [0.000,0.500,0.184]  8g
S_i_C4v_O2.68    [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.485]  4e
S_i_C4v_Y1.92    [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.418]  4e
S_i_Cs_S1.71     [-2,-1,0,+1,+2,+3,+4]  [0.180,0.180,0.143]  16m
S_i_Cs_Ti1.95    [-2,-1,0,+1,+2,+3,+4]  [0.325,0.325,0.039]  16m
S_i_D2d          [-2,-1,0,+1,+2,+3,+4]  [0.000,0.500,0.250]  4d
O_i_C2v          [-2,-1,0]              [0.000,0.500,0.184]  8g
O_i_C4v_O2.68    [-2,-1,0]              [0.000,0.000,0.485]  4e
O_i_C4v_Y1.92    [-2,-1,0]              [0.000,0.000,0.418]  4e
O_i_Cs_S1.71     [-2,-1,0]              [0.180,0.180,0.143]  16m
O_i_Cs_Ti1.95    [-2,-1,0]              [0.325,0.325,0.039]  16m
O_i_D2d          [-2,-1,0]              [0.000,0.500,0.250]  4d
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 2 formula unit(s) of Y2Ti2S2O5.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.lmno_defect_gen_string = (
            "DefectsGenerator for input composition Li2Mn3NiO8, space group P4_332 with 182 defect "
            "entries created."
        )
        self.lmno_defect_gen_info = (
            """Vacancies    Guessed Charges     Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Li         [-1,0,+1]           [0.004,0.004,0.004]  8c
v_Mn         [-4,-3,-2,-1,0,+1]  [0.121,0.129,0.625]  12d
v_Ni         [-2,-1,0,+1]        [0.625,0.625,0.625]  4b
v_O_C1       [-1,0,+1,+2]        [0.101,0.124,0.392]  24e
v_O_C3       [-1,0,+1,+2]        [0.385,0.385,0.385]  8c

Substitutions    Guessed Charges        Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Li_Mn            [-3,-2,-1,0]           [0.121,0.129,0.625]  12d
Li_Ni            [-1,0]                 [0.625,0.625,0.625]  4b
Li_O_C1          [0,+1,+2,+3]           [0.101,0.124,0.392]  24e
Li_O_C3          [0,+1,+2,+3]           [0.385,0.385,0.385]  8c
Mn_Li            [0,+1,+2,+3]           [0.004,0.004,0.004]  8c
Mn_Ni            [0,+1,+2]              [0.625,0.625,0.625]  4b
Mn_O_C1          [0,+1,+2,+3,+4,+5,+6]  [0.101,0.124,0.392]  24e
Mn_O_C3          [0,+1,+2,+3,+4,+5,+6]  [0.385,0.385,0.385]  8c
Ni_Li            [0,+1,+2,+3]           [0.004,0.004,0.004]  8c
Ni_Mn            [-3,-2,-1,0]           [0.121,0.129,0.625]  12d
Ni_O_C1          [0,+1,+2,+3,+4,+5]     [0.101,0.124,0.392]  24e
Ni_O_C3          [0,+1,+2,+3,+4,+5]     [0.385,0.385,0.385]  8c
O_Li             [-3,-2,-1,0]           [0.004,0.004,0.004]  8c
O_Mn             [-6,-5,-4,-3,-2,-1,0]  [0.121,0.129,0.625]  12d
O_Ni             [-4,-3,-2,-1,0]        [0.625,0.625,0.625]  4b

Interstitials    Guessed Charges    Conv. Cell Coords    Wyckoff
---------------  -----------------  -------------------  ---------
Li_i_C1_O1.72    [0,+1]             [0.248,0.480,0.249]  24e
Li_i_C1_O1.78    [0,+1]             [0.017,0.261,0.250]  24e
Li_i_C2_Li1.83   [0,+1]             [0.077,0.125,0.173]  12d
Li_i_C2_Li1.84   [0,+1]             [0.151,0.375,0.401]  12d
Li_i_C2_Li1.86   [0,+1]             [0.086,0.375,0.336]  12d
Li_i_C3          [0,+1]             [0.497,0.497,0.497]  8c
Mn_i_C1_O1.72    [0,+1,+2,+3,+4]    [0.248,0.480,0.249]  24e
Mn_i_C1_O1.78    [0,+1,+2,+3,+4]    [0.017,0.261,0.250]  24e
Mn_i_C2_Li1.83   [0,+1,+2,+3,+4]    [0.077,0.125,0.173]  12d
Mn_i_C2_Li1.84   [0,+1,+2,+3,+4]    [0.151,0.375,0.401]  12d
Mn_i_C2_Li1.86   [0,+1,+2,+3,+4]    [0.086,0.375,0.336]  12d
Mn_i_C3          [0,+1,+2,+3,+4]    [0.497,0.497,0.497]  8c
Ni_i_C1_O1.72    [0,+1,+2,+3,+4]    [0.248,0.480,0.249]  24e
Ni_i_C1_O1.78    [0,+1,+2,+3,+4]    [0.017,0.261,0.250]  24e
Ni_i_C2_Li1.83   [0,+1,+2,+3,+4]    [0.077,0.125,0.173]  12d
Ni_i_C2_Li1.84   [0,+1,+2,+3,+4]    [0.151,0.375,0.401]  12d
Ni_i_C2_Li1.86   [0,+1,+2,+3,+4]    [0.086,0.375,0.336]  12d
Ni_i_C3          [0,+1,+2,+3,+4]    [0.497,0.497,0.497]  8c
O_i_C1_O1.72     [-2,-1,0]          [0.248,0.480,0.249]  24e
O_i_C1_O1.78     [-2,-1,0]          [0.017,0.261,0.250]  24e
O_i_C2_Li1.83    [-2,-1,0]          [0.077,0.125,0.173]  12d
O_i_C2_Li1.84    [-2,-1,0]          [0.151,0.375,0.401]  12d
O_i_C2_Li1.86    [-2,-1,0]          [0.086,0.375,0.336]  12d
O_i_C3           [-2,-1,0]          [0.497,0.497,0.497]  8c
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 4 formula unit(s) of Li2Mn3NiO8.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")
        self.zns_defect_gen_string = (
            "DefectsGenerator for input composition ZnS, space group F-43m with 44 defect entries "
            "created."
        )
        self.zns_defect_gen_info = (
            """Vacancies    Guessed Charges    Conv. Cell Coords    Wyckoff
-----------  -----------------  -------------------  ---------
v_Zn         [-2,-1,0,+1]       [0.000,0.000,0.000]  4a
v_S          [-1,0,+1,+2]       [0.250,0.250,0.250]  4c

Substitutions    Guessed Charges        Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Zn_S             [0,+1,+2,+3,+4]        [0.250,0.250,0.250]  4c
S_Zn             [-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.000]  4a

Interstitials    Guessed Charges    Conv. Cell Coords    Wyckoff
---------------  -----------------  -------------------  ---------
Zn_i_C3v         [0,+1,+2]          [0.625,0.625,0.625]  16e
Zn_i_Td_S2.35    [0,+1,+2]          [0.500,0.500,0.500]  4b
Zn_i_Td_Zn2.35   [0,+1,+2]          [0.750,0.750,0.750]  4d
S_i_C3v          [-2,-1,0,+1,+2]    [0.625,0.625,0.625]  16e
S_i_Td_S2.35     [-2,-1,0,+1,+2]    [0.500,0.500,0.500]  4b
S_i_Td_Zn2.35    [-2,-1,0,+1,+2]    [0.750,0.750,0.750]  4d
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
            """Vacancies    Guessed Charges    Conv. Cell Coords    Wyckoff
-----------  -----------------  -------------------  ---------
v_Cu         [-1,0,+1]          [0.000,0.000,0.000]  4a

Interstitials    Guessed Charges    Conv. Cell Coords    Wyckoff
---------------  -----------------  -------------------  ---------
Cu_i_Oh          [0,+1,+2]          [0.500,0.500,0.500]  4b
Cu_i_Td          [0,+1,+2]          [0.250,0.250,0.250]  8c
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
            """Vacancies    Guessed Charges    Conv. Cell Coords    Wyckoff
-----------  -----------------  -------------------  ---------
v_Cu         [-1,0,+1]          [0.000,0.000,0.000]  3a
v_Ag         [-1,0,+1]          [0.000,0.000,0.500]  3b

Substitutions    Guessed Charges    Conv. Cell Coords    Wyckoff
---------------  -----------------  -------------------  ---------
Cu_Ag            [-1,0,+1,+2]       [0.000,0.000,0.500]  3b
Ag_Cu            [-1,0,+1]          [0.000,0.000,0.000]  3a

Interstitials                 Guessed Charges    Conv. Cell Coords    Wyckoff
----------------------------  -----------------  -------------------  ---------
Cu_i_C3v_Cu1.56Ag1.56Cu2.99a  [0,+1,+2]          [0.000,0.000,0.125]  6c
Cu_i_C3v_Cu1.56Ag1.56Cu2.99b  [0,+1,+2]          [0.000,0.000,0.375]  6c
Cu_i_C3v_Cu1.80               [0,+1,+2]          [0.000,0.000,0.250]  6c
Ag_i_C3v_Cu1.56Ag1.56Cu2.99a  [0,+1]             [0.000,0.000,0.125]  6c
Ag_i_C3v_Cu1.56Ag1.56Cu2.99b  [0,+1]             [0.000,0.000,0.375]  6c
Ag_i_C3v_Cu1.80               [0,+1]             [0.000,0.000,0.250]  6c
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 3 formula unit(s) of AgCu.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.cd_i_cdte_supercell_defect_gen_info = (
            """Vacancies                    Guessed Charges    Conv. Cell Coords    Wyckoff
---------------------------  -----------------  -------------------  ---------
v_Cd_C1_Te2.83Cd4.62Te5.42a  [-1,0,+1]          [0.000,0.250,0.000]  18c
v_Cd_C1_Te2.83Cd4.62Te5.42b  [-1,0,+1]          [0.250,0.250,0.500]  18c
v_Cd_C3v_Cd2.71              [-1,0,+1]          [0.000,0.000,0.688]  3a
v_Cd_C3v_Te2.83Cd4.25        [-1,0,+1]          [0.000,0.000,0.500]  3a
v_Cd_C3v_Te2.83Cd4.62        [-1,0,+1]          [0.000,0.000,0.000]  3a
v_Cd_Cs_Cd2.71               [-1,0,+1]          [0.500,0.250,0.000]  9b
v_Cd_Cs_Te2.83Cd4.25         [-1,0,+1]          [0.583,0.167,0.167]  9b
v_Cd_Cs_Te2.83Cd4.62Cd5.36   [-1,0,+1]          [0.000,0.500,0.000]  9b
v_Cd_Cs_Te2.83Cd4.62Te5.42a  [-1,0,+1]          [0.500,0.500,0.500]  9b
v_Cd_Cs_Te2.83Cd4.62Te5.42b  [-1,0,+1]          [0.083,0.167,0.167]  9b
v_Cd_Cs_Te2.83Cd4.62Te5.42c  [-1,0,+1]          [0.167,0.083,0.333]  9b
v_Te_C1_Cd2.83Te4.62Cd5.42a  [-1,0,+1]          [0.250,0.250,0.375]  18c
v_Te_C1_Cd2.83Te4.62Cd5.42b  [-1,0,+1]          [0.250,0.250,0.875]  18c
v_Te_C3v_Cd2.83Cd4.25        [-1,0,+1]          [0.000,0.000,0.875]  3a
v_Te_C3v_Cd2.83Te4.62        [-1,0,+1]          [0.000,0.000,0.375]  3a
v_Te_Cs_Cd2.71               [-1,0,+1]          [0.583,0.167,0.042]  9b
v_Te_Cs_Cd2.83Cd4.25         [-1,0,+1]          [0.083,0.167,0.542]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.36   [-1,0,+1]          [0.500,0.500,0.375]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.42a  [-1,0,+1]          [0.500,0.500,0.875]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.42b  [-1,0,+1]          [0.083,0.167,0.042]  9b
v_Te_Cs_Cd2.83Te4.62Cd5.42c  [-1,0,+1]          [0.167,0.083,0.208]  9b

Substitutions                 Guessed Charges        Conv. Cell Coords    Wyckoff
----------------------------  ---------------------  -------------------  ---------
Cd_Te_C1_Cd2.83Te4.62Cd5.42a  [-1,0,+1,+2]           [0.250,0.250,0.375]  18c
Cd_Te_C1_Cd2.83Te4.62Cd5.42b  [-1,0,+1,+2]           [0.250,0.250,0.875]  18c
Cd_Te_C3v_Cd2.83Cd4.25        [-1,0,+1,+2]           [0.000,0.000,0.875]  3a
Cd_Te_C3v_Cd2.83Te4.62        [-1,0,+1,+2]           [0.000,0.000,0.375]  3a
Cd_Te_Cs_Cd2.71               [-1,0,+1,+2]           [0.583,0.167,0.042]  9b
Cd_Te_Cs_Cd2.83Cd4.25         [-1,0,+1,+2]           [0.083,0.167,0.542]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.36   [-1,0,+1,+2]           [0.500,0.500,0.375]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.42a  [-1,0,+1,+2]           [0.500,0.500,0.875]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.42b  [-1,0,+1,+2]           [0.083,0.167,0.042]  9b
Cd_Te_Cs_Cd2.83Te4.62Cd5.42c  [-1,0,+1,+2]           [0.167,0.083,0.208]  9b
Te_Cd_C1_Te2.83Cd4.62Te5.42a  [-2,-1,0,+1,+2,+3,+4]  [0.000,0.250,0.000]  18c
Te_Cd_C1_Te2.83Cd4.62Te5.42b  [-2,-1,0,+1,+2,+3,+4]  [0.250,0.250,0.500]  18c
Te_Cd_C3v_Cd2.71              [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.688]  3a
Te_Cd_C3v_Te2.83Cd4.25        [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.500]  3a
Te_Cd_C3v_Te2.83Cd4.62        [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.000]  3a
Te_Cd_Cs_Cd2.71               [-2,-1,0,+1,+2,+3,+4]  [0.500,0.250,0.000]  9b
Te_Cd_Cs_Te2.83Cd4.25         [-2,-1,0,+1,+2,+3,+4]  [0.583,0.167,0.167]  9b
Te_Cd_Cs_Te2.83Cd4.62Cd5.36   [-2,-1,0,+1,+2,+3,+4]  [0.000,0.500,0.000]  9b
Te_Cd_Cs_Te2.83Cd4.62Te5.42a  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.500]  9b
Te_Cd_Cs_Te2.83Cd4.62Te5.42b  [-2,-1,0,+1,+2,+3,+4]  [0.083,0.167,0.167]  9b
Te_Cd_Cs_Te2.83Cd4.62Te5.42c  [-2,-1,0,+1,+2,+3,+4]  [0.167,0.083,0.333]  9b

Interstitials                Guessed Charges        Conv. Cell Coords    Wyckoff
---------------------------  ---------------------  -------------------  ---------
Cd_i_C1_Cd2.71Te2.71Cd4.00a  [0,+1,+2]              [0.458,0.167,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.00b  [0,+1,+2]              [0.167,0.458,0.271]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25a  [0,+1,+2]              [0.250,0.250,0.188]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25b  [0,+1,+2]              [0.125,0.125,0.438]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25c  [0,+1,+2]              [0.375,0.375,0.438]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25d  [0,+1,+2]              [0.250,0.250,0.688]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25e  [0,+1,+2]              [0.125,0.125,0.938]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25f  [0,+1,+2]              [0.375,0.375,0.938]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25g  [0,+1,+2]              [0.208,0.042,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25h  [0,+1,+2]              [0.083,0.292,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25i  [0,+1,+2]              [0.042,0.208,0.271]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25j  [0,+1,+2]              [0.292,0.083,0.271]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25k  [0,+1,+2]              [0.458,0.042,0.104]  18c
Cd_i_C1_Cd2.71Te2.71Cd4.25l  [0,+1,+2]              [0.042,0.458,0.271]  18c
Cd_i_C1_Cd2.83Te3.27Cd4.84   [0,+1,+2]              [0.250,0.250,0.625]  18c
Cd_i_C1_Cd2.83Te3.27Cd5.42   [0,+1,+2]              [0.250,0.250,0.125]  18c
Cd_i_C1_Te2.83Cd3.27Cd4.84   [0,+1,+2]              [0.750,0.750,0.750]  18c
Cd_i_C1_Te2.83Cd3.27Te5.42   [0,+1,+2]              [0.250,0.250,0.250]  18c
Cd_i_C3v_Cd2.71              [0,+1,+2]              [0.000,0.000,0.188]  3a
Cd_i_C3v_Cd2.83              [0,+1,+2]              [0.000,0.000,0.125]  3a
Cd_i_C3v_Te2.83              [0,+1,+2]              [0.000,0.000,0.250]  3a
Cd_i_Cs_Cd2.59Cd2.65         [0,+1,+2]              [0.563,0.281,0.108]  9b
Cd_i_Cs_Cd2.59Te2.65         [0,+1,+2]              [0.104,0.052,0.600]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25a  [0,+1,+2]              [0.500,0.500,0.188]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25b  [0,+1,+2]              [0.500,0.500,0.688]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25c  [0,+1,+2]              [0.083,0.042,0.104]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25d  [0,+1,+2]              [0.167,0.083,0.021]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25e  [0,+1,+2]              [0.042,0.083,0.271]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25f  [0,+1,+2]              [0.083,0.167,0.354]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25g  [0,+1,+2]              [0.208,0.417,0.104]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25h  [0,+1,+2]              [0.125,0.250,0.438]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25i  [0,+1,+2]              [0.417,0.208,0.271]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25j  [0,+1,+2]              [0.167,0.083,0.521]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25k  [0,+1,+2]              [0.500,0.250,0.188]  9b
Cd_i_Cs_Cd2.71Te2.71Cd4.25l  [0,+1,+2]              [0.542,0.083,0.271]  9b
Cd_i_Cs_Cd2.83Te3.27Cd3.56   [0,+1,+2]              [0.500,0.250,0.125]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42a  [0,+1,+2]              [0.500,0.500,0.125]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42b  [0,+1,+2]              [0.500,0.500,0.625]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42c  [0,+1,+2]              [0.083,0.167,0.292]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42d  [0,+1,+2]              [0.167,0.083,0.458]  9b
Cd_i_Cs_Cd2.83Te3.27Cd5.42e  [0,+1,+2]              [0.583,0.167,0.292]  9b
Cd_i_Cs_Te2.83Cd3.27Cd3.56   [0,+1,+2]              [0.250,0.500,0.250]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42a  [0,+1,+2]              [0.167,0.083,0.083]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42b  [0,+1,+2]              [0.500,0.250,0.250]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42c  [0,+1,+2]              [0.500,0.500,0.250]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42d  [0,+1,+2]              [0.500,0.500,0.750]  9b
Cd_i_Cs_Te2.83Cd3.27Te5.42e  [0,+1,+2]              [0.750,0.250,0.750]  9b
Te_i_C1_Cd2.71Te2.71Cd4.00a  [-2,-1,0,+1,+2,+3,+4]  [0.458,0.167,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.00b  [-2,-1,0,+1,+2,+3,+4]  [0.167,0.458,0.271]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25a  [-2,-1,0,+1,+2,+3,+4]  [0.250,0.250,0.188]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25b  [-2,-1,0,+1,+2,+3,+4]  [0.125,0.125,0.438]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25c  [-2,-1,0,+1,+2,+3,+4]  [0.375,0.375,0.438]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25d  [-2,-1,0,+1,+2,+3,+4]  [0.250,0.250,0.688]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25e  [-2,-1,0,+1,+2,+3,+4]  [0.125,0.125,0.938]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25f  [-2,-1,0,+1,+2,+3,+4]  [0.375,0.375,0.938]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25g  [-2,-1,0,+1,+2,+3,+4]  [0.208,0.042,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25h  [-2,-1,0,+1,+2,+3,+4]  [0.083,0.292,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25i  [-2,-1,0,+1,+2,+3,+4]  [0.042,0.208,0.271]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25j  [-2,-1,0,+1,+2,+3,+4]  [0.292,0.083,0.271]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25k  [-2,-1,0,+1,+2,+3,+4]  [0.458,0.042,0.104]  18c
Te_i_C1_Cd2.71Te2.71Cd4.25l  [-2,-1,0,+1,+2,+3,+4]  [0.042,0.458,0.271]  18c
Te_i_C1_Cd2.83Te3.27Cd4.84   [-2,-1,0,+1,+2,+3,+4]  [0.250,0.250,0.625]  18c
Te_i_C1_Cd2.83Te3.27Cd5.42   [-2,-1,0,+1,+2,+3,+4]  [0.250,0.250,0.125]  18c
Te_i_C1_Te2.83Cd3.27Cd4.84   [-2,-1,0,+1,+2,+3,+4]  [0.750,0.750,0.750]  18c
Te_i_C1_Te2.83Cd3.27Te5.42   [-2,-1,0,+1,+2,+3,+4]  [0.250,0.250,0.250]  18c
Te_i_C3v_Cd2.71              [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.188]  3a
Te_i_C3v_Cd2.83              [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.125]  3a
Te_i_C3v_Te2.83              [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.250]  3a
Te_i_Cs_Cd2.59Cd2.65         [-2,-1,0,+1,+2,+3,+4]  [0.563,0.281,0.108]  9b
Te_i_Cs_Cd2.59Te2.65         [-2,-1,0,+1,+2,+3,+4]  [0.104,0.052,0.600]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25a  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.188]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25b  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.688]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25c  [-2,-1,0,+1,+2,+3,+4]  [0.083,0.042,0.104]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25d  [-2,-1,0,+1,+2,+3,+4]  [0.167,0.083,0.021]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25e  [-2,-1,0,+1,+2,+3,+4]  [0.042,0.083,0.271]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25f  [-2,-1,0,+1,+2,+3,+4]  [0.083,0.167,0.354]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25g  [-2,-1,0,+1,+2,+3,+4]  [0.208,0.417,0.104]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25h  [-2,-1,0,+1,+2,+3,+4]  [0.125,0.250,0.438]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25i  [-2,-1,0,+1,+2,+3,+4]  [0.417,0.208,0.271]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25j  [-2,-1,0,+1,+2,+3,+4]  [0.167,0.083,0.521]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25k  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.250,0.188]  9b
Te_i_Cs_Cd2.71Te2.71Cd4.25l  [-2,-1,0,+1,+2,+3,+4]  [0.542,0.083,0.271]  9b
Te_i_Cs_Cd2.83Te3.27Cd3.56   [-2,-1,0,+1,+2,+3,+4]  [0.500,0.250,0.125]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42a  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.125]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42b  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.625]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42c  [-2,-1,0,+1,+2,+3,+4]  [0.083,0.167,0.292]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42d  [-2,-1,0,+1,+2,+3,+4]  [0.167,0.083,0.458]  9b
Te_i_Cs_Cd2.83Te3.27Cd5.42e  [-2,-1,0,+1,+2,+3,+4]  [0.583,0.167,0.292]  9b
Te_i_Cs_Te2.83Cd3.27Cd3.56   [-2,-1,0,+1,+2,+3,+4]  [0.250,0.500,0.250]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42a  [-2,-1,0,+1,+2,+3,+4]  [0.167,0.083,0.083]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42b  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.250,0.250]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42c  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.250]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42d  [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.750]  9b
Te_i_Cs_Te2.83Cd3.27Te5.42e  [-2,-1,0,+1,+2,+3,+4]  [0.750,0.250,0.750]  9b
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 3 formula unit(s) of Cd33Te32.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.N_doped_diamond_supercell = Structure.from_file(f"{self.data_dir}/N_C_diamond_POSCAR")

        self.N_diamond_defect_gen_info = (
            """Vacancies                 Guessed Charges    Conv. Cell Coords    Wyckoff
------------------------  -----------------  -------------------  ---------
v_C_C1_C1.54C2.52C2.95a   [-1,0,+1]          [0.167,0.167,0.056]  18c
v_C_C1_C1.54C2.52C2.95b   [-1,0,+1]          [0.167,0.167,0.139]  18c
v_C_C1_C1.54C2.52C2.95c   [-1,0,+1]          [0.056,0.278,0.278]  18c
v_C_C1_C1.54C2.52C2.95d   [-1,0,+1]          [0.167,0.167,0.389]  18c
v_C_C1_C1.54C2.52C2.95e   [-1,0,+1]          [0.333,0.333,0.056]  18c
v_C_C1_C1.54C2.52C2.95f   [-1,0,+1]          [0.333,0.333,0.139]  18c
v_C_C1_C1.54C2.52C2.95g   [-1,0,+1]          [0.389,0.278,0.278]  18c
v_C_C1_C1.54C2.52C2.95h   [-1,0,+1]          [0.167,0.167,0.722]  18c
v_C_C1_C1.54C2.52C2.95i   [-1,0,+1]          [0.167,0.167,0.806]  18c
v_C_C1_C1.54C2.52C2.95j   [-1,0,+1]          [0.056,0.278,0.028]  18c
v_C_C1_C1.54C2.52C2.95k   [-1,0,+1]          [0.278,0.055,0.167]  18c
v_C_C1_C1.54C2.52C2.95l   [-1,0,+1]          [0.278,0.056,0.250]  18c
v_C_C1_C1.54C2.52C2.95m   [-1,0,+1]          [0.389,0.111,0.028]  18c
v_C_C1_C1.54C2.52C2.95n   [-1,0,+1]          [0.111,0.389,0.167]  18c
v_C_C1_C1.54C2.52C2.95o   [-1,0,+1]          [0.056,0.444,0.028]  18c
v_C_C1_C1.54C2.52C2.95p   [-1,0,+1]          [0.111,0.389,0.250]  18c
v_C_C1_C1.54C2.52C2.95q   [-1,0,+1]          [0.445,0.056,0.167]  18c
v_C_C1_C1.54C2.52C2.95r   [-1,0,+1]          [0.445,0.056,0.250]  18c
v_C_C1_C1.54C2.52C2.95s   [-1,0,+1]          [0.056,0.445,0.278]  18c
v_C_C1_C1.54C2.52N2.52    [-1,0,+1]          [0.167,0.167,0.472]  18c
v_C_C3v_C1.54C2.52C2.95a  [-1,0,+1]          [0.000,0.000,0.056]  3a
v_C_C3v_C1.54C2.52C2.95b  [-1,0,+1]          [0.000,0.000,0.139]  3a
v_C_C3v_C1.54C2.52C2.95c  [-1,0,+1]          [0.000,0.000,0.722]  3a
v_C_C3v_C1.54C2.52C2.95d  [-1,0,+1]          [0.000,0.000,0.806]  3a
v_C_C3v_C1.54N1.54        [-1,0,+1]          [0.000,0.000,0.389]  3a
v_C_Cs_C1.54C2.52C2.95a   [-1,0,+1]          [0.556,0.278,0.278]  9b
v_C_Cs_C1.54C2.52C2.95b   [-1,0,+1]          [0.500,0.500,0.056]  9b
v_C_Cs_C1.54C2.52C2.95c   [-1,0,+1]          [0.500,0.500,0.139]  9b
v_C_Cs_C1.54C2.52C2.95d   [-1,0,+1]          [0.500,0.500,0.389]  9b
v_C_Cs_C1.54C2.52C2.95e   [-1,0,+1]          [0.500,0.500,0.472]  9b
v_C_Cs_C1.54C2.52C2.95f   [-1,0,+1]          [0.500,0.500,0.722]  9b
v_C_Cs_C1.54C2.52C2.95g   [-1,0,+1]          [0.500,0.500,0.806]  9b
v_C_Cs_C1.54C2.52C2.95h   [-1,0,+1]          [0.056,0.111,0.028]  9b
v_C_Cs_C1.54C2.52C2.95i   [-1,0,+1]          [0.111,0.056,0.167]  9b
v_C_Cs_C1.54C2.52C2.95j   [-1,0,+1]          [0.222,0.111,0.028]  9b
v_C_Cs_C1.54C2.52C2.95k   [-1,0,+1]          [0.111,0.056,0.250]  9b
v_C_Cs_C1.54C2.52C2.95l   [-1,0,+1]          [0.111,0.222,0.167]  9b
v_C_Cs_C1.54C2.52C2.95m   [-1,0,+1]          [0.056,0.111,0.278]  9b
v_C_Cs_C1.54C2.52C2.95n   [-1,0,+1]          [0.111,0.222,0.250]  9b
v_C_Cs_C1.54C2.52C2.95o   [-1,0,+1]          [0.222,0.111,0.278]  9b
v_C_Cs_C1.54C2.52C2.95p   [-1,0,+1]          [0.222,0.111,0.361]  9b
v_C_Cs_C1.54C2.52C2.95q   [-1,0,+1]          [0.444,0.222,0.167]  9b
v_C_Cs_C1.54C2.52C2.95r   [-1,0,+1]          [0.444,0.222,0.250]  9b
v_C_Cs_C1.54C2.52C2.95s   [-1,0,+1]          [0.111,0.222,0.500]  9b
v_C_Cs_C1.54C2.52C2.95t   [-1,0,+1]          [0.556,0.111,0.028]  9b
v_C_Cs_C1.54C2.52C2.95u   [-1,0,+1]          [0.222,0.445,0.278]  9b
v_C_Cs_C1.54C2.52C2.95v   [-1,0,+1]          [0.111,0.555,0.250]  9b
v_C_Cs_C1.54C2.52C2.95w   [-1,0,+1]          [0.556,0.278,0.028]  9b
v_C_Cs_C1.54C2.52C2.95x   [-1,0,+1]          [0.056,0.111,0.611]  9b
v_C_Cs_C1.54C2.52C2.95y   [-1,0,+1]          [0.556,0.111,0.278]  9b
v_C_Cs_C1.54C2.52C2.95z   [-1,0,+1]          [0.611,0.222,0.167]  9b
v_C_Cs_C1.54C2.52C2.95{   [-1,0,+1]          [0.611,0.222,0.250]  9b
v_C_Cs_C1.54C2.52N2.52a   [-1,0,+1]          [0.056,0.111,0.361]  9b
v_C_Cs_C1.54C2.52N2.52b   [-1,0,+1]          [0.111,0.056,0.583]  9b
v_C_Cs_C1.54N1.54         [-1,0,+1]          [0.111,0.056,0.500]  9b
v_N                       [-1,0,+1]          [0.000,0.000,0.472]  3a

Substitutions             Guessed Charges    Conv. Cell Coords    Wyckoff
------------------------  -----------------  -------------------  ---------
C_N                       [-1,0,+1]          [0.000,0.000,0.472]  3a
N_C_C1_C1.54C2.52C2.95a   [-1,0,+1]          [0.167,0.167,0.056]  18c
N_C_C1_C1.54C2.52C2.95b   [-1,0,+1]          [0.167,0.167,0.139]  18c
N_C_C1_C1.54C2.52C2.95c   [-1,0,+1]          [0.056,0.278,0.278]  18c
N_C_C1_C1.54C2.52C2.95d   [-1,0,+1]          [0.167,0.167,0.389]  18c
N_C_C1_C1.54C2.52C2.95e   [-1,0,+1]          [0.333,0.333,0.056]  18c
N_C_C1_C1.54C2.52C2.95f   [-1,0,+1]          [0.333,0.333,0.139]  18c
N_C_C1_C1.54C2.52C2.95g   [-1,0,+1]          [0.389,0.278,0.278]  18c
N_C_C1_C1.54C2.52C2.95h   [-1,0,+1]          [0.167,0.167,0.722]  18c
N_C_C1_C1.54C2.52C2.95i   [-1,0,+1]          [0.167,0.167,0.806]  18c
N_C_C1_C1.54C2.52C2.95j   [-1,0,+1]          [0.056,0.278,0.028]  18c
N_C_C1_C1.54C2.52C2.95k   [-1,0,+1]          [0.278,0.055,0.167]  18c
N_C_C1_C1.54C2.52C2.95l   [-1,0,+1]          [0.278,0.056,0.250]  18c
N_C_C1_C1.54C2.52C2.95m   [-1,0,+1]          [0.389,0.111,0.028]  18c
N_C_C1_C1.54C2.52C2.95n   [-1,0,+1]          [0.111,0.389,0.167]  18c
N_C_C1_C1.54C2.52C2.95o   [-1,0,+1]          [0.056,0.444,0.028]  18c
N_C_C1_C1.54C2.52C2.95p   [-1,0,+1]          [0.111,0.389,0.250]  18c
N_C_C1_C1.54C2.52C2.95q   [-1,0,+1]          [0.445,0.056,0.167]  18c
N_C_C1_C1.54C2.52C2.95r   [-1,0,+1]          [0.445,0.056,0.250]  18c
N_C_C1_C1.54C2.52C2.95s   [-1,0,+1]          [0.056,0.445,0.278]  18c
N_C_C1_C1.54C2.52N2.52    [-1,0,+1]          [0.167,0.167,0.472]  18c
N_C_C3v_C1.54C2.52C2.95a  [-1,0,+1]          [0.000,0.000,0.056]  3a
N_C_C3v_C1.54C2.52C2.95b  [-1,0,+1]          [0.000,0.000,0.139]  3a
N_C_C3v_C1.54C2.52C2.95c  [-1,0,+1]          [0.000,0.000,0.722]  3a
N_C_C3v_C1.54C2.52C2.95d  [-1,0,+1]          [0.000,0.000,0.806]  3a
N_C_C3v_C1.54N1.54        [-1,0,+1]          [0.000,0.000,0.389]  3a
N_C_Cs_C1.54C2.52C2.95a   [-1,0,+1]          [0.556,0.278,0.278]  9b
N_C_Cs_C1.54C2.52C2.95b   [-1,0,+1]          [0.500,0.500,0.056]  9b
N_C_Cs_C1.54C2.52C2.95c   [-1,0,+1]          [0.500,0.500,0.139]  9b
N_C_Cs_C1.54C2.52C2.95d   [-1,0,+1]          [0.500,0.500,0.389]  9b
N_C_Cs_C1.54C2.52C2.95e   [-1,0,+1]          [0.500,0.500,0.472]  9b
N_C_Cs_C1.54C2.52C2.95f   [-1,0,+1]          [0.500,0.500,0.722]  9b
N_C_Cs_C1.54C2.52C2.95g   [-1,0,+1]          [0.500,0.500,0.806]  9b
N_C_Cs_C1.54C2.52C2.95h   [-1,0,+1]          [0.056,0.111,0.028]  9b
N_C_Cs_C1.54C2.52C2.95i   [-1,0,+1]          [0.111,0.056,0.167]  9b
N_C_Cs_C1.54C2.52C2.95j   [-1,0,+1]          [0.222,0.111,0.028]  9b
N_C_Cs_C1.54C2.52C2.95k   [-1,0,+1]          [0.111,0.056,0.250]  9b
N_C_Cs_C1.54C2.52C2.95l   [-1,0,+1]          [0.111,0.222,0.167]  9b
N_C_Cs_C1.54C2.52C2.95m   [-1,0,+1]          [0.056,0.111,0.278]  9b
N_C_Cs_C1.54C2.52C2.95n   [-1,0,+1]          [0.111,0.222,0.250]  9b
N_C_Cs_C1.54C2.52C2.95o   [-1,0,+1]          [0.222,0.111,0.278]  9b
N_C_Cs_C1.54C2.52C2.95p   [-1,0,+1]          [0.222,0.111,0.361]  9b
N_C_Cs_C1.54C2.52C2.95q   [-1,0,+1]          [0.444,0.222,0.167]  9b
N_C_Cs_C1.54C2.52C2.95r   [-1,0,+1]          [0.444,0.222,0.250]  9b
N_C_Cs_C1.54C2.52C2.95s   [-1,0,+1]          [0.111,0.222,0.500]  9b
N_C_Cs_C1.54C2.52C2.95t   [-1,0,+1]          [0.556,0.111,0.028]  9b
N_C_Cs_C1.54C2.52C2.95u   [-1,0,+1]          [0.222,0.445,0.278]  9b
N_C_Cs_C1.54C2.52C2.95v   [-1,0,+1]          [0.111,0.555,0.250]  9b
N_C_Cs_C1.54C2.52C2.95w   [-1,0,+1]          [0.556,0.278,0.028]  9b
N_C_Cs_C1.54C2.52C2.95x   [-1,0,+1]          [0.056,0.111,0.611]  9b
N_C_Cs_C1.54C2.52C2.95y   [-1,0,+1]          [0.556,0.111,0.278]  9b
N_C_Cs_C1.54C2.52C2.95z   [-1,0,+1]          [0.611,0.222,0.167]  9b
N_C_Cs_C1.54C2.52C2.95{   [-1,0,+1]          [0.611,0.222,0.250]  9b
N_C_Cs_C1.54C2.52N2.52a   [-1,0,+1]          [0.056,0.111,0.361]  9b
N_C_Cs_C1.54C2.52N2.52b   [-1,0,+1]          [0.111,0.056,0.583]  9b
N_C_Cs_C1.54N1.54         [-1,0,+1]          [0.111,0.056,0.500]  9b
\n"""
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            "conventional ('conv.') unit cell, which comprises 3 formula unit(s) of C215N.\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        self.zn3p2 = Structure.from_file(f"{self.data_dir}/Zn3P2_POSCAR")
        self.sb2se3 = Structure.from_file(f"{self.data_dir}/Sb2Se3_bulk_supercell_POSCAR")
        self.ag2se = Structure.from_file(f"{self.data_dir}/Ag2Se_POSCAR")
        self.sb2si2te6 = Structure.from_file(f"{self.data_dir}/Sb2Si2Te6_POSCAR")

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

    def _general_defect_gen_check(self, defect_gen, charge_states_removed=False):
        assert self.structure_matcher.fit(
            defect_gen.primitive_structure * defect_gen.supercell_matrix,
            defect_gen.bulk_supercell,
        )
        assert len(defect_gen) == len(defect_gen.defect_entries)  # __len__()
        assert dict(defect_gen.items()) == defect_gen.defect_entries  # __iter__()
        assert all(
            defect_entry_name in defect_gen
            for defect_entry_name in defect_gen.defect_entries  # __contains__()
        )

        assert all(defect.defect_type == DefectType.Vacancy for defect in defect_gen.defects["vacancies"])
        if "substitutions" in defect_gen.defects:
            assert all(
                defect.defect_type == DefectType.Substitution
                for defect in defect_gen.defects["substitutions"]
            )
        assert all(
            defect.defect_type == DefectType.Interstitial
            for defect in defect_gen.defects.get("interstitials", [])
        )
        assert sum(vacancy.multiplicity for vacancy in defect_gen.defects["vacancies"]) == len(
            defect_gen.primitive_structure
        )

        assert all(
            isinstance(defect_entry, DefectEntry) for defect_entry in defect_gen.defect_entries.values()
        )

        for defect_list in defect_gen.defects.values():
            for defect in defect_list:
                assert isinstance(defect, Defect)
                assert defect.conv_cell_frac_coords in defect.equiv_conv_cell_frac_coords
                assert defect.structure == defect_gen.primitive_structure
                np.testing.assert_array_equal(  # test that defect structure uses primitive structure
                    defect.defect_structure.lattice.matrix,
                    defect_gen.primitive_structure.lattice.matrix,
                )

        for defect_name, defect_entry in defect_gen.defect_entries.items():
            self._check_defect_entry(defect_entry, defect_name, defect_gen, charge_states_removed)

        random_name, random_defect_entry = random.choice(list(defect_gen.defect_entries.items()))
        self._random_equiv_supercell_sites_check(random_defect_entry)
        self._check_editing_defect_gen(random_name, defect_gen)

    def _check_defect_entry(self, defect_entry, defect_name, defect_gen, charge_states_removed=False):
        assert defect_entry.name == defect_name
        assert defect_entry.charge_state == int(defect_name.split("_")[-1])
        assert defect_entry.wyckoff
        assert defect_entry.defect
        assert defect_entry.defect.wyckoff == defect_entry.wyckoff
        assert np.isclose(
            defect_entry.defect.conv_cell_frac_coords, defect_entry.conv_cell_frac_coords
        ).all()
        np.testing.assert_allclose(
            defect_entry.sc_entry.structure.lattice.matrix,
            defect_gen.bulk_supercell.lattice.matrix,
        )
        sga = SpacegroupAnalyzer(defect_gen.structure)
        reoriented_conv_structure = swap_axes(
            sga.get_conventional_standard_structure(), defect_gen._BilbaoCS_conv_cell_vector_mapping
        )
        assert np.allclose(
            defect_entry.conventional_structure.lattice.matrix,
            reoriented_conv_structure.lattice.matrix,
        )
        assert np.allclose(
            defect_entry.defect.conventional_structure.lattice.matrix,
            reoriented_conv_structure.lattice.matrix,
        )
        # get minimum distance of defect_entry.conv_cell_frac_coords to any site in
        # defect_entry.conventional_structure
        distance_matrix = np.linalg.norm(
            np.dot(
                pbc_diff(
                    np.array([site.frac_coords for site in defect_entry.conventional_structure]),
                    defect_entry.conv_cell_frac_coords,
                ),
                defect_entry.conventional_structure.lattice.matrix,
            ),
            axis=1,
        )
        min_dist = min(distance_matrix[distance_matrix > 0.01])
        if defect_gen.interstitial_gen_kwargs is not False:
            assert min_dist > defect_gen.interstitial_gen_kwargs.get(
                "min_dist", 0.9
            )  # default min_dist = 0.9
        for conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
            distance_matrix = np.linalg.norm(
                np.dot(
                    pbc_diff(
                        np.array([site.frac_coords for site in defect_entry.conventional_structure]),
                        conv_cell_frac_coords,
                    ),
                    defect_entry.conventional_structure.lattice.matrix,
                ),
                axis=1,
            )
            equiv_min_dist = min(distance_matrix[distance_matrix > 0.01])
            assert np.isclose(min_dist, equiv_min_dist, atol=0.01)

        # test equivalent_sites for defects:
        assert len(defect_entry.defect.equivalent_sites) == defect_entry.defect.multiplicity
        assert defect_entry.defect.site in defect_entry.defect.equivalent_sites
        for equiv_site in defect_entry.defect.equivalent_sites:
            nearest_atoms = defect_entry.defect.structure.get_sites_in_sphere(
                equiv_site.coords,
                5,
            )
            nn_distances = np.array([nn.distance_from_point(equiv_site.coords) for nn in nearest_atoms])
            nn_distance = min(nn_distances[nn_distances > 0.01])  # minimum nonzero distance
            print(defect_entry.name, equiv_site.coords, nn_distance, min_dist)
            assert np.isclose(min_dist, nn_distance, atol=0.01)  # same min_dist as from
            # conv_cell_frac_coords testing above

        assert np.allclose(
            defect_entry.bulk_supercell.lattice.matrix, defect_gen.bulk_supercell.lattice.matrix
        )
        num_prim_cells_in_conv_cell = len(defect_entry.conventional_structure) / len(
            defect_entry.defect.structure
        )
        assert defect_entry.defect.multiplicity * num_prim_cells_in_conv_cell == int(
            defect_entry.wyckoff[:-1]
        )
        assert len(defect_entry.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
        assert len(defect_entry.defect.equiv_conv_cell_frac_coords) == int(defect_entry.wyckoff[:-1])
        assert defect_entry.conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords
        for equiv_conv_cell_frac_coords in defect_entry.equiv_conv_cell_frac_coords:
            assert any(
                np.array_equal(equiv_conv_cell_frac_coords, x)
                for x in defect_entry.defect.equiv_conv_cell_frac_coords
            )
        assert len(defect_entry.equivalent_supercell_sites) == int(defect_entry.wyckoff[:-1]) * (
            len(defect_entry.bulk_supercell) / len(defect_entry.conventional_structure)
        )
        assert defect_entry.defect_supercell_site in defect_entry.equivalent_supercell_sites
        assert np.isclose(
            defect_entry.defect_supercell_site.frac_coords, defect_entry.sc_defect_frac_coords
        ).all()
        assert defect_entry.bulk_entry is None

        # Sb2Se3 and Ag2Se are the 2 test cases included so far where the lattice vectors swap,
        # both with the [1, 0, 2] mapping
        assert defect_entry._BilbaoCS_conv_cell_vector_mapping in [[0, 1, 2], [1, 0, 2]]

        assert (
            defect_entry._BilbaoCS_conv_cell_vector_mapping
            == defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
        )
        assert defect_entry.defect_supercell == defect_entry.sc_entry.structure
        assert not defect_entry.corrections
        assert defect_entry.corrected_energy == 0  # check doesn't raise error (with bugfix from SK)

        # test charge state guessing:
        if not charge_states_removed:
            for charge_state_dict in defect_entry.charge_state_guessing_log:
                charge_state = charge_state_dict["input_parameters"]["charge_state"]
                try:
                    assert np.isclose(
                        np.product(list(charge_state_dict["probability_factors"].values())),
                        charge_state_dict["probability"],
                    )
                except AssertionError as e:
                    if charge_state not in [-1, 0, 1]:
                        raise e

                if charge_state_dict["probability"] > charge_state_dict["probability_threshold"]:
                    assert any(
                        defect_name in defect_gen.defect_entries
                        for defect_name in defect_gen.defect_entries
                        if int(defect_name.split("_")[-1]) == charge_state
                        and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                    )
                else:
                    try:
                        assert all(
                            defect_name not in defect_gen.defect_entries
                            for defect_name in defect_gen.defect_entries
                            if int(defect_name.split("_")[-1]) == charge_state
                            and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                        )
                    except AssertionError as e:
                        # check if intermediate charge state:
                        if all(
                            defect_name not in defect_gen.defect_entries
                            for defect_name in defect_gen.defect_entries
                            if abs(int(defect_name.split("_")[-1])) > abs(charge_state)
                            and defect_name.startswith(defect_entry.name.rsplit("_", 1)[0])
                        ):
                            raise e

    def _random_equiv_supercell_sites_check(self, defect_entry):
        print(f"Randomly testing the equivalent supercell sites for {defect_entry.name}...")
        # get minimum distance of defect_entry.defect_supercell_site to any site in
        # defect_entry.bulk_supercell:
        distance_matrix = defect_entry.defect_supercell.distance_matrix
        min_dist = min(distance_matrix[distance_matrix > 0.01])
        print(min_dist)
        for equiv_defect_supercell_site in defect_entry.equivalent_supercell_sites:
            new_defect_structure = defect_entry.bulk_supercell.copy()
            new_defect_structure.append(
                equiv_defect_supercell_site.specie, equiv_defect_supercell_site.frac_coords
            )
            distance_matrix = new_defect_structure.distance_matrix
            equiv_min_dist = min(distance_matrix[distance_matrix > 0.01])
            print(equiv_min_dist)
            assert np.isclose(min_dist, equiv_min_dist, atol=0.01)

    def _check_editing_defect_gen(self, random_defect_entry_name, defect_gen):
        assert (
            defect_gen[random_defect_entry_name] == defect_gen.defect_entries[random_defect_entry_name]
        )  # __getitem__()
        assert (
            defect_gen.get(random_defect_entry_name) == defect_gen.defect_entries[random_defect_entry_name]
        )  # get()
        random_defect_entry = defect_gen.defect_entries[random_defect_entry_name]
        del defect_gen[random_defect_entry_name]  # __delitem__()
        assert random_defect_entry_name not in defect_gen
        defect_gen[random_defect_entry_name] = random_defect_entry  # __setitem__()
        # assert setting something else throws an error
        with self.assertRaises(TypeError) as e:
            defect_gen[random_defect_entry_name] = random_defect_entry.defect
            assert "Value must be a DefectEntry object, not Interstitial" in str(e.exception)

        with self.assertRaises(ValueError) as e:
            fd_up_random_defect_entry = copy.deepcopy(defect_gen.defect_entries[random_defect_entry_name])
            fd_up_random_defect_entry.defect.structure = self.cdte_bulk_supercell  # any structure that
            # isn't used as a primitive structure for any defect gen will do here
            defect_gen[random_defect_entry_name] = fd_up_random_defect_entry
            assert "Value must have the same primitive structure as the DefectsGenerator object, " in str(
                e.exception
            )

        with self.assertRaises(ValueError) as e:
            fd_up_random_defect_entry = copy.deepcopy(defect_gen.defect_entries[random_defect_entry_name])
            fd_up_random_defect_entry.sc_entry = copy.deepcopy(self.fd_up_sc_entry)
            defect_gen[random_defect_entry_name] = fd_up_random_defect_entry
            assert "Value must have the same supercell as the DefectsGenerator object," in str(e.exception)

    def _generate_and_test_no_warnings(self, structure, min_image_distance=None, **kwargs):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                defect_gen = DefectsGenerator(structure, **kwargs)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                if min_image_distance is None:
                    assert not non_ignored_warnings
                else:
                    assert len(non_ignored_warnings) == 1
                    assert issubclass(non_ignored_warnings[-1].category, UserWarning)
                    assert (
                        f"Input structure is <10 Å in at least one direction (minimum image distance ="
                        f" {min_image_distance:.2f} Å, which is usually too small for accurate defect "
                        f"calculations, but generate_supercell = False, so using input structure as "
                        f"defect & bulk supercells. Caution advised!"
                        in str(non_ignored_warnings[-1].message)
                    )
            output = sys.stdout.getvalue()  # Return a str containing the printed output
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        return defect_gen, output

    def test_extrinsic(self):
        def _split_and_check_orig_cdte_output(output):
            # split self.cdte_defect_gen_info into lines and check each line is in the output:
            for line in self.cdte_defect_gen_info.splitlines():
                if "Cd" in line or "Te" in line:
                    assert line in output

        def _test_cdte_interstitials_and_Te_sub(output, extrinsic_cdte_defect_gen, element="Se"):
            spacing = " " * (2 - len(element))
            charges = "[-2,-1,0,+1,+2]" if element == "S" else "[-2,-1,0]      "
            for info in [
                output,
                extrinsic_cdte_defect_gen._defect_generator_info(),
                repr(extrinsic_cdte_defect_gen),
            ]:
                assert (
                    f"{element}_i_C3v         {spacing}{charges}        [0.625,0.625,0.625]  16e" in info
                )
                assert f"{element}_i_Td_Cd2.83   {spacing}{charges}        [0.750,0.750,0.750]  4d" in info
                assert f"{element}_i_Td_Te2.83   {spacing}{charges}        [0.500,0.500,0.500]  4b" in info

                assert (
                    f"{element}_Te            {spacing}[0,+1]                 [0.250,0.250,0.250]  4c"
                    in info
                )

        def _check_Se_Te(extrinsic_cdte_defect_gen, element="Se", idx=-1):
            assert extrinsic_cdte_defect_gen.defects["substitutions"][idx].name == f"{element}_Te"
            assert extrinsic_cdte_defect_gen.defects["substitutions"][idx].oxi_state == 0
            assert extrinsic_cdte_defect_gen.defects["substitutions"][idx].multiplicity == 1
            assert extrinsic_cdte_defect_gen.defects["substitutions"][idx].defect_site == PeriodicSite(
                "Te", [0.25, 0.25, 0.25], extrinsic_cdte_defect_gen.primitive_structure.lattice
            )
            assert str(extrinsic_cdte_defect_gen.defects["substitutions"][idx].site.specie) == f"{element}"
            assert np.isclose(
                extrinsic_cdte_defect_gen.defects["substitutions"][idx].site.frac_coords,
                np.array([0.25, 0.25, 0.25]),
            ).all()
            assert (
                len(extrinsic_cdte_defect_gen.defects["substitutions"][idx].equiv_conv_cell_frac_coords)
                == 4
            )  # 4x conv cell

        extrinsic_input = "Se"
        cdte_se_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cdte, extrinsic=extrinsic_input
        )
        _split_and_check_orig_cdte_output(output)
        _test_cdte_interstitials_and_Te_sub(output, cdte_se_defect_gen)
        assert "Se_Cd            [-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.000]  4a" in output

        self._general_defect_gen_check(cdte_se_defect_gen)
        assert cdte_se_defect_gen.extrinsic == extrinsic_input  # explicitly test extrinsic attribute set

        # explicitly test defects
        assert len(cdte_se_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cdte_se_defect_gen.defects["vacancies"]) == 2
        assert len(cdte_se_defect_gen.defects["substitutions"]) == 4  # 2 extra
        assert len(cdte_se_defect_gen.defects["interstitials"]) == 9  # 3 extra

        # explicitly test some relevant defect attributes
        _check_Se_Te(cdte_se_defect_gen)
        # test defect entries
        assert len(cdte_se_defect_gen.defect_entries) == 68  # 18 more

        # explicitly test defect entry charge state log:
        assert cdte_se_defect_gen.defect_entries["Se_Cd_-1"].charge_state_guessing_log == [
            {
                "input_parameters": {
                    "charge_state": -4.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.767,
                    "oxi_state": -2.0,
                },
                "probability": 0.12079493065867099,
                "probability_factors": {
                    "charge_state_magnitude": 0.3968502629920499,
                    "charge_state_vs_max_host_charge": 0.3968502629920499,
                    "oxi_probability": 0.767,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability_threshold": 0.0075,
            },
            {
                "input_parameters": {
                    "charge_state": 2.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.111,
                    "oxi_state": 4.0,
                },
                "probability": 0.027750000000000004,
                "probability_factors": {
                    "charge_state_magnitude": 0.6299605249474366,
                    "charge_state_vs_max_host_charge": 1.0,
                    "oxi_probability": 0.111,
                    "oxi_state_vs_max_host_charge": 0.3968502629920499,
                },
                "probability_threshold": 0.0075,
            },
            {
                "input_parameters": {
                    "charge_state": -3.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.079,
                    "oxi_state": -1,
                },
                "probability": 0.023925421138956505,
                "probability_factors": {
                    "charge_state_magnitude": 0.4807498567691361,
                    "charge_state_vs_max_host_charge": 0.6299605249474366,
                    "oxi_probability": 0.079,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability_threshold": 0.0075,
            },
            {
                "input_parameters": {
                    "charge_state": 4.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.024,
                    "oxi_state": 6.0,
                },
                "probability": 0.000944940787421155,
                "probability_factors": {
                    "charge_state_magnitude": 0.3968502629920499,
                    "charge_state_vs_max_host_charge": 0.3968502629920499,
                    "oxi_probability": 0.024,
                    "oxi_state_vs_max_host_charge": 0.25,
                },
                "probability_threshold": 0.0075,
            },
        ]

        # test extrinsic with a dict:
        extrinsic_input = {"Te": "Se"}
        cdte_se_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cdte, extrinsic=extrinsic_input
        )
        _split_and_check_orig_cdte_output(output)
        _test_cdte_interstitials_and_Te_sub(output, cdte_se_defect_gen)
        assert "Se_Cd" not in output  # no Se_Cd substitution now

        self._general_defect_gen_check(cdte_se_defect_gen)
        assert cdte_se_defect_gen.extrinsic == extrinsic_input  # explicitly test extrinsic attribute set

        # explicitly test defects
        assert len(cdte_se_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cdte_se_defect_gen.defects["vacancies"]) == 2
        assert len(cdte_se_defect_gen.defects["substitutions"]) == 3  # only 1 extra
        assert len(cdte_se_defect_gen.defects["interstitials"]) == 9  # 3 extra

        # explicitly test some relevant defect attributes
        _check_Se_Te(cdte_se_defect_gen)
        # test defect entries
        assert len(cdte_se_defect_gen.defect_entries) == 61  # 11 more

        # explicitly test defect entry charge state log:
        assert cdte_se_defect_gen.defect_entries["Se_Te_0"].charge_state_guessing_log == [
            {
                "input_parameters": {
                    "charge_state": 0.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.767,
                    "oxi_state": -2.0,
                },
                "probability": 1,
                "probability_factors": {
                    "charge_state_magnitude": 1,
                    "charge_state_vs_max_host_charge": 1.0,
                    "oxi_probability": 0.767,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability_threshold": 0.0075,
            },
            {
                "input_parameters": {
                    "charge_state": 1.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.079,
                    "oxi_state": -1,
                },
                "probability": 0.079,
                "probability_factors": {
                    "charge_state_magnitude": 1.0,
                    "charge_state_vs_max_host_charge": 1.0,
                    "oxi_probability": 0.079,
                    "oxi_state_vs_max_host_charge": 1.0,
                },
                "probability_threshold": 0.0075,
            },
            {
                "input_parameters": {
                    "charge_state": 6.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.111,
                    "oxi_state": 4.0,
                },
                "probability": 0.0033352021313358825,
                "probability_factors": {
                    "charge_state_magnitude": 0.3028534321386899,
                    "charge_state_vs_max_host_charge": 0.25,
                    "oxi_probability": 0.111,
                    "oxi_state_vs_max_host_charge": 0.3968502629920499,
                },
                "probability_threshold": 0.0075,
            },
            {
                "input_parameters": {
                    "charge_state": 8.0,
                    "max_host_oxi_magnitude": 2.0,
                    "oxi_probability": 0.024,
                    "oxi_state": 6.0,
                },
                "probability": 0.00028617856063833296,
                "probability_factors": {
                    "charge_state_magnitude": 0.25,
                    "charge_state_vs_max_host_charge": 0.19078570709222198,
                    "oxi_probability": 0.024,
                    "oxi_state_vs_max_host_charge": 0.25,
                },
                "probability_threshold": 0.0075,
            },
        ]

        # test extrinsic with a dict, with a list as value:
        extrinsic_input = {"Te": ["Se", "S"]}
        cdte_se_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cdte, extrinsic=extrinsic_input
        )
        _split_and_check_orig_cdte_output(output)
        _test_cdte_interstitials_and_Te_sub(output, cdte_se_defect_gen)
        _test_cdte_interstitials_and_Te_sub(output, cdte_se_defect_gen, element="S")
        assert "Se_Cd" not in output  # no Se_Cd substitution now
        assert "S_Cd" not in output  # no S_Cd substitution

        self._general_defect_gen_check(cdte_se_defect_gen)
        assert cdte_se_defect_gen.extrinsic == extrinsic_input  # explicitly test extrinsic attribute set

        # explicitly test defects
        assert len(cdte_se_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cdte_se_defect_gen.defects["vacancies"]) == 2
        assert len(cdte_se_defect_gen.defects["substitutions"]) == 4  # now 2 extra
        assert len(cdte_se_defect_gen.defects["interstitials"]) == 12  # 3 extra

        # explicitly test some relevant defect attributes
        _check_Se_Te(cdte_se_defect_gen, element="S", idx=-2)
        _check_Se_Te(cdte_se_defect_gen, element="Se")
        # test defect entries
        assert len(cdte_se_defect_gen.defect_entries) == 78  # 28 more

        # test warning when specifying an intrinsic element as extrinsic:
        for extrinsic_arg in [
            "Cd",
            {"Te": "Cd"},
            {
                "Te": [
                    "Cd",
                ]
            },
            ["Cd"],
        ]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cdte_defect_gen = DefectsGenerator(self.prim_cdte, extrinsic=extrinsic_arg)
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                assert len(non_ignored_warnings) == 1
                assert (
                    "Specified 'extrinsic' elements ['Cd'] are present in the host structure, so do not "
                    "need to be specified as 'extrinsic' in DefectsGenerator(). These will be ignored."
                    in str(non_ignored_warnings[-1].message)
                )
                assert cdte_defect_gen.extrinsic == extrinsic_arg  # explicitly test extrinsic attribute

        self.cdte_defect_gen_check(cdte_defect_gen)

    def test_processes(self):
        # first test setting processes with a small primitive cell (so it makes no difference):
        cdte_defect_gen, output = self._generate_and_test_no_warnings(self.prim_cdte, processes=4)
        assert self.cdte_defect_gen_info in output
        self.cdte_defect_gen_check(cdte_defect_gen)

        # check with larger unit cell, where processes makes a difference:
        ytos_defect_gen, output = self._generate_and_test_no_warnings(
            self.ytos_bulk_supercell, processes=4
        )
        assert self.ytos_defect_gen_info in output
        self.ytos_defect_gen_check(ytos_defect_gen)

    def test_interstitial_coords(self):
        # first test that specifying the default interstitial coords for CdTe gives the same result as
        # default:
        cdte_interstitial_coords = [
            [0.625, 0.625, 0.625],  # C3v
            [0.750, 0.750, 0.750],  # Cd2.83
            [0.500, 0.500, 0.500],  # Te2.83
        ]
        cdte_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cdte, interstitial_coords=cdte_interstitial_coords
        )

        assert self.cdte_defect_gen_info in output
        assert cdte_defect_gen.interstitial_coords == cdte_interstitial_coords  # check attribute set

        expected = [
            (
                np.array([0.625, 0.625, 0.625]),
                4,
                [
                    np.array([0.125, 0.625, 0.625]),
                    np.array([0.625, 0.125, 0.625]),
                    np.array([0.625, 0.625, 0.125]),
                    np.array([0.625, 0.625, 0.625]),
                ],
            ),
            (np.array([0.75, 0.75, 0.75]), 1, [np.array([0.75, 0.75, 0.75])]),
            (np.array([0.5, 0.5, 0.5]), 1, [np.array([0.5, 0.5, 0.5])]),
        ]

        def _test_prim_interstitial_coords(result, expected):  # check attribute set
            assert len(result) == len(expected), "Lengths do not match"

            for (r_coord, r_num, r_list), (e_coord, e_num, e_list) in zip(result, expected):
                assert np.array_equal(r_coord, e_coord), "Coordinates do not match"
                assert r_num == e_num, "Number of coordinates do not match"
                assert all(
                    np.array_equal(r, e) for r, e in zip(r_list, e_list)
                ), "List of arrays do not match"

        _test_prim_interstitial_coords(cdte_defect_gen.prim_interstitial_coords, expected)

        # defect_gen_check changes defect_entries ordering, so save to json first:
        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

        cdte_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cdte, interstitial_coords=[[0.5, 0.5, 0.5]], extrinsic={"Te": ["Se", "S"]}
        )
        assert cdte_defect_gen.interstitial_coords == [[0.5, 0.5, 0.5]]  # check attribute set
        _test_prim_interstitial_coords(
            cdte_defect_gen.prim_interstitial_coords,
            [(np.array([0.5, 0.5, 0.5]), 1, [np.array([0.5, 0.5, 0.5])])],  # check attribute set
        )

        assert self.cdte_defect_gen_info not in output

        assert (
            """Interstitials    Guessed Charges        Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Cd_i_Td          [0,+1,+2]              [0.500,0.500,0.500]  4b
Te_i_Td          [-2,-1,0,+1,+2,+3,+4]  [0.500,0.500,0.500]  4b
S_i_Td           [-2,-1,0,+1,+2]        [0.500,0.500,0.500]  4b
Se_i_Td          [-2,-1,0]              [0.500,0.500,0.500]  4b"""
            in output
        )  # now only Td
        self._general_defect_gen_check(cdte_defect_gen)

        # test with YTOS conventional cell input
        ytos_interstitial_coords = [  # in conventional structure! subset of Voronoi vertices
            [0, 0.5, 0.18377232],  # C2v
            [0, 0, 0.48467759],  # C4v_O2.68
            [0, 0, 0.41783323],  # C4v_Y1.92
            [0, 0.5, 0.25],  # D2d
        ]
        ytos_conv_struc, _swap_array = get_BCS_conventional_structure(self.ytos_bulk_supercell)
        ytos_defect_gen, output = self._generate_and_test_no_warnings(
            ytos_conv_struc, interstitial_coords=ytos_interstitial_coords
        )

        for line in output.splitlines():
            assert line in self.ytos_defect_gen_info.splitlines()

        self._general_defect_gen_check(ytos_defect_gen)
        self._save_defect_gen_jsons(ytos_defect_gen)
        self._load_and_test_defect_gen_jsons(ytos_defect_gen)

        assert ytos_defect_gen.interstitial_coords == ytos_interstitial_coords  # check attribute set
        _test_prim_interstitial_coords(
            ytos_defect_gen.prim_interstitial_coords,
            [
                (
                    np.array([0.1838, 0.6838, 0.5]),
                    4,
                    [
                        np.array([0.6838, 0.1838, 0.5]),
                        np.array([0.3162, 0.8162, 0.5]),
                        np.array([0.1838, 0.6838, 0.5]),
                        np.array([0.8162, 0.3162, 0.5]),
                    ],
                ),
                (
                    np.array([0.4847, 0.4847, 0.0]),
                    2,
                    [np.array([0.4847, 0.4847, 0.0]), np.array([0.5153, 0.5153, 0.0])],
                ),
                (
                    np.array([0.4178, 0.4178, 0.0]),
                    2,
                    [np.array([0.4178, 0.4178, 0.0]), np.array([0.5822, 0.5822, 0.0])],
                ),
                (
                    np.array([0.25, 0.75, 0.5]),
                    2,
                    [np.array([0.75, 0.25, 0.5]), np.array([0.25, 0.75, 0.5])],
                ),
            ],
        )

        # test with CdTe supercell input:
        cdte_supercell_interstitial_coords = [
            [0.6875, 0.4375, 0.4375],  # C3v
            [0.625, 0.375, 0.375],  # Cd2.83
            [0.75, 0.5, 0.5],  # Te2.83
        ]  # note that cdte_bulk_supercell has the slightly different orientation to
        # defect_gen.bulk_supercell

        cdte_defect_gen, output = self._generate_and_test_no_warnings(
            self.cdte_bulk_supercell, interstitial_coords=cdte_supercell_interstitial_coords
        )

        assert self.cdte_defect_gen_info in output
        assert (
            cdte_defect_gen.interstitial_coords == cdte_supercell_interstitial_coords
        )  # check attribute set
        _test_prim_interstitial_coords(
            cdte_defect_gen.prim_interstitial_coords,
            [
                (
                    np.array([0.625, 0.625, 0.625]),
                    4,
                    [
                        np.array([0.625, 0.125, 0.625]),
                        np.array([0.625, 0.625, 0.125]),
                        np.array([0.625, 0.625, 0.625]),
                        np.array([0.125, 0.625, 0.625]),
                    ],
                ),
                (np.array([0.75, 0.75, 0.75]), 1, [np.array([0.75, 0.75, 0.75])]),
                (np.array([0.5, 0.5, 0.5]), 1, [np.array([0.5, 0.5, 0.5])]),
            ],
        )

        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

        # with supercell input, single interstitial coord, within min_dist of host atom:
        te_cd_1_metastable_c2v_antisite_supercell_frac_coords = [0.9999, 0.9999, 0.0313]

        with warnings.catch_warnings(record=True) as w:
            _cdte_defect_gen = DefectsGenerator(
                self.cdte_bulk_supercell,
                interstitial_coords=te_cd_1_metastable_c2v_antisite_supercell_frac_coords,
            )
            non_ignored_warnings = [
                warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
            ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
            assert len(non_ignored_warnings) == 1
            assert (
                "Note that some manually-specified interstitial sites were skipped due to being too "
                "close to host lattice sites (minimum distance = `min_dist` = 0.90 Å). If for some "
                "reason you still want to include these sites, you can adjust `min_dist` (default = 0.9 "
                "Å), or just use the default Voronoi tessellation algorithm for generating interstitials ("
                "by not setting the `interstitial_coords` argument)."
                in str(non_ignored_warnings[-1].message)
            )
            assert _cdte_defect_gen.interstitial_coords == [
                te_cd_1_metastable_c2v_antisite_supercell_frac_coords
            ]

        cdte_defect_gen, output = self._generate_and_test_no_warnings(
            self.cdte_bulk_supercell,
            interstitial_coords=te_cd_1_metastable_c2v_antisite_supercell_frac_coords,
            interstitial_gen_kwargs={"min_dist": 0.01},
        )
        assert "Cd_i_C2v         [0,+1,+2]              [0.000,0.000,0.063]  24f" in output
        assert "Te_i_C2v         [-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.063]  24f" in output

        self._general_defect_gen_check(cdte_defect_gen)
        assert cdte_defect_gen.interstitial_coords == [
            te_cd_1_metastable_c2v_antisite_supercell_frac_coords
        ]
        assert cdte_defect_gen.interstitial_gen_kwargs == {"min_dist": 0.01}  # check attribute set

    def test_target_frac_coords(self):
        defect_gen = DefectsGenerator(self.prim_cdte)
        target_frac_coords1 = [0, 0, 0]
        target_frac_coords2 = [0.15, 0.8, 0.777]
        target_frac1_defect_gen = DefectsGenerator(self.prim_cdte, target_frac_coords=target_frac_coords1)
        target_frac2_defect_gen = DefectsGenerator(self.prim_cdte, target_frac_coords=target_frac_coords2)

        for i in [target_frac1_defect_gen, target_frac2_defect_gen]:
            self._general_defect_gen_check(i)

            for name, defect_entry in defect_gen.items():
                # test equivalent supercell sites the same:
                assert defect_entry.equivalent_supercell_sites == i[name].equivalent_supercell_sites

                assert defect_entry.defect_supercell_site != i[name].defect_supercell_site

                # test that the defect supercell site chosen is indeed the closest site to the target
                # frac coords:
                assert i[name].defect_supercell_site == min(
                    i[name].equivalent_supercell_sites,
                    key=lambda x: x.distance_and_image_from_frac_coords(
                        i[name].defect_supercell_site.frac_coords
                    )[0],
                )

        assert target_frac1_defect_gen.target_frac_coords == target_frac_coords1  # check attribute set
        assert target_frac2_defect_gen.target_frac_coords == target_frac_coords2  # check attribute set

    def test_supercell_gen_kwargs(self):
        # test setting supercell_gen_kwargs
        cdte_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cdte, supercell_gen_kwargs={"min_length": 15}
        )
        assert self.cdte_defect_gen_info in output
        self._general_defect_gen_check(cdte_defect_gen)
        assert cdte_defect_gen.supercell_gen_kwargs == {"min_length": 15}  # check attribute set

        # check now with 216-atom supercell:
        assert len(cdte_defect_gen.bulk_supercell) == 216

        assert np.allclose(
            cdte_defect_gen["Cd_i_C3v_0"].defect_supercell_site.coords, [8.9933965, 8.9933965, 10.6285595]
        )

    def cdte_defect_gen_check(self, cdte_defect_gen):
        self._general_defect_gen_check(cdte_defect_gen)

        # test attributes:
        assert self.cdte_defect_gen_info in cdte_defect_gen._defect_generator_info()
        assert cdte_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        assert self.structure_matcher.fit(cdte_defect_gen.primitive_structure, self.prim_cdte)
        assert np.allclose(
            cdte_defect_gen.primitive_structure.lattice.matrix, self.prim_cdte.lattice.matrix
        )  # same lattice
        np.testing.assert_allclose(
            cdte_defect_gen.supercell_matrix, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
        )

        assert np.allclose(
            (cdte_defect_gen.primitive_structure * cdte_defect_gen.supercell_matrix).lattice.matrix,
            self.cdte_bulk_supercell.lattice.matrix,
        )
        assert self.structure_matcher.fit(cdte_defect_gen.conventional_structure, self.prim_cdte)
        assert np.allclose(
            cdte_defect_gen.conventional_structure.lattice.matrix,
            self.conv_cdte.lattice.matrix,
        )

        # explicitly test defects
        assert len(cdte_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cdte_defect_gen.defects["vacancies"]) == 2
        assert len(cdte_defect_gen.defects["substitutions"]) == 2
        assert len(cdte_defect_gen.defects["interstitials"]) == 6

        # explicitly test some relevant defect attributes
        assert cdte_defect_gen.defects["vacancies"][0].name == "v_Cd"
        assert cdte_defect_gen.defects["vacancies"][0].oxi_state == -2
        assert cdte_defect_gen.defects["vacancies"][0].multiplicity == 1
        assert cdte_defect_gen.defects["vacancies"][0].defect_site == PeriodicSite(
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )
        assert cdte_defect_gen.defects["vacancies"][0].site == PeriodicSite(
            "Cd", [0, 0, 0], cdte_defect_gen.primitive_structure.lattice
        )
        assert (
            len(cdte_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4
        )  # 4x conv cell

        # test defect entries
        assert len(cdte_defect_gen.defect_entries) == 50
        assert str(cdte_defect_gen) == self.cdte_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(cdte_defect_gen)
            == self.cdte_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.cdte_defect_gen_info
        )

        # explicitly test defect entry charge state log:
        assert cdte_defect_gen.defect_entries["v_Cd_-1"].charge_state_guessing_log == [
            {
                "input_parameters": {"charge_state": -2},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.0075,
                "padding": 1,
            },
            {
                "input_parameters": {"charge_state": -1},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.0075,
                "padding": 1,
            },
            {
                "input_parameters": {"charge_state": 0},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.0075,
                "padding": 1,
            },
            {
                "input_parameters": {"charge_state": 1},
                "probability_factors": {"oxi_probability": 1},
                "probability": 1,
                "probability_threshold": 0.0075,
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
                "probability_threshold": 0.0075,
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
                "probability_threshold": 0.0075,
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
                "probability_threshold": 0.0075,
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
                "probability_threshold": 0.0075,
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
                "probability_threshold": 0.0075,
            },
        ]

        # explicitly test defect entry attributes
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.defect_type == DefectType.Interstitial
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].wyckoff == "16e"
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].conv_cell_frac_coords,
            np.array([0.625, 0.625, 0.625]),
        )
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].sc_defect_frac_coords,
            np.array([0.3125, 0.4375, 0.4375]),
        )
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect_supercell_site.specie.symbol == "Cd"
        assert cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.multiplicity == 4
        np.testing.assert_allclose(
            cdte_defect_gen.defect_entries["Cd_i_C3v_0"].defect.site.frac_coords,
            np.array([0.625, 0.625, 0.625]),
        )

        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.name == "v_Cd"
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.oxi_state == -2
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.multiplicity == 1
        assert cdte_defect_gen.defect_entries["v_Cd_0"].wyckoff == "4a"
        assert cdte_defect_gen.defect_entries["v_Cd_0"].defect.defect_type == DefectType.Vacancy
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
            cdte_defect_gen.defect_entries["v_Cd_0"].sc_defect_frac_coords,
            np.array([0.5, 0.5, 0.5]),
        )

    def test_defects_generator_cdte(self):
        cdte_defect_gen, output = self._generate_and_test_no_warnings(self.prim_cdte)

        assert self.cdte_defect_gen_info in output  # matches expected 4b & 4d Wyckoff letters for Td
        # interstitials (https://doi.org/10.1016/j.solener.2013.12.017)

        assert cdte_defect_gen.structure == self.prim_cdte

        # defect_gen_check changes defect_entries ordering, so save to json first:
        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

        cdte_defect_gen.to_json(f"{self.data_dir}/cdte_defect_gen.json")  # for testing in test_vasp.py

    def test_defects_generator_cdte_supercell_input(self):
        cdte_defect_gen, output = self._generate_and_test_no_warnings(self.cdte_bulk_supercell)

        assert self.cdte_defect_gen_info in output
        assert cdte_defect_gen.structure == self.cdte_bulk_supercell

        self._save_defect_gen_jsons(cdte_defect_gen)
        self.cdte_defect_gen_check(cdte_defect_gen)
        self._load_and_test_defect_gen_jsons(cdte_defect_gen)

    def test_adding_charge_states(self):
        cdte_defect_gen, _output = self._generate_and_test_no_warnings(self.prim_cdte)

        cdte_defect_gen.add_charge_states("Cd_i_C3v_0", [-7, -6])
        self._general_defect_gen_check(cdte_defect_gen)

        assert "Cd_i_C3v_-6" in cdte_defect_gen.defect_entries
        assert cdte_defect_gen["Cd_i_C3v_-7"].charge_state == -7
        info_line = "Cd_i_C3v         [-7,-6,0,+1,+2]        [0.625,0.625,0.625]  16e"
        assert info_line in repr(cdte_defect_gen)

    def test_removing_charge_states(self):
        cdte_defect_gen, _output = self._generate_and_test_no_warnings(self.prim_cdte)
        cdte_defect_gen.remove_charge_states("Cd_i", [+1, +2])
        self._general_defect_gen_check(cdte_defect_gen, charge_states_removed=True)

        assert "Cd_i_C3v_+1" not in cdte_defect_gen.defect_entries
        assert "Cd_i_Td_Cd2.83_+2" not in cdte_defect_gen.defect_entries
        assert "Cd_i_Td_Te2.83_+1" not in cdte_defect_gen.defect_entries
        assert "Cd_i_C3v_0" in cdte_defect_gen.defect_entries
        assert "Cd_i_Td_Cd2.83_0" in cdte_defect_gen.defect_entries
        assert "Cd_i_Td_Te2.83_0" in cdte_defect_gen.defect_entries
        #            Cd_i_C3v         [0,+1,+2]              [0.625,0.625,0.625]  16e
        info_line = "Cd_i_C3v         [0]                    [0.625,0.625,0.625]  16e"
        assert info_line in repr(cdte_defect_gen)

    def test_cdte_no_generate_supercell_supercell_input(self):
        cdte_defect_gen, output = self._generate_and_test_no_warnings(
            self.cdte_bulk_supercell, generate_supercell=False
        )

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
        self._general_defect_gen_check(ytos_defect_gen)

        assert ytos_defect_gen.generate_supercell == generate_supercell

        assert self.ytos_defect_gen_info in ytos_defect_gen._defect_generator_info()
        assert ytos_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]

        # test attributes:
        assert self.structure_matcher.fit(  # reduces to primitive, but StructureMatcher still matches
            ytos_defect_gen.primitive_structure, self.ytos_bulk_supercell
        )
        assert self.structure_matcher.fit(
            ytos_defect_gen.primitive_structure, self.ytos_bulk_supercell.get_primitive_structure()
        )  # reduces to primitive, but StructureMatcher still matches
        assert not np.allclose(
            ytos_defect_gen.primitive_structure.lattice.matrix, self.ytos_bulk_supercell.lattice.matrix
        )
        assert np.allclose(
            ytos_defect_gen.primitive_structure.volume,
            self.ytos_bulk_supercell.get_primitive_structure().volume,
        )
        assert self.structure_matcher.fit(
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

        assert self.structure_matcher.fit(ytos_defect_gen.conventional_structure, self.ytos_bulk_supercell)

        # explicitly test defects
        assert len(ytos_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(ytos_defect_gen.defects["vacancies"]) == 5
        assert len(ytos_defect_gen.defects["substitutions"]) == 15
        assert len(ytos_defect_gen.defects["interstitials"]) == 24

        # test some relevant defect attributes
        assert ytos_defect_gen.defects["vacancies"][0].name == "v_Y"
        assert ytos_defect_gen.defects["vacancies"][0].oxi_state == -3
        assert ytos_defect_gen.defects["vacancies"][0].multiplicity == 2
        assert np.allclose(
            ytos_defect_gen.defects["vacancies"][0].site.frac_coords,
            np.array([0.6661, 0.6661, 0.0]),
            atol=1e-3,
        )
        assert (
            len(ytos_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4
        )  # 2x conv cell

        # explicitly test defect entries
        assert len(ytos_defect_gen.defect_entries) == 221
        assert str(ytos_defect_gen) == self.ytos_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(ytos_defect_gen)
            == self.ytos_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.ytos_defect_gen_info
        )

        # explicitly test defect entry attributes
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.defect_type == DefectType.Interstitial
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].wyckoff == "4d"
        assert ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.multiplicity == 2
        sc_frac_coords = np.array([0.41667, 0.41667, 0.5] if generate_supercell else [0.3333, 0.5, 0.25])
        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["O_i_D2d_0"].sc_defect_frac_coords,
            sc_frac_coords,
            rtol=1e-3,
        )
        assert ytos_defect_gen.defect_entries["O_i_D2d_0"].defect_supercell_site.specie.symbol == "O"

        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["O_i_D2d_-1"].conv_cell_frac_coords,
            np.array([0.000, 0.500, 0.25]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["O_i_D2d_-1"].defect.site.frac_coords,
            np.array([0.25, 0.75, 0.5]),
            atol=1e-3,
        )

        assert ytos_defect_gen.defect_entries["v_Y_0"].defect.name == "v_Y"
        assert ytos_defect_gen.defect_entries["v_Y_0"].defect.oxi_state == -3
        assert ytos_defect_gen.defect_entries["v_Y_-2"].defect.multiplicity == 2
        assert ytos_defect_gen.defect_entries["v_Y_-2"].wyckoff == "4e"
        assert ytos_defect_gen.defect_entries["v_Y_-2"].defect.defect_type == DefectType.Vacancy

        np.testing.assert_allclose(
            ytos_defect_gen.defect_entries["v_Y_0"].conv_cell_frac_coords,
            np.array([0, 0, 0.334]),
            atol=1e-3,
        )

        frac_coords = np.array(
            [0.4446, 0.5554, 0.3322] if generate_supercell else [0.3333, 0.3333, 0.3339]
        )
        try:
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["v_Y_0"].sc_defect_frac_coords,
                frac_coords,
                atol=1e-4,
            )
        except AssertionError:
            np.testing.assert_allclose(
                ytos_defect_gen.defect_entries["v_Y_0"].sc_defect_frac_coords,
                np.array([frac_coords[1], frac_coords[0], frac_coords[2]]),  # swap x and y coordinates
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
        ytos_defect_gen, output = self._generate_and_test_no_warnings(self.ytos_bulk_supercell)

        assert self.ytos_defect_gen_info in output

        self._save_defect_gen_jsons(ytos_defect_gen)
        self.ytos_defect_gen_check(ytos_defect_gen)
        self._load_and_test_defect_gen_jsons(ytos_defect_gen)

        ytos_defect_gen.to_json(f"{self.data_dir}/ytos_defect_gen.json")  # for testing in test_vasp.py

    def test_ytos_no_generate_supercell(self):
        # tests the case of an input structure which is >10 Å in each direction, has
        # more atoms (198) than the pmg supercell (99), but generate_supercell = False,
        # so the _input_ supercell is used
        ytos_defect_gen, output = self._generate_and_test_no_warnings(
            self.ytos_bulk_supercell, generate_supercell=False
        )

        assert self.ytos_defect_gen_info in output

        self._save_defect_gen_jsons(ytos_defect_gen)
        self.ytos_defect_gen_check(ytos_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(ytos_defect_gen)

        ytos_defect_gen.to_json(
            f"{self.data_dir}/ytos_defect_gen_supercell.json"
        )  # for testing in test_vasp.py

    def lmno_defect_gen_check(self, lmno_defect_gen, generate_supercell=True):
        self._general_defect_gen_check(lmno_defect_gen)
        assert lmno_defect_gen.generate_supercell == generate_supercell
        assert self.lmno_defect_gen_info in lmno_defect_gen._defect_generator_info()
        assert lmno_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        assert self.structure_matcher.fit(  # reduces to primitive, but StructureMatcher still matches
            lmno_defect_gen.primitive_structure, self.lmno_primitive
        )
        assert np.allclose(
            lmno_defect_gen.primitive_structure.lattice.matrix, self.lmno_primitive.lattice.matrix
        )
        supercell_matrix = np.array(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]] if generate_supercell else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        np.testing.assert_allclose(lmno_defect_gen.supercell_matrix, supercell_matrix)

        assert self.structure_matcher.fit(lmno_defect_gen.conventional_structure, self.lmno_primitive)

        # explicitly test defects
        assert len(lmno_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(lmno_defect_gen.defects["vacancies"]) == 5
        assert len(lmno_defect_gen.defects["substitutions"]) == 15
        assert len(lmno_defect_gen.defects["interstitials"]) == 24

        # explicitly test some relevant defect attributes
        assert lmno_defect_gen.defects["vacancies"][0].name == "v_Li"
        assert lmno_defect_gen.defects["vacancies"][0].oxi_state == -1
        assert lmno_defect_gen.defects["vacancies"][0].multiplicity == 8  # prim = conv structure in LMNO
        assert np.allclose(
            lmno_defect_gen.defects["vacancies"][0].site.frac_coords,
            np.array([0.0037, 0.0037, 0.0037]),
            atol=1e-4,
        )
        assert (
            len(lmno_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 8
        )  # prim = conv cell in LMNO

        # explicitly test defect entries
        assert len(lmno_defect_gen.defect_entries) == 182
        assert str(lmno_defect_gen) == self.lmno_defect_gen_string  # __str__()
        # __repr__() tested in other tests, skipped here due to slight difference in rounding behaviour
        # between local and GH Actions

        # explicitly test defect entry attributes
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].defect.defect_type
            == DefectType.Interstitial
        )
        assert lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].wyckoff == "24e"
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].defect.multiplicity == 24
        )  # prim = conv structure in LMNO
        sc_frac_coords = np.array(
            [0.494716, 0.616729, 0.500199] if generate_supercell else [0.500397, 0.510568, 0.766541]
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].sc_defect_frac_coords,
            sc_frac_coords,  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        assert (
            lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].defect_supercell_site.specie.symbol == "Ni"
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].conv_cell_frac_coords,
            np.array([0.017, 0.261, 0.250]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Ni_i_C1_O1.78_+2"].defect.site.frac_coords,
            np.array([0.017, 0.261, 0.250]),
            atol=1e-3,
        )

        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.name == "Li_O"
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.oxi_state == +3
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.multiplicity == 8
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].wyckoff == "8c"
        assert lmno_defect_gen.defect_entries["Li_O_C3_+3"].defect.defect_type == DefectType.Substitution

        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].conv_cell_frac_coords,
            np.array([0.385, 0.385, 0.385]),
            atol=1e-3,
        )
        frac_coords = np.array(
            [0.4328, 0.4328, 0.4328] if generate_supercell else [0.38447, 0.38447, 0.38447]
        )
        np.testing.assert_allclose(
            lmno_defect_gen.defect_entries["Li_O_C3_+3"].sc_defect_frac_coords,
            frac_coords,  # closest to middle of supercell
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
        lmno_defect_gen, output = self._generate_and_test_no_warnings(self.lmno_primitive)

        assert self.lmno_defect_gen_info in output

        self._save_defect_gen_jsons(lmno_defect_gen)
        self.lmno_defect_gen_check(lmno_defect_gen)
        self._load_and_test_defect_gen_jsons(lmno_defect_gen)

        lmno_defect_gen.to_json(f"{self.data_dir}/lmno_defect_gen.json")  # for testing in test_vasp.py

    def test_lmno_no_generate_supercell(self):
        # test inputting a non-diagonal supercell structure with a lattice vector <10 Å with
        # generate_supercell = False
        lmno_defect_gen, output = self._generate_and_test_no_warnings(
            self.lmno_primitive, min_image_distance=8.28, generate_supercell=False
        )

        assert self.lmno_defect_gen_info in output

        self._save_defect_gen_jsons(lmno_defect_gen)
        self.lmno_defect_gen_check(lmno_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(lmno_defect_gen)

    def zns_defect_gen_check(self, zns_defect_gen, generate_supercell=True, check_info=True):
        self._general_defect_gen_check(zns_defect_gen)
        assert zns_defect_gen.generate_supercell == generate_supercell
        if check_info:
            assert self.zns_defect_gen_info in zns_defect_gen._defect_generator_info()
        assert zns_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:

        assert self.structure_matcher.fit(zns_defect_gen.primitive_structure, self.non_diagonal_ZnS)
        assert self.structure_matcher.fit(
            zns_defect_gen.primitive_structure, zns_defect_gen.bulk_supercell
        )  # reduces to primitive, but StructureMatcher still matches (but below lattice doesn't match)
        assert not np.allclose(
            zns_defect_gen.primitive_structure.lattice.matrix, self.non_diagonal_ZnS.lattice.matrix
        )

        supercell_matrix = np.array(
            [[-2, 2, 2], [2, -2, 2], [2, 2, -2]]
            if generate_supercell
            else [[0, 0, -2], [0, -4, 2], [-4, 1, 2]]
        )
        np.testing.assert_allclose(zns_defect_gen.supercell_matrix, supercell_matrix)
        assert self.structure_matcher.fit(zns_defect_gen.conventional_structure, self.non_diagonal_ZnS)

        # explicitly test defects
        assert len(zns_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(zns_defect_gen.defects["vacancies"]) == 2
        assert len(zns_defect_gen.defects["substitutions"]) == 2
        assert len(zns_defect_gen.defects["interstitials"]) == 6

        # explicitly test some relevant defect attributes
        assert zns_defect_gen.defects["vacancies"][1].name == "v_S"
        assert zns_defect_gen.defects["vacancies"][1].oxi_state == +2
        assert zns_defect_gen.defects["vacancies"][1].multiplicity == 1
        assert np.allclose(
            zns_defect_gen.defects["vacancies"][1].site.frac_coords, np.array([0.25, 0.25, 0.25])
        )
        assert len(zns_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4  # 4x conv cell

        # explicitly test defect entries
        if check_info:
            assert len(zns_defect_gen.defect_entries) == 44
            assert str(zns_defect_gen) == self.zns_defect_gen_string  # __str__()
            assert (  # __repr__()
                repr(zns_defect_gen)
                == self.zns_defect_gen_string
                + "\n---------------------------------------------------------\n"
                + self.zns_defect_gen_info
            )

        # explicitly test defect entry attributes
        assert (
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.defect_type == DefectType.Interstitial
        )
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].wyckoff == "4b"
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.multiplicity == 1
        sc_frac_coords = np.array([0.25, 0.5, 0.5] if generate_supercell else [0.59375, 0.46875, 0.375])
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].sc_defect_frac_coords,
            sc_frac_coords,  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        assert zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect_supercell_site.specie.symbol == "S"
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].conv_cell_frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["S_i_Td_S2.35_-2"].defect.site.frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )

        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.name == "Zn_S"
        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.oxi_state == +4
        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.multiplicity == 1
        assert zns_defect_gen.defect_entries["Zn_S_+2"].wyckoff == "4c"
        assert zns_defect_gen.defect_entries["Zn_S_+2"].defect.defect_type == DefectType.Substitution

        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["Zn_S_+2"].conv_cell_frac_coords,
            np.array([0.25, 0.25, 0.25]),
            atol=1e-3,
        )
        sc_frac_coords = np.array(
            [0.375, 0.375, 0.625] if generate_supercell else [0.359375, 0.546875, 0.4375]
        )
        np.testing.assert_allclose(
            zns_defect_gen.defect_entries["Zn_S_+2"].sc_defect_frac_coords,
            sc_frac_coords,  # closest to middle of supercell
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
        zns_defect_gen, output = self._generate_and_test_no_warnings(
            self.non_diagonal_ZnS,
        )

        assert self.zns_defect_gen_info in output

        self._save_defect_gen_jsons(zns_defect_gen)
        self.zns_defect_gen_check(zns_defect_gen)
        self._load_and_test_defect_gen_jsons(zns_defect_gen)

    def test_zns_no_generate_supercell(self):
        # test inputting a non-diagonal supercell structure with a lattice vector <10 Å with
        # generate_supercell = False
        zns_defect_gen, output = self._generate_and_test_no_warnings(
            self.non_diagonal_ZnS, min_image_distance=7.59, generate_supercell=False
        )

        assert self.zns_defect_gen_info in output

        self._save_defect_gen_jsons(zns_defect_gen)
        self.zns_defect_gen_check(zns_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(zns_defect_gen)

    def cu_defect_gen_check(self, cu_defect_gen):
        self._general_defect_gen_check(cu_defect_gen)
        assert self.cu_defect_gen_info in cu_defect_gen._defect_generator_info()
        assert cu_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        assert self.structure_matcher.fit(cu_defect_gen.primitive_structure, self.prim_cu)

        np.testing.assert_allclose(
            cu_defect_gen.supercell_matrix, np.array([[-3, 3, 3], [3, -3, 3], [3, 3, -3]])
        )
        assert self.structure_matcher.fit(cu_defect_gen.conventional_structure, self.prim_cu)

        # explicitly test defects
        assert len(cu_defect_gen.defects) == 2  # vacancies, NO substitutions, interstitials
        assert len(cu_defect_gen.defects["vacancies"]) == 1
        assert cu_defect_gen.defects.get("substitutions") is None
        assert len(cu_defect_gen.defects["interstitials"]) == 2

        # explicitly test some relevant defect attributes
        assert cu_defect_gen.defects["vacancies"][0].name == "v_Cu"
        assert cu_defect_gen.defects["vacancies"][0].oxi_state == 0
        assert cu_defect_gen.defects["vacancies"][0].multiplicity == 1
        assert np.allclose(
            cu_defect_gen.defects["vacancies"][0].site.frac_coords, np.array([0.0, 0.0, 0.0])
        )
        assert len(cu_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 4  # 4x conv cell

        # explicitly test defect entries
        assert len(cu_defect_gen.defect_entries) == 9
        assert str(cu_defect_gen) == self.cu_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(cu_defect_gen)
            == self.cu_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.cu_defect_gen_info
        )

        # explicitly test defect entry attributes
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.defect_type == DefectType.Interstitial
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].wyckoff == "4b"
        assert cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.multiplicity == 1
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].sc_defect_frac_coords,
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
            cu_defect_gen.defect_entries["Cu_i_Oh_+1"].defect.site.frac_coords,
            np.array([0.5, 0.5, 0.5]),
            rtol=1e-2,
        )

        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.name == "v_Cu"
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.oxi_state == 0
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.multiplicity == 1
        assert cu_defect_gen.defect_entries["v_Cu_0"].wyckoff == "4a"
        assert cu_defect_gen.defect_entries["v_Cu_0"].defect.defect_type == DefectType.Vacancy

        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["v_Cu_0"].conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cu_defect_gen.defect_entries["v_Cu_0"].sc_defect_frac_coords,
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
        cu_defect_gen, output = self._generate_and_test_no_warnings(
            self.prim_cu,
        )

        assert self.cu_defect_gen_info in output
        assert cu_defect_gen.structure == self.prim_cu

        self._save_defect_gen_jsons(cu_defect_gen)
        self.cu_defect_gen_check(cu_defect_gen)
        self._load_and_test_defect_gen_jsons(cu_defect_gen)

        cu_defect_gen.to_json(f"{self.data_dir}/cu_defect_gen.json")  # for testing in test_vasp.py

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
        self._general_defect_gen_check(agcu_defect_gen)
        assert agcu_defect_gen.generate_supercell == generate_supercell
        assert self.agcu_defect_gen_info in agcu_defect_gen._defect_generator_info()
        assert agcu_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        assert self.structure_matcher.fit(agcu_defect_gen.primitive_structure, self.agcu)

        supercell_matrix = np.array(
            [[2, 2, 0], [-5, 5, 0], [-3, -3, 6]]
            if generate_supercell
            else [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
        )
        np.testing.assert_allclose(agcu_defect_gen.supercell_matrix, supercell_matrix)
        assert self.structure_matcher.fit(agcu_defect_gen.conventional_structure, self.agcu)

        # explicitly test defects
        assert len(agcu_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(agcu_defect_gen.defects["vacancies"]) == 2
        assert len(agcu_defect_gen.defects["substitutions"]) == 2
        assert len(agcu_defect_gen.defects["interstitials"]) == 6

        # explicitly test some relevant defect attributes
        assert agcu_defect_gen.defects["vacancies"][1].name == "v_Ag"
        assert agcu_defect_gen.defects["vacancies"][1].oxi_state == 0
        assert agcu_defect_gen.defects["vacancies"][1].multiplicity == 1
        assert agcu_defect_gen.defects["vacancies"][1].defect_type == DefectType.Vacancy
        assert np.allclose(
            agcu_defect_gen.defects["vacancies"][1].site.frac_coords, np.array([0.5, 0.5, 0.5])
        )
        assert (
            len(agcu_defect_gen.defects["vacancies"][0].equiv_conv_cell_frac_coords) == 3
        )  # 3x conv cell

        # explicitly test defect entries
        assert len(agcu_defect_gen.defect_entries) == 28
        assert str(agcu_defect_gen) == self.agcu_defect_gen_string  # __str__()
        assert (  # __repr__()
            repr(agcu_defect_gen)
            == self.agcu_defect_gen_string
            + "\n---------------------------------------------------------\n"
            + self.agcu_defect_gen_info
        )

        # explicitly test defect entry attributes
        assert (
            agcu_defect_gen.defect_entries["Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"].defect.defect_type
            == DefectType.Interstitial
        )
        assert agcu_defect_gen.defect_entries["Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"].wyckoff == "6c"
        assert agcu_defect_gen.defect_entries["Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"].defect.multiplicity == 2
        sc_frac_coords = np.array(
            [0.53125, 0.5, 0.395833] if generate_supercell else [0.375, 0.375, 0.375]
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"].sc_defect_frac_coords,
            sc_frac_coords,  # closest to [0.5, 0.5, 0.5]
            rtol=1e-2,
        )
        assert (
            agcu_defect_gen.defect_entries[
                "Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"
            ].defect_supercell_site.specie.symbol
            == "Cu"
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"].conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.375]),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Cu_i_C3v_Cu1.56Ag1.56Cu2.99b_+1"].defect.site.frac_coords,
            np.array([0.375, 0.375, 0.375]),
            rtol=1e-2,
        )

        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.name == "Ag_Cu"
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.oxi_state == 0
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.multiplicity == 1
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].wyckoff == "3a"
        assert agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.defect_type == DefectType.Substitution

        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].conv_cell_frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )
        sc_frac_coords = np.array([0.5, 0.5, 0.5] if generate_supercell else [0.0, 0.5, 0.5])
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].sc_defect_frac_coords,
            sc_frac_coords,  # closest to middle of supercell
            atol=1e-4,
        )
        np.testing.assert_allclose(
            agcu_defect_gen.defect_entries["Ag_Cu_-1"].defect.site.frac_coords,
            np.array([0.0, 0.0, 0.0]),
            atol=1e-3,
        )

    def test_agcu(self):
        # test initialising with an intermetallic (where pymatgen oxidation state guessing fails)
        agcu_defect_gen, output = self._generate_and_test_no_warnings(self.agcu)

        assert self.agcu_defect_gen_info in output

        self._save_defect_gen_jsons(agcu_defect_gen)
        self.agcu_defect_gen_check(agcu_defect_gen)
        self._load_and_test_defect_gen_jsons(agcu_defect_gen)

        agcu_defect_gen.to_json(f"{self.data_dir}/agcu_defect_gen.json")  # for testing in test_vasp.py

    def test_agcu_no_generate_supercell(self):
        # test high-symmetry intermetallic with generate_supercell = False
        agcu_defect_gen, output = self._generate_and_test_no_warnings(
            self.agcu, min_image_distance=4.42, generate_supercell=False
        )

        assert self.agcu_defect_gen_info in output

        self._save_defect_gen_jsons(agcu_defect_gen)
        self.agcu_defect_gen_check(agcu_defect_gen, generate_supercell=False)
        self._load_and_test_defect_gen_jsons(agcu_defect_gen)

    def cd_i_cdte_supercell_defect_gen_check(self, cd_i_defect_gen):
        self._general_defect_gen_check(cd_i_defect_gen)
        assert self.cd_i_cdte_supercell_defect_gen_info in cd_i_defect_gen._defect_generator_info()
        assert cd_i_defect_gen._BilbaoCS_conv_cell_vector_mapping == [0, 1, 2]
        # test attributes:
        assert not self.structure_matcher.fit(cd_i_defect_gen.primitive_structure, self.prim_cdte)
        assert not self.structure_matcher.fit(
            cd_i_defect_gen.primitive_structure, self.cdte_bulk_supercell
        )
        assert np.allclose(  # primitive cell of defect supercell here is same as bulk supercell
            cd_i_defect_gen.primitive_structure.lattice.matrix, self.cdte_bulk_supercell.lattice.matrix
        )

        np.testing.assert_allclose(cd_i_defect_gen.supercell_matrix, np.eye(3), atol=1e-3)

        # explicitly test defects
        assert len(cd_i_defect_gen.defects) == 3  # vacancies, substitutions, interstitials
        assert len(cd_i_defect_gen.defects["vacancies"]) == 21
        assert len(cd_i_defect_gen.defects["substitutions"]) == 21
        assert len(cd_i_defect_gen.defects["interstitials"]) == 94

        # explicitly test some relevant defect attributes
        assert cd_i_defect_gen.defects["vacancies"][1].name == "v_Cd"
        assert cd_i_defect_gen.defects["vacancies"][1].oxi_state == 0  # pmg fails oxi guessing with
        # defective supercell
        assert cd_i_defect_gen.defects["vacancies"][1].multiplicity == 3
        assert np.allclose(
            cd_i_defect_gen.defects["vacancies"][1].site.frac_coords, np.array([0.0, 0.75, 0.25])
        )
        assert (
            len(cd_i_defect_gen.defects["vacancies"][1].equiv_conv_cell_frac_coords) == 9
        )  # 3x conv cell

        # explicitly test defect entries
        assert len(cd_i_defect_gen.defect_entries) == 650

        # explicitly test defect entry attributes
        assert (
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.defect_type
            == DefectType.Interstitial
        )
        assert cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].wyckoff == "9b"
        assert cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.multiplicity == 3
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].sc_defect_frac_coords,
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
            cd_i_defect_gen.defect_entries["Te_i_Cs_Cd2.83Te3.27Cd5.42e_0"].defect.site.frac_coords,
            np.array([0.875, 0.625, 0.625]),
            rtol=1e-2,
        )

        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.name == "Cd_Te"
        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.oxi_state == 0
        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.multiplicity == 3
        assert cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].wyckoff == "9b"
        assert (
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].defect.defect_type
            == DefectType.Substitution
        )

        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].conv_cell_frac_coords,
            np.array([0.58333, 0.16666, 0.04167]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cd_i_defect_gen.defect_entries["Cd_Te_Cs_Cd2.71_-1"].sc_defect_frac_coords,
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

        cd_i_defect_gen, output = self._generate_and_test_no_warnings(
            cdte_defect_gen["Cd_i_C3v_0"].sc_entry.structure,
        )

        assert self.cd_i_cdte_supercell_defect_gen_info in output

        self._save_defect_gen_jsons(cd_i_defect_gen)
        self.cd_i_cdte_supercell_defect_gen_check(cd_i_defect_gen)
        self._load_and_test_defect_gen_jsons(cd_i_defect_gen)

        cd_i_defect_gen.to_json(
            f"{self.data_dir}/cd_i_supercell_defect_gen.json"
        )  # for testing in test_vasp.py

        # don't need to test generate_supercell = False with this one. Already takes long enough as is,
        # and we've tested the handling of input >10 Å supercells in CdTe tests above

    def test_supercell_w_substitution_N_doped_diamond(self):
        # test inputting a large (216-atom) N_C diamond supercell as input, to check oxi_state handling
        # and skipping of interstitial generation:
        if not _potcars_available():
            return

        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                N_diamond_defect_gen = DefectsGenerator(
                    self.N_doped_diamond_supercell, interstitial_gen_kwargs=False
                )
                non_ignored_warnings = [
                    warning for warning in w if "get_magnetic_symmetry" not in str(warning.message)
                ]  # pymatgen/spglib warning, ignored by default in doped but not here from setting
                assert len(non_ignored_warnings) == 1
                assert (
                    "\nOxidation states could not be guessed for the input structure. This is "
                    "required for charge state guessing, so defects will still be generated but all "
                    "charge states will be set to -1, 0, +1. You can manually edit these with the "
                    "add/remove_charge_states methods (see tutorials), or you can set the oxidation "
                    "states of the input structure (e.g. using "
                    "structure.add_oxidation_state_by_element()) and re-initialize DefectsGenerator()."
                    in str(non_ignored_warnings[-1].message)
                )
                assert N_diamond_defect_gen.interstitial_gen_kwargs is False  # check attribute set

                output = sys.stdout.getvalue()  # Return a str containing the printed output

        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value.

        assert self.N_diamond_defect_gen_info in output
        assert "_i_" not in output  # no interstitials generated

        self._general_defect_gen_check(N_diamond_defect_gen)

    def compare_doped_charges(self, tld_stable_charges, defect_gen):
        """
        Compare the charges of the generated defects to the expected charges.
        """
        false_positives = {}
        false_negatives = {}
        hits = {}
        for defect, charge_list in tld_stable_charges.items():
            generated_defect_charges = {
                defect_gen.defect_entries[defect_entry_name].charge_state
                for defect_entry_name in defect_gen.defect_entries
                if defect_entry_name.startswith(defect)
            }

            false_negatives[defect] = set(charge_list) - generated_defect_charges
            false_positives[defect] = generated_defect_charges - set(charge_list)
            hits[defect] = set(charge_list) & generated_defect_charges

        return (false_positives, false_negatives, hits)

    def test_charge_state_guessing(self):
        # Charge state guessing obvs already explicitly tested above,
        # but here we test for two representative tricky cases with
        # amphoteric species:

        zn3p2_tld_stable_charges = {  # from Yihuang and Geoffroy's: https://arxiv.org/abs/2306.13583
            "v_Zn": list(range(-2, 0 + 1)),
            "v_P": list(range(-1, +1 + 1)),
            "Zn_i": list(range(0, +2 + 1)),
            "P_i": list(range(-1, +3 + 1)),
            "Zn_P": list(range(-2, +3 + 1)),  # -2 just below CBM...
            "P_Zn": list(range(-1, +3 + 1)),
        }

        zn3p2_defect_gen, output = self._generate_and_test_no_warnings(self.zn3p2)
        self._general_defect_gen_check(zn3p2_defect_gen)

        false_positives, false_negatives, hits = self.compare_doped_charges(
            zn3p2_tld_stable_charges, zn3p2_defect_gen
        )
        assert sum(len(val) for val in false_positives.values()) == 13
        assert sum(len(val) for val in false_negatives.values()) == 2
        assert sum(len(val) for val in hits.values()) == 23

        sb2se3_tld_stable_charges = {  # from Xinwei's work
            "v_Se": list(range(-2, +2 + 1)),
            "v_Sb": list(range(-3, +1 + 1)),
            "Sb_Se": list(range(-1, +3 + 1)),
            "Se_Sb": list(range(-1, +1 + 1)),
            "Sb_i": list(range(-1, +3 + 1)),
            "Se_i": list(range(0, +4 + 1)),
        }

        # note Sb2Se3 is a case where the lattice vectors are swapped to match the BCS conv cell definition
        sb2se3_defect_gen, output = self._generate_and_test_no_warnings(self.sb2se3)
        self._general_defect_gen_check(sb2se3_defect_gen)

        false_positives, false_negatives, hits = self.compare_doped_charges(
            sb2se3_tld_stable_charges, sb2se3_defect_gen
        )
        assert sum(len(val) for val in false_positives.values()) == 14
        assert sum(len(val) for val in false_negatives.values()) == 1
        assert sum(len(val) for val in hits.values()) == 27

        # all Sb2Se3 atoms occupy the 4c Wyckoff positions (10.1002/pssb.202000063,
        # 10.1107/S0365110X57000298, https://next-gen.materialsproject.org/materials/mp-2160/), which is
        # incorrectly determined if the pymatgen/spglib lattice vectors aren't swapped to match the BCS
        # convention, so this is a good check:

        assert (
            (
                """Vacancies       Guessed Charges    Conv. Cell Coords    Wyckoff
--------------  -----------------  -------------------  ---------
v_Sb_Cs_Se2.57  [-3,-2,-1,0,+1]    [0.537,0.250,0.355]  4c
v_Sb_Cs_Se2.63  [-3,-2,-1,0,+1]    [0.328,0.250,0.032]  4c
v_Se_Cs_Sb2.57  [-1,0,+1,+2]       [0.628,0.250,0.553]  4c
v_Se_Cs_Sb2.63  [-1,0,+1,+2]       [0.192,0.250,0.210]  4c
v_Se_Cs_Sb2.65  [-1,0,+1,+2]       [0.445,0.750,0.128]  4c

Substitutions    Guessed Charges              Conv. Cell Coords    Wyckoff
---------------  ---------------------------  -------------------  ---------
Sb_Se_Cs_Sb2.57  [-1,0,+1,+2,+3,+4,+5,+6,+7]  [0.628,0.250,0.553]  4c
Sb_Se_Cs_Sb2.63  [-1,0,+1,+2,+3,+4,+5,+6,+7]  [0.192,0.250,0.210]  4c
Sb_Se_Cs_Sb2.65  [-1,0,+1,+2,+3,+4,+5,+6,+7]  [0.445,0.750,0.128]  4c
Se_Sb_Cs_Se2.57  [-5,-4,-3,-2,-1,0,+1]        [0.537,0.250,0.355]  4c
Se_Sb_Cs_Se2.63  [-5,-4,-3,-2,-1,0,+1]        [0.328,0.250,0.032]  4c

Interstitials         Guessed Charges              Conv. Cell Coords    Wyckoff
--------------------  ---------------------------  -------------------  ---------
Sb_i_C1               [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.367,0.043,0.279]  8d
Sb_i_Cs_Sb2.14        [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.503,0.250,0.700]  4c
Sb_i_Cs_Sb2.32        [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.295,0.250,0.378]  4c
Sb_i_Cs_Sb2.63        [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.591,0.250,0.090]  4c
Sb_i_Cs_Se2.32Sb2.33  [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.393,0.250,0.217]  4c
Sb_i_Cs_Se2.32Sb2.40  [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.074,0.250,0.035]  4c
Sb_i_Cs_Se2.38        [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.293,0.750,0.263]  4c
Se_i_C1               [-2,-1,0,+1,+2,+3,+4]        [0.367,0.043,0.279]  8d
Se_i_Cs_Sb2.14        [-2,-1,0,+1,+2,+3,+4]        [0.503,0.250,0.700]  4c
Se_i_Cs_Sb2.32        [-2,-1,0,+1,+2,+3,+4]        [0.295,0.250,0.378]  4c
Se_i_Cs_Sb2.63        [-2,-1,0,+1,+2,+3,+4]        [0.591,0.250,0.090]  4c
Se_i_Cs_Se2.32Sb2.33  [-2,-1,0,+1,+2,+3,+4]        [0.393,0.250,0.217]  4c
Se_i_Cs_Se2.32Sb2.40  [-2,-1,0,+1,+2,+3,+4]        [0.074,0.250,0.035]  4c
Se_i_Cs_Se2.38        [-2,-1,0,+1,+2,+3,+4]        [0.293,0.750,0.263]  4c
\n"""
                "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in "
                "the conventional ('conv.') unit cell, which comprises 4 formula unit(s) of Sb2Se3.\n"
                "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
                "standard structure, for which doped uses the spglib convention."
            )
            in output
        )

    def test_lattice_vector_swapping(self):
        # Already tested above with Sb2Se3, but Ag2Se is also another tricky case so quick test with it too
        # note that this Ag2Se structure is not the groundstate, but just a case where the lattice vectors
        # are not in the standard orientation (spacegroup 17)
        # https://next-gen.materialsproject.org/materials/mp-568889
        ag2se_defect_gen, output = self._generate_and_test_no_warnings(self.ag2se)
        self._general_defect_gen_check(ag2se_defect_gen)

        assert (
            (
                """Vacancies       Guessed Charges    Conv. Cell Coords    Wyckoff
--------------  -----------------  -------------------  ---------
v_Ag_C1         [-1,0,+1]          [0.103,0.005,0.244]  4e
v_Ag_C2_Ag2.80  [-1,0,+1]          [0.391,0.000,0.000]  2a
v_Ag_C2_Ag2.85  [-1,0,+1]          [0.615,0.500,0.500]  2b
v_Se            [-1,0,+1,+2]       [0.294,0.520,0.251]  4e

Substitutions    Guessed Charges        Conv. Cell Coords    Wyckoff
---------------  ---------------------  -------------------  ---------
Ag_Se            [0,+1,+2,+3]           [0.294,0.520,0.251]  4e
Se_Ag_C1         [-3,-2,-1,0,+1,+2,+3]  [0.103,0.005,0.244]  4e
Se_Ag_C2_Ag2.80  [-3,-2,-1,0,+1,+2,+3]  [0.391,0.000,0.000]  2a
Se_Ag_C2_Ag2.85  [-3,-2,-1,0,+1,+2,+3]  [0.615,0.500,0.500]  2b

Interstitials    Guessed Charges    Conv. Cell Coords    Wyckoff
---------------  -----------------  -------------------  ---------
Ag_i_C1_Ag2.04   [0,+1,+2]          [0.335,0.435,0.002]  4e
Ag_i_C1_Ag2.05   [0,+1,+2]          [0.405,0.109,0.250]  4e
Ag_i_C2_Ag2.02   [0,+1,+2]          [0.500,0.250,0.184]  2d
Ag_i_C2_Ag2.48   [0,+1,+2]          [0.091,0.500,0.500]  2b
Se_i_C1_Ag2.04   [-2,-1,0]          [0.335,0.435,0.002]  4e
Se_i_C1_Ag2.05   [-2,-1,0]          [0.405,0.109,0.250]  4e
Se_i_C2_Ag2.02   [-2,-1,0]          [0.500,0.250,0.184]  2d
Se_i_C2_Ag2.48   [-2,-1,0]          [0.091,0.500,0.500]  2b
\n"""
                "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in "
                "the conventional ('conv.') unit cell, which comprises 4 formula unit(s) of Ag2Se.\n"
                "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
                "standard structure, for which doped uses the spglib convention."
            )
            in output
        )

        assert ag2se_defect_gen._BilbaoCS_conv_cell_vector_mapping == [1, 0, 2]

    def test_sb2si2te6(self):
        # weird case where the oxidation state guessing with max_sites = -1 gives a different (bad) guess
        # still a tricky case as we have rare Si+3 due to dumbbell formation

        sb2si2te6_defect_gen, output = self._generate_and_test_no_warnings(self.sb2si2te6)
        self._general_defect_gen_check(sb2si2te6_defect_gen)
        assert sb2si2te6_defect_gen.structure == self.sb2si2te6

        assert (  # different charge states than when max_sites = -1 is used:
            (
                """Vacancies    Guessed Charges     Conv. Cell Coords    Wyckoff
-----------  ------------------  -------------------  ---------
v_Si         [-4,-3,-2,-1,0,+1]  [0.000,0.000,0.445]  6c
v_Sb         [-2,-1,0,+1]        [0.000,0.000,0.166]  6c
v_Te         [-1,0,+1,+2]        [0.335,0.003,0.073]  18f

Substitutions    Guessed Charges              Conv. Cell Coords    Wyckoff
---------------  ---------------------------  -------------------  ---------
Si_Sb            [0,+1,+2]                    [0.000,0.000,0.166]  6c
Si_Te            [-2,-1,0,+1,+2,+3,+4,+5,+6]  [0.335,0.003,0.073]  18f
Sb_Si            [-7,-6,-5,-4,-3,-2,-1,0,+1]  [0.000,0.000,0.445]  6c
Sb_Te            [-1,0,+1,+2,+3,+4,+5,+6,+7]  [0.335,0.003,0.073]  18f
Te_Si            [-6,-5,-4,-3,-2,-1,0,+1,+2]  [0.000,0.000,0.445]  6c
Te_Sb            [-4,-3,-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.166]  6c

Interstitials    Guessed Charges              Conv. Cell Coords    Wyckoff
---------------  ---------------------------  -------------------  ---------
Si_i_C1_Si2.20   [-4,-3,-2,-1,0,+1,+2,+3,+4]  [0.179,0.359,0.167]  18f
Si_i_C1_Si2.28   [-4,-3,-2,-1,0,+1,+2,+3,+4]  [0.209,0.208,0.361]  18f
Si_i_C1_Si2.39   [-4,-3,-2,-1,0,+1,+2,+3,+4]  [0.001,0.336,0.243]  18f
Si_i_C3_Si2.64   [-4,-3,-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.318]  6c
Si_i_C3i_Te2.81  [-4,-3,-2,-1,0,+1,+2,+3,+4]  [0.000,0.000,0.000]  3a
Sb_i_C1_Si2.20   [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.179,0.359,0.167]  18f
Sb_i_C1_Si2.28   [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.209,0.208,0.361]  18f
Sb_i_C1_Si2.39   [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.001,0.336,0.243]  18f
Sb_i_C3_Si2.64   [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.000,0.000,0.318]  6c
Sb_i_C3i_Te2.81  [-3,-2,-1,0,+1,+2,+3,+4,+5]  [0.000,0.000,0.000]  3a
Te_i_C1_Si2.20   [-2,-1,0,+1,+2,+3,+4]        [0.179,0.359,0.167]  18f
Te_i_C1_Si2.28   [-2,-1,0,+1,+2,+3,+4]        [0.209,0.208,0.361]  18f
Te_i_C1_Si2.39   [-2,-1,0,+1,+2,+3,+4]        [0.001,0.336,0.243]  18f
Te_i_C3_Si2.64   [-2,-1,0,+1,+2,+3,+4]        [0.000,0.000,0.318]  6c
Te_i_C3i_Te2.81  [-2,-1,0,+1,+2,+3,+4]        [0.000,0.000,0.000]  3a
\n"""
                "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect "
                "in the conventional ('conv.') unit cell, which comprises 6 formula unit(s) of "
                "SiSbTe3.\n"
                "Note that Wyckoff letters can depend on the ordering of elements in the "
                "conventional standard structure, for which doped uses the spglib convention."
            )
            in output
        )

    def test_charge_state_gen_kwargs(self):
        # test adjusting probability_threshold with ZnS:
        zns_defect_gen, output = self._generate_and_test_no_warnings(
            self.non_diagonal_ZnS, charge_state_gen_kwargs={"probability_threshold": 0.1}
        )
        assert zns_defect_gen.charge_state_gen_kwargs == {"probability_threshold": 0.1}  # check
        # attribute set

        assert self.zns_defect_gen_info not in output
        for prev_string, new_string in [
            (
                "S_i_C3v          [-2,-1,0,+1,+2]",
                "S_i_C3v          [-2,-1,0]      ",
            ),
            (
                "S_i_Td_S2.35     [-2,-1,0,+1,+2]",
                "S_i_Td_S2.35     [-2,-1,0]      ",
            ),
            (
                "S_i_Td_Zn2.35    [-2,-1,0,+1,+2]",
                "S_i_Td_Zn2.35    [-2,-1,0]      ",
            ),
            (
                "S_Zn             [-4,-3,-2,-1,0,+1,+2]",
                "S_Zn             [-4,-3,-2,-1,0]    ",  # reduced column width with less charge states
            ),
        ]:
            assert prev_string not in output
            assert new_string in output

        self.zns_defect_gen_check(zns_defect_gen, check_info=False)
