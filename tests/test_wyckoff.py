"""
Tests for the `doped.utils.wyckoff` module.
"""
import os
import unittest

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from doped.utils.wyckoff import get_wyckoff_dict_from_sgn, get_wyckoff_label_and_equiv_coord_list


class WyckoffTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.cdte_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.prim_cdte = Structure.from_file(f"{self.example_dir}/CdTe/relaxed_primitive_POSCAR")
        sga = SpacegroupAnalyzer(self.prim_cdte)
        self.conv_cdte = sga.get_conventional_standard_structure()

    def test_wyckoff_dict_from_sgn(self):
        for sgn in range(1, 231):
            wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)
            assert isinstance(wyckoff_dict, dict)
            assert all(isinstance(k, str) for k in wyckoff_dict)
            assert all(isinstance(v, list) for v in wyckoff_dict.values())

    def test_wyckoff_label_and_equiv_coord_list(self):
        """
        Here we test the `conv_cell_site` input to
        `get_wyckoff_label_and_equiv_coord_list`.

        The `defect_entry` input option is thoroughly (implicitly) tested in
        `test_generation.py`.
        """
        label, equiv_coord_list = get_wyckoff_label_and_equiv_coord_list(
            conv_cell_site=self.conv_cdte[0], sgn=216
        )
        assert label == "4a"

        for coord_array in equiv_coord_list:
            assert any(
                np.allclose(coord_array, x)
                for x in [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 0.5, 0.5]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.5, 0.5, 0.0]),
                ]
            )

        # test with a whack sgn it still runs fine:
        label, equiv_coord_list = get_wyckoff_label_and_equiv_coord_list(
            conv_cell_site=self.conv_cdte[0], sgn=21
        )

        # test by inputting wyckoff_dict and not sgn:
        wyckoff_dict = get_wyckoff_dict_from_sgn(216)
        label, equiv_coord_list = get_wyckoff_label_and_equiv_coord_list(
            conv_cell_site=self.conv_cdte[0], wyckoff_dict=wyckoff_dict
        )
        assert label == "4a"

        for coord_array in equiv_coord_list:
            assert any(
                np.allclose(coord_array, x)
                for x in [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 0.5, 0.5]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.5, 0.5, 0.0]),
                ]
            )

        with self.assertRaises(ValueError) as e:
            no_sgn_or_dict_error = ValueError(
                "If inputting `conv_cell_site` and not `defect_entry`, either `sgn` or `wyckoff_dict` "
                "must be provided."
            )
            label, equiv_coord_list = get_wyckoff_label_and_equiv_coord_list(
                conv_cell_site=self.conv_cdte[0],  # no sgn
            )
            assert no_sgn_or_dict_error in e.exception
