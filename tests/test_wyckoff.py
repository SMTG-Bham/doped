"""
Tests for the `doped.utils.symmetry` module.
"""

import unittest

import numpy as np
import pytest
import spglib
from pymatgen.core import Lattice, Species
from test_utils import EXAMPLE_DIR, data_dir

from doped.utils.efficiency import SpacegroupAnalyzer, Structure
from doped.utils.symmetry import (
    _hermann_point_symmetry,
    _symmetrised_orbit_coords,
    get_all_equiv_sites,
    get_equiv_frac_coords_in_primitive,
    get_min_dist_between_equiv_sites,
    get_sga,
    get_wyckoff,
    get_wyckoff_dict_from_sgn,
    get_wyckoff_label_and_equiv_coord_list,
    point_symmetry_from_site,
    schoenflies_from_hermann,
    schoenflies_from_spacegroup_number,
)


class WyckoffTest(unittest.TestCase):
    def setUp(self):
        self.prim_cdte = Structure.from_file(f"{EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")
        sga = SpacegroupAnalyzer(self.prim_cdte)
        self.conv_cdte = sga.get_conventional_standard_structure()

    def test_wyckoff_dict_from_sgn(self):
        for sgn in range(1, 231):
            wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)
            assert isinstance(wyckoff_dict, dict)
            assert all(isinstance(k, str) for k in wyckoff_dict)
            assert all(isinstance(v, list) for v in wyckoff_dict.values())

    def test_schoenflies_from_spacegroup_number(self):
        """
        Cross-check ``doped``'s space group number -> point group (crystal
        class) table against ``spglib``'s own space-group-type data, for all
        230 space groups.
        """
        for hall_number in range(1, 531):  # all 230 space groups, over all their settings
            sg_type = spglib.get_spacegroup_type(hall_number)
            assert schoenflies_from_spacegroup_number(sg_type.number) == schoenflies_from_hermann(
                sg_type.pointgroup_international
            ), f"{sg_type.number} ({sg_type.international_short})"

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

        no_sgn_or_dict_error = ValueError(
            "If inputting `conv_cell_site` and not `defect_entry`, either `sgn` or `wyckoff_dict` "
            "must be provided."
        )
        with pytest.raises(ValueError) as e:
            _label, _equiv_coord_list = get_wyckoff_label_and_equiv_coord_list(
                conv_cell_site=self.conv_cdte[0],  # no sgn
            )
        assert str(no_sgn_or_dict_error) in str(e.value)


class SitePointSymmetryTest(unittest.TestCase):
    """
    Tests for determining site point symmetries from the symmetry operations
    which stabilise a site.
    """

    def test_matches_spglib_for_all_atomic_sites(self):
        """
        ``spglib`` labels the site symmetry of `atoms` directly, so its own
        ``site_symmetry_symbols`` are an independent ground truth to check the
        stabiliser-based determination against.
        """
        for filename in [
            "CdTe/relaxed_primitive_POSCAR",
            "YTOS/Bulk/POSCAR",
        ]:
            structure = Structure.from_file(f"{EXAMPLE_DIR}/{filename}")
            symm_dataset = get_sga(structure, symprec=0.01).get_symmetry_dataset()
            for i, site in enumerate(structure):
                assert schoenflies_from_hermann(
                    _hermann_point_symmetry(site.frac_coords, structure, symprec=0.01, dist_tol=0.01)
                ) == schoenflies_from_hermann(symm_dataset.site_symmetry_symbols[i]), (
                    f"{filename} site {i} ({site.specie})"
                )

    def test_orbit_stabilizer_relation_holds_by_construction(self):
        """
        The orbit and the stabiliser are read off a single partition of one
        operation set, so ``|orbit| x |stabiliser| == |G|`` must hold exactly.
        """
        structure = Structure.from_file(f"{EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")
        n_ops = len(get_sga(structure, symprec=0.01)._get_symmetry()[0])
        for frac_coords, expected_symmetry in [
            ([0.5, 0.5, 0.5], "Td"),  # tetrahedral interstitial void
            ([0.6, 0.6, 0.6], "C3v"),  # along the <111> axis
            ([0.6, 0.55, 0.4], "C1"),  # general position
        ]:
            point_symmetry = point_symmetry_from_site(np.array(frac_coords), structure, symprec=0.01)
            assert point_symmetry == expected_symmetry, f"{frac_coords}: {point_symmetry}"
            n_equiv = len(get_all_equiv_sites(np.array(frac_coords), structure, symprec=0.01))
            site_pg_order = {"Td": 24, "C3v": 6, "C1": 1}[expected_symmetry]
            assert n_equiv * site_pg_order == n_ops

    def test_off_ideal_site_multiplicity_and_wyckoff_label(self):
        """
        A site sitting slightly off an ideal position must still be recognised
        as being on that position (within ``symprec``).
        """
        structure = Structure.from_file(f"{data_dir}/Zn3P2_POSCAR")
        conv_structure = get_sga(structure, symprec=0.01).get_conventional_standard_structure()
        frac_coords = np.array([0.4994, 0.0, 0.0323])  # ~0.0096 Å off the 4d position

        assert len(get_all_equiv_sites(frac_coords, conv_structure, symprec=0.01)) == 4
        assert get_wyckoff(frac_coords, conv_structure, symprec=0.01) == "4d"
        assert point_symmetry_from_site(frac_coords, conv_structure, symprec=0.01) == "C2v"

        # the symmetry-averaged coordinates should snap onto the exact special positions:
        symmetrised_coords = _symmetrised_orbit_coords(frac_coords, conv_structure, 4, 0.01, 0.01)
        assert len(symmetrised_coords) == 4
        for coords in symmetrised_coords:
            assert any(np.allclose(coords[:2], xy, atol=1e-6) for xy in [(0.5, 0.0), (0.0, 0.5)])

    def test_oxidation_state_distinct_sublattices_not_merged(self):
        """
        The orientation-preserving primitive cell determination must
        distinguish species the same way ``SpacegroupAnalyzer`` does, i.e. by
        the full ``site.species``.

        Removing the oxidation states merges the two Fe sublattices into a
        single bcc lattice (Pm-3m -> Im-3m, with the input cell becoming the
        body-centred conventional cell), so the same fractional positions must
        then give different point symmetries / multiplicities.
        """
        mixed_valence = Structure(  # Pm-3m, so the two Fe sites are crystallographically distinct
            Lattice.cubic(4.0), [Species("Fe", 2), Species("Fe", 3)], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        single_valence = mixed_valence.copy()
        single_valence.remove_oxidation_states()  # now bcc Fe
        assert get_sga(mixed_valence, symprec=0.01).get_space_group_symbol() == "Pm-3m"
        assert get_sga(single_valence, symprec=0.01).get_space_group_symbol() == "Im-3m"

        for frac_coords, expected_syms_and_mults in [  # ITA special positions; (Pm-3m, Im-3m):
            ([0.25, 0.25, 0.25], (("C3v", 8), ("D3d", 8))),  # 8g .3m -> 8c .-3m
            ([0.25, 0.0, 0.0], (("C4v", 6), ("C4v", 12))),  # 6e 4m.m -> 12e 4m.m
            ([0.25, 0.25, 0.0], (("C2v", 12), ("C2v", 24))),  # 12i m.m2 -> 24h m.m2
        ]:
            for structure, (expected_symmetry, expected_multiplicity) in zip(
                (mixed_valence, single_valence), expected_syms_and_mults, strict=True
            ):
                assert (
                    point_symmetry_from_site(np.array(frac_coords), structure, symprec=0.01)
                    == expected_symmetry
                )
                assert (
                    len(get_all_equiv_sites(np.array(frac_coords), structure, symprec=0.01))
                    == expected_multiplicity
                )


class PrimitiveFoldingTest(unittest.TestCase):
    """
    Tests for folding sites from (possibly distorted) supercells to the
    primitive cell, via ``get_equiv_frac_coords_in_primitive`` and its
    consumers.
    """

    def test_distorted_supercell_folds_to_ideal_primitive_orbit(self):
        """
        A strained (1%) and rattled supercell has no strict-tolerance affine
        map to the ideal primitive cell, so folding falls back to a default-
        tolerance ``StructureMatcher`` match -- which must still fold the
        tetrahedral interstitial onto its ideal primitive-cell orbit.
        """
        prim_cdte = Structure(
            Lattice([[0, 3.29, 3.29], [3.29, 0, 3.29], [3.29, 3.29, 0]]),
            ["Cd", "Te"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        supercell = prim_cdte * [2, 2, 2]
        tet_interstitial_frac_coords = np.array([0.375, 0.375, 0.375])  # (0.75, 0.75, 0.75) in prim
        clean_orbit = get_equiv_frac_coords_in_primitive(
            tet_interstitial_frac_coords, prim_cdte, supercell
        )
        assert np.allclose(clean_orbit, [[0.75, 0.75, 0.75]])

        rng = np.random.default_rng(42)
        strained = Structure(
            Lattice(supercell.lattice.matrix * 1.01), supercell.species, supercell.frac_coords
        )
        rattled = Structure(
            strained.lattice,
            supercell.species,
            supercell.frac_coords
            + rng.normal(0, 0.01, (len(supercell), 3)) / np.array(strained.lattice.abc),
        )
        for distorted_supercell in [strained, rattled]:
            orbit = get_equiv_frac_coords_in_primitive(
                tet_interstitial_frac_coords, prim_cdte, distorted_supercell
            )
            assert orbit is not None  # fold map found despite no strict affine match
            assert len(orbit) == len(clean_orbit)
            assert np.allclose(sorted(map(tuple, orbit)), sorted(map(tuple, clean_orbit)), atol=2e-3)

    def test_min_dist_inf_for_inequivalent_same_formula_hosts(self):
        """
        ``get_min_dist_between_equiv_sites`` with two genuinely different host
        structures of the same reduced formula and primitive cell size (here
        rocksalt- vs CsCl-type NaCl) must return ``inf`` -- the default-
        tolerance ``StructureMatcher`` fallback used for folding approximately-
        equivalent hosts must not loosely match inequivalent structures.
        """
        rocksalt = Structure(
            Lattice([[0, 2.8, 2.8], [2.8, 0, 2.8], [2.8, 2.8, 0]]),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        cscl_type = Structure(Lattice.cubic(4.1), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        min_dist_between_hosts = get_min_dist_between_equiv_sites(
            [0.25, 0.25, 0.25], [0.25, 0.25, 0.25], structure=rocksalt, structure_2=cscl_type
        )
        assert min_dist_between_hosts == np.inf
