"""
Tests for the ``doped.utils.configurations`` module.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.core.structure_matcher import get_linear_assignment_solution, pbc_shortest_vectors
from test_utils import EXAMPLE_DIR, _run_func_and_capture_stdout_warnings, data_dir

from doped.core import DefectEntry
from doped.thermodynamics import DefectThermodynamics
from doped.utils.configurations import (
    _smart_round,
    apply_s2_to_s1_transformation,
    get_dQ,
    get_path_structures,
    get_s2_like_s1,
    get_transformation_from_s2_to_s1,
    orient_s2_like_s1,
    write_path_structures,
)
from doped.utils.mappings import (
    _cached_min_separation,
    _get_site_mapping_from_coords_and_indices,
    _min_separation,
    _nearest_neighbour_site_mapping,
    check_atom_mapping_far_from_defect,
    find_missing_idx,
    get_site_mappings,
    get_wigner_seitz_radius,
)
from doped.utils.supercells import min_dist
from doped.utils.symmetry import get_clean_structure, point_symmetry_from_structure, summed_dist


class ConfigurationsTestCase(unittest.TestCase):
    """
    Base test case for ``doped.utils.configurations`` tests; loads the defect
    supercells used in the `CC / NEB tutorial
    <https://doped.readthedocs.io/en/latest/CCD_NEB_tutorial.html>`__.
    """

    @classmethod
    def setUpClass(cls):
        cls.Se_example_dir = os.path.join(EXAMPLE_DIR, "Se")
        cls.Se_intrinsic_thermo = DefectThermodynamics.from_json(
            f"{cls.Se_example_dir}/Se_Intrinsic_Thermo.json.gz"
        )
        # uses old (v1) names:
        cls.V_Se_m1_supercell = cls.Se_intrinsic_thermo["vac_1_Se_-1"].defect_supercell
        cls.V_Se_m2_supercell = cls.Se_intrinsic_thermo["vac_1_Se_-2"].defect_supercell

        # binary CdTe vacancies (different charge states), from ``tests/data``:
        # remember v_Cd_0 has vacancy at [0.5, 0.5, 0.5], v_Cd_m1 has it at [0, 0, 0]
        cls.v_Cd_0 = DefectEntry.from_json(f"{data_dir}/v_Cd_defect_entry.json.gz")
        cls.v_Cd_m1 = DefectEntry.from_json(f"{data_dir}/v_Cd_m1_defect_entry.json.gz")


class TestGetDQ(ConfigurationsTestCase):
    """
    Tests for ``get_dQ``.
    """

    def test_identical_structures(self):
        """
        ``get_dQ`` should be zero for a structure against itself.
        """
        assert np.isclose(get_dQ(self.V_Se_m1_supercell, self.V_Se_m1_supercell), 0.0)

    def test_known_value_Se_vacancies(self):
        """
        Check the reference ΔQ value between ``V_Se^-1`` and ``V_Se^-2`` from
        the CCD/NEB tutorial (before re-orienting).
        """
        # tutorial: ΔQ(s1/s2) = 9.91 amu^(1/2)Å
        assert np.isclose(get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell), 9.91, atol=1e-2)

    def test_mismatched_structures_returns_inf(self):
        """
        If the structures have different numbers of sites, ``get_dQ`` should
        return ``np.inf`` (i.e. indicate no valid mapping).
        """
        reduced = Structure.from_sites(self.V_Se_m1_supercell.sites[:-1])
        assert get_dQ(self.V_Se_m1_supercell, reduced) == np.inf

    def test_ignored_species(self):
        """
        ``ignored_species`` should remove sites of that species from the ΔQ
        calculation.

        When all species are ignored, ΔQ should be zero.
        """
        dQ_all_ignored = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell, ignored_species=["Se"])
        assert np.isclose(dQ_all_ignored, 0.0)

        dQ_none_ignored = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell)
        assert not np.isclose(dQ_none_ignored, 0.0)

    def test_reorient_true_matches_manual_reorientation(self):
        """
        ``reorient=True`` should compute ΔQ after internally applying
        ``orient_s2_like_s1`` to ``struct2``.
        """
        dQ_no_reorient = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell, reorient=False)
        dQ_no_reorient_default = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell)
        assert np.isclose(dQ_no_reorient, dQ_no_reorient_default, atol=1e-6)

        dQ_reorient_kwarg = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell, reorient=True)
        V_Se_m2_like_m1 = orient_s2_like_s1(self.V_Se_m1_supercell, self.V_Se_m2_supercell)
        dQ_manual = get_dQ(self.V_Se_m1_supercell, V_Se_m2_like_m1)

        assert np.isclose(dQ_reorient_kwarg, dQ_manual, atol=1e-6)
        assert not np.isclose(dQ_no_reorient, dQ_manual, atol=1e-6)

    def test_reorient_true_with_ignored_species(self):
        """
        ``ignored_species`` should also be respected when ``reorient=True``.
        """
        dQ_reorient_kwarg = get_dQ(
            self.v_Cd_0.defect_supercell,
            self.v_Cd_m1.defect_supercell,
            ignored_species=["Te"],
            reorient=True,
        )
        dQ_manual = get_dQ(
            self.v_Cd_0.defect_supercell,
            orient_s2_like_s1(
                self.v_Cd_0.defect_supercell,
                self.v_Cd_m1.defect_supercell,
                ignored_species=["Te"],
            ),
            ignored_species=["Te"],
        )

        assert np.isclose(dQ_reorient_kwarg, dQ_manual, atol=1e-6)

    def test_reorient_true_matches_site_mapping_formula(self):
        """
        ``get_dQ(..., reorient=True)`` should match the equivalent ΔQ computed
        from ``get_site_mappings`` with a mapped weighted-distance sum, **if**
        no re-orientation is required (only re-ordering).
        """
        struct1 = self.V_Se_m1_supercell
        struct2 = get_clean_structure(self.V_Se_m2_supercell)  # re-order to break ordering match

        raw_dQ_no_reorient = get_dQ(struct1, struct2)
        dQ_reorient = get_dQ(struct1, struct2, reorient=True)
        dQ_from_mapping = np.sqrt(
            sum(
                (struct1[i].distance(struct2[j]) ** 2) * struct1[i].specie.atomic_mass
                for _, i, j in get_site_mappings(struct1, struct2)
                if i is not None and j is not None
            )
        )
        assert raw_dQ_no_reorient > 500  # large dQ due to ordering mismatch
        assert dQ_from_mapping < 10  # much lower dQ with mapping
        assert dQ_reorient < dQ_from_mapping
        assert dQ_reorient < 9  # lower dQ again with re-orientation


class TestOrientS2LikeS1(ConfigurationsTestCase):
    """
    Tests for ``orient_s2_like_s1`` / ``get_s2_like_s1``.
    """

    def test_reorientation_reduces_dQ(self):
        """
        Re-orienting ``V_Se^-2`` to match ``V_Se^-1`` should (monotonically)
        reduce the mass-weighted displacement (ΔQ) between the two structures,
        matching the reference values from the CC/NEB tutorial.
        """
        V_Se_m2_like_m1 = orient_s2_like_s1(self.V_Se_m1_supercell, self.V_Se_m2_supercell)

        dQ_original = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_supercell)
        dQ_reoriented = get_dQ(self.V_Se_m1_supercell, V_Se_m2_like_m1)
        dQ_s2_to_s2_like_s1 = get_dQ(V_Se_m2_like_m1, self.V_Se_m2_supercell)

        assert dQ_reoriented < dQ_original
        # reference values from the CC/NEB tutorial notebook:
        assert np.isclose(dQ_original, 9.91, atol=1e-2)
        assert np.isclose(dQ_reoriented, 8.63, atol=1e-2)
        assert np.isclose(dQ_s2_to_s2_like_s1, 4.88, atol=1e-2)

    def test_symmetry_and_min_dist_preserved(self):
        """
        Re-orienting should be a symmetry-equivalent transformation (same point
        group symmetry and same minimum interatomic distance), since the
        geometry should not change.
        """
        V_Se_m2_like_m1 = orient_s2_like_s1(self.V_Se_m1_supercell, self.V_Se_m2_supercell)

        assert point_symmetry_from_structure(V_Se_m2_like_m1) == point_symmetry_from_structure(
            self.V_Se_m2_supercell
        )
        assert np.isclose(min_dist(V_Se_m2_like_m1), min_dist(self.V_Se_m2_supercell), atol=1e-3)

    def test_get_s2_like_s1_alias(self):
        """
        ``get_s2_like_s1`` is just an alias for ``orient_s2_like_s1``.
        """
        assert get_s2_like_s1 is orient_s2_like_s1

    def test_verbose_prints_dQ_info(self):
        """
        ``verbose=True`` should print the three ΔQ values, with the numbers
        matching the CC/NEB tutorial reference values (9.91, 4.88, 8.63).
        """
        _, output, w = _run_func_and_capture_stdout_warnings(
            orient_s2_like_s1, self.V_Se_m1_supercell, self.V_Se_m2_supercell, verbose=True
        )
        # tutorial reference: ΔQ(s1/s2) = 9.91, ΔQ(s2_like_s1/s2) = 4.88,
        # ΔQ(s1/s2_like_s1) = 8.63 amu^(1/2)Å
        assert "\u0394Q(s1/s2) = 9.91 amu^(1/2)Å" in output
        assert "\u0394Q(s2_like_s1/s2) = 4.88 amu^(1/2)Å" in output
        assert "\u0394Q(s1/s2_like_s1) = 8.63 amu^(1/2)Å" in output
        assert not w

        # without ``verbose=True`` nothing is printed:
        _, output, w = _run_func_and_capture_stdout_warnings(
            orient_s2_like_s1, self.V_Se_m1_supercell, self.V_Se_m2_supercell
        )
        assert "ΔQ" not in output
        assert not w

    def test_matching_site_indices_after_reorientation_CdTe(self):
        """
        The output structure should have sites of matching species at matching
        indices as ``struct1``, suitable for VASP NEB / ``nonrad`` usage.

        Uses the CdTe Cd-vacancy defect supercells which have significantly
        different lattice-positional offsets (ΔQ > 200 amu^(1/2)Å between the
        two structures before re-orientation, dropping to sub-amu^(1/2)Å scale
        after re-orientation).
        """
        # different vacancy sites:
        assert np.allclose(self.v_Cd_0.defect_supercell_site.frac_coords, [0.5, 0.5, 0.5])
        assert np.allclose(self.v_Cd_m1.defect_supercell_site.frac_coords, [0.0, 0.0, 0.0])

        v_Cd_m1_like_0 = orient_s2_like_s1(self.v_Cd_0.defect_supercell, self.v_Cd_m1.defect_supercell)
        assert len(v_Cd_m1_like_0) == len(self.v_Cd_0.defect_supercell)
        # ensure species (and frac_coords) match index-by-index in the re-oriented output:
        for i, site in enumerate(v_Cd_m1_like_0.sites):
            assert site.specie == self.v_Cd_0.defect_supercell[i].specie
            assert np.allclose(site.frac_coords, self.v_Cd_0.defect_supercell[i].frac_coords, atol=5e-2)
        assert not all(
            np.allclose(site.frac_coords, self.v_Cd_0.defect_supercell[i].frac_coords, atol=5e-2)
            for i, site in enumerate(self.v_Cd_m1.defect_supercell.sites)
        )
        assert {site.specie.symbol for site in v_Cd_m1_like_0} == {"Cd", "Te"}

        # reorientation should substantially reduce ΔQ here (original
        # ``v_Cd_0`` and ``v_Cd_m1`` structures have inconsistent atomic
        # orderings and vacancy located at different fractional positions;
        # expected drop of at least 5x from the un-reoriented value:
        dQ_original = get_dQ(self.v_Cd_0.defect_supercell, self.v_Cd_m1.defect_supercell)
        dQ_reoriented = get_dQ(self.v_Cd_0.defect_supercell, v_Cd_m1_like_0)
        assert dQ_original > 100.0
        assert dQ_reoriented < dQ_original / 5

    def test_inequivalent_lattices_warn(self):
        """
        Re-orienting between structures with inequivalent lattices (but
        matching compositions) should raise a warning about the lattices being
        ``(symmetry-)inequivalent``.

        Here we anisotropically stretch
        ``V_Se^-2`` to get an inequivalent lattice with matching composition.
        """
        stretched_lattice = self.V_Se_m2_supercell.lattice.matrix.copy()
        stretched_lattice[2] *= 1.1  # stretch c-axis by 10 %
        stretched_V_Se_m2 = Structure(
            lattice=stretched_lattice,
            species=[site.specie for site in self.V_Se_m2_supercell],
            coords=self.V_Se_m2_supercell.frac_coords,
            coords_are_cartesian=False,
        )

        _, _, w = _run_func_and_capture_stdout_warnings(
            orient_s2_like_s1, self.V_Se_m1_supercell, stretched_V_Se_m2
        )
        assert len(w) == 2
        assert (
            "The lattices of the two input structures have been detected to be (symmetry-)inequivalent. "
        ) in str(w[0].message)
        assert ("Note that the lattice definitions may differ between the output structure") in str(
            w[-1].message
        )

    def test_mismatched_compositions_raises(self):
        """
        Trying to orient structures with different compositions / too different
        lattices should raise a ``RuntimeError``.
        """
        # create a structure with clearly incompatible composition:
        other = Structure(lattice=np.eye(3) * 5, species=["Si"], coords=[[0, 0, 0]])
        with pytest.raises(RuntimeError, match="get_transformation"):
            orient_s2_like_s1(other, self.V_Se_m1_supercell)

    def test_check_mapping_well_matched_no_warning(self):
        """
        With the default ``check_mapping=True`` and well-matched structures
        (V_Se^-1 vs V_Se^-2, which re-orient to ΔQ ~8.6 amu^(1/2)Å with
        matching atomic basis), no lattice-mismatch warning should be raised.
        """
        _, _, w = _run_func_and_capture_stdout_warnings(
            orient_s2_like_s1, self.V_Se_m1_supercell, self.V_Se_m2_supercell
        )
        assert not any("significant atomic displacements remain" in str(warning.message) for warning in w)

    def test_check_mapping_detects_lattice_basis_mismatch(self):
        """
        When re-orientation cannot reconcile an atomic-basis mismatch between
        the two input structures (e.g. different primitive-cell tiling giving
        identical lattice vectors but inequivalent atomic positions), the
        ``check_mapping`` check should raise a warning.

        Here we simulate a basis mismatch by applying a large (non-rigid)
        shear-like perturbation to the upper half of ``V_Se^-2``, which
        cannot be resolved by any rigid transformation applied during re-
        orientation -> leaves significant residual displacements at sites
        far from the cell centre.
        """
        perturbed = self.V_Se_m2_supercell.copy()
        for i, site in enumerate(perturbed):
            if site.frac_coords[2] > 0.5:  # displace upper-half atoms by 1 Å along x
                perturbed.translate_sites(i, [1.0, 0, 0], frac_coords=False, to_unit_cell=False)

        # ``stol=0.6`` to loosen ``StructureMatcher`` tolerances enough to find a transformation:
        _, _, w = _run_func_and_capture_stdout_warnings(
            orient_s2_like_s1, self.V_Se_m1_supercell, perturbed, stol=0.6
        )
        assert any(
            "significant site mismatches remain" in str(warning.message)
            and "lattice definitions" in str(warning.message)
            and "check_mapping=False" in str(warning.message)
            for warning in w
        )
        assert len(w) == 1

        # with ``check_mapping=False`` the warning should be suppressed:
        _, _, w = _run_func_and_capture_stdout_warnings(
            orient_s2_like_s1,
            self.V_Se_m1_supercell,
            perturbed,
            stol=0.6,
            check_mapping=False,
        )
        assert not w

    def test_check_atom_mapping_detects_partial_mismatch(self):
        """
        ``check_atom_mapping_far_from_defect`` should detect a *partial* atomic
        basis mismatch -- where only a fraction of far-from-defect sites are
        strongly displaced and the rest are well-matched -- which a mean/median
        displacement metric would dilute/hide, while remaining robust to a
        small number of outlier sites.
        """
        bulk = self.Se_intrinsic_thermo["vac_1_Se_-1"].bulk_supercell
        defect_coords = np.array([0.5, 0.5, 0.5])
        ws_radius = get_wigner_seitz_radius(bulk.lattice)
        far_indices = [  # sites outside the defect Wigner-Seitz radius
            i
            for i in range(len(bulk))
            if bulk.lattice.get_all_distances(bulk[i].frac_coords, defect_coords).ravel()[0] > ws_radius
        ]

        def _displace_fraction(fraction, magnitude):
            # displace ``fraction`` of the far-from-defect sites by ``magnitude`` Å
            structure = bulk.copy()
            chosen = np.random.default_rng(0).choice(
                far_indices, size=round(fraction * len(far_indices)), replace=False
            )
            for i in chosen:
                structure.translate_sites(int(i), [magnitude, 0, 0], frac_coords=False, to_unit_cell=True)
            return structure

        # 40% strongly displaced -> mismatch detected (returns False), even though 60% match perfectly
        # (a median displacement would sit in the well-matched 60% and miss this):
        assert not check_atom_mapping_far_from_defect(
            _displace_fraction(0.4, 3.0), bulk, defect_coords, warning=False
        )
        # a small fraction (10%) of outlier sites should not trigger a (false-positive) mismatch:
        assert check_atom_mapping_far_from_defect(
            _displace_fraction(0.1, 3.0), bulk, defect_coords, warning=False
        )
        # a well-matched supercell (no displacements) should not be flagged:
        assert check_atom_mapping_far_from_defect(bulk, bulk, defect_coords, warning=False)

        # test displacing 25% of far-from-defect atoms by 1 Å (mean would not catch,
        # but significant-fraction check should):
        assert not check_atom_mapping_far_from_defect(
            _displace_fraction(0.25, 1.0), bulk, defect_coords, warning=False
        )

        # but then bumping ``fraction_tol`` then accepts this partial mismatch:
        assert check_atom_mapping_far_from_defect(
            _displace_fraction(0.25, 1.0), bulk, defect_coords, fraction_tol=0.3, warning=False
        )


class TestGetTransformationAndApply(ConfigurationsTestCase):
    """
    Tests for ``get_transformation_from_s2_to_s1`` and
    ``apply_s2_to_s1_transformation``.
    """

    def test_transformation_matches_orient_s2_like_s1(self):
        """
        Manually applying the transformation returned by
        ``get_transformation_from_s2_to_s1`` should reproduce the output of
        ``orient_s2_like_s1``.
        """
        supercell_matrix, trans_vector, mapping = get_transformation_from_s2_to_s1(
            self.V_Se_m1_supercell, self.V_Se_m2_supercell
        )
        assert supercell_matrix.shape == (3, 3)
        assert trans_vector.shape == (3,)
        assert len(mapping) >= len(self.V_Se_m1_supercell)

        manual = apply_s2_to_s1_transformation(
            self.V_Se_m1_supercell,
            self.V_Se_m2_supercell,
            supercell_matrix,
            trans_vector,
            mapping,
        )
        auto = orient_s2_like_s1(self.V_Se_m1_supercell, self.V_Se_m2_supercell)

        # compositions must match and dQ must be equivalent (up to tiny numerical noise):
        assert manual.composition == auto.composition
        assert np.isclose(get_dQ(manual, auto), 0.0, atol=1e-6)

    def test_primitive_cell_raises(self):
        """
        ``primitive_cell=True`` is not supported.
        """
        with pytest.raises(ValueError, match="primitive_cell=True"):
            get_transformation_from_s2_to_s1(
                self.V_Se_m1_supercell, self.V_Se_m2_supercell, primitive_cell=True
            )

    def test_apply_new_lattice_options(self):
        """
        Check the different ``new_lattice`` options for
        ``apply_s2_to_s1_transformation``.

        ``V_Se_m1_supercell`` and ``V_Se_m2_supercell`` have identical lattice
        matrices as loaded from ``doped``, so to make the ``"struct1"``,
        ``"struct2"`` and ``"s2_like_s1"`` options produce (potentially)
        distinct output lattices we first apply a symmetry-equivalent 120°
        rotation about the hexagonal c-axis to ``struct2`` (which produces a
        different lattice matrix, equivalent by lattice symmetry).
        """
        # rotate V_Se^-2 by 120° about c-axis (a lattice symmetry of hexagonal
        # Se), producing an equivalent lattice with a different matrix:
        rotated_V_Se_m2 = self.V_Se_m2_supercell.copy()
        rotated_V_Se_m2.apply_operation(SymmOp.from_axis_angle_and_translation([0, 0, 1], 120.0))
        assert not np.allclose(rotated_V_Se_m2.lattice.matrix, self.V_Se_m1_supercell.lattice.matrix), (
            "Rotated V_Se^-2 lattice matrix should now differ from V_Se^-1"
        )

        supercell_matrix, trans_vector, mapping = get_transformation_from_s2_to_s1(
            self.V_Se_m1_supercell, rotated_V_Se_m2
        )

        out_struct1 = apply_s2_to_s1_transformation(
            self.V_Se_m1_supercell,
            rotated_V_Se_m2,
            supercell_matrix,
            trans_vector,
            mapping,
            new_lattice="struct1",
        )
        np.testing.assert_allclose(
            out_struct1.lattice.matrix, self.V_Se_m1_supercell.lattice.matrix, atol=1e-6
        )

        out_struct2 = apply_s2_to_s1_transformation(
            self.V_Se_m1_supercell,
            rotated_V_Se_m2,
            supercell_matrix,
            trans_vector,
            mapping,
            new_lattice="struct2",
        )
        np.testing.assert_allclose(out_struct2.lattice.matrix, rotated_V_Se_m2.lattice.matrix, atol=1e-6)
        # ``"struct2"`` lattice should differ from ``"struct1"``:
        assert not np.allclose(out_struct1.lattice.matrix, out_struct2.lattice.matrix)

        out_s2_like_s1 = apply_s2_to_s1_transformation(
            self.V_Se_m1_supercell,
            rotated_V_Se_m2,
            supercell_matrix,
            trans_vector,
            mapping,
            new_lattice="s2_like_s1",
        )
        # ``s2_like_s1`` lattice can (and does often) differ from ``struct1``:
        assert not np.allclose(out_s2_like_s1.lattice.matrix, self.V_Se_m1_supercell.lattice.matrix)
        # ``s2_like_s1`` lattice often matches ``struct2``, as here, but not guaranteed (may change in
        # future):
        assert np.allclose(out_s2_like_s1.lattice.matrix, rotated_V_Se_m2.lattice.matrix)

        # all three output structures should have the same composition and be
        # symmetry-equivalent (same point group and min distance):
        for out in (out_struct1, out_struct2, out_s2_like_s1):
            assert out.composition == rotated_V_Se_m2.composition
            assert np.isclose(min_dist(out), min_dist(rotated_V_Se_m2), atol=1e-3)

    def test_apply_invalid_new_lattice_raises(self):
        """
        An invalid ``new_lattice`` string should raise a ``ValueError``.
        """
        supercell_matrix, trans_vector, mapping = get_transformation_from_s2_to_s1(
            self.V_Se_m1_supercell, self.V_Se_m2_supercell
        )
        with pytest.raises(ValueError, match="Invalid value for ``new_lattice``"):
            apply_s2_to_s1_transformation(
                self.V_Se_m1_supercell,
                self.V_Se_m2_supercell,
                supercell_matrix,
                trans_vector,
                mapping,
                new_lattice="not_a_valid_option",
            )


class TestGetPathStructures(ConfigurationsTestCase):
    """
    Tests for ``get_path_structures``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.V_Se_m2_like_m1 = orient_s2_like_s1(cls.V_Se_m1_supercell, cls.V_Se_m2_supercell)

        # ``V_Se^-1`` supercell shifted by (1/3, 0, 0) — one primitive `a` lattice vector in this 3x3x1
        # supercell — so the structure is periodically equivalent to ``V_Se_m1_supercell`` (same defect
        # at a symmetry-equivalent site). ``orient_s2_like_s1()`` maps it back onto the unshifted
        # ``V_Se_m1_supercell`` exactly (ΔQ_before large, ΔQ_after ~ 0), used to test the
        # NEB-between-symmetry-equivalent-sites warning path:
        cls.shifted_V_Se_m1 = cls.V_Se_m1_supercell.copy()
        cls.shifted_V_Se_m1.translate_sites(
            list(range(len(cls.shifted_V_Se_m1))), [1 / 3, 0, 0], to_unit_cell=True
        )

    def test_neb_mode_default_n_images(self):
        """
        ``get_path_structures`` with default ``n_images=7`` (NEB mode) should
        return a single ``dict`` with keys ``00``...``07`` and 8 interpolated
        structures matching the endpoints.

        Additionally, the mass-weighted displacements ``ΔQ(struct1, struct_i)``
        should (initially) scale linearly with image index (``i/7``) along the
        path from ``struct1`` to ``struct2``, consistent with linear
        interpolation.
        """
        result = get_path_structures(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
        assert isinstance(result, dict)
        assert list(result) == [f"0{i}" for i in range(8)]
        # endpoints should be the input structures (atomic positions match):
        np.testing.assert_allclose(result["00"].frac_coords, self.V_Se_m1_supercell.frac_coords, atol=1e-6)
        np.testing.assert_allclose(result["07"].frac_coords, self.V_Se_m2_like_m1.frac_coords, atol=1e-6)

        # ΔQ scales linearly with (image index / n_images):
        dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
        dQs = [get_dQ(self.V_Se_m1_supercell, result[f"0{i}"]) for i in range(8)]
        # monotonically increasing, with expected linear scaling:
        assert dQs == sorted(dQs), f"ΔQ values should be monotonically increasing, got {dQs}"
        for i, dQ in enumerate(dQs):
            assert np.isclose(dQ, (i / 7) * dQ_total, atol=1e-3)

    def test_neb_mode_with_n_images_list(self):
        """
        When ``n_images`` is a list of fractional displacements (still in NEB
        mode since ``displacements`` is not set), the keys should be the
        ``delQ_<x>`` labels and one dict should be returned.

        Again we verify ``ΔQ(struct1, struct_d)`` scales linearly with ``|d|``.
        """
        displacements = [0.0, 0.25, 0.5, 1.0]
        result = get_path_structures(self.V_Se_m1_supercell, self.V_Se_m2_like_m1, n_images=displacements)
        assert isinstance(result, dict)
        assert list(result) == [f"delQ_{d}" for d in displacements]

        dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
        for d in displacements:
            dQ = get_dQ(self.V_Se_m1_supercell, result[f"delQ_{d}"])
            assert np.isclose(dQ, d * dQ_total, atol=1e-3)

    def test_cc_mode_with_displacements(self):
        """
        Setting ``displacements`` should enable CC-diagram mode, returning a
        tuple of two dictionaries (with ``delQ_<x>`` keys) -- one for each set
        of interpolated structures.

        In ``disp_dict_1``, ``ΔQ(struct1, disp_dict_1[delQ_d])`` should equal
        ``|d| * ΔQ(struct1, struct2)``, and in ``disp_dict_2``,
        ``ΔQ(struct2, disp_dict_2[delQ_d])`` should equal
        ``|d| * ΔQ(struct1, struct2)``.
        """
        displacements = [-0.5, 0.0, 0.5, 1.0]
        result = get_path_structures(
            self.V_Se_m1_supercell, self.V_Se_m2_like_m1, displacements=displacements
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        disp_dict_1, disp_dict_2 = result
        expected_keys = ["delQ_-0.5", "delQ_0.0", "delQ_0.5", "delQ_1.0"]
        assert list(disp_dict_1) == expected_keys
        assert list(disp_dict_2) == expected_keys

        dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
        for d in displacements:
            dQ_1 = get_dQ(self.V_Se_m1_supercell, disp_dict_1[f"delQ_{d}"])
            dQ_2 = get_dQ(self.V_Se_m2_like_m1, disp_dict_2[f"delQ_{d}"])
            assert np.isclose(dQ_1, abs(d) * dQ_total, atol=1e-3)
            assert np.isclose(dQ_2, abs(d) * dQ_total, atol=1e-3)

        # ``delQ_0.0`` should reproduce the corresponding endpoint structure:
        np.testing.assert_allclose(
            disp_dict_1["delQ_0.0"].frac_coords, self.V_Se_m1_supercell.frac_coords, atol=1e-6
        )
        np.testing.assert_allclose(
            disp_dict_2["delQ_0.0"].frac_coords, self.V_Se_m2_like_m1.frac_coords, atol=1e-6
        )

    def test_reorient_default_with_matched_inputs(self):
        """
        With already-matched inputs (``struct2`` already re-oriented to match
        ``struct1``), the default ``reorient=None`` should be a no-op in
        practice: no warnings should be raised, and the result should match the
        ``reorient=False`` case (which skips re-orientation entirely).
        """
        result_default, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_like_m1
        )
        assert not w

        result_no_reorient, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_like_m1, reorient=False
        )
        assert not w
        for key in result_default:
            np.testing.assert_allclose(
                result_default[key].frac_coords, result_no_reorient[key].frac_coords, atol=1e-6
            )

    def test_reorient_none_default_with_mismatched_inputs_warns_and_reorients(self):
        """
        With mismatched inputs (``struct2`` not re-oriented to match
        ``struct1``) and the default ``reorient=None``, ``struct2`` should be
        re-oriented (producing the same output as if re-oriented externally
        with ``reorient=False``) and a warning should be raised indicating re-
        orientation was required.

        With ``reorient=True`` on mismatched inputs, ``struct2`` should be
        re-oriented (matching the ``reorient=None`` re-oriented output) but
        `no` re-orientation warning should be raised (user explicitly requested
        re-orientation).
        """
        result, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_supercell
        )
        assert any(
            "did not have a matching orientation" in str(warning.message)
            and "reorient=False" in str(warning.message)
            and "ΔQ decreased from" in str(warning.message)
            for warning in w
        )
        assert len(w) == 1

        # result should match pre-orienting ``struct2`` first (i.e. equivalent to feeding in
        # ``V_Se_m2_like_m1`` with ``reorient=False``):
        reference, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_like_m1, reorient=False
        )
        assert not w

        # re-orientation with no warning with reorient=True:
        result_true, _, w_true = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_supercell, reorient=True
        )
        assert not w_true

        for key in result:
            np.testing.assert_allclose(result[key].frac_coords, reference[key].frac_coords, atol=1e-6)
            np.testing.assert_allclose(result_true[key].frac_coords, reference[key].frac_coords, atol=1e-6)

    def test_reorient_false_skips_reorientation(self):
        """
        With ``reorient=False``, mismatched inputs should not be re-oriented
        (``orient_s2_like_s1`` is not applied) and no warning should be raised,
        even if re-orientation would have meaningfully reduced ΔQ.

        The raw endpoint should differ from the re-oriented output (note:
        ``pymatgen``'s ``Structure.interpolate`` still auto-sorts atoms to
        match ``struct1`` via ``autosort_tol``, but this only reorders sites
        and does not apply the supercell transformation / unit cell
        translations performed by ``orient_s2_like_s1``).
        """
        result_no_reorient, stdout, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_supercell, reorient=False
        )
        assert not w
        assert not stdout

        result_reorient, stdout, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.V_Se_m2_supercell, reorient=True
        )
        assert not w
        assert not stdout

        # the two endpoint structures should differ (re-orientation has a real effect):
        assert not np.allclose(
            result_no_reorient["07"].frac_coords, result_reorient["07"].frac_coords, atol=1e-3
        )

    def test_neb_between_symmetry_equivalent_sites_warns_and_skips_reorient(self):
        """
        In NEB mode (``displacements=None``) with the default
        ``reorient=None``, if re-orienting ``struct2`` would reduce ΔQ below
        ``0.1`` amu^(1/2)Å, this is assumed to be an NEB between different
        symmetry-equivalent configurations (where re-orientation would collapse
        the intended migration path) and re-orientation should be `skipped`,
        with a warning raised (mentioning the small mass-weighted atomic
        displacement).

        Here ``struct2`` is ``struct1`` shifted by one primitive lattice
        vector (a symmetry-equivalent configuration in the supercell), so
        ``orient_s2_like_s1`` would map it back onto ``struct1`` exactly
        (ΔQ_after ~ 0).
        """
        result, stdout, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.shifted_V_Se_m1
        )
        assert not stdout
        assert any(
            "small mass-weighted atomic displacement" in str(warning.message)
            and "symmetry-equivalent" in str(warning.message)
            and "reorient" in str(warning.message)
            for warning in w
        )

        # verify re-orientation was skipped by comparing against the ``reorient=True`` output (which
        # `does` apply re-orientation, mapping ``shifted_V_Se_m1`` back onto ``V_Se_m1_supercell``
        # exactly -- giving ΔQ ~ 0). With re-orientation skipped, the endpoint retains the lattice
        # shift and ΔQ relative to ``V_Se_m1_supercell`` should be substantially larger:
        result_reoriented, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.shifted_V_Se_m1, reorient=True
        )
        assert not w
        assert get_dQ(self.V_Se_m1_supercell, result_reoriented["07"]) < 0.1
        assert get_dQ(self.V_Se_m1_supercell, result["07"]) > 10
        assert np.isclose(get_dQ(self.V_Se_m1_supercell, result["00"]), 0)
        # note that result["07"] is still a re-`ordered` (but not reoriented) version of
        # ``self.shifted_V_Se_m1``, due to the auto-sorting in ``Structure.interpolate()``
        assert not np.isclose(get_dQ(self.shifted_V_Se_m1, result["07"]), 0)
        assert np.isclose(
            get_dQ(get_clean_structure(self.shifted_V_Se_m1), get_clean_structure(result["07"])), 0
        )

    def test_neb_between_symmetry_equivalent_sites_reorient_true_suppresses_warning(self):
        """
        With the same symmetry-equivalent NEB inputs but ``reorient=True``
        explicit, re-orientation is forced and no warning is raised; the two
        endpoint structures become effectively identical (ΔQ ~ 0 across all
        images since ``struct2`` is mapped onto ``struct1``).
        """
        result, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures, self.V_Se_m1_supercell, self.shifted_V_Se_m1, reorient=True
        )
        assert not w

        # after re-orientation, ``shifted_V_Se_m1`` maps back to ``V_Se_m1_supercell``, so all
        # intermediate images should have ΔQ ~ 0 relative to ``V_Se_m1_supercell``:
        for key in result:
            assert get_dQ(self.V_Se_m1_supercell, result[key]) < 0.1

    def test_cc_mode_between_symmetry_equivalent_sites_reorients(self):
        """
        In CC mode (``displacements`` set) the NEB-between-symmetry-equivalent-
        sites skip heuristic should not trigger -- i.e. re-orientation proceeds
        normally and the regular "re-oriented" warning is raised instead.
        """
        _, _, w = _run_func_and_capture_stdout_warnings(
            get_path_structures,
            self.V_Se_m1_supercell,
            self.shifted_V_Se_m1,
            displacements=[0.0, 1.0],
        )
        assert any(
            "did not have a matching orientation" in str(warning.message)
            and "reorient=False" in str(warning.message)
            for warning in w
        )
        assert not any("small mass-weighted atomic displacement" in str(warning.message) for warning in w)
        assert len(w) == 1

    def test_cc_mode_different_displacements2(self):
        """
        Setting a different ``displacements2`` should use different fractional
        displacements for the two sets of interpolated structures, and ``ΔQ``
        should scale linearly with the corresponding displacement for each set.
        """
        displacements = [0.0, 0.5, 1.0]
        displacements2 = [-1.0, 0.0]
        disp_dict_1, disp_dict_2 = get_path_structures(
            self.V_Se_m1_supercell,
            self.V_Se_m2_like_m1,
            displacements=displacements,
            displacements2=displacements2,
        )
        assert list(disp_dict_1) == ["delQ_0.0", "delQ_0.5", "delQ_1.0"]
        assert list(disp_dict_2) == ["delQ_-1.0", "delQ_0.0"]

        dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
        for d in displacements:
            dQ = get_dQ(self.V_Se_m1_supercell, disp_dict_1[f"delQ_{d}"])
            assert np.isclose(dQ, abs(d) * dQ_total, atol=1e-3)
        for d in displacements2:
            dQ = get_dQ(self.V_Se_m2_like_m1, disp_dict_2[f"delQ_{d}"])
            assert np.isclose(dQ, abs(d) * dQ_total, atol=1e-3)


class TestWritePathStructures(TestGetPathStructures):
    """
    Tests for ``write_path_structures``.

    Inherits ``setupclass`` from ``TestGetPathStructures``.
    """

    def test_neb_output_structure(self):
        """
        NEB write (``n_images`` set) creates ``output_dir/00``,
        ``output_dir/01``, ... subfolders each containing a ``POSCAR`` file (no
        intermediate ``PES_x`` subfolder).

        Re-reading the POSCARs should also reproduce the same linear ``ΔQ``
        scaling across images as ``get_path_structures``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "NEB_test")
            n_images = 3
            result = write_path_structures(
                self.V_Se_m1_supercell,
                self.V_Se_m2_like_m1,
                output_dir=output_dir,
                n_images=n_images,
            )
            for folder_name in ["00", "01", "02", "03"]:
                assert os.path.isfile(os.path.join(output_dir, folder_name, "POSCAR"))

            # the function should also return the structures dict (NEB mode -> single dict with the
            # same keys as the written folder names):
            assert isinstance(result, dict)
            assert list(result) == ["00", "01", "02", "03"]

            # round-trip: written POSCARs can be reloaded with the right
            # composition, and ΔQ scales linearly with image index:
            dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
            for i in range(n_images + 1):
                written = Structure.from_file(os.path.join(output_dir, f"0{i}", "POSCAR"))
                assert written.composition == self.V_Se_m1_supercell.composition
                dQ = get_dQ(self.V_Se_m1_supercell, written)
                assert np.isclose(dQ, (i / n_images) * dQ_total, atol=1e-3)
                # returned structures should match what was written (w.r.t. ΔQ from struct1):
                assert np.isclose(get_dQ(self.V_Se_m1_supercell, result[f"0{i}"]), dQ, atol=1e-5)

    def test_cc_output_structure(self):
        """
        CC mode write (``displacements`` set) creates two ``PES_1`` and
        ``PES_2`` subfolders with ``delQ_<x>`` subfolders.

        Re-reading the POSCARs, ``ΔQ`` from each endpoint should scale
        linearly with ``|d|`` for both ``PES_1`` and ``PES_2``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "CC_test")
            displacements = [-1.0, 0.0, 0.5, 1.0]
            result = write_path_structures(
                self.V_Se_m1_supercell,
                self.V_Se_m2_like_m1,
                output_dir=output_dir,
                displacements=displacements,
            )
            for pes in ("PES_1", "PES_2"):
                for d in displacements:
                    assert os.path.isfile(os.path.join(output_dir, pes, f"delQ_{d}", "POSCAR"))

            # the function should also return the structures (CC mode -> tuple of two dicts with
            # keys matching the ``delQ_<x>`` folder names under ``PES_1`` / ``PES_2``):
            assert isinstance(result, tuple)
            assert len(result) == 2
            expected_keys = [f"delQ_{d}" for d in displacements]
            assert list(result[0]) == expected_keys
            assert list(result[1]) == expected_keys

            # re-read POSCARs and check ΔQ linear scaling w.r.t. the
            # corresponding endpoint:
            dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
            for endpoint, pes, returned in zip(
                (self.V_Se_m1_supercell, self.V_Se_m2_like_m1),
                ("PES_1", "PES_2"),
                result,
                strict=True,
            ):
                for d in displacements:
                    written = Structure.from_file(os.path.join(output_dir, pes, f"delQ_{d}", "POSCAR"))
                    dQ = get_dQ(endpoint, written)
                    assert np.isclose(dQ, abs(d) * dQ_total, atol=1e-3)
                    # returned structure for this (PES, delQ) should match what was written:
                    assert np.isclose(get_dQ(endpoint, returned[f"delQ_{d}"]), dQ, atol=1e-5)

    def test_reorient_none_default_with_mismatched_inputs_warns_and_reorients(self):
        """
        With mismatched inputs, the default ``reorient=None`` should re-orient
        ``struct2`` before writing (so the written endpoint matches the re-
        oriented ``struct2``), and raise a warning indicating re-orientation
        was required.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "NEB_reorient")
            result, _, w = _run_func_and_capture_stdout_warnings(
                write_path_structures,
                self.V_Se_m1_supercell,
                self.V_Se_m2_supercell,
                output_dir=output_dir,
                n_images=2,
            )
            assert any(
                "did not have a matching orientation" in str(warning.message)
                and "reorient=False" in str(warning.message)
                for warning in w
            )
            assert len(w) == 1
            written_end = Structure.from_file(os.path.join(output_dir, "02", "POSCAR"))
            # after re-orientation, endpoint should match ``V_Se_m2_like_m1`` (not raw ``V_Se_m2``):
            assert np.isclose(get_dQ(written_end, self.V_Se_m2_like_m1), 0.0, atol=1e-3)
            # returned dict endpoint should also match the re-oriented ``V_Se_m2_like_m1``:
            assert isinstance(result, dict)
            assert list(result) == ["00", "01", "02"]
            assert np.isclose(get_dQ(result["02"], self.V_Se_m2_like_m1), 0.0, atol=1e-3)

    def test_reorient_false_skips_reorientation(self):
        """
        With ``reorient=False`` on mismatched inputs, no re-orientation is
        performed (written endpoint matches the raw ``struct2``) and no warning
        is raised.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "NEB_no_reorient")
            result, _, w = _run_func_and_capture_stdout_warnings(
                write_path_structures,
                self.V_Se_m1_supercell,
                self.V_Se_m2_supercell,
                output_dir=output_dir,
                n_images=2,
                reorient=False,
            )
            assert not w
            written_end = Structure.from_file(os.path.join(output_dir, "02", "POSCAR"))
            # without re-orientation, endpoint should match the raw ``V_Se_m2``:
            assert np.isclose(get_dQ(written_end, self.V_Se_m2_supercell), 0.0, atol=1e-3)
            # returned dict endpoint should also match the raw (un-reoriented) ``V_Se_m2``:
            assert isinstance(result, dict)
            assert list(result) == ["00", "01", "02"]
            assert np.isclose(get_dQ(result["02"], self.V_Se_m2_supercell), 0.0, atol=1e-3)

    def test_neb_between_symmetry_equivalent_sites_warns_and_skips_reorient(self):
        """
        In NEB mode with the default ``reorient=None``, when re-orientation
        would reduce ΔQ below ``0.1`` amu^(1/2)Å (assumed to be NEB between
        symmetry-equivalent sites), re-orientation should be `skipped` and a
        warning raised about the small mass-weighted atomic displacement.
        """
        # using ``V_Se_m1`` shifted by one primitive a-axis lattice vector in the 3x3x1 supercell:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "NEB_equiv_sites")
            result, _, w = _run_func_and_capture_stdout_warnings(
                write_path_structures,
                self.V_Se_m1_supercell,
                self.shifted_V_Se_m1,
                output_dir=output_dir,
                n_images=2,
            )
            assert any(
                "small mass-weighted atomic displacement" in str(warning.message)
                and "symmetry-equivalent" in str(warning.message)
                for warning in w
            )
            assert len(w) == 1
            # verify re-orientation was skipped: the endpoint should `not` match ``V_Se_m1_supercell``
            # (which is what re-orientation would produce, mapping ``shifted_V_Se_m1`` back onto
            # ``V_Se_m1_supercell`` exactly with ΔQ ~ 0). ``Structure.interpolate``'s ``autosort_tol``
            # reorders sites but retains the lattice shift, giving a substantially larger ΔQ:
            written_end = Structure.from_file(os.path.join(output_dir, "02", "POSCAR"))
            assert get_dQ(self.V_Se_m1_supercell, written_end) > 10
            # returned dict endpoint should match the written one (i.e. also un-reoriented):
            assert isinstance(result, dict)
            assert list(result) == ["00", "01", "02"]
            assert get_dQ(self.V_Se_m1_supercell, result["02"]) > 10

    def test_default_output_dir(self):
        """
        The default output directory is ``"NEB"`` for NEB mode, and
        ``"Configuration_Coordinate"`` for CC mode.
        """
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                neb_result = write_path_structures(
                    self.V_Se_m1_supercell, self.V_Se_m2_like_m1, n_images=2
                )
                assert os.path.isdir("NEB")
                # NEB mode returns a single dict:
                assert isinstance(neb_result, dict)
                assert list(neb_result) == ["00", "01", "02"]

                cc_result = write_path_structures(
                    self.V_Se_m1_supercell,
                    self.V_Se_m2_like_m1,
                    displacements=[0.0, 1.0],
                )
                assert os.path.isdir("Configuration_Coordinate")
                # CC mode returns a tuple of two dicts:
                assert isinstance(cc_result, tuple)
                assert len(cc_result) == 2
                assert list(cc_result[0]) == ["delQ_0.0", "delQ_1.0"]
                assert list(cc_result[1]) == ["delQ_0.0", "delQ_1.0"]
        finally:
            os.chdir(original_cwd)


class TestSmartRound(unittest.TestCase):
    """
    Tests for the ``_smart_round`` helper function.
    """

    def test_single_float(self):
        """
        Values just above or below an integer/tenth/hundredth value (within
        tolerance) should be rounded down to the simplest form; values further
        from the rounded form should not.
        """
        assert _smart_round(0.5000001) == 0.5
        assert _smart_round(0.5780001) == 0.578
        assert _smart_round(-1.2) == -1.2

    def test_return_decimals(self):
        """
        ``return_decimals=True`` should also return the number of decimals used
        for rounding.
        """
        rounded, decimals = _smart_round(0.5000001, return_decimals=True)
        assert rounded == 0.5
        assert decimals == 1

        rounded, decimals = _smart_round(0.5780001, return_decimals=True)
        assert rounded == 0.578
        assert decimals == 3

    def test_list_consistent_decimals(self):
        """
        For a list of numbers, ``consistent_decimals=True`` (default) uses the
        same number of decimals for all elements -- the minimum required for
        the tolerance to be satisfied for all elements -- whereas
        ``consistent_decimals=False`` rounds each element to the minimum number
        of decimals required for that element alone.
        """
        rounded = _smart_round([0.5000001, 0.5780001, 0.12], consistent_decimals=True)
        # needs 3 decimals for 0.578, so everything is rounded to 3 decimals:
        assert rounded == [0.5, 0.578, 0.12]

        rounded, decimals = _smart_round([0.5, 0.578], return_decimals=True)
        assert decimals == 3
        assert rounded == [0.5, 0.578]

        # ``consistent_decimals=False`` rounds each element independently to
        # the minimum decimals required for that element (i.e. the returned
        # elements can have differing decimal precision):
        # With ``tol=0.5``: 1.25 rounds to 1 at 0 decimals (diff=0.25 < 0.5),
        # whereas 0.5 rounds to 0 at 0 decimals (diff=0.5, NOT strictly
        # less than 0.5) so it needs 1 decimal (-> 0.5 exactly). Thus:
        # - ``consistent_decimals=False`` -> [1.0, 0.5] (uses min-per-element)
        # - ``consistent_decimals=True`` -> [1.2, 0.5] (reuses 1 decimal for
        #   both, and ``round(1.25, 1)`` gives 1.2 via banker's rounding)
        assert _smart_round([1.25, 0.5], tol=0.5, consistent_decimals=False) == [1.0, 0.5]
        assert _smart_round([1.25, 0.5], tol=0.5, consistent_decimals=True) == [1.2, 0.5]

        # ``return_decimals=True`` further reveals the difference:
        result_inconsistent = _smart_round(
            [1.25, 0.5], tol=0.5, consistent_decimals=False, return_decimals=True
        )
        assert result_inconsistent == [(1.0, 0), (0.5, 1)]

        result_consistent = _smart_round(
            [1.25, 0.5], tol=0.5, consistent_decimals=True, return_decimals=True
        )
        assert result_consistent == ([1.2, 0.5], 1)

    def test_array_input(self):
        """
        A ``numpy`` array input with ``consistent_decimals=False`` returns an
        ``np.ndarray``.
        """
        arr = np.array([0.5000001, 0.12])
        rounded = _smart_round(arr, consistent_decimals=False)
        assert isinstance(rounded, np.ndarray)
        np.testing.assert_allclose(rounded, [0.5, 0.12])

    def test_tolerance(self):
        """
        A tighter tolerance should require more decimals.
        """
        # With default tol=1e-5, 0.500001 -> 0.5; with tighter tol=1e-8 it stays:
        assert _smart_round(0.5000001) == 0.5
        rounded, decimals = _smart_round(0.5000001, tol=1e-9, return_decimals=True)
        assert decimals == 7
        assert np.isclose(rounded, 0.5000001)


class TestGetSiteMappings(unittest.TestCase):
    """
    Tests for ``get_site_mappings``; specifically the branches which aren't
    exercised by the main parsing/analysis workflows.
    """

    def setUp(self):
        self.lattice = Lattice.cubic(8)
        # both ``struct1`` Na sites are 0.4 Å from the _same_ ``struct2`` Na site (index 0), and no
        # site pair is separated by a periodic boundary (so PBC and Cartesian distances are equal):
        self.struct1 = Structure(
            self.lattice, ["Na", "Na", "Cl"], [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.5, 0.5, 0.5]]
        )
        self.struct2 = Structure(
            self.lattice, ["Na", "Na", "Cl"], [[0.15, 0.1, 0.1], [0.55, 0.55, 0.55], [0.5, 0.5, 0.5]]
        )

    def test_linear_assignment_avoids_duplicate_matches(self):
        """
        With the default ``allow_duplicates=False``, linear assignment must
        give a 1-to-1 site mapping even when multiple ``struct1`` sites share
        the same closest ``struct2`` site.
        """
        matches = {i: (d, j) for d, i, j in get_site_mappings(self.struct1, self.struct2, threshold=1e10)}
        assert len(matches) == 3
        assert sorted(j for _d, j in matches.values()) == [0, 1, 2]  # no duplicate matches
        assert np.isclose(matches[0][0], 0.4)  # [0.1, 0.1, 0.1] -> [0.15, 0.1, 0.1]
        assert np.isclose(matches[1][0], 5.8103, atol=1e-4)  # forced onto the further Na site
        assert np.isclose(matches[2][0], 0.0)

    def test_allow_duplicates(self):
        """
        With ``allow_duplicates=True``, each ``struct1`` site independently
        takes its closest ``struct2`` site, so both ``Na`` sites match the same
        site here (each 0.4 Å away).
        """
        matches = {
            i: (d, j)
            for d, i, j in get_site_mappings(
                self.struct1, self.struct2, threshold=1e10, allow_duplicates=True
            )
        }
        # both Na sites matched to ``struct2`` site 0:
        assert matches[0] == pytest.approx((0.4, 0))
        assert matches[1] == pytest.approx((0.4, 0))
        assert np.isclose(matches[2][0], 0.0)

    def test_cartesian_matching(self):
        """
        ``frac_coords=False`` matches on Cartesian distances with no PBC, so
        should give the same mapping when no site pair is separated by a
        periodic boundary, but much larger distances when they are.
        """
        assert get_site_mappings(self.struct1, self.struct2, threshold=1e10, frac_coords=False) == (
            get_site_mappings(self.struct1, self.struct2, threshold=1e10)
        )

        # Na at [0.95, 0.1, 0.1] is 1.2 Å from [0.1, 0.1, 0.1] under PBC, but 6.8 Å without:
        s1 = Structure(self.lattice, ["Na"], [[0.1, 0.1, 0.1]])
        s2 = Structure(self.lattice, ["Na"], [[0.95, 0.1, 0.1]])
        assert np.isclose(get_site_mappings(s1, s2, threshold=1e10)[0][0], 1.2)
        assert np.isclose(get_site_mappings(s1, s2, threshold=1e10, frac_coords=False)[0][0], 6.8)

    def test_species_absent_from_struct2(self):
        """
        ``struct1`` sites of a species with no ``struct2`` sites (e.g.
        extrinsic dopants) should be returned as unmatched.
        """
        extrinsic_struct1 = self.struct1.copy()
        extrinsic_struct1.append("Mg", [0.25, 0.25, 0.25])
        mapping = get_site_mappings(extrinsic_struct1, self.struct2, threshold=1e10)
        assert (None, 3, None) in mapping
        assert len([entry for entry in mapping if entry[0] is not None]) == 3

    def test_rms_vs_summed_distance_assignment(self):
        """
        ``rms=True`` minimises the summed `squared` distances, which can select
        a different pairing to the default (summed distances).

        Note that the two can only ever differ when some site is displaced by
        more than half the nearest same-species site separation ``r``: a swap
        costs at least ``2*r`` minus the two identity distances (reverse
        triangle inequality), so it can only win if those sum to more than
        ``r``.

        Minimal discriminating case: two ``struct2`` sites ``P``, ``Q`` a
        distance ``r`` apart, with one ``struct1`` site sitting exactly on
        ``P``, and the other (``Y``) a distance ``a`` from ``P`` and ``b`` from
        ``Q``. The two candidate pairings cost ``b`` (identity) vs ``r + a``
        (swap), so with one site left exactly coincident the default
        (``rms=False``) never takes the swap (by the triangle inequality),
        whereas ``rms=True`` takes it whenever ``b**2 > r**2 + a**2``, i.e.
        whenever angle ``YPQ`` is obtuse.

        Here ``P = (0, 0, 0)``, ``Q = (2, 0, 0)``, ``Y = (-1.2, 1.6, 0)``, so
        ``r = a = 2`` Å and ``b = sqrt(12.8) = 3.578`` Å: identity costs
        ``3.578`` Å (sum) / ``12.8`` Å² (sum of squares), and the swap costs
        ``4.0`` Å / ``8.0`` Å².
        """
        lattice = Lattice.cubic(20)  # large enough that PBC images are irrelevant here
        struct2 = Structure(lattice, ["Na", "Na"], [[0, 0, 0], [0.1, 0, 0]])  # P, Q
        struct1 = Structure(lattice, ["Na", "Na"], [[0, 0, 0], [-0.06, 0.08, 0]])  # on P, and Y

        summed = get_site_mappings(struct1, struct2, threshold=1e10)
        assert [(i, j) for _d, i, j in summed] == [(0, 0), (1, 1)]  # identity pairing
        assert sum(d for d, _i, _j in summed) == pytest.approx(np.sqrt(12.8))

        rms = get_site_mappings(struct1, struct2, threshold=1e10, rms=True)
        assert [(i, j) for _d, i, j in rms] == [(0, 1), (1, 0)]  # swapped pairing
        assert sum(d for d, _i, _j in rms) == pytest.approx(4.0)
        assert all(d == pytest.approx(2.0) for d, _i, _j in rms)

        # each objective does indeed minimise its own cost:
        assert sum(d for d, *_ in summed) < sum(d for d, *_ in rms)
        assert sum(d**2 for d, *_ in rms) < sum(d**2 for d, *_ in summed)


class TestSummedDistAndFindMissingIdx(unittest.TestCase):
    """
    Tests for ``summed_dist`` and ``find_missing_idx``, which build on
    ``get_site_mappings`` / ``_get_site_mapping_from_coords_and_indices``.
    """

    def setUp(self):
        self.lattice = Lattice.cubic(6)
        self.struct_a = Structure(self.lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.struct_b = Structure(self.lattice, ["Na", "Cl"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])

    def test_summed_dist(self):
        """
        ``summed_dist`` should return a native ``float`` (not a ``numpy``
        scalar, which would otherwise propagate to output metadata etc).
        """
        dist = summed_dist(self.struct_a, self.struct_b)
        assert type(dist) is float  # not ``isinstance``; ``np.float64`` subclasses ``float``
        assert dist == pytest.approx(0.06)
        assert summed_dist(self.struct_a, self.struct_a) == 0.0

    def test_summed_dist_unmatched_sites(self):
        """
        Structures with differing compositions have unmatched sites, which give
        an infinite summed distance (rather than being silently dropped, which
        would make such structures rank as `closer` matches for callers).
        """
        extrinsic = self.struct_a.copy()
        extrinsic.append("Mg", [0.25, 0.25, 0.25])
        assert summed_dist(extrinsic, self.struct_b) == float("inf")
        assert summed_dist(extrinsic, self.struct_b, ignored_species=["Mg"]) == pytest.approx(0.06)

    def test_find_missing_idx(self):
        """
        ``find_missing_idx`` should return the index of the missing/outlier
        coordinate in the larger of the two sets, for either input ordering.
        """
        rng = np.random.default_rng(42)
        for _ in range(100):
            n_coords = int(rng.integers(2, 12))
            full = rng.random((n_coords, 3))
            dropped_idx = int(rng.integers(0, n_coords))
            # delete one site, and jitter the rest (well within half the site separation):
            partial = np.delete(full, dropped_idx, axis=0) + 0.005 * rng.standard_normal((n_coords - 1, 3))

            for coords_1, coords_2 in ((partial, full), (full, partial)):  # both orderings
                assert find_missing_idx(coords_1, coords_2, self.lattice) == dropped_idx


class TestNearestNeighbourSiteMapping(unittest.TestCase):
    """
    Tests for ``_nearest_neighbour_site_mapping``; the neighbour list search
    used by ``_get_site_mapping_from_coords_and_indices`` when possible, which
    must give exactly the linear assignment solution whenever it is used.
    """

    def setUp(self):
        # 6x6x6 grids (216 sites), in orthogonal and non-orthogonal lattices;
        # site separation ``r`` is 2.0 Å (cubic) / 1.91 Å (sheared):
        self.coords = np.stack(np.meshgrid(*[np.arange(6) / 6] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        self.lattices = [Lattice.cubic(12), Lattice([[12, 0, 0], [4.5, 11, 0], [3.5, 3, 10.5]])]
        self.rng = np.random.default_rng(42)

    @staticmethod
    def _linear_assignment_mapping(subset_coords, superset_coords, lattice):
        """
        Reference mapping, from the full distance matrix and linear assignment.
        """
        dists = np.sqrt(pbc_shortest_vectors(lattice, subset_coords, superset_coords, return_d2=True)[1])
        site_matches, _ = get_linear_assignment_solution(dists)
        return dists[np.arange(len(site_matches)), site_matches], site_matches

    def test_min_separation_and_caching(self):
        """
        ``_min_separation`` should give the true minimum separation (under
        PBC), and be cached on repeat calls with the same coordinates and
        lattice.
        """
        for lattice in self.lattices:
            _cached_min_separation.cache_clear()
            separation = _min_separation(self.coords, lattice)
            all_dists = lattice.get_all_distances(self.coords, self.coords)
            assert separation == pytest.approx(all_dists[all_dists > 1e-8].min())
            assert type(separation) is float  # not a ``numpy`` scalar

            # check caching working:
            hits_before = _cached_min_separation.cache_info().hits
            assert _min_separation(self.coords.copy(), lattice) == separation  # equal, not identical
            assert _cached_min_separation.cache_info().hits == hits_before + 1

    def test_matches_linear_assignment(self):
        """
        When each site's nearest neighbour is distinct, the neighbour list
        search should be used, and give exactly the same result as the linear
        assignment.
        """
        # 0.01 fractional noise = 0.12 Å per component: displacements ~0.2 Å (max ~0.5 Å << r ~ 2 Å)
        for lattice in self.lattices:
            displaced = self.coords + 0.01 * self.rng.standard_normal(self.coords.shape)
            for subset in (displaced, np.delete(displaced, 5, axis=0)):  # equal-size, and vacancy-like
                result = _nearest_neighbour_site_mapping(subset, self.coords, lattice)
                assert result is not None
                ref_dists, ref_matches = self._linear_assignment_mapping(subset, self.coords, lattice)
                np.testing.assert_array_equal(result[1], ref_matches)
                np.testing.assert_allclose(result[0], ref_dists)

    def test_explicit_search_radius(self):
        """
        An explicit search radius ``r`` only affects the acceptance rate;
        accepted mappings are identical at any radius.
        """
        for lattice in self.lattices:
            displaced = self.coords + 0.01 * self.rng.standard_normal(self.coords.shape)
            default = _nearest_neighbour_site_mapping(displaced, self.coords, lattice)
            widened = _nearest_neighbour_site_mapping(displaced, self.coords, lattice, r=10.0)
            assert default is not None
            assert widened is not None
            np.testing.assert_array_equal(widened[1], default[1])
            np.testing.assert_allclose(widened[0], default[0])

            # while a tiny radius finds no matches (displacements are ~0.2 Å here), so is rejected:
            assert _nearest_neighbour_site_mapping(displaced, self.coords, lattice, r=0.01) is None

    def test_competing_sites_rejected(self):
        """
        The rejection mode seen in practice: one site displaced past the
        midpoint toward a neighbouring site, so that two sites share the same
        nearest neighbour and the (greedy) match is no longer a valid
        assignment -- while every other site is barely displaced.
        """
        for lattice in self.lattices:
            displaced = self.coords + 0.002 * self.rng.standard_normal(self.coords.shape)
            # site 1 moved just past halfway towards site 0, so both are nearest to site 0:
            displaced[1] = self.coords[0] + 0.45 * (self.coords[1] - self.coords[0])
            assert _nearest_neighbour_site_mapping(displaced, self.coords, lattice) is None

            # ...and the fallback still gives the exact linear assignment solution:
            ref_dists, _ref_matches = self._linear_assignment_mapping(displaced, self.coords, lattice)
            mapping = _get_site_mapping_from_coords_and_indices(displaced, self.coords, lattice=lattice)
            assert sum(d for d, *_ in mapping) == pytest.approx(ref_dists.sum())

    def test_periodic_image_duplicates(self):
        """
        In slab-like cells the shortest lattice repeat can be the minimum site
        separation, so the search returns the same site pair via several
        periodic images.

        The nearest neighbour per site (and hence the mapping) must still be
        correct.
        """
        # one site per cell along c, so ``r`` is set by the c-axis periodic image:
        xy = np.stack(np.meshgrid(*[np.arange(12) / 12] * 2, indexing="ij"), axis=-1).reshape(-1, 2)
        coords = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
        lattice = Lattice([[60.0, 0, 0], [9.0, 60.0, 0], [1.1, 0.9, 3.0]])
        cart = lattice.get_cartesian_coords(coords)
        displaced = lattice.get_fractional_coords(cart + 0.1 * self.rng.standard_normal(cart.shape))

        result = _nearest_neighbour_site_mapping(displaced, coords, lattice)
        assert result is not None
        ref_dists, ref_matches = self._linear_assignment_mapping(displaced, coords, lattice)
        np.testing.assert_array_equal(result[1], ref_matches)
        np.testing.assert_allclose(result[0], ref_dists)

    def test_large_displacements_rejected(self):
        """
        Once displacements are large enough that two sites share the same
        nearest neighbour (or that some site has none within the search
        radius), the nearest neighbour match is no longer the linear assignment
        solution, so the search must decline (returning ``None``, to fall back
        to the full distance matrix).
        """
        for lattice in self.lattices:
            displaced = self.coords + 0.15 * self.rng.standard_normal(self.coords.shape)
            assert _nearest_neighbour_site_mapping(displaced, self.coords, lattice) is None

            # ...but the full mapping still works, falling back to the linear assignment. Compared on
            # total cost rather than pairing, as the optimal pairing need not be unique here:
            ref_dists, _ref_matches = self._linear_assignment_mapping(displaced, self.coords, lattice)
            mapping = _get_site_mapping_from_coords_and_indices(displaced, self.coords, lattice=lattice)
            assert len(mapping) == len(self.coords)
            assert sum(d for d, *_ in mapping) == pytest.approx(ref_dists.sum())

    def test_get_site_mappings_with_vacancy(self):
        """
        End-to-end check through ``get_site_mappings``, including the unmatched
        (vacant) site, which the search itself does not return.
        """
        for lattice in self.lattices:
            bulk = Structure(lattice, ["Na"] * len(self.coords), self.coords)
            # corresponding noisy vacancy structure:
            displaced = self.coords + 0.02 * self.rng.standard_normal(self.coords.shape)
            defect = Structure(lattice, ["Na"] * (len(self.coords) - 1), np.delete(displaced, 5, axis=0))

            _dists, matches = self._linear_assignment_mapping(defect.frac_coords, self.coords, lattice)
            assert [(i, j) for _d, i, j in get_site_mappings(defect, bulk, threshold=1e10)] == list(
                enumerate(matches)
            )

            # ``get_site_mappings`` drops unmatched sites, so check these are retained one level down:
            mapping = _get_site_mapping_from_coords_and_indices(
                defect.frac_coords, self.coords, lattice=lattice
            )
            assert [j for dist, i, j in mapping if dist is None and i is None] == [5]  # the vacant site
