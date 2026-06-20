"""
Tests for the ``doped.utils.configurations`` module.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
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
from doped.utils.parsing import get_site_mapping_indices
from doped.utils.supercells import min_dist
from doped.utils.symmetry import get_clean_structure, point_symmetry_from_structure


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
        from ``get_site_mapping_indices`` with a mapped weighted-distance sum,
        **if** no re-orientation is required (only re-ordering).
        """
        struct1 = self.V_Se_m1_supercell
        struct2 = get_clean_structure(self.V_Se_m2_supercell)  # re-order to break ordering match

        raw_dQ_no_reorient = get_dQ(struct1, struct2)
        dQ_reorient = get_dQ(struct1, struct2, reorient=True)
        dQ_from_mapping = np.sqrt(
            sum(
                (struct1[i].distance(struct2[j]) ** 2) * struct1[i].specie.atomic_mass
                for _, i, j in get_site_mapping_indices(struct1, struct2)
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
        # ΔQ(s1/s2_like_s1) = 8.63 amu^(1/2)Å (\u212b = angstrom sign)
        assert "\u0394Q(s1/s2) = 9.91 amu^(1/2)\u212b" in output
        assert "\u0394Q(s2_like_s1/s2) = 4.88 amu^(1/2)\u212b" in output
        assert "\u0394Q(s1/s2_like_s1) = 8.63 amu^(1/2)\u212b" in output
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
