"""
Tests for the ``doped.utils.configurations`` module.
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from test_utils import EXAMPLE_DIR, _print_warning_info, _run_func_and_capture_stdout_warnings, data_dir

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
from doped.utils.supercells import min_dist
from doped.utils.symmetry import point_symmetry_from_structure


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

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            orient_s2_like_s1(self.V_Se_m1_supercell, stretched_V_Se_m2)

        _print_warning_info(w)
        assert any("(symmetry-)inequivalent" in str(warning.message) for warning in w)

    def test_mismatched_compositions_raises(self):
        """
        Trying to orient structures with different compositions / too different
        lattices should raise a ``RuntimeError``.
        """
        # create a structure with clearly incompatible composition:
        other = Structure(lattice=np.eye(3) * 5, species=["Si"], coords=[[0, 0, 0]])
        with pytest.raises(RuntimeError, match="get_transformation"):
            orient_s2_like_s1(other, self.V_Se_m1_supercell)


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
        assert not np.allclose(
            rotated_V_Se_m2.lattice.matrix, self.V_Se_m1_supercell.lattice.matrix
        ), "Rotated V_Se^-2 lattice matrix should now differ from V_Se^-1"

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


class TestWritePathStructures(ConfigurationsTestCase):
    """
    Tests for ``write_path_structures``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.V_Se_m2_like_m1 = orient_s2_like_s1(cls.V_Se_m1_supercell, cls.V_Se_m2_supercell)

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
            write_path_structures(
                self.V_Se_m1_supercell,
                self.V_Se_m2_like_m1,
                output_dir=output_dir,
                n_images=n_images,
            )
            for folder_name in ["00", "01", "02", "03"]:
                assert os.path.isfile(os.path.join(output_dir, folder_name, "POSCAR"))

            # round-trip: written POSCARs can be reloaded with the right
            # composition, and ΔQ scales linearly with image index:
            dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
            for i in range(n_images + 1):
                written = Structure.from_file(os.path.join(output_dir, f"0{i}", "POSCAR"))
                assert written.composition == self.V_Se_m1_supercell.composition
                dQ = get_dQ(self.V_Se_m1_supercell, written)
                assert np.isclose(dQ, (i / n_images) * dQ_total, atol=1e-3)

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
            write_path_structures(
                self.V_Se_m1_supercell,
                self.V_Se_m2_like_m1,
                output_dir=output_dir,
                displacements=displacements,
            )
            for pes in ("PES_1", "PES_2"):
                for d in displacements:
                    assert os.path.isfile(os.path.join(output_dir, pes, f"delQ_{d}", "POSCAR"))

            # re-read POSCARs and check ΔQ linear scaling w.r.t. the
            # corresponding endpoint:
            dQ_total = get_dQ(self.V_Se_m1_supercell, self.V_Se_m2_like_m1)
            for endpoint, pes in zip(
                (self.V_Se_m1_supercell, self.V_Se_m2_like_m1), ("PES_1", "PES_2"), strict=True
            ):
                for d in displacements:
                    written = Structure.from_file(os.path.join(output_dir, pes, f"delQ_{d}", "POSCAR"))
                    dQ = get_dQ(endpoint, written)
                    assert np.isclose(dQ, abs(d) * dQ_total, atol=1e-3)

    def test_default_output_dir(self):
        """
        The default output directory is ``"NEB"`` for NEB mode, and
        ``"Configuration_Coordinate"`` for CC mode.
        """
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                write_path_structures(self.V_Se_m1_supercell, self.V_Se_m2_like_m1, n_images=2)
                assert os.path.isdir("NEB")

                write_path_structures(
                    self.V_Se_m1_supercell,
                    self.V_Se_m2_like_m1,
                    displacements=[0.0, 1.0],
                )
                assert os.path.isdir("Configuration_Coordinate")
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
