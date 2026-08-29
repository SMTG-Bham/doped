"""
Tests for the pure beam-zoom global search core in ``doped.utils._optimise``.

``doped.utils._optimise`` is the engine behind ``FermiSolver.optimise``), using
fast synthetic landscapes -- no DFT data or Fermi solving required.

Integration tests of ``FermiSolver.optimise`` itself (which wraps this core)
are in ``test_fermisolver.py``.
"""

import itertools
import warnings

import numpy as np
import pytest

from doped.chemical_potentials import _lattice_in_hull
from doped.utils._optimise import (
    _beam_zoom_search,
    _default_grid_resolution,
    _landscape_smoothness_scale,
    _PolytopeInterp,
    _roughness,
    _select_seeds,
    _typical_spacing,
    kB,
)

# Synthetic two-basin landscape on a triangular ("ternary") stability polytope: a broad basin (amplitude 1,
# sigma = 0.35 eV) plus a narrow global basin (amplitude 100, sigma = 0.03 eV), which provably defeats the
# legacy greedy single-basin contraction search at its default (30-point) grid density. The third column
# plays the role of the dependent chemical potential (= -(x+y) at the vertices, affine over hull):
TRIANGLE = np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [0.0, -2.0, 2.0]])
BROAD, NARROW = np.array([-1.5, -0.4]), np.array([-0.3, -1.55])


def two_basin(points, amp=100.0, centre=NARROW, sigma=0.03):
    """
    Broad (amplitude-1) + narrow (amplitude-``amp``) Gaussian basins.
    """
    d_broad = np.sum((points[:, :2] - BROAD) ** 2, axis=-1)
    d_narrow = np.sum((points[:, :2] - centre) ** 2, axis=-1)
    return np.exp(-d_broad / (2 * 0.35**2)) + amp * np.exp(-d_narrow / (2 * sigma**2))


def make_grid(vertices, n_points=30, resolution=None):
    """
    Barycentric grid over the triangle (as ``optimise`` grid generators).
    """
    return _lattice_in_hull(vertices[:, :2], vertices[:, 2], n_points=n_points, resolution=resolution)


def recording_grid(resolutions):
    """
    ``make_grid``, appending each dense-pass ``resolution`` to ``resolutions``.
    """

    def wrapped(vertices, n_points=30, resolution=None):
        if resolution is not None and np.isfinite(resolution):  # dense first pass / densification only
            resolutions.append(resolution)
        return make_grid(vertices, n_points=n_points, resolution=resolution)

    return wrapped


LEGACY_CONFIG = {  # reproduces the pre-doped-v4 greedy single-branch contraction search
    "beam_width": 1,
    "initial_grid_resolution": np.inf,
    "n_audit_points": 0,
    "polish": False,
}


class TestTwoBasinRegression:
    """
    The canonical multi-modal edge case (§ ``optimise`` docstring): a narrow
    global basin easily missed by coarse greedy contraction.
    """

    def test_new_defaults_find_global_basin(self):
        result = _beam_zoom_search(
            two_basin, TRIANGLE, make_grid=make_grid, initial_grid_resolution=kB * 900
        )
        assert result.value >= 99.5
        assert np.max(np.abs(result.point[:2] - NARROW)) < 0.02

    def test_polish_refines_stopping_rule_floor(self):
        # the relative-change stopping rule alone leaves a small value error; the final Nelder-Mead polish
        # refines the returned optimum to solver precision (audit off to isolate the effect):
        unpolished = _beam_zoom_search(
            two_basin,
            TRIANGLE,
            make_grid=make_grid,
            initial_grid_resolution=kB * 900,
            n_audit_points=0,
            polish=False,
        )
        polished = _beam_zoom_search(
            two_basin,
            TRIANGLE,
            make_grid=make_grid,
            initial_grid_resolution=kB * 900,
            n_audit_points=0,
        )
        assert 96 <= unpolished.value < polished.value
        assert abs(polished.value - 100) < 1e-3

    def test_legacy_config_converges_to_wrong_basin(self):
        # the legacy configuration provably fails on this landscape (finds the broad amplitude-1 basin,
        # missing the narrow amplitude-100 global basin), demonstrating this test suite has teeth:
        result = _beam_zoom_search(two_basin, TRIANGLE, make_grid=make_grid, **LEGACY_CONFIG)
        assert result.value < 2
        assert np.max(np.abs(result.point[:2] - BROAD)) < 0.1

    def test_plateau_early_exit_fixed(self):
        # a ~0.15 eV first-pass spacing seeds the right basin but the legacy single-hit relative-change
        # rule plateau-stopped ~10% short (89.9 vs 100); the two-consecutive rule alone closes this to
        # <0.2%, with a single beam and the audit, roughness-densification and polish layers all off,
        # so that nothing else can mask a regression of the stopping rule (which gives 98.98 here):
        result = _beam_zoom_search(
            two_basin,
            TRIANGLE,
            make_grid=make_grid,
            beam_width=1,
            initial_grid_resolution=0.15,
            n_audit_points=0,
            roughness_threshold=0,
            polish=False,
        )
        assert abs(result.value - 100) < 0.5
        assert np.max(np.abs(result.point[:2] - NARROW)) < 0.02

    def test_sample_suppressed_global_basin(self):
        # place the narrow global basin (amplitude 2, sigma = 0.02 eV) mid-cell in the first-pass lattice,
        # so its best sample (~0.3) sits well below the broad basin's peak samples (~1.0) -- the case that
        # defeats dense-argmax + local polish (and DE-style refinement), proving the local-maxima beam
        # seeding layer is required for robustness/completeness:
        spacing = 2.0 / 37  # first-pass lattice leg spacing at kB*900 resolution (r = 37 subdivisions)
        centre = np.array([-2 * 6 / 37 - spacing / 2, -2 * 28 / 37 - spacing / 2])  # mid-cell
        suppressed = lambda pts: two_basin(pts, amp=2.0, centre=centre, sigma=0.02)  # noqa: E731

        first_pass = make_grid(TRIANGLE, resolution=kB * 900)
        near_basin = np.linalg.norm(first_pass[:, :2] - centre, axis=1) < 3 * 0.02
        assert suppressed(first_pass)[near_basin].max() < 0.5  # precondition: basin sample-suppressed

        argmax_and_polish = _beam_zoom_search(  # (a) single branch from the dense argmax + polish:
            suppressed,
            TRIANGLE,
            make_grid=make_grid,
            beam_width=1,
            initial_grid_resolution=kB * 900,
            n_audit_points=0,
            roughness_threshold=0,
        )
        assert abs(argmax_and_polish.value - 1.0) < 0.1  # wrong (broad) basin

        beam = _beam_zoom_search(  # (b) full beam pipeline (audit off; the beam layer must catch it):
            suppressed,
            TRIANGLE,
            make_grid=make_grid,
            initial_grid_resolution=kB * 900,
            n_audit_points=0,
        )
        assert abs(beam.value - 2.0) < 0.02


class TestSearchMechanics:
    """
    Unit tests of the individual search-core mechanisms.
    """

    def test_seed_selection_local_maxima_and_separation(self):
        points = make_grid(TRIANGLE, resolution=0.1)[:, :2]
        spacing = _typical_spacing(points)
        values = np.zeros(len(points))
        top = int(np.argmin(np.linalg.norm(points - [-0.4, -0.4], axis=1)))
        near_duplicate = int(  # a slightly-lower peak 2 lattice spacings from ``top`` (a distinct...
            np.argmin(np.linalg.norm(points - (points[top] - [2 * spacing, 0]), axis=1))
        )  # ...discrete local maximum, but within the 3*spacing Chebyshev dedup separation)
        distant = int(np.argmin(np.linalg.norm(points - [-1.5, -0.1], axis=1)))
        values[[top, near_duplicate, distant]] = [10, 9.9, 5]
        # a monotonically-decreasing ridge descending from ``top`` in +y: its points are `not` discrete
        # local maxima (each has a higher ridge neighbour), but their values exceed ``distant``'s and the
        # outer ones lie beyond the 3*spacing dedup separation from ``top`` -- so naive top-value seeding
        # (even with dedup) would seed the ridge instead of the distant basin:
        ridge = [
            int(np.argmin(np.linalg.norm(points - (points[top] + [0, i * spacing]), axis=1)))
            for i in range(1, 6)
        ]
        values[ridge] = [9.8, 9.6, 9.4, 9.2, 9.0]

        seeds = _select_seeds(points, values, spacing, beam_width=3)
        assert len(seeds) <= 3
        assert seeds[0] == top  # ranked by value
        assert distant in seeds
        assert near_duplicate not in seeds  # deduplicated against ``top``
        assert not set(ridge) & set(seeds)  # ridge points are not local maxima -> never seeded
        for i, j in itertools.combinations(seeds, 2):
            assert np.max(np.abs(points[i] - points[j])) >= 3 * spacing

    def test_typical_spacing_ignores_duplicates_and_rounded_vertex_twins(self):
        """
        Grids prepend exact polytope vertices alongside their 6-dp-rounded
        lattice twins (~1e-7 apart), and densified grids can nest the coarse
        grid exactly (e.g. halved-resolution 1D lines, >50% duplicates); either
        would collapse the median NN distance to ~1e-7/zero -- giving
        degenerate seed pre-contraction boxes, downstream qhull failures and
        broken seed selection -- so ``_typical_spacing`` must merge both
        (rounding then ``np.unique``), and report a degenerate single-point set
        as ``inf``.
        """
        exact = np.array([[0.0000004, 0.0], [1.0000004, 0.0], [0.0, 1.0000004], [0.5000004, 0.5]])
        points = np.vstack([exact, np.round(exact, 6)])  # every point's NN is its twin, ~4e-7 off
        assert _typical_spacing(points) == pytest.approx(np.sqrt(2) / 2, rel=1e-3)

        line = np.linspace(0, 1, 11)[:, None]
        assert np.isclose(_typical_spacing(np.vstack([line, line, line])), 0.1)  # 2/3 exact duplicates
        assert _typical_spacing(np.zeros((5, 2))) == np.inf  # degenerate single-point case

    def test_audit_reseeds_spike_invisible_to_first_pass(self):
        # a sigma = 0.02 eV spike is invisible to a deliberately coarse (0.3 eV) first pass with the
        # roughness check disabled; the random audit (seeded rng; deterministic) must find it and re-seed a
        # search branch which refines it to the global optimum:
        spike = lambda pts: two_basin(pts, centre=np.array([-0.913, -0.617]), sigma=0.02)  # noqa: E731
        result = _beam_zoom_search(
            spike,
            TRIANGLE,
            make_grid=make_grid,
            initial_grid_resolution=0.3,
            roughness_threshold=0,
            n_audit_points=3000,
        )
        assert result.value > 99
        assert len(result.branches) >= 2  # audit re-seed ran a new branch (not just a lucky sample)

    def test_3d_simplex(self):
        tetrahedron = np.hstack(  # 3 independent dimensions + fake dependent column
            [np.array([[0, 0, 0], [-2, 0, 0], [0, -2, 0], [0, 0, -2]], dtype=float), np.zeros((4, 1))]
        )
        centre = np.array([-0.3, -1.0, -0.4])

        def landscape(pts):
            broad = np.exp(-np.sum((pts[:, :3] + 1.0) ** 2, axis=-1) / (2 * 0.35**2))
            return broad + 100 * np.exp(-np.sum((pts[:, :3] - centre) ** 2, axis=-1) / (2 * 0.05**2))

        def make_grid_3d(vertices, n_points=30, resolution=None):
            return _lattice_in_hull(
                vertices[:, :3], vertices[:, 3], n_points=n_points, resolution=resolution
            )

        result = _beam_zoom_search(
            landscape, tetrahedron, make_grid=make_grid_3d, initial_grid_resolution=0.1
        )
        assert result.value > 99
        assert np.max(np.abs(result.point[:3] - centre)) < 0.02

    def test_nested_densification_grids_terminate(self):
        # regression test: a halved-resolution 1D line grid can nest the coarse grid `exactly`, giving
        # >50% duplicate points after the roughness densification merge -- with a naive median
        # nearest-neighbour spacing (0.0), the branch pre-contraction loop would then iterate to a
        # floating-point fixed point 1 ulp above the seed and hang forever:
        segment = np.array([[0.0, 5.0], [-2.0, 7.0]])

        def line_grid(verts, n_points=30, resolution=None):
            start, end = np.asarray(verts, dtype=float)
            if resolution is not None and np.isfinite(resolution):
                n_points = int(np.ceil(np.max(np.abs(end - start)) / resolution)) + 1
            points = start + np.linspace(0, 1, n_points)[:, None] * (end - start)
            points[0], points[-1] = start, end
            return points

        quasi_step = lambda pts: 10 ** (6 * np.tanh((pts[:, 0] + 0.7) / 0.001))  # noqa: E731
        result = _beam_zoom_search(  # rough landscape -> densification fires; kB*250 nests exactly
            quasi_step, segment, make_grid=line_grid, initial_grid_resolution=kB * 250, n_audit_points=0
        )
        assert result.value == pytest.approx(10**6, rel=0.01)

    def test_max_iterations_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            _beam_zoom_search(
                two_basin,
                TRIANGLE,
                make_grid=make_grid,
                max_iterations=1,
                n_audit_points=0,
                polish=False,
                roughness_threshold=0,
                initial_grid_resolution=0.15,
            )
        assert any("max_iterations" in str(warning.message) for warning in caught)


class TestRoughnessCheck:
    """
    Tests for the empirical under-resolution (roughness) detector and its one-
    shot densification response.
    """

    # log-target ∝ tanh((x - x0)/w) with w << grid spacing (a compensation crossover-style quasi-step):
    quasi_step = staticmethod(lambda pts: 10 ** (6 * np.tanh((pts[:, 0] + 0.7) / 0.005)))

    def test_roughness_metric(self):
        grid = make_grid(TRIANGLE, resolution=0.1)
        spacing = _typical_spacing(grid[:, :2])
        assert _roughness(grid[:, :2], self.quasi_step(grid), spacing) > 2
        smooth_values = two_basin(grid, amp=0)  # broad basin only; smooth at this spacing
        assert _roughness(grid[:, :2], smooth_values, spacing) < 2

        # near-duplicate rows (exact polytope vertices + their rounded lattice copies, sharing one cached
        # solve value) must not blind the check -- without collapsing them, each twin is the other's
        # nearest neighbour with |Δv| = 0, hiding every feature (roughness 0.0 here):
        line = np.linspace(0, 1, 21)[:, None]
        kink_values = np.where(line[:, 0] < 0.024, 1e6, 1.0)  # sharp feature at the x = 0 corner
        twinned = np.vstack([line, line + 5e-7])  # rounded lattice copy of every point, 5e-7 away
        twinned_values = np.concatenate([kink_values, kink_values])  # twins share one cached value
        assert _roughness(twinned, twinned_values, 0.05) == pytest.approx(
            _roughness(line, kink_values, 0.05), rel=1e-3
        )
        assert _roughness(twinned, twinned_values, 0.05) > 2

        # non-finite values (failed solves are mapped to -inf) must be masked without warnings:
        with_failures = smooth_values.copy()
        with_failures[10:20] = -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any numpy RuntimeWarning (e.g. inf - inf) -> failure
            assert np.isfinite(_roughness(grid[:, :2], with_failures, spacing))

    def test_quasi_step_triggers_single_densification_and_warning(self):
        resolutions = []
        with warnings.catch_warnings(record=True) as caught:
            _beam_zoom_search(
                self.quasi_step,
                TRIANGLE,
                make_grid=recording_grid(resolutions),
                initial_grid_resolution=0.1,
                n_audit_points=0,
                polish=False,
            )
        assert resolutions == [0.1, 0.05]  # exactly one densification (at halved resolution)
        assert any("under-resolved" in str(warning.message) for warning in caught)

    def test_smooth_landscape_no_densification(self):
        # the roughness check must not fire (and densify, doubling the first-pass cost) when the landscape
        # is smooth at the first-pass grid spacing:
        resolutions = []
        with warnings.catch_warnings(record=True) as caught:
            _beam_zoom_search(
                lambda pts: two_basin(pts, amp=0),
                TRIANGLE,
                make_grid=recording_grid(resolutions),
                initial_grid_resolution=kB * 900,
                n_audit_points=0,
                polish=False,
            )
        assert resolutions == [kB * 900]  # no densification
        assert not any("under-resolved" in str(warning.message) for warning in caught)


class TestFirstPassResolutionDefaults:
    """
    Tests for ``_landscape_smoothness_scale`` (per-protocol dispatch) and
    ``_default_grid_resolution`` (opportunistic refinement + budget).
    """

    def test_smoothness_scale_dispatch(self):
        assert np.isclose(_landscape_smoothness_scale(temperature=500), kB * 500)
        assert np.isclose(  # frozen totals; c(μ) ~ exp(-dH/kT_anneal):
            _landscape_smoothness_scale(annealing_temperature=900, quenched_temperature=300), kB * 900
        )
        assert np.isclose(  # free defects re-equilibrate against the reservoir at T_quench:
            _landscape_smoothness_scale(
                annealing_temperature=900, quenched_temperature=300, free_defects=["v_"]
            ),
            kB * 300,
        )
        assert np.isclose(  # conservative kB * min(temperatures) fallback (T_quench > T_anneal here):
            _landscape_smoothness_scale(
                annealing_temperature=300, quenched_temperature=600, free_defects=["v_"]
            ),
            kB * 300,
        )

    def test_opportunistic_refinement_2d_reaches_quench_floor(self):
        # 2D anneal(900 K) + quench(300 K): refinement to the kB*300 floor is cheap (well within the 25%
        # soft budget of ``max_initial_points``), insuring against dispatch misclassification:
        resolution = _default_grid_resolution(
            make_grid, TRIANGLE, annealing_temperature=900, quenched_temperature=300
        )
        assert np.isclose(resolution, kB * 300)

    def test_4d_stays_at_dispatch_scale_budget_bound(self):
        simplex_4d = np.hstack([np.vstack([np.zeros(4), -2 * np.eye(4)]), np.zeros((5, 1))])

        def make_grid_4d(vertices, n_points=30, resolution=None):
            return _lattice_in_hull(
                vertices[:, :4], vertices[:, 4], n_points=n_points, resolution=resolution
            )

        resolution = _default_grid_resolution(
            make_grid_4d, simplex_4d, annealing_temperature=900, quenched_temperature=300
        )
        assert np.isclose(resolution, kB * 900)  # budget-bound; no refinement below dispatch scale

    def test_equilibrium_no_refinement_below_kT(self):
        # for full equilibrium the dispatch scale equals the floor (kB*T); no refinement below it:
        resolution = _default_grid_resolution(make_grid, TRIANGLE, temperature=500)
        assert np.isclose(resolution, kB * 500)

    def test_max_points_clamps_before_materialising(self):
        # ``max_points`` must clamp the implied grid size (with a single clear warning) `before` the grid
        # is materialised -- the mechanism preventing memory blow-ups for fine resolutions in
        # high-dimensional chemical spaces (``max_initial_points`` in ``optimise``):
        with warnings.catch_warnings(record=True) as caught:
            clamped = _lattice_in_hull(TRIANGLE[:, :2], TRIANGLE[:, 2], resolution=1e-4, max_points=500)
        assert len(clamped) <= 2 * 500  # S * C(r+d, d) bound; a little slack for shared simplex faces
        assert sum("max_points" in str(warning.message) for warning in caught) == 1
        achieved_spacing = _typical_spacing(clamped[:, :2])
        assert achieved_spacing > 1e-4  # coarser than requested (clamped)

        with warnings.catch_warnings(record=True) as caught:
            unclamped = _lattice_in_hull(TRIANGLE[:, :2], TRIANGLE[:, 2], resolution=0.1, max_points=10**6)
        assert not caught  # fits within the cap; no warning, resolution honoured
        assert len(unclamped) == len(_lattice_in_hull(TRIANGLE[:, :2], TRIANGLE[:, 2], resolution=0.1))


class TestPolytopeInterp:
    """
    Tests for the piecewise-linear polytope interpolation/sampling helper.
    """

    def test_lift_interpolates_dependent_column(self):
        interp = _PolytopeInterp(TRIANGLE)  # dependent column = -(x+y) over this polytope
        lifted = interp.lift(np.array([[-0.5, -0.5], [-1.0, -0.25]]))
        np.testing.assert_allclose(lifted[:, :2], [[-0.5, -0.5], [-1.0, -0.25]], atol=1e-8)
        np.testing.assert_allclose(lifted[:, 2], [1.0, 1.25], atol=1e-6)

    def test_lift_outside_hull_is_nan(self):
        interp = _PolytopeInterp(TRIANGLE)
        assert np.isnan(interp.lift(np.array([[0.5, 0.5]]))).all()
        assert np.isnan(interp.lift(np.array([[-1.5, -1.5]]))).all()  # beyond the x+y = -2 edge

    def test_sample_uniform_within_hull(self):
        interp = _PolytopeInterp(TRIANGLE)
        samples = interp.sample(500, np.random.default_rng(0))
        assert samples.shape == (500, 3)
        assert (samples[:, 0] <= 1e-9).all()
        assert (samples[:, 1] <= 1e-9).all()
        assert (samples[:, 0] + samples[:, 1] >= -2 - 1e-9).all()
        np.testing.assert_allclose(samples[:, 2], -(samples[:, 0] + samples[:, 1]), atol=1e-6)

    def test_1d_segment(self):
        segment = np.array([[0.0, 5.0], [-2.0, 7.0]])  # binary system line; dependent = 5 - x
        interp = _PolytopeInterp(segment)
        np.testing.assert_allclose(interp.lift(np.array([[-1.0]])), [[-1.0, 6.0]], atol=1e-8)
        assert np.isnan(interp.lift(np.array([[0.5]]))).all()
        samples = interp.sample(100, np.random.default_rng(0))
        assert ((samples[:, 0] <= 1e-9) & (samples[:, 0] >= -2 - 1e-9)).all()
        np.testing.assert_allclose(samples[:, 1], 5 - samples[:, 0], atol=1e-8)

    def test_constant_columns_excluded_from_geometry(self):
        # e.g. a ``fixed_elements``-constrained sub-polytope: the fixed column is constant and must be
        # excluded from geometry (else ``scipy`` Delaunay fails on the degenerate dimension), but still
        # carried through interpolation/sampling:
        rows = np.array([[0.0, -1.32, 0.0, 0.0], [-2.0, -1.32, 0.0, 2.0], [0.0, -1.32, -2.0, 2.0]])
        interp = _PolytopeInterp(rows)
        np.testing.assert_array_equal(interp.varying, [True, False, True, False])
        lifted = interp.lift(np.array([[-0.5, -0.5]]))  # (x, z) geometry coordinates
        np.testing.assert_allclose(lifted, [[-0.5, -1.32, -0.5, 1.0]], atol=1e-6)
        samples = interp.sample(50, np.random.default_rng(0))
        np.testing.assert_allclose(samples[:, 1], -1.32, atol=1e-9)

    def test_dependent_column_not_last(self):
        # with ``fixed_elements`` constraining the `final` element, the constrained sub-polytope's
        # dependent chemical potential is the last non-fixed column (not the last column overall); the
        # linear-dependence dropping must identify it (dropping ``c2 = -(c0 + c1)`` here, not the constant
        # fixed column ``c3``), else a degenerate over-dimensioned Delaunay results:
        rows = np.array([[0.0, 0.0, 0.0, -1.32], [-2.0, 0.0, 2.0, -1.32], [0.0, -2.0, 2.0, -1.32]])
        interp = _PolytopeInterp(rows)
        np.testing.assert_array_equal(interp.varying, [True, True, False, False])
        lifted = interp.lift(np.array([[-0.5, -0.5]]))
        np.testing.assert_allclose(lifted, [[-0.5, -0.5, 1.0, -1.32]], atol=1e-6)
        samples = interp.sample(50, np.random.default_rng(0))
        np.testing.assert_allclose(samples[:, 2], -(samples[:, 0] + samples[:, 1]), atol=1e-6)
        np.testing.assert_allclose(samples[:, 3], -1.32, atol=1e-9)
