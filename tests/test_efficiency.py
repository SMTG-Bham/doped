"""
Contract tests for the custom ``__eq__``/``__hash__`` implementations in
``doped`` (``doped.utils.efficiency``, ``doped.core``) and the shared-cache
mutation guards.

The core contract is the Python data-model invariant: ``a == b`` implies
``hash(a) == hash(b)`` (hashes may be coarser than equality -- collisions are
resolved by ``__eq__`` -- but (ideally) not finer, else ``set``/``dict``/cache
lookups silently break). Where hashes are deliberately finer than
tolerance-based equality (near-equal coordinates), that is documented at the
definition site and not asserted here.
"""

import copy
import warnings

import numpy as np
import pytest
from pymatgen.core import Composition, Element, Lattice, PeriodicSite, Species, Structure
from pymatgen.core.ion import Ion
from pymatgen.core.structure import Molecule
from pymatgen.entries.computed_entries import ComputedStructureEntry

import doped.utils.efficiency  # noqa: F401  (applies the ``pymatgen`` patches)
from doped.core import DefectEntry, Vacancy
from doped.utils.symmetry import get_all_equiv_sites, get_distance_matrix, get_primitive_structure, get_sga

CUBIC_LATTICE = Lattice.cubic(5.0)


def _simple_structure():
    return Structure(CUBIC_LATTICE, ["Cd", "Te"], [[0, 0, 0], [0.25, 0.25, 0.25]])


class TestCompositionHashEq:
    def test_tolerance_equal_compositions_share_hash(self):
        c1 = Composition({"Ga": 1.0, "As": 1.0})
        c2 = Composition({"Ga": 1.0 + 1e-9, "As": 1.0})  # within ``amount_tolerance``
        assert c1 == c2
        assert hash(c1) == hash(c2)  # eq -> hash invariant
        assert len({c1, c2}) == 1  # set dedup works
        assert {c1: "x"}.get(c2) == "x"  # dict lookup works

    def test_different_stoichiometries_unequal(self):
        # coarse (chemical-system) hash must not leak into equality:
        assert Composition("Fe2O3") != Composition("Fe3O2")
        assert Composition("Fe2O3") == Composition("Fe2O3")

    def test_composition_dict_comparison(self):
        # documented ``pymatgen`` behaviour: comparison with Element-keyed dicts returns a bool
        # (previously crashed with AttributeError in ``doped``):
        assert Composition("Fe2O3") == {Element("Fe"): 2.0, Element("O"): 3.0}
        assert Composition("Fe2O3") != {Element("Fe"): 3.0, Element("O"): 2.0}
        # Composition.__eq__ supports dict -- Composition comparison
        assert {Element("Fe"): 2.0, Element("O"): 3.0} == Composition("Fe2O3")

    def test_ion_hash_invariant(self):
        # ``Ion.__hash__`` = hash((composition, charge)); inherits the Composition fix:
        i1 = Ion(Composition({"Ga": 1.0, "As": 1.0}), 1)
        i2 = Ion(Composition({"Ga": 1.0 + 1e-9, "As": 1.0}), 1)
        assert i1 == i2
        assert hash(i1) == hash(i2)


class TestPeriodicSiteHashEq:
    def test_identical_sites_invariant(self):
        s1 = PeriodicSite("Fe", [0.1, 0.2, 0.3], CUBIC_LATTICE)
        s2 = PeriodicSite("Fe", [0.1, 0.2, 0.3], CUBIC_LATTICE)
        assert s1 == s2
        assert hash(s1) == hash(s2)
        assert len({s1, s2}) == 1

    def test_ndarray_properties_comparison_returns_bool(self):
        # e.g. ``selective_dynamics`` from POSCAR parsing (previously raised ValueError):
        kwargs = {"coords": [0, 0, 0], "lattice": CUBIC_LATTICE}
        s1 = PeriodicSite("Fe", properties={"sd": np.array([True, False, True])}, **kwargs)
        s2 = PeriodicSite("Fe", properties={"sd": np.array([True, False, True])}, **kwargs)
        s3 = PeriodicSite("Fe", properties={"sd": np.array([False, False, True])}, **kwargs)
        assert s1 == s2
        assert hash(s1) == hash(s2)
        assert s1 != s3

    def test_0d_ndarray_property_hashable(self):
        site = PeriodicSite("Fe", [0, 0, 0], CUBIC_LATTICE, properties={"x": np.array(1.0)})
        assert isinstance(hash(site), int)  # hash with array properties previously raised TypeError

    def test_lattice_compared(self):
        # matches pristine ``pymatgen`` semantics (previously ignored the lattice):
        s1 = PeriodicSite("Fe", [0, 0, 0], CUBIC_LATTICE)
        s2 = PeriodicSite("Fe", [0, 0, 0], Lattice.cubic(4.0))
        assert s1 != s2
        # equal-but-distinct lattice objects still compare equal (allclose semantics, cached):
        s3 = PeriodicSite("Fe", [0, 0, 0], Lattice.cubic(5.0))
        assert s1 == s3

    def test_partial_occupancy_in_hash(self):
        # species amounts are hashed, so occupancy-differing sites/structures don't collide in the
        # hash-equal structure eq fast path:
        s1 = PeriodicSite({"Fe": 0.5}, [0, 0, 0], CUBIC_LATTICE)
        s2 = PeriodicSite({"Fe": 1.0}, [0, 0, 0], CUBIC_LATTICE)
        assert s1 != s2
        assert hash(s1) != hash(s2)
        st1 = Structure(CUBIC_LATTICE, [{"Fe": 0.5}], [[0, 0, 0]])
        st2 = Structure(CUBIC_LATTICE, [{"Fe": 1.0}], [[0, 0, 0]])
        assert st1 != st2


class TestStructureHashEq:
    def test_identical_structures_invariant(self):
        s1, s2 = _simple_structure(), _simple_structure()
        assert s1 == s2
        assert hash(s1) == hash(s2)
        assert len({s1, s2}) == 1

    def test_properties_only_difference_unequal(self):
        # previously falsely equal via the hash-keyed ``__instances__`` registry overwrite:
        s1, s2 = _simple_structure(), _simple_structure()
        s1.properties, s2.properties = {"x": 1}, {"x": 2}
        assert s1 != s2
        s2.properties = {"x": 1}
        assert s1 == s2
        assert hash(s1) == hash(s2)

    def test_rotated_lattice_twin_unequal(self):
        # same lengths/angles and frac coords, different lattice matrix (previously falsely equal,
        # poisoning e.g. the ``get_sga`` cache with wrongly-oriented Cartesian symmetry operations):
        s1 = _simple_structure()
        rotated_lattice = Lattice(CUBIC_LATTICE.matrix[[1, 0, 2]] * np.array([1, 1, -1])[:, None])
        s2 = Structure(rotated_lattice, ["Cd", "Te"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        assert rotated_lattice.lengths == CUBIC_LATTICE.lengths
        assert rotated_lattice.angles == CUBIC_LATTICE.angles
        assert s1 != s2
        assert get_sga(s1) is not get_sga(s2)  # separate (correctly-oriented) SGA cache entries

    def test_site_property_difference_unequal(self):
        # unhashable (dict-valued) site properties previously dropped from the hash -> falsely equal:
        s1 = Structure(CUBIC_LATTICE, ["Fe"], [[0, 0, 0]], site_properties={"d": [{"a": 1}]})
        s2 = Structure(CUBIC_LATTICE, ["Fe"], [[0, 0, 0]], site_properties={"d": [{"a": 2}]})
        assert s1 != s2

    def test_freeze_type_tags(self):
        # ``_freeze`` type-tags container images, so eq-unequal values of different types don't collide in
        # the hash-equal structure eq fast path:
        list_props = Structure(CUBIC_LATTICE, ["Fe"], [[0, 0, 0]], site_properties={"v": [[1, 2]]})
        tuple_props = Structure(CUBIC_LATTICE, ["Fe"], [[0, 0, 0]], site_properties={"v": [(1, 2)]})
        assert list_props != tuple_props  # [1, 2] != (1, 2) under dict eq
        int_array = np.array([1], dtype=np.int64)
        float_array = np.frombuffer(int_array.tobytes())  # same bytes, float64 dtype
        a1 = Structure(CUBIC_LATTICE, ["Fe"], [[0, 0, 0]], site_properties={"v": [int_array]})
        a2 = Structure(CUBIC_LATTICE, ["Fe"], [[0, 0, 0]], site_properties={"v": [float_array]})
        assert a1 != a2  # dtype in the frozen image; same-bytes different-dtype arrays don't collide

    def test_ndarray_structure_properties_comparison_returns_bool(self):
        # structure-level ndarray properties: both the hash-equal fast path and the full comparison path
        # return a bool (pristine pymatgen raises ValueError here; doped improves on it):
        s1, s2 = _simple_structure(), _simple_structure()
        s1.properties = {"m": np.array([1.0, 2.0])}
        s2.properties = {"m": np.array([1.0, 2.0])}
        assert s1 == s2
        s3 = Structure(CUBIC_LATTICE, ["Cd", "Te"], [[0, 0, 0.1], [0.25, 0.25, 0.25]])
        s3.properties = {"m": np.array([1.0, 2.0])}
        assert s1 != s3  # different coords; full comparison path

    def test_tolerant_equality_kept(self):
        s1 = _simple_structure()
        permuted = Structure.from_sites(list(reversed(s1.sites)))
        assert s1 == permuted  # site-order-independent equality
        noisy = Structure(CUBIC_LATTICE, ["Cd", "Te"], [[0, 0, 1e-9], [0.25, 0.25, 0.25]])
        # tolerance-based equality (noise above 1e-10 hash rounding, so hashes may differ, but still eq):
        assert s1 == noisy

    def test_float_noise_twins_share_hash(self):
        # coords/lattices differing only by float noise (<1e-10, e.g. from symmop round-trips) share a hash
        # -> cache hits & set dedup work for them:
        s1 = _simple_structure()
        noisy = Structure(Lattice.cubic(5.0 + 1e-13), ["Cd", "Te"], [[0, 0, 1e-13], [0.25, 0.25, 0.25]])
        assert s1 == noisy
        assert hash(s1) == hash(noisy)
        assert len({s1, noisy}) == 1
        negative_zero = Structure(CUBIC_LATTICE, ["Cd", "Te"], [[0, 0, -1e-13], [0.25, 0.25, 0.25]])
        assert hash(s1) == hash(negative_zero)  # -0.0 normalised to 0.0
        site1 = PeriodicSite("Fe", [0.1, 0.2, 0.3], CUBIC_LATTICE)
        site2 = PeriodicSite("Fe", [0.1, 0.2, 0.3 + 1e-13], CUBIC_LATTICE)
        assert site1 == site2
        assert hash(site1) == hash(site2)


class TestSGACaching:
    def test_per_instance_memoisation_and_frozen_arrays(self):
        sga = get_sga(_simple_structure())
        rotations, translations = sga._get_symmetry()
        assert sga._get_symmetry()[0] is rotations  # memoised
        assert not rotations.flags.writeable  # frozen; mutation raises instead of corrupting the cache
        with pytest.raises(ValueError):
            rotations[0, 0, 0] = 5
        assert not translations.flags.writeable

    def test_fresh_symmetry_operations_list_per_call(self):
        sga = get_sga(_simple_structure())
        ops_a = sga.get_symmetry_operations()
        ops_b = sga.get_symmetry_operations()
        assert ops_a == ops_b
        assert ops_a is not ops_b  # fresh list each call
        ops_a.clear()  # caller mutation...
        assert sga.get_symmetry_operations() == ops_b  # ...does not corrupt the cache

    def test_get_symmetry_reuses_init_dataset_bit_identically(self):
        # patched ``_get_symmetry`` extracts rotations/translations from the symmetry dataset already
        # computed at ``SpacegroupAnalyzer`` init for non-magnetic cells, rather than re-calling
        # ``spglib.get_symmetry``; outputs must be identical to the original method (magnetic cells fall
        # back to it):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        from doped.utils.efficiency import _original_get_symmetry

        magmom_struct = Structure(CUBIC_LATTICE, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        magmom_struct.add_site_property("magmom", [1.0, 2.0])  # magnetic cell -> fallback branch
        perturbed = Structure(
            CUBIC_LATTICE, ["Cd", "Te"], [[0.0001, -0.0002, 0.0001], [0.2499, 0.2502, 0.2498]]
        )
        for struct in [_simple_structure(), _simple_structure() * 2, perturbed, magmom_struct]:
            sga = SpacegroupAnalyzer(struct)
            magnetic = "magmom" in struct.site_properties
            # pin the fast-path predicate itself, so the optimisation can't be silently disabled (e.g.
            # by a ``pymatgen`` attribute rename) leaving these legs vacuously comparing like with like:
            assert len(sga._cell) == (4 if magnetic else 3), struct.formula
            assert magnetic or hasattr(sga._space_group_data, "rotations"), struct.formula
            rotations, translations = sga._get_symmetry()  # patched, fresh SGA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # magnetic-path spglib DeprecationWarning
                orig_rotations, orig_translations = _original_get_symmetry(SpacegroupAnalyzer(struct))
            assert rotations.dtype == orig_rotations.dtype, struct.formula
            assert translations.dtype == orig_translations.dtype, struct.formula
            assert np.array_equal(rotations, orig_rotations), struct.formula
            assert translations.tobytes() == orig_translations.tobytes(), struct.formula  # bit-identical


class TestSharedCacheMutationGuards:
    def test_get_all_equiv_sites_returns_fresh_list(self):
        struct = _simple_structure()
        sites1 = get_all_equiv_sites([0.0, 0.0, 0.0], struct)
        n_sites = len(sites1)
        assert n_sites > 0
        sites1.clear()  # caller mutation...
        assert len(get_all_equiv_sites([0.0, 0.0, 0.0], struct)) == n_sites  # ...cache unaffected

    def test_get_primitive_structure_returns_fresh_structure(self):
        struct = _simple_structure()
        prim1 = get_primitive_structure(struct)
        n_sites = len(prim1)
        prim1.remove_sites([0])  # caller mutation...
        assert len(get_primitive_structure(struct)) == n_sites  # ...cache unaffected

    def test_get_primitive_structure_does_not_mutate_input(self):
        # mixed None/non-None site properties (e.g. slab-like inputs) previously had their properties
        # silently deleted from the *caller's* structure (which was also the captured cache key):
        struct = _simple_structure()
        struct[0].properties["bulk_wyckoff"] = "a"
        struct[1].properties["bulk_wyckoff"] = None
        get_primitive_structure(struct)
        assert struct[0].properties.get("bulk_wyckoff") == "a"

    def test_get_distance_matrix_read_only(self):
        dist_matrix = get_distance_matrix([[0, 0, 0], [0.5, 0.5, 0.5]], CUBIC_LATTICE)
        with pytest.raises(ValueError):
            dist_matrix[0, 0] = 5.0  # shared cached array is frozen; loud failure
        mutable = dist_matrix.copy()
        mutable[0, 0] = 5.0  # callers mutate a copy

    def test_cached_composition_init_returns_fresh_object(self):
        from doped.utils.efficiency import _cache_ready_Composition_init

        c1 = _cache_ready_Composition_init("Fe2O3")
        c2 = _cache_ready_Composition_init("Fe2O3")
        assert c1 == c2
        assert c1 is not c2  # fresh copy each call (incl. cache hits)

    def test_get_orientation_preserving_primitive_returns_fresh_copies(self):
        from doped.utils.symmetry import _get_orientation_preserving_primitive

        supercell = _simple_structure() * 2
        prim_and_matrix = _get_orientation_preserving_primitive(supercell)
        assert prim_and_matrix is not None
        prim, matrix = prim_and_matrix
        n_prim_sites, matrix_00 = len(prim), matrix[0, 0]
        prim.remove_sites([0])  # caller mutation of both returned objects...
        matrix[0, 0] = 99
        prim2, matrix2 = _get_orientation_preserving_primitive(supercell)
        assert len(prim2) == n_prim_sites  # ...does not corrupt the cache
        assert matrix2[0, 0] == matrix_00

    def test_get_orientation_preserving_primitive_handles_dummy_species(self):
        # ``DummySpecies.Z`` is ``hash(symbol)`` -- randomised per process and far outside ``spglib``'s
        # int32 atomic-number range -- which raised ``SpglibError`` for any "X"-decorated structure;
        # now handled without issue:
        from doped.utils.symmetry import _get_orientation_preserving_primitive

        supercell = _simple_structure() * 2
        for frac_coords in np.array(list(np.ndindex(2, 2, 2))) / 2 + 0.125:
            supercell.append("X", frac_coords)  # one X per sub-cell -> still a 2x2x2 supercell
        prim, matrix = _get_orientation_preserving_primitive(supercell)
        assert len(prim) == 3  # Cd + Te + X
        assert "X0+" in [str(specie) for specie in prim.types_of_species]  # X survives the round-trip
        assert round(float(np.linalg.det(matrix))) == 8


class TestNonMagneticSymmetryDefault:
    """
    ``doped`` ignores magnetism in symmetry analysis unless
    ``USE_MAGNETIC_SYMMETRY=1``; spins carried on ``Species`` objects must be
    stripped for that, just like ``magmom`` site properties.
    """

    @staticmethod
    def _spin_structure():
        return Structure(
            CUBIC_LATTICE,
            [Species("Fe", 2, spin=4), Species("Fe", 2, spin=-4), Element("O")],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )

    def test_species_spins_ignored_by_default(self, monkeypatch):
        # spins live on the ``Species`` objects rather than in ``site_properties``, so they need to be
        # pruned to avoid magnetic symmetry handling
        struct = self._spin_structure()
        monkeypatch.delenv("USE_MAGNETIC_SYMMETRY", raising=False)
        sga = get_sga(struct)
        assert len(sga._cell) == 3  # no magmoms handed to spglib
        assert sga.get_space_group_symbol() == "R-3m"  # spin-degenerate Fe -> inversion retained

        monkeypatch.setenv("USE_MAGNETIC_SYMMETRY", "1")
        magnetic_sga = get_sga(struct)
        assert len(magnetic_sga._cell) == 4
        assert magnetic_sga.get_space_group_symbol() == "R3m"  # spins split the Fe sites -> no inversion

    def test_spin_stripping_preserves_occupancies_and_oxidation_states(self, monkeypatch):
        # only ``spin`` may be dropped: occupancies must be summed when spin variants merge onto one site,
        # and oxidation states kept (else mixed-valence sites merge, giving spuriously `higher` symmetry):
        monkeypatch.delenv("USE_MAGNETIC_SYMMETRY", raising=False)
        disordered = Structure(
            CUBIC_LATTICE, [{Species("Fe", 2, spin=4): 0.5, Species("Fe", 2, spin=-4): 0.5}], [[0, 0, 0]]
        )
        assert get_sga(disordered)._structure[0].species == Composition({Species("Fe", 2): 1.0})

        mixed_valence = Structure(
            CUBIC_LATTICE,
            [Species("Fe", 2, spin=4), Species("Fe", 3, spin=-4), Element("O")],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )  # Fe2+ and Fe3+ stay distinct -> no inversion centre, unlike the same-valence case above
        assert get_sga(mixed_valence).get_space_group_symbol() == "R3m"

    def test_spin_stripping_does_not_mutate_input_or_break_dummy_species(self, monkeypatch):
        monkeypatch.delenv("USE_MAGNETIC_SYMMETRY", raising=False)  # else the strip block is skipped
        # ``Structure.remove_spin()`` is not usable here: it rebuilds every species as a ``Species``, which
        # raises on the ``DummySpecies`` ("X") sites used throughout ``doped``:
        struct = self._spin_structure()
        struct.append("X", [0.75, 0.75, 0.75])
        species_before = struct.types_of_species
        get_sga(struct)  # must not raise
        assert struct.types_of_species == species_before  # caller's structure untouched


class TestDefectAndDefectEntryHashEq:
    def _make_vacancy(self, structure=None, site_index=0, perturbation=0.0):
        structure = structure if structure is not None else _simple_structure()
        site = structure[site_index]
        if perturbation:
            site = PeriodicSite(
                site.species,
                site.frac_coords + perturbation,
                structure.lattice,
                properties=site.properties,
            )
        return Vacancy(structure=structure, site=site, oxi_state=0)

    def test_equal_defects_share_hash(self):
        v1 = self._make_vacancy()
        v2 = self._make_vacancy(perturbation=1e-5)  # within symprec -> equal
        assert v1 == v2
        assert hash(v1) == hash(v2)  # eq -> hash invariant
        assert len({v1, v2}) == 1  # set dedup works

    def test_unequal_defects(self):
        v_cd = self._make_vacancy(site_index=0)
        v_te = self._make_vacancy(site_index=1)
        assert v_cd != v_te

    def test_non_defect_comparison_returns_false(self):
        v1 = self._make_vacancy()
        # previously raised TypeError, breaking e.g. ``in``-membership on mixed lists:
        assert v1 is not None
        assert (v1 == None) is False  # noqa: E711
        assert v1 != "v_Cd"
        assert v1 in [None, "x", v1]

    def test_symmetric_equality_with_differing_symprec(self):
        v1 = self._make_vacancy()
        v2 = self._make_vacancy(perturbation=1e-5)
        v2.symprec = 0.5  # stricter (smaller) symprec of the two is used -> symmetric equality
        assert (v1 == v2) == (v2 == v1)

    def _make_entry(self, vacancy, sc_energy=-100.0, bulk_energy=-102.0, name="v_Cd_0"):
        structure = vacancy.structure
        return DefectEntry(
            defect=vacancy,
            charge_state=0,
            sc_entry=ComputedStructureEntry(structure=structure, energy=sc_energy),
            bulk_entry=ComputedStructureEntry(structure=structure, energy=bulk_energy),
            name=name,
        )

    def test_equal_entries_share_hash(self):
        e1 = self._make_entry(self._make_vacancy())
        e2 = self._make_entry(self._make_vacancy(perturbation=1e-5))
        assert e1 == e2
        assert hash(e1) == hash(e2)  # previously broken via the exact defect hash
        assert len({e1, e2}) == 1

    def test_entry_non_entry_comparison_returns_false(self):
        e1 = self._make_entry(self._make_vacancy())
        # previously raised AttributeError:
        assert (e1 == None) is False  # noqa: E711
        assert e1 != 5

    def test_entry_energy_identity_guard(self):
        e1 = self._make_entry(self._make_vacancy())
        assert e1.sc_entry_energy == -100.0
        e1.sc_entry = ComputedStructureEntry(structure=e1.defect.structure, energy=-99.0)
        assert e1.sc_entry_energy == -99.0  # in-place replacement is picked up by the staleness guard
        e1.bulk_entry = ComputedStructureEntry(structure=e1.defect.structure, energy=-101.0)
        assert e1.bulk_entry_energy == -101.0

    def test_entry_as_dict_strips_session_state(self):
        e1 = self._make_entry(self._make_vacancy())
        _ = e1.sc_entry_energy, e1.bulk_entry_energy  # populate identity refs
        entry_dict = e1.as_dict()
        assert not [key for key in entry_dict if "_hash" in key or key.endswith("_entry_ref")]

    def test_entry_from_dict_accepts_legacy_hash_keys(self):
        e1 = self._make_entry(self._make_vacancy())
        entry_dict = e1.as_dict()
        entry_dict["_bulk_entry_hash"] = 123  # keys present in JSONs from older ``doped`` versions (<v4)
        entry_dict["_sc_entry_hash"] = 456
        e2 = DefectEntry.from_dict(entry_dict)
        assert e2.name == e1.name
        assert not hasattr(e2, "_bulk_entry_hash")


class TestStructureDeepcopy:
    def test_deepcopy_does_not_alias_properties(self):
        structure = _simple_structure()
        structure.properties["info"] = {"origin": "test"}
        structure.add_site_property("magmom", [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])])

        deep_copy = copy.deepcopy(structure)
        assert deep_copy == structure

        # mutating the deepcopy must not affect the original (previously aliased via ``.copy()``):
        deep_copy.properties["info"]["origin"] = "mutated"
        deep_copy[0].properties["magmom"][2] = 99.0
        assert structure.properties["info"]["origin"] == "test"
        assert structure[0].properties["magmom"][2] == 1.0


class TestMoleculeHashEq:
    def test_hash_invariant_and_set_dedup(self):
        coords = [[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]]
        water = Molecule(["O", "H", "H"], coords)
        permuted_water = Molecule(["H", "O", "H"], [coords[1], coords[0], coords[2]])
        assert water == permuted_water  # eq is order-insensitive
        assert hash(water) == hash(permuted_water)  # eq -> hash (broken by the old z-matrix hash)
        assert len({water, permuted_water}) == 1  # set dedup works
        # conformer with same composition collides but is resolved by ``==``:
        stretched_water = Molecule(["O", "H", "H"], np.array(coords) * 1.5)
        assert water != stretched_water
        assert len({water, stretched_water}) == 2
