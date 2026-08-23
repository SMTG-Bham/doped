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

import numpy as np
import pytest
from pymatgen.core import Composition, Element, Lattice, PeriodicSite, Structure
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
