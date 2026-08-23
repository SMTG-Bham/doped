"""
Utility functions to improve the efficiency of common
functions/workflows/calculations in ``doped``.
"""

import contextlib
import copy
import itertools
import operator
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from functools import cached_property, lru_cache
from string import digits
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.defects.utils import VoronoiPolyhedron, remove_collisions
from pymatgen.core.composition import Composition, DummySpecies
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import IStructure, Molecule, Structure
from pymatgen.core.structure_matcher import (
    AbstractComparator,
    ElementComparator,
    FrameworkComparator,
    StructureMatcher,
)
from pymatgen.io.vasp.sets import get_valid_magmom_struct
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.misc import is_np_dict_equal
from scipy.spatial import Voronoi

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from doped.core import Vacancy

# Note: overrides of ``__eq__`` should also override ``__hash__`` (and vice versa), preserving the
# invariant that ``a == b`` implies ``hash(a) == hash(b)`` -- hashes may be coarser than equality
# (collisions are resolved by ``__eq__`` in ``dict``/``set``/``lru_cache`` lookups) but not finer.
# A perfectly eq-consistent hash is impossible for tolerance-based equality (not transitive), so the
# ``PeriodicSite``/``Structure`` hashes below are deliberately finer. Safe for caching (worst case a
# spurious miss & recompute) and for sets of bitwise-identical copies, but ``set``/``dict`` dedup of
# near-equal (eq-equal, not bitwise-identical) objects can silently fail -- dedup those with explicit
# ``==`` loops instead. These patches also make some mutable objects hashable: don't mutate hashed fields
# while keyed!

_TRANSLATE_REMOVE_CHARGE = str.maketrans("", "", digits + "+-.")


def _freeze(obj):
    """
    Recursively convert ``obj`` (e.g. nested ``properties`` values) into a
    hashable canonical form, whose equality mirrors ``__eq__`` (e.g. order-
    insensitive dict equality) -- required for the eq/hash invariant.

    Container images are type-tagged (tuples excepted; already-hashable
    composites) so eq-unequal values of different types (``[1, 2]`` vs
    ``(1,2)``) don't collide into equal hashes (keeping hash equality a sound
    proxy for equality in the ``_Structure__eq__`` hash-equal fast path).
    """
    if isinstance(obj, dict):  # frozenset of items; insertion-order independent (like dict eq)
        return ("dict", frozenset((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return ("list", tuple(_freeze(v) for v in obj))
    if isinstance(obj, tuple):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, set):
        return ("set", frozenset(_freeze(v) for v in obj))
    if isinstance(obj, np.ndarray):
        return ("ndarray", obj.shape, obj.dtype.str, obj.tobytes())
    # else assume it's already hashable (int, str, custom ...)
    return obj


def _frozen_properties(properties: dict):
    """
    Hashable image of a ``properties`` dict, for use in ``__hash__`` methods.
    """
    try:  # fast path: property values already hashable (floats/strings/tuples etc.)
        return frozenset(properties.items())
    except TypeError:  # unhashable values (arrays/lists/dicts etc.); canonicalise with ``_freeze``
        pass
    try:
        return _freeze(properties)
    except Exception:  # exotic unhashable values; repr fallback (worst case: benign cache misses)
        return repr(sorted(properties.items(), key=repr))


def _properties_equal(props_1: dict, props_2: dict) -> bool:
    """
    ``properties`` dict equality, tolerating ``np.ndarray`` values (plain dict
    ``==`` raises ``ValueError`` on these, e.g. ``selective_dynamics``).
    """
    try:
        return props_1 == props_2
    except ValueError:  # array-aware (slower) comparison, only used when needed:
        return is_np_dict_equal(props_1, props_2)


# Composition overrides:
# First, define operand slots for the cached equality functions below. Those ``lru_cache``\s are keyed on
# the operands' exact int fingerprints -- they can't take the objects themselves, as ``lru_cache`` keys
# args via their ``hash``/``==``, recursing into the very ``__eq__`` being implemented -- but hash
# fingerprints aren't invertible, so on a cache miss the function body gets the actual operands from these
# slots instead (set immediately before each cached call, read at function entry; cache hits never enter
# the function body). Avoids unbounded hash->instance registry (memory leak) and associated per-call
# insertion overhead.
_composition_eq_pair: list = [None, None]


@lru_cache(maxsize=int(1e5))  # maxsize on the order of 30 Mb
def doped_Composition_eq_func(self_hash, other_hash):
    r"""
    Updated equality function for |Composition| instances, which breaks early
    for mismatches and also uses caching, making it orders of magnitude faster
    than ``pymatgen``\s equality function.
    """
    self_comp, other_comp = _composition_eq_pair  # read at entry; only runs on cache misses

    return fast_Composition_eq(self_comp, other_comp)


def fast_Composition_eq(self, other):
    """
    Fast equality function for |Composition| instances, breaking early for
    mismatches.
    """
    if len(self) != len(other):  # skip type check here, already checked upstream in ``_Composition__eq__``
        return False

    return all(abs(amt - other[el]) <= type(self).amount_tolerance for el, amt in self.items())


def _composition_fingerprint(composition):
    """
    Exact-stoichiometry fingerprint for |Composition| instances, used as a fast
    equality key below (exact amounts distinguish e.g. Fe2O3 vs Fe3O2).
    """
    return hash(frozenset(composition._data.items()))


def _Composition__eq__(self, other):
    """
    Custom ``__eq__`` method for |Composition| instances, using a cached
    equality function to speed up comparisons.
    """
    if not isinstance(other, type(self) | dict):
        return NotImplemented

    if not isinstance(other, type(self)):  # plain dicts have no ``_data``/hash-cache support;
        return fast_Composition_eq(self, other)  # compare directly

    _composition_eq_pair[:] = (self, other)  # slice-assign to mutate the module-level slots
    return doped_Composition_eq_func(_composition_fingerprint(self), _composition_fingerprint(other))


class Hashabledict(dict):
    def __hash__(self):
        """
        Make the dictionary hashable by recursively "freezing" into only
        hashable built-ins (with ``_freeze``), then hash that.

        Handles nested dicts, lists, sets, tuples, arrays etc.
        """
        return hash(_freeze(self))


def _get_hashable_dict(d: dict) -> Hashabledict:
    if isinstance(d, Hashabledict):
        return d
    if isinstance(d, dict):
        return Hashabledict(d)  # convert to hashable dict for caching purposes
    return d


def _fast_dict_deepcopy_max_two_levels(d: dict) -> dict:
    """
    Fast deepcopy of a dict with at most two levels of nested dicts (i.e. d ->
    dict -> dict -> values).

    Implemented to allow fast deep-copying of nested chemical potential dicts,
    avoiding the overhead of ``deepcopy`` when looping over many chemical
    potential dicts.
    """
    return {
        k: (
            {
                k2: (v2.copy() if isinstance(v2, dict) else v2)  # final level, shallow copy sufficient
                for k2, v2 in v1.items()
            }
            if isinstance(v1, dict)
            else v1
        )
        for k, v1 in d.items()
    }


@lru_cache(maxsize=int(1e5))
def _cached_Composition_init(comp_input):
    return Composition(comp_input)


def _cache_ready_Composition_init(comp_input):
    # copy on every call (incl. cache hits) so caller mutation can't corrupt the cached object:
    return _cached_Composition_init(_get_hashable_dict(comp_input)).copy()


def _fast_get_composition_from_sites(sites: Sequence[Site], assume_full_occupancy: bool = False):
    """
    Helper function to quickly get the composition of a collection of sites,
    faster than initializing a |Structure| object.

    Used in initial drafts of defect stenciling code, but replaced by faster
    methods.
    """
    elem_map: dict[Species, float] = defaultdict(float)
    for site in sites:
        if assume_full_occupancy:
            elem_map[next(iter(site._species))] += 1
        else:
            for species, occu in site._species.items():
                elem_map[species] += occu
    return Composition(elem_map)


Composition.__eq__ = _Composition__eq__


def _parse_site_species_str(site: Site, wout_charge: bool = False) -> str:
    """
    Get the species string of a :class:`~pymatgen.core.sites.Site`, with or
    without charge information.

    Much faster than direct ``str(Site)``. Note that this is faster without
    caching.

    Args:
        site (Site):
            :class:`~pymatgen.core.sites.Site` to get the species string of.
        wout_charge (bool):
            Whether to remove charge information from the species string.

    Returns:
        str:
            Species string of the :class:`~pymatgen.core.sites.Site`, with or
            without charge information.
    """
    return _parse_species_str(site._species, wout_charge=wout_charge)


def _parse_species_str(sp_el: Species | Element, wout_charge: bool = False) -> str:
    """
    Get the string representation of a
    :class:`~pymatgen.core.periodic_table.Species` or
    :class:`~pymatgen.core.periodic_table.Element`, with or without charge
    information.

    Much faster than direct ``str(sp_el)``.

    Args:
        sp_el (Species | Element):
            :class:`~pymatgen.core.periodic_table.Species` or
            :class:`~pymatgen.core.periodic_table.Element` for which to get the
            string representation.
        wout_charge (bool):
            Whether to remove charge information from the species string.
            Default is ``False``.

    Returns:
        str:
            String representation of ``sp_el``, with or without charge
            information.
    """
    if isinstance(sp_el, str):
        species_string = sp_el
    elif isinstance(sp_el, Species) and wout_charge:
        return sp_el.element.symbol  # no charge info, return element sybmol
    elif isinstance(sp_el, Element):
        return sp_el.symbol
    elif isinstance(sp_el, Composition | Species):
        species_string = str(sp_el)
    else:
        species_string = str(Composition(sp_el))

    # remove all digits, +, - or . from species string, if `wout_charge` is True
    return species_string.translate(_TRANSLATE_REMOVE_CHARGE) if wout_charge else species_string


# Species overrides:
_orig_species__str__ = Species.__str__


def _species__str__(self):
    """
    Memoized ``Species.__str__`` (immutable objects); avoids heavy string
    formatting in the many millions of ``Species.__hash__`` (= ``hash(str)``)
    calls from ``Composition`` ``dict`` operations.

    We memoize the string, not
    the hash: changing ``__hash__`` values breaks pre-built ``Species``-keyed
    ``dict``s in ``pymatgen`` (e.g. ``bond_valence``), which store key hashes
    at insertion.
    """
    try:
        return self.__dict__["_doped_str"]
    except KeyError:
        self.__dict__["_doped_str"] = string = _orig_species__str__(self)
        return string


Species.__str__ = _species__str__


def _noise_rounded_bytes(arr) -> bytes:
    """
    Byte-image of a float array with noise below ``1e-10`` collapsed (rounded
    to 10 dp; ``+ 0.0`` normalises ``-0.0``), so coordinates differing only by
    float noise (e.g. from symmetry operations / Cartesian <-> fractional
    round-trips) share a hash.

    ``1e-10`` sits safely below the ``__eq__`` tolerances (``atol=1e-8``) for
    ``Structure``/``PeriodicSite``, so hash-merged values are always still
    eq-equal -- required for the ``_Structure__eq__`` hash-equality fast path
    to stay sound.
    """
    return (arr.round(10) + 0.0).tobytes()


def _species_info(species: dict) -> tuple:
    # avoid ``str(el)`` (``Species.__str__``/format machinery); equal species give equal
    # (symbol, oxi, amount) tuples, incl. amounts to distinguish partial occupancies:
    return tuple((el.symbol, getattr(el, "_oxi_state", None), amt) for el, amt in species.items())


# PeriodicSite overrides:
def _periodic_site__hash__(self):
    """
    Custom ``__hash__`` method for |PeriodicSite| instances; deliberately finer
    than the tolerance-based ``__eq__`` (see module comment above), though
    noise-rounded (``_noise_rounded_bytes``) so float-noise twins share a hash.

    All fields ``__eq__`` compares exactly (incl. amounts & properties) are
    hashed, which the ``doped`` |Structure| ``__eq__`` hash-equality fast path
    relies on.
    """
    base_hash_tuple = (
        _species_info(self._species),
        _noise_rounded_bytes(self.lattice.matrix),
        self.lattice.pbc,
        _noise_rounded_bytes(self.frac_coords),
    )
    if not self.properties:
        return hash(base_hash_tuple)
    return hash((*base_hash_tuple, _frozen_properties(self.properties)))


@lru_cache(maxsize=int(1e3))
def _cached_lattice_eq(matrix_bytes_1: bytes, matrix_bytes_2: bytes, pbc_1: tuple, pbc_2: tuple) -> bool:
    """
    Cached lattice equality from raw matrix bytes and pbc; ``Lattice.__eq__``
    (``np.allclose``) semantics but ~10x faster in hot containment loops, where
    the same lattice pair repeats.
    """
    if pbc_1 != pbc_2:
        return False
    if matrix_bytes_1 == matrix_bytes_2:
        return True
    return bool(
        np.allclose(
            np.frombuffer(matrix_bytes_1).reshape(3, 3), np.frombuffer(matrix_bytes_2).reshape(3, 3)
        )
    )


def cache_ready_Site__eq__(self, other):
    """
    Custom ``__eq__`` method for ``Site``  and |PeriodicSite| instances, using
    a cached equality function to speed up comparisons.
    """
    if self is other:
        return True

    needed_attrs = ("_species", "coords", "properties")

    if not all(hasattr(other, attr) for attr in needed_attrs):
        return NotImplemented

    if not (
        self._species == other._species  # should always work fine (and is faster) if ``Site`` initialised
        and (  # without ``skip_checks`` (default)
            self.coords is other.coords  # if coords are the same object
            or cached_allclose(tuple(self.coords), tuple(other.coords), atol=type(self).position_atol)
        )
    ):
        return False

    # lattice checked only for otherwise-matching sites (cheap checks above dominate hot containment loops)
    other_lattice = getattr(other, "lattice", None)  # plain ``Site``s have no lattice -> skip:
    if (
        other_lattice is not None
        and self.lattice is not other_lattice
        and not _cached_lattice_eq(
            self.lattice.matrix.tobytes(),
            other_lattice.matrix.tobytes(),
            self.lattice.pbc,
            other_lattice.pbc,
        )
    ):
        return False

    return _properties_equal(self.properties, other.properties)


@lru_cache(maxsize=int(2e3))  # maxsize on the order of 1 Mb
def cached_allclose(a: tuple, b: tuple, rtol: float = 1e-05, atol: float = 1e-08):
    """
    Cached version of ``np.allclose``, taking tuples as inputs (so that they
    are hashable and thus cacheable).

    Implemented directly in Python (same semantics as ``np.allclose`` for
    finite inputs), as the ``numpy`` machinery has ~10x higher overhead for the
    small (len-3 coordinate) tuples this is mostly used for (and full
    vectorised allclose should be used otherwise, anyway).
    """
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b, strict=True))


PeriodicSite.__eq__ = cache_ready_Site__eq__
Site.__eq__ = cache_ready_Site__eq__
PeriodicSite.__hash__ = _periodic_site__hash__


# Lattice overrides:
# (note: memoizing ``Lattice.__hash__`` was tested and found to give negligible speedup (~0.1% of
# ``DefectsGenerator`` runtime), as ``pymatgen`` already caches the lengths/angles used in its hash)
_orig_lattice_get_all_distances = Lattice.get_all_distances


@lru_cache(maxsize=int(1e4))  # maxsize on the order of 20 Mb for typical use cases
def _cached_get_all_distances(self: Lattice, frac_coords1: tuple, frac_coords2: tuple):
    return _orig_lattice_get_all_distances(self, np.array(frac_coords1), np.array(frac_coords2))


def array_to_tuple(array: "ArrayLike | tuple") -> tuple:
    """
    Convert an array-like input to tuple.
    """
    array = np.array(array)
    if array.ndim == 1:
        return tuple(array)
    return tuple(map(tuple, array))


def get_all_distances(
    self,
    frac_coords1: "ArrayLike",
    frac_coords2: "ArrayLike",
) -> NDArray[np.float64]:
    """
    Get the distances between two lists of coordinates taking into account
    periodic boundary conditions and the lattice.

    See :meth:`~pymatgen.core.lattice.get_all_distances`.
    """
    return _cached_get_all_distances(
        self, array_to_tuple(frac_coords1), array_to_tuple(frac_coords2)
    ).copy()


Lattice.get_all_distances = get_all_distances

# Structure overrides:


def _structure__hash__(self):
    """
    Custom ``__hash__`` method for |Structure| instances; deliberately finer
    than the tolerance-based, site-order-independent ``__eq__`` (see module
    comment above), though noise-rounded (``_noise_rounded_bytes``) so float-
    noise twins share a hash.

    Includes the full lattice matrix (the coarser ``Lattice.__hash__`` can't
    distinguish rotated/reflected settings) and properties, so every field
    ``__eq__`` compares exactly contributes to the hash -- which the ``__eq__``
    hash-equality fast path relies on.
    """
    lattice_info = (_noise_rounded_bytes(self.lattice.matrix), self.lattice.pbc)
    sites_info = tuple(
        (
            _species_info(site._species),
            _frozen_properties(site.properties) if site.properties else None,
        )
        for site in self._sites
    )
    coords_info = _noise_rounded_bytes(self.frac_coords)  # vectorised coord rounding, rather than per-site
    if not self.properties:
        return hash((lattice_info, coords_info, sites_info))
    return hash((lattice_info, coords_info, sites_info, _frozen_properties(self.properties)))


@contextlib.contextmanager
def cache_species(structure_cls):
    """
    Context manager that makes ``Structure.species`` a cached property, which
    significantly speeds up ``pydefect`` eigenvalue parsing in large structures
    (due to repeated use of ``Structure.indices_from_symbol``.
    """
    original_species = structure_cls.species
    try:
        cached = cached_property(original_species.fget)
        cached.__set_name__(structure_cls, "species")  # Explicit initialization
        structure_cls.species = cached
        yield
    finally:
        structure_cls.species = original_species


def doped_Structure__eq__(self, other: IStructure) -> bool:
    """
    Copied from ``pymatgen``, but updated to break early once a mis-matching
    site is found, to speed up structure matching by ~2x.
    """
    # skip matching object type check here, as already checked upstream in ``_Structure__eq__``
    if other is self:
        return True
    if len(self) != len(other):
        return False
    if self.lattice != other.lattice:
        return False
    if not _properties_equal(self.properties, other.properties):
        return False
    for site in self:  # noqa: SIM110
        if site not in other:
            return False  # break early!
    return True


# operand slots for the cached structure equality function; see ``_composition_eq_pair`` comment above:
_structure_eq_pair: list = [None, None]


@lru_cache(maxsize=int(1e4))
def cached_Structure_eq_func(self_hash, other_hash):
    """
    Cached equality function for |Structure| instances.
    """
    self_struct, other_struct = _structure_eq_pair  # read at entry; only runs on cache misses

    return doped_Structure__eq__(self_struct, other_struct)


def _Structure__eq__(self, other):
    """
    Custom ``__eq__`` method for |Structure|/``IStructure`` instances, using
    both caching and an updated, faster equality function to speed up
    comparisons.
    """
    if self is other:
        return True  # identity fast-path, avoids hashing both structures below

    needed_attrs = ("lattice", "sites", "properties")

    if not all(hasattr(other, attr) for attr in needed_attrs):
        return NotImplemented

    self_hash = _structure__hash__(self)
    other_hash = _structure__hash__(other)

    if self_hash == other_hash:
        return True

    _structure_eq_pair[:] = (self, other)  # slice-assign to mutate the module-level slots
    return cached_Structure_eq_func(self_hash, other_hash)


def _structure__deepcopy__(self, memo):
    """
    Fast ``__deepcopy__`` for ``Structure``: shallow ``.copy()``, then deep-
    copy only the mutable ``properties`` dicts (structure- and site-level) so
    the copy shares no state with the original.
    """
    new_structure = self.copy()
    new_structure.properties = copy.deepcopy(self.properties, memo)
    for new_site, site in zip(new_structure, self, strict=True):
        new_site.properties = copy.deepcopy(site.properties, memo)
    return new_structure


IStructure.__eq__ = _Structure__eq__
IStructure.__hash__ = _structure__hash__
Structure.__eq__ = _Structure__eq__
Structure.__hash__ = _structure__hash__
Structure.__deepcopy__ = _structure__deepcopy__


# Molecule overrides:
def _DopedMolecule__hash__(self):
    """
    ``__hash__`` for (mutable, unhashable-by-default) ``Molecule``, using only
    fields compared _exactly_ by ``Molecule.__eq__`` (composition, charge,
    spin) -- not tolerance-compared coordinates -- so eq-equal molecules always
    share a hash (sets/dicts/caches dedup correctly) and collisions
    (e.g. conformers) are resolved by ``__eq__``.
    """
    return hash((self.composition, self.charge, self.spin_multiplicity))


Molecule.__hash__ = _DopedMolecule__hash__


# SpacegroupAnalyzer overrides:
def _sga__hash__(self):
    """
    Custom ``__hash__`` for ``SpacegroupAnalyzer`` (e.g. for cache keys);
    coarser than the (default, identity) ``__eq__``, so invariant-safe.
    """
    return hash((self._cell, self._symprec, self._angle_tol))


_original_get_symmetry = SpacegroupAnalyzer._get_symmetry


def _get_symmetry(self) -> tuple[NDArray, NDArray]:
    """
    Get the symmetry operations associated with the structure, memoised per-
    instance and ``get_sga`` already caches SGA construction by structure.

    The cached arrays are frozen so caller mutation raises loudly rather than
    silently corrupting the shared values.
    """
    try:
        return self._doped_symmetry
    except AttributeError:
        rotations, translations = _original_get_symmetry(self)
        rotations.flags.writeable = False  # freeze mutatable arrays
        translations.flags.writeable = False
        self._doped_symmetry = (rotations, translations)
        return self._doped_symmetry


_original_get_symmetry_operations = SpacegroupAnalyzer.get_symmetry_operations


def _get_symmetry_operations(self, cartesian: bool = False) -> list[SymmOp]:
    """
    Get the symmetry operations associated with the structure, memoised per-
    instance (as with ``_get_symmetry``).

    A fresh ``list`` is returned on every call (incl. hits) so callers can
    freely modify it; the shared ``SymmOp`` objects themselves should not be
    mutated in-place.
    """
    cache = self.__dict__.setdefault("_doped_symmetry_operations", {})
    if cartesian not in cache:
        cache[cartesian] = _original_get_symmetry_operations(self, cartesian=cartesian)
    return list(cache[cartesian])


SpacegroupAnalyzer.__hash__ = _sga__hash__
SpacegroupAnalyzer._get_symmetry = _get_symmetry
SpacegroupAnalyzer.get_symmetry_operations = _get_symmetry_operations


def _get_symbol(element: Element | Species, comparator: AbstractComparator | None = None) -> str:
    """
    Convenience function to get the symbol of an ``Element`` or ``Species`` as
    a string, with charge information included or excluded depending on the
    choice of ``comparator``.

    By default, the returned symbol does not include any charge / oxidation
    state information. If ``comparator`` is provided and is not
    ``ElementComparator`` / ``FrameworkComparator``, then the ``str(element)``
    representation is returned (which will include charge information if
    ``element`` is a ``Species``).

    Args:
        element (Element | Species):
            ``Element`` or ``Species`` to get the symbol of.
        comparator (AbstractComparator | None):
            Comparator to check if we should return the ``str(element)``
            representation (which includes charge information if ``element`` is
            a ``Species``), or just the element symbol (i.e.
            ``element.symbol``, or ``element.element.symbol`` if ``element`` is
            a ``Species`` object) -- which is the case when ``comparator`` is
            ``None`` (default) or ``ElementComparator`` /
            ``FrameworkComparator``.

    Returns:
        str: Symbol of the element as a string.
    """
    if (
        comparator is not None
        and not isinstance(comparator, ElementComparator | FrameworkComparator)
        and isinstance(element, Species)
    ):
        return str(element)
    return element.symbol if isinstance(element, Element | DummySpecies) else element.element.symbol


def get_element_indices(
    structure: Structure,
    elements: list[Element | Species | str] | None = None,
    comparator: AbstractComparator | None = None,
) -> dict[str, list[int]]:
    """
    Convenience function to generate a dictionary of ``{element: [indices]}``
    for a given |Structure|, where ``indices`` are the indices of the sites in
    the structure corresponding to the given ``elements`` (default is all
    elements in the structure).

    Args:
        structure (|Structure|):
            |Structure| to get the indices from.
        elements (list[Element | Species | str] | None):
            List of elements to get the indices of. If ``None`` (default), all
            elements in the structure are used.
        comparator (AbstractComparator | None):
            Comparator to check if we should return the ``str(element)``
            representation (which includes charge information if ``element`` is
            a ``Species``), or just the element symbol (i.e.
            ``element.element.symbol``) -- which is the case when
            ``comparator`` is ``None`` (default) or ``ElementComparator`` /
            ``FrameworkComparator``.

    Returns:
        dict[str, list[int]]:
            Dictionary of ``{element: [indices]}`` for the given ``elements``
            in the structure.
    """
    if elements is None:
        elements = _fast_get_composition_from_sites(structure).elements

    if not all(isinstance(element, str) for element in elements):
        elements = [_get_symbol(element, comparator) for element in elements]
    species = np.array([_get_symbol(site.specie, comparator) for site in structure])
    return {element: np.where(species == element)[0].tolist() for element in elements}


def get_element_min_max_bond_length_dict(structure: Structure, **sm_kwargs) -> dict:
    r"""
    Get a dictionary of ``{element: (min_bond_length, max_bond_length)}`` for a
    given |Structure|, where ``min_bond_length`` and ``max_bond_length`` are
    the minimum and maximum `smallest` interatomic bond lengths for each
    element in the structure.

    Args:
        structure (|Structure|):
            |Structure| to calculate bond lengths for.
        **sm_kwargs:
            Additional keyword arguments to pass to ``StructureMatcher()``.
            Just used to check if ``comparator`` has been set here (if
            ``ElementComparator``/``FrameworkComparator`` used, then we use
            ``Element``\s rather than ``Species`` as the keys), or if
            ``ignored_species`` is set (in which case these species are
            ignored when calculating bond lengths).

    Returns:
        dict: Dictionary of ``{element: (min_bond_length, max_bond_length)}``.
    """
    comparator = sm_kwargs.get("comparator")

    if len(structure) == 1:
        structure *= 2  # need at least two sites to calculate bond lengths
    elif len(structure) == 0:  # edge case of a 'defect structure' in a primitive cell with 1 atom
        return {}

    # get the distance matrix broken down by species:
    element_idx_dict = get_element_indices(structure, comparator=comparator)
    ignored_indices = [
        idx for elt in sm_kwargs.get("ignored_species", []) for idx in element_idx_dict.get(elt, [])
    ]

    # Note: The repeated distance_matrix call here, particularly when this function is called twice in
    # ``apply_s2_to_s1_transformation()``, can become expensive for large structures (due to the NxN
    # distance calculation). In ``apply_s2_to_s1_transformation()``, where this function is called
    # twice, we sub-select sites in the structure to run the min/max bond lengths test, for this reason
    distance_matrix = structure.distance_matrix
    np.fill_diagonal(distance_matrix, np.inf)  # set diagonal to np.inf to ignore self-distances of 0
    distance_matrix[:, ignored_indices] = np.inf  # set ignored indices to np.inf to ignore these distances
    distance_matrix[ignored_indices, :] = np.inf  # set ignored indices to np.inf to ignore these distances
    element_min_max_bond_length_dict = {
        elt: np.array([np.min(structure.lattice.abc), np.max(structure.lattice.abc)])
        for elt in element_idx_dict
    }  # default to min/max lattice vectors (for cases where there are no other matching non-ignored atoms)

    for elt, site_indices in element_idx_dict.items():
        element_dist_matrix = distance_matrix[:, site_indices]  # (N_of_that_element, N_sites) matrix
        if element_dist_matrix.size != 0:
            min_interatomic_distances_per_atom = np.min(element_dist_matrix, axis=0)  # min along columns
            if np.min(min_interatomic_distances_per_atom) != np.inf:  # other non-ignored matching atoms
                element_min_max_bond_length_dict[elt] = np.array(
                    [
                        np.min(min_interatomic_distances_per_atom),
                        np.max(min_interatomic_distances_per_atom),
                    ]
                )

    return element_min_max_bond_length_dict


def get_dist_equiv_stol(dist: float, structure: Structure) -> float:
    """
    Get the equivalent ``stol`` value for a given Cartesian distance (``dist``)
    in a given |Structure|.

    ``stol`` is a site tolerance parameter used in ``pymatgen``
    |StructureMatcher| functions, defined as the fraction of the average free
    length per atom := ( V / Nsites ) ** (1/3).

    Args:
        dist (float): Cartesian distance in Å.
        structure (|Structure|): |Structure| to calculate ``stol`` for.

    Returns:
        float: Equivalent ``stol`` value for the given distance.
    """
    return dist / (structure.volume / max(len(structure), 1)) ** (1 / 3)  # max to ensure no divide-by-zero


def get_min_stol_for_s1_s2(struct1: Structure, struct2: Structure, **sm_kwargs) -> float:
    """
    Get the minimum possible ``stol`` value which will give a match between
    ``struct1`` and ``struct2`` using |StructureMatcher|, based on the ranges
    of per-element minimum interatomic distances in the two structures.

    Args:
        struct1 (|Structure|): Initial structure.
        struct2 (|Structure|): Final structure.
        **sm_kwargs:
            Additional keyword arguments to pass to ``StructureMatcher()``.
            Just used to check if ``ignored_species`` or ``comparator`` has
            been set here.

    Returns:
        float:
            Minimum ``stol`` value for a match between ``struct1`` and
            ``struct2``. If a direct match is detected (corresponding to min
            ``stol`` = 0, then ``1e-4`` is returned).
    """
    s1_min_max_bond_length_dict = get_element_min_max_bond_length_dict(struct1, **sm_kwargs)
    s2_min_max_bond_length_dict = get_element_min_max_bond_length_dict(struct2, **sm_kwargs)
    common_elts = set(s1_min_max_bond_length_dict.keys()) & set(s2_min_max_bond_length_dict.keys())
    if not common_elts:  # try without oxidation states
        struct1_wout_oxi = struct1.copy()
        struct2_wout_oxi = struct2.copy()
        struct1_wout_oxi.remove_oxidation_states()
        struct2_wout_oxi.remove_oxidation_states()
        s1_min_max_bond_length_dict = get_element_min_max_bond_length_dict(struct1_wout_oxi, **sm_kwargs)
        s2_min_max_bond_length_dict = get_element_min_max_bond_length_dict(struct2_wout_oxi, **sm_kwargs)
        common_elts = set(s1_min_max_bond_length_dict.keys()) & set(s2_min_max_bond_length_dict.keys())

    min_min_dist_change = 1e-4
    with contextlib.suppress(Exception):
        min_min_dist_change = (
            max(
                {
                    elt: max(np.abs(s1_min_max_bond_length_dict[elt] - s2_min_max_bond_length_dict[elt]))
                    for elt in common_elts
                    if elt not in sm_kwargs.get("ignored_species", [])
                }.values()
            )
            / 2
        )  # divide by two as sites may have displaced toward each other (so Δbond-length = 2*Δsite)

    return max(get_dist_equiv_stol(min_min_dist_change, struct1), 1e-4)


def _sm_get_atomic_disps(sm: StructureMatcher, struct1: Structure, struct2: Structure):
    """
    Get the root-mean-square displacement `and atomic displacements` between
    two structures, normalized by the mean free length per atom:
    ``(Vol/Nsites)^(1/3)``.

    These values are not directly returned by |StructureMatcher| methods.
    This function replicates ``StructureMatcher.get_rms_dist()``, but changes
    the return value from ``match[0], max(match[1])`` to ``match[0], match[1]``
    to allow further analysis of displacements. Mainly intended for use by
    |ShakeNBreak|.

    Args:
        sm (|StructureMatcher|): ``pymatgen`` |StructureMatcher| object.
        struct1 (|Structure|): Initial structure.
        struct2 (|Structure|): Final structure.

    Returns:
        tuple:

            - float: Normalised RMS displacement between the two structures.
            - np.ndarray: Normalised displacements between the two structures.

        or ``None`` if no match is found.
    """
    struct1, struct2 = sm._process_species([struct1, struct2])
    struct1, struct2, fu, s1_supercell = sm._preprocess(struct1, struct2)
    match = sm._match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=False)

    return None if match is None else (match[0], match[1])


def StructureMatcher_scan_stol(
    struct1: Structure,
    struct2: Structure,
    func_name: str = "get_s2_like_s1",
    min_stol: float | None = None,
    max_stol: float = 0.3,
    stol_factor: float = 0.5,
    **sm_kwargs,
):
    r"""
    Utility function to scan through a range of ``stol`` values for
    |StructureMatcher| until a match is found between ``struct1`` and
    ``struct2`` (i.e. ``StructureMatcher.{func_name}`` returns a result).

    The ``StructureMatcher.match()`` function (used in most
    |StructureMatcher| methods) speed is heavily dependent on ``stol``, with
    smaller values being faster, so we can speed up evaluation by starting with
    small values and increasing until a match is found (especially with the
    ``doped`` efficiency tools which implement caching (and other improvements)
    to ensure no redundant work here).

    Note that ``ElementComparator()`` is used by default here! (So sites with
    different species but the same element (e.g. "S2-" & "S0+") will be
    considered match-able). This can be controlled with
    ``sm_kwargs['comparator']``.

    Note: If you know reduction to primitive cells is not possible/needed, then
    setting ``primitive_cell=False`` in ``sm_kwargs`` can significantly speed
    up matching here (by avoiding expensive reduction to primitive cells for
    large structures).

    Args:
        struct1 (|Structure|): ``struct1`` for ``StructureMatcher.match()``.
        struct2 (|Structure|): ``struct2`` for ``StructureMatcher.match()``.
        func_name (str):
            The name of the |StructureMatcher| method to return the result
            of ``StructureMatcher.{func_name}(struct1, struct2)`` for, such
            as:

            - "get_s2_like_s1" (default)
            - "get_rms_dist"
            - "fit"
            - "fit_anonymous"
            - "get_rms_anonymous"
        min_stol (float):
            Minimum ``stol`` value to try. Default is to use ``doped``\s
            ``get_min_stol_for_s1_s2()`` function to estimate the minimum
            ``stol`` necessary, and start with 2x this value to achieve fast
            structure-matching in most cases.
        max_stol (float):
            Maximum ``stol`` value to try. Default: 0.3 (matching
            |StructureMatcher| default).
        stol_factor (float):
            Fractional increment to increase ``stol`` by each time (when a
            match is not found). Default value of 0.5 increases ``stol`` by 50%
            each time.
        **sm_kwargs:
            Additional keyword arguments to pass to ``StructureMatcher()``.

    Returns:
        Result of ``StructureMatcher.{func_name}(struct1, struct2)`` or
        ``None`` if no match is found.
    """
    # use doped efficiency tools to make structure-matching as fast as possible:
    if func_name == "_get_atomic_disps":  # only used by SnB; add get-atomic-disps method:
        StructureMatcher._get_atomic_disps = _sm_get_atomic_disps

    if "comparator" not in sm_kwargs:
        sm_kwargs["comparator"] = ElementComparator()

    if min_stol is None:
        min_stol = get_min_stol_for_s1_s2(struct1, struct2, **sm_kwargs) * 2

    # here we cycle through a range of stols, because we just need to find the closest match so we could
    # use a high ``stol`` from the start and it would give correct result, but higher ``stol``\s take
    # much longer to run as it cycles through multiple possible matches. So we start with a low ``stol``
    # and break once a match is found:
    stol = min_stol
    while stol <= max_stol:
        if user_stol := sm_kwargs.pop("stol", False):  # first run, try using user-provided stol first:
            sm_full_user_custom = StructureMatcher(stol=user_stol, **sm_kwargs)
            result = getattr(sm_full_user_custom, func_name)(struct1, struct2)
            if result is not None:
                return result

        sm = StructureMatcher(stol=stol, **sm_kwargs)
        result = getattr(sm, func_name)(struct1, struct2)
        if (
            result is not None
            and result is not False
            and not (isinstance(result, tuple) and result[0] is None)  # for ``get_rms_anonymous()``
        ):
            return result

        if stol == max_stol:  # failed with max_stol; break
            break

        stol = min(stol * (1 + stol_factor), max_stol)
        # Note: this function could possibly be sped up if ``StructureMatcher._match()`` was updated to
        # return the guessed ``best_match`` value (even if larger than ``stol``), which will always be
        # >= the best possible match it seems, and then using this to determine the next ``stol`` value
        # to trial. Seems like it could give a ~50% speedup in some cases? Not clear though,
        # as once you're getting a reasonable guessed value out, the trial ``stol`` should be pretty
        # close to the necessary value anyway.

    return None


class DopedTopographyAnalyzer:
    """
    This is a modified version of
    :class:`~pymatgen.analysis.defects.utils.TopographyAnalyzer` to lean down
    the input options and make initialisation far more efficient (~2 orders of
    magnitude faster).

    The original code was written by Danny Broberg and colleagues
    (10.1016/j.cpc.2018.01.004), which was then added to ``pymatgen`` before
    being cut.
    """

    def __init__(
        self,
        structure: Structure,
        image_tol: float = 0.0001,
        max_cell_range: int = 1,
        constrained_c_frac: float = 0.5,
        thickness: float = 0.5,
    ) -> None:
        """
        Args:
            structure (|Structure|):
                |Structure| to analyse.
            image_tol (float):
                A tolerance distance for the analysis, used to determine if
                sites are periodic images of each other. Default (of 1e-4) is
                usually fine.
            max_cell_range (int):
                This is the range of periodic images to construct the Voronoi
                tessellation. A value of 1 means that we include all points
                from ``(x +- 1, y +- 1, z+- 1)`` in the Voronoi construction.
                This is because the Voronoi polyhedra extend beyond the
                standard unit cell because of PBC. Typically, the default value
                of 1 works fine for most structures and is fast. But for very
                small unit cells with high symmetry, this may need to be
                increased to 2 or higher. If there are < 5 atoms in the input
                structure and ``max_cell_range`` is 1, this will automatically
                be increased to 2.
            constrained_c_frac (float):
                Constrain the region where topology analysis is performed.
                Only sites with ``z`` fractional coordinates between
                ``constrained_c_frac +/- thickness`` are considered. Default of
                0.5 (with ``thickness`` of 0.5) includes all sites in the unit
                cell.
            thickness (float):
                Constrain the region where topology analysis is performed.
                Only sites with ``z`` fractional coordinates between
                ``constrained_c_frac +/- thickness`` are considered. Default of
                0.5 (with ``thickness`` of 0.5) includes all sites in the unit
                cell.
        """
        # if input cell is very small (< 5 atoms) and max cell range is 1 (default), bump to 2 for
        # accurate Voronoi tessellation:
        if len(structure) < 5 and max_cell_range == 1:
            max_cell_range = 2

        self.structure = structure.copy()
        self.structure.remove_oxidation_states()

        constrained_sites = []
        for _i, site in enumerate(self.structure):
            if (
                site.frac_coords[2] >= constrained_c_frac - thickness
                and site.frac_coords[2] <= constrained_c_frac + thickness
            ):
                constrained_sites.append(site)
        constrained_struct = Structure.from_sites(sites=constrained_sites)
        lattice = constrained_struct.lattice

        coords = []
        cell_range = list(range(-max_cell_range, max_cell_range + 1))
        for shift in itertools.product(cell_range, cell_range, cell_range):
            for site in constrained_struct.sites:
                shifted = site.frac_coords + shift
                coords.append(lattice.get_cartesian_coords(shifted))

        # Perform the voronoi tessellation.
        voro = Voronoi(coords)
        node_points_map: defaultdict[int, set] = defaultdict(set)
        for pts, vs in voro.ridge_dict.items():
            for v in vs:
                node_points_map[v].update(pts)

        vnodes: list[VoronoiPolyhedron] = []

        def get_mapping(vnodes, poly: VoronoiPolyhedron):
            """
            Check if a Voronoi Polyhedron is a periodic image of one of the
            existing polyhedra.

            Modified to avoid expensive ``np.allclose()`` calls.
            """
            if not vnodes:
                return None
            distance_matrix = lattice.get_all_distances([v.frac_coords for v in vnodes], poly.frac_coords)
            if np.any(distance_matrix < image_tol):
                for v in vnodes:
                    if v.is_image(poly, image_tol):
                        return v
            return None

        # Filter all the voronoi polyhedra so that we only consider those
        # which are within the unit cell:
        for i, vertex in enumerate(voro.vertices):
            if i == 0:
                continue
            fcoord = lattice.get_fractional_coords(vertex)
            if np.all([-image_tol <= c < 1 + image_tol for c in fcoord]):
                poly = VoronoiPolyhedron(lattice, fcoord, node_points_map[i], coords, i)
                if get_mapping(vnodes, poly) is None:
                    vnodes.append(poly)

        self.coords = coords
        self.vnodes = vnodes


def get_voronoi_nodes(structure: Structure) -> list[PeriodicSite]:
    """
    Get the Voronoi nodes of a ``pymatgen`` |Structure|.

    Maximises efficiency by mapping down to the primitive cell, doing Voronoi
    analysis (with the efficient ``DopedTopographyAnalyzer`` class), and then
    mapping back to the original structure (typically a supercell).

    Args:
        structure (|Structure|):
            ``pymatgen`` |Structure| object.

    Returns:
        list[PeriodicSite]:
            List of |PeriodicSite| objects representing the Voronoi nodes.
    """
    # fresh ``list`` on every call (incl. cache hits), so caller mutation cannot corrupt the cache:
    return list(_hashable_get_voronoi_nodes(structure))


@lru_cache(maxsize=int(1e2))
def _hashable_get_voronoi_nodes(structure: Structure) -> list[PeriodicSite]:
    from doped.utils.symmetry import _get_orientation_preserving_primitive, doped_cluster_frac_coords

    # map all sites to the unit cell; 0 ≤ xyz < 1.
    structure = Structure.from_sites(structure, to_unit_cell=True)
    # get Voronoi nodes in the primitive structure and then map back to the supercell; using the
    # orientation-preserving primitive, so that mapping back is just the integer supercell matrix expansion
    # (no structure-matching or origin-shift handling needed):
    try:
        prim_and_matrix = _get_orientation_preserving_primitive(structure)
    except ValueError:  # non-integer supercell matrix (no other primitive); analyse structure directly
        prim_and_matrix = None
    prim_structure, supercell_matrix = prim_and_matrix or (structure, np.eye(3, dtype=int))

    top_analyzer = DopedTopographyAnalyzer(prim_structure)
    voronoi_coords = [v.frac_coords for v in top_analyzer.vnodes]
    # remove nodes less than 0.5 Å from sites in the structure
    voronoi_coords = remove_collisions(voronoi_coords, structure=prim_structure, min_dist=0.5)
    # cluster nodes within 0.2 Å of each other:
    prim_vnodes = doped_cluster_frac_coords(voronoi_coords, prim_structure, tol=0.2)

    voronoi_struct = Structure.from_sites(
        [PeriodicSite("X", fpos, prim_structure.lattice, skip_checks=True) for fpos in prim_vnodes]
    )  # Structure with Voronoi nodes as sites
    voronoi_struct.make_supercell(supercell_matrix)  # Map back to the supercell

    return voronoi_struct.sites.copy()  # copy() to help avoid mutability issues with cached outputs


def _generic_group_labels(list_in: Sequence, comp: Callable = operator.eq) -> list[int]:
    """
    Group a list of unsortable objects, using a given comparator function.

    Templated off the ``pymatgen-analysis-defects`` function, but fixed to
    avoid broken reassignment logic and overwriting of labels (resulting in
    sites being incorrectly dropped).

    Previously in ``doped`` interstitial generation, but then removed after
    updates in commit ``4699f38`` (for v3.0.0) to use faster site-matching
    functions from ``doped``.

    Args:
        list_in (Sequence): A sequence of objects to group using ``comp``.
        comp (Callable): A comparator function.

    Returns:
        list[int]: list of labels for the input list
    """
    list_out = [-1] * len(list_in)  # Initialize with -1 instead of None for clarity
    label_num = 0

    for i1 in range(len(list_in)):
        if list_out[i1] != -1:  # Already labeled
            continue
        list_out[i1] = label_num
        for i2 in range(i1 + 1, len(list_in)):
            if list_out[i2] == -1 and comp(list_in[i1], list_in[i2]):
                list_out[i2] = label_num
        label_num += 1

    return list_out


class DopedVacancyGenerator(VacancyGenerator):
    """
    Vacancy defects generator, subclassed from ``pymatgen-analysis-defects`` to
    improve efficiency (particularly when handling defect complexes).
    """

    def generate(
        self,
        structure: Structure,
        rm_species: set[str | Species] | list[str | Species] | None = None,
        **kwargs,
    ) -> Generator["Vacancy", None, None]:
        """
        Generate vacancy defects.

        Args:
            structure (|Structure|):
                The structure to generate vacancy defects in.
            rm_species (set[str | Species] | list[str | Species] | None):
                List/set of species to be removed (i.e. to consider for vacancy
                generation). If ``None``, considers all species.
            **kwargs:
                Additional keyword arguments for the ``Vacancy`` constructor.

        Returns:
            Generator[Vacancy, None, None]:
                Generator that yields a list of ``Vacancy`` objects.
        """
        from doped.core import Vacancy
        from doped.utils.symmetry import get_sga

        # core difference is the removal of unnecessary `remove_oxidation_states` calls
        structure = get_valid_magmom_struct(structure)
        all_species = {elt.symbol for elt in structure.composition.elements}
        rm_species = all_species if rm_species is None else {*map(str, rm_species)}

        if not set(rm_species).issubset(all_species):
            raise ValueError(
                f"rm_species ({rm_species}) must be a subset of the structure's species ({all_species})."
            )

        sga = get_sga(structure)
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            if site.specie.symbol in rm_species:
                yield Vacancy(
                    structure=structure,  # note that we no longer remove oxi states here! or in get_sga
                    site=site,
                    equivalent_sites=site_group,
                    **kwargs,
                )
