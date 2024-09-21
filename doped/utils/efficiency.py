"""
Utility functions to improve the efficiency of common
functions/workflows/calculations in ``doped``.
"""

from functools import lru_cache

import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite

# Make composition comparisons faster (used in structure matching etc)
pmg_Comp_eq = Composition.__eq__


@lru_cache(maxsize=int(1e4))
def cached_Comp_eq_func(self_id, other_id):
    """
    Cached equality function for ``Composition`` instances.
    """
    return pmg_Comp_eq(Composition.__instances__[self_id], Composition.__instances__[other_id])


def _comp__eq__(self, other):
    """
    Custom ``__eq__`` method for ``Composition`` instances, using a cached
    equality function to speed up comparisons.
    """
    if not isinstance(other, Composition):
        return NotImplemented

    self_id = id(self)  # Use object id to prevent recursion issues
    other_id = id(other)

    Composition.__instances__[self_id] = self  # Ensure instances are stored for caching
    Composition.__instances__[other_id] = other

    return cached_Comp_eq_func(self_id, other_id)


Composition.__instances__ = {}
Composition.__eq__ = _comp__eq__


# similar for PeriodicSite:
def cache_ready_PeriodicSite__eq__(self, other):
    """
    Custom ``__eq__`` method for ``PeriodicSite`` instances, using a cached
    equality function to speed up comparisons.
    """
    if not isinstance(other, type(self)):
        return NotImplemented

    return (
        self.species == other.species
        and cached_allclose(tuple(self.coords), tuple(other.coords), atol=type(self).position_atol)
        and self.properties == other.properties
    )


@lru_cache(maxsize=int(1e8))
def cached_allclose(a: tuple, b: tuple, rtol: float = 1e-05, atol: float = 1e-08):
    """
    Cached version of ``np.allclose``, taking tuples as inputs (so that they
    are hashable and thus cacheable).
    """
    return np.allclose(np.array(a), np.array(b), rtol=rtol, atol=atol)


PeriodicSite.__instances__ = {}
PeriodicSite.__eq__ = cache_ready_PeriodicSite__eq__
