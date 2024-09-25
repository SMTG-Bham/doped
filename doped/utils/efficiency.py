"""
Utility functions to improve the efficiency of common
functions/workflows/calculations in ``doped``.
"""

import itertools
from collections import defaultdict
from functools import lru_cache

import numpy as np
from pymatgen.analysis.defects.utils import VoronoiPolyhedron
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite
from scipy.spatial import Voronoi

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


class DopedTopographyAnalyzer:
    """
    This is a modified version of
    pymatgen.analysis.defects.utils.TopographyAnalyzer to lean down the input
    options and make initialisation far more efficient (~2 orders of magnitude
    faster).
    """

    def __init__(
        self,
        structure: Structure,
        image_tol: float = 0.0001,
        max_cell_range: int = 1,
        constrained_c_frac: float = 0.5,
        thickness: float = 0.5,
        clustering_tol: float = 0.5,
        min_dist: float = 0.9,
    ) -> None:
        """
        Args:
            structure (Structure): An initial structure.
            image_tol (float): A tolerance distance for the analysis, used to
                determine if something are actually periodic boundary images of
                each other. Default is usually fine.
            max_cell_range (int): This is the range of periodic images to
                construct the Voronoi tessellation. A value of 1 means that we
                include all points from (x +- 1, y +- 1, z+- 1) in the
                voronoi construction. This is because the Voronoi poly
                extends beyond the standard unit cell because of PBC.
                Typically, the default value of 1 works fine for most
                structures and is fast. But for very small unit
                cells with high symmetry, this may need to be increased to 2
                or higher. If there are < 5 atoms in the input structure and
                max_cell_range is 1, this will automatically be increased to 2.
            constrained_c_frac (float): Constraint the region where users want
                to do Topology analysis the default value is 0.5, which is the
                fractional coordinate of the cell
            thickness (float): Along with constrained_c_frac, limit the
                thickness of the regions where we want to explore. Default is
                0.5, which is mapping all the site of the unit cell.
            clustering_tol (float): Tolerance for clustering nodes. Default is
                0.5.
            min_dist (float): Minimum distance between nodes. Default is 0.9.
        """
        self.clustering_tol = clustering_tol
        self.min_dist = min_dist

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
        node_points_map = defaultdict(set)
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
