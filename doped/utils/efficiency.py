"""
Utility functions to improve the efficiency of common
functions/workflows/calculations in ``doped``.
"""

import itertools
import operator
import re
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Callable

import numpy as np
from pymatgen.analysis.defects.utils import VoronoiPolyhedron, remove_collisions
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition, Element, Species
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import IStructure, Structure
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Voronoi
from scipy.spatial.distance import squareform

from doped.utils import symmetry

# Make composition comparisons faster (used in structure matching etc)


def _composition__hash__(self):
    """
    Custom ``__hash__`` method for ``Composition`` instances.

    ``pymatgen`` composition has just hashes the chemical system
    (without stoichiometry), which cannot then be used to
    distinguish different compositions.
    """
    return hash(frozenset(self._data.items()))


@lru_cache(maxsize=int(1e8))
def doped_Composition_eq_func(self_hash, other_hash):
    """
    Update equality function for ``Composition`` instances, which breaks early
    for mismatches and also uses caching, making orders of magnitude faster
    than ``pymatgen`` equality function.
    """
    self_comp = Composition.__instances__[self_hash]
    other_comp = Composition.__instances__[other_hash]

    return fast_Composition_eq(self_comp, other_comp)


def fast_Composition_eq(self, other):
    """
    Fast equality function for ``Composition`` instances, breaking early for
    mismatches.
    """
    # skip matching object type check here, as already checked upstream in ``_Composition__eq__``
    if len(self) != len(other):
        return False

    for el, amt in self.items():  # noqa: SIM110
        if abs(amt - other[el]) > type(self).amount_tolerance:
            return False

    return True


def _Composition__eq__(self, other):
    """
    Custom ``__eq__`` method for ``Composition`` instances, using a cached
    equality function to speed up comparisons.
    """
    if not isinstance(other, type(self) | dict):
        return NotImplemented

    # use object hash with instances to avoid recursion issues (for class method)
    self_hash = _composition__hash__(self)
    other_hash = _composition__hash__(other)

    Composition.__instances__[self_hash] = self  # Ensure instances are stored for caching
    Composition.__instances__[other_hash] = other

    return doped_Composition_eq_func(self_hash, other_hash)


Composition.__instances__ = {}
Composition.__eq__ = _Composition__eq__
Composition.__hash__ = _composition__hash__


class Hashabledict(dict):
    def __hash__(self):
        """
        Make the dictionary hashable by converting it to a tuple of key-value
        pairs.
        """
        return hash(tuple(sorted(self.items())))


@lru_cache(maxsize=int(1e5))
def _cached_Composition_init(comp_dict):
    return Composition(comp_dict)


def _fast_get_composition_from_sites(sites, assume_full_occupancy=False):
    """
    Helper function to quickly get the composition of a collection of sites,
    faster than initializing a ``Structure`` object.

    Used in initial drafts of defect stenciling code, but replaced by faster
    methods.
    """
    elem_map: dict[Species, float] = defaultdict(float)
    for site in sites:
        if assume_full_occupancy:
            elem_map[next(iter(site._species))] += 1
        else:
            for species, occu in site.species.items():
                elem_map[species] += occu
    return Composition(elem_map)


@lru_cache(maxsize=int(1e5))
def _parse_site_species_str(site: Site, wout_charge: bool = False):
    if isinstance(site._species, Element):
        return site._species.symbol
    if isinstance(site._species, str):
        species_string = site._species
    elif isinstance(site._species, (Composition, dict)):
        species_string = str(next(iter(site._species)))
    else:
        raise ValueError(f"Unexpected species type: {type(site._species)}")

    if wout_charge:  # remove all digits, + or - from species string
        return re.sub(r"\d+|[\+\-]", "", species_string)
    return species_string


# similar for PeriodicSite:
def cache_ready_PeriodicSite__eq__(self, other):
    """
    Custom ``__eq__`` method for ``PeriodicSite`` instances, using a cached
    equality function to speed up comparisons.
    """
    needed_attrs = ("_species", "coords", "properties")

    if not all(hasattr(other, attr) for attr in needed_attrs):
        return NotImplemented

    return (
        self._species == other._species  # should always work fine (and is faster) if Site initialised
        # without ``skip_checks`` (default)
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


PeriodicSite.__eq__ = cache_ready_PeriodicSite__eq__


# make PeriodicSites hashable:
def _periodic_site__hash__(self):
    """
    Custom ``__hash__`` method for ``PeriodicSite`` instances.
    """
    property_dict = (
        {k: tuple(v) if isinstance(v, (list, np.ndarray)) else v for k, v in self.properties.items()}
        if self.properties
        else {}
    )
    try:
        site_hash = hash((self.species, tuple(self.coords), frozenset(property_dict.items())))
    except Exception:  # hash without the property dict
        site_hash = hash((self.species, tuple(self.coords)))
    return site_hash  # who robbed the hash from the gaff


PeriodicSite.__hash__ = _periodic_site__hash__


# make Structure objects hashable, using lattice and sites:
def _structure__hash__(self):
    """
    Custom ``__hash__`` method for ``Structure`` instances.
    """
    return hash((self.lattice, frozenset(self.sites)))


Structure.__hash__ = _structure__hash__
Structure.__deepcopy__ = lambda x, y: x.copy()  # make deepcopying faster, shallow copy fine for structures
IStructure.__hash__ = _structure__hash__


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
    if self.properties != other.properties:
        return False
    for site in self:  # noqa: SIM110
        if site not in other:
            return False  # break early!
    return True


@lru_cache(maxsize=int(1e4))
def cached_Structure_eq_func(self_hash, other_hash):
    """
    Cached equality function for ``Composition`` instances.
    """
    return doped_Structure__eq__(IStructure.__instances__[self_hash], IStructure.__instances__[other_hash])


def _Structure__eq__(self, other):
    """
    Custom ``__eq__`` method for ``Structure``/``IStructure`` instances, using
    both caching and an updated, faster equality function to speed up
    comparisons.
    """
    needed_attrs = ("lattice", "sites", "properties")

    if not all(hasattr(other, attr) for attr in needed_attrs):
        return NotImplemented

    self_hash = _structure__hash__(self)
    other_hash = _structure__hash__(other)

    IStructure.__instances__[self_hash] = self  # Ensure instances are stored for caching
    IStructure.__instances__[other_hash] = other

    return cached_Structure_eq_func(self_hash, other_hash)


IStructure.__instances__ = {}
IStructure.__eq__ = _Structure__eq__
Structure.__eq__ = _Structure__eq__


class DopedTopographyAnalyzer:
    """
    This is a modified version of
    ``pymatgen.analysis.defects.utils.TopographyAnalyzer`` to lean down the
    input options and make initialisation far more efficient (~2 orders of
    magnitude faster).

    The original code was written by Danny Broberg and colleagues
    (10.1016/j.cpc.2018.01.004), which was then added to ``pymatgen`` before being
    cut.
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


def get_voronoi_nodes(structure: Structure) -> list[PeriodicSite]:
    """
    Get the Voronoi nodes of a ``pymatgen`` ``Structure``.

    Maximises efficiency by mapping down to the primitive cell,
    doing Voronoi analysis (with the efficient ``DopedTopographyAnalyzer``
    class), and then mapping back to the original structure (typically
    a supercell).

    Args:
        structure (:obj:`Structure`):
            pymatgen `Structure` object.

    Returns:
        list[PeriodicSite]:
            List of `PeriodicSite` objects representing the Voronoi nodes.
    """
    structure.__hash__ = _structure__hash__  # make sure Structure is hashable
    return _hashable_get_voronoi_nodes(structure)


@lru_cache(maxsize=int(1e2))
def _hashable_get_voronoi_nodes(structure: Structure) -> list[PeriodicSite]:
    # map all sites to the unit cell; 0 ≤ xyz < 1.
    structure = Structure.from_sites(structure, to_unit_cell=True)
    # get Voronoi nodes in primitive structure and then map back to the supercell:
    prim_structure = structure.get_primitive_structure()

    top_analyzer = DopedTopographyAnalyzer(prim_structure)
    voronoi_coords = [v.frac_coords for v in top_analyzer.vnodes]
    # remove nodes less than 0.5 Å from sites in the structure
    voronoi_coords = remove_collisions(voronoi_coords, structure=prim_structure, min_dist=0.5)
    # cluster nodes within 0.2 Å of each other:
    prim_vnodes: np.array = _doped_cluster_frac_coords(voronoi_coords, prim_structure, tol=0.2)

    # map back to the supercell
    sm = StructureMatcher(primitive_cell=False, attempt_supercell=True)
    mapping = sm.get_supercell_matrix(structure, prim_structure)
    voronoi_struct = Structure.from_sites(
        [PeriodicSite("X", fpos, structure.lattice) for fpos in prim_vnodes]
    )  # Structure with Voronoi nodes as sites
    voronoi_struct.make_supercell(mapping)  # Map back to the supercell

    # check if there was an origin shift between primitive and supercell
    regenerated_supercell = prim_structure.copy()
    regenerated_supercell.make_supercell(mapping)
    fractional_shift = sm.get_transformation(structure, regenerated_supercell)[1]
    if not np.allclose(fractional_shift, 0):
        voronoi_struct.translate_sites(range(len(voronoi_struct)), fractional_shift, frac_coords=True)

    return voronoi_struct.sites


def _doped_cluster_frac_coords(
    fcoords: np.typing.ArrayLike,
    structure: Structure,
    tol: float = 0.55,
    symmetry_preference: float = 0.1,
) -> np.typing.NDArray:
    """
    Cluster fractional coordinates that are within a certain distance tolerance
    of each other, and return the cluster site.

    Modified from the ``pymatgen-analysis-defects``` function as follows:
    For each site cluster, the possible sites to choose from are the sites
    in the cluster _and_ the cluster midpoint (average position). Of these
    sites, the site with the highest symmetry, and then largest ``min_dist``
    (distance to any host lattice site), is chosen -- if its ``min_dist`` is
    no more than ``symmetry_preference`` (0.1 Å by default) smaller than
    the site with the largest ``min_dist``. This is because we want to favour
    the higher symmetry interstitial sites (as these are typically the more
    intuitive sites for placement, cleaner, easier for analysis etc, and work
    well when combined with ``ShakeNBreak`` or other structure-searching
    techniques to account for symmetry-breaking), but also interstitials are
    often lowest-energy when furthest from host atoms (i.e. in the largest
    interstitial voids -- particularly for fully-ionised charge states), and
    so this approach tries to strike a balance between these two goals.

    In ``pymatgen-analysis-defects``, the average cluster position is used,
    which breaks symmetries and is less easy to manipulate in the following
    interstitial generation functions.

    Args:
        fcoords (npt.ArrayLike): Fractional coordinates of points to cluster.
        structure (Structure): The host structure.
        tol (float):
            A distance tolerance for clustering Voronoi nodes. Default is 0.55 Å.
        symmetry_preference (float):
            A distance preference for symmetry over minimum distance to host atoms,
            as detailed in docstring above.
            Default is 0.1 Å.

    Returns:
        np.typing.NDArray: Clustered fractional coordinates
    """
    if len(fcoords) <= 1:
        return None

    lattice = structure.lattice
    sga = symmetry.get_sga(structure, symprec=0.1)  # for getting symmetries of different sites
    symm_ops = sga.get_symmetry_operations()  # fractional symm_ops by default
    dist_matrix = np.array(lattice.get_all_distances(fcoords, fcoords))
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    for i in range(len(dist_matrix)):
        dist_matrix[i, i] = 0
    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    cn = fcluster(z, tol, criterion="distance")
    unique_fcoords = []

    for n in set(cn):
        frac_coords = []
        for i, j in enumerate(np.where(cn == n)[0]):
            if i == 0:
                frac_coords.append(fcoords[j])
            else:
                fcoord = fcoords[j]  # We need the image to combine the frac_coords properly:
                d, image = lattice.get_distance_and_image(frac_coords[0], fcoord)
                frac_coords.append(fcoord + image)

        frac_coords.append(np.average(frac_coords, axis=0))  # midpoint of cluster
        frac_coords_scores = {
            tuple(x): (
                -symmetry.group_order_from_schoenflies(
                    symmetry.point_symmetry_from_site(x, structure, symm_ops=symm_ops)
                ),  # higher order = higher symmetry
                -np.min(lattice.get_all_distances(x, structure.frac_coords), axis=1),
                *symmetry._frac_coords_sort_func(x),
            )
            for x in frac_coords
        }
        symmetry_favoured_site = sorted(frac_coords_scores.items(), key=lambda x: x[1])[0][0]
        dist_favoured_site = sorted(
            frac_coords_scores.items(), key=lambda x: (x[1][1], x[1][0], *x[1][2:])
        )[0][0]

        if (
            np.min(lattice.get_all_distances(dist_favoured_site, structure.frac_coords), axis=1)
            < np.min(lattice.get_all_distances(symmetry_favoured_site, structure.frac_coords), axis=1)
            - symmetry_preference
        ):
            unique_fcoords.append(dist_favoured_site)
        else:  # prefer symmetry over distance if difference is sufficiently small
            unique_fcoords.append(symmetry_favoured_site)

    return symmetry._vectorized_custom_round(
        np.mod(symmetry._vectorized_custom_round(unique_fcoords, 5), 1), 4
    )  # to unit cell


def _generic_group_labels(list_in: Sequence, comp: Callable = operator.eq) -> list[int]:
    """
    Group a list of unsortable objects.

    Templated off the ``pymatgen-analysis-defects`` function,
    but fixed to avoid broken reassignment logic and overwriting
    of labels (resulting in sites being incorrectly dropped).

    Args:
        list_in: A list of objects to group using ``comp``.
        comp: Comparator function.

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
