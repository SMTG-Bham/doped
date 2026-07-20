"""
Calculator-agnostic defect identification and site-mapping utilities
(site/structure matching between bulk and defect supercells, defect type
identification etc.).
"""

import warnings
from copy import deepcopy
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core.structure import Composition, Lattice, PeriodicSite, SiteCollection, Structure
from pymatgen.core.structure_matcher import get_linear_assignment_solution, pbc_shortest_vectors
from pymatgen.util.coord import all_distances
from pymatgen.util.typing import SpeciesLike

from doped.utils import _warn_parameter_order
from doped.utils.efficiency import _parse_site_species_str


def get_defect_type_and_composition_diff(
    defect: Structure | Composition,
    bulk: Structure | Composition,
    _parameter_order_warn: bool = True,
) -> tuple[str, dict]:
    """
    Get the difference in composition between a bulk structure and a defect
    structure.

    Args:
        defect (|Structure| | |Composition|):
            The defect structure or composition.
        bulk (|Structure| | |Composition|):
            The bulk structure or composition.

    Returns:
        tuple[str, dict[str, int]]:
            The defect type (``interstitial``, ``vacancy``, ``substitution`` or
            ``complex``) and the composition difference between the bulk and
            defect structures as a dictionary.
    """
    if _parameter_order_warn:
        _warn_parameter_order("get_defect_type_and_composition_diff")  # TODO: Remove in doped v4.1
    bulk_comp = bulk.composition if isinstance(bulk, Structure) else bulk
    defect_comp = defect.composition if isinstance(defect, Structure) else defect

    bulk_comp_dict = bulk_comp.get_el_amt_dict()
    defect_comp_dict = defect_comp.get_el_amt_dict()

    composition_diff = {
        element: int(defect_amount - bulk_comp_dict.get(element, 0))
        for element, defect_amount in defect_comp_dict.items()
        if int(defect_amount - bulk_comp_dict.get(element, 0)) != 0
    }

    if len(composition_diff) == 1 and next(iter(composition_diff.values())) == 1:
        defect_type = "interstitial"
    elif len(composition_diff) == 1 and next(iter(composition_diff.values())) == -1:
        defect_type = "vacancy"
    elif len(composition_diff) == 2 and all(i in composition_diff.values() for i in [-1, 1]):
        defect_type = "substitution"
    else:
        defect_type = "complex"

    if len(composition_diff) > 5:  # likely a mistake, warn user:
        warnings.warn(
            f"The composition difference between the bulk ({bulk_comp_dict}) and defect "
            f"({defect_comp_dict}) structures is quite large, suggesting either a large complex defect "
            f"or a mistake in the inputs. Beware!"
        )

    return defect_type, composition_diff


def get_defect_type_and_site_indices(
    defect_supercell: Structure,
    bulk_supercell: Structure,
    site_tol: float | None = None,  # TODO: Change to 0.5 and add complex defect handling
    abs_tol: bool = False,
    use_oxi_states: bool = False,
    use_rms: bool = False,
) -> tuple[str, list[int], list[int]]:
    """
    Get the defect type, and indices of defect sites in the bulk (vacancies /
    substitutions) and defect (interstitials / substitutions) supercells.

    Defect sites are determined by matching sites in the bulk and defect
    structures (by element and distances), according to ``site_tol``.

    Note that this assumes consistent cell definitions (lattice vectors and
    bases) for the input defect and bulk supercells, and does not perform any
    structural re-orientations.

    Args:
        defect_supercell (|Structure|):
            The defect supercell structure.
        bulk_supercell (|Structure|):
            The bulk supercell structure.
        site_tol (float | None):
            The (fractional) tolerance for matching sites between the defect
            and bulk structures. If ``abs_tol`` is ``False`` (default), then
            the distance threshold for matching is set to the product of
            ``site_tol`` and the shortest bond length in the bulk structure for
            the given species, otherwise the value is used directly (as a
            length in Å).
            If ``None`` (default), the defect is assumed to be a point defect,
            and the largest site mismatch is assigned as the defect site.
        abs_tol (bool):
            Whether to use ``site_tol`` as an absolute distance tolerance (in
            Å) instead of a fractional tolerance (in terms of the shortest bond
            length in the structure). Default is ``False``.
        use_oxi_states (bool):
            Whether to use the oxidation states of the sites in the bulk and
            defect structures when considering matching sites (such that e.g.
            ``Fe3+`` and ``Fe2+`` would be considered different species).
            Default is ``False``.
        use_rms (bool):
            Site mapping (using linear assignment) -- used to determine defect
            sites -- will be that which minimises either the summed RMS
            distances (if ``use_rms`` is ``True``) or just simple linear sum of
            distances (if ``False``, default) between all paired sites.

    Returns:
        defect_type (str):
            The type of defect as a string (``interstitial``, ``vacancy`` or
            ``substitution``).
        missing_bulk_site_indices (list[int]):
            Indices of sites in the bulk structure that do not match any site
            in the defect structure (according to ``site_tol`` choice).
        additional_defect_site_indices (list[int]):
            Indices of sites in the defect structure that do not match any site
            in the bulk structure (according to ``site_tol`` choice).
    """
    # TODO: Default is 0.5 (i.e. half the shortest bond length in the bulk structure for the given
    #  species). -- add to site_tol docstring
    bulk_composition = bulk_supercell.composition
    defect_composition = defect_supercell.composition

    try:
        defect_type, comp_diff = get_defect_type_and_composition_diff(
            defect_composition, bulk_composition, _parameter_order_warn=False
        )  # internal call with correct (defect, bulk) ordering; don't warn
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_supercell)} in bulk vs. {len(defect_supercell)} in defect?"
        ) from exc

    if site_tol is None and defect_type == "complex":
        raise ValueError(
            f"Based on the composition difference between defect and bulk structures ({comp_diff}), "
            f"the defect is a complex defect, but ``site_tol`` is set to ``None`` which enforces the "
            f"assumption of a point defect. Please set ``site_tol`` to allow parsing of complex defect "
            f"sites."
        )

    oxi_state_decorated = [  # if all sites in both structures are not oxi-state decorated / neutral
        any(i in site.species_string for i in ["+", "-", "0"])
        for site in [*bulk_supercell.sites, *defect_supercell.sites]
    ]
    if len(set(oxi_state_decorated)) > 1 and use_oxi_states:  # not consistent, ignore oxi states:
        warnings.warn(
            "`use_oxi_states` was set to `True`, but not all sites in the bulk and defect structures are "
            "oxidation state decorated. Setting `use_oxi_states` to `False`."
        )
        use_oxi_states = False

    elt_symbols = {
        str(species) if use_oxi_states else species.symbol
        for species in bulk_composition.elements + defect_composition.elements
    }
    additional_defect_site_indices = []
    missing_bulk_site_indices = []
    distance_matrix = bulk_supercell.distance_matrix

    for elt_symbol in elt_symbols:
        bulk_species_fcoords, bulk_species_indices = get_coords_and_idx_of_species(
            bulk_supercell, elt_symbol, use_oxi_states=use_oxi_states
        )
        defect_species_fcoords, defect_species_indices = get_coords_and_idx_of_species(
            defect_supercell, elt_symbol, use_oxi_states=use_oxi_states
        )
        if bulk_species_indices.size == 0:  # extrinsic species
            site_dist_tol = None
        else:
            species_distances = distance_matrix[bulk_species_indices]
            species_min_dist = max(species_distances[np.nonzero(species_distances)].min(), 1)
            site_dist_tol = site_tol if site_tol is None or abs_tol else site_tol * species_min_dist

        site_mapping = _get_site_mapping_from_coords_and_indices(
            defect_species_fcoords,
            bulk_species_fcoords,
            lattice=bulk_supercell.lattice,
            s1_indices=defect_species_indices,
            s2_indices=bulk_species_indices,
            use_rms=use_rms,
        )
        defect_site_mappings = [
            mapping
            for mapping in site_mapping
            if mapping[0] is None or (site_dist_tol is not None and mapping[0] > site_dist_tol)
        ]
        for _dist, defect_idx, bulk_idx in defect_site_mappings:
            if bulk_idx is not None:  # missing bulk site
                missing_bulk_site_indices.append(bulk_idx)
            if defect_idx is not None:  # additional defect site (may be from same matched tuple if dist
                additional_defect_site_indices.append(defect_idx)  # greater than site_dist_tol)

    return defect_type, missing_bulk_site_indices, additional_defect_site_indices


def get_coords_and_idx_of_species(
    structure_or_sites: SiteCollection,
    species_name: str,
    frac_coords: bool = True,
    use_oxi_states: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get arrays of the coordinates and indices of the given species in the
    structure/list of sites.
    """
    coords = []
    idx = []
    for i, site in enumerate(structure_or_sites):
        if _parse_site_species_str(site, wout_charge=not use_oxi_states) == species_name:
            coords.append(site.frac_coords if frac_coords else site.coords)
            idx.append(i)

    return np.array(coords), np.array(idx)


def get_matching_site(
    site: PeriodicSite | np.ndarray, structure: Structure, anonymous: bool = False, tol: float = 0.5
) -> PeriodicSite:
    """
    Get the (closest) matching |PeriodicSite| in ``structure`` for the input
    ``site``, which can be a |PeriodicSite| or fractional coordinates.

    If the closest matching site in ``structure`` is > ``tol`` Å (0.5 Å by
    default) away from the input ``site`` coordinates, an error is raised.

    Automatically accounts for possible differences in assigned oxidation
    states, site property dicts etc.

    Args:
        site (|PeriodicSite| | np.ndarray):
            The site for which to find the closest matching site in
            ``structure``, either as a |PeriodicSite| or fractional
            coordinates array. If fractional coordinates, then ``anonymous``
            is set to ``True``.
        structure (|Structure|):
            The structure in which to search for matching sites to ``site``.
        anonymous (bool):
            Whether to use anonymous matching, allowing different
            species/elements to match each other (i.e. just matching based on
            coordinates). Default is ``False`` if ``site`` is a
            |PeriodicSite|, and ``True`` if ``site`` is fractional
            coordinates.
        tol (float):
            A distance tolerance (in Å), where an error will be thrown if the
            closest matching site is > ``tol`` Å away from the input ``site``.
            Default is 0.5 Å.

    Returns:
        PeriodicSite:
            The closest matching site in ``structure`` to the input ``site``.
    """
    from doped.core import remove_site_oxi_state

    if (
        isinstance(site, PeriodicSite) and not anonymous
    ):  # try directly match first         if site in structure:
        if site in structure:
            return site

        site_w_no_ox_state = deepcopy(site)
        remove_site_oxi_state(site_w_no_ox_state)
        site_w_no_ox_state.properties = {}

        bulk_sites_w_no_ox_state = structure.copy().sites
        for bulk_site in bulk_sites_w_no_ox_state:
            remove_site_oxi_state(bulk_site)
            bulk_site.properties = {}

        if site_w_no_ox_state in bulk_sites_w_no_ox_state:
            return structure.sites[bulk_sites_w_no_ox_state.index(site_w_no_ox_state)]

    site_frac_coords = (
        site.frac_coords if hasattr(site, "frac_coords") else np.array(site, dtype=float)
    )  # ensure site is in fractional coords

    # else get closest site in structure, raising error if not within tol Å:
    if isinstance(site, PeriodicSite) and not anonymous:  # reduce to only matching species
        candidate_frac_coords, candidate_indices = get_coords_and_idx_of_species(
            structure, site.specie.symbol
        )
    else:
        candidate_frac_coords = structure.frac_coords
        candidate_indices = np.arange(len(structure))

    closest_site_idx = candidate_indices[
        np.argmin(structure.lattice.get_all_distances(site_frac_coords, candidate_frac_coords).ravel())
    ]
    closest_site = structure.sites[closest_site_idx]

    closest_site_dist = closest_site.distance_and_image_from_frac_coords(site_frac_coords)[0]
    if closest_site_dist > tol:
        raise ValueError(
            f"Closest site to input defect site ({site}) in bulk supercell is {closest_site} "
            f"with distance {closest_site_dist:.2f} Å (greater than {tol} Å and suggesting a likely "
            f"mismatch in sites/structures here!)."
        )

    if (
        not anonymous
        and isinstance(site, PeriodicSite)
        and site.specie.symbol != closest_site.specie.symbol
    ):
        raise ValueError(
            f"Closest site to input defect site ({site}) in bulk supercell is {closest_site} "
            f"with distance {closest_site_dist:.2f} Å which is a different element! Set `anonymous=True` "
            f"to allow matching of different elements/species if this is desired."
        )

    return closest_site


def _create_unrelaxed_defect_structure(
    defect_supercell: Structure,
    bulk_supercell: Structure,
    defect_site_idx: int | None = None,
    bulk_site_idx: int | None = None,
    defect_coords: bool = False,
) -> Structure:
    """
    Create the unrelaxed defect structure, which corresponds to the bulk
    supercell with the unrelaxed defect site.

    The unrelaxed defect site corresponds to the vacancy/substitution site in
    the pristine (bulk) supercell for vacancies/substitutions, and the `final`
    relaxed interstitial site for interstitials (as the assignment of their
    initial site is ambiguous).

    Args:
        defect_supercell (Structure):
            The defect structure.
        bulk_supercell (Structure):
            The bulk structure.
        defect_site_idx (int):
            The index of the defect site to use in the unreleaxed defect
            structure. Just for consistency with the relaxed defect structure.
        bulk_site_idx (int):
            The index of the site in the bulk structure that corresponds to the
            defect site in the defect structure.
        defect_coords (bool):
            Whether to use the fractional coordinates of the defect site in the
            defect structure, or the bulk structure. Irrelevant for vacancies.
            Parent functions in ``doped`` use ``True`` for interstitials, and
            ``False`` for substitutions (i.e. use bulk site coords).

    Returns:
        Structure:
            The unrelaxed defect structure.
    """
    unrelaxed_defect_structure = bulk_supercell.copy()  # create unrelaxed defect structure

    if bulk_site_idx is not None:
        unrelaxed_defect_structure.remove_sites([bulk_site_idx])

    if defect_site_idx is not None:
        defect_site_in_defect = defect_supercell[defect_site_idx]
        if not defect_coords and bulk_site_idx is not None:
            defect_coords = bulk_supercell[bulk_site_idx].frac_coords
        else:
            defect_coords = defect_site_in_defect.frac_coords

        unrelaxed_defect_structure.insert(defect_site_idx, defect_site_in_defect.species, defect_coords)

    return unrelaxed_defect_structure


def get_wigner_seitz_radius(lattice: Structure | Lattice) -> float:
    """
    Calculates the Wigner-Seitz radius of the structure, which corresponds to
    the maximum radius of a sphere fitting inside the cell.

    Templated on the ``calc_max_sphere_radius`` function from ``pydefect``,
    but rewritten to avoid calling ``vise`` which causes hanging on Windows.
    (https://github.com/SMTG-Bham/doped/issues/147).

    Args:
        lattice (|Structure| | |Lattice|):
            The lattice of the structure (either a ``pymatgen`` |Structure|
            or |Lattice| object).

    Returns:
        float:
            The Wigner-Seitz radius of the structure.
    """
    lattice_matrix = lattice.matrix if isinstance(lattice, Lattice) else lattice.lattice.matrix
    distances = np.zeros(3, dtype=float)  # copied over from pydefect v0.9.4; avoid vise issues
    for i in range(3):
        a_i_a_j = np.cross(lattice_matrix[i - 2], lattice_matrix[i - 1])
        a_k = lattice_matrix[i]
        distances[i] = abs(np.dot(a_i_a_j, a_k)) / np.linalg.norm(a_i_a_j)
    return max(distances) / 2.0


def check_atom_mapping_far_from_defect(
    defect_supercell: Structure,
    bulk_supercell: Structure,
    defect_coords: np.ndarray,
    coords_are_cartesian: bool = False,
    displacement_tol: float = 0.5,
    fraction_tol: float = 0.2,
    warning: bool | str = "verbose",
) -> bool:
    """
    Check the displacement of atoms far from the determined defect site, and
    warn the user if they are large (often indicates a mismatch between the
    bulk and defect supercell definitions).

    For sites of a given species outside the Wigner-Seitz radius of the defect
    (the radius of the largest sphere which can fit in the cell), a 'large'
    displacement is flagged if either the *mean* displacement exceeds
    ``displacement_tol`` Ångströms (capturing a systematic/global mismatch),
    *or* the *fraction* of such sites individually displaced by more than
    ``displacement_tol`` exceeds ``fraction_tol`` (capturing a partial mismatch
    without being triggered by single outlier sites).

    Args:
        defect_supercell (|Structure|):
            The defect structure.
        bulk_supercell (|Structure|):
            The bulk structure.
        defect_coords (np.ndarray):
            The coordinates of the defect site.
        coords_are_cartesian (bool):
            Whether the defect coordinates are in Cartesian or fractional
            coordinates. Default is ``False`` (fractional).
        displacement_tol (float):
            The tolerance for the displacement of individual atoms far from the
            defect site, in Ångströms. Default is 0.5 Å.
        fraction_tol (float):
            The tolerance for the fraction of far-from-defect sites (of a given
            species) displaced by more than ``displacement_tol``, above which a
            mismatch is flagged. Default is 0.2 (i.e. 20%).
        warning (bool, str):
            Whether to throw a warning if a mismatch is detected. If
            ``warning = "verbose"`` (default), the individual atomic
            displacements are included in the warning message.

    Returns:
        bool:
            Returns ``False`` if a mismatch is detected, else ``True``.
    """
    wigner_seitz_radius = get_wigner_seitz_radius(bulk_supercell.lattice)
    defect_frac_coords = (
        defect_coords
        if not coords_are_cartesian
        else bulk_supercell.lattice.get_fractional_coords(defect_coords)
    )

    bulk_sites_outside_or_at_ws_radius = [  # vectorised for fast computation
        bulk_supercell[i]
        for i in np.where(
            bulk_supercell.lattice.get_all_distances(
                bulk_supercell.frac_coords, defect_frac_coords
            ).ravel()
            > np.max((wigner_seitz_radius - 1, 1))
        )[0]
    ]
    defect_sites_outside_wigner_radius = [  # vectorised for fast computation
        defect_supercell[i]
        for i in np.where(
            defect_supercell.lattice.get_all_distances(
                defect_supercell.frac_coords, defect_frac_coords
            ).ravel()
            > wigner_seitz_radius
        )[0]
    ]

    disps_outside_ws: dict[str, list[float]] = {site.specie.symbol: [] for site in bulk_supercell}
    for species in bulk_supercell.composition.elements:  # divide and vectorise calc for efficiency
        bulk_species_outside_near_ws_fcoords = get_coords_and_idx_of_species(
            bulk_sites_outside_or_at_ws_radius, species.name
        )[0]
        defect_species_outside_ws_fcoords = get_coords_and_idx_of_species(
            defect_sites_outside_wigner_radius, species.name
        )[0]
        if (
            min(
                len(bulk_species_outside_near_ws_fcoords),
                len(defect_species_outside_ws_fcoords),
            )
            == 0
        ):
            continue  # if no sites of this species outside the WS radius, skip

        site_mapping_outside_ws = _get_site_mapping_from_coords_and_indices(
            defect_species_outside_ws_fcoords,
            bulk_species_outside_near_ws_fcoords,
            lattice=bulk_supercell.lattice,
        )
        displacement_dists = [dist for dist, _i, _j in site_mapping_outside_ws if dist is not None]
        disps_outside_ws[species.name].extend(np.round(displacement_dists, 2))

    if large_disps_outside_ws := {
        specie: list
        for specie, list in disps_outside_ws.items()
        if list
        and (
            np.mean(list) > displacement_tol  # mean displacement of ``specie`` sites exceeds tolerance
            or np.mean(np.array(list) > displacement_tol) > fraction_tol  # significant fraction exceed
        )
    }:
        message = (
            f"Detected atoms far from the defect site (>{wigner_seitz_radius:.2f} Å) with major "
            f"displacements (>{displacement_tol} Å) in the defect supercell. This likely indicates a "
            f"mismatch between the bulk and defect supercell definitions (-> see troubleshooting docs) or "
            f"an unconverged supercell size, both of which could cause errors in parsing. The mean (or at "
            f"least {fraction_tol:.0%}) of displacements of the following species, at sites far from the "
            f"determined defect position, is >{displacement_tol} Å: {list(large_disps_outside_ws.keys())}"
        )
        if warning == "verbose":
            message += f", with displacements (Å): {large_disps_outside_ws}"
        if warning:
            warnings.warn(message)

        return False

    return True


def _get_site_mapping_from_coords_and_indices(
    s1_frac_coords: ArrayLike,
    s2_frac_coords: ArrayLike,
    s1_indices: np.ndarray | None = None,
    s2_indices: np.ndarray | None = None,
    lattice: Lattice | None = None,
    use_rms: bool = False,
) -> list[tuple[float | None, int | None, int | None]]:
    """
    Get the site mapping between two sets of coordinates and indices, based on
    the shortest distances between sites.

    Args:
        s1_frac_coords (np.ndarray[float]):
            The fractional coordinates of the first set of sites.
        s2_frac_coords (np.ndarray[float]):
            The fractional coordinates of the second set of sites.
        s1_indices (np.ndarray[int] | None):
            The indices of the first set of sites. If ``None``, the indices are
            assumed to be the range of the number of sites in
            ``s1_frac_coords``.
        s2_indices (np.ndarray[int] | None):
            The indices of the second set of sites. If ``None``, the indices
            are assumed to be the range of the number of sites in
            ``s2_frac_coords``.
        lattice (Lattice | None):
            The lattice of the structures. If ``None``, the identity matrix is
            used.
        use_rms (bool):
            The returned site mapping (using linear assignment) will be that
            which minimises either the summed RMS distances (if ``use_rms`` is
            ``True``) or just simple linear sum of distances (if ``False``,
            default) between all paired sites.

    Returns:
        list:
            A list of lists containing the distance, index from ``s1_indices``
            and index from ``s2_indices`` for each matched site.
    """
    lattice = lattice or Lattice(np.eye(3))
    s1_frac_coords = np.asarray(s1_frac_coords)
    s2_frac_coords = np.asarray(s2_frac_coords)
    if s1_indices is None:
        s1_indices = np.arange(len(s1_frac_coords))
    if s2_indices is None:
        s2_indices = np.arange(len(s2_frac_coords))

    for empty_coords, indices, tuple_idx in [
        (s1_frac_coords, s2_indices, 2),
        (s2_frac_coords, s1_indices, 1),
    ]:
        if empty_coords.size == 0:  # handly case of empty input coords
            if indices is None:
                return [(None, None, None)]
            return [
                (None, None if tuple_idx == 2 else int(i), None if tuple_idx == 1 else int(i))
                for i in indices
            ]

    s1_is_subset = len(s1_frac_coords) < len(s2_frac_coords)
    subset_fcoords, subset_indices = (
        (s1_frac_coords, s1_indices) if s1_is_subset else (s2_frac_coords, s2_indices)
    )
    superset_fcoords, superset_indices = (
        (s2_frac_coords, s2_indices) if s1_is_subset else (s1_frac_coords, s1_indices)
    )
    # Note: if needed in future, could be sped up by using k-D trees and/or k-NN searching (rather than
    # global PBC dists over all sites of the same species), but not a bottleneck for typical (~<10,000
    # atom) supercells currently
    _vecs, d_2 = pbc_shortest_vectors(lattice, subset_fcoords, superset_fcoords, return_d2=True)
    dists = np.sqrt(d_2)
    site_matches, _ = get_linear_assignment_solution(d_2 if use_rms else dists)
    site_mapping = [  # site_matches -> matching superset indices, of len(subset)
        (dists[i, j], subset_indices[i], superset_indices[j]) for i, j in enumerate(site_matches)
    ]
    for missing_index in set(range(len(superset_fcoords))) - set(site_matches):
        site_mapping.append((None, None, superset_indices[missing_index]))  # unmatched sites

    if not s1_is_subset:  # swap tuple order, to match (dist, s1_index, s2_index)
        site_mapping = [(dist, index2, index1) for dist, index1, index2 in site_mapping]

    return site_mapping


def get_site_mappings(
    struct1: Structure,
    struct2: Structure,
    species: SpeciesLike | None = None,
    allow_duplicates: bool = False,
    threshold: float = 2.0,
    anonymous: bool = False,
    ignored_species: list[str] | None = None,
    frac_coords: bool = True,
    use_rms: bool = False,
) -> list[tuple[float | None, int | None, int | None]]:
    """
    Get the site mappings between two structures (from ``struct1`` to
    ``struct2``), based on the shortest distances between sites.

    The two structures may have different species orderings.

    NOTE: If ``frac_coords = True`` (default), this assumes that both
    structures have the same lattice definitions (i.e. that they match, and
    aren't rigidly translated/rotated with respect to each other), which is
    mostly the case unless we have a mismatching defect/bulk supercell (in
    which case the ``check_atom_mapping_far_from_defect`` warning should be
    thrown anyway during parsing).

    Args:
        struct1 (|Structure|):
            The input structure.
        struct2 (|Structure|):
            The template structure.
        species (str):
            If provided, only sites of this species will be considered when
            matching sites. Default is ``None`` (all species).
        allow_duplicates (bool):
            If ``True``, allow multiple sites in ``struct1`` to be matched to
            the same site in ``struct2``. Default is ``False``.
        threshold (float):
            If the distance between a pair of matched sites is larger than
            this, then a warning will be thrown. Default is 2.0 Å.
        anonymous (bool):
            If ``True``, the species of the sites will not be considered when
            matching sites. Default is ``False`` (only matching species can be
            matched together).
        ignored_species (list[str]):
            A list of species to ignore when matching sites. Default is no
            species ignored.
        frac_coords (bool):
            Whether to match sites based on their fractional coordinate
            distances (i.e. assuming PBC with matching lattice definitions,
            using the lattice of ``struct1``)(default). If ``False``, instead
            matches sites based on distances between their Cartesian
            coordinates, with no consideration of PBC.
        use_rms (bool):
            The returned site mapping (using linear assignment -- only
            applicable when ``allow_duplicates`` is ``False``) will be that
            which minimises either the summed RMS distances (if ``use_rms`` is
            ``True``) or just simple linear sum of distances (if ``False``,
            default) between all paired sites.

    Returns:
        list:
            A list of lists containing the distance, index in ``struct1`` and
            index in ``struct2`` for each matched site.
    """

    def get_coords(site: PeriodicSite):
        return list(site.frac_coords) if frac_coords else list(site.coords)

    def get_distances(
        coords1: np.ndarray | list, coords2: np.ndarray | list, lattice: Lattice | None = None
    ):
        if frac_coords:
            assert lattice is not None, "Lattice needs to be given if frac_coords is True!"
            return lattice.get_all_distances(coords1, coords2)
        return all_distances(coords1, coords2)

    # Generate a site matching table between the input and the template
    min_dist_with_index: list[tuple] = []
    s1_species_symbols = (
        [
            species.symbol
            for species in struct1.composition.elements
            if species.symbol not in (ignored_species or [])
        ]
        if not anonymous
        else [None]
    )

    for s1_species_symbol in s1_species_symbols:
        if species is not None and s1_species_symbol != species and not anonymous:
            continue
        # Build (struct1_index, coords) pairs for this species, preserving ``struct1`` order:
        species_input = [
            (i, get_coords(site))
            for i, site in enumerate(struct1)
            if (site.specie.symbol == s1_species_symbol or anonymous)
        ]
        input_coords = [coords for _, coords in species_input]
        species_s2_indices = [
            i for i, site in enumerate(struct2) if (site.specie.symbol == s1_species_symbol or anonymous)
        ]
        template_coords = [get_coords(struct2[i]) for i in species_s2_indices]

        dmat = (
            get_distances(input_coords, template_coords, lattice=struct1.lattice)
            if template_coords
            else None
        )
        dmat = dmat**2 if (use_rms and dmat is not None) else dmat  # square if use_rms is True

        # TODO: Can _get_site_mapping_from_coords_and_indices be used instead here (with minimal
        #  efficiency) loss:?
        if not allow_duplicates and dmat is not None:
            # Use linear assignment for order-independent optimal matching.
            # get_linear_assignment_solution returns (col_ind, total_cost), where col_ind[i] is the
            # template index assigned to input row i (requires n_rows <= n_cols). For n > m, transpose
            # the problem (assign each template to one input) and invert the mapping:
            if len(input_coords) <= len(template_coords):
                tmpl_col_indices, _ = get_linear_assignment_solution(dmat)
                input_to_template = dict(enumerate(tmpl_col_indices.tolist()))
            else:
                # dmat.T is (n_templates, n_inputs): each template row j is assigned input column
                # input_col_indices[j]. We need input_idx -> tmpl_idx for the loop below:
                input_col_indices, _ = get_linear_assignment_solution(dmat.T)
                input_to_template = {int(input_col_indices[j]): j for j in range(len(template_coords))}
        else:
            input_to_template = None

        for input_idx, (index, _) in enumerate(species_input):
            if dmat is None:
                min_dist_with_index.append((None, index, None))
                continue

            if input_to_template is not None:
                if input_idx not in input_to_template:
                    # No unique template available (more inputs than templates for this species)
                    min_dist_with_index.append((None, index, None))
                    continue
                tmpl_idx = input_to_template[input_idx]

            else:  # allow_duplicates=True: each input independently picks its closest template
                dists = dmat[input_idx]
                tmpl_idx = dists.argmin()

            current_dist = float(dmat[input_idx, tmpl_idx]) ** (0.5 if use_rms else 1)
            # Map species-local template index (tmpl_idx) to global struct2 index (species_s2_indices):
            template_index = species_s2_indices[tmpl_idx]

            if current_dist > threshold:
                warnings.warn(
                    f"Large site displacement {current_dist:.2f} Å detected when matching atomic sites: "
                    f"{struct1[index]} -> {struct2[template_index]}."
                )

            min_dist_with_index.append((current_dist, index, template_index))

    if not min_dist_with_index:
        raise RuntimeError(
            f"No matching sites for species {species} found between the two structures!\n"
            f"Struct1 composition: {struct1.composition}, Struct2 composition: {struct2.composition}"
        )

    return min_dist_with_index


def reorder_s2_like_s1(s1_structure: Structure, s2_structure: Structure, threshold=5.0) -> Structure:
    """
    Reorder the atoms of a (relaxed) structure, ``s2_structure``, to match the
    ordering of the atoms in ``s1_structure``.

    s1/s2 structures may have a different species orderings.

    NOTE: This assumes that both structures have the same lattice definitions
    (i.e. that they match, and aren't rigidly translated/rotated with respect
    to each other), which is mostly the case unless we have a mismatching
    defect/bulk supercell (in which case the
    ``check_atom_mapping_far_from_defect`` warning should be thrown anyway
    during parsing).

    Args:
        s1_structure (|Structure|):
            The template structure.
        s2_structure (|Structure|):
            The structure to reorder, to match ``s1_structure``.
        threshold (float):
            If the distance between a pair of matched sites is larger than
            this value in Å, then a warning will be thrown. Default is 5.0 Å.

    Returns:
        Structure:
            ``s2_structure`` reordered to match ``s1_structure``.
    """
    # This function was previously used to ensure correct site matching when pulling site potentials for
    # the eFNV Kumagai correction, though no longer used for this purpose. If threshold is set to a low
    # value, it will raise a warning if there is a large site displacement detected.
    if len(s2_structure) != len(s1_structure):
        raise ValueError("Structure reordering not possible, structures have different number of sites.")

    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mappings(s1_structure, s2_structure, threshold=threshold)
    mapping = sorted(mapping, key=lambda x: cast("int", x[1]))  # sort by s1 index (to match s1 ordering)

    # Reorder s2_structure so that it matches the ordering of s1_structure
    reordered_sites = [s2_structure[mapping_tuple[-1]] for mapping_tuple in mapping]

    # Avoid warning about selective_dynamics properties (can happen if user explicitly set selective
    # dynamics flags (e.g. "T T T" in a VASP POSCAR) for the bulk):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Not all sites have property")
        return Structure.from_sites(reordered_sites)


def get_dimer_bonds(structure: Structure, rtol: float = 1.05) -> dict[str, list[float]]:
    """
    Get a dictionary of all homoionic (dimer) bonds in the structure.

    This function uses the ``get_homoionic_bonds`` and
    ``get_dimer_bond_length`` functions from ``shakenbreak`` to identify dimer
    bonds in the structure (where any pair of atoms of the same element with
    distance < ``rtol * get_dimer_bond_length(elt, elt)`` are considered a
    dimer bond), returning a dictionary of the site names and the dimer bond
    length.

    Args:
        structure (|Structure|): The structure to get the dimer bond lengths for.
        rtol (float):
            The relative tolerance to use for classifying bonds as dimer bonds,
            where distances < ``rtol * get_dimer_bond_length(elt, elt)`` are
            considered dimer bonds. Default is 1.05.

    Returns:
        dict[str, list[float]]:
            A dictionary of element names with values being sub-dictionaries of
            site names and their homoionic neighbours and distances (in Å)
            which are classified as dimer bonds.
            (e.g. {'O': {'O(1)': {'O(3)': '1.44 Å'}}})
    """
    from shakenbreak.analysis import get_homoionic_bonds
    from shakenbreak.distortions import get_dimer_bond_length

    dimer_bond_dict = {
        str(elt): get_homoionic_bonds(
            structure=structure,
            elements=str(elt),
            radius=rtol * get_dimer_bond_length(elt, elt),
            verbose=False,
        )
        for elt in structure.composition.elements
    }
    return {k: v for k, v in dimer_bond_dict.items() if v}
