"""
Calculator-agnostic defect identification and site-mapping utilities
(site/structure matching between bulk and defect supercells, defect type
identification etc.).
"""

import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core.structure import Composition, Lattice, PeriodicSite, SiteCollection, Structure
from pymatgen.core.structure_matcher import get_linear_assignment_solution, pbc_shortest_vectors
from pymatgen.optimization.neighbors import find_points_in_spheres
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
    rms: bool = False,
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
            length in Å).
            If ``None`` (default), we assume a single point defect and return
            the site of largest mismatch.
        abs_tol (bool):
            Whether to use ``site_tol`` as an absolute distance tolerance (in
            Å) instead of a fractional tolerance (in terms of the shortest bond
            length in the structure). Default is ``False``.
        use_oxi_states (bool):
            Whether to use the oxidation states of the sites in the bulk and
            defect structures when considering matching sites (such that e.g.
            ``Fe3+`` and ``Fe2+`` would be considered different species).
            Default is ``False``.
        rms (bool):
            Site mapping (using linear assignment) -- used to determine defect
            sites -- will be that which minimises either the summed `squared`
            distances (i.e. the RMS displacement; if ``rms`` is ``True``) or
            the summed distances (if ``False``, default) between all paired
            sites.

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

    defect_type, comp_diff = get_defect_type_and_composition_diff(
        defect_composition, bulk_composition, _parameter_order_warn=False
    )  # internal call with correct (defect, bulk) ordering; don't warn

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
            species_distances = bulk_supercell.lattice.get_all_distances(
                bulk_supercell.frac_coords[bulk_species_indices], bulk_supercell.frac_coords
            )
            species_min_dist = max(species_distances[np.nonzero(species_distances)].min(), 1)
            site_dist_tol = site_tol if site_tol is None or abs_tol else site_tol * species_min_dist

        site_mapping = _get_site_mapping_from_coords_and_indices(
            defect_species_fcoords,
            bulk_species_fcoords,
            lattice=bulk_supercell.lattice,
            s1_indices=defect_species_indices,
            s2_indices=bulk_species_indices,
            rms=rms,
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

    # Sanity checks; reasonable number of defects detected, and matching lattice definitions:
    n_defect_sites = max(len(missing_bulk_site_indices), len(additional_defect_site_indices))
    if not n_defect_sites:
        warnings.warn(
            f"No defect sites could be identified from the defect and bulk supercells (composition "
            f"difference: {comp_diff}) with ``site_tol`` = {site_tol}, suggesting that ``site_tol`` is "
            f"too large, or that the defect and bulk supercells are equivalent."
        )
        return defect_type, missing_bulk_site_indices, additional_defect_site_indices

    if n_defect_sites > 0.1 * len(bulk_supercell):  # >10% of sites flagged; likely spurious
        warnings.warn(
            f"{n_defect_sites} sites were identified as defect sites (more than 10% of the "
            f"{len(bulk_supercell)} sites in the bulk supercell) with ``site_tol`` = {site_tol}, "
            f"suggesting that ``site_tol`` is too small, or that the defect and bulk supercells do not "
            f"match -- unless this is expected (e.g. alloying / defect-ordering)."
        )

    # check lattice definitions; use a detected site as the reference position for the atom-mapping
    # diagnostic; any site in the cell will do if the supercells globally mismatch
    check_atom_mapping_far_from_defect(
        defect_supercell,
        bulk_supercell,
        defect_supercell[additional_defect_site_indices[0]].frac_coords
        if additional_defect_site_indices
        else bulk_supercell[missing_bulk_site_indices[0]].frac_coords,
    )  # throws informative warning about global site (lattice definition) mismatch
    # Note: This function checks (and warns, if necessary) for large mismatches between defect and bulk
    # supercells, where a common case is a symmetry-equivalent bulk supercell but with a different
    # basis/definition for the atomic positions (discussion:
    # doped.readthedocs.io/en/latest/Troubleshooting.html#mis-matching-bulk-and-defect-supercells )
    # In theory, we could use orient_s2_like_s1 with allow_subset to shift the defect cell to match the
    # (different definition) bulk cell, tracking the site matches, and accounting for the site matches
    # properly with the charge corrections. But, beyond being a lot of work to allow the unnecessary (and
    # usually easily fixed) case of mismatching supercells, which can also lead to other issues, it would
    # require different definitions of 'defect supercell sites' (e.g. for a vacancy with a mismatching
    # supercell definition, the supercell site should be the exact atom site in the bulk supercell, but
    # this is now entirely different from the defect supercell). Also, the choice of matching orientation
    # for the bulk supercell (and thus defect site) can become arbitrary in these situations, where there
    # are many possible defect cell translations etc which match the bulk cell... Also difficulties with
    # handling this for finite-size corrections.

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

    If the closest matching site in ``structure`` is > ``tol`` Å (0.5 Å by
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
    supercell with the `"unrelaxed"` defect site.

    The unrelaxed defect site corresponds to the vacancy/substitution site in
    the pristine (bulk) supercell for vacancies/substitutions, and the
    `relaxed` interstitial site for interstitials (as the assignment of their
    initial site is ambiguous).

    Args:
        defect_supercell (Structure):
            The defect structure.
        bulk_supercell (Structure):
            The bulk structure.
        defect_site_idx (int):
            The index of the defect site to use in the `"unreleaxed"` defect
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
            The `"unrelaxed"` defect structure.
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


def _guess_initial_defect_structure(
    unrelaxed_defect_structure: Structure,
    bulk_supercell: Structure,
    defect_type: str,
    defect_site: PeriodicSite,
    defect_site_in_bulk: PeriodicSite,
    defect_site_index: int | None,
) -> tuple[Structure, PeriodicSite]:
    """
    Guess the initial defect structure, corresponding to
    ``unrelaxed_defect_structure`` but with interstitials placed at the closest
    candidate interstitial site in the bulk supercell (based on default
    ``doped`` interstitial generation settings) to the relaxed interstitial
    site -- as this is likely the `initial` interstitial site.

    ``defect_site_in_bulk`` is updated to this guessed initial site if it is
    within 1 Å of the relaxed site, otherwise left unchanged (the relaxed
    site). For vacancies/substitutions, just returns a copy of
    ``unrelaxed_defect_structure`` and the unchanged ``defect_site_in_bulk``.

    Returns:
        tuple[Structure, PeriodicSite]:
            The guessed initial defect structure, and the (possibly updated)
            ``defect_site_in_bulk``.
    """
    guessed_initial_defect_structure = unrelaxed_defect_structure.copy()
    if defect_type != "interstitial":
        return guessed_initial_defect_structure, defect_site_in_bulk

    from doped.generation import get_interstitial_sites

    # get closest candidate interstitial site in bulk supercell (based on default interstitial gen
    # settings) to the relaxed interstitial site, as this is likely the _initial_ interstitial site
    int_site = guessed_initial_defect_structure.pop(defect_site_index)
    int_gen_kwargs: dict[str, Any] = {"min_dist": 0.5} if int_site.species_string == "H" else {}
    all_equiv_fpos = [  # all candidate interstitial frac coords in the bulk supercell
        fpos
        for *_, equiv_fpos in get_interstitial_sites(bulk_supercell, **int_gen_kwargs)
        for fpos in equiv_fpos
    ]
    closest_cand_int_fcoords = all_equiv_fpos[  # closest candidate interstitial frac coords
        np.argmin(bulk_supercell.lattice.get_all_distances(defect_site.frac_coords, all_equiv_fpos))
    ]
    guessed_initial_defect_structure.insert(
        defect_site_index,  # place defect at same position as in supercell calculation
        int_site.species_string,
        closest_cand_int_fcoords,
        coords_are_cartesian=False,
        validate_proximity=True,
    )
    # if guessed initial site is sufficiently close to the relaxed site, then use it as
    # "defect_site_in_bulk", otherwise use the relaxed site:
    if defect_site_in_bulk.distance_and_image_from_frac_coords(closest_cand_int_fcoords)[0] < 1:
        defect_site_in_bulk = guessed_initial_defect_structure[defect_site_index]

    return guessed_initial_defect_structure, defect_site_in_bulk


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
        if not (len(bulk_species_outside_near_ws_fcoords) and len(defect_species_outside_ws_fcoords)):
            continue  # no sites of this species outside the WS radius, skip

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
            f"Detected atoms far from the defect site (>{wigner_seitz_radius:.2f} Å) with major "
            f"displacements (>{displacement_tol} Å) in the defect supercell. This likely indicates a "
            f"mismatch between the bulk and defect supercell definitions (-> see troubleshooting docs) or "
            f"an unconverged supercell size, both of which could cause errors in parsing. The mean (or at "
            f"least {fraction_tol:.0%}) of displacements of the following species, at sites far from the "
            f"determined defect position, is >{displacement_tol} Å: {list(large_disps_outside_ws.keys())}"
        )
        if warning == "verbose":
            message += f", with displacements (Å): {large_disps_outside_ws}"
        if warning:
            warnings.warn(message)

        return False

    return True


_PBC = np.array([1, 1, 1], dtype=np.int64)  # for ``find_points_in_spheres``, takes ints not bools for PBC


def _min_separation(coords: np.ndarray, lattice: Lattice) -> float | None:
    """
    Get the minimum separation (under PBC) between the given fractional
    coordinates.

    Cached for efficiency. Returns ``None`` if no non-coincident neighbour is
    found within ~twice the mean site spacing -- geometrically impossible (by
    sphere packing) unless sites are stacked at identical positions (degenerate
    input).
    """
    # keyed on the raw coordinate bytes, which is both exact (no hash collisions) and much cheaper than
    # tuple conversion for arrays this size (~0.03 vs ~1.7 ms for 7k sites):
    return _cached_min_separation(np.asarray(coords, dtype=float).tobytes(), lattice)


@lru_cache(maxsize=int(1e2))  # maxsize on the order of 20 Mb for typical (large) supercells
def _cached_min_separation(coords_bytes: bytes, lattice: Lattice) -> float | None:
    coords = np.frombuffer(coords_bytes, dtype=float).reshape(-1, 3)
    cart = lattice.get_cartesian_coords(coords)  # C-contiguous float64 arrays, as required for:
    *_, self_dists = find_points_in_spheres(  # <- find_points_in_spheres
        all_coords=cart,
        center_coords=cart,
        r=2 * (lattice.volume / len(coords)) ** (1 / 3),  # ~2x mean site spacing; always sufficient
        pbc=_PBC,
        lattice=lattice.matrix,
        tol=1e-8,
    )
    separations = self_dists[self_dists > 1e-8]  # excluding each site's distance to itself
    return float(separations.min()) if separations.size else None


def _nearest_neighbour_site_mapping(
    subset_coords: np.ndarray, superset_coords: np.ndarray, lattice: Lattice | None, r: float | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Get the site mapping from a neighbour list search, rather than the global
    distance matrix, or ``None`` if this is not provably the exact solution.

    Matching each ``subset`` site to its nearest ``superset`` site (i.e.
    allowing duplicates) minimises each term of the cost independently, and so
    lower-bounds the cost of `any` assignment. If those nearest matches happen
    to all be distinct (no duplicate matches), then this assignment is the
    linear assignment solution -- whether for linear or squared distance cost
    functions. This is almost always the case (for defect supercells), and
    avoids both the ``O(N*M)`` distance matrix and ``O(N^3)`` assignment,
    giving large speedups for big (>~1,000 atom) supercells.

    Only the search radius is a heuristic here (and never a correctness
    concern); by default, the minimum separation within ``superset`` is used,
    which is ample for any physically-reasonable displacement, and ``None`` is
    returned if some ``subset`` site has no ``superset`` site within it (so its
    nearest is unknown).

    Args:
        subset_coords (np.ndarray[float]):
            Fractional coordinates of the smaller set of sites.
        superset_coords (np.ndarray[float]):
            Fractional coordinates of the larger set of sites.
        lattice (Lattice | None):
            The lattice of the structures, for PBC distances. If ``None``
            (i.e. Cartesian coordinates, no PBC), the neighbour search is
            skipped (returning ``None``).
        r (float | None):
            Search radius in Å. Only affects the acceptance rate and not
            correctness. Default is the minimum separation between ``superset``
            sites.

    Returns:
        tuple | None:
            The matched distances and matching ``superset`` indices, for each
            ``subset`` site, or ``None`` if this could not be shown to be the
            exact solution.
    """
    if lattice is None or (r := r or _min_separation(superset_coords, lattice)) is None:
        return None

    # anything found within ``r`` is that site's true nearest neighbour (nothing closer can lie outside the
    # search radius), so the conditions reduce to each subset site having found a match, and no two subset
    # sites having matched the same superset site:
    centres, neighbours, _offsets, dists = find_points_in_spheres(
        all_coords=lattice.get_cartesian_coords(superset_coords),  # = np.dot; C-contiguous f64, required
        center_coords=lattice.get_cartesian_coords(subset_coords),
        r=r,
        pbc=_PBC,
        lattice=lattice.matrix,
        tol=1e-8,
    )
    # sort by distance within each subset site, then take the first entry per site (= its nearest):
    order = np.lexsort((dists, centres))
    nearest = order[np.unique(centres[order], return_index=True)[1]]
    matched_dists, site_matches = dists[nearest], neighbours[nearest]

    if len(site_matches) == len(subset_coords) == len(np.unique(site_matches)):
        return matched_dists, site_matches
    return None  # some site with no match within ``r``, or two sites matched to the same superset site


def _get_site_mapping_from_coords_and_indices(
    s1_coords: ArrayLike,
    s2_coords: ArrayLike,
    s1_indices: np.ndarray | None = None,
    s2_indices: np.ndarray | None = None,
    lattice: Lattice | None = None,
    rms: bool = False,
) -> list[tuple[float | None, int | None, int | None]]:
    """
    Get the site mapping between two sets of coordinates and indices, based on
    the shortest distances between sites.

    Args:
        s1_coords (np.ndarray[float]):
            The coordinates of the first set of sites; fractional if
            ``lattice`` is given, otherwise Cartesian.
        s2_coords (np.ndarray[float]):
            The coordinates of the second set of sites; fractional if
            ``lattice`` is given, otherwise Cartesian.
        s1_indices (np.ndarray[int] | None):
            The indices of the first set of sites. If ``None``, assumed to be
            the range of the number of sites in ``s1_coords``.
        s2_indices (np.ndarray[int] | None):
            The indices of the second set of sites. If ``None``, assumed to be
            the range of the number of sites in ``s2_coords``.
        lattice (Lattice | None):
            The lattice of the structures, for which the input coordinates are
            fractional and distances are computed under PBC. If ``None``
            (default), the inputs are instead taken as Cartesian coordinates
            and distances are computed directly, with no consideration of PBC.
        rms (bool):
            The returned site mapping (using linear assignment) will be that
            which minimises either the summed `squared` distances (i.e. the RMS
            displacement; if ``rms`` is ``True``) or the summed distances (if
            ``False``, default) between all paired sites.

    Returns:
        list:
            A list of lists containing the distance, index from ``s1_indices``
            and index from ``s2_indices`` for each matched site.
    """
    s1_coords = np.asarray(s1_coords)
    s2_coords = np.asarray(s2_coords)
    if s1_indices is None:
        s1_indices = np.arange(len(s1_coords))
    if s2_indices is None:
        s2_indices = np.arange(len(s2_coords))

    for empty_coords, indices, tuple_idx in [  # handle case of empty input coords
        (s1_coords, s2_indices, 2),
        (s2_coords, s1_indices, 1),
    ]:
        if empty_coords.size == 0:
            if indices is None:
                return [(None, None, None)]
            return [
                (None, None if tuple_idx == 2 else int(i), None if tuple_idx == 1 else int(i))
                for i in indices
            ]

    s1_is_subset = len(s1_coords) < len(s2_coords)
    subset_coords, subset_indices = (s1_coords, s1_indices) if s1_is_subset else (s2_coords, s2_indices)
    superset_coords, superset_indices = (
        (s2_coords, s2_indices) if s1_is_subset else (s1_coords, s1_indices)
    )
    # try the (much faster) neighbour list search first; only valid under PBC and when it can prove itself
    # exact, otherwise fall back to the global distance matrix and linear assignment:
    nn_mapping = _nearest_neighbour_site_mapping(subset_coords, superset_coords, lattice)
    if nn_mapping is not None:
        matched_dists, site_matches = nn_mapping
    else:
        dists = (  # ``pbc_shortest_vectors`` only gives squared distances, so sqrt for matched dists
            all_distances(subset_coords, superset_coords)
            if lattice is None
            else np.sqrt(pbc_shortest_vectors(lattice, subset_coords, superset_coords, return_d2=True)[1])
        )
        site_matches, _ = get_linear_assignment_solution(dists**2 if rms else dists)
        matched_dists = dists[np.arange(len(site_matches)), site_matches]

    # convert to native Python types:
    matched_dists = matched_dists.tolist()
    site_matches = site_matches.tolist()
    subset_indices = np.asarray(subset_indices).tolist()
    superset_indices = np.asarray(superset_indices).tolist()

    site_mapping = [  # site_matches -> matching superset indices, of len(subset)
        (matched_dists[i], subset_indices[i], superset_indices[j]) for i, j in enumerate(site_matches)
    ]
    for missing_index in set(range(len(superset_coords))) - set(site_matches):
        site_mapping.append((None, None, superset_indices[missing_index]))  # unmatched sites

    if not s1_is_subset:  # swap tuple order, to match (dist, s1_index, s2_index)
        site_mapping = [(dist, index2, index1) for dist, index1, index2 in site_mapping]

    return site_mapping


def find_missing_idx(
    frac_coords1: list | np.ndarray,
    frac_coords2: list | np.ndarray,
    lattice: Lattice,
):
    """
    Find the missing/outlier index between two sets of fractional coordinates
    (differing in size by 1), by grouping the coordinates based on the minimum
    distances between coordinates or, if that doesn't give a unique match, the
    site combination that gives the minimum summed squared distances between
    paired sites.

    The index returned is the index of the missing/outlier coordinate in the
    larger set of coordinates.

    Args:
        frac_coords1 (list | np.ndarray):
            First set of fractional coordinates.
        frac_coords2 (list | np.ndarray):
            Second set of fractional coordinates.
        lattice (|Lattice|):
            The lattice object to use with the fractional coordinates.
    """
    # the unmatched entry (i.e. that with a ``dist`` of ``None``) has exactly one non-``None`` index,
    # which is that of the missing/outlier coordinate in the larger set of coordinates:
    return next(
        idx1 if idx2 is None else idx2
        for dist, idx1, idx2 in _get_site_mapping_from_coords_and_indices(
            frac_coords1, frac_coords2, lattice=lattice, rms=True
        )
        if dist is None
    )


def get_site_mappings(
    struct1: Structure,
    struct2: Structure,
    species: SpeciesLike | None = None,
    allow_duplicates: bool = False,
    threshold: float = 2.0,
    anonymous: bool = False,
    ignored_species: list[str] | None = None,
    frac_coords: bool = True,
    rms: bool = False,
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
            The first structure, for which the mappings to sites in ``struct2``
            will be returned in the order of its sites.
        struct2 (|Structure|):
            The second structure, for which to determine the mappings to sites
            in ``struct1``.
        species (str):
            If provided, only sites of this species will be considered when
            matching sites. Default is ``None`` (all species).
        allow_duplicates (bool):
            If ``True``, allow multiple sites in ``struct1`` to be matched to
            the same site in ``struct2``. Default is ``False``.
        threshold (float):
            If the distance between a pair of matched sites is larger than
            this, then a warning will be thrown. Default is 2.0 Å.
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
        rms (bool):
            The returned site mapping (using linear assignment -- only
            applicable when ``allow_duplicates`` is ``False``) will be that
            which minimises either the summed `squared` distances (i.e. the RMS
            displacement; if ``rms`` is ``True``) or the summed distances (if
            ``False``, default) between all paired sites.

    Returns:
        list:
            A list of lists containing the distance, index in ``struct1`` and
            index in ``struct2`` for each matched site.
    """

    def get_coords(site: PeriodicSite):
        return list(site.frac_coords) if frac_coords else list(site.coords)

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
        s1_species_indices = [
            i for i, site in enumerate(struct1) if (site.specie.symbol == s1_species_symbol or anonymous)
        ]
        s1_coords = [get_coords(struct1[i]) for i in s1_species_indices]
        s2_species_indices = [
            i for i, site in enumerate(struct2) if (site.specie.symbol == s1_species_symbol or anonymous)
        ]
        s2_coords = [get_coords(struct2[i]) for i in s2_species_indices]

        if not s2_coords:  # no sites of this species in struct2
            min_dist_with_index.extend((None, index, None) for index in s1_species_indices)
            continue

        # mapping entries are (dist, species-local struct1 index, species-local struct2 index):
        if allow_duplicates:  # each input independently picks its closest template
            dmat = (
                struct1.lattice.get_all_distances(s1_coords, s2_coords)
                if frac_coords
                else all_distances(s1_coords, s2_coords)
            )
            mapping: list[tuple[float | None, int | None, int | None]] = [
                (float(row.min()), i, int(row.argmin())) for i, row in enumerate(dmat)
            ]

        else:  # linear assignment, for order-independent optimal matching
            mapping = _get_site_mapping_from_coords_and_indices(
                s1_coords,
                s2_coords,
                lattice=struct1.lattice if frac_coords else None,
                rms=rms,
            )

        # struct2 sites with no matching struct1 site aren't reported by this function, so are dropped:
        matched_s1 = [(dist, i, j) for dist, i, j in mapping if i is not None]

        for dist, i, j in sorted(matched_s1, key=lambda entry: entry[1]):  # keep ``struct1`` ordering
            index = s1_species_indices[i]  # map species-local indices to global ``struct1/2`` indices
            s2_index = s2_species_indices[j] if j is not None else None
            if dist is not None and dist > threshold:
                warnings.warn(
                    f"Large site displacement {dist:.2f} Å detected when matching atomic sites: "
                    f"{struct1[index]} -> {struct2[s2_index]}."
                )
            min_dist_with_index.append((dist, index, s2_index))

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
            this value in Å, then a warning will be thrown. Default is 5.0 Å.

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
            site names and their homoionic neighbours and distances (in Å)
            which are classified as dimer bonds.
            (e.g. {'O': {'O(1)': {'O(3)': '1.44 Å'}}})
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
