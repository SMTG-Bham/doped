"""
Utility functions to re-generate a relaxed defect structure in a different
supercell.

The code in this sub-module is still in development! (TODO)
"""

import math
import warnings
from collections import Counter
from functools import lru_cache
from itertools import combinations, product
from typing import Optional, Union

import numpy as np
from pymatgen.core.composition import Composition, Species, defaultdict
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from tqdm import tqdm

from doped.core import DefectEntry
from doped.utils.configurations import orient_s2_like_s1
from doped.utils.efficiency import _Comp__eq__
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    get_coords_and_idx_of_species,
    get_defect_type_and_composition_diff,
    get_wigner_seitz_radius,
)
from doped.utils.symmetry import (
    _round_floats,
    apply_symm_op_to_site,
    apply_symm_op_to_struct,
    get_sga,
    translate_structure,
)


def get_defect_in_supercell(
    defect_entry: DefectEntry,
    target_supercell: Structure,
    target_frac_coords: Union[np.ndarray[float], list[float], bool] = True,
    edge_tol: float = 1,
) -> Structure:
    """
    Re-generate a relaxed defect structure in a different supercell.

    This function takes the relaxed defect structure of the input ``DefectEntry``
    (from ``DefectEntry.defect_supercell``) and re-generates it in the
    ``target_supercell`` structure, and the closest possible position to
    ``target_frac_coords`` (if provided, else closest to centre = [0.5, 0.5, 0.5]).

    Briefly, this function works by:

    - Translating the defect site to the centre of the original supercell.
    - Identifying a super-supercell which fully encompasses the target supercell
      (regardless of orientation).
    - Generate this super-supercell, using one copy of the original defect
      supercell (``DefectEntry.defect_supercell``), and the rest of the sites
      (outside of the original defect supercell box, with the defect translated
      to the centre) are populated using the bulk supercell
      (``DefectEntry.bulk_supercell``).
    - Translate the defect site in this super-supercell to the Cartesian coordinates
      of the centre of ``target_supercell``, then stencil out all sites in the
      ``target_supercell`` portion of the super-supercell, accounting for possible
      site displacements in the relaxed defect supercell (e.g. if ``target_supercell``
      has a different shape and does not fully encompass the original defect
      supercell).
      This is done by scanning over possible combinations of sites near the boundary
      regions of the ``target_supercell`` portion, and identifying the combination
      which maximises the minimum inter-atomic distance in the new supercell (i.e. the
      most bulk-like arrangement).
    - Re-orient this new stenciled supercell to match the orientation and site
      positions of ``target_supercell``.
    - If ``target_frac_coords`` is not ``False``, scan over all symmetry operations
      of ``target_supercell`` and apply that which places the defect site closest
      to ``target_frac_coords``.

    Args:
        defect_entry (DefectEntry):
            A ``DefectEntry`` object for which to re-generate the relaxed
            structure (taken from ``DefectEntry.defect_supercell``) in the
            ``target_supercell`` lattice.
        target_supercell (Structure):
            The supercell structure to re-generate the relaxed defect
            structure in.
        target_frac_coords (Union[np.ndarray[float], list[float], bool]):
            The fractional coordinates to target for defect placement in the
            new supercell. If just set to ``True`` (default), will try to place
            the defect nearest to the centre of the superset cell (i.e.
            ``target_frac_coords = [0.5, 0.5, 0.5]``), as is default in ``doped``
            defect generation. Note that defect placement is harder in this
            case than in generation with ``DefectsGenerator``, as we are not
            starting from primitive cells and we are working with relaxed
            geometries.
        edge_tol (float):
            A tolerance (in Angstrom) for site displacements at the edge of the
            ``target_supercell`` supercell, when determining the best match of
            sites to stencil out in the new supercell.
            Default is 1 Angstrom, and then this is sequentially increased up
            to 4.5 Angstrom if the initial scan fails.

    Returns:
        Structure: The relaxed defect structure in the new supercell.
    """
    # Note to self; using Pycharm breakpoints throughout is likely easiest way to debug these functions
    # TODO: Function needs to be cleaned up more, modularise into more functions, update docstrings etc
    # TODO: Tests!! (At least one of each defect, Se good test case, then at least one or two with
    #  >unary compositions and extrinsic substitution/interstitial)

    pbar = tqdm(
        total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
    )  # tqdm progress bar. 100% is completion
    pbar.set_description("Getting super-supercell (relaxed defect + bulk sites)")

    try:
        orig_supercell = _get_defect_supercell(defect_entry)
        orig_min_dist = min_dist(orig_supercell)
        orig_bulk_supercell = _get_bulk_supercell(defect_entry)
        orig_defect_frac_coords = defect_entry.sc_defect_frac_coords
        target_supercell = target_supercell.copy()
        bulk_min_bond_length = min_dist(orig_bulk_supercell)
        if target_frac_coords is True:
            target_frac_coords = [0.5, 0.5, 0.5]

        # ensure no oxidation states (for easy composition matching later)
        for struct in [orig_supercell, orig_bulk_supercell, target_supercell]:
            struct.remove_oxidation_states()

        # first translate both orig supercells to put defect in the middle, to aid initial stenciling:
        orig_supercell = translate_structure(
            orig_supercell, np.array([0.5, 0.5, 0.5]) - orig_defect_frac_coords, frac_coords=True
        )  # checked, working as expected
        orig_bulk_supercell = translate_structure(
            orig_bulk_supercell, np.array([0.5, 0.5, 0.5]) - orig_defect_frac_coords, frac_coords=True
        )

        # get big_supercell, which is expanded version of orig_supercell so that _each_ lattice vector is
        # now bigger than the _largest_ lattice vector in target_supercell (so that target_supercell is
        # fully encompassed by big_supercell):
        supercell_matrix, big_supercell, big_supercell_with_X, orig_supercell_with_X = (
            _get_superset_matrix_and_supercells(orig_defect_frac_coords, orig_supercell, target_supercell)
        )

        # this big_supercell is with the defect now repeated in it, but we want it with just one defect,
        # so only take the first repeated defect supercell, and then the rest of the sites from the
        # expanded bulk supercell:
        # only keep atoms in big supercell which are within the original supercell bounds:
        big_defect_supercell = _get_matching_sites_from_s1_then_s2(
            orig_supercell_with_X,
            big_supercell_with_X,
            orig_bulk_supercell * supercell_matrix,
            orig_min_dist,
        )
        _check_min_dist(big_defect_supercell, orig_min_dist, warning=False, ignored_species=["X"])
        big_supercell_defect_site = next(
            site for site in big_defect_supercell.sites if site.specie.symbol == "X"
        )
        # translate structure to put defect at the centre:
        big_defect_supercell = translate_structure(
            big_defect_supercell,
            np.array([0.5, 0.5, 0.5]) - big_supercell_defect_site.frac_coords,
            frac_coords=True,
        )
        big_supercell_defect_site = next(
            site for site in big_defect_supercell.sites if site.specie.symbol == "X"
        )

        pbar.update(20)  # 20% of progress bar
        pbar.set_description("Getting sites in border region")

        # get Cartesian translation to move the defect site to centre of the target supercell:
        translation_from_big_defect_site_to_target_middle = (
            target_supercell.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
            - big_supercell_defect_site.coords
        )
        # translate big supercell to place defect site at the cartesian coordinates matching the centre of
        # the new supercell:
        big_defect_supercell = translate_structure(
            big_defect_supercell,
            translation_from_big_defect_site_to_target_middle,
            frac_coords=False,
            to_unit_cell=False,
        )

        # get all atoms in big supercell within the cartesian bounds of the target_supercell supercell:
        # we need to ensure the same number of sites, and that the sites we choose are appropriate for the
        # new supercell (i.e. that if we have e.g. an in-plane contraction, we don't take duplicate atoms
        # that then correspond to tiny inter-atomic distances in the new supercell due to imperfect
        # stenciling under PBC -- so we can't just take the atoms that are closest to the defect). So,
        # scan over possible choices of atoms to include, and take that which maximises the _minimum_
        # inter-atomic distance in the new supercell, when accounting for PBCs:
        # sequentially increase edge_tol by half an angstrom, up to 4.5 Å, until match is found:
        while edge_tol <= 4.5:
            try:
                target_composition = (
                    target_supercell.composition  # this is bulk, need to account for defect:
                    + orig_supercell.composition
                    - orig_bulk_supercell.composition
                )
                (
                    def_new_supercell_sites,
                    def_new_supercell_sites_to_check_in_target,
                    candidate_sites_in_target,
                    combo_composition,
                ) = _get_candidate_supercell_sites(
                    big_defect_supercell, target_supercell, edge_tol, target_composition
                )
                num_sites_up_for_grabs = int(
                    sum(combo_composition.values())
                )  # number of sites to be placed
                candidate_sites_in_target = _remove_overlapping_sites(
                    candidate_sites_in_target,
                    def_new_supercell_sites_to_check_in_target,
                    big_supercell_defect_site,
                    bulk_min_bond_length,
                    pbar,
                )

                if len(candidate_sites_in_target) < num_sites_up_for_grabs:
                    raise RuntimeError(
                        f"Too little candidate sites ({len(candidate_sites_in_target)}) to match target "
                        f"composition ({num_sites_up_for_grabs} sites to be placed). Aborting."
                    )
                num_possible_combos = math.comb(len(candidate_sites_in_target), num_sites_up_for_grabs)
                if num_possible_combos > 1e10:
                    raise RuntimeError(
                        "Too many possible combinations to check. Code will take forever, aborting."
                    )

                pbar.set_description(
                    f"Calculating best match (edge_tol = {edge_tol} Å, possible "
                    f"combos = {num_possible_combos})"
                )  # 40% of pbar
                species_symbols = [site.specie.symbol for site in candidate_sites_in_target]
                min_interatomic_distances_tuple_combo_dict = {}
                idx_combos = list(
                    combinations(range(len(candidate_sites_in_target)), num_sites_up_for_grabs)
                )

                if idx_combos and idx_combos != [()]:
                    for idx_combo in idx_combos:
                        comp = _cached_Comp_init(
                            Hashabledict(Counter([species_symbols[i] for i in idx_combo]))
                        )
                        if _Comp__eq__(comp, combo_composition):
                            # could early break cases where the distances are too small? if a bottleneck,
                            # currently not. And/or could loop over subsets of each possible combo first,
                            # culling any which break this
                            combo_sites_in_superset = [candidate_sites_in_target[i] for i in idx_combo]
                            fake_candidate_struct_sites = (
                                def_new_supercell_sites_to_check_in_target + combo_sites_in_superset
                            )
                            frac_coords = [site.frac_coords for site in fake_candidate_struct_sites]
                            distance_matrix = target_supercell.lattice.get_all_distances(
                                frac_coords, frac_coords
                            )
                            sorted_distances = np.sort(distance_matrix.flatten())
                            min_interatomic_distances_tuple_combo_dict[
                                tuple(sorted_distances[len(frac_coords) : len(frac_coords) + 10])
                                # take the 10 smallest distances, in case defect site is smallest distance
                            ] = [candidate_sites_in_target[i] for i in idx_combo]

                    # get the candidate structure with the largest minimum interatomic distance:
                    min_interatomic_distances_tuple_combo_list = sorted(
                        min_interatomic_distances_tuple_combo_dict.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                    new_supercell_sites = def_new_supercell_sites + list(
                        min_interatomic_distances_tuple_combo_list[0][1]
                    )
                else:
                    new_supercell_sites = def_new_supercell_sites

                new_supercell = Structure(
                    lattice=target_supercell.lattice,
                    species=[site.specie for site in new_supercell_sites],
                    coords=[site.coords for site in new_supercell_sites],
                    coords_are_cartesian=True,
                    to_unit_cell=True,
                )
                new_supercell_w_defect_comp = new_supercell.copy()
                new_supercell_w_defect_comp.remove_species("X")
                # raise RuntimeError and dynamically increase edge_tol if resulting min_dist too small:
                _check_min_dist(new_supercell_w_defect_comp, orig_min_dist, warning=False)

                try:
                    defect_type, comp_diff = get_defect_type_and_composition_diff(
                        target_supercell, new_supercell_w_defect_comp
                    )
                    break  # match found!
                except ValueError as e:
                    raise RuntimeError("Incorrect defect cell obtained. Aborting.") from e

            except RuntimeError as e:
                edge_tol += 0.5
                if edge_tol > 4.5:
                    raise e

                pbar.n = 20  # decrease pbar progress back to 20%
                pbar.refresh()
                pbar.set_description(f"Trying edge_tol = {edge_tol} Å")
                continue

        pbar.update(15)  # 55% of progress bar
        pbar.set_description("Ensuring matching orientation w/target_supercell")

        # now we just do get s2 like s1 to get the orientation right:
        defect_site = next(site for site in new_supercell if site.specie.symbol == "X")

        # Note: this function is typically the main bottleneck in this workflow. We have already
        # optimised the underlying ``StructureMatcher`` workflow in many ways (caching,
        # fast structure/site/composition comparisons, skipping comparison of defect neighbourhood to
        # reduce requisite ``stol`` etc; being many orders of magnitude faster than the base
        # ``pymatgen`` ``StructureMatcher``), however the ``_cart_dists()`` function call is
        # still quite expensive, especially with large structures with significant noise in the atomic
        # positions...
        new_supercell_w_defect_neighbours_as_X = _convert_defect_neighbours_to_X(
            new_supercell, defect_site.frac_coords, coords_are_cartesian=False
        )
        oriented_new_supercell_w_defect_neighbours_as_X = orient_s2_like_s1(
            target_supercell,
            new_supercell_w_defect_neighbours_as_X,
            verbose=False,
            ignored_species=["X"],  # ignore X site
            allow_subset=True,  # allow defect supercell composition to differ from target
        )
        oriented_new_supercell = _convert_X_back_to_orig_species(
            oriented_new_supercell_w_defect_neighbours_as_X
        )
        oriented_new_defect_site = next(  # get defect site and remove X from sites
            oriented_new_supercell.pop(i)
            for i, site in enumerate(oriented_new_supercell.sites)
            if site.specie.symbol == "X"
        )

        pbar.update(35)  # 90% of progress bar

        if target_frac_coords is not False:
            pbar.set_description("Placing defect closest to target_frac_coords")

            # translate to put defect at closest possible site to target_frac_coords
            sga = get_sga(target_supercell)
            symm_ops = sga.get_symmetry_operations()
            symm_op_pos_dict = {}
            for i, symm_op in enumerate(symm_ops):  # should check if frac or cartesian is faster
                symm_opped_site = apply_symm_op_to_site(
                    symm_op, oriented_new_defect_site, fractional=True, rotate_lattice=False
                )
                symm_op_pos_dict[i] = symm_opped_site.frac_coords

            # get symm_op which puts defect closest to target_frac_coords:
            closest_site = min(
                symm_op_pos_dict.items(), key=lambda x: np.linalg.norm(x[1] - target_frac_coords)
            )
            target_symm_op = symm_ops[closest_site[0]]

            # apply symm_op to structure
            oriented_new_supercell = apply_symm_op_to_struct(
                target_symm_op, oriented_new_supercell, fractional=True, rotate_lattice=False
            )  # reordered inputs in updated doped

        pbar.update(pbar.total - pbar.n)  # set to 100% of progress bar
        oriented_new_supercell = Structure.from_dict(_round_floats(oriented_new_supercell.as_dict()))
        oriented_new_supercell = Structure.from_sites(
            [site.to_unit_cell() for site in oriented_new_supercell]
        )

    except Exception as e:
        pbar.close()
        raise e

    finally:
        pbar.close()

    _check_min_dist(oriented_new_supercell, orig_min_dist)  # check interatomic distances are reasonable
    return oriented_new_supercell


def _convert_defect_neighbours_to_X(
    defect_supercell: Structure,
    defect_position: np.ndarray[float],
    coords_are_cartesian: bool = False,
    ws_radius_fraction: float = 0.5,
):
    """
    Convert all neighbouring sites of a defect site in a supercell, within half
    the Wigner-Seitz radius, to have their species as "X" (storing their
    original species in the site property dict as "orig_species").

    Intended to then be used to make structure-matching far more
    efficient, by ignoring the highly-perturbed defect neighbourhood
    (which requires larger ``stol`` values which grealy slow down
    structure-matching).

    Args:
        defect_supercell (Structure):
            The defect supercell to edit.
        defect_position (np.ndarray[float]):
            The coordinates of the defect site, either fractional
            or cartesian depending on ``coords_are_cartesian``.
        coords_are_cartesian (bool):
            Whether the defect position is in cartesian coordinates.
            Default is ``False`` (fractional coordinates).
        ws_radius_fraction (float):
            The fraction of the Wigner-Seitz radius to use as the
            cut-off distance for neighbouring sites to convert to
            species "X". Default is 0.5 (50%).

    Returns:
        Structure:
            The supercell structure with the defect site and all
            neighbouring sites converted to species X.
    """
    converted_defect_supercell = defect_supercell.copy()
    ws_radius = get_wigner_seitz_radius(defect_supercell)
    defect_frac_coords = (
        defect_position
        if not coords_are_cartesian
        else defect_supercell.lattice.get_fractional_coords(defect_position)
    )
    site_indices_to_convert = np.where(  # vectorised for fast computation
        defect_supercell.lattice.get_all_distances(
            defect_supercell.frac_coords, defect_frac_coords
        ).ravel()
        < np.max((ws_radius * ws_radius_fraction, 1))
    )[0]

    for i, orig_site in enumerate(defect_supercell):
        if i in site_indices_to_convert:
            converted_defect_supercell.replace(i, "X")
        # we set properties for all sites, because ``Structure.from_sites()`` in ``pymatgen``'s
        # ``StructureMatcher`` adds properties to all sites, which can then mess with site comparisons...
        converted_defect_supercell[i].properties["orig_species"] = orig_site.specie.symbol

    return converted_defect_supercell


def _convert_X_back_to_orig_species(converted_defect_supercell: Structure) -> Structure:
    """
    Convert all sites in a supercell with species "X" and "orig_species" in the
    site property dict back to their original species.

    Mainly intended just for internal ``doped`` usage, to convert back
    sites which had been converted to "X" for efficient structure-matching
    (see ``_convert_defect_neighbours_to_X``).

    Args:
        converted_defect_supercell (Structure):
            The defect supercell to convert back.

    Returns:
        Structure:
            The supercell structure with all sites with species "X"
            and "orig_species" in the site property dict converted back
            to their original species.
    """
    defect_supercell = converted_defect_supercell.copy()

    for i, site in enumerate(defect_supercell):
        if site.specie.symbol == "X" and site.properties.get("orig_species"):
            defect_supercell.replace(i, site.properties.pop("orig_species", site.specie.symbol))

    return defect_supercell


def min_dist(structure: Structure, ignored_species: Optional[list[str]] = None) -> float:
    """
    Return the minimum interatomic distance in a structure.

    Uses numpy vectorisation for fast computation.

    Args:
        structure (Structure):
            The structure to check.
        ignored_species (list[str]):
            A list of species symbols to ignore when calculating
            the minimum interatomic distance. Default is ``None``.

    Returns:
        float:
            The minimum interatomic distance in the structure.
    """
    if ignored_species is not None:
        structure = structure.copy()
        structure.remove_species(ignored_species)

    distances = structure.distance_matrix.flatten()
    return (  # fast vectorised evaluation of minimum distance
        0
        if len(np.nonzero(distances)[0]) < (len(distances) - structure.num_sites)
        else np.min(distances[np.nonzero(distances)])
    )


def _check_min_dist(
    structure: Structure,
    orig_min_dist: float = 5.0,
    warning: bool = True,
    ignored_species: Optional[list[str]] = None,
):
    """
    Helper function to check if the minimum interatomic distance in the
    provided ``structure`` are reasonable.

    Args:
        structure (Structure):
            The structure to check.
        orig_min_dist (float):
            The minimum interatomic distance in the original structure.
            If the minimum interatomic distance in the new structure is
            smaller than this, and smaller than a reasonable minimum
            distance (0.65 Å if H in structure, else 1.0 Å), a warning
            or error is raised.
            Default is 5.0 Å.
        warning (bool):
            Whether to raise a warning or an error if the minimum interatomic
            distance is too small. Default is ``True`` (warning).
        ignored_species (list[str]):
            A list of species symbols to ignore when calculating
            the minimum interatomic distance. Default is ``None``.
    """
    H_in_struct = any(site for site in structure.sites if site.specie.symbol == "H")
    reasonable_min_dist = 0.65 if H_in_struct else 1.0
    struct_min_dist = min_dist(structure, ignored_species)
    if struct_min_dist < min(orig_min_dist, reasonable_min_dist):
        message = (
            f"Generated structure has a minimum interatomic distance of {struct_min_dist:.2f} Å, smaller "
            f"than the original defect supercell ({orig_min_dist:.2f} Å), which may be unreasonable. "
            f"Please check if this minimum distance and structure make sense, and if not please report "
            f"this issue to the developers!"
        )
        if warning:
            warnings.warn(message)
        else:
            raise RuntimeError(message)


def _get_candidate_supercell_sites(big_defect_supercell, target_supercell, edge_tol, target_composition):
    """
    Get all atoms in ``big_defect_supercell`` which are within the Cartesian
    bounds of the ``target_supercell`` supercell.

    We need to ensure the same number of sites, and that the sites
    we choose are appropriate for the new supercell (i.e. that if
    we have e.g. an in-plane contraction, we don't take duplicate atoms
    that then correspond to tiny inter-atomic distances in the new
    supercell due to imperfect stenciling under PBC -- so we can't just
    take the atoms that are closest to the defect). So, this function
    determines the possible sites to include in the new supercell,
    the sites which are definitely in the target supercell, and the
    sites which are near the bordering regions of the target supercell
    and so may or may not be included (i.e. ``candidate_sites_in_target``),
    using the provided ``edge_tol``.

    Args:
        big_defect_supercell (Structure):
            The super-supercell with a single defect supercell and
            rest of the sites populated by the bulk supercell.
        target_supercell (Structure):
            The supercell structure to re-generate the relaxed defect
            structure in.
        edge_tol (float):
            A tolerance (in Angstrom) for site displacements at the edge of the
            ``target_supercell`` supercell, when determining the best match of
            sites to stencil out in the new supercell.
        target_composition (Composition):
            The composition of the target supercell.

    Returns:
        Tuple[list[PeriodicSite], list[PeriodicSite], list[PeriodicSite], Composition, int]:
            - ``def_new_supercell_sites``: List of sites in the super-supercell
              which are within the bounds of the target supercell (minus ``edge_tol``)
            - ``def_new_supercell_sites_to_check_in_target``: List of sites
              in the super-supercell which are near the bordering regions of
              the target supercell (within ``edge_tol*2`` of the target cell edge).
            - ``candidate_sites_in_target``: List of candidate sites in the
              target supercell to check if overlapping with each other or
              ``def_new_supercell_sites_to_check_in_target``.
            - ``combo_composition``: The composition we need to add to the
              target supercell.
    """
    # Note: This could possibly be made faster by reducing `edge_tol` when the `orig_supercell` is
    # fully encompassed by `target_supercell` (meaning there should be no defect-induced
    # displacements at the stenciled cell edges here), by getting the minimum encompassing cube
    # length (sphere diameter), and seeing if this is smaller than the largest sphere (diameter)
    # which can be inscribed in the target supercell
    # either way, this portion of the workflow is not a bottleneck (rather `get_orientation..` is)
    possible_new_supercell_sites = [
        site
        for site in big_defect_supercell.sites
        if is_within_frac_bounds(target_supercell.lattice, site.coords, tol=edge_tol)
    ]  # tolerance of 2 Angstrom displacement at cell edges
    def_new_supercell_sites = [
        site
        for site in possible_new_supercell_sites
        if is_within_frac_bounds(target_supercell.lattice, site.coords, tol=-edge_tol)
    ]  # at least 2 Angstrom inside cell edges
    def_new_supercell_sites_to_check_in_target = [
        PeriodicSite(site.specie, site.coords, lattice=target_supercell.lattice, coords_are_cartesian=True)
        for site in def_new_supercell_sites
        if not is_within_frac_bounds(target_supercell.lattice, site.coords, tol=-edge_tol * 2)
    ]
    candidate_sites_in_target = [
        PeriodicSite(site.specie, site.coords, lattice=target_supercell.lattice, coords_are_cartesian=True)
        for site in possible_new_supercell_sites
        if site not in def_new_supercell_sites
    ]
    # target additional composition:
    def_new_supercell_sites_struct = Structure.from_sites(def_new_supercell_sites)
    def_new_supercell_sites_struct.remove_species("X")
    def_new_supercell_sites_comp = def_new_supercell_sites_struct.composition
    combo_composition = target_composition - def_new_supercell_sites_comp  # composition we need to add
    # def_new_supercell_sites also has X in it

    return (
        def_new_supercell_sites,
        def_new_supercell_sites_to_check_in_target,
        candidate_sites_in_target,
        combo_composition,
    )


def _remove_overlapping_sites(
    candidate_sites_in_target: list[PeriodicSite],
    def_new_supercell_sites_to_check_in_target: list[PeriodicSite],
    big_supercell_defect_site: PeriodicSite,
    bulk_min_bond_length: float,
    pbar: tqdm = None,
) -> list[PeriodicSite]:
    """
    Remove sites in ``candidate_sites_in_target`` which overlap either with
    each other (within 50% of the bulk bond length; in which case the site
    closest to the defect coords is removed) or with sites in
    ``def_new_supercell_sites_to_check_in_target`` (within 50% of the bulk bond
    length).

    Args:
        candidate_sites_in_target (list[PeriodicSite]):
            List of candidate sites in the target supercell to check if
            overlapping with each other or
            ``def_new_supercell_sites_to_check_in_target``.
        def_new_supercell_sites_to_check_in_target (list[PeriodicSite]):
            List of sites that are in the target supercell but are near
            the bordering regions, so are used to check for overlapping.
        big_supercell_defect_site (PeriodicSite):
            The defect site in the super-supercell.
        bulk_min_bond_length (float):
            The minimum bond length in the bulk supercell.
        pbar (tqdm):
            ``tqdm`` progress bar object to update (for internal ``doped``
            usage). Default is ``None``.

    Returns:
        list[PeriodicSite]:
            The list of candidate sites in the target supercell which do
            not overlap with each other or with
            ``def_new_supercell_sites_to_check_in_target``.
    """
    if not candidate_sites_in_target:
        if pbar is not None:
            pbar.update(20)
        return candidate_sites_in_target

    # scan over all possible combinations of num_sites_up_for_grabs sites in candidate_sites_in_target:
    check_other_candidate_sites_first = len(candidate_sites_in_target) < len(
        def_new_supercell_sites_to_check_in_target
    )  # check smaller list first for efficiency

    overlapping_site_indices: list[int] = []  # using indices as faster for comparing than actual sites
    _pbar_increment_per_iter = max(
        0, 20 / len(candidate_sites_in_target) - 0.0001
    )  # up to 20% of progress bar

    def _check_other_sites(
        idx,
        candidate_sites_in_target,
        overlapping_site_indices,
        big_supercell_defect_site,
        bulk_min_bond_length,
    ):
        for other_idx, other_site in enumerate(candidate_sites_in_target):
            if (
                idx == other_idx
                or other_idx in overlapping_site_indices
                or candidate_site.specie.symbol != other_site.specie.symbol
            ):
                continue
            if candidate_site.distance(other_site) < bulk_min_bond_length * 0.5:
                # if distance is less than 50% of bulk bond length, add the site with smaller
                # distance from defect to overlapping_sites (i.e. taking the site with larger
                # distance to defect as a remaining candidate site)
                overlapping_site_indices.append(
                    min(
                        [(idx, candidate_site), (other_idx, other_site)],
                        key=lambda x: x[1].distance_from_point(big_supercell_defect_site.coords),
                    )[0]
                )
        return overlapping_site_indices

    for idx, candidate_site in list(enumerate(candidate_sites_in_target)):
        if pbar is not None:
            pbar.update(_pbar_increment_per_iter)
        if idx in overlapping_site_indices:
            continue

        if check_other_candidate_sites_first:
            overlapping_site_indices = _check_other_sites(
                idx,
                candidate_sites_in_target,
                overlapping_site_indices,
                big_supercell_defect_site,
                bulk_min_bond_length,
            )

        for site in def_new_supercell_sites_to_check_in_target:
            if candidate_site.distance(site) < bulk_min_bond_length * 0.5:
                overlapping_site_indices.append(idx)
                break
        if idx in overlapping_site_indices:
            continue

        if not check_other_candidate_sites_first:
            overlapping_site_indices = _check_other_sites(
                idx,
                candidate_sites_in_target,
                overlapping_site_indices,
                big_supercell_defect_site,
                bulk_min_bond_length,
            )

    return [site for i, site in enumerate(candidate_sites_in_target) if i not in overlapping_site_indices]


class Hashabledict(dict):
    def __hash__(self):
        """
        Make the dictionary hashable by converting it to a tuple of key-value
        pairs.
        """
        return hash(tuple(sorted(self.items())))


@lru_cache(maxsize=int(1e5))
def _cached_Comp_init(comp_dict):
    return Composition(comp_dict)


def _get_matching_sites_from_s1_then_s2(
    template_struct: Structure,
    struct1_pool: Structure,
    struct2_pool: Structure,
    orig_min_dist: float = 5.0,
) -> Structure:
    """
    Generate a stenciled structure from a template sub-set structure and two
    pools of sites/structures.

    Given two pairs of sites/structures, returns a new structure which has (1)
    the closest site in ``struct1_pool`` to each site in ``template_struct``,
    and (2) ``struct2_pool.num_sites - struct2.num_sites`` sites from
    ``struct2_pool``, chosen as those with the largest minimum distances
    from sites in the first set of sites (i.e. the single defect subcell
    matching ``template_struct``).

    The targeted use case is transforming a large periodically-repeated
    super-supercell of a defect supercell (``struct1_pool``) into the
    same super-supercell but with only one copy of the original defect
    supercell (``template_struct``), and the rest of the sites populated by the
    bulk super-supercell (``struct2_pool``; with ``struct2`` being the
    original bulk supercell).

    Args:
        template_struct (Structure):
            The template structure to match.
        struct1_pool (Structure):
            The first pool of sites to match to the template structure.
        struct2_pool (Structure):
            The second pool of sites to match to the template structure.
        orig_min_dist (float):
            The minimum interatomic distance in the original (bulk)
            structure. Used to sanity check the output; if the minimum
            interatomic distance in the templated structure is smaller
            than this, and smaller than a reasonable minimum distance
            (0.65 Å if H in structure, else 1.0 Å), an error is raised.
            Default is 5.0 Å.

    Returns:
        Structure:
            The stenciled structure.
    """
    num_super_supercells = len(struct1_pool) // len(template_struct)  # both also have X
    single_defect_subcell_sites = []

    for (
        sub_site
    ) in template_struct.sites:  # get closest site in big supercell to site, using cartesian coords:
        closest_site = min(
            [site for site in struct1_pool if site.species == sub_site.species],
            key=lambda x: x.distance_from_point(sub_site.coords),
        )
        single_defect_subcell_sites.append(closest_site)

    assert len(set(single_defect_subcell_sites)) == len(single_defect_subcell_sites)  # no repeats

    species_coord_dict = {}  # avoid recomputing coords for each site
    for element in _fast_get_composition_from_sites(struct2_pool).elements:
        species_coord_dict[element.name] = get_coords_and_idx_of_species(
            single_defect_subcell_sites, element.name, frac_coords=True  # frac_coords
        )[0]

    # this could be made faster by vectorising with ``find_idx_of_nearest_coords`` from
    # ``doped.utils.parsing`` but it's far from being the bottleneck in this workflow:
    struct2_pool_idx_min_dist_dict = {}
    struct2_pool_dists_to_template_centre = struct2_pool.lattice.get_all_distances(
        struct2_pool.frac_coords,
        struct2_pool.lattice.get_fractional_coords(
            template_struct.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
        ),
    ).ravel()  # template centre is defect site in stenciling workflow
    template_ws_radius = get_wigner_seitz_radius(template_struct)
    for i, super_site in enumerate(struct2_pool):
        if struct2_pool_dists_to_template_centre[i] > template_ws_radius * 0.75:
            # check that it's outside WS radius, so not defect site itself
            struct2_pool_idx_min_dist_dict[i] = np.min(  # vectorised for fast computation
                struct2_pool.lattice.get_all_distances(
                    species_coord_dict[super_site.specie.symbol], super_site.frac_coords
                )
            )

    # sort possible_bulk_outer_cell_sites by (largest) min dist to single_defect_subcell_sites:
    possible_bulk_outer_cell_sites = [
        struct2_pool[i]
        for i in sorted(
            struct2_pool_idx_min_dist_dict.keys(),
            key=lambda x: struct2_pool_idx_min_dist_dict[x],
            reverse=True,
        )
    ]
    bulk_outer_cell_sites = possible_bulk_outer_cell_sites[
        : int(len(struct2_pool) * (1 - 1 / num_super_supercells))
    ]
    _check_min_dist(Structure.from_sites(bulk_outer_cell_sites), orig_min_dist, warning=False)

    return Structure.from_sites(single_defect_subcell_sites + bulk_outer_cell_sites)


def _get_superset_matrix_and_supercells(defect_frac_coords, orig_supercell, target_supercell):
    """
    Given a defect site (frac coords) in a supercell, the original supercell,
    and a target supercell, return the supercell matrix which makes all lattice
    vectors for ``orig_supercell`` larger than or equal to the largest lattice
    vector in ``target_supercell``, and the corresponding supercells with 'X'
    at the defect site(s).
    """
    min_cell_length = _get_all_encompassing_cube_length(target_supercell.lattice)

    # get supercell matrix which makes all lattice vectors for orig_supercell larger
    # than min_cell_length:
    superset_matrix = np.ceil(min_cell_length / orig_supercell.lattice.abc)
    # could possibly also use non-diagonal supercells to make this more efficient in some cases,
    # but a lot of work, shouldn't really contribute too much to slowdowns, and only relevant in some
    # rare cases (?)
    big_supercell = orig_supercell * superset_matrix  # get big supercell

    # get defect coords in big supercell:
    orig_supercell_with_X = orig_supercell.copy()
    # we've translated the defect to the middle, so X marker goes here:
    orig_supercell_with_X.append("X", [0.5, 0.5, 0.5], coords_are_cartesian=False)
    big_supercell_with_X = orig_supercell_with_X * superset_matrix

    return superset_matrix, big_supercell, big_supercell_with_X, orig_supercell_with_X


def _get_all_encompassing_cube_length(lattice):
    """
    Get the smallest possible cube that fully encompasses the cell, _regardless
    of orientation_.

    This is determined by getting the 8 vertices of the cell and computing the
    max distance between any two vertices, giving the side length of an all-
    encompassing cube. Equivalent to getting the diameter of an encompassing
    sphere.
    """
    # get all vertices of the cell:
    cart_vertices = np.array([lattice.get_cartesian_coords(perm) for perm in product([0, 1], repeat=3)])
    # get 2D matrix of all distances between vertices:
    distances = np.linalg.norm(cart_vertices[:, None] - cart_vertices, axis=-1)

    return max(distances.flatten())


def is_within_frac_bounds(lattice, cart_coords, tol=1e-5):
    """
    Check if a given Cartesian coordinate is inside the unit cell defined by
    the lattice matrix.

    Args:
        lattice:
            ``Lattice`` object defining the unit cell.
        cart_coords:
            The Cartesian coordinates to check.
        tol:
            A tolerance (in Angstrom / cartesian units) for
            frac_coords to be considered within the unit cell.
            If positive, expands the bounds of the unit cell
            by this amount, if negative, shrinks the bounds.
    """
    frac_coords = lattice.get_fractional_coords(cart_coords)
    frac_tols = np.array([tol, tol, tol]) / lattice.abc

    # Check if fractional coordinates are in the range [0, 1)
    return np.all((frac_coords + frac_tols >= 0) & (frac_coords - frac_tols < 1))


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
