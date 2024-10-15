"""
Utility functions to re-generate a relaxed defect structure in a different
supercell.

The code in this sub-module is still in development! (TODO)
"""

import math
from collections import Counter
from functools import lru_cache
from itertools import combinations, product
from typing import Union

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
    find_idx_of_nearest_coords,
    get_defect_type_and_composition_diff,
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
        orig_bulk_supercell = _get_bulk_supercell(defect_entry)
        orig_defect_frac_coords = defect_entry.sc_defect_frac_coords
        target_supercell = target_supercell.copy()
        bulk_min_bond_length = np.sort(
            orig_bulk_supercell.distance_matrix[orig_bulk_supercell.distance_matrix > 0.5].flatten()
        )[0]
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
        big_defect_supercell = _get_corresponding_sites_from_struct1_then_2(
            big_supercell_with_X,
            orig_supercell_with_X,
            orig_bulk_supercell * supercell_matrix,
            orig_bulk_supercell,
        )
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

                new_supercell_w_defect_comp = _fast_get_composition_from_sites(
                    [site for site in new_supercell_sites if site.specie.symbol != "X"]
                )
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

        new_supercell = Structure(
            lattice=target_supercell.lattice,
            species=[site.specie for site in new_supercell_sites],
            coords=[site.coords for site in new_supercell_sites],
            coords_are_cartesian=True,
            to_unit_cell=True,
        )

        # now we just do get s2 like s1 to get the orientation right:
        new_supercell_w_defect_comp = new_supercell.copy()
        new_supercell_w_defect_comp.remove_species("X")
        x_site = next(site for site in new_supercell if site.specie.symbol == "X")
        if defect_type == "vacancy":
            new_supercell.append(
                next(iter(comp_diff.keys())), x_site.frac_coords, coords_are_cartesian=False
            )
        elif defect_type == "substitution":
            # bulk species is key with value = -1 in comp_diff:
            bulk_species = next(k for k, v in comp_diff.items() if v == -1)
            idx = find_idx_of_nearest_coords(
                new_supercell.frac_coords, x_site.frac_coords, new_supercell_w_defect_comp.lattice
            )
            new_supercell.replace(
                idx, bulk_species, new_supercell[idx].frac_coords, coords_are_cartesian=False
            )
        else:  # interstitial
            idx = find_idx_of_nearest_coords(
                new_supercell.frac_coords, x_site.frac_coords, new_supercell_w_defect_comp.lattice
            )
            new_supercell.remove_sites([idx])

        oriented_new_supercell = orient_s2_like_s1(
            target_supercell,
            new_supercell,
            verbose=False,
            ignored_species=["X"],  # ignore X site
        )
        x_site = next(
            oriented_new_supercell.pop(i)
            for i, site in enumerate(oriented_new_supercell.sites)
            if site.specie.symbol == "X"
        )
        # need to put back in correct defect:
        if defect_type == "vacancy":
            added_idx = find_idx_of_nearest_coords(
                oriented_new_supercell.frac_coords, x_site.frac_coords, oriented_new_supercell.lattice
            )
            oriented_new_supercell.remove_sites([added_idx])

        elif defect_type == "substitution":
            # defect species is key with value = +1 in comp_diff:
            defect_species = next(k for k, v in comp_diff.items() if v == +1)
            idx = find_idx_of_nearest_coords(
                oriented_new_supercell.frac_coords, x_site.frac_coords, oriented_new_supercell.lattice
            )
            oriented_new_supercell.replace(
                idx, defect_species, oriented_new_supercell[idx].frac_coords, coords_are_cartesian=False
            )
        else:  # interstitial
            defect_species = next(k for k, v in comp_diff.items() if v == +1)
            oriented_new_supercell.append(defect_species, x_site.frac_coords, coords_are_cartesian=False)

        pbar.update(35)  # 90% of progress bar

        if target_frac_coords is not False:
            pbar.set_description("Placing defect closest to target_frac_coords")

            # translate to put defect at closest possible site to target_frac_coords
            sga = get_sga(target_supercell)
            symm_ops = sga.get_symmetry_operations()
            symm_op_pos_dict = {}
            for i, symm_op in enumerate(symm_ops):  # should check if frac or cartesian is faster
                symm_opped_site = apply_symm_op_to_site(
                    symm_op, x_site, fractional=True, rotate_lattice=False
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

    return oriented_new_supercell


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


def _get_corresponding_sites_from_struct1_then_2(struct1_pool, struct1, struct2_pool, struct2):
    """
    Given two pairs of sites/structures, returns a new structure which has (1)
    the closest site in ``struct1_pool`` to each site in ``struct1``, and (2)
    each site in ``struct2_pool`` which is further away from any site in
    ``struct2`` than the minimum bond length in ``struct2``, and away from any
    site in the first set of sites by at least 80% of this minimum bond length.

    The targeted use case is transforming a large periodically-repeated
    super-supercell of a defect supercell (``struct1_pool``) into the
    same super-supercell but with only one copy of the original defect
    supercell (``struct1``), and the rest of the sites populated by the
    bulk super-supercell (``struct2_pool``; with ``struct2`` being the
    original bulk supercell).
    """
    single_defect_subcell_sites = []
    bulk_min_bond_length = np.sort(struct2.distance_matrix[struct2.distance_matrix > 0.5].flatten())[0]

    for sub_site in struct1.sites:  # get closest site in big supercell to site, using cartesian coords:
        closest_site = min(
            [site for site in struct1_pool if site.species == sub_site.species],
            key=lambda x: x.distance_from_point(sub_site.coords),
        )
        single_defect_subcell_sites.append(closest_site)

    # should have no repeats:
    assert len(set(single_defect_subcell_sites)) == len(single_defect_subcell_sites)
    assert len(single_defect_subcell_sites) == len(struct1.sites)  # and one site for each

    bulk_outer_cell_sites = []
    for super_site in struct2_pool:
        matching_site_coords = np.array(
            [site.coords for site in struct2.sites if site.species == super_site.species]
        )
        closest_site_dist = min(np.linalg.norm(matching_site_coords - super_site.coords, axis=-1))
        if closest_site_dist > bulk_min_bond_length * 0.99:
            # check min dist to sites in single_defect_subcell_sites:
            matching_site_coords = np.array(
                [site.coords for site in single_defect_subcell_sites if site.species == super_site.species]
            )
            min_dist = min(np.linalg.norm(matching_site_coords - super_site.coords, axis=-1))
            if min_dist > bulk_min_bond_length * 0.8:
                bulk_outer_cell_sites.append(super_site)

    combined_sites = single_defect_subcell_sites + bulk_outer_cell_sites
    # should have total number of sites equal to len(struct1) + len(struct2_pool) - len(struct2)
    assert len(combined_sites) == len(struct1.sites) + len(struct2_pool) - len(struct2)

    return Structure.from_sites(combined_sites)


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
