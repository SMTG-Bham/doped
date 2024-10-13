"""
Utility functions to re-generate a relaxed defect structure in a different
supercell.

The code in this sub-module is still in development! (TODO)
"""

import math
from collections import Counter
from functools import lru_cache
from itertools import combinations
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
) -> Structure:
    """
    Re-generate a relaxed defect structure in a different supercell.

    This function takes the relaxed defect structure of the input ``DefectEntry``
    (from ``DefectEntry.defect_supercell``) and re-generates it in the
    ``target_supercell`` structure, and the closest possible position to
    ``target_frac_coords`` (if provided, else closest to centre = [0.5, 0.5, 0.5]).

    Briefly, this function works by:

    - Translating the defect site to the centre of the original supercell.
    - Identifying a super-supercell where all lattice vectors are larger than
      the largest lattice vector in the target supercell, to fully encompass
      the target supercell.
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
      supercell).  (TODO: Test this again)
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
            defect generation.

    Returns:
        Structure: The relaxed defect structure in the new supercell.
    """
    # Note to self; using Pycharm breakpoints throughout is likely easiest way to debug these functions
    # TODO: Function needs to be cleaned up more, modularise into more functions, etc

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
            big_supercell_with_X.sites,
            orig_supercell_with_X,
            orig_bulk_supercell * supercell_matrix,
            orig_bulk_supercell,
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
            big_defect_supercell, translation_from_big_defect_site_to_target_middle, frac_coords=False
        )

        # get all atoms in big supercell within the cartesian bounds of the target_supercell supercell:
        # we need to ensure the same number of sites, and that the sites we choose are appropriate for the
        # new supercell (i.e. that if we have e.g. an in-plane contraction, we don't take duplicate atoms
        # that then correspond to tiny inter-atomic distances in the new supercell due to imperfect
        # stenciling under PBC -- so we can't just take the atoms that are closest to the defect). So,
        # scan over possible choices of atoms to include, and take that which maximises the _minimum_
        # inter-atomic distance in the new supercell, when accounting for PBCs:
        # maybe for robustness, can make tol an adjustable parameter, and can scan over larger values
        # if initial scan fails? TODO
        edge_tol = 2  # Angstrom
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
            PeriodicSite(
                site.specie, site.coords, lattice=target_supercell.lattice, coords_are_cartesian=True
            )
            for site in def_new_supercell_sites
            if not is_within_frac_bounds(target_supercell.lattice, site.coords, tol=-edge_tol * 2)
        ]
        candidate_sites_in_target = [
            PeriodicSite(
                site.specie, site.coords, lattice=target_supercell.lattice, coords_are_cartesian=True
            )
            for site in possible_new_supercell_sites
            if site not in def_new_supercell_sites
        ]
        num_sites_up_for_grabs = len(target_supercell) - len(def_new_supercell_sites)

        # scan over all possible combinations of num_sites_up_for_grabs sites in candidate_sites_in_target:
        check_other_candidate_sites_first = len(candidate_sites_in_target) < len(
            def_new_supercell_sites_to_check_in_target
        )  # check smaller list first for efficiency

        overlapping_site_indices = []  # using indices as faster for comparing than actual sites
        _pbar_increment_per_iter = max(
            0, 20 / len(candidate_sites_in_target) - 0.0001
        )  # up to 20% of progress bar
        for idx, candidate_site in list(enumerate(candidate_sites_in_target)):
            pbar.update(_pbar_increment_per_iter)
            if idx in overlapping_site_indices:
                continue

            if check_other_candidate_sites_first:
                for other_idx, other_site in enumerate(candidate_sites_in_target):
                    if (
                        idx == other_idx
                        or other_idx in overlapping_site_indices
                        or candidate_site.specie.symbol != other_site.specie.symbol
                    ):
                        continue
                    if candidate_site.distance(other_site) < bulk_min_bond_length * 0.5:
                        # if distance is greater than 50% of bulk bond length,
                        # append site with greater distance from defect to overlapping_sites
                        overlapping_site_indices.append(
                            min(
                                [(idx, candidate_site), (other_idx, other_site)],
                                key=lambda x: x[1].distance_from_point(big_supercell_defect_site.coords),
                            )[0]
                        )

            for site in def_new_supercell_sites_to_check_in_target:
                if candidate_site.distance(site) < bulk_min_bond_length * 0.5:
                    overlapping_site_indices.append(idx)
                    break
            if idx in overlapping_site_indices:
                continue

            if not check_other_candidate_sites_first:
                for other_idx, other_site in enumerate(candidate_sites_in_target):
                    if (
                        idx == other_idx
                        or other_idx in overlapping_site_indices
                        or candidate_site.specie.symbol != other_site.specie.symbol
                    ):
                        continue
                    if candidate_site.distance(other_site) < bulk_min_bond_length * 0.5:
                        # append site with greater distance from defect to overlapping_sites
                        overlapping_site_indices.append(
                            min(
                                [(idx, candidate_site), (other_idx, other_site)],
                                key=lambda x: x[1].distance_from_point(big_supercell_defect_site.coords),
                            )[0]
                        )

        candidate_sites_in_target = [
            site for i, site in enumerate(candidate_sites_in_target) if i not in overlapping_site_indices
        ]
        pbar.set_description("Calculating best match")  # now at 40% of progress bar

        # target additional composition:
        target_composition = (
            target_supercell.composition  # this is bulk, need to account for defect:
            + orig_supercell.composition
            - orig_bulk_supercell.composition
        )
        def_new_supercell_sites_struct = Structure.from_sites(def_new_supercell_sites)
        def_new_supercell_sites_struct.remove_species("X")
        def_new_supercell_sites_comp = def_new_supercell_sites_struct.composition
        combo_composition = target_composition - def_new_supercell_sites_comp  # composition we need to add

        # get composition of list of sites:
        num_possible_combos = math.comb(len(candidate_sites_in_target), num_sites_up_for_grabs)
        if num_possible_combos > 1e10:
            # TODO: Handle this by dynamically adjusting tol??
            raise RuntimeError(
                "Too many possible combinations to check. Code will take forever, aborting."
            )

        species_symbols = [site.specie.symbol for site in candidate_sites_in_target]
        min_interatomic_distances_tuple_combo_dict = {}

        for idx_combo in list(combinations(range(len(candidate_sites_in_target)), num_sites_up_for_grabs)):
            comp = _cached_Comp_init(Hashabledict(Counter([species_symbols[i] for i in idx_combo])))
            if _Comp__eq__(comp, combo_composition):
                # could early break cases where the distances are too small? if a bottleneck, currently not
                # and/or could loop over subsets of each possible combo first, culling any which break this
                combo_sites_in_superset = [candidate_sites_in_target[i] for i in idx_combo]
                fake_candidate_struct_sites = (
                    def_new_supercell_sites_to_check_in_target + combo_sites_in_superset
                )
                frac_coords = [site.frac_coords for site in fake_candidate_struct_sites]
                distance_matrix = target_supercell.lattice.get_all_distances(frac_coords, frac_coords)
                sorted_distances = np.sort(distance_matrix.flatten())
                min_interatomic_distances_tuple_combo_dict[
                    tuple(sorted_distances[len(frac_coords) : len(frac_coords) + 10])
                    # take the 10 smallest distances, in case defect site is smallest distance
                ] = [candidate_sites_in_target[i] for i in idx_combo]

        # get the candidate structure with the largest minimum interatomic distance:
        min_interatomic_distances_tuple_combo_list = sorted(
            min_interatomic_distances_tuple_combo_dict.items(), key=lambda x: x[0], reverse=True
        )
        new_supercell_sites = def_new_supercell_sites + list(
            min_interatomic_distances_tuple_combo_list[0][1]
        )
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
        defect_type, comp_diff = get_defect_type_and_composition_diff(
            target_supercell, new_supercell_w_defect_comp
        )
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
    Using ``framework_sites`` as the framework of lattice sites to.
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

    for super_site in struct2_pool:
        closest_site = min(
            [site for site in struct2.sites if site.species == super_site.species],
            key=lambda x: x.distance_from_point(super_site.coords),
        )
        if (
            closest_site.distance_from_point(super_site.coords) > bulk_min_bond_length * 0.99
            and min(super_site.distance_from_point(s.coords) for s in single_defect_subcell_sites)
            > bulk_min_bond_length * 0.8
        ):
            single_defect_subcell_sites.append(super_site)

    # should have total number of sites equal to len(struct1) + len(struct2_pool) - len(struct2)
    assert len(single_defect_subcell_sites) == len(struct1.sites) + len(struct2_pool) - len(struct2)

    return Structure.from_sites(single_defect_subcell_sites)


def _get_superset_matrix_and_supercells(defect_frac_coords, orig_supercell, target_supercell):
    """
    Given a defect site (frac coords) in a supercell, the original supercell,
    and a target supercell, return the supercell matrix which makes all lattice
    vectors for ``orig_supercell`` larger than or equal to the largest lattice
    vector in ``target_supercell``, and the corresponding supercells with 'X'
    at the defect site(s).
    """
    largest_target_cell_vector = np.max(target_supercell.lattice.abc)

    # get supercell matrix which makes all lattice vectors for orig_supercell larger
    # than largest_superset_lattice_vector:
    superset_matrix = np.ceil(largest_target_cell_vector / orig_supercell.lattice.abc)

    big_supercell = orig_supercell * superset_matrix  # get big supercell

    # get defect coords in big supercell:
    orig_supercell_with_X = orig_supercell.copy()
    # we've translated the defect to the middle, so X marker goes here:
    orig_supercell_with_X.append("X", [0.5, 0.5, 0.5], coords_are_cartesian=False)
    big_supercell_with_X = orig_supercell_with_X * superset_matrix

    return superset_matrix, big_supercell, big_supercell_with_X, orig_supercell_with_X


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
