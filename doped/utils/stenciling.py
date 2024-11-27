"""
Utility functions to re-generate a relaxed defect structure in a different
supercell.
"""

import math
import warnings
from collections import Counter
from collections.abc import Sequence
from itertools import combinations, product
from typing import Optional, Union

import numpy as np
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Composition, Lattice, Structure
from tqdm import tqdm

from doped.core import DefectEntry
from doped.utils.configurations import orient_s2_like_s1
from doped.utils.efficiency import (
    Hashabledict,
    _cached_Composition_init,
    _Composition__eq__,
    _fast_get_composition_from_sites,
)
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    check_atom_mapping_far_from_defect,
    get_coords_and_idx_of_species,
    get_defect_type_and_composition_diff,
    get_wigner_seitz_radius,
)
from doped.utils.supercells import _largest_cube_length_from_matrix, min_dist
from doped.utils.symmetry import (
    SymmOp,
    apply_symm_op_to_site,
    apply_symm_op_to_struct,
    get_clean_structure,
    get_sga,
    translate_structure,
)


def get_defect_in_supercell(
    defect_entry: DefectEntry,
    target_supercell: Structure,
    check_bulk: bool = True,
    target_frac_coords: Union[np.ndarray[float], list[float], bool] = True,
    edge_tol: float = 1,
) -> tuple[Structure, Structure]:
    """
    Re-generate a relaxed defect structure in a different supercell.

    This function takes the relaxed defect structure of the input ``DefectEntry``
    (from ``DefectEntry.defect_supercell``) and re-generates it in the
    ``target_supercell`` structure, and the closest possible position to
    ``target_frac_coords`` (if provided, else closest to centre = [0.5, 0.5, 0.5]),
    also providing the corresponding bulk supercell (which should be the same for
    each generated defect supercell given the same ``target_supercell`` and base
    supercell for ``defect_entry``, see note below).

    ``target_supercell`` should be the same host crystal structure, just with
    different supercell dimensions, having the same lattice parameters and bond
    lengths.

    Note: This function does _not_ guarantee that the generated defect supercell
    atomic position basis exactly matches that of ``target_supercell``, which may
    have come from a different primitive structure definition (e.g. CdTe with
    ``{"Cd": [0,0,0], "Te": [0.25,0.25,0.25]}`` vs
    ``{"Cd": [0,0,0], "Te": [0.75,0.75,0.75]}``). The generated supercell _will_
    have the exact same lattice/cell definition with fully symmetry-equivalent atom
    positions, but if the actual position basis differs then this can cause issues
    with parsing finite-size corrections (which rely on site-matched potentials).
    This is perfectly fine if it occurs, just will require the use of a matching
    bulk/reference supercell when parsing (rather than the input ``target_supercell``)
    -- ``doped`` will also throw a warning about this when parsing if a non-matching
    bulk supercell is used anyway.
    This function will automatically check if the position basis in the generated
    supercell differs from that of ``target_supercell``, printing a warning if so
    (unless ``check_bulk`` is ``False``) and returning the corresponding bulk
    supercell which should be used for parsing defect calculations with the
    generated supercell. Of course, if generating multiple defects in the same
    ``target_supercell``, only one such bulk supercell calculation should be required
    (should correspond to the same bulk supercell in each case).

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
        check_bulk (bool):
            Whether to check if the generated defect/bulk supercells have
            different atomic position bases to ``target_supercell`` (as described
            above) -- if so, a warning will be printed (unless ``check_bulk`` is
            ``False``). Default is ``True``.
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
            stenciled supercell, when determining the best match of
            sites to stencil out in the new supercell (of ``target_supercell``
            dimension). Default is 1 Angstrom, and then this is sequentially
            increased up to 4.5 Angstrom if the initial scan fails.

    Returns:
        tuple[Structure, Structure]:
            The re-generated defect supercell in the ``target_supercell`` lattice,
            and the corresponding bulk/reference supercell for the generated defect
            supercell (see explanations above).
    """
    # Note to self; using Pycharm breakpoints throughout is likely easiest way to debug these functions
    # TODO: Tests!! (At least one of each defect, Se good test case, then at least one or two with
    #  >unary compositions and extrinsic substitution/interstitial)
    # TODO: We should now be able to use these functions (without the final re-orientation step,
    #  for speed) to determine the point symmetries of relaxed defects in non-symmetry-conserving
    #  supercells, by stenciling into a small symmetry-conserving cell and getting the point symmetry
    #  for that -- will do!

    pbar = tqdm(
        total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
    )  # tqdm progress bar. 100% is completion
    pbar.set_description("Getting super-supercell (relaxed defect + bulk sites)")

    bulk_mismatch_warning = False

    try:
        orig_supercell = _get_defect_supercell(defect_entry)
        orig_min_dist = min_dist(orig_supercell)
        orig_bulk_supercell = _get_bulk_supercell(defect_entry)
        orig_defect_frac_coords = defect_entry.sc_defect_frac_coords
        target_supercell = target_supercell.copy()
        bulk_min_bond_length = min_dist(orig_bulk_supercell)
        target_frac_coords = [0.5, 0.5, 0.5] if target_frac_coords is True else target_frac_coords

        # ensure no oxidation states (for easy composition matching later)
        for struct in [orig_supercell, orig_bulk_supercell, target_supercell]:
            struct.remove_oxidation_states()

        # first translate both orig supercells to put defect in the middle, to aid initial stenciling:
        orig_def_to_centre = np.array([0.5, 0.5, 0.5]) - orig_defect_frac_coords
        orig_supercell = translate_structure(orig_supercell, orig_def_to_centre, frac_coords=True)
        trans_orig_bulk_supercell = translate_structure(
            orig_bulk_supercell, orig_def_to_centre, frac_coords=True
        )

        # get big_supercell, which is expanded version of orig_supercell so that _each_ lattice vector is
        # now bigger than the _largest_ lattice vector in target_supercell (so that target_supercell is
        # fully encompassed by big_supercell):
        superset_matrix, big_supercell, big_supercell_with_X, orig_supercell_with_X = (  # supa-set!!
            _get_superset_matrix_and_supercells(orig_supercell, target_supercell, [0.5, 0.5, 0.5])
        )
        big_bulk_supercell = orig_bulk_supercell * superset_matrix  # get big bulk supercell

        # this big_supercell is with the defect now repeated in it, but we want it with just one defect,
        # so only take the first repeated defect supercell, and then the rest of the sites from the
        # expanded bulk supercell:
        # only keep atoms in big supercell which are within the original supercell bounds:
        big_defect_supercell = _get_matching_sites_from_s1_then_s2(
            orig_supercell_with_X,
            big_supercell_with_X,
            trans_orig_bulk_supercell * superset_matrix,
            orig_min_dist,
        )

        # translate structure to put defect at the centre of the big supercell (w/frac_coords)
        big_supercell_defect_site = next(s for s in big_defect_supercell.sites if s.specie.symbol == "X")
        def_to_centre = np.array([0.5, 0.5, 0.5]) - big_supercell_defect_site.frac_coords
        big_defect_supercell = translate_structure(big_defect_supercell, def_to_centre, frac_coords=True)

        pbar.update(20)  # 20% of progress bar
        pbar.set_description("Getting sites in border region")

        # get all atoms in big supercell within the cartesian bounds of the target_supercell supercell:
        new_defect_supercell = _stencil_target_cell_from_big_cell(
            big_defect_supercell,
            target_supercell,
            bulk_min_bond_length=bulk_min_bond_length,
            orig_min_dist=orig_min_dist,
            edge_tol=edge_tol,
            pbar=pbar,
        )
        new_bulk_supercell = _stencil_target_cell_from_big_cell(
            big_bulk_supercell,
            target_supercell,
            bulk_min_bond_length=bulk_min_bond_length,
            orig_min_dist=bulk_min_bond_length,
            edge_tol=1e-3,
            pbar=None,
        )  # shouldn't need `edge_tol`, should be much faster than defect supercell stencil

        pbar.update(15)  # 55% of progress bar
        pbar.set_description("Ensuring matching orientation w/target_supercell")

        # now we just do get s2 like s1 to get the orientation right:
        defect_site = next(site for site in new_defect_supercell if site.specie.symbol == "X")

        # Note: this function is typically the main bottleneck in this workflow. We have already
        # optimised the underlying ``StructureMatcher`` workflow in many ways (caching,
        # fast structure/site/composition comparisons, skipping comparison of defect neighbourhood to
        # reduce requisite ``stol`` etc; being many orders of magnitude faster than the base
        # ``pymatgen`` ``StructureMatcher``), however the ``_cart_dists()`` function call is
        # still quite expensive, especially with large structures with significant noise in the atomic
        # positions...
        new_defect_supercell_w_defect_neighbours_as_X = _convert_defect_neighbours_to_X(
            new_defect_supercell, defect_site.frac_coords, coords_are_cartesian=False
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Not all sites have property orig_species")
            # first we orient the generated _bulk_ supercell to match the ``target_supercell``,
            # to try ensure consistency in the generated supercells
            oriented_new_bulk_supercell = orient_s2_like_s1(  # speed should be >= defect orienting
                target_supercell,
                new_bulk_supercell,
                verbose=False,
                allow_subset=True,
            )
            # return new_defect_supercell, _round_struct_coords(oriented_new_bulk_supercell,
            #                                                   dist_precision=0.01, to_unit_cell=True)
            oriented_new_defect_supercell_w_defect_neighbours_as_X = orient_s2_like_s1(
                oriented_new_bulk_supercell,
                new_defect_supercell_w_defect_neighbours_as_X,
                verbose=False,
                ignored_species=["X"],  # ignore X site
                allow_subset=True,  # allow defect supercell composition to differ
            )
            oriented_new_defect_supercell = _convert_X_back_to_orig_species(
                oriented_new_defect_supercell_w_defect_neighbours_as_X
            )
            oriented_new_defect_site = next(  # get defect site and remove X from sites
                oriented_new_defect_supercell.pop(i)
                for i, site in enumerate(oriented_new_defect_supercell.sites)
                if site.specie.symbol == "X"
            )

            pbar.update(35)  # 90% of progress bar

            if target_frac_coords is not False:
                pbar.set_description("Placing defect closest to target_frac_coords")

                target_symm_op = _scan_symm_ops_to_place_site_closest_to_frac_coords(
                    target_supercell, oriented_new_defect_site, target_frac_coords
                )
                oriented_new_defect_supercell = get_clean_structure(
                    apply_symm_op_to_struct(  # apply symm_op to structure
                        target_symm_op,
                        oriented_new_defect_supercell,
                        fractional=True,
                        rotate_lattice=False,
                    ),
                )  # reordered inputs in updated doped
                if check_bulk:
                    bulk_mismatch_warning = not check_atom_mapping_far_from_defect(
                        target_supercell,
                        oriented_new_bulk_supercell,
                        oriented_new_defect_site.frac_coords,
                        warning=False,
                    )

                oriented_new_bulk_supercell = get_clean_structure(
                    apply_symm_op_to_struct(
                        target_symm_op,
                        oriented_new_bulk_supercell,
                        fractional=True,
                        rotate_lattice=False,
                    ),
                )

        pbar.update(pbar.total - pbar.n)  # set to 100% of progress bar

    except Exception as e:
        pbar.close()
        raise e

    finally:
        pbar.close()

    if bulk_mismatch_warning:  # print warning after closing pbar; cleaner
        warnings.warn(
            "Note that the atomic position basis of the generated defect/bulk supercell "
            "differs from that of the ``target_supercell``. This is likely fine, "
            "and just due to differences in (symmetry-equivalent) primitive cell definitions "
            "(e.g. {'Cd': [0,0,0], 'Te': [0.25,0.25,0.25]} vs "
            "{'Cd(': [0,0,0], 'Te': [0.75,0.75,0.75]}``) -- see ``get_defect_in_supercell`` "
            "docstring for more info. For accurate finite-size charge corrections when "
            "parsing, a matching bulk and defect supercell should be used, and so the "
            "matching bulk supercell for the generated defect supercell (also returned "
            "by this function) should be used for its reference host cell calculation.",
        )

    _check_min_dist(oriented_new_defect_supercell, orig_min_dist)  # check distances are reasonable
    _check_min_dist(oriented_new_bulk_supercell, bulk_min_bond_length)
    return oriented_new_defect_supercell, oriented_new_bulk_supercell


def _scan_symm_ops_to_place_site_closest_to_frac_coords(
    symm_ops: Union[Structure, Sequence[SymmOp]],
    site: PeriodicSite,
    target_frac_coords: Optional[Union[np.ndarray[float], list[float]]] = None,
) -> SymmOp:
    """
    Given either a list of symmetry operations or a structure (to extract
    symmetry operations from), scan over all symmetry operations to find the
    one which places the provided ``site`` closest to the target fractional
    coordinates.

    Args:
        symm_ops (Union[Structure, Sequence[SymmOp]]):
            Either a list of symmetry operations or a structure from
            which to extract symmetry operations.
        site (PeriodicSite):
            The site to place closest to the target fractional coordinates.
        target_frac_coords (Optional[Union[np.ndarray[float], list[float]]]):
            The target fractional coordinates to place the site closest to.
            Default is ``None``, in which case the site is placed closest to
            the centre of the supercell (i.e. [0.5, 0.5, 0.5]).
    """
    if isinstance(symm_ops, Structure):
        sga = get_sga(symm_ops)
        symm_ops = sga.get_symmetry_operations()

    target_frac_coords = [0.5, 0.5, 0.5] if target_frac_coords is None else target_frac_coords

    # translate to put defect at closest possible site to target_frac_coords
    symm_op_pos_dict = {}
    for i, symm_op in enumerate(symm_ops):  # should check if frac or cartesian is faster
        symm_opped_site = apply_symm_op_to_site(symm_op, site, fractional=True, rotate_lattice=False)
        symm_op_pos_dict[i] = symm_opped_site.to_unit_cell().frac_coords

    # get symm_op which puts defect closest to target_frac_coords:
    closest_site = min(symm_op_pos_dict.items(), key=lambda x: np.linalg.norm(x[1] - target_frac_coords))
    return symm_ops[closest_site[0]]


def _stencil_target_cell_from_big_cell(
    big_supercell: Structure,
    target_supercell: Structure,
    edge_tol: float = 1.0,
    bulk_min_bond_length: Optional[float] = None,
    orig_min_dist: Optional[float] = None,
    pbar: Optional[tqdm] = None,
) -> Structure:
    """
    Given the input ``big_supercell`` and ``target_supercell`` (which should be
    fully encompassed by the former), stencil out the sites in
    ``big_supercell`` which correspond to the sites in ``target_supercell``
    (i.e. are within the Cartesian bounds of ``target_supercell``).

    Note that this function assumes that the defect is roughly centred
    within ``big_supercell`` (i.e. near [0.5, 0.5, 0.5])! The midpoints of
    ``target_supercell`` and ``big_supercell`` are then aligned within this
    function, before stenciling.

    We need to ensure the appropriate number of sites (and their composition) are
    taken, and that the sites we choose are appropriate for the new supercell (i.e.
    that if we have e.g. an in-plane contraction, we don't take duplicate atoms
    that then correspond to tiny inter-atomic distances in the new supercell due to
    imperfect stenciling under PBC -- so we can't simply take the atoms that are
    closest to the defect). So, here we scan over possible choices of atoms to
    include, and take the combination which maximises the _minimum_ inter-atomic
    distance in the new supercell, when accounting for PBCs.

    Args:
        big_supercell (Structure):
            The supercell structure which fully encompasses ``target_supercell``,
            from which to stencil out the sites.
        target_supercell (Structure):
            The supercell structure giving the cell dimensions to stencil out
            from ``big_supercell``.
        edge_tol (float):
            A tolerance (in Angstrom) for site displacements at the edge of the
            stenciled supercell, when determining the best match of
            sites to stencil out in the new supercell (of ``target_supercell``
            dimension). Default is 1 Angstrom, and then this is sequentially
            increased up to 4.5 Angstrom if the initial scan fails.
        bulk_min_bond_length (float):
            The minimum interatomic distance in the bulk supercell. Default is
            ``None``, in which case it is calculated from ``target_supercell``.
        orig_min_dist (float):
            The minimum interatomic distance in the original defect supercell.
            Default is ``None``, in which case it is calculated from
            ``big_supercell``.
        pbar (tqdm):
            ``tqdm`` progress bar object to update (for internal ``doped``
            usage). Default is ``None``.

    Returns:
        Structure: The stenciled supercell structure.
    """
    # first, translate sites to put the centre of big_supercell (which should have defect site at the
    # centre) to the centre of the target supercell (w/cart coords)
    target_supercell_midpoint_cart_coords = target_supercell.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    translation_from_big_defect_middle_to_target_middle = (
        target_supercell_midpoint_cart_coords - big_supercell.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    )
    big_supercell = translate_structure(
        big_supercell,
        translation_from_big_defect_middle_to_target_middle,
        frac_coords=False,
        to_unit_cell=False,
    )

    if bulk_min_bond_length is None:
        bulk_min_bond_length = min_dist(target_supercell)
    if orig_min_dist is None:
        orig_min_dist = min_dist(big_supercell.copy().remove_species(["X"]))

    # get target composition accounting for defect presence in big supercell:
    num_sc = big_supercell.volume / target_supercell.volume  # may be fractional
    big_supercell_comp_wout_X = big_supercell.copy().remove_species(["X"]).composition
    target_composition = big_supercell_comp_wout_X - ((num_sc - 1) * target_supercell.composition)
    target_composition = Composition({k: round(v) for k, v in target_composition.items()})

    while edge_tol <= 4.5:  # sequentially increase edge_tol by 0.5 Å, up to 4.5 Å, until match is found:
        try:
            (
                def_new_supercell_sites,
                def_new_supercell_sites_to_check_in_target,
                candidate_sites_in_target,
                combo_composition,
            ) = _get_candidate_supercell_sites(
                big_supercell, target_supercell, target_composition, edge_tol
            )
            num_sites_up_for_grabs = int(sum(combo_composition.values()))  # number of sites to be placed
            candidate_sites_in_target = _remove_overlapping_sites(
                candidate_sites_in_target=candidate_sites_in_target,
                def_new_supercell_sites_to_check_in_target=def_new_supercell_sites_to_check_in_target,
                big_supercell_defect_coords=target_supercell_midpoint_cart_coords,
                bulk_min_bond_length=bulk_min_bond_length,
                pbar=pbar,
            )

            if len(candidate_sites_in_target) < num_sites_up_for_grabs:
                raise RuntimeError(
                    f"Too little candidate sites ({len(candidate_sites_in_target)}) to match target "
                    f"composition ({num_sites_up_for_grabs} sites to be placed). Aborting."
                )
            num_combos = math.comb(len(candidate_sites_in_target), num_sites_up_for_grabs)
            if num_combos > 1e10:
                raise RuntimeError(
                    "Far too many possible site combinations to check, indicating a code failure. "
                    "Aborting, please report this case to the developers!"
                )

            if pbar is not None:
                pbar.set_description(
                    f"Calculating best match (edge_tol = {edge_tol} Å, possible combos = {num_combos})"
                )  # 40% of pbar

            species_symbols = [site.specie.symbol for site in candidate_sites_in_target]
            min_interatomic_distances_tuple_combo_dict = {}
            idx_combos = list(combinations(range(len(candidate_sites_in_target)), num_sites_up_for_grabs))

            if idx_combos and idx_combos != [()]:
                for idx_combo in idx_combos:
                    if _Composition__eq__(
                        _cached_Composition_init(
                            Hashabledict(Counter([species_symbols[i] for i in idx_combo]))
                        ),
                        combo_composition,
                    ):
                        # could early break cases where the distances are too small? if a bottleneck,
                        # currently not. And/or could loop over subsets of each possible combo first,
                        # culling any which break this
                        fake_candidate_struct_sites = def_new_supercell_sites_to_check_in_target + [
                            candidate_sites_in_target[i] for i in idx_combo
                        ]
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
                if target_composition != target_supercell.composition:  # defect, not bulk
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

            if pbar is not None:
                pbar.n = 20  # decrease pbar progress back to 20%
                pbar.refresh()
                pbar.set_description(f"Trying edge_tol = {edge_tol} Å")
            continue

    return new_supercell


def _convert_defect_neighbours_to_X(
    defect_supercell: Structure,
    defect_position: np.ndarray[float],
    coords_are_cartesian: bool = False,
    ws_radius_fraction: float = 0.5,
) -> Structure:
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


def _check_min_dist(
    structure: Structure,
    orig_min_dist: float = 5.0,
    warning: bool = True,
    ignored_species: Optional[list[str]] = None,
) -> None:
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


def _get_candidate_supercell_sites(
    big_supercell: Structure,
    target_supercell: Structure,
    target_composition: Composition,
    edge_tol: float = 1.0,
) -> tuple[list[PeriodicSite], list[PeriodicSite], list[PeriodicSite], Composition]:
    """
    Get all atoms in ``big_supercell`` which are within the Cartesian bounds of
    the ``target_supercell`` supercell.

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

    Note that this function assumes that the defect (or any significant
    atomic displacements) is (are) roughly centred within ``big_supercell``
    (i.e. near [0.5, 0.5, 0.5])!

    Args:
        big_supercell (Structure):
            The super-supercell with a single defect supercell and
            rest of the sites populated by the bulk supercell.
        target_supercell (Structure):
            The supercell structure to re-generate the relaxed defect
            structure in.
        target_composition (Composition):
            The composition of the target supercell.
        edge_tol (float):
            A tolerance (in Angstrom) for site displacements at the edge of the
            ``target_supercell`` supercell, when determining the best match of
            sites to stencil out in the new supercell. Default is 1 Angstrom.

    Returns:
        tuple[list[PeriodicSite], list[PeriodicSite], list[PeriodicSite], Composition, int]:
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
        for site in big_supercell.sites
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
    big_supercell_defect_coords: np.ndarray[float],
    bulk_min_bond_length: Optional[float] = None,
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
        big_supercell_defect_coords (np.ndarray[float]):
            *Cartesian* coordinates of the defect site in the big supercell.
            Used to choose between overlapping sites (favouring those which
            have a larger distance from the defect site).
        bulk_min_bond_length (float):
            The minimum bond length in the bulk supercell, used to check
            if inter-site distances are reasonable. If ``None`` (default),
            determined automatically from ``candidate_sites_in_target``.
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

    if bulk_min_bond_length is None:
        bulk_min_bond_length = min_dist(Structure.from_sites(candidate_sites_in_target))

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
        big_supercell_defect_coords,
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
                        key=lambda x: x[1].distance_from_point(big_supercell_defect_coords),
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
                big_supercell_defect_coords,
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
                big_supercell_defect_coords,
                bulk_min_bond_length,
            )

    return [site for i, site in enumerate(candidate_sites_in_target) if i not in overlapping_site_indices]


def _get_matching_sites_from_s1_then_s2(
    template_struct: Structure,
    struct1_pool: Structure,
    struct2_pool: Structure,
    orig_min_dist: float = 5.0,
) -> Structure:
    """
    Generate a stenciled structure from a template sub-set structure and two
    pools of sites/structures.

    Given two pools of sites/structures, returns a new structure which has (1)
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

    species_coord_dict = {}  # avoid recomputing coords for each site
    species_idx_dict = {}
    for element in _fast_get_composition_from_sites(struct1_pool).elements:
        species_coord_dict[element.symbol], species_idx_dict[element.symbol] = (
            get_coords_and_idx_of_species(struct1_pool, element.symbol, frac_coords=False)
        )

    for (
        sub_site
    ) in template_struct.sites:  # get closest site in big supercell to site, using cartesian coords:
        closest_site_dict_idx = np.argmin(
            np.linalg.norm(species_coord_dict[sub_site.specie.symbol] - sub_site.coords, axis=1),
        )
        single_defect_subcell_sites.append(
            struct1_pool[species_idx_dict[sub_site.specie.symbol][closest_site_dict_idx]]
        )

    assert len(set(single_defect_subcell_sites)) == len(single_defect_subcell_sites)  # no repeats

    species_coord_dict = {}  # avoid recomputing coords for each site
    for element in _fast_get_composition_from_sites(struct2_pool).elements:
        species_coord_dict[element.symbol] = get_coords_and_idx_of_species(
            single_defect_subcell_sites, element.symbol, frac_coords=True  # frac_coords
        )[0]

    struct2_pool_dists_to_template_centre = struct2_pool.lattice.get_all_distances(
        struct2_pool.frac_coords,
        struct2_pool.lattice.get_fractional_coords(
            template_struct.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
        ),
    ).ravel()  # template centre is defect site in stenciling workflow
    largest_encompassed_cube_length = _largest_cube_length_from_matrix(template_struct.lattice.matrix)
    candidate_struct2_pool_species_sites: dict[str, list[PeriodicSite]] = {
        super_site.specie.symbol: [] for super_site in struct2_pool
    }
    for dist_to_template_centre, super_site in zip(struct2_pool_dists_to_template_centre, struct2_pool):
        # screen to sites outside defect WS radius, for efficiency:
        if dist_to_template_centre > largest_encompassed_cube_length * 0.49:  # 2% buffer (cube length / 2)
            candidate_struct2_pool_species_sites[super_site.specie.symbol].append(super_site)

    struct2_pool_site_min_dist_dict = {}
    for species_symbol, species_sites in candidate_struct2_pool_species_sites.items():
        dist_matrix = struct2_pool.lattice.get_all_distances(  # vectorised for fast computation
            species_coord_dict[species_symbol], [site.frac_coords for site in species_sites]
        )  # M x N
        min_dists = np.min(dist_matrix, axis=0)  # down columns
        struct2_pool_site_min_dist_dict.update(dict(zip(species_sites, min_dists)))

    # sort possible_bulk_outer_cell_sites by (largest) min dist to single_defect_subcell_sites:
    possible_bulk_outer_cell_sites = sorted(
        [site for sites in candidate_struct2_pool_species_sites.values() for site in sites],
        key=lambda x: struct2_pool_site_min_dist_dict[x],
        reverse=True,
    )
    bulk_outer_cell_sites = possible_bulk_outer_cell_sites[
        : int(len(struct2_pool) * (1 - 1 / num_super_supercells))
    ]

    return Structure.from_sites(single_defect_subcell_sites + bulk_outer_cell_sites)


def _get_superset_matrix_and_supercells(
    structure: Structure,
    target_supercell: Structure,
    defect_frac_coords: Optional[Union[np.ndarray[float], list[float]]] = None,
) -> tuple:  # tuple[np.ndarray[int], Structure, Structure, Structure] or tuple[np.ndarray[int], Structure]
    """
    Given a structure and a target supercell, return the transformation
    ('superset') matrix which makes all lattice vectors for the structure
    larger than or equal to the largest lattice vector in ``target_supercell``.

    Args:
        structure (Structure):
            The original structure for which to get the superset matrix
            that fully encompasses the target supercell.
        target_supercell (Structure):
            The target supercell.
        defect_frac_coords (Optional: Union[np.ndarray[float], list[float]]):
            The fractional coordinates of a defect site in the structure.
            If provided, will add an "X" marker (fake species) to this site
            in a copy of ``structure``, and additionally return the big
            supercell with copies of "X", and the original structure with "X".

    Returns:
        Union[tuple[np.ndarray[int], Structure, Structure, Structure], tuple[np.ndarray[int]]:
            - If ``defect_frac_coords`` is not provided, returns a tuple
              containing the superset matrix and the big supercell.
            - If ``defect_frac_coords`` is provided, returns a tuple
              containing the superset matrix, the big supercell, the big
              supercell with _repeated_ defect sites marked by "X", and
              a copy of the original structure with the defect site as "X".
    """
    min_cell_length = _get_all_encompassing_cube_length(target_supercell.lattice)

    # get supercell matrix which makes all lattice vectors for orig_supercell larger than min_cell_length:
    superset_matrix = np.ceil(min_cell_length / structure.lattice.abc)
    # could possibly use non-diagonal supercells to make this more efficient in some cases, but a lot of
    # work, shouldn't really contribute much to slowdowns, and only relevant in some rare cases (?)
    big_supercell = structure * superset_matrix

    if defect_frac_coords is None:
        return superset_matrix, big_supercell

    structure_with_X = structure.copy()  # get defect coords in big supercell:
    structure_with_X.append("X", defect_frac_coords, coords_are_cartesian=False)
    big_supercell_with_X = structure_with_X * superset_matrix

    return superset_matrix, big_supercell, big_supercell_with_X, structure_with_X


def _get_all_encompassing_cube_length(lattice: Lattice) -> float:
    """
    Get the smallest possible cube that fully encompasses the cell, _regardless
    of orientation_.

    This is determined by getting the 8 vertices of the cell and computing the
    max distance between any two vertices, giving the side length of an all-
    encompassing cube. Equivalent to getting the diameter of an encompassing
    sphere.

    Args:
        lattice (Lattice):
            The lattice to get the all-encompassing cube length for.

    Returns:
        float: The side length of the all-encompassing cube.
    """
    # get all vertices of the cell:
    cart_vertices = np.array([lattice.get_cartesian_coords(perm) for perm in product([0, 1], repeat=3)])
    # get 2D matrix of all distances between vertices:
    distances = np.linalg.norm(cart_vertices[:, None] - cart_vertices, axis=-1)

    return max(distances.flatten())


def is_within_frac_bounds(
    lattice: Lattice, cart_coords: Union[np.ndarray[float], list[float]], tol: float = 1e-5
) -> bool:
    """
    Check if a given Cartesian coordinate is inside the unit cell defined by
    the lattice object.

    Args:
        lattice (Lattice):
            ``Lattice`` object defining the unit cell.
        cart_coords (Union[np.ndarray[float], list[float]]):
            The Cartesian coordinates to check.
        tol (float):
            A tolerance (in Angstrom / cartesian units) for
            coordinates to be considered within the unit cell.
            If positive, expands the bounds of the unit cell
            by this amount, if negative, shrinks the bounds.

    Returns:
        bool:
            Whether the Cartesian coordinates are within the
            fractional bounds of the unit cell, accounting for
            ``tol``.
    """
    frac_coords = lattice.get_fractional_coords(cart_coords)
    frac_tols = np.array([tol, tol, tol]) / lattice.abc

    # Check if fractional coordinates are in the range [0, 1)
    return np.all((frac_coords + frac_tols >= 0) & (frac_coords - frac_tols < 1))
