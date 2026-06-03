"""
Utility functions to re-generate a relaxed defect structure in a different
supercell.
"""

import math
import warnings
from collections import Counter
from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
from tqdm import tqdm

from doped.analysis import defect_site_from_structures
from doped.core import DefectEntry
from doped.thermodynamics import _ensure_list
from doped.utils.configurations import apply_s2_to_s1_transformation, get_transformation_from_s2_to_s1
from doped.utils.efficiency import (
    Composition,
    Hashabledict,
    Lattice,
    PeriodicSite,
    Structure,
    _cached_Composition_init,
    _Composition__eq__,
)
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    check_atom_mapping_far_from_defect,
    get_coords_and_idx_of_species,
    get_defect_type_and_composition_diff,
    get_site_mapping_indices,
    get_wigner_seitz_radius,
)
from doped.utils.supercells import _largest_cube_length_from_matrix, min_dist
from doped.utils.symmetry import (
    SymmOp,
    apply_symm_op_to_site,
    apply_symm_op_to_struct,
    get_clean_structure,
    get_distance_matrix,
    get_sga,
    translate_structure,
)


def get_defect_in_supercell(
    defect_entry: (
        DefectEntry | tuple[Structure, Structure] | tuple[Structure, Structure, list | np.ndarray]
    ),
    target_supercell: Structure,
    check_bulk: bool = True,
    target_frac_coords: np.ndarray | list[float] | bool = True,
    edge_tol_range: float | range | list | np.ndarray | None = None,
    min_dist_tol_factor_range: float | range | list | np.ndarray | None = None,
    min_dist_warning_tol_factor: float = 0.9,
    orientation_template_radii_range: float | range | list | np.ndarray | None = None,
    show_pbar: bool = True,
) -> tuple[Structure, Structure]:
    """
    Re-generate a (relaxed) defect structure in a (arbitrarily) different
    supercell.

    This function takes the relaxed defect structure of the input
    |DefectEntry| (from ``DefectEntry.defect_supercell``), or input
    |Structure| objects (if ``defect_entry`` is a tuple, see argument
    descriptions) and re-generates it in the ``target_supercell`` structure
    (which may be smaller or larger than the original supercell), using the
    bulk supercell to intelligently pad out the additional missing positions in
    the new supercell as needed. The defect is placed at the closest possible
    position to ``target_frac_coords`` (default is the supercell centre =
    [0.5, 0.5, 0.5]). In most cases, the generated supercell should correspond
    to the same supercell basis / tiling as the input ``target_supercell``. In
    some cases, however, this is not possible (in which case a warning is
    thrown), and so this function also returns the corresponding bulk supercell
    for the generated supercell (which should be the same for each generated
    defect supercell given the same ``target_supercell`` and base supercell for
    ``defect_entry``, see notes below).

    ``target_supercell`` should be the same host crystal structure, just with
    different supercell dimensions. The stenciling algorithm is typically
    robust to small differences in lattice parameters (i.e. volume per atom)
    and bond lengths, but large differences in these between the original and
    target bulk supercells can cause issues.

    Briefly, this function works by:

    - Translating the defect site to the centre of the original supercell (and
      applying to the bulk supercell as well).
    - Identifying a super-supercell of the original supercell which fully
      encompasses the target supercell (regardless of orientation).
    - Generate this super-supercell, using one copy of the original defect
      supercell (``DefectEntry.defect_supercell``), with the rest of the sites
      (outside of the original defect supercell box, with the defect translated
      to the centre) populated using the bulk supercell
      (``DefectEntry.bulk_supercell``). Same for translated bulk supercell.
    - Orient these super-supercells to (try) match the orientation of
      ``target_supercell``, to (try) avoid outputting stenciled supercells with
      different primitive cell tiling (i.e. different structural bases) to the
      target supercell.
    - Translate the defect site in these super-supercells to the Cartesian
      coordinates of the centre of ``target_supercell``, then stencil out all
      sites in the ``target_supercell`` portion of the super-supercell,
      accounting for possible site displacements in the relaxed defect
      supercell (e.g. if ``target_supercell`` has a different shape and does
      not fully encompass the original defect supercell).
      This is done by scanning over possible combinations of sites near the
      boundary regions of the ``target_supercell`` portion (using ``edge_tol``
      and ``min_dist_tol_factor`` etc), and identifying the combination which
      maximises the minimum inter-atomic distance in the new supercell (i.e.
      the most bulk-like arrangement). Same for bulk super-supercell.
    - Re-orient this new stenciled supercell (and the corresponding bulk) to
      (attempt to) match the orientation and site positions of
      ``target_supercell``.
    - If ``target_frac_coords`` is not ``False``, scan over all symmetry
      operations of ``target_supercell`` and apply that which places the defect
      site closest to ``target_frac_coords``, to both the defect and bulk
      supercells.

    Note: While this function tries to ensure that the generated defect/bulk
    supercell position basis exactly matches that of ``target_supercell``, this
    is `not` guaranteed. A mismatch can arise due to different tiling of
    primitive cells within ``target_supercell`` and the original defect
    supercell, which are symmetry-equivalent for the bulk (pristine) supercell
    but not for the defect supercell (due to the broken periodicity/symmetry).
    The generated supercell `is` guaranteed to have the exact same lattice
    parameters etc. This is perfectly fine if it occurs (in which case a
    warning will be thrown, unless ``check_bulk`` is ``False``), just will
    require the use of a matching bulk/reference supercell when parsing (rather
    than the input ``target_supercell``) to avoid any issues with finite-size
    corrections and defect site-matching -- ``doped`` will also throw a warning
    about this when parsing if a non-matching bulk supercell is used anyway.
    This function returns the corresponding bulk supercell which should be used
    for parsing defect calculations with the generated supercell. Of course, if
    generating multiple defects in the same ``target_supercell`` with this
    issue occurring, only one such bulk supercell calculation should be
    required (`should` correspond to the same bulk supercell in each case).

    We note that the algorithm employed here is not guaranteed to be
    deterministic. It can depend on site orderings, small differences in cell
    definitions etc. For these small differences, however, the returned
    stenciled supercell should still be effectively the same, just with small
    differences in atomic positions near the cell boundaries due to
    defect-induced strain (which for reasonably large original & stenciling
    supercells should be negligible).

    Args:
        defect_entry (|DefectEntry| | tuple[|Structure|, |Structure|(, np.ndarray)?]):
            A |DefectEntry| object for which to re-generate the relaxed
            structure (taken from ``DefectEntry.defect_supercell``) in the
            ``target_supercell`` lattice. Alternatively, a tuple of
            ``(defect_supercell, bulk_supercell)`` structures can be provided,
            in which case the defect fractional coordinates are determined
            automatically (by comparing the two supercells). The defect
            fractional coordinates can optionally be provided as a third
            element with the tuple input option (as a ``numpy`` array or list),
            to skip auto-determination in this case.
        target_supercell (|Structure|):
            The supercell structure to re-generate the relaxed defect structure
            in.
        check_bulk (bool):
            Whether to check if the generated defect/bulk supercells have
            different atomic position bases to ``target_supercell`` (as
            described above) -- if so, a warning will be printed (unless
            ``check_bulk`` is ``False``). Default is ``True``.
        target_frac_coords (np.ndarray | list[float] | bool):
            The fractional coordinates to target for defect placement in the
            new supercell. If just set to ``True`` (default), will try to place
            the defect nearest to the centre of the superset cell (i.e.
            ``target_frac_coords = [0.5, 0.5, 0.5]``), as is default in
            ``doped`` defect generation. Note that defect placement is harder
            in this case than in generation with |DefectsGenerator|, as we
            are not starting from primitive cells, and we are working with
            relaxed geometries. If ``False``, does not attempt any defect
            re-positioning.
        edge_tol_range (float | range | list | np.ndarray | None):
            A range or list of tolerances (in Angstrom) for site displacements
            at the edge of the stenciled supercell to scan over, when
            determining the best match of sites to stencil out in the new
            supercell (of ``target_supercell`` dimension). Default is ``None``,
            in which case the default range is used (0.2 to 4 Å in 0.2 Å
            increments). Once a stenciling solution which satisfies the
            ``min_dist_tol_factor`` tolerance is found for a given
            ``edge_tol``, the search is terminated. See
            ``stencil_target_cell_from_big_cell`` for further details.
        min_dist_tol_factor_range (float | range | list | np.ndarray | None):
            A range or list of tolerance factors for the minimum interatomic
            distance (relative to the ``bulk_min_bond_length``) at the edge
            region of the stenciled defect supercell to scan over, when
            determining the best match of sites to stencil out in the new
            supercell. Default is ``None``, in which case the default range is
            used (0.95 to 0.5 in 0.05 increments). See
            ``stencil_target_cell_from_big_cell`` for further details.
        min_dist_warning_tol_factor (float):
            A tolerance factor for the minimum interatomic distance (relative
            to the ``bulk_min_bond_length``) in the stenciled defect supercell
            to use as a warning threshold. If the minimum interatomic distance
            near the edge of the stenciled defect supercell is less than this
            threshold, a warning is issued. Default is 0.9 (i.e. 90% of the
            ``bulk_min_bond_length``).
        orientation_template_radii_range (float | range | list | np.ndarray | None):
            A range or list of scale factors (relative to the Wigner-Seitz
            radius of ``target_supercell``) to scan over when constructing the
            template sub-structures of ``target_supercell`` and the bulk
            super-supercell used for pre-stenciling orientation matching (to
            try to ensure matching supercell atomic bases (i.e. tiling of
            primitive cells in the supercells) in the output stenciled cells
            with ``target_supercell``). Default is ``None``, in which case the
            default test range of ``[0.8, 1.0, 0.6, 0.4, 1.2]`` is used. It is
            expected that this parameter should rarely be required to tune.
        show_pbar (bool):
            Whether to show a ``tqdm`` progress bar. Default is ``True``.

    Returns:
        tuple[Structure, Structure]:
            The re-generated defect supercell in the ``target_supercell``
            lattice, and the corresponding bulk/reference supercell for the
            generated defect supercell (see explanations above).
    """
    # Note to self; using Pycharm breakpoints throughout is likely easiest for debugging
    if show_pbar:
        pbar = tqdm(
            total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
        )  # tqdm progress bar. 100% is completion
        pbar.set_description("Getting super-supercell (relaxed defect + bulk sites)")
    else:
        pbar = None

    bulk_mismatch_warning = False

    try:
        orig_defect_frac_coords: np.ndarray | tuple[float, float, float] | None
        if isinstance(defect_entry, tuple):
            orig_supercell = defect_entry[0].copy()
            orig_bulk_supercell = defect_entry[1].copy()
            if len(defect_entry) > 2:
                orig_defect_frac_coords = np.array(defect_entry[2])
            else:
                defect_site = defect_site_from_structures(
                    orig_supercell, orig_bulk_supercell, _parameter_order_warn=False
                )
                assert isinstance(defect_site, PeriodicSite)
                orig_defect_frac_coords = defect_site.frac_coords
        else:
            orig_supercell = _get_defect_supercell(defect_entry)
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
        trans_orig_supercell = translate_structure(orig_supercell, orig_def_to_centre, frac_coords=True)
        trans_orig_bulk_supercell = translate_structure(
            orig_bulk_supercell, orig_def_to_centre, frac_coords=True
        )

        # get big_supercell, which is an expanded version of trans_orig_supercell so that _each_ lattice
        # vector is now bigger than the _largest_ lattice vector in target_supercell (so that
        # target_supercell is fully encompassed by big_supercell):
        superset_matrix = _get_superset_matrix_and_supercells(trans_orig_supercell, target_supercell)

        # add "X" marker (fake species) to defect site in trans_orig_supercell"
        trans_orig_supercell.append("X", [0.5, 0.5, 0.5], coords_are_cartesian=False)
        big_defect_supercell_with_X = trans_orig_supercell * superset_matrix
        big_bulk_supercell = trans_orig_bulk_supercell * superset_matrix  # get big bulk supercell

        # this big_defect_supercell_with_X is with the defect now repeated in it, but we want it with just
        # one defect, so only take the first repeated defect supercell, and then the rest of the sites from
        # the expanded bulk supercell:
        # only keep atoms in big_defect_supercell_with_X which are within the original supercell bounds:
        big_defect_supercell = _get_matching_sites_from_s1_then_s2(
            template_struct=trans_orig_supercell,
            struct1_pool=big_defect_supercell_with_X,
            struct2_pool=big_bulk_supercell,
        )

        # now we try to preemptively reorient the big supercell to match the orientation of
        # target_supercell (to try avoid output stenciled supercells with different tiling, requiring
        # additional bulk supercell calculations).
        # First, we reduce to only atoms closest to the centre for both target and super-supercells, to
        # greatly reduce comp costs with site-matching enormous supercells; for this, we scan over possible
        # radii for the sampling sphere as some choices give matches (transformations) and some fail:
        orientation_template_radii_range = _ensure_list(
            orientation_template_radii_range
            if orientation_template_radii_range is not None
            else [0.8, 1.0, 0.6, 0.4, 1.2]  # scale factors of get_wigner_seitz_radius(target_supercell)
        )
        assert isinstance(orientation_template_radii_range, list | np.ndarray)  # typing
        wsr_target = get_wigner_seitz_radius(target_supercell)
        trans_to_match_target = None
        fake_target_supercell_w_big_supercell_lattice = None
        for r_factor in orientation_template_radii_range:
            fake_target_supercell_sites = target_supercell.get_sites_in_sphere(
                target_supercell.lattice.get_cartesian_coords([0.5, 0.5, 0.5]),
                r=r_factor * wsr_target,
            )
            if not fake_target_supercell_sites:
                continue
            fake_target_supercell_w_big_supercell_lattice = Structure(
                big_bulk_supercell.lattice,
                [site.species for site in fake_target_supercell_sites],
                [site.coords for site in fake_target_supercell_sites],
                coords_are_cartesian=True,
            )
            fake_big_supercell = Structure.from_sites(
                big_bulk_supercell.get_sites_in_sphere(
                    big_bulk_supercell.lattice.get_cartesian_coords([0.5, 0.5, 0.5]),
                    r=r_factor * wsr_target / 0.8,  # slightly larger r for fake big supercell
                )
            )
            trans_to_match_target = get_transformation_from_s2_to_s1(
                fake_target_supercell_w_big_supercell_lattice,
                fake_big_supercell,
                allow_subset=True,  # allow subset to match, likely different number of atoms here
                max_stol=0.5,
            )
            if trans_to_match_target is not None:
                break

        if trans_to_match_target is not None:
            oriented_big_bulk_supercell = _orient_to_match_target(
                big_bulk_supercell, fake_target_supercell_w_big_supercell_lattice, trans_to_match_target
            )
            oriented_big_defect_supercell = _orient_to_match_target(
                big_defect_supercell, fake_target_supercell_w_big_supercell_lattice, trans_to_match_target
            )
        else:
            oriented_big_bulk_supercell = big_bulk_supercell
            oriented_big_defect_supercell = big_defect_supercell

        if pbar is not None:
            pbar.update(20)  # 20% of progress bar
            pbar.set_description("Getting sites in border region")

        # translate structure to put defect at the centre of the big supercell (w/frac_coords)
        big_supercell_defect_site = next(
            s for s in oriented_big_defect_supercell.sites if s.specie.symbol == "X"
        )
        def_to_centre = np.array([0.5, 0.5, 0.5]) - big_supercell_defect_site.frac_coords
        oriented_big_defect_supercell = translate_structure(
            oriented_big_defect_supercell, def_to_centre, frac_coords=True
        )
        oriented_big_bulk_supercell = translate_structure(
            oriented_big_bulk_supercell, def_to_centre, frac_coords=True
        )

        # get all atoms in big supercell within the cartesian bounds of the target_supercell supercell,
        # accounting for positional noise:
        new_bulk_supercell = stencil_target_cell_from_big_cell(
            oriented_big_bulk_supercell,
            target_supercell,
            target_composition=target_supercell.composition,
            bulk_min_bond_length=bulk_min_bond_length,
            edge_tol_range=1e-3,
            min_dist_tol_factor_range=0.99,
            pbar=None,
        )  # shouldn't need `edge_tol`, should be much faster than defect supercell stencil
        new_defect_supercell = stencil_target_cell_from_big_cell(
            oriented_big_defect_supercell,
            target_supercell,
            target_composition=(
                target_supercell.composition + orig_supercell.composition - orig_bulk_supercell.composition
            ),
            bulk_min_bond_length=bulk_min_bond_length,
            edge_tol_range=edge_tol_range,
            min_dist_tol_factor_range=min_dist_tol_factor_range,
            min_dist_warning_tol_factor=min_dist_warning_tol_factor,
            pbar=pbar,
        )
        if pbar is not None:
            pbar.update(15)  # 55% of progress bar
            pbar.set_description("Ensuring matching orientation w/target_supercell")

        # now we try to re-orient the new defect (and bulk) supercells to match the orientation of the
        # input ``target_supercell``. First we orient the generated _bulk_ supercell to match the
        # ``target_supercell``, to try to ensure consistency in the generated supercells.

        # Note: this function is typically the main bottleneck in this workflow. We have already
        # optimised the underlying |StructureMatcher| workflow in many ways (caching,
        # fast structure/site/composition comparisons, skipping comparison of defect neighbourhood to
        # reduce requisite ``stol`` etc; being many orders of magnitude faster than the base
        # ``pymatgen`` |StructureMatcher|), however the ``_cart_dists()`` function call (used by
        # ``orient_s2_like_s1``, ``get_transformation_from_s2_to_s1``) is still quite expensive,
        # especially with large structures with significant noise in the atomic positions...
        trans = get_transformation_from_s2_to_s1(target_supercell, new_bulk_supercell, max_stol=5)
        # Note: Here we use a relatively large max stol, because we want to get `some` transformation
        # regardless of whether a perfect match is possible, as we want a consistent output bulk supercell
        # (in the case of the stenciled bulk supercell not matching the target supercell), so that only one
        # additional bulk supercell calculation should be required in this worst case scenario. This
        # approach gives us a consistent/deterministic output. In theory, we could be faster by breaking at
        # a lower stol and using an alternative deterministic setting for the new bulk supercell, but not
        # straightforward and use of fast algorithms and LRU caching minimises the cost here
        if trans:
            oriented_new_bulk_supercell = _orient_to_match_target(
                new_bulk_supercell, target_supercell, trans
            )
            oriented_new_defect_supercell = _orient_to_match_target(
                new_defect_supercell, target_supercell, trans
            )
        else:
            warnings.warn(  # shouldn't happen for most cases...
                "No mapping from the (bulk) stenciled cell to ``target_supercell`` could be found. This "
                "may indicate a severe issue with stenciling for this defect. Please report this issue to "
                "the developers if there are no obvious causes of this."
            )
            oriented_new_bulk_supercell = new_bulk_supercell
            oriented_new_defect_supercell = new_defect_supercell

        oriented_new_defect_site = next(  # get defect site and remove X from sites
            oriented_new_defect_supercell.pop(i)
            for i, site in enumerate(oriented_new_defect_supercell.sites)
            if site.specie.symbol == "X"
        )
        if pbar is not None:
            pbar.update(35)  # 90% of progress bar

        if target_frac_coords is not False:
            # Scan symm_ops to try and place defect closest to target_frac_coords as possible
            if pbar is not None:
                pbar.set_description("Placing defect closest to target_frac_coords")

            target_symm_op = _scan_symm_ops_to_place_site_closest_to_frac_coords(
                target_supercell, oriented_new_defect_site, target_frac_coords
            )

            def _apply_symm_op_and_clean_structure(structure: Structure, symm_op: SymmOp) -> Structure:
                return get_clean_structure(
                    apply_symm_op_to_struct(symm_op, structure, fractional=True, rotate_lattice=False),
                )

            oriented_new_defect_supercell = _apply_symm_op_and_clean_structure(
                oriented_new_defect_supercell, target_symm_op
            )
            oriented_new_bulk_supercell = _apply_symm_op_and_clean_structure(
                oriented_new_bulk_supercell, target_symm_op
            )

        if check_bulk:  # check orientation
            bulk_mismatch_warning = not check_atom_mapping_far_from_defect(
                oriented_new_bulk_supercell,  # could use defect or bulk supercell here, bulk cleaner
                target_supercell,
                np.array([0.5, 0.5, 0.5]),  # 'defect' site choice irrelevant here
                warning=False,
            )

        if pbar is not None:
            pbar.update(pbar.total - pbar.n)  # set to 100% of progress bar

    finally:
        if pbar is not None:
            pbar.close()

    if bulk_mismatch_warning:  # print warning after closing pbar; cleaner
        warnings.warn(
            "Note that the atomic position basis of the generated defect/bulk supercell differs from that "
            "of the ``target_supercell``. This is typically due to different tiling of primitive cells "
            "within ``target_supercell`` and the original defect supercell, which are symmetry-equivalent "
            "for the bulk (pristine) supercell but not for the defect supercell (due to the broken "
            "periodicity/symmetry). For accurate finite-size charge corrections when parsing, a matching "
            "bulk and defect supercell should be used, and so the matching bulk supercell for the "
            "generated defect supercell (also returned by this function) should be used for its reference "
            "host cell calculation.",
        )
        # Two symmetry-equivalent supercells can have identical supercell lattice vectors but different
        # atomic positions, due to different tiling of primitive cells within the same total supercell --
        # think e.g. parallelepids tiled along the long or short sides in a larger (supercell)
        # parallelepid; so e.g. the max distance between atoms _in the same cell_ can differ etc -- which
        # causes them to become symmetry-inequivalent upon periodicity-breaking (e.g. introduction of a
        # defect)
        # When these different primitive cells are tiled into supercells, the supercell shapes end up
        # identical but the atoms sit at different Cartesian positions. Crucially, the mapping between the
        # two is not a single rigid-body transformation (rotation + translation) applied uniformly to all
        # atoms.

        # if the original (defect entry) bulk supercell and target supercell differ in bond lengths,
        # then this can cause some irregularities in bonding within the output bulk supercell (as we
        # don't rescale when stenciling, so we take original Cartesian coordinates with the target
        # supercell lattice parameters) here we warn if the minimum bond length in the output bulk
        # supercell is less than 99% of the original bond length, _and_ if we have a bulk mismatch
        # warning (as otherwise the output bulk cell shouldn't be used anyway):
        new_bulk_min_bond_length = min_dist(oriented_new_bulk_supercell)
        if new_bulk_min_bond_length < bulk_min_bond_length * 0.995:
            warnings.warn(
                f"Note that the stenciled `bulk` supercell has a minimum interatomic distance of "
                f"{new_bulk_min_bond_length:.2f} Å, smaller than 99.5% of the original bond length "
                f"({bulk_min_bond_length:.2f} Å). This is typically the result of different bond lengths "
                f"/ interatomic distances between the original bulk and target supercells, which affects "
                f"the stenciling process. If the difference is relatively small, then this is easily "
                f"resolved by relaxing the output bulk supercell to re-equilibrate the interatomic "
                f"distances."
            )

    assert (
        get_defect_type_and_composition_diff(
            orig_supercell, orig_bulk_supercell, _parameter_order_warn=False
        )[1]
        == get_defect_type_and_composition_diff(
            oriented_new_defect_supercell, oriented_new_bulk_supercell, _parameter_order_warn=False
        )[1]
    ), (
        f"Output composition ({oriented_new_defect_supercell.composition}) does not match expected "
        f"stenciled defect composition, indicating fatal issues with stenciling here!"
    )
    return oriented_new_defect_supercell, oriented_new_bulk_supercell


def _scan_symm_ops_to_place_site_closest_to_frac_coords(
    symm_ops: Structure | Sequence[SymmOp],
    site: PeriodicSite,
    target_frac_coords: np.ndarray | list[float] | None = None,
) -> SymmOp:
    """
    Given either a list of symmetry operations or a structure (to extract
    symmetry operations from), scan over all symmetry operations to find the
    one which places the provided ``site`` closest to the target fractional
    coordinates.

    Args:
        symm_ops (|Structure| | Sequence[SymmOp]):
            Either a list of symmetry operations or a structure from which to
            extract symmetry operations.
        site (|PeriodicSite|):
            The site to place closest to the target fractional coordinates.
        target_frac_coords (np.ndarray | list[float] | None):
            The target fractional coordinates to place the site closest to.
            Default is ``None``, in which case the site is placed closest to
            the centre of the supercell (i.e. [0.5, 0.5, 0.5]).
    """
    if isinstance(symm_ops, Structure):
        sga = get_sga(symm_ops)
        symm_ops = sga.get_symmetry_operations()

    target_frac_coords = [0.5, 0.5, 0.5] if target_frac_coords is None else target_frac_coords

    # get transformed site for each possible ``symm_op``
    symm_op_pos_dict = {}
    for i, symm_op in enumerate(symm_ops):  # should check if frac or cartesian is faster
        symm_opped_site = apply_symm_op_to_site(symm_op, site, fractional=True, rotate_lattice=False)
        symm_opped_site.to_unit_cell(in_place=True)  # faster with in_place
        symm_op_pos_dict[i] = symm_opped_site.frac_coords

    # get symm_op which puts defect closest to target_frac_coords:
    closest_site = min(symm_op_pos_dict.items(), key=lambda x: np.linalg.norm(x[1] - target_frac_coords))
    return symm_ops[closest_site[0]]


def stencil_target_cell_from_big_cell(
    big_supercell: Structure,
    target_supercell: Structure,
    target_composition: Composition | None,
    edge_tol_range: float | range | list | np.ndarray | None = None,
    bulk_min_bond_length: float | None = None,
    min_dist_tol_factor_range: float | range | list | np.ndarray | None = None,
    min_dist_warning_tol_factor: float = 0.9,
    pbar: tqdm | None = None,
) -> Structure:
    """
    Given the input ``big_supercell`` and ``target_supercell`` (which should be
    fully encompassed by the former), stencil out the sites in
    ``big_supercell`` which correspond to the sites in ``target_supercell``
    (i.e. are within the Cartesian bounds of ``target_supercell``).

    Note that this function assumes that the defect is roughly centred within
    ``big_supercell`` (i.e. near [0.5, 0.5, 0.5])! The midpoints of
    ``target_supercell`` and ``big_supercell`` are then aligned within this
    function, before stenciling.

    We need to ensure the appropriate number of sites (and their composition)
    are taken, and that the sites we choose are appropriate for the new
    supercell (i.e. that if we have e.g. an in-plane contraction, we don't take
    duplicate atoms that then correspond to tiny inter-atomic distances in the
    new supercell due to imperfect stenciling under PBC -- so we can't simply
    take the atoms that are closest to the defect). So, here we scan over
    possible choices of atoms to include, and take the combination which
    maximises the `minimum` inter-atomic distance in the new supercell, when
    accounting for PBCs.

    Note that differences between the lattice parameters (volumes per atom) of
    ``big_supercell`` and ``target_supercell`` can cause issues with the
    stenciling approach herein, particularly when ``target_composition`` is not
    supplied and/or for defective ``big_supercell`` cells.

    Args:
        big_supercell (|Structure|):
            The supercell structure which fully encompasses
            ``target_supercell``, from which to stencil out the sites.
        target_supercell (|Structure|):
            The supercell structure giving the cell dimensions to stencil out
            from ``big_supercell``.
        target_composition (|Composition| | None):
            Expected composition of the output stenciled cell (used to
            determine candidate sites to stencil out). Auto-determined by
            comparing ``big_supercell`` and ``target_supercell`` if ``None``
            (default).
        edge_tol_range (float | range | list | np.ndarray | None):
            A range or list of tolerances (in Angstrom) for site displacements
            at the edge of the stenciled supercell to scan over, when
            determining the best match of sites to stencil out in the new
            supercell (of ``target_supercell`` dimension). Default is ``None``,
            in which case the default range is used (0.2 to 4 Å in 0.2 Å
            increments).
            In the stenciling search, we scan over ``edge_tol_range`` first,
            before iterating over ``min_dist_tol_factor_range``.
            Once a stenciling solution which satisfies the
            ``min_dist_tol_factor`` tolerance is found for a given
            ``edge_tol``, the search is terminated.
        bulk_min_bond_length (float):
            The minimum interatomic distance in the bulk supercell. Default is
            ``None``, in which case it is calculated from ``target_supercell``.
        min_dist_tol_factor_range (float | range | list | np.ndarray | None):
            A range or list of tolerance factors for the minimum interatomic
            distance (relative to the ``bulk_min_bond_length``) at the edge
            region of the stenciled defect supercell to scan over, when
            determining the best match of sites to stencil out in the new
            supercell. Default is ``None``, in which case the default range is
            used (0.95 to 0.5 in 0.05 increments).
            In the stenciling search, we scan over ``edge_tol_range`` first,
            before iterating over ``min_dist_tol_factor_range``.
            Once a stenciling solution which satisfies the
            ``min_dist_tol_factor`` tolerance is found for a given
            ``edge_tol``, the search is terminated.
        min_dist_warning_tol_factor (float):
            A tolerance factor for the minimum interatomic distance (relative
            to the ``bulk_min_bond_length``) in the stenciled defect supercell
            to use as a warning threshold. If the minimum interatomic distance
            near the edge of the stenciled defect supercell is less than this
            threshold, a warning is issued. Default is 0.9 (i.e. 90% of the
            ``bulk_min_bond_length``).
        pbar (tqdm):
            ``tqdm`` progress bar object to update (for internal ``doped``
            usage). Default is ``None``.

    Returns:
        Structure: The stenciled supercell structure.
    """
    edge_tol_range = _ensure_list(edge_tol_range if edge_tol_range is not None else np.arange(0.2, 4, 0.2))
    min_dist_tol_factor_range = _ensure_list(
        min_dist_tol_factor_range if min_dist_tol_factor_range is not None else np.arange(0.95, 0.5, -0.05)
    )
    assert isinstance(edge_tol_range, list | np.ndarray)
    assert isinstance(min_dist_tol_factor_range, list | np.ndarray)

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

    if target_composition is None:
        # get target composition accounting for defect presence in big supercell:
        num_sc = big_supercell.volume / target_supercell.volume  # may be fractional
        big_supercell_comp_wout_X = big_supercell.copy().remove_species(["X"]).composition
        target_composition = big_supercell_comp_wout_X - ((num_sc - 1) * target_supercell.composition)
        target_composition = Composition({k: round(v) for k, v in target_composition.items()})

    # here we stencil out sites from the big supercell using the ``target_supercell`` cell (i.e. stencil)
    # for the bulk supercell this is trivial, as we can just take the ``target_supercell`` box as is, and
    # it will have the correct composition (assuming the unit cell is the same in both)
    # for defective supercells (or slightly different bulk supercells, which are also handled natively
    # here), there can be noise/displacements in the atomic positions, which may be anisotropic etc, such
    # that the ``target_supercell`` stencil may miss some sites or include additional ones.
    # Thus, we need to be smart and account for positional noise in this stenciling approach.

    # To do this, we use an 'edge tolerance' with the ``_get_candidate_supercell_sites`` function to
    # section out the big supercell sites into:
    # - ``def_new_supercell_sites``: Super-supercell sites within ``target_supercell`` minus ``edge_tol``
    # - ``def_new_supercell_sites_to_check``: Sites in ``def_new_supercell_sites`` within
    #   ``[max(bulk_min_bond_length, edge_tol*2), edge_tol]`` from the target cell edge, used to check for
    #   site overlaps/displacements.
    # - ``candidate_new_supercell_sites``: Candidate sites for the new supercell, within +/-``edge_tol`` of
    #   the target cell edge, used to determine which sites to include in new supercell, by checking site
    #   overlaps/displacements (combined with ``def_new_supercell_sites_to_check``) and target composition.

    # We then need to determine the subset of ``candidate_new_supercell_sites`` which (1; easy) matches the
    # correct composition for our new target supercell, and (2; harder) which minimises discontinuities at
    # the supercell boundary (i.e. avoiding (nearly-)overlapping sites, which could occur due to
    # periodically-repeated atomic displacements combined with stenciling of a different supercell
    # shape/size). For (2), one could try to be very smart and track the atomic displacements in the defect
    # supercell as a function of distance to defect, or even for each specific site, keep track in the big
    # supercell, and try determine when the best match has been obtained by comparing max site displacement
    # near the edge to the known bulk-vs-defect displacement at this range, but would be tricky to
    # implement, likely not much (if at all) better (as different combinations of supercell shapes will
    # give different strain fields, such that the best choice of atoms near the stencil boundaries is
    # not necessarily immediately obvious) and in practice some basic displacement analysis should suffice.
    # Here for each tested ``edge_tol``, we take the set of candidate/variable sites, remove any which are
    # overlapping (within 50% of the bulk bond length / min distance; see ``_remove_overlapping_sites``),
    # from these take the site combinations which match the target composition, determine that which
    # minimises structural noise by giving the largest minimum interatomic distance in the near-edge
    # region. If the min interatomic distance here is greater than
    # ``bulk_min_bond_length * min_dist_tol_factor`` then this is treated as an acceptable solution

    # iterate over edge_tol first, then min_dist_tol_factor:
    for min_dist_tol_factor, edge_tol in product(min_dist_tol_factor_range, edge_tol_range):
        try:
            (
                def_new_supercell_sites,
                def_new_supercell_sites_to_check,
                candidate_new_supercell_sites,
            ) = _get_candidate_supercell_sites(
                big_supercell, target_supercell, edge_tol, bulk_min_bond_length
            )
            # Determine the composition of sites required to add to ``def_new_supercell_sites`` and match
            # the target composition:
            def_new_supercell_sites_struct = Structure.from_sites(def_new_supercell_sites)
            def_new_supercell_sites_struct.remove_species("X")  # def_new_supercell_sites also has X in it
            def_new_supercell_sites_comp = def_new_supercell_sites_struct.composition
            target_candidate_composition = (
                target_composition - def_new_supercell_sites_comp
            )  # composition to add
            num_sites_up_for_grabs = int(
                sum(target_candidate_composition.values())
            )  # number of sites to be placed
            candidate_new_supercell_sites = _remove_overlapping_sites(
                candidate_new_supercell_sites=candidate_new_supercell_sites,
                def_new_supercell_sites_to_check=def_new_supercell_sites_to_check,
                bulk_min_bond_length=bulk_min_bond_length,
                pbar=pbar,
            )

            if len(candidate_new_supercell_sites) < num_sites_up_for_grabs:
                raise RuntimeError(
                    f"Too little candidate sites ({len(candidate_new_supercell_sites)}) to match target "
                    f"composition ({num_sites_up_for_grabs} sites to be placed). Aborting."
                )
            num_combos = math.comb(len(candidate_new_supercell_sites), num_sites_up_for_grabs)
            if num_combos > 1e10:
                raise RuntimeError(
                    "Far too many possible site combinations to check, indicating a code failure. "
                    "Aborting, please report this case to the developers!"
                )

            if pbar is not None:
                pbar.set_description(
                    f"Calculating best match (edge_tol = {edge_tol} Å, possible combos = {num_combos})"
                )  # 40% of pbar

            species_symbols = [site.specie.symbol for site in candidate_new_supercell_sites]
            min_interatomic_distances_tuple_combo_dict = {}
            idx_combos = list(
                combinations(range(len(candidate_new_supercell_sites)), num_sites_up_for_grabs)
            )

            if idx_combos and idx_combos != [()]:
                for idx_combo in idx_combos:
                    if _Composition__eq__(
                        _cached_Composition_init(
                            Hashabledict(Counter([species_symbols[i] for i in idx_combo]))
                        ),
                        target_candidate_composition,
                    ):
                        # could early break cases where the distances are too small? if a bottleneck,
                        # currently not (supercell re-orientation is) -- likely to be a bottleneck with
                        # the latest approach...
                        # And/or could loop over subsets of each combo first, culling any which break this

                        idx_combo_sites_in_target = [candidate_new_supercell_sites[i] for i in idx_combo]
                        fake_candidate_struct_sites = (
                            def_new_supercell_sites_to_check + idx_combo_sites_in_target
                        )
                        frac_coords = [site.frac_coords for site in fake_candidate_struct_sites]
                        distance_matrix = get_distance_matrix(frac_coords, target_supercell.lattice)
                        sorted_distances = np.sort(distance_matrix.flatten())
                        min_interatomic_distances_tuple_combo_dict[
                            tuple(sorted_distances[len(frac_coords) : len(frac_coords) + 10])
                            # take the 10 smallest distances for comparison purposes
                        ] = idx_combo_sites_in_target

                # get the candidate structure with the largest minimum interatomic distance:
                min_interatomic_distances_tuple_combo_list = sorted(
                    min_interatomic_distances_tuple_combo_dict.items(),
                    key=lambda x: x[0],
                    reverse=True,
                )

                if not min_interatomic_distances_tuple_combo_list:
                    raise RuntimeError("No matching compositions for stenciling found.")

                # raise RuntimeError and dynamically increase ``edge_tol`` (first, then decrease
                # ``min_dist_tol_factor``) if resulting min_dist is too small:
                idx_combo_min_dist = min_interatomic_distances_tuple_combo_list[0][0][0]
                if idx_combo_min_dist < (bulk_min_bond_length * min_dist_tol_factor):
                    raise RuntimeError(
                        f"Minimum interatomic distance ({idx_combo_min_dist:.2f} Å) near the edge (within "
                        f"{edge_tol:.2f} Å) of the target cell is less than the minimum distance "
                        f"tolerance ({bulk_min_bond_length * min_dist_tol_factor:.2f} Å), indicating a "
                        f"fatal issue with the stenciling process. Aborting."
                    )

                new_supercell_sites = def_new_supercell_sites + list(
                    min_interatomic_distances_tuple_combo_list[0][1]
                )

                min_dist_warning_tol = bulk_min_bond_length * min_dist_warning_tol_factor
                if idx_combo_min_dist < min_dist_warning_tol:
                    inner_range = max(bulk_min_bond_length, edge_tol * 2)
                    warnings.warn(
                        f"Note that the generated stenciled structure has a minimum interatomic distance "
                        f"of {idx_combo_min_dist:.2f} Å near the cell edge (within {inner_range:.2f} Å), "
                        f"smaller than the warning threshold ({min_dist_warning_tol_factor} of the bulk "
                        f"minimum interatomic distance ({bulk_min_bond_length:.2f} Å) = "
                        f"{min_dist_warning_tol:.2f} Å). Some remnant structural noise is of course "
                        f"expected when stenciling with relatively small original/target supercells, so "
                        f"consider if this is reasonable for your system!"
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
            break

        except (RuntimeError, ValueError) as e:
            if edge_tol == edge_tol_range[-1] and min_dist_tol_factor == min_dist_tol_factor_range[-1]:
                raise e  # end of the line and no matches; raise error

            if pbar is not None:
                pbar.n = 20  # decrease pbar progress back to 20%
                pbar.refresh()
                pbar.set_description(
                    f"Trying edge_tol = {edge_tol:.2f} Å, min_dist_tol_factor = {min_dist_tol_factor:.2f}"
                )
            continue

    return new_supercell


def _orient_to_match_target(
    structure: Structure, target_structure: Structure, transformation: tuple
) -> Structure:
    return apply_s2_to_s1_transformation(
        target_structure,
        structure,
        supercell_matrix=transformation[0],
        trans_vector=transformation[1],
        mapping=transformation[2],
        # new_lattice="struct1",  # struct1 by default, no warning if not possible
        ignored_species=["X"],  # ignore X site
    )


def _get_candidate_supercell_sites(
    big_supercell: Structure,
    target_supercell: Structure,
    edge_tol: float = 0.2,
    bulk_min_bond_length: float | None = None,
) -> tuple[list[PeriodicSite], list[PeriodicSite], list[PeriodicSite]]:
    """
    Get all atoms in ``big_supercell`` which are within the Cartesian bounds of
    the ``target_supercell`` supercell.

    We need to ensure the same number of sites, and that the sites we choose
    are appropriate for the new supercell (i.e. that if we have e.g. an in-
    plane contraction, we don't take duplicate atoms that then correspond to
    tiny inter-atomic distances in the new supercell due to imperfect
    stenciling under PBC -- so we can't just take the atoms that are closest to
    the defect). So, this function determines the possible sites to include in
    the new supercell, the sites which are definitely in the target supercell,
    and the sites which are near the bordering regions of the target supercell
    and so may or may not be included (i.e. ``candidate_new_supercell_sites``),
    using the provided ``edge_tol``.

    Note that this function assumes that the defect (or any significant atomic
    displacements) is (are) roughly centred within ``big_supercell`` (i.e. near
    ``[0.5, 0.5, 0.5]``)!

    Args:
        big_supercell (|Structure|):
            The super-supercell with a single defect supercell and rest of the
            sites populated by the bulk supercell.
        target_supercell (|Structure|):
            The supercell structure to re-generate the relaxed defect structure
            in.
        edge_tol (float):
            A tolerance (in Angstrom) for site displacements at the edge of the
            ``target_supercell`` supercell, when determining the best match of
            sites to stencil out in the new supercell. Default is 0.2 Angstrom.
        bulk_min_bond_length (float):
            The minimum interatomic distance in the bulk supercell. Default is
            ``None``, in which case it is calculated from ``target_supercell``.

    Returns:
        tuple[list[PeriodicSite], list[PeriodicSite], list[PeriodicSite], Composition, int]:
            - ``def_new_supercell_sites``: List of sites in the super-supercell
              which are within the bounds of the target supercell (minus
              ``edge_tol``) -- 'definite sites in the new supercell'.
            - ``def_new_supercell_sites_to_check``: List of sites
              in ``def_new_supercell_sites`` which are near the bordering
              regions of the target supercell, to check for site
              overlaps/displacements. Specifically, includes sites within
              -``[max(bulk_min_bond_length, edge_tol*2), edge_tol]`` of the
              target cell edges).
            - ``candidate_new_supercell_sites``: List of candidate sites for
              the new (target) supercell, within +/-``edge_tol`` of the target
              cell edge, used to determine which sites to include in the new
              supercell, by checking site overlaps/displacements (combined with
              ``def_new_supercell_sites_to_check``) and target composition.
    """
    if bulk_min_bond_length is None:
        bulk_min_bond_length = min_dist(target_supercell)

    # Note: This could possibly be made faster by reducing `edge_tol` when the `orig_supercell` is
    # fully encompassed by `target_supercell` (meaning there should be no defect-induced
    # displacements at the stenciled cell edges here), by getting the minimum encompassing cube
    # length (sphere diameter), and seeing if this is smaller than the largest sphere (diameter)
    # which can be inscribed in the target supercell
    # either way, this portion of the workflow is not a bottleneck (rather `get_orientation..` is)
    possible_new_supercell_sites = [
        PeriodicSite(
            site.species,
            site.coords,
            lattice=target_supercell.lattice,
            coords_are_cartesian=True,
            skip_checks=True,
        )
        for site in big_supercell.sites
        if is_within_frac_bounds(target_supercell.lattice, site.coords, tol=edge_tol)
    ]
    def_new_supercell_site_indices = []
    for i, site in enumerate(possible_new_supercell_sites):
        if is_within_frac_bounds(target_supercell.lattice, site.coords, tol=-edge_tol):
            def_new_supercell_site_indices.append(i)
    def_new_supercell_sites = [
        site for i, site in enumerate(possible_new_supercell_sites) if i in def_new_supercell_site_indices
    ]
    candidate_new_supercell_sites = [
        site
        for i, site in enumerate(possible_new_supercell_sites)
        if i not in def_new_supercell_site_indices
    ]

    def_new_supercell_sites_to_check = [
        site
        for site in def_new_supercell_sites
        if not is_within_frac_bounds(
            target_supercell.lattice, site.coords, tol=-max(bulk_min_bond_length, edge_tol * 2)
        )
    ]

    return (
        def_new_supercell_sites,
        def_new_supercell_sites_to_check,
        candidate_new_supercell_sites,
    )


def _remove_overlapping_sites(
    candidate_new_supercell_sites: list[PeriodicSite],
    def_new_supercell_sites_to_check: list[PeriodicSite],
    bulk_min_bond_length: float | None = None,
    pbar: tqdm | None = None,
) -> list[PeriodicSite]:
    """
    Remove sites in ``candidate_new_supercell_sites`` which overlap (within 50%
    of the bulk bond length) either with each other (in which case the site
    which is furthest from any other site in ``candidate_new_supercell_sites``
    or ``def_new_supercell_sites_to_check`` is kept) or with sites in
    ``def_new_supercell_sites_to_check``.

    Args:
        candidate_new_supercell_sites (list[|PeriodicSite|]):
            List of candidate sites in the target supercell to check if
            overlapping with each other or
            ``def_new_supercell_sites_to_check``.
        def_new_supercell_sites_to_check (list[|PeriodicSite|]):
            List of sites that are in the new (target) supercell but are near
            the bordering regions, so are used to check for overlapping.
        bulk_min_bond_length (float):
            The minimum bond length in the bulk supercell, used to check if
            inter-site distances are reasonable. If ``None`` (default),
            determined automatically from ``def_new_supercell_sites_to_check``.
        pbar (tqdm):
            ``tqdm`` progress bar object to update (for internal ``doped``
            usage). Default is ``None``.

    Returns:
        list[PeriodicSite]:
            The list of candidate sites in the target supercell which do
            not overlap with each other or with
            ``def_new_supercell_sites_to_check``.
    """
    if not candidate_new_supercell_sites:
        if pbar is not None:
            pbar.update(20)
        return candidate_new_supercell_sites

    if bulk_min_bond_length is None:
        bulk_min_bond_length = min_dist(Structure.from_sites(def_new_supercell_sites_to_check))

    check_other_candidate_sites_first = len(candidate_new_supercell_sites) < len(
        def_new_supercell_sites_to_check
    )  # check smaller list first for efficiency

    overlapping_site_indices: set[int] = set()  # using indices as faster for comparing than actual sites
    _pbar_increment_per_iter = max(
        0, 20 / len(candidate_new_supercell_sites) - 0.0001
    )  # up to 20% of progress bar

    # pre-compute arrays for vectorised distance checks:
    lattice = candidate_new_supercell_sites[0].lattice  # same lattice for all sites
    n = len(candidate_new_supercell_sites)
    cand_frac_coords = np.array([s.frac_coords for s in candidate_new_supercell_sites])
    cand_species = [s.specie.symbol for s in candidate_new_supercell_sites]
    threshold = bulk_min_bond_length * 0.5

    # NxN candidate-candidate distance matrix (diagonal set to inf to ignore self-distances):
    cand_cand_dists = lattice.get_all_distances(cand_frac_coords, cand_frac_coords)
    np.fill_diagonal(cand_cand_dists, np.inf)

    # per-candidate minimum distance to any definite site (inf if no definite sites):
    def_frac_coords = np.array([s.frac_coords for s in def_new_supercell_sites_to_check])
    min_cand_def_dists = (
        np.min(lattice.get_all_distances(cand_frac_coords, def_frac_coords), axis=1)
        if def_frac_coords.size
        else np.full(n, np.inf)
    )

    def _check_other_sites(idx):
        for other_idx in range(n):
            if (
                idx == other_idx
                or other_idx in overlapping_site_indices
                or cand_species[idx] != cand_species[other_idx]
            ):
                continue
            if cand_cand_dists[idx, other_idx] < threshold:
                # remove the site closest to any other candidate/def site (excluding the overlapping pair)
                pair_mask = np.ones(n, dtype=bool)
                pair_mask[[idx, other_idx]] = False
                min_d_idx = min(
                    np.min(cand_cand_dists[idx, pair_mask], initial=np.inf), min_cand_def_dists[idx]
                )
                min_d_other = min(
                    np.min(cand_cand_dists[other_idx, pair_mask], initial=np.inf),
                    min_cand_def_dists[other_idx],
                )
                overlapping_site_indices.add(idx if min_d_idx <= min_d_other else other_idx)

    for idx in range(n):
        if pbar is not None:
            pbar.update(_pbar_increment_per_iter)
        if idx in overlapping_site_indices:
            continue

        if check_other_candidate_sites_first:
            _check_other_sites(idx)

        if (
            idx not in overlapping_site_indices
            and def_frac_coords.size
            and min_cand_def_dists[idx] < threshold
        ):
            overlapping_site_indices.add(idx)

        if idx not in overlapping_site_indices and not check_other_candidate_sites_first:
            _check_other_sites(idx)

    return [
        site for i, site in enumerate(candidate_new_supercell_sites) if i not in overlapping_site_indices
    ]


def _get_matching_sites_from_s1_then_s2(
    template_struct: Structure,
    struct1_pool: Structure,
    struct2_pool: Structure,
) -> Structure:
    """
    Generate a stenciled structure from a template sub-set structure and two
    pools of sites/structures.

    Given two pools of sites/structures, returns a new structure which has (1)
    the closest site in ``struct1_pool`` to each site in ``template_struct``,
    and (2) ``struct2_pool.num_sites - template_struct.num_sites`` sites from
    ``struct2_pool``, chosen as those with the largest minimum distances from
    sites in the first set of sites (i.e. the single defect subcell matching
    ``template_struct``).

    The targeted use case is transforming a large periodically-repeated super-
    supercell of a defect supercell (``struct1_pool``) into the same super-
    supercell but with only one copy of the original defect supercell
    (``template_struct``), and the rest of the sites populated by the bulk
    super-supercell (``struct2_pool``; with ``struct2`` being the original bulk
    supercell).

    Args:
        template_struct (|Structure|):
            The template structure to match.
        struct1_pool (|Structure|):
            The first pool of sites to match to the template structure.
        struct2_pool (|Structure|):
            The second pool of sites to match to the template structure.

    Returns:
        Structure:
            The stenciled structure.
    """
    num_super_supercells = len(struct1_pool) // len(template_struct)  # both also have X
    single_defect_subcell_sites = []

    # Uses linear assignment per species for unambiguous optimal matching, for template_struct sites
    mapping = get_site_mapping_indices(template_struct, struct1_pool, frac_coords=False, threshold=np.inf)
    pool1_indices = [  # mapping entries are (dist, template_idx, pool_idx)
        pool_idx
        for _, template_idx, pool_idx in mapping
        if (template_idx is not None and pool_idx is not None)
    ]
    assert len(set(pool1_indices)) == len(pool1_indices)  # check no duplicate indices
    single_defect_subcell_sites = [struct1_pool[i] for i in pool1_indices]

    # for struct2_pool, it's not as straightforward to use ``get_site_mapping_indices``, as we are trying
    # to compare defect-containing supercells to a bulk supercell, so have to account for different
    # compositions / number of sites etc; so more straightforward/robust to follow this approach, where we
    # get the required number of sites from ``struct2_pool``, which are furthest from those in
    # ``single_defect_subcell_sites``:
    species_coord_dict = {}  # avoid recomputing coords for each site
    for element in struct2_pool.composition.elements:
        species_coord_dict[element.symbol] = get_coords_and_idx_of_species(
            single_defect_subcell_sites,
            element.symbol,
            frac_coords=True,  # frac_coords
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
    for dist_to_template_centre, super_site in zip(
        struct2_pool_dists_to_template_centre, struct2_pool, strict=False
    ):
        # screen to sites outside defect WS radius, for efficiency:
        if dist_to_template_centre > largest_encompassed_cube_length * 0.49:  # 2% buffer (cube length / 2)
            candidate_struct2_pool_species_sites[super_site.specie.symbol].append(super_site)

    struct2_pool_site_min_dist_dict = {}
    for species_symbol, species_sites in candidate_struct2_pool_species_sites.items():
        dist_matrix = struct2_pool.lattice.get_all_distances(  # vectorised for fast computation
            species_coord_dict[species_symbol], [site.frac_coords for site in species_sites]
        )  # M x N
        min_dists = np.min(dist_matrix, axis=0)  # down columns
        struct2_pool_site_min_dist_dict.update(dict(zip(species_sites, min_dists, strict=False)))

    # sort possible_bulk_outer_cell_sites by (largest) min dist to single_defect_subcell_sites:
    possible_bulk_outer_cell_sites = sorted(
        [site for sites in candidate_struct2_pool_species_sites.values() for site in sites],
        key=lambda x: struct2_pool_site_min_dist_dict[x],
        reverse=True,
    )
    bulk_outer_cell_sites = possible_bulk_outer_cell_sites[
        : len(struct2_pool) - len(struct2_pool) // num_super_supercells
    ]

    return Structure.from_sites(single_defect_subcell_sites + bulk_outer_cell_sites)


def _get_superset_matrix_and_supercells(
    structure: Structure,
    target_supercell: Structure,
) -> np.ndarray:
    """
    Given a structure and a target supercell, return the transformation
    ('superset') matrix which makes all lattice vectors for the structure
    larger than or equal to the largest lattice vector in ``target_supercell``.

    Args:
        structure (|Structure|):
            The original structure for which to get the superset matrix that
            fully encompasses the target supercell.
        target_supercell (|Structure|):
            The target supercell.

    Returns:
        superset_matrix (np.ndarray):
            Transformation matrix to make all lattice vectors for ``structure``
            larger than or equal to the largest lattice vector in
            ``target_supercell``.
    """
    min_cell_length = _get_all_encompassing_cube_length(target_supercell.lattice)

    # get supercell matrix which makes all lattice vectors for orig_supercell larger than min_cell_length:
    return np.ceil(min_cell_length / structure.lattice.abc)
    # could possibly use non-diagonal supercells to make this more efficient in some cases, but a lot of
    # work, shouldn't really contribute much to slowdowns, and only relevant in some rare cases (?)


def _get_all_encompassing_cube_length(lattice: Lattice) -> float:
    """
    Get the smallest possible cube that fully encompasses the cell, `regardless
    of orientation`.

    This is determined by getting the 8 vertices of the cell and computing the
    max distance between any two vertices, giving the side length of an all-
    encompassing cube. Equivalent to getting the diameter of an encompassing
    sphere.

    Args:
        lattice (|Lattice|):
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
    lattice: Lattice, cart_coords: np.ndarray | list[float], tol: float = 1e-5
) -> bool:
    """
    Check if a given Cartesian coordinate is inside the unit cell defined by
    the lattice object.

    Args:
        lattice (|Lattice|):
            |Lattice| object defining the unit cell.
        cart_coords (np.ndarray | list[float]):
            The Cartesian coordinates to check.
        tol (float):
            A tolerance (in Angstrom / cartesian units) for coordinates to be
            considered within the unit cell. If positive, expands the bounds of
            the unit cell by this amount, if negative, shrinks the bounds.

    Returns:
        bool:
            Whether the Cartesian coordinates are within the fractional bounds
            of the unit cell, accounting for ``tol``.
    """
    frac_coords = lattice.get_fractional_coords(cart_coords)
    frac_tols = np.array([tol, tol, tol]) / lattice.abc

    # Check if fractional coordinates are in the range [0, 1)
    return bool(np.all((frac_coords + frac_tols >= 0) & (frac_coords - frac_tols < 1)))


def _convert_defect_neighbours_to_X(
    defect_supercell: Structure,
    defect_position: np.ndarray,
    coords_are_cartesian: bool = False,
    ws_radius_fraction: float = 0.5,
) -> Structure:
    """
    Convert all neighbouring sites of a defect site in a supercell, within
    ``ws_radius_fraction`` of the Wigner-Seitz radius, to have their species as
    "X" (storing their original species in the site property dict under the
    "orig_species" key).

    Intended to then be used to make structure-matching far more efficient, by
    ignoring the highly-perturbed defect neighbourhood (which requires larger
    ``stol`` values which greatly slow down structure-matching).

    No longer used in default stenciling approach.

    Args:
        defect_supercell (|Structure|):
            The defect supercell to edit.
        defect_position (np.ndarray):
            The coordinates of the defect site, either fractional or Cartesian
            depending on ``coords_are_cartesian``.
        coords_are_cartesian (bool):
            Whether the defect position is in Cartesian coordinates. Default is
            ``False`` (fractional coordinates).
        ws_radius_fraction (float):
            The fraction of the Wigner-Seitz radius to use as the cut-off
            distance for neighbouring sites to convert to species "X". Default
            is 0.5 (50%).

    Returns:
        Structure:
            The supercell structure with the defect site and all neighbouring
            sites converted to species X.
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
        # |StructureMatcher| adds properties to all sites, which can then mess with site comparisons...
        converted_defect_supercell[i].properties["orig_species"] = orig_site.specie.symbol

    return converted_defect_supercell


def _convert_X_back_to_orig_species(converted_defect_supercell: Structure) -> Structure:
    """
    Convert all sites in a supercell with species "X" and "orig_species" in the
    site property dict back to their original species.

    Mainly intended just for internal ``doped`` usage, to convert back sites
    which had been converted to "X" for efficient structure-matching (see
    ``_convert_defect_neighbours_to_X``). No longer used in default stenciling
    approach.

    Args:
        converted_defect_supercell (|Structure|):
            The defect supercell to convert back.

    Returns:
        Structure:
            The supercell structure with all sites with species "X" and
            "orig_species" in the site property dict converted back to their
            original species.
    """
    defect_supercell = converted_defect_supercell.copy()

    for i, site in enumerate(defect_supercell):
        if site.specie.symbol == "X" and site.properties.get("orig_species"):
            defect_supercell.replace(i, site.properties.pop("orig_species", site.specie.symbol))

    return defect_supercell
