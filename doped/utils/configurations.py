"""
Utility functions for generating and parsing configurational coordinate (CC)
diagrams, for potential energy surfaces (PESs), Nudged Elastic Band (NEB), non-
radiative recombination calculations etc.
"""

import contextlib
import os
import warnings
from functools import lru_cache

import numpy as np
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.util.typing import PathLike

from doped.utils.efficiency import StructureMatcher_scan_stol, get_element_min_max_bond_length_dict
from doped.utils.parsing import check_atom_mapping_far_from_defect


def get_transformation_from_s2_to_s1(
    struct1: Structure,
    struct2: Structure,
    **sm_kwargs,
) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    """
    Get the supercell transformation, fractional translation vector, and a
    mapping to transform ``struct2`` to be similar to ``struct1``.

    Copied over from the ``pymatgen`` ``StructureMatcher`` class, to allow
    usage with the fast ``StructureMatcher_scan_stol`` function from ``doped``,
    along with caching to reduce redundancy; e.g. when looping over multiple
    defects for stenciling etc.

    Args:
        struct1 (Structure):
            Reference structure
        struct2 (Structure):
            |Structure| to transform to be as similar as possible to ``struct1``.
        **sm_kwargs:
            Additional keyword arguments to pass to ``StructureMatcher()``
            (e.g. ``ignored_species``, ``comparator`` etc).

    Returns:
        tuple[np.ndarray, np.ndarray, list[int | None]]:
            ``(supercell_matrix, trans_vector, mapping)`` — the supercell
            transformation, fractional translation, and site mapping produced
            by ``StructureMatcher.get_transformation`` (in that order).

            * ``supercell_matrix``: shape ``(3, 3)`` — supercell matrix for the
              transformation.
            * ``trans_vector``: shape ``(3,)`` — fractional translation vector
              for the transformation.
            * ``mapping``: Mapping of the sites in ``struct2`` to the sites in
              ``struct1``. The first ``len(struct1)`` items of the mapping
              vector are the indices of ``struct1``'s corresponding sites in
              ``struct2`` (or ``None`` if there is no corresponding site), and
              the other items are the remaining site indices of ``struct2``.
    """
    if ignored_species := sm_kwargs.pop("ignored_species", None):  # ensure it's hashable!
        sm_kwargs["ignored_species"] = tuple(ignored_species)

    return _cache_ready_get_transformation_from_s2_to_s1(struct1, struct2, **sm_kwargs)


@lru_cache(maxsize=100)  # cache transformation generation (if running multiple times with e.g. stenciling)
def _cache_ready_get_transformation_from_s2_to_s1(
    struct1: Structure,
    struct2: Structure,
    **sm_kwargs,
):
    if sm_kwargs.get("primitive_cell", False):
        raise ValueError(
            "``primitive_cell=True`` is not supported for the ``get_transformation`` method of "
            "``StructureMatcher`` (and hence ``get_s2_like_s1``)."
        )
    sm_kwargs["primitive_cell"] = False

    return StructureMatcher_scan_stol(struct1, struct2, func_name="get_transformation", **sm_kwargs)


def apply_s2_to_s1_transformation(
    struct1: Structure,
    struct2: Structure,
    supercell_matrix: np.ndarray,
    trans_vector: np.ndarray,
    mapping: list[int | None],
    include_ignored_species: bool = True,
    ignored_species: list[str] | None = None,
    new_lattice: str | None = None,
) -> Structure:
    """
    Apply a transformation (e.g. as determined by
    ``get_transformation_from_s2_to_s1``) from ``struct2`` to ``struct1``, with
    a given supercell matrix, translation vector, and site mapping.

    This will give a fully symmetry-equivalent orientation (i.e. **will not
    change the actual geometry**) of ``struct2``, except if ``struct1`` and
    ``struct2`` have different inequivalent lattices (e.g. different space
    groups) `and` ``new_lattice`` is explicitly set to ``"struct1"``. This
    function uses an accelerated version of the
    :meth:`~pymatgen.analysis.structure_matcher.StructureMatcher.get_s2_like_s1`
    method, extended to ensure the correct atomic indices matching and lattice
    vector definitions, as well as allowing for cases where ``mapping`` does
    not include all sites in ``struct2`` (e.g. when using a subset of sites to
    do matching and determine the transformation matrix and translation vector,
    as in the stenciling workflow, without needing the ordering of sites in the
    ``Structure`` objects to match).

    Templated from the ``pymatgen`` ``StructureMatcher`` class, to allow direct
    usage without repeating the expensive ``get_transformation`` call (e.g.
    when applying the same transformation to the bulk and defect supercells in
    defect stenciling).

    Args:
        struct1 (Structure):
            Reference structure.
        struct2 (Structure):
            |Structure| to transform to be as similar as possible to ``struct1``.
        supercell_matrix (np.ndarray):
            Supercell matrix for the transformation.
        trans_vector (np.ndarray):
            Fractional translation vector for the transformation.
        mapping (list[int | None]):
            Mapping of the sites in ``struct2`` to the sites in ``struct1``.
            The first ``len(struct1)`` items of the mapping vector are the
            indices of ``struct1``'s corresponding sites in ``struct2`` (or
            ``None`` if there is no corresponding site), and the other items
            are the remaining site indices of ``struct2``.
        include_ignored_species (bool):
            Whether to include ignored species / sites not in ``mapping`` in
            the output structure. Default: ``True``
        ignored_species (list[str] | None):
            List of species to ignore in ``struct1`` (for mapping), should
            match that used for ``get_transformation_from_s2_to_s1`` (if used
            to generate the transformation mapping).
            Default: ``None``
        new_lattice (str | None):
            If ``"struct1"``, then the lattice of ``struct1`` is used for the
            re-oriented structure, if ``"struct2"``, then the lattice of
            ``struct2`` is used, or if ``"s2_like_s1"``, then the output
            lattice of ``StructureMatcher.get_s2_like_s1`` (a
            symmetry-equivalent version of ``struct2.lattice``) is used.
            Default is ``None``, where ``new_lattice`` is set to ``"struct1"``
            if ``struct1`` and ``struct2`` have equivalent lattices (expected
            to be the case for defect NEBs/CC diagrams) and using the
            ``struct1`` lattice returns a fully symmetry-equivalent structure,
            or ``"s2_like_s1"`` otherwise.
            If ``new_lattice`` is explicitly set to ``"struct1"`` and this
            causes an inequivalent structure to be returned, a warning will be
            raised.

    Returns:
        Structure:
            ``struct2`` transformed to ``struct1`` as closely as possible.
    """
    ignored_set = set(ignored_species) if ignored_species else None
    if ignored_set:  # equivalent but more efficient version of code in ``get_s2_like_s1``
        kept_sites: list[PeriodicSite] = []
        ignored_sites: list[PeriodicSite] = []
        for site in struct2:  # only one pass over ``struct2``, and compare based on species not sites
            (kept_sites if site.specie.symbol not in ignored_set else ignored_sites).append(site)
        temp = Structure.from_sites(kept_sites + ignored_sites)  # ignored species at end
    else:
        temp = struct2.copy()

    temp.make_supercell(supercell_matrix)  # make supercell
    temp.translate_sites(list(range(len(temp))), trans_vector)  # translate by fractional vector

    s1 = struct1.copy().remove_species(ignored_species) if ignored_species else struct1  # ignore ignored

    # translate sites to the correct unit cell (only integer translations)
    for ii, jj in enumerate(mapping[: len(s1)]):
        if jj is not None:
            vec = np.round(s1[ii].frac_coords - temp[jj].frac_coords)  # round to nearest integer
            temp.translate_sites(jj, vec, to_unit_cell=False)  # translate to correct unit cell

    sites = [temp.sites[i] for i in mapping if i is not None]  # get sites in correct order (from mapping)

    if include_ignored_species:  # add back in ignored species / any sites not in ``mapping``
        # note that this differs slightly from the ``pymatgen`` ``StructureMatcher`` implementation, which
        # assumes that any sites not in ``mapping`` are ignored species (not the case when using a subset
        # of sites to determine the transformation matrix and translation vector, as in stenciling)
        sites.extend([temp.sites[i] for i in range(len(temp)) if i not in mapping])

    trans_struct = Structure.from_sites(sites)

    # get new_lattice choice:
    from doped.utils.symmetry import are_equivalent_lattices  # avoid circular import

    explicit_lattice_choice = new_lattice is not None
    if not are_equivalent_lattices(struct1.lattice, struct2.lattice):
        warnings.warn(
            "The lattices of the two input structures have been detected to be (symmetry-)inequivalent. "
            "This is usually not desirable for defect NEBs/CC diagrams, but may be the case for e.g. "
            "NEBs between polymorphs. "
        )
        if new_lattice is None:
            warnings.warn(
                "Note that the lattice definitions may differ between the output structure and "
                "``struct1``. See the NEB/CC diagram tutorial for details."
            )
            new_lattice = "s2_like_s1"

    elif new_lattice is None:
        new_lattice = "struct1"

    if not new_lattice:
        return trans_struct

    # Note: ``get_element_min_max_bond_length_dict`` can take a bit of time when run repeatedly with large
    # structures (due to the expensive O(N^2) distance matrix calculation). For large structures (N>50), we
    # therefore only use a subset of sites for this check (not all sites are required)
    if len(trans_struct) > 50:  # take a subset of the structure to get the min/max bond lengths
        sampled_indices = np.random.choice(len(trans_struct), size=50, replace=False)
        test_trans_struct = Structure.from_sites([trans_struct.sites[i] for i in sampled_indices])
    else:
        test_trans_struct = trans_struct

    orig_min_max_bond_lengths = get_element_min_max_bond_length_dict(test_trans_struct)

    lattice = (
        struct1.lattice
        if new_lattice == "struct1"
        else (
            trans_struct.lattice
            if new_lattice == "s2_like_s1"
            else struct2.lattice if new_lattice == "struct2" else None
        )
    )
    if lattice is None:
        raise ValueError(
            f"Invalid value for ``new_lattice``: {new_lattice}. Must be one of ``'struct1'``, "
            f"``'struct2'``, or ``'s2_like_s1'``."
        )

    trans_struct_w_lattice_choice = Structure.from_sites(
        [  # sometimes this get_s2_like_s1 doesn't fully work as desired, giving different (but equivalent)
            PeriodicSite(  # lattice vectors (e.g. a=(010) instead of (100) etc.), so we redefine with the
                site.species,  # chosen lattice to be sure
                site.frac_coords,
                lattice,
                properties=site.properties,
                to_unit_cell=False,
                skip_checks=True,
            )
            for site in trans_struct.sites
        ]
    )

    # in some cases, if the match between structures isn't perfect, then swapping the lattices here can
    # lead to a structural change, which is not desired. So here we test this by looking at the min/max
    # smallest bond lengths in the structure:
    if len(trans_struct_w_lattice_choice) > 50:
        trans_struct_w_lattice_choice_min_max_bond_lengths = get_element_min_max_bond_length_dict(
            Structure.from_sites([trans_struct_w_lattice_choice.sites[i] for i in sampled_indices])
        )
    else:
        trans_struct_w_lattice_choice_min_max_bond_lengths = get_element_min_max_bond_length_dict(
            trans_struct_w_lattice_choice
        )

    min_min_dist_change = 1e-4
    with contextlib.suppress(Exception):
        min_min_dist_change = max(
            {
                elt: max(
                    np.abs(
                        orig_min_max_bond_lengths[elt]
                        - trans_struct_w_lattice_choice_min_max_bond_lengths[elt]
                    )
                )
                for elt in orig_min_max_bond_lengths
                if elt not in (ignored_species or [])
            }.values()
        )

    if min_min_dist_change > 1e-2:
        if not explicit_lattice_choice:
            return trans_struct  # return without the new lattice, as it changes the structure

        warnings.warn(
            f"The chosen `new_lattice` ({new_lattice}) changes the minimum/maximum smallest bond lengths "
            f"by more than 1%; i.e. changing the structure, which is typically not desired!"
        )

    return trans_struct_w_lattice_choice


def orient_s2_like_s1(
    struct1: Structure,
    struct2: Structure,
    new_lattice: str | None = None,
    verbose: bool = False,
    check_mapping: bool = True,
    **sm_kwargs,
) -> Structure:
    """
    Re-orient ``struct2`` to match the orientation of ``struct1`` as closely as
    possible , with matching atomic indices as needed for VASP NEB calculations
    and other structural transformation analyses (e.g. configuration coordinate
    (CC) diagrams via ``nonrad``, ``CarrierCapture.jl`` etc.).

    This will give a fully symmetry-equivalent orientation (i.e. **will not
    change the actual geometry**) of ``struct2``, except if ``struct1`` and
    ``struct2`` have different inequivalent lattices (e.g. different space
    groups) `and` ``new_lattice`` is explicitly set to ``"struct1"``.

    This corresponds to minimising the root-mean-square displacement for the
    shortest `linear` path from ``struct1`` to a symmetry-equivalent definition
    of ``struct2``, with matched atomic indices and lattices as required by
    VASP NEB and ``nonrad`` functions. This function uses an accelerated
    version of the
    :meth:`~pymatgen.analysis.structure_matcher.StructureMatcher.get_s2_like_s1`
    method, extended to ensure the correct atomic indices matching and lattice
    vector definitions.

    If ``verbose=True``, information about the mass-weighted displacement (ΔQ
    in amu^(1/2)Å) between the input and re-oriented structures is printed.
    This is the typical x-axis unit in configurational coordinate diagrams (see
    e.g. 10.1103/PhysRevB.90.075202).

    Args:
        struct1 (Structure): Initial structure.
        struct2 (Structure): Final structure.
        new_lattice (str | None):
            If ``"struct1"``, then the lattice of ``struct1`` is used for the
            re-oriented structure, if ``"struct2"``, then the lattice of
            ``struct2`` is used, or if ``"s2_like_s1"``, then the output
            lattice of ``StructureMatcher.get_s2_like_s1`` (a
            symmetry-equivalent version of ``struct2.lattice``) is used.
            Default is ``None``, where ``new_lattice`` is set to ``"struct1"``
            if ``struct1`` and ``struct2`` have equivalent lattices (expected
            to be the case for defect NEBs/CC diagrams) and using the
            ``struct1`` lattice returns a fully symmetry-equivalent structure,
            or ``"s2_like_s1"`` otherwise.
            If ``new_lattice`` is explicitly set to ``"struct1"`` and this
            causes an inequivalent structure to be returned, a warning will be
            raised.
        verbose (bool):
            Print information about the mass-weighted displacement
            (ΔQ in amu^(1/2)Å) between the input and re-oriented structures.
            Default: ``False``
        check_mapping (bool):
            If ``True`` (default), check the atom mapping between ``struct1``
            and the re-oriented ``struct2`` (using
            ``check_atom_mapping_far_from_defect`` from
            ``doped.utils.parsing``), warning if a significant mismatch
            remains throughout the cell after re-orientation. This typically
            indicates a mismatch in the lattice definitions (e.g. different
            tiling of primitive cells within identical supercell lattice
            vectors) between the two input structures, which cannot be resolved
            by reorientation alone. Set to ``False`` to skip the check.
        **sm_kwargs:
            Additional keyword arguments to pass to ``StructureMatcher()`` /
            ``StructureMatcher_scan_stol`` (e.g. ``ignored_species``,
            ``comparator``, ``max_stol``, ``min_stol`` etc).

    Returns:
        Structure:
        ``struct2`` re-oriented to match ``struct1`` as closely as possible.

        # TODO: Option to return RMSD, just displacement, anything else?
    """
    trans = get_transformation_from_s2_to_s1(
        struct1,
        struct2,
        **sm_kwargs,
    )
    if not trans:
        raise RuntimeError(
            f"``StructureMatcher.get_transformation()`` failed. Note that this requires your input "
            f"structures to have matching compositions and similar lattices. Input structures have "
            f"compositions:\nstruct1: {struct1.composition}\nstruct2: {struct2.composition}\n"
            f"and lattices:\nstruct1: {struct1.lattice}\nstruct2: {struct2.lattice}"
        )

    ignored_species = sm_kwargs.get("ignored_species")  # used throughout this function
    struct2_really_like_struct1 = apply_s2_to_s1_transformation(
        struct1,
        struct2,
        supercell_matrix=trans[0],
        trans_vector=trans[1],
        mapping=trans[2],
        new_lattice=new_lattice,
        include_ignored_species=sm_kwargs.get("include_ignored_species", True),
        ignored_species=ignored_species,
    )

    # we see that this rearranges the structure so the atom indices should now match correctly. This should
    # give a lower dQ as we see here (or the same if the original structures matched perfectly)
    delQ_s1_s2 = get_dQ(struct1, struct2, ignored_species=ignored_species)
    delQ_s2_like_s1_s2 = get_dQ(struct2_really_like_struct1, struct2, ignored_species=ignored_species)
    delQ_s1_s2_like_s1 = get_dQ(struct1, struct2_really_like_struct1, ignored_species=ignored_species)

    if not sm_kwargs.get("allow_subset") and delQ_s1_s2_like_s1 > delQ_s1_s2 + 0.1:
        # shouldn't happen!  (and ignore cases where we're using it in defect stenciling)
        struct2_like_struct1 = StructureMatcher_scan_stol(
            struct1, struct2, func_name="get_s2_like_s1", **sm_kwargs
        )
        delQ_s1_s2_like_s1_pmg = get_dQ(struct1, struct2_like_struct1, ignored_species=ignored_species)
        warnings.warn(
            f"StructureMatcher.get_s2_like_s1() appears to have failed. The mass-weighted displacement "
            f"(ΔQ in amu^(1/2)Å) for the input structures is:\n"
            f"ΔQ(s1/s2) = {delQ_s1_s2:.2f} amu^(1/2)Å\n"
            f"Using ``StructureMatcher.get_s2_like_s1()`` directly gives:\n"
            f"ΔQ(s1/s2_like_s1_pmg) = {delQ_s1_s2_like_s1_pmg:.2f} amu^(1/2)Å\n"
            f"Then using the re-oriented structure gives:\n"
            f"ΔQ(s1/s2_like_s1_doped) = {delQ_s1_s2_like_s1:.2f} amu^(1/2)Å\n"
            f"which should always be less than or equal to the previous two values... Please report this "
            f"case to the doped developers!"
            f"\n{struct1}"
            f"\n{struct2}"
            f"\n{struct2_like_struct1}"
            f"\n{struct2_really_like_struct1}"
        )

    if verbose:
        print(f"ΔQ(s1/s2) = {delQ_s1_s2:.2f} amu^(1/2)Å")
        print(f"ΔQ(s2_like_s1/s2) = {delQ_s2_like_s1_s2:.2f} amu^(1/2)Å")
        print(f"ΔQ(s1/s2_like_s1) = {delQ_s1_s2_like_s1:.2f} amu^(1/2)Å")

    if check_mapping and not sm_kwargs.get("allow_subset"):
        # check that after re-orientation the atomic basis matches between ``struct1`` and the re-oriented
        # ``struct2``. Mismatches (far from the guess defect position(s)) typically indicate differing
        # lattice definitions (e.g. different primitive-cell tiling within identical supercell lattice
        # vectors), which re-orientation alone cannot reconcile -- see ``doped.utils.stenciling`` for
        # related usage:
        s1_for_check = struct1.copy().remove_species(ignored_species) if ignored_species else struct1
        s2_for_check = (
            struct2_really_like_struct1.copy().remove_species(ignored_species)
            if ignored_species
            else struct2_really_like_struct1
        )

        from doped.analysis import guess_defect_position  # avoid circular import

        if (
            s1_for_check.composition == s2_for_check.composition
            and not check_atom_mapping_far_from_defect(
                defect_supercell=s2_for_check,
                bulk_supercell=s1_for_check,
                defect_coords=guess_defect_position(s2_for_check, bulk_supercell=s1_for_check),
                warning=False,
            )
        ):
            warnings.warn(
                "After re-orientation, significant site mismatches remain between ``struct1`` and "
                "``struct2`` throughout the cell. This often indicates a mismatch in the lattice "
                "definitions (e.g. different tiling of primitive cells within identical supercell "
                "lattice vectors) between the two input structures, which cannot be resolved by "
                "re-orientation alone. This warning can be disabled by setting ``check_mapping=False``."
            )

    return struct2_really_like_struct1


get_s2_like_s1 = orient_s2_like_s1  # alias similar to pymatgen's get_s2_like_s1


def get_dQ(
    struct1: Structure,
    struct2: Structure,
    ignored_species: list[str] | None = None,
    reorient: bool = False,
    **sm_kwargs,
) -> float:
    """
    Get the mass-weighted displacement (ΔQ in amu^(1/2)Å) between two
    structures, assuming matched atomic indices (unless ``reorient`` is set to
    ``True``).

    Args:
        struct1 (Structure): Initial structure.
        struct2 (Structure): Final structure.
        ignored_species (list[str] | None):
            List of species to ignore when computing ΔQ (and re-orienting, if
            relevant). Default: ``None``
        reorient (bool):
            If ``True``, first re-orient ``struct2`` to match ``struct1``
            using ``orient_s2_like_s1()`` (with ``ignored_species`` forwarded),
            then compute ΔQ. Useful when input structures are symmetry-
            equivalent but have mismatched orientations / site indices.
            Default: ``False``
        **sm_kwargs:
            Additional keyword arguments to forward to ``orient_s2_like_s1()``
            (and hence ``StructureMatcher`` / ``StructureMatcher_scan_stol``),
            if/when re-orientation is performed.

    Returns:
        float:
            The mass-weighted displacement (ΔQ in amu^(1/2)Å) between the two
            structures. Returns ``np.inf`` if the structures are not matching.
    """
    if reorient:
        reorient_kwargs = {**sm_kwargs, "ignored_species": ignored_species or []}
        struct2 = orient_s2_like_s1(struct1, struct2, **reorient_kwargs)

    struct1_sites = [site for site in struct1 if site.specie.symbol not in (ignored_species or [])]
    struct2_sites = [site for site in struct2 if site.specie.symbol not in (ignored_species or [])]

    try:
        return np.sqrt(
            sum(
                (a.distance(b) ** 2) * a.specie.atomic_mass
                for a, b in zip(struct1_sites, struct2_sites, strict=True)
            )
        )
    except Exception:
        return np.inf  # if the structures are not matching, return inf


def _reorient_struct2_and_warn(
    struct1: Structure,
    struct2: Structure,
    reorient: bool | None = None,
    neb: bool = False,
    verbose: bool = False,
    **sm_kwargs,
) -> Structure:
    """
    Re-orient ``struct2`` to match ``struct1`` (using ``orient_s2_like_s1``),
    with behaviour controlled by ``reorient`` (and ``neb``).

    If ``reorient`` is ``None`` (default), ``struct2`` is re-oriented to match
    ``struct1`` and a warning is raised if re-orientation meaningfully
    decreased the mass-weighted displacement (ΔQ in amu^(1/2)Å) -- i.e. if the
    input ``struct2`` did not already have a matched orientation / atomic
    indexing with ``struct1``. In this case, if additionally ``neb`` is
    ``True`` (``False`` by default) and re-orientation reduces ΔQ to
    <0.1 amu^(1/2)Å (i.e. essentially the same structure), re-orientation is
    instead `skipped` and a different warning is raised, since we assume the
    goal was a NEB calculation between different but symmetry-equivalent
    geometries (e.g. defect migration between equivalent sites).

    If ``reorient`` is ``True``, ``struct2`` is always re-oriented and no
    warning is raised. If ``reorient`` is ``False``, ``struct2`` is returned
    unchanged and no warning is raised.

    Args:
        struct1 (Structure): Reference |Structure|.
        struct2 (Structure): |Structure| to re-orient to match ``struct1``.
        reorient (bool | None):
            Controls re-orientation behaviour; see function description above
            for details. One of ``True`` (always re-orient, no warning),
            ``False`` (never re-orient, no warning), or ``None`` (re-orient
            and warn if re-orientation was actually necessary, with special
            handling for ``neb=True`` case). Default: ``None``
        neb (bool):
            If ``True``, treat the function call as setting up an NEB
            calculation (only relevant when ``reorient`` is ``None``); if
            re-orientation would reduce ΔQ below ``0.1`` amu^(1/2)Å, this is
            assumed to be an NEB between symmetry-equivalent sites and
            re-orientation is skipped (with a corresponding warning).
            Default: ``False``
        verbose (bool):
            Forwarded to ``orient_s2_like_s1``; prints ΔQ information.
            Default: ``False``
        **sm_kwargs:
            Additional keyword arguments forwarded to ``orient_s2_like_s1``
            (and hence ``StructureMatcher`` / ``StructureMatcher_scan_stol``).

    Returns:
        Structure:
            ``struct2`` re-oriented to match ``struct1`` as closely as
            possible (if ``reorient`` is ``True`` or ``None``, and the
            NEB-between-symmetry-equivalent-sites case is not detected),
            otherwise the input ``struct2`` unchanged.
    """
    if reorient is False:
        return struct2

    delQ_before = get_dQ(struct1, struct2, ignored_species=sm_kwargs.get("ignored_species"))
    struct2_oriented = orient_s2_like_s1(struct1, struct2, verbose=verbose, **sm_kwargs)
    delQ_after = get_dQ(struct1, struct2_oriented, ignored_species=sm_kwargs.get("ignored_species"))

    # warn when re-orientation was actually required (i.e. ΔQ meaningfully decreased), and reorient is None
    if delQ_after < delQ_before - 1e-2 and reorient is None:
        if delQ_after < 0.1 and neb:
            warnings.warn(
                f"The input ``struct2`` did not have a matching orientation / atomic indexing with "
                f"``struct1``, but re-orienting to give the closest match gives a small mass-weighted "
                f"atomic displacement ΔQ of {delQ_after:.2f} amu^(1/2)Å (compared to {delQ_before:.2f} "
                f"amu^(1/2)Å before), suggesting that this could be a NEB calculation between different "
                f"symmetry-equivalent sites, and so re-orientation has been skipped. Set ``reorient`` to "
                f"True/False to control this behaviour, or see the ``doped`` configuration coordinate / "
                f"NEB path generation tutorial for details."
            )
            return struct2

        warnings.warn(
            f"The input ``struct2`` did not have a matching orientation / atomic indexing with "
            f"``struct1``, so has been re-oriented using ``orient_s2_like_s1()`` to give the shortest "
            f"linear interpolation path between them (ΔQ decreased from {delQ_before:.2f} to "
            f"{delQ_after:.2f} amu^(1/2)Å). Set ``reorient=False`` to disable this behaviour, or see the "
            f"``doped`` configuration coordinate / NEB path generation tutorial for details."
        )

    return struct2_oriented


def get_path_structures(
    struct1: Structure,
    struct2: Structure,
    n_images: int | np.ndarray | list[float] = 7,
    displacements: np.ndarray | list[float] | None = None,
    displacements2: np.ndarray | list[float] | None = None,
    reorient: bool | None = None,
    verbose: bool = False,
    **sm_kwargs,
) -> dict[str, Structure] | tuple[dict[str, Structure], dict[str, Structure]]:
    """
    Generate a series of interpolated structures along the linear path between
    ``struct1`` and ``struct2``, typically for use in NEB calculations or
    configuration coordinate (CC) diagrams.

    Structures are output as a dictionary with keys corresponding to either the
    index of the interpolated structure (0-indexed; ``00``, ``01`` etc as for
    ``VASP`` NEB calculations) or the fractional displacement along the
    interpolation path between structures, and values corresponding to the
    interpolated structure. If ``displacements`` is set (and thus two sets of
    structures are generated), a tuple of such dictionaries is returned.

    Note that for NEB calculations, the lattice vectors and order of sites
    (atomic indices) must be consistent in both ``struct1`` and ``struct2``.
    This is also desirable for CC diagrams, as the atomic indices are assumed
    to match for many parsing and plotting functions (e.g. in ``nonrad`` and
    ``CarrierCapture.jl``), but is not strictly necessary -- though is
    typically required for appropriate structure interpolation. By default
    (``reorient=None``), this function uses ``orient_s2_like_s1()`` to (attempt
    to) re-orient ``struct2`` to match the lattice vectors and site ordering of
    ``struct1`` as closely as possible, and warns if this re-orientation was
    actually required (i.e. if the input structures did not already correspond
    to the shortest linear interpolation path between them).
    In the case of NEB for defect migration between symmetry-equivalent sites,
    this is not desired, and so re-orientation will be skipped if it results in
    a near-zero final mass-weighted displacement (ΔQ) between the structures
    (<0.1 amu^(1/2)Å), ``reorient`` is ``None`` (default), and
    ``displacements`` is ``None`` (i.e. assuming an NEB / PES calculation).
    Otherwise set ``reorient`` explicitly to ``True/False`` to control this
    behaviour. See the ``doped`` configuration coordinate / NEB path generation
    `tutorial <https://doped.readthedocs.io/en/latest/CCD_NEB_tutorial.html>`__
    for further discussion.

    If only ``n_images`` is set (and ``displacements`` is ``None``)(default),
    then only one set of interpolated structures is generated (in other words,
    assuming a standard NEB/PES calculation is being performed). If
    ``displacements`` (and possibly ``displacements2``) is set, then two sets
    of interpolated structures are generated (in other words, assuming a CC /
    non-radiative recombination calculation is being performed, where the two
    sets of structures are to be calculated in separate charge/spin etc
    states).

    Args:
        struct1 (Structure): Initial structure.
        struct2 (Structure): Final structure.
        n_images (int):
            Number of images to interpolate between ``struct1`` and
            ``struct2``, or a list of fractional interpolation values
            (displacements) to use. Note that ``n_images`` is ignored if
            ``displacements`` is set (in which case CC / non-radiative
            recombination calculations are assumed -- generating two sets of
            interpolated structures -- otherwise a standard NEB / PES
            calculation is assumed -- generating one set of structures).
            Default: 7
        displacements (np.ndarray or list):
            Displacements to use for ``struct1`` along the linear
            transformation path to ``struct2``. If set, then CC / non-radiative
            recombination calculations are assumed, and two sets of
            interpolated structures will be generated. If set and
            ``displacements2`` is not set, then the same set of displacements
            is used for both sets of interpolated structures. Default: ``None``
        displacements2 (np.ndarray or list):
            Displacements to use for ``struct2`` along the linear
            transformation path to ``struct1``. If not set and
            ``displacements`` is not ``None``, then the same set of
            displacements is used for both sets of interpolated structures.
            Default: ``None``
        reorient (bool | None):
            Controls whether to automatically re-orient ``struct2`` to match
            ``struct1`` (using ``orient_s2_like_s1()``) before generating the
            interpolated path structures, which ensures matched lattice
            vectors / atomic indices and the shortest linear interpolation
            path between the endpoints. One of:

            - ``True``: always re-orient, no warnings.
            - ``False``: never re-orient (use ``struct2`` as provided), no
              warnings.
            - ``None`` (default): re-orient ``struct2`` and warn if
              re-orientation was actually necessary (i.e. if the
              mass-weighted displacement ΔQ decreased as a result). In NEB
              mode (``displacements=None``), if re-orientation reduces ΔQ
              below ``0.1`` amu^(1/2)Å this is assumed to be an NEB between
              symmetry-equivalent sites (where re-orientation is not
              desired); thus re-orientation is skipped and a different warning
              is raised.
        verbose (bool):
            If ``True`` and re-orientation is performed, ``orient_s2_like_s1()``
            prints information about the mass-weighted displacement (ΔQ in
            amu^(1/2)Å) between ``struct1`` and ``struct2`` (pre and post
            re-orientation). Default: ``False``
        **sm_kwargs:
            Additional keyword arguments to forward to ``orient_s2_like_s1()``
            (and hence ``StructureMatcher`` / ``StructureMatcher_scan_stol``),
            when re-orientation is performed.

    Returns:
        dict[str, Structure] | tuple[dict[str, Structure], dict[str, Structure]]:
            Dictionary of structures (for NEB/PES calculations), or tuple of
            two dictionaries of structures (for CC / non-radiative
            calculations, when ``displacements`` is not ``None``).
    """
    struct2 = _reorient_struct2_and_warn(
        struct1,
        struct2,
        reorient=reorient,
        neb=displacements is None,
        verbose=verbose,
        **sm_kwargs,
    )
    if displacements is None:
        disp_1 = struct1.interpolate(
            struct2, n_images, interpolate_lattices=True, pbc=True, autosort_tol=1.2
        )
        disp_2 = None
        if not isinstance(n_images, int | float):
            displacements = n_images  # displacement magnitudes provided instead of n_images

    else:
        displacements2 = displacements2 if displacements2 is not None else displacements
        disp_1 = struct1.interpolate(
            struct2, displacements, interpolate_lattices=True, pbc=True, autosort_tol=1.2
        )
        disp_2 = struct2.interpolate(
            struct1,
            displacements2,
            interpolate_lattices=True,
            pbc=True,
            autosort_tol=1.2,
        )

    disp_1_dict: dict[str, Structure] = {}
    disp_2_dict: dict[str, Structure] = {}
    for structs_disps_dict_tuple in [
        (disp_1, displacements, disp_1_dict),
        (disp_2, displacements2, disp_2_dict),
    ]:
        structs, disps, disp_dict = structs_disps_dict_tuple
        if structs is None:
            continue  # NEB, only one directory written
        if disps is not None:
            disps = _smart_round(disps)

        for i, struct in enumerate(structs):
            key = f"0{i}" if displacements is None else f"delQ_{disps[i]}"  # type: ignore
            disp_dict[key] = struct

    return (disp_1_dict, disp_2_dict) if disp_2_dict else disp_1_dict


def _smart_round(
    numbers: float | list[float] | np.ndarray[float],
    tol: float = 1e-5,
    return_decimals: bool = False,
    consistent_decimals: bool = True,
) -> float | list[float] | np.ndarray[float] | tuple[float | list[float] | np.ndarray[float], float]:
    """
    Custom rounding function that rounds the input number(s) to the lowest
    number of decimals which gives the same numerical value as the input, to
    within a specified tolerance value.

    For example, 0.5000001 with tol = 1e-5 (default) would be rounded to 0.5,
    but 0.5780001 would be rounded to 0.578.

    Primary use case is for rounding float values returned by ``numpy`` which
    may be e.g. 0.50000001 instead of 0.5.

    If a list or array of numbers is provided (rather than a single float),
    then rounding is applied to each element in the list/array. If
    ``consistent_decimals=True`` (default), then the same number of decimals is
    used for rounding for all elements, where this number of decimals is the
    minimum required for rounding to satisfy the tolerance for all elements.

    Args:
        numbers (float, list[float], np.ndarray[float]):
            The number(s) to round.
        tol (float): The tolerance value for rounding.
            Default: ``1e-5``
        return_decimals (bool):
            If ``True``, also returns the identified appropriate number of
            decimals to round to. Default: ``False``
        consistent_decimals (bool):
            If ``True``, the same number of decimals is used for rounding for
            all elements in the input list/array, where this number of decimals
            is the minimum required for rounding to satisfy the tolerance for
            all elements. Default: ``True``

    Returns:
        float | list[float] | np.ndarray[float] | tuple[..., float]:
            The rounded number(s). If ``return_decimals=True``, then a tuple is
            returned with the rounded number(s) and the number of decimals used
            for rounding.
    """

    def _smart_round_number(number: float, tol: float, return_decimals: bool):
        for decimals in range(20):
            rounded_number = round(number, decimals)
            if abs(rounded_number - number) < tol:
                return (rounded_number, decimals) if return_decimals else rounded_number

        raise ValueError("Failed to round number to within tolerance???")

    if isinstance(numbers, float):
        return _smart_round_number(numbers, tol, return_decimals)

    # list / array:
    rounded_list = [
        _smart_round_number(number, tol, return_decimals or consistent_decimals) for number in numbers
    ]

    if not consistent_decimals:
        return rounded_list if isinstance(numbers, list) else np.array(rounded_list)

    decimals = max(decimals for _, decimals in rounded_list)
    rounded_list = [round(number, decimals) for number in numbers]
    return (rounded_list, decimals) if return_decimals else rounded_list


def write_path_structures(
    struct1: Structure,
    struct2: Structure,
    output_dir: PathLike | None = None,
    n_images: int | list = 7,
    displacements: np.ndarray | list[float] | None = None,
    displacements2: np.ndarray | list[float] | None = None,
    reorient: bool | None = None,
    verbose: bool = False,
    **sm_kwargs,
) -> dict[str, Structure] | tuple[dict[str, Structure], dict[str, Structure]]:
    """
    Generate a series of interpolated structures along the linear path between
    ``struct1`` and ``struct2``, typically for use in NEB calculations or
    configuration coordinate (CC) diagrams, and write to folders.

    Folder names are labelled by the index of the interpolated structure
    (0-indexed; ``00``, ``01`` etc as for VASP NEB calculations) or the
    fractional displacement along the interpolation path between structures
    (e.g. ``delQ_0.0``, ``delQ_0.1``, ``delQ_-0.1`` etc), depending on the
    input ``n_images``/``displacements`` settings.

    Note that for NEB calculations, the lattice vectors and order of sites
    (atomic indices) must be consistent in both ``struct1`` and ``struct2``.
    This is also desirable for CC diagrams, as the atomic indices are assumed
    to match for many parsing and plotting functions (e.g. in ``nonrad`` and
    ``CarrierCapture.jl``), but is not strictly necessary -- though is
    typically required for appropriate structure interpolation. By default
    (``reorient=None``), this function uses ``orient_s2_like_s1()`` to (attempt
    to) re-orient ``struct2`` to match the lattice vectors and site ordering of
    ``struct1`` as closely as possible, and warns if this re-orientation was
    actually required (i.e. if the input structures did not already correspond
    to the shortest linear interpolation path between them).
    In the case of NEB for defect migration between symmetry-equivalent sites,
    this is not desired, and so re-orientation will be skipped if it results in
    a near-zero final mass-weighted displacement (ΔQ) between the structures
    (<0.1 amu^(1/2)Å), ``reorient`` is ``None`` (default), and
    ``displacements`` is ``None`` (i.e. assuming an NEB / PES calculation).
    Otherwise set ``reorient`` explicitly to ``True/False`` to control this
    behaviour. See the ``doped`` configuration coordinate / NEB path generation
    `tutorial <https://doped.readthedocs.io/en/latest/CCD_NEB_tutorial.html>`__
    for further discussion.

    If only ``n_images`` is set (and ``displacements`` is ``None``)(default),
    then only one set of interpolated structures is written (in other words,
    assuming a standard NEB/PES calculation is being performed). If
    ``displacements`` (and possibly ``displacements2``) is set, then two sets
    of interpolated structures are written (in other words, assuming a CC /
    non-radiative recombination calculation is being performed, where the two
    sets of structures are to be calculated in separate charge/spin etc
    states).

    Args:
        struct1 (Structure): Initial structure.
        struct2 (Structure): Final structure.
        output_dir (PathLike):
            Directory to write the interpolated structures to. Defaults to
            "Configuration_Coordinate" if ``displacements`` is set, otherwise
            "NEB".
        n_images (int):
            Number of images to interpolate between ``struct1`` and
            ``struct2``, or a list of fractional interpolation values
            (displacements) to use. Note that ``n_images`` is ignored if
            ``displacements`` is set (in which case CC / non-radiative
            recombination calculations are assumed -- generating two sets of
            interpolated structures -- otherwise a standard NEB / PES
            calculation is assumed -- generating one set of structures).
            Default: 7
        displacements (np.ndarray or list):
            Displacements to use for ``struct1`` along the linear
            transformation path to ``struct2``. If set, then CC / non-radiative
            recombination calculations are assumed, and two sets of
            interpolated structures will be written to file. If set and
            ``displacements2`` is not set, then the same set of displacements
            is used for both sets of interpolated structures. Default: ``None``
        displacements2 (np.ndarray or list):
            Displacements to use for ``struct2`` along the linear
            transformation path to ``struct1``. If not set and
            ``displacements`` is not ``None``, then the same set of
            displacements is used for both sets of interpolated structures.
            Default: ``None``
        reorient (bool | None):
            Controls whether to automatically re-orient ``struct2`` to match
            ``struct1`` (using ``orient_s2_like_s1()``) before generating the
            interpolated path structures, which ensures matched lattice
            vectors / atomic indices and the shortest linear interpolation
            path between the endpoints. One of:

            - ``True``: always re-orient, no warnings.
            - ``False``: never re-orient (use ``struct2`` as provided), no
              warnings.
            - ``None`` (default): re-orient ``struct2`` and warn if
              re-orientation was actually necessary (i.e. if the
              mass-weighted displacement ΔQ decreased as a result). In NEB
              mode (``displacements=None``), if re-orientation reduces ΔQ
              below ``0.1`` amu^(1/2)Å this is assumed to be an NEB between
              symmetry-equivalent sites (where re-orientation is not
              desired); thus re-orientation is skipped and a different warning
              is raised.
        verbose (bool):
            If ``True`` and re-orientation is performed, ``orient_s2_like_s1()``
            prints information about the mass-weighted displacement (ΔQ in
            amu^(1/2)Å) between ``struct1`` and ``struct2`` (pre and post
            re-orientation). Default: ``False``
        **sm_kwargs:
            Additional keyword arguments to forward to ``orient_s2_like_s1()``
            (and hence ``StructureMatcher`` / ``StructureMatcher_scan_stol``),
            when re-orientation is performed.

    Returns:
        dict[str, Structure] | tuple[dict[str, Structure], dict[str, Structure]]:
            Dictionary of structures (for NEB/PES calculations), or tuple of
            two dictionaries of structures (for CC / non-radiative
            calculations, when ``displacements`` is not ``None``).
    """
    path_structs = get_path_structures(
        struct1,
        struct2,
        n_images=n_images,
        displacements=displacements,
        displacements2=displacements2,
        reorient=reorient,
        verbose=verbose,
        **sm_kwargs,
    )
    path_struct_dicts = [path_structs] if isinstance(path_structs, dict) else list(path_structs)
    output_dir = output_dir or ("Configuration_Coordinate" if displacements is not None else "NEB")

    for i, path_struct_dict in enumerate(path_struct_dicts):
        for folder, struct in path_struct_dict.items():
            PES_dir = f"PES_{i + 1}" if len(path_struct_dicts) > 1 else ""
            path_to_folder = f"{output_dir}/{PES_dir}/{folder}"
            os.makedirs(path_to_folder, exist_ok=True)
            struct.to(filename=f"{path_to_folder}/POSCAR", fmt="poscar")

    return path_structs
