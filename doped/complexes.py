"""
Code for generating and analysing defect complexes.
"""

import contextlib
import math
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from itertools import combinations, product
from typing import Any

import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher

from doped.utils.efficiency import Molecule, PeriodicSite, Structure
from doped.utils.parsing import (
    _get_species_from_composition_diff,
    get_defect_type_and_composition_diff,
    get_matching_site,
    get_site_mapping_indices,
)
from doped.utils.supercells import min_dist
from doped.utils.symmetry import (
    get_equiv_frac_coords_in_primitive,
    get_primitive_structure,
    is_periodic_image,
)


def classify_vacancy_geometry(
    vacancy_supercell: Structure,
    bulk_supercell: Structure,
    tol: float = 0.5,
    abs_tol: bool = False,
    verbose: bool = False,
) -> str:
    """
    Classify the geometry of a given vacancy in a supercell, as either a
    'simple' point vacancy, a 'split' vacancy, or a 'non-trivial' vacancy.

    Split vacancy geometries are those where 2 vacancies and 1 interstitial
    are found to be present in the defect structure, as determined using
    site-matching between the defect and bulk structures with a fractional
    distance tolerance (``tol``), such that the absence of any site of
    matching species within the distance tolerance to the original bulk site
    is considered a vacancy, and vice versa in comparing the bulk to the defect
    structure is an interstitial. This corresponds to the 2 V_X + X_i
    definition of split vacancies discussed in
    https://doi.org/10.48550/arXiv.2412.19330

    A simple vacancy corresponds to cases where 1 site from the bulk structure
    cannot be matched to the defect structure while all defect structure sites
    can be matched to bulk sites, and 'non-trivial' vacancies refer to all
    other cases.

    Inspired by the vacancy geometry classification used in Kumagai et al.
    `Phys Rev Mater` 2021. See https://doi.org/10.48550/arXiv.2412.19330 for
    further details.

    Args:
        vacancy_supercell (Structure):
            The defect supercell containing the vacancy to be classified.
        bulk_supercell (Structure):
            The bulk supercell structure to compare against for site-matching.
        tol (float):
            The (fractional) tolerance for matching sites between the defect
            and bulk structures. If ``abs_tol`` is ``False`` (default), then
            this value multiplied by the shortest bond length in the bulk
            structure will be used as the distance threshold for matching,
            otherwise the value is used directly (as a length in Å).
            Default is 0.5.
        abs_tol (bool):
            Whether to use ``tol`` as an absolute distance tolerance (in Å)
            instead of a fractional tolerance (in terms of the shortest bond
            length in the structure).
            Default is ``False``.
        verbose (bool):
            Whether to print additional information about the classification
            for non-trivial vacancies.
            Default is ``False``.
    """
    # if not all sites in both structures are oxi-state decorated / neutral, then remove oxi states:
    oxi_state_decorated = [
        "+" in site.species_string or "-" in site.species_string or "0" in site.species_string
        for site in [*vacancy_supercell.sites, *bulk_supercell.sites]
    ]
    if len(set(oxi_state_decorated)) > 1:  # not consistent with all sites, remove oxi states:
        vacancy_supercell = vacancy_supercell.copy()
        bulk_supercell = bulk_supercell.copy()
        vacancy_supercell.remove_oxidation_states()
        bulk_supercell.remove_oxidation_states()

    def_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, vacancy_supercell)
    old_species = _get_species_from_composition_diff(comp_diff, -1)
    bulk_bond_length = max(min_dist(bulk_supercell), 1)
    dist_tol = tol * bulk_bond_length if not abs_tol else tol
    num_offsite_bulk_to_defect = np.sum(
        np.array(
            get_site_mapping_indices(
                bulk_supercell,
                vacancy_supercell,
                species=old_species,
                dists_only=True,
                allow_duplicates=True,
                threshold=np.inf,  # don't warn for large detected off-site displacements (e.g. split vacs)
            )
        )
        > dist_tol
    )
    num_offsite_defect_to_bulk = np.sum(
        np.array(
            get_site_mapping_indices(
                vacancy_supercell,
                bulk_supercell,
                species=old_species,
                dists_only=True,
                allow_duplicates=True,
                threshold=np.inf,  # don't warn for large detected off-site displacements (e.g. split vacs)
            )
        )
        > dist_tol
    )

    if num_offsite_bulk_to_defect == 1 and num_offsite_defect_to_bulk == 0:
        return "Simple Vacancy"

    if num_offsite_bulk_to_defect == 2 and num_offsite_defect_to_bulk == 1:
        return "Split Vacancy"

    # otherwise, we have a non-trivial vacancy
    if verbose:
        print(f"{num_offsite_defect_to_bulk} offsite atoms in defect compared to bulk")
        print(f"{num_offsite_bulk_to_defect} offsite atoms in bulk compared to defect")
    return "Non-Trivial"


def get_es_energy(structure: Structure, oxi_states: dict | None = None) -> float:
    """
    Calculate the electrostatic (Madelung) energy of a structure using Ewald
    summation.

    The oxidation states of the structure should be set already (via site
    species), or they should be provided with the ``oxi_states`` argument.

    Args:
        structure (Structure):
            Structure object for which to calculate the energy.
        oxi_states (dict, optional):
            Dictionary of oxidation states for the input structure.
            If ``None`` (default), the oxidation states of the structure are
            used. An error will be raised if the oxidation states are not set
            and are not provided.

    Returns:
        float: The electrostatic energy of the structure.
    """
    if oxi_states is not None:
        structure.add_oxidation_state_by_element(oxi_states)
    structure._charge = structure.charge  # requires oxidation states to be set
    return EwaldSummation(structure).total_energy


def generate_complex_from_defect_sites(
    bulk_supercell: Structure,
    vacancy_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    interstitial_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    substitution_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
) -> Structure:
    """
    Generate the supercell containing a defect complex, given the bulk
    supercell and the sites of the defects to be included in the complex.

    The coordinates of the input defect sites should correspond to the input
    bulk supercell. For substitutions, the closest site in the bulk supercell
    to the supplied site(s) will be removed, and replaced with the input
    ``substitution_sites``.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure in which to generate the defect
            complex.
        vacancy_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of vacancies to include in the defect complex
            supercell. Default is None.
        interstitial_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of interstitials to include in the defect complex
            supercell. Default is None.
        substitution_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of substitutions to include in the defect complex
            supercell. Default is None.

    Returns:
        Structure: The defect complex supercell structure.
    """
    defect_dict = {
        "vacancy_sites": vacancy_sites or [],
        "interstitial_sites": interstitial_sites or [],
        "substitution_sites": substitution_sites or [],
    }
    for key, value in list(defect_dict.items()):
        if isinstance(value, PeriodicSite):
            defect_dict[key] = [value]  # convert to Iterable

        if defect_dict[key] and (
            not isinstance(defect_dict[key], Iterable)
            or not isinstance(next(iter(defect_dict[key])), PeriodicSite)
        ):
            raise TypeError(
                f"Defect sites input arguments must be a list, set, or tuple of defect sites. Got "
                f"{type(defect_dict[key])} for {key}."
            )

    defect_struct = bulk_supercell.copy()

    for site in defect_dict["vacancy_sites"]:
        vac_site = get_matching_site(site, bulk_supercell)
        defect_struct.remove(vac_site)

    for site in defect_dict["interstitial_sites"]:
        defect_struct.insert(
            0,
            species=site.specie,
            coords=site.frac_coords,
        )

    for site in defect_dict["substitution_sites"]:
        bulk_site_idx = defect_struct.index(get_matching_site(site, bulk_supercell, anonymous=True))
        defect_struct.remove_sites([bulk_site_idx])
        defect_struct.insert(bulk_site_idx, species=site.specie, coords=site.frac_coords)

    return defect_struct


def get_equivalent_complex_defect_sites_in_primitive(
    bulk_supercell: Structure,
    vacancy_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    interstitial_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    substitution_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    primitive_structure: Structure | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    return_molecules: bool = False,
    **kwargs,
) -> list[list[PeriodicSite]] | list[Molecule]:
    r"""
    Generate all equivalent complex defect site configurations in the primitive
    unit cell, for the input constituent point defect sites of the complex.

    The input sites should correspond to the input bulk supercell.

    The approach followed in this function is:

    1. Generate all symmetry-equivalent sites in the primitive unit cell, for
    the input constituent point defect sites of the complex.

    2. Choose one constituent point defect as the 'anchor' site, based on
    estimated computational efficiency.

    3. Generate a 'template' complex defect molecule, using the ``pymatgen``
    ``Molecule`` class.

    4. From the sets of symmetry-equivalent defect sites in the primitive unit
    cell, generate all possible combinations of candidate point defect sites
    that have inter-defect distances matching the input complex defect sites
    (+/- 2 :math:`\times` ``symprec`` :math:`\times`  ``dist_tol_factor``).

    5. From these candidate site combinations, generate complex defect
    molecules (as ``pymatgen`` ``Molecule``\s), and reduce to only those which
    are symmetry-equivalent to the input complex defect, and are distinct (i.e.
    not identical or periodic images; to avoid potential double counting).

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure to which the input sites correspond.
        vacancy_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of vacancies in the defect complex. Default is
            ``None``.
        interstitial_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of interstitials in the defect complex. Default is
            ``None``.
        substitution_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of substitutions in the defect complex. Default is
            ``None``.
        primitive_structure (Structure | None):
            The primitive unit cell structure, in which to get equivalent
            complex defect sites. If ``None`` (default), the primitive
            structure will be determined from the bulk supercell.
        symprec (float):
            Symmetry precision for determining the primitive structure (if not
            provided), supercell symmetry operations and equivalent defect
            sites in the primitive unit cell. Defaults to 0.01. Note that this
            should match the value used for determining the point defect
            multiplicities (e.g. with the ``Defect.get_multiplicity()``
            methods) for appropriate comparisons -- the same default of 0.01 is
            used in all relevant ``doped`` functions.
            If ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default),
            this value will be automatically adjusted (up to 10x, down to 0.1x)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``symprec`` (and
            ``dist_tol_factor`` values).
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), searching for equivalent sites (by
            inter-defect distances) and matching equivalent complex defect
            geometries (as ``Molecule``\s), as a multiplicative factor of
            ``symprec``. Default is 1.0 (i.e. ``dist_tol = symprec``, in Å).
            Note that this should match the value used for determining the
            point defect multiplicities (e.g. with the
            ``Defect.get_multiplicity()`` methods) for appropriate comparisons
            -- the same default of 0.01 Å is used in all relevant ``doped``
            functions. If ``fixed_symprec_and_dist_tol_factor`` is ``False``
            (default), this value will also be automatically adjusted if
            necessary (up to 10x, down to 0.1x)(after ``symprec`` adjustments)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``dist_tol_factor`` (and
            ``symprec``) values.
        return_molecules (bool):
            Whether to return the equivalent complex defect molecules as
            ``Molecule`` objects, or as lists of ``PeriodicSite``
            objects. Default is ``False`` (return lists of ``PeriodicSite``\s).
        **kwargs:
            Additional keyword arguments to pass to ``get_all_equiv_sites``
            (via ``get_equiv_frac_coords_in_primitive()``), such as
            ``fixed_symprec_and_dist_tol_factor`` and ``verbose``.

    Returns:
        list[list[PeriodicSite]] | list[Molecule]:
            List of equivalent complex defect sites as ``Molecule``\s
            or lists of ``PeriodicSite``\s, depending on the value of
            ``return_molecules``.
    """
    # Note: An alternative approach here would be to use the intersection of host crystal symmetry
    # operations and those of the complex defect (molecule), though unlikely to be any more efficient
    primitive_structure = primitive_structure or get_primitive_structure(bulk_supercell, symprec=symprec)

    vacancy_sites = vacancy_sites or []
    interstitial_sites = interstitial_sites or []
    substitution_sites = substitution_sites or []
    raw_complex_defect_sites = [*vacancy_sites, *interstitial_sites, *substitution_sites]

    # generate our template complex defect molecule:
    # later, if the complex is composed of >=3 defects, we use the centre of mass to determine the 'anchor'
    # site, and so here we ensure that the coordinates used in our template molecule are the closest
    # periodic image site to the first site, to ensure we obtain a correct center of mass
    complex_mol = Molecule(
        [site.species for site in raw_complex_defect_sites],
        bulk_supercell.lattice.get_cartesian_coords(
            [
                site.frac_coords + raw_complex_defect_sites[0].distance_and_image(site)[1]
                for site in raw_complex_defect_sites
            ]
        ),
    )
    complex_com = complex_mol.center_of_mass

    # now we set the species of vacancy sites to X, to distinguish from occupied sites (we didn't do this
    # earlier as this breaks the ``center_of_mass`` attribute of ``Molecule``\s)
    X_vacancy_sites = []
    for site in vacancy_sites:
        X_site = deepcopy(site)
        X_site.species = "X"
        X_vacancy_sites.append(X_site)
    complex_defect_sites = [*X_vacancy_sites, *interstitial_sites, *substitution_sites]

    # set efficient and unique hash for Site and PeriodicSite, to allow use as dict keys
    complex_equiv_prim_frac_coords_dict = {
        site: {  # dict of sets of equivalent frac coords in primitive unit cell, for each point defect
            tuple(frac_coords.round(4))  # type: ignore  # (return_symprec_and_dist_tol_factor=False)
            for frac_coords in get_equiv_frac_coords_in_primitive(
                site.frac_coords,
                primitive_structure,
                bulk_supercell,
                symprec=symprec,
                dist_tol_factor=dist_tol_factor,
                return_symprec_and_dist_tol_factor=False,
                **kwargs,
            )
        }
        for site in complex_defect_sites
    }
    _, symprec, dist_tol_factor = get_equiv_frac_coords_in_primitive(
        complex_defect_sites[-1].frac_coords,
        primitive_structure,
        bulk_supercell,
        symprec=symprec,
        dist_tol_factor=dist_tol_factor,
        return_symprec_and_dist_tol_factor=True,
        **kwargs,
    )  # get dynamically-adjusted symprec / dist_tol_factor
    # Note: In original drafts, we used ``get_matching_site`` and ``SymmetrizedStructure.equivalent_sites``
    # to get the symmetry-equivalent sites of vacancies/substitutions in the primitive unit cell at this
    # step, but _should_ be fully redundant/equivalent to ``get_equiv_frac_coords_in_primitive``

    # min multiplicity is the maximum of m_i - choose - n_i, where m_i is the multiplicity of the i-th
    # point defect in the complex, and n_i is the number of symmetry-equivalent i-th point defects in the
    # complex:
    dist_tol = symprec * dist_tol_factor  # distance tolerance for clustering generated sites
    frac_tol = np.max(np.array([dist_tol, dist_tol, dist_tol]) / primitive_structure.lattice.abc)

    unique_complex_site_m_n_equiv_prim_frac_dict: dict[PeriodicSite, dict[str, Any]] = {}
    for site, equiv_coords_set in complex_equiv_prim_frac_coords_dict.items():
        m_i = len(equiv_coords_set)  # (try to) find matching site configuration
        for site_data in unique_complex_site_m_n_equiv_prim_frac_dict.values():
            if site_data["m_i"] == m_i and any(
                np.allclose(next(iter(equiv_coords_set)), frac_coords, atol=frac_tol)
                for frac_coords in site_data["equiv_prim_frac_coords"]
            ):
                site_data["n_i"] += 1
                break
        else:  # No match found
            unique_complex_site_m_n_equiv_prim_frac_dict[site] = {
                "m_i": m_i,
                "n_i": 1,
                "equiv_prim_frac_coords": equiv_coords_set,
            }

    min_multiplicity = max(
        min(  # minimum m-choose-n for each inequivalent constituent point defect
            math.comb(site_data["m_i"], site_data["n_i"])
            for site_data in unique_complex_site_m_n_equiv_prim_frac_dict.values()
        ),
        1,  # at least 1
    )

    complex_defect_defect_dist_dict = {  # dict of inter-defect distances for each pair of point defects
        frozenset((site1, site2)): site1.distance(site2)
        for site1, site2 in combinations(complex_defect_sites, 2)
    }

    # choose anchor point defect site; the result should be invariant to the choice of anchor site, but
    # computational efficiency will be optimised by minimising the points_in_sphere search space and
    # candidate complexes scanned over:
    anchor_site: PeriodicSite | None = None
    with contextlib.suppress(AttributeError):
        if len(complex_defect_sites) > 2:  # for complexes with 3+ defects, choose the most central site:
            anchor_site = min(
                complex_defect_sites,
                key=lambda site: site.distance_and_image_from_frac_coords(
                    bulk_supercell.lattice.get_fractional_coords(complex_com)
                )[0],
            )
    if anchor_site is None:
        anchor_site = max(
            complex_defect_sites,
            key=lambda site: len(complex_equiv_prim_frac_coords_dict[site]),
        )  # choose that with the highest multiplicity, for 2-defect complexes, or if COM approach fails

    # generate our template complex defect molecule, with the anchor site first in the sites list, followed
    # by the order of ``complex_defect_sites`` (without the anchor site):
    # technically this is not required due to the use of (efficient) permutation-invariant
    # BruteForceOrderMatcher in molecule matching, but it should aid efficiency for early breaking in the
    # permutation searches
    complex_mol = Molecule(
        [
            site.species
            for site in [anchor_site, *[site for site in complex_defect_sites if site != anchor_site]]
        ],
        bulk_supercell.lattice.get_cartesian_coords(
            [
                site.frac_coords + anchor_site.distance_and_image(site)[1]
                for site in [
                    anchor_site,
                    *[site for site in complex_defect_sites if site != anchor_site],
                ]
            ]
        ),
    )  # Note that an alternative complex defect featurisation approach could use e.g. SOAP vectors, but
    # the molecular representation is more intuitive, queryable, physically guided, and is sufficiently
    # efficient in this implementation

    def sites_lists_from_molecules(molecules: list[Molecule]) -> list[list[PeriodicSite]]:
        """
        Convert list of ``Molecule`` to list of lists of ``PeriodicSite``.
        """
        return [
            [
                PeriodicSite(
                    species=site.species,
                    coords=site.coords,
                    lattice=primitive_structure.lattice,
                    coords_are_cartesian=True,
                )
                for site in molecule.sites
            ]
            for molecule in molecules
        ]

    other_complex_equiv_prim_frac_coords_dict = {
        k: v for k, v in complex_equiv_prim_frac_coords_dict.items() if k != anchor_site
    }  # dict of sets of equivalent frac coords in primitive unit cell, for each point defect (exc anchor)

    equiv_complex_molecules: list[Molecule] = []
    for equiv_anchor_frac_coords in complex_equiv_prim_frac_coords_dict[anchor_site]:
        # for each anchor site, generate a dict of (other) candidate point defect sites, with the other
        # site as the key, and the set of its equivalent frac coords (wrt primitive unit cell) with
        # inter-defect distances matching the input/template defect complex (+/-5%) as the value
        candidate_point_defect_frac_coords = {
            other_site: {
                tuple(i[0].round(4))  # rounded frac coords for efficiency
                for i in primitive_structure.lattice.get_points_in_sphere(
                    [list(i) for i in other_equiv_frac_coords],  # equivalent frac coords in primitive
                    primitive_structure.lattice.get_cartesian_coords(equiv_anchor_frac_coords),  # center
                    r=complex_defect_defect_dist_dict[frozenset((anchor_site, other_site))] * 1.05,
                )
                if i[1]  # i[1] is the distance, compare to the stored inter-defect distance:
                > complex_defect_defect_dist_dict[frozenset((anchor_site, other_site))] * 0.95
            }
            for other_site, other_equiv_frac_coords in other_complex_equiv_prim_frac_coords_dict.items()
        }  # dict of sets of candidate point defect sites

        candidate_frac_coords_sets_combinations = product(  # all combinations of candidate frac coords
            *list(candidate_point_defect_frac_coords.values())
        )  # product retains the A,B,C etc order of the input iterables, so we can track site species with
        # the list of dict keys:
        candidate_site_combinations = list(candidate_point_defect_frac_coords.keys())

        # generate all candidate complex defect molecules from these frac coords / site combinations, with
        # the order matching the template complex defect molecule (anchor site, *complex_defect_sites --
        # with the latter having been the order of the dict generation, retained by itertools.product):
        candidate_equiv_molecules = {  # set of candidate complex defect molecules
            Molecule(
                [site.species for site in [anchor_site, *candidate_site_combinations]],
                primitive_structure.lattice.get_cartesian_coords(
                    [list(i) for i in [equiv_anchor_frac_coords, *candidate_frac_coords_combo]]
                ),
            )
            for candidate_frac_coords_combo in candidate_frac_coords_sets_combinations
            if not any(  # avoid potential duplicate sites
                np.allclose(cand_frac_coords_1, cand_frac_coords_2, atol=frac_tol * 2)
                for cand_frac_coords_1, cand_frac_coords_2 in combinations(
                    [equiv_anchor_frac_coords, *candidate_frac_coords_combo], 2
                )
            )
        }

        # now reduce to only those which are symmetry-equivalent to the template complex defect molecule,
        # but are distinct (i.e. not identical or periodic images; to only count those per unit cell)

        matching_candidate_equiv_molecules: list[Molecule] = [
            candidate_equiv_mol  # complex defect molecules which are symmetry-equivalent to the template
            for candidate_equiv_mol in candidate_equiv_molecules
            if are_equivalent_molecules(complex_mol, candidate_equiv_mol, tol=dist_tol * 2)
        ]
        unique_candidate_equiv_molecules: list[Molecule] = []
        # note that we don't use a list comprehension here, so we can compare each matching candidate
        # molecule to those in ``equiv_complex_molecules`` _and_ to all others in the _current_
        # ``unique_candidate_equiv_molecules`` list
        for candidate_equiv_mol in matching_candidate_equiv_molecules:
            if not any(  # complex defect molecules which are not not identical or periodic images
                is_periodic_image(
                    [
                        primitive_structure.lattice.get_fractional_coords(site.coords)
                        for site in other_equiv_mol.sites
                    ],
                    [
                        primitive_structure.lattice.get_fractional_coords(site.coords)
                        for site in candidate_equiv_mol.sites
                    ],
                    frac_tol=frac_tol,
                )
                for other_equiv_mol in [*equiv_complex_molecules, *unique_candidate_equiv_molecules]
            ):
                unique_candidate_equiv_molecules.append(candidate_equiv_mol)

        equiv_complex_molecules.extend(unique_candidate_equiv_molecules)

    if len(equiv_complex_molecules) < min_multiplicity:
        raise RuntimeError(
            f"Calculated complex defect site multiplicity {len(equiv_complex_molecules)} (in the "
            f"primitive cell) is lower than the theoretical minimum multiplicity {min_multiplicity}, "
            f"indicating an error in the analysis. Please report this bug to the developers!"
        )

    return (
        equiv_complex_molecules
        if return_molecules
        else sites_lists_from_molecules(equiv_complex_molecules)
    )


def get_complex_defect_multiplicity(
    bulk_supercell: Structure,
    vacancy_sites: Iterable | PeriodicSite | None = None,
    interstitial_sites: Iterable | PeriodicSite | None = None,
    substitution_sites: Iterable | PeriodicSite | None = None,
    primitive_structure: Structure | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    primitive_cell_multiplicity: bool = True,
    **kwargs,
) -> int:
    r"""
    Get the multiplicity of a given complex defect configuration (as given by
    the combination of input constituent point defect sites).

    The complex defect multiplicity is given by the number of distinct
    symmetry-equivalent complex defect site configurations in the unit cell.
    The returned multiplicity value corresponds to the `primitive cell` site
    multiplicity by default (``primitive_cell_multiplicity = True``).

    The input sites should correspond to the input bulk supercell.

    The multiplicity is calculated by generating all possible
    symmetry-equivalent complex defect site configurations in the primitive
    unit cell; see ``get_equivalent_complex_defect_sites_in_primitive()`` for
    details.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure to which the input sites correspond.
        vacancy_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of vacancies in the defect complex. Default is
            ``None``.
        interstitial_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of interstitials in the defect complex. Default is
            ``None``.
        substitution_sites (Iterable[PeriodicSite] | PeriodicSite | None):
            The site(s) of substitutions in the defect complex. Default is
            ``None``.
        primitive_structure (Structure | None):
            The primitive unit cell structure, in which to get equivalent
            complex defect sites. If ``None`` (default), the primitive
            structure will be determined from the bulk supercell.
        symprec (float):
            Symmetry precision for determining the primitive structure (if not
            provided), supercell symmetry operations and equivalent defect
            sites in the primitive unit cell. Defaults to 0.01. Note that this
            should match the value used for determining the point defect
            multiplicities (e.g. with the ``Defect.get_multiplicity()``
            methods) for appropriate comparisons -- the same default of 0.01 is
            used in all relevant ``doped`` functions.
            If ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default),
            this value will be automatically adjusted (up to 10x, down to 0.1x)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``symprec`` (and
            ``dist_tol_factor`` values).
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), searching for equivalent sites (by
            inter-defect distances) and matching equivalent complex defect
            geometries (as ``Molecule``\s), as a multiplicative factor of
            ``symprec``. Default is 1.0 (i.e. ``dist_tol = symprec``, in Å).
            Note that this should match the value used for determining the
            point defect multiplicities (e.g. with the
            ``Defect.get_multiplicity()`` methods) for appropriate comparisons
            -- the same default of 0.01 Å is used in all relevant ``doped``
            functions. If ``fixed_symprec_and_dist_tol_factor`` is ``False``
            (default), this value will also be automatically adjusted if
            necessary (up to 10x, down to 0.1x)(after ``symprec`` adjustments)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``dist_tol_factor`` (and
            ``symprec``) values.
        primitive_cell_multiplicity (bool):
            Whether to return the site multiplicity in the primitive unit cell
            (``True``) or the bulk supercell (``False``). Default is ``True``.
        **kwargs:
            Additional keyword arguments to pass to ``get_all_equiv_sites``
            (via ``get_equiv_frac_coords_in_primitive()``), such as
            ``fixed_symprec_and_dist_tol_factor`` and ``verbose``.

    Returns:
        int:
            The multiplicity of the input complex defect configuration, in the
            primitive unit cell if ``primitive_cell_multiplicity = True``
            (default), or in the bulk supercell if
            ``primitive_cell_multiplicity = False``.
    """
    equiv_complex_molecules = get_equivalent_complex_defect_sites_in_primitive(
        bulk_supercell,
        vacancy_sites=vacancy_sites,
        interstitial_sites=interstitial_sites,
        substitution_sites=substitution_sites,
        primitive_structure=primitive_structure,
        symprec=symprec,
        dist_tol_factor=dist_tol_factor,
        return_molecules=True,  # for speed
        **kwargs,
    )
    return len(equiv_complex_molecules) * (
        1
        if primitive_cell_multiplicity
        else round(
            len(bulk_supercell) / len(primitive_structure or get_primitive_structure(bulk_supercell))
        )
    )


def are_equivalent_molecules(molecule_1: Molecule, molecule_2: Molecule, tol: float = 0.02) -> bool:
    """
    Determine if two ``Molecule`` objects are equivalent, using the Kabsch
    algorithm (which minimizes the root-mean-square-deviation (RMSD) of two
    molecules which are topologically (atom types, geometry) similar) as
    implemented in the ``BruteForceOrderMatcher`` class, which allows
    permutation invariance in the molecule definitions.

    Uses caching to speed up the comparison.
    """
    return _cached_equivalent_molecules(molecule_1, molecule_2, tol)


@lru_cache(maxsize=int(1e4))
def _cached_equivalent_molecules(molecule_1: Molecule, molecule_2: Molecule, tol: float = 0.1) -> bool:
    # Note that in most cases with the ``get_equivalent_complex_defect_sites_in_primitive`` implementation,
    # brute-force permutation matching should not be required, due to our controlled ordering and
    # comparisons of equivalent sites, but current implementation is super-fast (particularly for small
    # molecules, like complexes), so worth being safe
    rmsd = BruteForceOrderMatcher(molecule_1).fit(molecule_2, break_on_tol=tol)[-1]
    return rmsd < tol


# TODO: In future, should be able to use similar code to generate all possible complexes in a given
# supercell
# TODO: Update DOIs here and throughout when published
# TODO: Tests!!
