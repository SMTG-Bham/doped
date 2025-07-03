"""
Code for generating and analysing defect complexes.
"""

import contextlib
import math
import warnings
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from itertools import combinations, product
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher
from pymatgen.util.coord import get_angle, pbc_shortest_vectors
from tqdm import tqdm

from doped.core import _get_single_valence_oxi_states, guess_and_set_oxi_states_with_timeout
from doped.utils.efficiency import Composition, DopedVacancyGenerator, Molecule, PeriodicSite, Structure
from doped.utils.parsing import (
    get_coords_and_idx_of_species,
    get_defect_type_and_composition_diff,
    get_defect_type_and_site_indices,
    get_matching_site,
)
from doped.utils.symmetry import (
    get_equiv_frac_coords_in_primitive,
    get_primitive_structure,
    is_periodic_image,
)

if TYPE_CHECKING:
    from doped.generation import DefectsGenerator


def classify_vacancy_geometry(
    bulk_supercell: Structure,
    vacancy_supercell: Structure,
    site_tol: float = 0.5,
    abs_tol: bool = False,
    verbose: bool = False,
    use_oxi_states: bool = False,
) -> str:
    """
    Classify the geometry of a given vacancy in a supercell, as either a
    'simple' point vacancy, a 'split' vacancy, or a 'non-trivial' vacancy.

    Split vacancy geometries are those where 2 vacancies and 1 interstitial
    are found to be present in the defect structure, as determined using
    site-matching between the defect and bulk structures with a distance
    tolerance (``site_tol``), such that the absence of any site of matching
    species within the distance tolerance to the original bulk site is
    considered a vacancy, and vice versa in comparing the bulk to the defect
    structure is an interstitial. This corresponds to the 2 V_X + X_i
    definition of split vacancies geometries as discussed in
    https://doi.org/10.1088/2515-7655/ade916

    On the other hand, a simple (point) vacancy corresponds to cases where 1
    site from the bulk structure cannot be matched to the defect structure
    while all defect structure sites can be matched to bulk sites, and
    'non-trivial' vacancies refer to all other cases (multiple off-site atoms,
    which don't match the split vacancy classification).

    Inspired by the vacancy geometry classification used in Kumagai et al.
    `Phys Rev Mater` 2021. See https://doi.org/10.1088/2515-7655/ade916 for
    further details.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure to compare against for site-matching.
        vacancy_supercell (Structure):
            The defect supercell containing the vacancy to be classified.
        site_tol (float):
            The (fractional) tolerance for matching sites between the defect
            and bulk structures. If ``abs_tol`` is ``False`` (default), then
            this value multiplied by the shortest bond length in the bulk
            structure will be used as the distance threshold for matching,
            otherwise the value is used directly (as a length in Å).
            Default is 0.5 (i.e. half the shortest bond length in the bulk
            structure).
        abs_tol (bool):
            Whether to use ``site_tol`` as an absolute distance tolerance (in
            Å) instead of a fractional tolerance (in terms of the shortest bond
            length in the structure). Default is ``False``.
        verbose (bool):
            Whether to print additional information about the classification
            for non-trivial vacancies.
            Default is ``False``.
        use_oxi_states (bool):
            Whether to use the oxidation states of the sites in the bulk and
            defect structures when considering matching sites (such that e.g.
            ``Fe3+`` and ``Fe2+`` would be considered different species).
            Default is ``False``.

    Returns:
        str:
            The classification of the vacancy geometry, which can be one of
            "Simple Vacancy", "Split Vacancy", or "Non-Trivial".
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

    defect_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, vacancy_supercell)
    if defect_type != "vacancy":
        warnings.warn(
            f"Note that based on the defect/bulk structure composition difference ({comp_diff}), the "
            f"defect supercell is not classified as a vacancy, but as a {defect_type}."
        )
    old_species = next(el for el, amt in comp_diff.items() if amt == -1)
    _, bulk_species_indices = get_coords_and_idx_of_species(
        bulk_supercell, old_species, use_oxi_states=use_oxi_states
    )
    _, vacancy_species_indices = get_coords_and_idx_of_species(
        vacancy_supercell, old_species, use_oxi_states=use_oxi_states
    )

    defect_type, missing_bulk_site_indices, additional_defect_site_indices = (
        get_defect_type_and_site_indices(
            bulk_supercell,
            vacancy_supercell,
            site_tol=site_tol,
            abs_tol=abs_tol,
        )
    )
    # ignore indices for other species:
    missing_bulk_site_indices = np.asarray(missing_bulk_site_indices)[
        np.isin(missing_bulk_site_indices, bulk_species_indices)
    ]
    additional_defect_site_indices = np.asarray(additional_defect_site_indices)[
        np.isin(additional_defect_site_indices, vacancy_species_indices)
    ]

    if len(missing_bulk_site_indices) == 1 and len(additional_defect_site_indices) == 0:
        return "Simple Vacancy"

    if len(missing_bulk_site_indices) == 2 and len(additional_defect_site_indices) == 1:
        return "Split Vacancy"

    if verbose:  # otherwise, we have a non-trivial vacancy
        print(f"{len(additional_defect_site_indices)} offsite atoms in defect compared to bulk")
        print(f"{len(missing_bulk_site_indices)} offsite atoms in bulk compared to defect")

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


def _estimate_ES_time(num_candidates: int, cell_size: int) -> float:
    """
    Convenience function to electrostatic estimation compute time per process,
    in seconds, using ``EwaldSummation(candidate_structure).total_energy``
    (i.e. ``get_es_energy()``).

    From fitting and testing with/without multiprocessing over different
    sizes / compositions, on a Macbook Pro 2021. Quadratic as expected.

    This was previously used to dynamically adjust the search radius for
    candidate low energy complex defects (e.g. split vacancies), but has been
    mostly deprecated by the use of ``EwaldMinimizer`` to efficiently estimate
    the electrostatic energies of many candidate configurations within the same
    supercell.

    Args:
        num_candidates (int):
            The number of candidate structures (with ``cell_size`` sites) to be
            estimated.
        cell_size (int):
            The size of the cell, in number of sites/atoms.

    Returns:
        float: The estimated time to compute the electrostatic energy of the
        candidate structures, in seconds.
    """
    return 1.7e-5 * num_candidates * cell_size**2


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
    # TODO: Combine with _create_unrelaxed_defect_structure, and just make it
    #  create_defect_structure_from_sites?
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
    complex_mol = molecule_from_sites(raw_complex_defect_sites, anchor_idx=0)
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
    complex_mol = molecule_from_sites(
        [anchor_site, *[site for site in complex_defect_sites if site != anchor_site]], anchor_idx=0
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
# TODO: Tests!! (See notebooks for draft tests)


def molecule_from_sites(sites: list[PeriodicSite], anchor_idx: int = 0) -> Molecule:
    """
    Generate a ``Molecule`` from a list of ``PeriodicSite`` objects, accounting
    for periodic boundary conditions.

    ``sites[anchor_idx]`` is taken as the 'anchor' site, from which the closest
    periodic image of each other site is used in constructing the molecule.

    Args:
        sites (list[PeriodicSite]):
            The list of ``PeriodicSite`` objects to generate a ``Molecule``
            from.
        anchor_idx (int):
            The index of the anchor site in the list of ``PeriodicSite``
            objects. Default is 0.

    Returns:
        Molecule:
            The ``Molecule`` object generated from the list of ``PeriodicSite``
            objects.
    """
    lattice = sites[0].lattice
    return Molecule(
        [site.species for site in sites],
        lattice.get_cartesian_coords(
            [site.frac_coords + sites[anchor_idx].distance_and_image(site)[1] for site in sites]
        ),
    )


# TODO: Use check like this to determine whether to call this function in default DefectsGenerator
#  workflow
# if self.kwargs.get("skip_split_vacancies", False):
#     print(
#         "Skipping split vacancies as 'skip_split_vacancies' is set to True in kwargs."
#     )
#     return False
def get_split_vacancies(
    defect_gen: "DefectsGenerator",
    elements: Iterable[str] | str | None = None,
    bulk_oxi_states: Structure | Composition | dict[str, int] | None = None,
    relative_electrostatic_energy_tol: float = 1.1,
    split_vac_dist_tol: float = 5,
    verbose: bool = True,  # TODO: Use verbose = False in DefectsGen
    **kwargs,
):
    """
    TODO. Elements can elt strings, "all" or None (cations only).

    Todo:
    Note that this function requires the bulk oxidation states to be set
    (in the ``DefectsGenerator._bulk_oxi_states`` attribute), which is
    done automatically in ``DefectsGenerator`` initialisation when oxidation
    states can be successfully guessed. Additionally, this function assumes
    single oxidation states for each element in the bulk structure (i.e.
    does not account for any mixed valence).
    """
    bulk_supercell = defect_gen.bulk_supercell
    bulk_oxi_states = bulk_oxi_states or defect_gen._bulk_oxi_states
    if not bulk_oxi_states and (bulk_oxi_states := guess_and_set_oxi_states_with_timeout(bulk_supercell)):
        print(
            f"Guessed oxidation states for input structure: "
            f"{_get_single_valence_oxi_states(bulk_oxi_states)}"
        )

    if elements is None and bulk_oxi_states is None:
        raise ValueError(  # TODO: Will need to handle this in DefectsGenerator usage
            "Elements for split vacancy generation were not explicitly set; the default is to generate "
            "for cations only, however oxidation states could not be guessed for the input structure. "
            "Please explicitly provide oxidation states using `bulk_oxi_states`."
        )

    single_valence_oxi_states = _get_single_valence_oxi_states(bulk_oxi_states)
    cations = {elt for elt, oxi in single_valence_oxi_states.items() if oxi > 0}

    if elements is None:
        elements = cations
    elif elements == "all":
        elements = set(defect_gen.bulk_supercell.composition.elements)
    elif isinstance(elements, str):
        elements = {elements}
    elif isinstance(elements, Iterable):
        elements = set(elements)
    else:
        raise ValueError(
            f"Invalid elements input: {elements}. Please provide a list of elements, 'all', or None."
        )

    combined_split_vacancies_dict = {}
    for element in elements:
        interstitial_sites = {
            entry.defect_supercell_site
            for entry in defect_gen.defect_entries.values()
            if entry.name.startswith(f"{element}_i")
        }
        if not interstitial_sites:
            warnings.warn(
                f"No interstitial sites (from DefectEntry.defect_supercell_site) found for element "
                f"{element} -- required for split vacancy generation!"
            )
            continue

        try:  # try from database, otherwise print info message and try from electrostatics
            split_vacancies_dict = get_split_vacancies_from_database(
                verbose=(
                    "Will use electrostatic analysis to check for candidate low-energy split "
                    "vacancies. Set skip_split_vacancies=True in DefectsGenerator() to skip this step."
                )
            )
        except Exception as exc:
            warnings.warn(
                f"Error getting split vacancies from database: {exc}\nGenerating from electrostatic "
                f"analysis instead."
            )
            split_vacancies_dict = get_split_vacancies_from_electrostatics(
                bulk_supercell=bulk_supercell,
                interstitial_sites=interstitial_sites,
                bulk_oxi_states=bulk_oxi_states,
                relative_electrostatic_energy_tol=relative_electrostatic_energy_tol,
                split_vac_dist_tol=split_vac_dist_tol,
                verbose=verbose,
                **kwargs,
            )

        combined_split_vacancies_dict[element] = split_vacancies_dict

    return combined_split_vacancies_dict


def get_split_vacancies_by_geometry(
    bulk_supercell: Structure,
    interstitial_sites: Iterable[PeriodicSite] | PeriodicSite,
    split_vac_dist_tol: float = 5.0,
    all_species: bool = False,
    prune_symmetry_equivalent: bool = True,
    show_pbar: bool = False,
    **kwargs,
) -> dict[frozenset[float] | int, dict]:
    """
    Generate inequivalent split vacancy configurations (i.e. vacancy -
    interstitial - vacancy complexes) for the given interstitial sites with a
    maximum vacancy-interstitial distance of ``split_vac_dist_tol`` in Å.

    Split vacancies are generated by finding all possible
    vacancy-interstitial-vacancy complexes with vacancy-interstitial
    distances less than ``split_vac_dist_tol`` Å using the set of input
    interstitial sites, and removing any symmetry-equivalent duplicates (based
    on the VIV distances, rounded to 0.01 Å, and V-I-V bond angle, rounded to
    0.1°).

    See https://doi.org/10.1088/2515-7655/ade916 for further details.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure in which to generate split vacancies.
        interstitial_sites (Iterable[PeriodicSite] | PeriodicSite):
            The set of interstitial sites to consider for split vacancy
            generation.
        split_vac_dist_tol (float):
            The maximum distance between vacancy and interstitial sites
            to allow for candidate split vacancies (which are
            vacancy-interstitial-vacancy complexes). Default is 5.0 Å.
        all_species (bool):
            Whether to consider `all species for vacancy generation, rather
            than just vacancies for species matching the interstitial species.
            Default is ``False``.
        prune_symmetry_equivalent (bool):
            Whether to prune symmetry-equivalent split vacancies based on the
            VIV distances and bond angle as mentioned above. In rare /
            low-symmetry cases, this could lead to unintentional reduction of
            similar but not fully symmetry-equivalent candidates. Default is
            ``True``.
        show_pbar (bool):
            Whether to show a progress bar during generation.
            Default is ``True``.
        **kwargs:
            Additional keyword arguments to use for vacancy generation, such as
            ``symprec``.

    Returns:
        dict:
            A dictionary of candidate split vacancies, with the
            vacancy-interstitial distances as keys and a dictionary
            of the defect information as values.
    """
    if isinstance(interstitial_sites, PeriodicSite):
        interstitial_sites = [interstitial_sites]
    candidate_split_vacancies: dict[frozenset[float] | int, dict] = {}
    defect_init_kwargs = {
        "symprec": 0.01,
        "oxi_state": 0,
        "multiplicity": 1,
    }  # to avoid slow pymatgen functions for these
    defect_init_kwargs.update(kwargs)
    doped_vacancy_generator = DopedVacancyGenerator()

    if show_pbar:
        interstitial_sites = tqdm(
            interstitial_sites, desc="Generating split vacancies from geometry analysis..."
        )

    lattice = bulk_supercell.lattice
    split_vac_dist_tol_squared = split_vac_dist_tol**2
    i = 0
    for interstitial_site in interstitial_sites:
        vac_gen_kwargs = {
            "rm_species": None if all_species else {interstitial_site.specie.symbol},
            **defect_init_kwargs,
        }
        defect_supercell = bulk_supercell.copy()
        defect_supercell.append(
            interstitial_site.species, interstitial_site.coords, coords_are_cartesian=True
        )
        for vac in doped_vacancy_generator.generate(structure=defect_supercell, **vac_gen_kwargs):  # type: ignore
            vec_1, v1i_squared_dists = pbc_shortest_vectors(
                lattice, vac.site.frac_coords, interstitial_site.frac_coords, return_d2=True
            )
            v1i_dist_squared = v1i_squared_dists[0][0]
            if (
                v1i_dist_squared < split_vac_dist_tol_squared and v1i_dist_squared > 0.1
            ):  # d=0 for vac @ int
                for second_vac in doped_vacancy_generator.generate(
                    structure=vac.defect_structure, **vac_gen_kwargs  # type: ignore
                ):
                    vec_2, v2i_squared_dists = pbc_shortest_vectors(
                        lattice,
                        second_vac.site.frac_coords,
                        interstitial_site.frac_coords,
                        return_d2=True,
                    )
                    v2i_dist_squared = v2i_squared_dists[0][0]
                    if v2i_dist_squared < split_vac_dist_tol_squared and v2i_dist_squared > 0.1:
                        v1i_dist = round(np.sqrt(v1i_dist_squared), 3)
                        v2i_dist = round(np.sqrt(v2i_dist_squared), 3)
                        angle = get_angle(vec_1[0][0], vec_2[0][0], units="degrees")
                        candidate_split_vacancies[
                            (
                                frozenset((round(v1i_dist, 2), round(v2i_dist, 2), round(angle, 1)))
                                if prune_symmetry_equivalent
                                else i
                            )
                        ] = {
                            "interstitial_site": interstitial_site,
                            "vacancy_1_site": vac.site,
                            "vacancy_2_site": second_vac.site,
                            "vacancy_1_interstitial_distance": v1i_dist,
                            "vacancy_2_interstitial_distance": v2i_dist,
                            "VIV_distances": f"{v1i_dist:.2f}_{v2i_dist:.2f}",
                            "VIV_bond_angle_degrees": round(angle, 3),
                        }
                        i += 1

    return candidate_split_vacancies


def get_split_vacancies_from_electrostatics(
    bulk_supercell: Structure,
    interstitial_sites: Iterable[PeriodicSite] | PeriodicSite,
    bulk_oxi_states: Structure | Composition | dict[str, int] | None = None,
    relative_electrostatic_energy_tol: float = 1.1,
    split_vac_dist_tol: float = 5,
    verbose: bool = True,
    ndigits: int = 2,
    prune_symmetry_equivalent: bool = True,
    **kwargs,
):
    """
    Generate inequivalent split vacancy configurations (i.e. vacancy -
    interstitial - vacancy complexes) for the given interstitial sites, using
    geometric and electrostatic analyses.

    Candidate split vacancies are generated by finding all possible
    vacancy-interstitial-vacancy complexes with vacancy-interstitial
    distances less than ``split_vac_dist_tol`` Å using the set of input
    interstitial sites, and removing any symmetry-equivalent duplicates (based
    on the VIV distances, rounded to 0.01 Å, and V-I-V bond angle, rounded to
    0.1°). The electrostatic formation energies (i.e. electrostatic energy of
    the split vacancy supercell minus that of the bulk supercell) are then
    calculated, and only those within ``relative_electrostatic_energy_tol``
    (default = 1.1) times the lowest point vacancy electrostatic formation
    energy are returned.

    See https://doi.org/10.1088/2515-7655/ade916 for further details.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure in which to generate split vacancies.
        interstitial_sites (Iterable[PeriodicSite] | PeriodicSite):
            The set of interstitial sites to consider for split vacancy
            generation.
        bulk_oxi_states (Structure | Composition | dict[str, int] | None):
            The oxidation states of elements to use for electrostatic energy
            evaluations. If not provided, oxidation states will be taken from
            the bulk supercell, or otherwise guessed. Default is ``None``.
        relative_electrostatic_energy_tol (float):
            Relative tolerance for selecting candidate split vacancy
            configurations. Split vacancies with electrostatic formation
            energies (i.e. electrostatic energy of the split vacancy supercell
            minus that of the bulk supercell) less than
            ``relative_electrostatic_energy_tol`` times the lowest point
            vacancy electrostatic formation energy are returned. Default
            ``1.1``.
        split_vac_dist_tol (float):
            The maximum distance between vacancy and interstitial sites
            to allow for candidate split vacancies (which are
            vacancy-interstitial-vacancy complexes). Default is 5.0 Å.
        verbose (bool):
            Whether to print verbose information about the generation process.
            Default is ``True``.
        ndigits (int):
            The number of decimal places to round the electrostatic formation
            energies to, which is used to avoid degenerate configurations.
            Default is 2.
        prune_symmetry_equivalent (bool):
            Whether to prune symmetry-equivalent split vacancies based on the
            VIV distances and bond angle (see
            ``get_split_vacancies_by_geometry``). In rare / low-symmetry cases,
            this could lead to unintentional reduction of similar but not fully
            symmetry-equivalent candidates. Default is ``True``.
        **kwargs:
            Additional keyword arguments to use for vacancy generation, such as
            ``symprec``.

    Returns:
        dict:
            A dictionary with information about candidate split vacancies,
            including:

            - dictionary of split vacancy electrostatic formation energies as
              keys and the constituent vacancy and interstitial sites as a
              dictionary for values,
            - the minimum electrostatic formation energies for split vacancies,
              point vacancies, interstitials and isolated vacancy-interstitial-
              vacancy combinations,
            - the number of split vacancies within the
              ``relative_electrostatic_energy_tol`` tolerance, or lower
              electrostatic formation energy that the lowest point vacancy
              electrostatic formation energy,
            - the input structure (bulk supercell) with oxidation states added,
            - the absolute electrostatic energy of the input structure.
    """
    if (  # if input structure was oxi-state-decorated, use these oxi states:
        all(hasattr(site.specie, "oxi_state") for site in bulk_supercell.sites)
        and all(isinstance(site.specie.oxi_state, int | float) for site in bulk_supercell.sites)
        and bulk_oxi_states is None
    ):
        bulk_oxi_states = bulk_supercell
    if bulk_oxi_states is None:
        if bulk_oxi_states := guess_and_set_oxi_states_with_timeout(bulk_supercell):
            print(
                f"Guessed oxidation states for input structure: "
                f"{_get_single_valence_oxi_states(bulk_oxi_states)}"
            )
        else:
            raise ValueError(
                "Oxidation states could not be guessed for the input structure, please explicitly "
                "provide oxidation states using `bulk_oxi_states`."
            )

    single_valence_oxi_states = _get_single_valence_oxi_states(bulk_oxi_states)
    bulk_supercell_w_oxi_states = bulk_supercell.copy()
    bulk_supercell_w_oxi_states.add_oxidation_state_by_element(single_valence_oxi_states)

    try:  # generate candidate split vacancies from geometric analysis:
        candidate_split_vacancies = get_split_vacancies_by_geometry(
            bulk_supercell_w_oxi_states,
            interstitial_sites,
            split_vac_dist_tol=split_vac_dist_tol,
            all_species=False,
            prune_symmetry_equivalent=prune_symmetry_equivalent,
            symprec=kwargs.get("symprec", 0.01),
        )
    except Exception as exc:
        print(f"Error generating split vacancies: {exc}")

    if not candidate_split_vacancies:
        print("No candidate split vacancies found!")
        return None

    if verbose:
        print(
            f"{len(candidate_split_vacancies)} candidate split vacancies found, with vacancy-interstitial "
            f"distance tolerance of {split_vac_dist_tol} Å. Evaluating electrostatic energies..."
        )

    # determine electrostatic energies of candidate split vacancies:
    # note that here we use a modified implementation of that used in the original work, using
    # EwaldMinimizer for fast electrostatic energy evaluations (rather than EwaldSummation with
    # multiprocessing), which results in greater initial pruning of high electrostatic energy
    # candidates:
    unique_interstitial_sites = {
        subdict["interstitial_site"] for subdict in candidate_split_vacancies.values()
    }
    unique_interstitial_sites_idx_dict = {
        len(bulk_supercell_w_oxi_states) + i: interstitial_site
        for i, interstitial_site in enumerate(unique_interstitial_sites)
    }
    bulk_supercell_with_all_interstitial_sites = bulk_supercell_w_oxi_states.copy()
    for interstitial_site in unique_interstitial_sites_idx_dict.values():
        bulk_supercell_with_all_interstitial_sites.append(
            species=interstitial_site.specie,
            coords=interstitial_site.frac_coords,
            properties=interstitial_site.properties,
        )  # idx will be len(bulk_supercell_with_all_interstitial_sites) + i
    bulk_supercell_with_all_interstitial_sites.add_oxidation_state_by_element(single_valence_oxi_states)

    # generate manipulations for EwaldMinimizer; manipulations are of the format:
    # [oxi_state_multiplication_factor, num_sites_to_place, allowed_site_indices, species]
    unique_vacancy_sites = {
        site
        for d in candidate_split_vacancies.values()
        for site in (d["vacancy_1_site"], d["vacancy_2_site"])
    }
    unique_vacancy_sites_idx_dict = {}
    for vac_site in unique_vacancy_sites:
        matching_site = get_matching_site(vac_site, bulk_supercell_w_oxi_states)
        unique_vacancy_sites_idx_dict[bulk_supercell_w_oxi_states.index(matching_site)] = vac_site

    base_manipulations = [
        [0, 0, list(unique_vacancy_sites_idx_dict.keys()), None],  # no vacancies
        [
            0,
            len(unique_interstitial_sites_idx_dict),
            list(unique_interstitial_sites_idx_dict.keys()),
            None,
        ],  # remove all interstitials
    ]  # this would correspond to getting the base bulk supercell, no vacancies or interstitials
    split_vac_manipulations = deepcopy(base_manipulations)
    split_vac_manipulations[0][1] = 2  # two vacancies per split vac
    split_vac_manipulations[1][1] = (
        len(unique_interstitial_sites_idx_dict) - 1
    )  # remove all but 1 interstitial
    bulk_w_all_int_matrix = EwaldSummation(bulk_supercell_with_all_interstitial_sites).total_energy_matrix

    # get bulk ES energy; deepcopy to avoid modifying base_manipulations
    ewald_m = EwaldMinimizer(bulk_w_all_int_matrix, deepcopy(base_manipulations), num_to_return=1)
    abs_bulk_es_energy = round(ewald_m.output_lists[0][0], ndigits)

    split_vac_ewald_m = EwaldMinimizer(
        bulk_w_all_int_matrix,
        split_vac_manipulations,
        num_to_return=len(candidate_split_vacancies) * 10,
    )  # 10x number to return to partially account for duplicate generation in EwaldMinimizer (should only
    # really affect higher energy configurations, and we later round to ndigits to prune)

    split_vacancies_energy_dict = {}
    for output in split_vac_ewald_m.output_lists:
        int_idx = next(  # interstitial site that wasn't removed (i.e., the one that remains)
            i for i in unique_interstitial_sites_idx_dict if all(i != manip[0] for manip in output[1])
        )
        interstitial_site = unique_interstitial_sites_idx_dict[int_idx]
        vacancy_indices = [  # get vacancy indices (those not in interstitial sites)
            i[0] for i in output[1] if i[0] not in unique_interstitial_sites_idx_dict
        ]
        vacancy_sites = {unique_vacancy_sites_idx_dict[i] for i in vacancy_indices}

        split_vacancies_energy_dict[round(output[0], ndigits)] = {
            "interstitial_site": interstitial_site,
            "vacancy_sites": vacancy_sites,
        }

    # get single vacancy ES energy:
    point_vac_manipulations = deepcopy(base_manipulations)
    point_vac_manipulations[0][1] = 1  # one vacancy
    point_vac_ewald_m = EwaldMinimizer(
        bulk_w_all_int_matrix, point_vac_manipulations, num_to_return=1
    )  # just lowest energy
    abs_point_vac_es_energy = round(
        point_vac_ewald_m.output_lists[0][0], ndigits
    )  # min point vacancy ES energy

    # get interstitial ES energies:
    int_manipulations = deepcopy(base_manipulations)
    int_manipulations[1][1] = len(unique_interstitial_sites) - 1  # one interstitial
    int_ewald_m = EwaldMinimizer(bulk_w_all_int_matrix, int_manipulations, num_to_return=1)
    abs_int_es_energy = round(int_ewald_m.output_lists[0][0], ndigits)  # min interstitial ES energy

    # get relative ES energies:
    rel_point_vac_es_energy = abs_point_vac_es_energy - abs_bulk_es_energy
    rel_int_es_energy = abs_int_es_energy - abs_bulk_es_energy
    rel_isolated_viv_es_energy = rel_int_es_energy + (rel_point_vac_es_energy * 2)
    rel_min_split_vac_es_energy = min(split_vacancies_energy_dict) - abs_bulk_es_energy

    if verbose:
        print(f"Lowest point vacancy electrostatic energy: {rel_point_vac_es_energy:.2f}")
        print(f"Lowest interstitial electrostatic energy: {rel_int_es_energy:.2f}")
        print(
            f"Lowest isolated vacancy-interstitial-vacancy combination electrostatic energy: "
            f"{rel_isolated_viv_es_energy:.2f}"
        )
        print(
            f"Lowest split-vacancy (vacancy-interstitial-vacancy complex) electrostatic energy: "
            f"{rel_min_split_vac_es_energy:.2f}"
        )

    split_vacancies_energy_dict = {
        round(energy - abs_bulk_es_energy, ndigits): split_vacancies_energy_dict[energy]
        for energy in sorted(split_vacancies_energy_dict)
    }
    num_lower_energy_split_vacancies = len(
        [energy for energy in split_vacancies_energy_dict if energy < rel_point_vac_es_energy]
    )
    num_split_vacancies_within_tol = len(
        [
            energy
            for energy in split_vacancies_energy_dict
            if energy < rel_point_vac_es_energy * (relative_electrostatic_energy_tol)
        ]
    )
    split_vacancies_energy_dict = dict(
        sorted(split_vacancies_energy_dict.items())[:num_split_vacancies_within_tol]
    )

    if verbose:
        print(
            f"Found {num_lower_energy_split_vacancies} lower energy split vacancies, "
            f"{num_split_vacancies_within_tol} within {relative_electrostatic_energy_tol}x of the "
            f"lowest point vacancy electrostatic formation energy."
        )

    return {
        "split_vacancies_electrostatic_energy_dict": split_vacancies_energy_dict,
        "min_point_vacancy_electrostatic_formation_energy": rel_point_vac_es_energy,
        "min_interstitial_electrostatic_formation_energy": rel_int_es_energy,
        "min_isolated_viv_electrostatic_formation_energy": rel_isolated_viv_es_energy,
        "min_split_vac_electrostatic_formation_energy": rel_min_split_vac_es_energy,
        "bulk_supercell_with_oxi_states": bulk_supercell_w_oxi_states,
        "bulk_supercell_absolute_electrostatic_energy": abs_bulk_es_energy,
        "num_lower_energy_split_vacancies": num_lower_energy_split_vacancies,
        "num_split_vacancies_within_tol": num_split_vacancies_within_tol,
    }


def get_split_vacancies_from_database(*args, verbose: bool | str = False):
    """
    TODO.
    """
    raise NotImplementedError("Not implemented yet.")
