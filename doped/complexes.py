"""
Code for generating and analysing defect complexes.
"""

import contextlib
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
from functools import lru_cache
from itertools import combinations, pairwise, permutations, product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure_matcher import get_linear_assignment_solution
from pymatgen.util.coord import get_angle, lattice_points_in_supercell, pbc_shortest_vectors
from scipy.cluster.hierarchy import fcluster, linkage
from spglib import get_pointgroup
from tqdm import tqdm

from doped.core import Defect, DefectComplex, _get_oxi_state_modes, guess_and_set_oxi_states_with_timeout
from doped.utils.efficiency import Composition, DopedVacancyGenerator, Molecule, PeriodicSite, Structure
from doped.utils.parsing import (
    get_coords_and_idx_of_species,
    get_defect_type_and_composition_diff,
    get_defect_type_and_site_indices,
    get_matching_site,
)
from doped.utils.supercells import _get_unwrapped_complex_fc
from doped.utils.symmetry import (
    FclusterCriterion,
    LinkageMethod,
    _frac_coords_sort_func,
    cluster_coords,
    get_equiv_frac_coords_in_primitive,
    get_primitive_structure,
    get_sga_and_symprec,
    is_periodic_image_old,
    schoenflies_from_hermann,
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
            otherwise the value is used directly (as a length in Å).
            Default is 0.5 (i.e. half the shortest bond length in the bulk
            structure).
        abs_tol (bool):
            Whether to use ``site_tol`` as an absolute distance tolerance (in
            Å) instead of a fractional tolerance (in terms of the shortest bond
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
    oxi_state_decorated = [  # if all sites in both structures are not oxi-state decorated / neutral
        any(i in site.species_string for i in ["+", "-", "0"])
        for site in [*vacancy_supercell.sites, *bulk_supercell.sites]
    ]
    if len(set(oxi_state_decorated)) > 1 and use_oxi_states:  # not consistent, ignore oxi states:
        warnings.warn(
            "`use_oxi_states` was set to `True`, but not all sites in the bulk and defect structures are "
            "oxidation state decorated. Setting `use_oxi_states` to `False`."
        )
        use_oxi_states = False

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
    num_missing_bulk_sites = np.isin(missing_bulk_site_indices, bulk_species_indices).sum()
    num_additional_defect_sites = np.isin(additional_defect_site_indices, vacancy_species_indices).sum()

    if num_missing_bulk_sites == 1 and num_additional_defect_sites == 0:
        return "Simple Vacancy"

    if num_missing_bulk_sites == 2 and num_additional_defect_sites == 1:
        return "Split Vacancy"

    if verbose:  # otherwise, we have a non-trivial vacancy
        print(f"{num_additional_defect_sites} offsite atoms in defect compared to bulk")
        print(f"{num_missing_bulk_sites} offsite atoms in bulk compared to defect")

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


# TODO DEPRECATED
def get_equivalent_complex_defect_sites_in_primitive(
    bulk_supercell: Structure,
    vacancy_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    interstitial_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    substitution_sites: Iterable[PeriodicSite] | PeriodicSite | None = None,
    primitive_structure: Structure | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    return_molecules: bool = False,
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
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), searching for equivalent sites (by
            inter-defect distances) and matching equivalent complex defect
            geometries (as ``Molecule``\s), as a multiplicative factor of
            ``symprec``. Default is 1.0 (i.e. ``dist_tol = symprec``, in Å).
            Note that this should match the value used for determining the
            point defect multiplicities (e.g. with the
            ``Defect.get_multiplicity()`` methods) for appropriate comparisons
            -- the same default of 0.01 Å is used in all relevant ``doped``
            functions.
        return_molecules (bool):
            Whether to return the equivalent complex defect molecules as
            ``Molecule`` objects, or as lists of ``PeriodicSite``
            objects. Default is ``False`` (return lists of ``PeriodicSite``\s).

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
            tuple(frac_coords.round(4))  # type: ignore
            for frac_coords in cast(
                "list[np.ndarray] | np.ndarray",
                get_equiv_frac_coords_in_primitive(
                    site.frac_coords,
                    primitive_structure,
                    bulk_supercell,
                    symprec=symprec,
                    dist_tol_factor=dist_tol_factor,
                ),
            )
        }
        for site in complex_defect_sites
    }
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
            if are_equivalent_molecules(complex_mol, candidate_equiv_mol, tol=dist_tol)
        ]
        unique_candidate_equiv_molecules: list[Molecule] = []
        # note that we don't use a list comprehension here, so we can compare each matching candidate
        # molecule to those in ``equiv_complex_molecules`` _and_ to all others in the _current_
        # ``unique_candidate_equiv_molecules`` list
        for candidate_equiv_mol in matching_candidate_equiv_molecules:
            if not any(  # complex defect molecules which are not not identical or periodic images
                is_periodic_image_old(
                    [
                        primitive_structure.lattice.get_fractional_coords(site.coords)
                        for site in other_equiv_mol.sites
                    ],
                    [
                        primitive_structure.lattice.get_fractional_coords(site.coords)
                        for site in candidate_equiv_mol.sites
                    ],
                    frac_tol=frac_tol,
                    same_image=True,
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


# TODO DEPRECATED
def get_complex_defect_multiplicity(
    bulk_supercell: Structure,
    vacancy_sites: Iterable | PeriodicSite | None = None,
    interstitial_sites: Iterable | PeriodicSite | None = None,
    substitution_sites: Iterable | PeriodicSite | None = None,
    primitive_structure: Structure | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    primitive_cell_multiplicity: bool = True,
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
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), searching for equivalent sites (by
            inter-defect distances) and matching equivalent complex defect
            geometries (as ``Molecule``\s), as a multiplicative factor of
            ``symprec``. Default is 1.0 (i.e. ``dist_tol = symprec``, in Å).
            Note that this should match the value used for determining the
            point defect multiplicities (e.g. with the
            ``Defect.get_multiplicity()`` methods) for appropriate comparisons
            -- the same default of 0.01 Å is used in all relevant ``doped``
            functions.
        primitive_cell_multiplicity (bool):
            Whether to return the site multiplicity in the primitive unit cell
            (``True``) or the bulk supercell (``False``). Default is ``True``.

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
    )
    return len(equiv_complex_molecules) * (
        1
        if primitive_cell_multiplicity
        else round(
            len(bulk_supercell) / len(primitive_structure or get_primitive_structure(bulk_supercell))
        )
    )


# TODO: Add class links for others in this module, and update docstrings to use these.


def are_equivalent_molecules(molecule_1: Molecule, molecule_2: Molecule, tol: float = 0.01) -> bool:
    """
    Determine if two ``Molecule`` objects are equivalent, using the Kabsch
    algorithm (which minimizes the root-mean-square-deviation (RMSD) of two
    molecules which are topologically (atom types, geometry) similar) as
    implemented in the |BruteForceOrderMatcher| class, which allows permutation
    invariance in the molecule definitions.

    Uses caching to speed up the comparison.

    Args:
        molecule_1 (|Molecule|):
            The first ``pymatgen`` |Molecule| object to compare.
        molecule_2 (|Molecule|):
            The second ``pymatgen`` |Molecule| object to compare.
        tol (float):
            The tolerance for the Kabsch algorithm. Default is 0.01 Å.

    Returns:
        bool:
            ``True`` if the two |Molecule| objects are equivalent, ``False`` otherwise.
    """
    return _cached_equivalent_molecules(molecule_1, molecule_2, tol)


@lru_cache(maxsize=int(1e4))
def _cached_equivalent_molecules(molecule_1: Molecule, molecule_2: Molecule, tol: float = 0.01) -> bool:
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
        print(f"Guessed oxidation states for input structure: {_get_oxi_state_modes(bulk_oxi_states)}")

    if elements is None and bulk_oxi_states is None:
        raise ValueError(  # TODO: Will need to handle this in DefectsGenerator usage
            "Elements for split vacancy generation were not explicitly set; the default is to generate "
            "for cations only, however oxidation states could not be guessed for the input structure. "
            "Please explicitly provide oxidation states using `bulk_oxi_states`."
        )

    mode_oxi_states = _get_oxi_state_modes(bulk_oxi_states)
    cations = {elt for elt, oxi in mode_oxi_states.items() if oxi > 0}

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
                    structure=vac.defect_structure,
                    **vac_gen_kwargs,  # type: ignore
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
            print(f"Guessed oxidation states for input structure: {_get_oxi_state_modes(bulk_oxi_states)}")
        else:
            raise ValueError(
                "Oxidation states could not be guessed for the input structure, please explicitly "
                "provide oxidation states using `bulk_oxi_states`."
            )

    mode_oxi_states = _get_oxi_state_modes(bulk_oxi_states)
    bulk_supercell_w_oxi_states = bulk_supercell.copy()
    bulk_supercell_w_oxi_states.add_oxidation_state_by_element(mode_oxi_states)

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
    bulk_supercell_with_all_interstitial_sites.add_oxidation_state_by_element(mode_oxi_states)

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


def is_periodic_image(
    complex_1: list[PeriodicSite],
    complex_2: list[PeriodicSite],
    dist_tol: float = 0.01,  # TODO wout_charge?
) -> bool:
    r"""
    Determine whether two defect complexes are equal, up to a rigid integer
    lattice vector translation, and within dist_tol. Accepts a list of
    PeriodicSite objects in any order. Does not check for symmetry equivalent
    complexes (i.e. involving any rotation).

    Args:
        complex_1 (list[PeriodicSite]):
            First defect complex, as a list of |PeriodicSite|\\s.
        complex_2 (list[PeriodicSite]):
            Second defect complex, in the same format as ``complex_1``.
        dist_tol (float):
            Distance tolerance (in Å) within which paired point defects are
            taken to coincide.
            (Default: 0.01)

    Returns:
        bool:
            ``True`` if the two complexes are equivalent, else ``False``.
    """
    return (
        _complex_frac_coords_dist(
            [site.species_string for site in complex_1],
            np.asarray([site.frac_coords for site in complex_1]),
            [site.species_string for site in complex_2],
            np.asarray([site.frac_coords for site in complex_2]),
            complex_1[0].lattice,
        )
        <= dist_tol
    )


# maybe should be changed to a consistent metric for large tolerances
# ie check over all periodic images and assignments for min dist
# TODO switch to min over max to match vectorised
def _complex_frac_coords_dist(
    labels_1: Sequence[Hashable],
    frac_coords_1: np.ndarray,
    labels_2: Sequence[Hashable],
    frac_coords_2: np.ndarray,
    lattice: Lattice,
) -> float:
    """
    Metric between (similar) complexes, defined as the maximum distance between
    corresponding sites, after per-label linear assignment between the two
    complexes. Invariant to integer lattice translations of either complex.

    Args:
        labels_1 (Sequence[Hashable]):
            Labels (e.g. species strings) of the point defects in the first
            complex.
        frac_coords_1 (np.ndarray):
            Fractional coordinates of the point defects in the first complex.
        labels_2 (Sequence[Hashable]):
            Labels of the point defects in the second complex.
        frac_coords_2 (np.ndarray):
            Fractional coordinates of the point defects in the second complex.
        lattice (|Lattice|):
            Lattice to which the fractional coordinates correspond.

    Returns:
        float:
            Maximum matched per-site distance (in Å) between the two
            complexes, or ``np.inf`` if their compositions differ.
    """
    # compositions differ - no tolerance
    if Counter(labels_1) != Counter(labels_2):
        return np.inf

    # match centroids
    int_shift = np.round(np.mean(frac_coords_1, axis=0) - np.mean(frac_coords_2, axis=0))

    labels_arr_1 = np.asarray(labels_1)
    labels_arr_2 = np.asarray(labels_2)

    # per label linear assignment, and get max of max distance within each label
    max_dist = 0.0
    for label in set(labels_1):
        cart_1 = lattice.get_cartesian_coords(frac_coords_1[labels_arr_1 == label])
        cart_2 = lattice.get_cartesian_coords(frac_coords_2[labels_arr_2 == label] + int_shift)
        cart_dists = np.linalg.norm(cart_1[:, None, :] - cart_2[None, :, :], axis=-1)
        matches, _ = get_linear_assignment_solution(cart_dists)
        max_dist = max(max_dist, np.max(cart_dists[np.arange(len(matches)), matches]))

    return float(max_dist)


def _complex_frac_coords_dists(
    labels_1: Sequence[Hashable],
    frac_coords_1: np.ndarray,
    labels_2: Sequence[Hashable],
    frac_coords_2: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    """
    Batched form of ``_complex_frac_coords_dist``: the smallest, max per-site
    distance from each of ``k`` complex configurations to each of ``m`` complex
    configurations, over all possible assignments, vectorised.

    Args:
        labels_1 (Sequence[Hashable]):
            Labels (e.g. species strings) of the point defects in the
            first complexes.
        frac_coords_1 (np.ndarray):
            ``(k, n, 3)`` fractional coordinates of the ``k`` first
            complexes (``(n, 3)`` is also accepted, for one).
        labels_2 (Sequence[Hashable]):
            Labels of the point defects in the second complexes, matching
            the ordering of ``frac_coords_2``.
        frac_coords_2 (np.ndarray):
            ``(m, n, 3)`` fractional coordinates of the ``m`` complexes to
            compare against (``(n, 3)`` is also accepted, for one).
        lattice (|Lattice|):
            Lattice to which the fractional coordinates correspond.

    Returns:
        np.ndarray:
            ``(k, m)`` array of maximum matched per-site distances (in Å), or
            ``np.inf`` if the compositions differ. Always 2D.
    """
    # single complex given on either arg: (n,3) -> (1,n,3)
    fcs_1 = np.asarray(frac_coords_1, dtype=float)
    if fcs_1.ndim == 2:
        fcs_1 = fcs_1[None]
    fcs_2 = np.asarray(frac_coords_2, dtype=float)
    if fcs_2.ndim == 2:
        fcs_2 = fcs_2[None]

    # compositions differ - no tolerance
    if Counter(labels_1) != Counter(labels_2):
        return np.full((len(fcs_1), len(fcs_2)), np.inf)

    # integer lattice translation to align centroids
    int_shifts = np.round(  # (k, m, 3)
        fcs_1.mean(axis=1)[:, None, :] - fcs_2.mean(axis=1)[None, :, :]
    )
    shifted = fcs_2[None] + int_shifts[:, :, None, :]  # (k, m, n, 3)

    # evaluate all distances
    cart_disps = (  # (k, m, n, n, 3)
        fcs_1[:, None, :, None, :] - shifted[:, :, None, :, :]
    ) @ lattice.matrix
    dists = np.linalg.norm(cart_disps, axis=-1)  # (k, m, n, n)

    # per label, min over all possible assignments, get max dist
    labels_arr_1, labels_arr_2 = np.asarray(labels_1), np.asarray(labels_2)
    max_dists = np.zeros((len(fcs_1), len(fcs_2)))  # (k, m)
    for label in set(labels_1):
        # get block for this label
        idx_1 = np.nonzero(labels_arr_1 == label)[0]
        idx_2 = np.nonzero(labels_arr_2 == label)[0]
        block = dists[:, :, idx_1[:, None], idx_2[None, :]]  # (k, m, b, b) for num of label b

        # min over all possible assignments, get max
        max_dists = np.maximum(
            max_dists,
            np.min(
                [
                    block[:, :, np.arange(len(idx_1)), perm].max(axis=2)
                    for perm in permutations(range(len(idx_2)))
                ],
                axis=0,
            ),
        )

    return max_dists


def _get_min_dist_between_equiv_complexes(
    complex_1: "DefectComplex | Sequence[PeriodicSite]",
    complex_2: "DefectComplex | Sequence[PeriodicSite]",
    structure: Structure | None = None,
    structure_2: Structure | None = None,
    strip_oxi_states: bool | None = None,
    symprec: float = 0.01,
) -> float:
    """
    Get the minimum distance (in Å) between symmetry-equivalent configurations
    of two defect complexes, according to the ``_complex_frac_coords_dist``
    metric, otherwise analogous to ``get_min_dist_between_equiv_sites``.

    Args:
        complex_1 (|DefectComplex| | Sequence[|PeriodicSite|]):
            First defect complex, as a |DefectComplex| object or a sequence
            of its (unwrapped) constituent point defect sites.
        complex_2 (|DefectComplex| | Sequence[|PeriodicSite|]):
            Second defect complex, similarly.
        structure (|Structure| | None):
            Host structure in which ``complex_1`` is defined. Required if
            ``complex_1`` is not a |DefectComplex| object. Default: None.
        structure_2 (|Structure| | None):
            Separate host |Structure| for ``complex_2``, if the two sites are
            potentially defined in different (but equivalent) host frames --
            e.g. differently-oriented/-defined cells, primitive vs supercell
            definitions, or differently oxi-state-decorated hosts. Each site
            is then folded via its own host into a shared canonical primitive
            cell (from ``get_primitive_structure``) for comparison, returning
            ``np.inf`` if the two hosts do not correspond to matching primitive
            structures. If ``None`` (default), taken from ``complex_2`` if it is a
            |Defect|/|DefectEntry| object, otherwise assumed to match
            ``structure``.
        strip_oxi_states (bool | None):
            Whether to strip oxidation states from the host structure(s)
            before symmetry analysis / host matching. If ``None`` (default),
            oxidation states are only stripped when the two host structures
            (``structure``/``structure_2``) have mismatching oxi-state
            decorations (which can otherwise hinder host matching) -- so
            consistently-decorated hosts retain any decoration-dependent
            symmetry (e.g. inequivalent sites in mixed-valence hosts). Set to
            ``True``/``False`` to always/never strip oxidation states.
        symprec (float):
            Symmetry precision for determining the shared primitive cell and
            its space group operations. Default is 0.01.

    Returns:
        float:
            Minimum distance (in Å) between symmetry-equivalent
            configurations of ``complex_1`` and ``complex_2``, or ``np.inf``
            if their constituent species or host structures do not match.
    """
    # TODO align symmetry kwargs with point version

    # parsing
    def _parse_sites_and_structure(
        cplx: "DefectComplex | Sequence[PeriodicSite]",
        structure: Structure | None,
    ) -> tuple[list[PeriodicSite], Structure | None]:
        if isinstance(cplx, DefectComplex):
            return [defect.site for defect in cplx.defects], cplx.structure
        return list(cplx), structure

    sites_1, structure = _parse_sites_and_structure(complex_1, structure)
    sites_2, structure_2 = _parse_sites_and_structure(complex_2, structure_2)
    if structure is None:
        raise ValueError("``structure`` must be provided if ``complex_1`` is not a ``DefectComplex``.")
    if structure_2 is None:
        structure_2 = structure

    if strip_oxi_states is None:  # default: strip only when mismatching decorations
        strip_oxi_states = {str(sp) for sp in structure.composition} != {
            str(sp) for sp in structure_2.composition
        }  # compare based on species string sets; ``Composition`` equality is oxi-state-insensitive

    if strip_oxi_states:
        structure, structure_2 = structure.copy(), structure_2.copy()
        structure.remove_oxidation_states()
        structure_2.remove_oxidation_states()

    # use oxi-state-insensitive labels if hosts stripped
    labels_1, labels_2 = [
        [
            str(site.species.element_composition) if strip_oxi_states else site.species_string
            for site in cplx_sites
        ]
        for cplx_sites in (sites_1, sites_2)
    ]

    # different constituent point defects -> no tolerance
    if Counter(labels_1) != Counter(labels_2):
        return np.inf

    primitive = get_primitive_structure(structure, symprec=symprec)
    if structure_2 != structure:  # fast-fail for clearly-different host crystals:
        prim_2 = get_primitive_structure(structure_2, symprec=symprec)
        if (
            len(prim_2) != len(primitive)
            or prim_2.composition.reduced_formula != primitive.composition.reduced_formula
        ):
            return np.inf

    try:  # try fold each complex into the shared primitive cell:
        prim_fcs_1, prim_fcs_2 = [
            np.array(
                [
                    site.frac_coords
                    for site in _unwrap_and_transform_to_prim(
                        host,
                        cplx_sites,
                        primitive_structure=primitive,
                        symprec=symprec,
                        image_cells=np.zeros((len(cplx_sites), 3), dtype=int),  # already unwrapped
                    )
                ]
            )
            for cplx_sites, host in ((sites_1, structure), (sites_2, structure_2))
        ]
    except RuntimeError:  # host structures don't match
        return np.inf

    # generate the full orbit of complex_1 (all operations, skip clustering):
    sga, _symprec = get_sga_and_symprec(primitive, symprec)
    quotient_ops = sga.get_symmetry_operations()
    rotations = np.array([op.rotation_matrix for op in quotient_ops])  # (n_ops, 3, 3)
    translations = np.array([op.translation_vector for op in quotient_ops])  # (n_ops, 3)
    images = np.einsum("oij,nj->oni", rotations, prim_fcs_1) + translations[:, None, :]
    images -= np.floor(np.mean(images, axis=1, keepdims=True))  # (n_ops, n_sites, 3)

    return float(
        _complex_frac_coords_dists(labels_2, prim_fcs_2, labels_1, images, primitive.lattice).min()
    )


def cluster_complexes_by_dist_tol(
    complexes: list[list[PeriodicSite]],
    dist_tol: float = 0.01,
    method: LinkageMethod = "single",
    criterion: FclusterCriterion = "distance",
) -> tuple[list[list[PeriodicSite]], list[list[int]]]:
    r"""
    Cluster defect complexes based on their distances, analogous to
    ``cluster_sites_by_dist_tol`` for point defect sites.

    The metric for clustering is maximum distance between corresponding
    sites after linear assignment of sites per label. Invariant to
    translation of any complexes by an integer lattice vector.

    Args:
        complexes (Sequence[list[PeriodicSite]]):
            Complexes to cluster, each as a list of constituent point defect
            sites (|PeriodicSite| objects on a common lattice).
        dist_tol (float):
            Distance tolerance for clustering, in Å (default: 0.01).
        method (str):
            Clustering algorithm to use. (Default: ``"single"``.)
            See  `~doped.utils.symmetry.cluster_coords` for details.
        criterion (str):
            Criterion for flattening clusters. (Default: ``"distance"``.)
            See  `~doped.utils.symmetry.cluster_coords` for details.

    Returns:
        tuple[list[list[PeriodicSite]], list[list[int]]]:
            One representative complex per cluster (the first member),
            and the indices of the input complexes in each cluster, ordered
            as input.
    """
    cluster_indices = _cluster_complex_frac_coords(
        [[site.species_string for site in complex_sites] for complex_sites in complexes],
        [np.asarray([site.frac_coords for site in complex_sites]) for complex_sites in complexes],
        complexes[0][0].lattice,
        dist_tol=dist_tol,
        method=method,
        criterion=criterion,
    )
    return [list(complexes[idx[0]]) for idx in cluster_indices], cluster_indices


def _cluster_complex_frac_coords(
    all_labels: Sequence[Sequence[Hashable]],
    all_fcs: np.ndarray | list[np.ndarray],
    lattice: Lattice,
    dist_tol: float = 0.01,
    method: LinkageMethod = "single",
    criterion: FclusterCriterion = "distance",
) -> list[list[int]]:
    """
    See ``cluster_complexes_by_dist_tol``, but takes frac coords and labels.
    """
    if len(all_fcs) == 1:
        return [[0]]

    # mean constituent displacement <= max constituent displacement so centroids displacement
    # > dist_tol cannot cluster together
    centroids = np.array([np.mean(fcs, axis=0) for fcs in all_fcs])
    diffs = centroids[:, None] - centroids  # match centroids, as in _complex_frac_coords_dist
    no_match = np.linalg.norm((diffs - np.round(diffs)) @ lattice.matrix, axis=-1) > dist_tol
    no_match &= method == "single" and criterion == "distance"

    # make condensed distance array using _complex_frac_coords_dist metric
    condensed_dists = np.array(
        [
            np.inf  # too far apart to cluster
            if no_match[i, j]
            else _complex_frac_coords_dist(all_labels[i], all_fcs[i], all_labels[j], all_fcs[j], lattice)
            for i, j in combinations(range(len(all_fcs)), 2)
        ]
    )

    # linkage doesn't take inf but to avoid clustering different composition
    condensed_dists[~np.isfinite(condensed_dists)] = 1e10

    # cluster
    cn = fcluster(linkage(condensed_dists, method=method), dist_tol, criterion=criterion)

    # group member indices by cluster ordered by first appearance
    return sorted(
        (np.where(cn == n)[0].tolist() for n in set(cn)),
        key=lambda idx: idx[0],
    )


def _transform_complex_fc_to_prim(
    bulk_supercell: Structure,
    unwrapped_fc: Sequence[np.ndarray],
    primitive_structure: Structure,
    symprec: float = 0.01,
    **sm_kwargs,
) -> list[np.ndarray]:
    """
    Transform unwrapped (defect complex) fractional coordinates from the bulk
    supercell frame into the primitive host frame.

    The coordinates are translated such that the complex centroid lands in the
    unit primitive cell; they are `not` individually mapped back to the unit
    cell, so the relative geometry of the complex is preserved. The ordering of
    the input coordinates is unchanged.

    Args:
        bulk_supercell (|Structure|):
            Bulk supercell host structure, in which ``unwrapped_fc`` are
            defined.
        unwrapped_fc (Sequence[np.ndarray]):
            Unwrapped fractional coordinates of the constituent (defect) sites,
            in the ``bulk_supercell`` frame; see
            ``_get_unwrapped_complex_fc``.
        primitive_structure (|Structure|):
            Primitive host structure to transform the coordinates into.
        symprec (float):
            Symmetry precision for the supercell-to-primitive fold map.
            (Default: 0.01)
        **sm_kwargs:
            Additional keyword arguments for the supercell-to-primitive
            transformation (``get_transformation_from_s2_to_s1``, i.e.
            |StructureMatcher| keyword arguments such as ``ltol``, ``stol``,
            ``angle_tol``, ``comparator``).

    Returns:
        list[np.ndarray]:
            The fractional coordinates in the primitive host frame (same
            ordering as ``unwrapped_fc``).
    """
    from doped.utils.configurations import get_transformation_from_s2_to_s1
    from doped.utils.symmetry import _get_supercell_to_prim_fold_map

    # try prim fold map, else fallback to StructureMatcher?
    # TODO see new implementation in get_equiv_frac_coords_in_primitive
    fold_map = _get_supercell_to_prim_fold_map(bulk_supercell, primitive_structure, symprec=symprec)
    if fold_map is not None:
        M, translation = fold_map
        sc_matrix = np.asarray(M)
        offset = np.asarray(translation)
    else:
        sm_kwargs.setdefault("attempt_supercell", True)
        sm_kwargs.setdefault("scale", False)
        transformation = get_transformation_from_s2_to_s1(bulk_supercell, primitive_structure, **sm_kwargs)
        if transformation is None:  # no match to primitive
            raise RuntimeError(
                "Could not map the input structure onto the given primitive cell (no "
                "``StructureMatcher`` match) - they likely do not correspond to the same structure. "
                "Try passing without ``primitive_structure`` argument?"
            )
        sc_matrix, trans_vector, _mapping = transformation
        sc_matrix = np.asarray(sc_matrix)
        offset = -np.asarray(trans_vector) @ sc_matrix

    # shift offset such that the centroid lands in the unit primitive cell:
    offset = offset - np.floor(np.mean(unwrapped_fc, axis=0) @ sc_matrix + offset)

    return [sc_fc @ sc_matrix + offset for sc_fc in unwrapped_fc]


# TODO is this wrapper still useful really
def _unwrap_and_transform_to_prim(
    bulk_supercell: Structure,
    sites: list[PeriodicSite],
    primitive_structure: Structure | None = None,
    symprec: float = 0.01,
    image_cells: list[np.ndarray] | np.ndarray | None = None,
    **sm_kwargs,
) -> list[PeriodicSite]:
    """
    Unwrap a set of (defect complex) sites in a bulk supercell and transform
    them into the primitive host frame.

    The sites are first unwrapped to the unique complex or (with warning)
    to the complex with the smallest span relative to the cell. They are then
    transformed to the primitive basis, such that the centroid lands in the
    unit primitive cell. The ordering of the input sites is unchanged.

    Args:
        bulk_supercell (|Structure|):
            Bulk supercell host structure, in which ``sites`` are defined.
        sites (list[|PeriodicSite|]):
            Constituent (defect) sites to unwrap and transform, defined in
            the ``bulk_supercell`` frame.
        primitive_structure (|Structure| | None):
            Primitive host structure to transform the sites into. If ``None``
            (default), computed from ``bulk_supercell`` using
            ``get_primitive_structure`` (with ``symprec``).
        symprec (float):
            Symmetry precision for computing the primitive structure, if not
            provided. (Default: 0.01)
        image_cells (list[np.ndarray] | np.ndarray | None):
            Provide a list of cells (np.ndarray[int]) for a specified unwrapping,
            to skip automatic unwrapping.
        **sm_kwargs:
            Additional keyword arguments for the supercell-to-primitive
            transformation (``get_transformation_from_s2_to_s1``, i.e.
            |StructureMatcher| keyword arguments such as ``ltol``, ``stol``,
            ``angle_tol``, ``comparator``).

    Returns:
        list[|PeriodicSite|]:
            The unwrapped sites recreated in the primitive host frame (same
            species and ordering as ``sites``).
    """
    if primitive_structure is None:
        primitive_structure = get_primitive_structure(bulk_supercell, symprec=symprec)

    unwrapped_fc = (
        _get_unwrapped_complex_fc(bulk_supercell, sites)
        if image_cells is None
        else [site.frac_coords + image for site, image in zip(sites, image_cells, strict=True)]
    )
    rel_obj_fcs = _transform_complex_fc_to_prim(
        bulk_supercell,
        unwrapped_fc,
        primitive_structure,
        symprec=symprec,
        **sm_kwargs,
    )
    return [  # recreate sites in primitive
        PeriodicSite(site.species, rel_fc, primitive_structure.lattice, coords_are_cartesian=False)
        for site, rel_fc in zip(sites, rel_obj_fcs, strict=True)
    ]


def _get_complex_orbit_in_prim(
    point_defects: list[PeriodicSite],
    primitive: Structure,
    quotient_ops: list[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
) -> tuple[list[list[PeriodicSite]], str]:
    """
    Get the orbit and point group of a defect complex, given a primitive
    structure. The order of sites within each complex is maintained from input
    to output.

    Constituents are matched by species when clustering equal complexes; see
    ``_get_complex_orbit_fcs_in_prim`` to match by arbitrary labels instead.

    The orbit is deterministically sorted (independent of the input
    configuration and constituent ordering), such that ``orbit[0]`` is a consistent
    representative of the orbit, returned if any element of the orbit were input.
    The point group (Schoenflies symbol) is that of the complex (i.e. of its
    stabiliser).

    Args:
        point_defects (list[PeriodicSite]):
            Constituent point defects of the complex, defined in primitive frame.
        primitive (|Structure|):
            Primitive host structure.
        quotient_ops (list[SymmOp] | None):
            Space group symmetry operations, in fractional coordinates,
            to test. If ``None``, determined from ``primitive``.
        symprec (float):
            If ``quotient_ops`` is None, the precision for determining the
            space group operations, which may be dynamically adjusted
            when finding the space group. Else the precision at which the
            ``quotient_ops`` were determined. (Default: 0.01)
        dist_tol_factor (float):
            Factor by which symprec is multiplied to give the distance
            tolerance for clustering equal complexes. (Default: 1.0)

    Returns:
        tuple[list[list[PeriodicSite]], str]:
            The orbit of the complex in the crystal (per primitive lattice
            cell) and its Schoenflies point group symbol. The orbit is
            deterministically sorted, with ``orbit[0]`` the representative
            complex, and all elements given in the unit primitive cell
            (i.e. with complex centroids translated to lie within the unit
            cell).
    """
    point_labels = [site.species_string for site in point_defects]
    orbit_fcs, point_group = _get_complex_orbit_fcs_in_prim(
        np.array([site.frac_coords for site in point_defects]),
        point_labels,
        primitive,
        quotient_ops=quotient_ops,
        symprec=symprec,
        dist_tol_factor=dist_tol_factor,
    )

    # recreate PeriodicSite objects
    orbit = []
    for member_fcs in orbit_fcs:
        member = []
        for site, frac_coords in zip(point_defects, member_fcs, strict=True):
            new_site = deepcopy(site)
            new_site.frac_coords = frac_coords
            member.append(new_site)
        orbit.append(member)

    # sort the orbit deterministically, so orbit[0] is the representative element
    orbit, _sort_index = _sort_complex_orbit(orbit, labels=point_labels)

    return orbit, point_group


def _get_complex_orbit_fcs_in_prim(
    point_fcs: np.ndarray,
    labels: Sequence[Hashable],
    primitive: Structure,
    quotient_ops: list[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
) -> tuple[np.ndarray, str]:
    """
    Get the orbit and point group of a defect complex, given its constituent
    fractional coordinates and a primitive structure; the coordinate-only core
    of ``_get_complex_orbit_in_prim`` (see there for details). Note output is
    not sorted as labels may be arbitrary: use ``_get_complex_orbit_in_prim`` if a
    deterministic order is required.

    Args:
        point_fcs (np.ndarray):
            ``(n_constituents, 3)`` fractional coordinates of the constituent
            point defects, in the primitive frame.
        labels (Sequence[Hashable]):
            Labels by which constituents are matched when clustering equal
            complexes.
        primitive (|Structure|):
            Primitive host structure.
        quotient_ops (list[SymmOp] | None):
            Space group symmetry operations, in fractional coordinates,
            to test. If ``None``, determined from ``primitive``.
        symprec (float):
            Symmetry precision; see ``_get_complex_orbit_in_prim``.
            (Default: 0.01)
        dist_tol_factor (float):
            Factor by which symprec is multiplied to give the distance
            tolerance for clustering equal complexes. (Default: 1.0)

    Returns:
        tuple[np.ndarray, str]:
            The ``(n_orbit, n_constituents, 3)`` fractional coordinates of the
            orbit, with the constituent ordering of the input maintained, and
            the Schoenflies point group symbol.
    """
    # check that input structure is primitive
    if len(get_primitive_structure(primitive, symprec=symprec)) != len(primitive):
        # TODO better way to deal with this?
        raise ValueError(
            "``_get_complex_orbit_in_prim`` requires a primitive host structure, but the provided "
            "structure is not primitive. Try ``get_all_equiv_complexes``, or adjust symmetry tolerances."
        )

    # get space group if not provided
    if quotient_ops is None:
        bulk_sga, symprec = get_sga_and_symprec(primitive, symprec)
        quotient_ops = bulk_sga.get_symmetry_operations()

    dist_tol = dist_tol_factor * symprec
    lattice = primitive.lattice

    # centre the complex (centroid to unit primitive)
    point_labels = list(labels)
    point_fcs = np.asarray(point_fcs, dtype=float)
    point_fcs = point_fcs - np.floor(np.mean(point_fcs, axis=0))

    # apply operations
    rotations = np.array([op.rotation_matrix for op in quotient_ops])  # (n_ops, 3, 3)
    translations = np.array([op.translation_vector for op in quotient_ops])  # (n_ops, 3)
    transformed_fcs = np.einsum("oij,nj->oni", rotations, point_fcs) + translations[:, None, :]
    # (n_ops, n_sites, 3)

    # centre all complexes to unit primitive
    transformed_fcs -= np.floor(np.mean(transformed_fcs, axis=1, keepdims=True))

    # add the original complex back in for ease of identifying stabiliser if quotient_ops
    # don't necessarily come identity first
    all_fcs = np.concatenate([point_fcs[None, :, :], transformed_fcs])  # (n_ops + 1, n_sites, 3)

    # cluster
    cluster_indices = _cluster_complex_frac_coords(
        [point_labels] * len(all_fcs), all_fcs, lattice, dist_tol=dist_tol
    )

    # clusters come by first appearance so first cluster corresponds to original complex
    # and stabiliser is just corresponding ops
    stabiliser = [quotient_ops[i - 1] for i in cluster_indices[0] if i > 0]

    # TODO check group closure?

    # check group theory results (orbit-stabiliser theorem):
    if len(cluster_indices) * len(stabiliser) != len(quotient_ops):
        warnings.warn(
            f"Orbit size ({len(cluster_indices)}) x stabiliser size ({len(stabiliser)}) does not equal "
            f"the number of symmetry operations ({len(quotient_ops)}). Check the symmetry tolerances "
            f"used (symprec = {symprec}, dist_tol = {dist_tol})."
        )

    # get orbit representatives (one configuration per cluster)
    fcs_orbit = np.array([all_fcs[idx[0]] for idx in cluster_indices])

    # TODO align with new local_point_symmetry methods - eg schoenflies from cartesian ops etc
    pg_ops = [np.rint(op.rotation_matrix).astype(int) for op in stabiliser]
    if (pointgroup := get_pointgroup(pg_ops)) is None:
        raise RuntimeError("Could not determine the point group of the defect complex stabiliser.")
    hermann_symbol, _number, _transform = pointgroup
    return fcs_orbit, schoenflies_from_hermann(hermann_symbol.strip())


# TODO public function update docstring
def get_all_equiv_complexes(
    sites: list[PeriodicSite],
    bulk_supercell: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    return_point_group: bool = False,
) -> list[list[PeriodicSite]] | tuple[list[list[PeriodicSite]], str]:
    """
    Get all symmetry-equivalent defect complexes in a cell (supercell or
    primitive).

    The cell is folded to a primitive cell, the orbit is generated by the
    space group operations of the structure and clustering, and the resulting
    configurations are transformed back to the original cell.

    The order of sites within each complex is unchanged. Note the constituent
    sites should be provided unwrapped.

    Args:
        sites (list[|PeriodicSite|]):
            Constituent (defect) sites of the complex, defined in the
            ``bulk_supercell`` frame, and provided unwrapped.
        bulk_supercell (|Structure|):
            Bulk (super)cell host structure, in which ``sites`` are defined.
        symprec (float):
            Symmetry precision for the primitive-cell determination and space
            group operations. (Default: 0.01)
        dist_tol_factor (float):
            Factor by which ``symprec`` is multiplied to give the distance
            tolerance for clustering equal complexes. (Default: 1.0)
        return_point_group (bool):
            If ``True``, also return the Schoenflies point group symbol of the
            complex (as determined in the primitive cell). (Default: False)

    Returns:
        list[list[|PeriodicSite|]] | tuple[list[list[|PeriodicSite|]], str]:
            All symmetry-equivalent complexes in ``bulk_supercell``, each as a
            list of |PeriodicSite| objects (same species and ordering as
            ``sites``). If ``return_point_group`` is ``True``, also returns the
            Schoenflies point group symbol of the complex as a second element.
    """
    from doped.utils.symmetry import _get_orientation_preserving_primitive

    prim_and_matrix = _get_orientation_preserving_primitive(bulk_supercell, symprec=symprec)
    if prim_and_matrix is None:  # already primitive (get_complex_orbit should catch spglib fail)
        orbit, point_group = _get_complex_orbit_in_prim(
            sites, bulk_supercell, symprec=symprec, dist_tol_factor=dist_tol_factor
        )
        return (orbit, point_group) if return_point_group else orbit
    primitive, M = prim_and_matrix

    # fold the complex into the primitive cell and get its orbit
    prim_fcs = _transform_complex_fc_to_prim(
        bulk_supercell, [site.frac_coords for site in sites], primitive, symprec=symprec
    )
    prim_sites = [
        PeriodicSite(site.species, fc, primitive.lattice, coords_are_cartesian=False)
        for site, fc in zip(sites, prim_fcs, strict=True)
    ]
    orbit, point_group = _get_complex_orbit_in_prim(
        prim_sites, primitive, symprec=symprec, dist_tol_factor=dist_tol_factor
    )

    # transform back and add all images
    inv_M = np.linalg.inv(M)
    cosets = lattice_points_in_supercell(M)
    all_complexes = []
    for complex_sites in orbit:
        sc_fcs = np.array([site.frac_coords for site in complex_sites]) @ inv_M
        for coset in cosets:
            shifted_fcs = sc_fcs + coset
            shifted_fcs -= np.floor(np.mean(shifted_fcs, axis=0))  # centroid lands in unit supercell
            all_complexes.append(
                [
                    PeriodicSite(site.species, fc, bulk_supercell.lattice, coords_are_cartesian=False)
                    for site, fc in zip(complex_sites, shifted_fcs, strict=True)
                ]
            )

    # deterministic ordering
    all_complexes, _sort_index = _sort_complex_orbit(all_complexes)

    return (all_complexes, point_group) if return_point_group else all_complexes


def _sort_complex_orbit(
    orbit: list[list[PeriodicSite]],
    prec: int = 5,
    labels: Sequence[Hashable] | None = None,
) -> tuple[list[list[PeriodicSite]], list[int]]:
    r"""
    Deterministically sort the members of a defect complex orbit (used within
    ``_get_complex_orbit_in_prim``). The order of the point defects within each
    complex on input is maintained, and the complex sort order is independent
    of the order of these point defects.

    The sorting order is based on the point defects, but is not the same order as
    _sort_defects, and is intended only to give a deterministic output order of
    symmetry-equivalent complexes.

    Args:
        orbit (list[list[PeriodicSite]]):
            Orbit of a defect complex (i.e. with all members sharing a common
            constituent ordering, and centroids translated to lie within the
            unit cell, and given relative to primitive cell).
        prec (int):
            Number of decimal places to which fractional coordinates are
            rounded for sort keys.
            (Default: 5)
        labels (Sequence[Hashable] | None):
            Labels of the constituent point defects, used in the sort keys.
            If ``None`` (default), the constituent species are used.

    Returns:
        tuple[list[list[PeriodicSite]], list[int]]:
            The sorted orbit, and the indices which sort the input orbit
            members, as ``(sorted_orbit, sort_index)``.
    """
    labels = labels if labels is not None else [site.species_string for site in orbit[0]]

    # sort complexes by key
    sort_index = sorted(
        range(len(orbit)),
        key=lambda k: _complex_config_sort_key(labels, [site.frac_coords for site in orbit[k]], prec=prec),
    )
    sorted_orbit = [orbit[k] for k in sort_index]

    return sorted_orbit, sort_index


def _complex_config_sort_key(
    labels: Sequence[Hashable],
    frac_coords: Sequence | np.ndarray,
    prec: int = 5,
) -> tuple:
    """
    Return a deterministic sort key for a defect complex configuration, based
    on labels and frac coords. Orders configurations up to integer lattice
    translation, but no other symmetry operations.

    This is keyed on largely non-invariant properties of the complex, as it is
    meant largely for sorting symmetry-equivalent configurations of a single
    complex, unlike ``_defect_sort_key``, for sorting inequivalent complexes.

    Args:
        labels (Sequence[Hashable]):
            Labels (e.g. species strings) of the constituent point defects.
        frac_coords (Sequence | np.ndarray):
            ``(n, 3)`` array of (unwrapped) fractional coordinates of the
            constituent point defects.
        prec (int):
            Number of decimal places to which the centroid-relative
            fractional coordinates are rounded.
            (Default: 5)

    Returns:
        tuple: Sort key for the defect complex configuration.
    """
    frac_coords = np.asarray(frac_coords)
    centroid = np.mean(frac_coords, axis=0)
    rel_fcs = np.round(frac_coords - centroid, prec)
    # round before wrapping to unit cell for float noise
    centroid = np.mod(np.round(centroid, prec), 1.0)
    return (
        _frac_coords_sort_func(centroid),
        tuple(sorted(zip(labels, map(tuple, rel_fcs), strict=True))),
    )


def _complex_config_key(
    labels: Sequence[Hashable],
    frac_coords: Sequence | np.ndarray,
    prec: int = 5,
) -> tuple:
    """
    Return a unique hashable key identifying a defect complex configuration,
    for matching identical configurations, invariant to integer lattice
    translation and to the constituent ordering, but no other symmetry
    operations.

    Used for fast exact-match path for checking equal configurations.

    Args:
        labels (Sequence[Hashable]):
            Labels (e.g. species strings) of the constituent point defects.
        frac_coords (Sequence | np.ndarray):
            ``(n, 3)`` array of (unwrapped) fractional coordinates of the
            constituent point defects.
        prec (int):
            Number of decimal places to which the centroid and
            centroid-relative fractional coordinates are rounded (should be
            finer than the fractional equivalent of the matching distance tolerance). (Default: 5)

    Returns:
        tuple: Hashable key for the defect complex configuration.
    """
    frac_coords = np.asarray(frac_coords)
    centroid = np.mean(frac_coords, axis=0)
    rel_fcs = np.round(frac_coords - centroid, prec)
    # round before wrapping to the unit cell for float noise, then round again
    # for exact matching as mod can reintroduce it
    centroid = np.round(np.mod(np.round(centroid, prec), 1.0), prec)
    return (
        tuple(centroid),
        tuple(sorted(zip(labels, map(tuple, rel_fcs), strict=True))),
    )


def _reduce_equivalent_complexes(
    complexes: Iterable[tuple[Sequence[Hashable], np.ndarray]],
    primitive: Structure,
    quotient_ops: list[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
) -> list[tuple[int, np.ndarray, str]]:
    """
    Reduce a set of candidate defect complexes to only symmetry inequivalent
    sets of orbits.

    Args:
        complexes (Iterable[tuple[Sequence[Hashable], np.ndarray]]):
            Candidate complexes, as ``(labels, frac_coords)`` tuples.
        primitive (|Structure|):
            Primitive host structure.
        quotient_ops (list[SymmOp] | None):
            Space group symmetry operations, in fractional coordinates. If
            ``None``, these are redetermined for each generated orbit, so
            can be provided here. (Default: None)
        symprec (float):
            Symmetry precision for the space group operations. (Default: 0.01)
        dist_tol_factor (float):
            Factor by which ``symprec`` is multiplied to give the distance
            tolerance for matching equal complexes. (Default: 1.0)

    Returns:
        list[tuple[int, np.ndarray, str]]:
            For each symmetry-inequivalent complex: the index of the first
            candidate which generated it; its orbit ``(n_orbit,
            n_constituents, 3)`` array of fractional coordinates, with
            constituent ordering matching the input; the Schoenflies
            point group symbol.
    """
    lattice = primitive.lattice
    dist_tol = dist_tol_factor * symprec

    # get fractional rounding tolerance
    max_frac_deviation = (
        dist_tol * np.linalg.norm(lattice.reciprocal_lattice_crystallographic.matrix, axis=1).max()
    )  # largest fractional coordinate shift allowed by dist_tol: with r=fL, f_i=L*_i.r,
    # |df_i| <= |L*_i|dr, +3dp excess precision for exact match to be safe
    key_prec = max(int(np.floor(-np.log10(max(max_frac_deviation, 1e-12)))) + 3, 3)

    # initialise recording sets
    seen: set[tuple] = set()  # keys of all recorded orbit elements for all seen complexes
    by_composition: dict[tuple, list[np.ndarray]] = {}  # complexes by composition for matching
    inequivalent = []  # for output, see Returns

    for idx, (labels, complex_fcs) in enumerate(complexes):
        # check for exact match in seen to skip more expensive distance check
        frac_coords = np.asarray(complex_fcs)
        if _complex_config_key(labels, frac_coords, prec=key_prec) in seen:
            continue

        # no exact match -> actual distance check
        order = np.argsort(np.asarray(labels, dtype=object))
        composition = tuple(np.asarray(labels, dtype=object)[order])
        members = by_composition.get(composition)  # only check against same composition
        if (
            members is not None
            and (
                _complex_frac_coords_dists(
                    composition, frac_coords[order], composition, np.asarray(members), lattice
                )
                <= dist_tol
            ).any()
        ):
            continue  # already seen

        # not seen -> generate orbit
        orbit_fcs, point_group = _get_complex_orbit_fcs_in_prim(
            frac_coords,
            labels,
            primitive,
            quotient_ops=quotient_ops,
            symprec=symprec,
            dist_tol_factor=dist_tol_factor,
        )

        # mark whole orbit as seen, and record composition
        recorded = by_composition.setdefault(composition, [])
        for member_fcs in orbit_fcs:
            seen.add(_complex_config_key(labels, member_fcs, prec=key_prec))
            recorded.append(np.asarray(member_fcs)[order])

        # return inequivalent complexes generated
        inequivalent.append((idx, orbit_fcs, point_group))
    return inequivalent


def get_defect_complex_name(
    defect_complex: DefectComplex,
    names: Sequence[str] | None = None,
    element_list: list[str] | None = None,
    prec: int = 2,
    include_point_group: bool = True,
    include_separations: bool = True,
) -> str:
    r"""
    Get an intrinsic name for a defect complex, dependent only on a set of
    symmetry invariant properties. The ordering of the point defects is chosen
    such that the largest distance between successive point defects in the
    chain is minimised, in order to give the most visualisable representation
    of the complex.

    The name is then constructed from this chain, with shortest separation
    first, with the distances between successive point defects interleaved,
    and the point group appended, e.g. ``v_Cd-1.87-Cd_i-1.87-v_Cd-C3v``.

    This name is not necessarily unique, and may be shared by distinct
    inequivalent complexes. The name is invariant under permutation of the
    constituents, the choice of symmetry-equivalent configuration of the complex,
    and lattice translation. Tie-breaking for ordering the constituent point defects
    is by ``_defect_sort_key`` then by fractional coordinates.

    All orderings of the constituents are enumerated, which is factorial in
    the number of constituents, but negligible at realistic complex sizes
    relative to generating such a complex.

    Args:
        defect_complex (DefectComplex):
            The defect complex to name.
        names (Sequence[str]):
            Names to use for the constituent point defects, ordered as
            ``defect_complex.defects``; e.g. the extended point defect names
            from |DefectsGenerator| (``Cd_i_C3v`` etc), rather than the bare
            ``Defect.name`` (``Cd_i``). Does not affect the ordering. If
            ``None`` (default), ``Defect.name`` is used.
        element_list (list[str]):
            Sorted list of elements in the host structure, used for
            deterministic tie-breaking between constituents (as in
            ``name_defect_entries``). If ``None`` (default), determined from
            the constituent point defects.
        prec (int):
            Number of decimal places to which separations are rounded, both
            in the name and when identifying ties. (Default: 2)
        include_point_group (bool):
            Whether to include the point group of the complex in the name.
            (Default: True)
        include_separations (bool):
            Whether to include the separations between successive
            constituents in the name. Setting this and ``include_point_group``
            to ``False`` gives just the constituent names (e.g.
            ``v_Cd-Cd_i-v_Cd``), which shows the composition of the complex at
            a glance. (Default: True)

    Returns:
        str: The name of the defect complex (without charge state).
    """
    from doped.generation import _defect_sort_key, _get_element_list

    if element_list is None:
        element_list = _get_element_list(defect_complex.defects)
    if names is None:
        names = [defect.name for defect in defect_complex.defects]
    lattice = cast("Structure", defect_complex.structure).lattice
    frac_coords = np.array([defect.site.frac_coords for defect in defect_complex.defects])
    cart_coords = lattice.get_cartesian_coords(frac_coords)
    seps = np.round(np.linalg.norm(cart_coords[:, None, :] - cart_coords[None, :, :], axis=-1), prec)
    sort_keys = [  # as for point defects, including position (which orders repeated constituents)
        _defect_sort_key(defect, cast("list[str]", element_list), include_frac_coords=True)
        for defect in defect_complex.defects
    ]

    def _successive_seps(order: Sequence[int]) -> list[float]:
        return [seps[i, j] for i, j in pairwise(order)]

    def _ordering_key(order: Sequence[int]) -> tuple:
        successive_seps = _successive_seps(order)
        return (
            tuple(sorted(successive_seps, reverse=True)),  # min over max separations
            tuple(successive_seps),  # shortest first
            tuple(sort_keys[idx] for idx in order),  # tie break on constituent sort keys
        )

    order = min(permutations(range(len(cart_coords))), key=_ordering_key)
    parts = [names[order[0]]]  # constituents and their separations, interleaved
    for sep, idx in zip(_successive_seps(order), order[1:], strict=True):
        parts += ([f"{sep:.{prec}f}"] if include_separations else []) + [names[idx]]

    point_group = defect_complex.point_group if include_point_group else None
    return "-".join(parts) + (f"-{point_group}" if point_group else "")


@contextlib.contextmanager
def _complex_generation_pbar(total: int, pbar: tqdm | bool = True) -> Iterator[tqdm]:
    """
    Yield a progress bar for defect complex generation, over ``total`` steps
    (i.e. successive increases in complex size).

    Args:
        total (int):
            Number of steps (increases in complex size) to be taken.
        pbar (tqdm | bool):
            An existing progress bar to report into -- allowing several
            generation calls to share one bar, in which case ``total`` is
            added to its own total, and it is neither reset nor closed here --
            or ``True``/``False`` to create a new bar which is shown/hidden.
            (Default: True)

    Yields:
        tqdm: The progress bar to update.
    """
    if isinstance(pbar, tqdm):
        pbar.total = (pbar.total or 0) + total
        pbar.refresh()
        yield pbar
        return

    with tqdm(  # matching the |DefectsGenerator| initialisation progress bar:
        total=total,
        disable=not pbar,
        bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]",
    ) as new_pbar:
        yield new_pbar


def get_complex_chains(
    defects: dict[str, Defect],
    chain: Sequence[Iterable[str]],
    max_separation: float = 2.5,
    min_separation: float = 0.0,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    pbar: tqdm | bool = True,
) -> list[tuple[DefectComplex, list[str]]]:
    r"""
    Generate a set of symmetry-inequivalent defect complexes as ordered chains
    of point defects. The set of possible point defects at each position in the
    chain can be specified, as well as the maximum and minimum separation
    between consecutive point defects in the chain.

    ``chain`` sets both the length of the chains and the point defects allowed
    at each position along them, so e.g. ``[["v_Cd"], ["Cd_i"], ["v_Cd"]]``
    gives split vacancies (vacancy-interstitial-vacancy complexes), or
    ``[["v_Cd", "v_Te"], ["v_Cd", "v_Te"]]`` gives all possible divacancies.

    For an alternative method of defect complex generation, see
    ``get_complex_clusters``.

    Args:
        defects (dict[str, Defect]):
            The point defects available as chain constituents, as
            ``{defect name: Defect}`` (sharing a common primitive host
            structure), and against which the names in ``chain`` are matched.
        chain (Sequence[Iterable[str]]):
            The names of the point defects allowed at each position along
            the chain, as one iterable per position: this sets both the
            chain length and the allowed constituents.
        max_separation (float):
            Maximum distance (in Å) between consecutive constituent point
            defects in a chain. (Default: 2.5)
        min_separation (float):
            Minimum distance (in Å) between consecutive constituent point
            defects in a chain. (Default: 0.0)
        symprec (float):
            Symmetry precision for matching symmetry-equivalent chains.
            (Default: 0.01)
        dist_tol_factor (float):
            Factor by which ``symprec`` is multiplied to give the distance
            tolerance for matching equal complexes. (Default: 1.0)
        pbar (tqdm | bool):
            An existing progress bar to report into (allowing several
            generation calls to share one bar), or ``True``/``False`` to
            create a new bar which is shown/hidden. One step is taken per
            position added to the chains. (Default: True)

    Returns:
        list[tuple[DefectComplex, list[str]]]:
            The symmetry-inequivalent defect complexes, each with the names
            (from ``defects``) of its constituent point defects, ordered as
            ``DefectComplex.defects``.
    """
    # TODO worth adding per link max dist? trivial along with min_separation check
    # also could make min_separation check over all previous constituents ie can't fold back?

    chain = [list(position) for position in chain]
    names = list(dict.fromkeys(name for position in chain for name in position))  # used defects only
    pool = [defects[name] for name in names]

    # list[list[int]] ie list of defect indices for each position in chain
    positions = [[names.index(name) for name in position] for position in chain]
    primitive = pool[0].structure

    frac_coords, site_defects, site_neighbours = _get_candidate_site_graph(
        pool, max_separation, dist_tol=symprec * dist_tol_factor
    )
    sga, symprec = get_sga_and_symprec(primitive, symprec)
    quotient_ops = sga.get_symmetry_operations()
    seed_sites = {
        defect: index for index, defects_here in enumerate(site_defects) for defect in defects_here
    }  # {int: int}, only one site (ie index) kept per defect

    # each partial chain is (unwrapped) constituent frac coords, constituent point
    # defect indices, and site index and image of each constituent:
    partials = [
        (frac_coords[[seed_sites[defect]]], [defect], [seed_sites[defect]], [np.zeros(3, int)])
        for defect in positions[0]
    ]

    # loop over position in chain
    with _complex_generation_pbar(len(positions) - 1, pbar) as chain_pbar:
        for position, position_defects in enumerate(positions[1:], start=1):
            final = position == len(positions) - 1
            extended = []
            chain_pbar.set_description(  # the number of partials is what sets the cost of this step
                f"Extending {len(partials)} chain(s) to length {position + 1}"
            )

            # loop over partial chains of length position
            for coords, chain_defects, sites, images in partials:
                # loop over neighbours of head of chain
                for site, rel_image, dist in zip(*site_neighbours[sites[-1]], strict=True):
                    image = images[-1] + rel_image  # neighbour images are relative to the head
                    if dist < min_separation or any(  # too close or already in chain
                        s == site and np.array_equal(i, image) for s, i in zip(sites, images, strict=True)
                    ):
                        continue

                    for defect in position_defects:
                        if defect not in site_defects[site]:
                            continue
                        new_coords = np.vstack([coords, frac_coords[site] + image])
                        extended.append(
                            (new_coords, [*chain_defects, defect], [*sites, site], [*images, image])
                        )

            # generate candidate set for reduction: note the head of the chain is labelled
            # so we don't lose it by reduction to an otherwise symmetry equivalent chain
            candidates = [
                ([*chain_defects[:-1], chain_defects[-1] + len(pool) * (not final)], coords)
                for coords, chain_defects, _sites, _images in extended
            ]

            # reduce to symmetry inequivalent chains
            inequiv_parts = _reduce_equivalent_complexes(
                candidates, primitive, quotient_ops, symprec=symprec, dist_tol_factor=dist_tol_factor
            )
            chain_pbar.update(1)
            if final:  # build the DefectComplex objects, with the names of their constituents
                complexes = []
                for idx, orbit, pg in inequiv_parts:
                    indices = extended[idx][1]  # constituent indices into ``pool``
                    complexes.append(
                        _complex_from_orbit_coords(
                            [pool[i] for i in indices], [names[i] for i in indices], orbit, pg
                        )
                    )
                return complexes
            partials = [extended[idx] for idx, _orbit, _pg in inequiv_parts]

    return []  # chain length 1 no complexes...? or should return point defects anyway


def get_complex_clusters(
    defects: dict[str, Defect],
    names: Sequence[str | Sequence[str]] | None = None,
    size: int | tuple[int, int] = 2,
    max_diameter: float = 3.0,
    min_separation: float = 0.0,
    allow_repeats: bool = True,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    pbar: tqdm | bool = True,
) -> list[tuple[DefectComplex, list[str]]]:
    r"""
    Generate a set of symmetry-inequivalent defect complexes as unordered
    clusters of point defects. The set of possible point defects, the maximum
    and minimum size of the cluster, and the maximum diameter of the complex
    can be specified.

    Constituents are drawn from ``defects`` in any combination, so e.g.
    ``[v_Cd, v_Te]`` with ``size = 2`` gives all divacancies, and with
    ``size = (2, 3)`` all di- and tri-vacancies. With ``allow_repeats = False``,
    complexes with an exact composition can be specified, e.g.
    ``[v_Cd, v_Cd, Cd_i, Cd_Te]`` with ``size = 4`` will only give clusters
    with this exact composition of point defects.

    For an alternative method of defect complex generation, see
    ``get_complex_chains``.

    Args:
        defects (dict[str, Defect]):
            The point defects available as cluster constituents (sharing a
            common primitive host structure), as ``{name: Defect}``.
        names (Sequence[str | Sequence[str]]):
            Names of the point defects (from ``defects``) to use as cluster
            constituents. A name may be listed more than once, giving the
            number of times that point defect can appear in a cluster when
            ``allow_repeats`` is ``False`` (e.g. ``["v_Cd", "v_Cd", "Cd_i"]``
            with ``size=3, allow_repeats=False`` gives only
            ``v_Cd+v_Cd+Cd_i`` type complexes). An entry may itself be a
            group of names, which then count as one constituent between
            them; e.g. ``["v_Cd", ["Cd_i", "Te_i"]]`` to permit a single Cd interstitial of any
            type in the cluster. Note groups are assumed disjoint or identical,
            partially overlapping groups are not supported.
        size (int | tuple[int, int]):
            Number of constituent point defects in the generated clusters,
            or a ``(min, max)`` range. (Default: 2)
        max_diameter (float):
            Maximum distance (in Å) between any two constituent point
            defects in a cluster. (Default: 3.0)
        min_separation (float):
            Minimum distance (in Å) between `any` two constituent point
            defects in a cluster, e.g. to scan separation shells by
            excluding clusters already generated with a smaller
            ``max_diameter``. (Default: 0.0)
        allow_repeats (bool):
            Whether a point defect can appear in a cluster more times than
            it is listed in ``defects`` (e.g. ``v_Cd + v_Cd`` from a singly-
            listed ``v_Cd``). Distinct sites are required either way, so a
            repeated point defect must occupy a different site (or periodic
            image) of its orbit. (Default: True)
        symprec (float):
            Symmetry precision for matching symmetry-equivalent clusters.
            (Default: 0.01)
        dist_tol_factor (float):
            Factor by which ``symprec`` is multiplied to give the distance
            tolerance for matching equal complexes. (Default: 1.0)
        pbar (tqdm | bool):
            An existing progress bar to report into (allowing several
            generation calls to share one bar), or ``True``/``False`` to
            create a new bar which is shown/hidden. One step is taken per
            constituent added to the clusters. (Default: True)

    Returns:
        list[tuple[DefectComplex, list[str]]]:
            The symmetry-inequivalent defect complexes, each with the names
            (from ``defects``) of its constituent point defects, ordered as
            ``DefectComplex.defects``.
    """
    min_size, max_size = (size, size) if isinstance(size, int) else size
    if names is None:
        names = list(defects)  # each defect available once
    groups = [[name] if isinstance(name, str) else list(name) for name in names]
    pool_names = list(dict.fromkeys(n for group in groups for n in group))  # distinct point defects
    pool = [defects[name] for name in pool_names]

    # count available number of each group of defects: because we allow for eg [v_Cd,v_Cd,Cd_i],
    # allow_repeats=False,size=3 to give v_Cd+v_cd+Cd_i type complexes only.
    # grouped names share a single count between them
    counts = Counter(frozenset(pool_names.index(name) for name in group) for group in groups)
    constraints = defaultdict(list)  # {pool index: [(group, count), ...]} it could violate
    for group, count in counts.items():
        for idx in group:
            constraints[idx].append((group, count))
    primitive = pool[0].structure

    frac_coords, site_defects, site_neighbours = _get_candidate_site_graph(
        pool, max_diameter, dist_tol=symprec * dist_tol_factor
    )
    sga, symprec = get_sga_and_symprec(primitive, symprec)
    quotient_ops = sga.get_symmetry_operations()
    seed_sites = {
        defect: index for index, defects_here in enumerate(site_defects) for defect in defects_here
    }  # {int: int}, only one site (ie index) kept per defect

    # each partial chain is (unwrapped) constituent frac coords, constituent point
    # defect indices, and site index and image of each constituent:
    partials = [
        (frac_coords[[seed_sites[defect]]], [defect], [seed_sites[defect]], [np.zeros(3, int)])
        for defect in range(len(pool))
    ]
    complexes: list[tuple[DefectComplex, list[str]]] = []

    with _complex_generation_pbar(max_size - 1, pbar) as cluster_pbar:
        for cluster_size in range(2, max_size + 1):
            extended = []
            cluster_pbar.set_description(  # the number of partials is what sets the cost of this step
                f"Extending {len(partials)} cluster(s) to size {cluster_size}"
            )

            for coords, cluster_defects, sites, images in partials:
                # get neighbours of any constituent
                for site, rel_image, _dist in zip(*site_neighbours[sites[-1]], strict=True):
                    image = images[-1] + rel_image  # neighbour images are relative to the last-added
                    if any(
                        s == site and np.array_equal(i, image) for s, i in zip(sites, images, strict=True)
                    ):
                        continue  # already in the cluster

                    # check satisfies max_diameter and min_separation
                    new_fcs = frac_coords[site] + image
                    dists = np.linalg.norm((coords - new_fcs) @ primitive.lattice.matrix, axis=1)
                    if dists.max() > max_diameter or dists.min() < min_separation:
                        continue

                    new_coords = np.vstack([coords, new_fcs])
                    for defect in site_defects[site]:
                        if not allow_repeats and any(
                            sum(d in group for d in cluster_defects) >= count
                            for group, count in constraints[defect]
                        ):
                            continue  # unacceptable composition

                        extended.append(
                            (new_coords, [*cluster_defects, defect], [*sites, site], [*images, image])
                        )

            # reduce symmetry equivalent complexes
            inequiv_parts = _reduce_equivalent_complexes(
                [(cluster_defects, coords) for coords, cluster_defects, _sites, _images in extended],
                primitive,
                quotient_ops,
                symprec=symprec,
                dist_tol_factor=dist_tol_factor,
            )
            if cluster_size >= min_size:
                for idx, orbit, pg in inequiv_parts:
                    indices = extended[idx][1]  # constituent indices into ``pool``
                    complexes.append(
                        _complex_from_orbit_coords(
                            [pool[i] for i in indices], [pool_names[i] for i in indices], orbit, pg
                        )
                    )
            partials = [extended[idx] for idx, _orbit, _pg in inequiv_parts]
            cluster_pbar.update(1)

    return complexes


def _complex_from_orbit_coords(
    defects: Sequence[Defect],
    names: Sequence[str],
    orbit: np.ndarray,
    point_group: str | None = None,
    **doped_kwargs,
) -> tuple[DefectComplex, list[str]]:
    """
    Create a |DefectComplex| from its constituent point defects and the full
    orbit coordinates of its symmetry-equivalent configurations. This generates
    a |DefectComplex| with a standard, deterministic ordering of its defects
    and orbit, matching the ordering of an object generated from parsing.

    Args:
        defects (Sequence[Defect]):
            Constituent point defects, ordered as the constituents of
            ``orbit``. Only their species and defect types are used; their
            sites are taken from ``orbit``.
        names (Sequence[str]):
            Names of the constituent point defects, in the same order as
            ``defects``. Returned reordered to match ``DefectComplex.defects``
            - just for convenience. TODO make optional?
        orbit (np.ndarray):
            ``(n_orbit, n_constituents, 3)`` array of fractional coordinates
            of the symmetry-equivalent configurations of the complex. The
            members (axis 0) may be given in any order, but their
            constituents (axis 1) must match the order of ``defects``.
        point_group (str):
            Schoenflies point group symbol of the complex.
        **doped_kwargs:
            Additional keyword arguments for the |DefectComplex| initialisation.

    Returns:
        tuple[DefectComplex, list[str]]:
            The defect complex, and ``names`` reordered to match
            ``DefectComplex.defects`` (which is canonically sorted).
    """
    # sort before constructing Defect objects such that Defect objects correspond to
    # equivalent_complexes[0]
    defect_names = [defect.name for defect in defects]
    orbit = orbit[
        sorted(range(len(orbit)), key=lambda idx: _complex_config_sort_key(defect_names, orbit[idx]))
    ]

    # Defect object construction
    lattice = defects[0].structure.lattice
    constituents = []
    for defect, frac_coords in zip(defects, orbit[0], strict=True):
        constituent = deepcopy(defect)
        constituent.site = PeriodicSite(defect.site.species, frac_coords, lattice)
        constituents.append(constituent)

    # orbit PeriodicSite construction
    sites_orbit = [
        [
            PeriodicSite(constituent.site.species, frac_coords, lattice)
            for constituent, frac_coords in zip(constituents, member, strict=True)
        ]
        for member in orbit
    ]
    # construct DefectComplex
    complex_defect, sort_index = _sorted_defect_complex(
        constituents, sites_orbit, point_group=point_group, **doped_kwargs
    )
    return complex_defect, [names[i] for i in sort_index]


def _sorted_defect_complex(
    defects: list[Defect],
    orbit: Sequence[Sequence[PeriodicSite]],
    point_group: str | None = None,
    **doped_kwargs,
) -> tuple[DefectComplex, list[int]]:
    """
    Create a |DefectComplex| from its constituent point defects and the orbit
    of its symmetry-equivalent configurations, with both the constituent
    defects and the orbit sorted to a standard, deterministic order. The same
    |DefectComplex| should be obtained for any permutation of the orbit and
    defects, other than the stored |Defect| objects from input, whose sites are
    left unchanged (up to an integer lattice translation to centre the
    centroid), i.e. their sites are the input Defect.site, and don't
    necessarily correspond to equivalent_complexes[0].

    Args:
        defects (list[Defect]):
            Constituent point defects of the complex, with their sites set,
            in any order.
        orbit (Sequence[Sequence[|PeriodicSite|]]):
            The symmetry-equivalent configurations of the complex, in any
            order, with the constituents of each ordered as ``defects``.
        point_group (str | None):
            Schoenflies point group symbol of the complex.
        **doped_kwargs:
            Additional keyword arguments for the |DefectComplex| initialisation.

    Returns:
        tuple[DefectComplex, list[int]]:
            The defect complex, and the indices which sort the input
            constituents (for reordering any accompanying per-constituent
            information).
    """
    # TODO currently the constituent ordering within each orbit member is only deterministic up to
    # permutations of constituents whose defect sort keys collide. worth checking if this could ever
    # be problematic - in which case a full 2D sort index would be needed over constituents within
    # each orbit element. but then note this could mess up that each orbit element is an exact per
    # constituent image under a certain symmetry operation?
    from doped.generation import _defect_sort_key, _get_element_list  # avoids circular import

    # sort point defects deterministically (matching ``_sort_defects``, then by prim
    # site coords instead of conv cell for same name)
    element_list = _get_element_list(defects)
    sort_index = sorted(
        range(len(defects)),
        key=lambda i: (
            _defect_sort_key(defects[i], element_list),
            tuple(np.round(defects[i].site.frac_coords, 5)),
        ),
    )
    sorted_defects = [defects[i] for i in sort_index]
    sorted_orbit = [[member[i] for i in sort_index] for member in orbit]

    # then sort the orbit members: _complex_config_sort_key is invariant to the constituent ordering, so
    # this is independent of both input orderings (may already be sorted, e.g. as currently the
    # case in both generation and analysis, but keeps this ordering deterministic)
    names = [defect.name for defect in sorted_defects]
    sorted_orbit.sort(
        key=lambda member: _complex_config_sort_key(names, [site.frac_coords for site in member])
    )

    return (
        DefectComplex(
            sorted_defects,
            equivalent_complexes=sorted_orbit,
            point_group=point_group,
            **doped_kwargs,
        ),
        sort_index,
    )


def _get_candidate_site_graph(
    defects: Sequence[Defect], max_dist: float, dist_tol: float = 0.01
) -> tuple[np.ndarray, list[list[int]], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    For a set of defects in a given structure, generate the set of possible
    sites, the defects which can occupy each sites, and the other defect sites
    within a certain distance of each site. Coincident sites are merged, and
    all neighbours are included (i.e. all distinct periodic images, excluding
    the site itself, but including its own periodic images).

    Args:
        defects (Sequence[Defect]):
            Point defects to take the candidate sites of, sharing a common
            (primitive) host structure.
        max_dist (float):
            Maximum neighbour distance (in Å).
        dist_tol (float):
            Distance tolerance (in Å) just used for matching sites. (Default: 0.01)

    Returns:
        tuple[np.ndarray, list[list[int]], list[tuple[np.ndarray, ...]]]:
            The ``(n_sites, 3)`` fractional coordinates of the candidate
            defect sites; the indices in ``defects`` of the point defects which
            can occupy each site; the neighbours of each as
            ``(index, image, distance)`` arrays ordered by increasing distance.
    """
    # get all equivalent defect sites
    lattice = defects[0].structure.lattice
    for defect in defects:
        if not defect.equivalent_sites:
            defect.get_multiplicity()  # populate equivalent_sites
    site_defects = [index for index, defect in enumerate(defects) for _ in defect.equivalent_sites]
    frac_coords = np.array([site.frac_coords for defect in defects for site in defect.equivalent_sites])

    # group coincident sites, get defects corresponding to each site
    clusters = cluster_coords(frac_coords, lattice, dist_tol=dist_tol)
    sites_and_defects = [
        (frac_coords[idxs[0]], [site_defects[i] for i in idxs])
        for idxs in (np.where(clusters == cluster)[0] for cluster in set(clusters))
    ]
    sites_and_defects.sort(key=lambda site_and_defects: _frac_coords_sort_func(site_and_defects[0]))
    site_fcs, defects_per_site = (list(i) for i in zip(*sites_and_defects, strict=True))

    # construct structure for convenient get_neighbor_list method
    sites_struct = Structure(lattice, ["X"] * len(site_fcs), site_fcs)
    centres, neighbours, images, dists = sites_struct.get_neighbor_list(r=max_dist, exclude_self=True)

    # sort with nearest neighbours first
    order = np.argsort(dists)
    centres, neighbours, dists = centres[order], neighbours[order], dists[order]
    images = np.rint(images[order]).astype(int)  # integer cells
    site_neighbours = [
        (neighbours[matches], images[matches], dists[matches])
        for matches in (centres == index for index in range(len(site_fcs)))
    ]

    # TODO legibility
    return np.array(site_fcs), defects_per_site, site_neighbours
