"""
Code for generating and analysing defect complexes.
"""

from collections.abc import Iterable
from copy import deepcopy

import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.structure import PeriodicSite, Structure

from doped.core import remove_site_oxi_state
from doped.utils.parsing import (
    _get_species_from_composition_diff,
    get_defect_type_and_composition_diff,
    get_site_mapping_indices,
)
from doped.utils.supercells import min_dist


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
    structure is an interstitial. This corresponds to the 2 V_X + X_i definition
    of split vacancies discussed in https://doi.org/10.48550/arXiv.2412.19330

    A simple vacancy corresponds to cases where 1 site from the bulk structure
    cannot be matched to the defect structure while all defect structure sites can
    be matched to bulk sites, and 'non-trivial' vacancies refer to all other cases.

    Inspired by the vacancy geometry classification used in Kumagai et al.
    _Phys Rev Mater_ 2021. See https://doi.org/10.48550/arXiv.2412.19330 for further
    details.

    Args:
        vacancy_supercell (Structure):
            The defect supercell containing the vacancy to be classified.
        bulk_supercell (Structure):
            The bulk supercell structure to compare against for site-matching.
        tol (float):
            The (fractional) tolerance for matching sites between the defect and bulk
            structures. If ``abs_tol`` is ``False`` (default), then this value multiplied
            by the shortest bond length in the bulk structure will be used as the distance
            threshold for matching, otherwise the value is used directly (as a length in Å).
            Default is 0.5.
        abs_tol (bool):
            Whether to use ``tol`` as an absolute distance tolerance (in Å) instead of a
            fractional tolerance (in terms of the shortest bond length in the structure).
            Default is ``False``.
        verbose (bool):
            Whether to print additional information about the classification for non-trivial
            vacancies.
            Default is ``False``.
    """
    def_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, vacancy_supercell)
    old_species = _get_species_from_composition_diff(comp_diff, -1)
    bulk_bond_length = max(min_dist(bulk_supercell), 1)
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
        > bulk_bond_length * tol
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
        > bulk_bond_length * tol
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
            If ``None`` (default), the oxidation states of the structure
            are used. An error will be raised if the oxidation states are
            not set and are not provided.

    Returns:
        float: The electrostatic energy of the structure.
    """
    if oxi_states is not None:
        structure.add_oxidation_state_by_element(oxi_states)
    structure._charge = structure.charge  # requires oxidation states to be set
    return EwaldSummation(structure).total_energy


def generate_complex_from_defect_sites(
    bulk_supercell: Structure,
    vacancy_sites: Iterable | PeriodicSite | None = None,
    interstitial_sites: Iterable | PeriodicSite | None = None,
    substitution_sites: Iterable | PeriodicSite | None = None,
) -> Structure:
    """
    Generate the supercell containing a defect complex, given the bulk
    supercell and the sites of the defects to be included in the complex.

    The coordinates of the input defect sites should correspond to the
    input bulk supercell. For substitutions, the closest site in the bulk
    supercell to the supplied site(s) will be removed, and replaced with
    the input ``substitution_sites``.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure in which to generate the defect complex.
        vacancy_sites (Iterable | PeriodicSite | None):
            The site(s) of vacancies to include in the defect complex supercell.
            Default is None.
        interstitial_sites (Iterable | PeriodicSite | None):
            The site(s) of interstitials to include in the defect complex supercell.
            Default is None.
        substitution_sites (Iterable | PeriodicSite | None):
            The site(s) of substitutions to include in the defect complex supercell.
            Default is None.

    Returns:
        Structure: The defect complex supercell structure.
    """
    defect_dict = {
        "vacancy_sites": vacancy_sites or [],
        "interstitial_sites": interstitial_sites or [],
        "substitution_sites": substitution_sites or [],
    }
    for key, value in defect_dict.items():
        if isinstance(value, PeriodicSite):
            defect_dict[key] = [value]  # convert to Iterable

        if not isinstance(defect_dict[key], Iterable) and defect_dict[key] is not None:
            raise TypeError(
                f"Defect sites input arguments must be a list, set, or tuple of defect sites. Got "
                f"{type(defect_dict[key])} for {key} instead!"
            )

    defect_struct = bulk_supercell.copy()

    def _get_matching_site(site, structure, sub_site=False):
        if not sub_site:
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

        # else get closest site in structure, raising error if not within 0.5 Å:
        closest_site_idx = np.argmin(np.linalg.norm(structure.cart_coords - site.coords, axis=1))
        closest_site = structure.sites[closest_site_idx]

        closest_site_dist = closest_site.distance(site)
        if closest_site_dist > 0.5:
            raise ValueError(
                f"Closest site to input defect site ({site}) in bulk supercell is {closest_site} "
                f"with distance {closest_site_dist:.2f} Å (greater than 0.5 Å and suggesting a likely "
                f"mismatch in sites/structures here!)."
            )
        return closest_site

    for site in defect_dict["vacancy_sites"]:
        vac_site = _get_matching_site(site, bulk_supercell)
        defect_struct.remove(vac_site)

    for site in defect_dict["interstitial_sites"]:
        defect_struct.insert(
            0,
            species=site.specie,
            coords=site.frac_coords,
        )

    for site in defect_dict["substitution_sites"]:
        bulk_site_idx = defect_struct.index(_get_matching_site(site, bulk_supercell))
        defect_struct.remove_sites([bulk_site_idx])
        defect_struct.insert(bulk_site_idx, species=site.specie, coords=site.frac_coords)

    return defect_struct


# TODO: Update DOIs here and throughout when published
# TODO: Add some quick tests when done too
