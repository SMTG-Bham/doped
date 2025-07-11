"""
Code to analyse VASP defect calculations.

These functions are built from a combination of useful modules from
``pymatgen``, alongside substantial modification, in the efforts of making an
efficient, user-friendly package for managing and analysing defect
calculations, with publication-quality outputs.
"""

import contextlib
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from monty.serialization import dumpfn
from pymatgen.analysis.defects import core
from pymatgen.analysis.defects.finder import cosine_similarity, get_site_vecs
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Composition, Structure
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Procar, Vasprun
from pymatgen.util.typing import PathLike
from tqdm import tqdm

from doped import _doped_obj_properties_methods, _ignore_pmg_warnings, get_mp_context, pool_manager
from doped.core import Defect, DefectEntry, guess_and_set_oxi_states_with_timeout
from doped.generation import (
    get_defect_name_from_defect,
    get_defect_name_from_entry,
    name_defect_entries,
    sort_defect_entries,
)
from doped.thermodynamics import DefectThermodynamics
from doped.utils.efficiency import StructureMatcher_scan_stol, _parse_site_species_str, get_voronoi_nodes
from doped.utils.parsing import (
    _compare_incar_tags,
    _compare_kpoints,
    _compare_potcar_symbols,
    _format_mismatching_incar_warning,
    _get_bulk_locpot_dict,
    _get_bulk_site_potentials,
    _get_defect_supercell_frac_coords,
    _get_output_files_and_check_if_multiple,
    _multiple_files_warning,
    _vasp_file_parsing_action_dict,
    check_atom_mapping_far_from_defect,
    get_core_potentials_from_outcar,
    get_defect_type_site_idxs_and_unrelaxed_structure,
    get_locpot,
    get_matching_site,
    get_procar,
    get_vasprun,
    spin_degeneracy_from_vasprun,
    total_charge_from_vasprun,
)
from doped.utils.plotting import format_defect_name
from doped.utils.symmetry import (
    _frac_coords_sort_func,
    get_equiv_frac_coords_in_primitive,
    get_orientational_degeneracy,
    get_primitive_structure,
    point_symmetry_from_defect_entry,
)


def _custom_formatwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """
    Reformat warnings to just print the warning message, and add two newlines
    for spacing.
    """
    return f"{message}\n\n"


warnings.formatwarning = _custom_formatwarning
_ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings

_aniso_dielectric_but_outcar_problem_warning = (
    "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to compute the "
    "_anisotropic_ Kumagai eFNV charge correction) "
)
# Neither new nor old pymatgen FNV correction can do anisotropic dielectrics (while new sxdefectalign can)
_aniso_dielectric_but_using_locpot_warning = (
    "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) charge "
    "correction developed for _isotropic_ materials will be applied here, which corresponds to using the "
    "effective isotropic average of the supplied anisotropic dielectric. This could lead to significant "
    "errors for very anisotropic systems and/or relatively small supercells!"
)

_CALC_OUTPUT_MASK = ("vasprun.xml", "vasprun.xml.gz")  # mask for identifying calculation files
_SUBFOLDER_PRIORITY = [
    "vasp_ncl",
    "vasp_std",
    "vasp_nkred_std",
    "vasp_gam",
]  # priority order for subfolders
_BULK_FOLDER_PATTERN = "bulk"


def _convert_dielectric_to_tensor(dielectric: float | np.ndarray | list) -> np.ndarray:
    # check if dielectric in required 3x3 matrix format
    if not isinstance(dielectric, float | int):
        dielectric = np.array(dielectric)
        if dielectric.shape == (3,):
            dielectric = np.diag(dielectric)
        elif dielectric.shape != (3, 3):
            raise ValueError(
                f"Dielectric constant must be a float/int or a 3x1 matrix or 3x3 matrix, "
                f"got type {type(dielectric)} and shape {dielectric.shape}"
            )

    else:
        dielectric = np.eye(3) * dielectric

    return dielectric


def _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(
    aniso_dielectric: np.ndarray | list,
) -> float:
    """
    Convert an anisotropic dielectric tensor to the equivalent isotropic
    dielectric constant using the harmonic mean (closest physically reasonable
    choice for finite-size charge corrections).
    """
    return 3 / sum(1 / diagonal_elt for diagonal_elt in np.diag(aniso_dielectric))


def check_and_set_defect_entry_name(
    defect_entry: DefectEntry,
    possible_defect_name: str = "",
) -> None:
    """
    Check that ``possible_defect_name`` is a recognised format by doped (i.e.
    in the format ``"{defect_name}_{optional_site_info}_{charge_state}"``).

    If the ``DefectEntry.name`` attribute is not defined or does not end with
    charge state, then the entry will be renamed with the doped default name
    for the `unrelaxed` defect (i.e. using the point symmetry of the defect
    site in the bulk cell).

    Args:
        defect_entry (DefectEntry): ``DefectEntry`` object.
        possible_defect_name (str):
            Possible defect name (usually the folder name) to check if
            recognised by ``doped``, otherwise defect name is re-determined.
    """
    formatted_defect_name = None
    charge_state = defect_entry.charge_state
    # check if defect folder name ends with charge state:
    defect_name_w_charge_state = (
        possible_defect_name
        if (possible_defect_name.endswith((f"_{charge_state}", f"_{charge_state:+}")))
        else f"{possible_defect_name}_{'+' if charge_state > 0 else ''}{charge_state}"
    )

    with contextlib.suppress(Exception):  # check if defect name is recognised
        formatted_defect_name = format_defect_name(
            defect_name_w_charge_state, include_site_info_in_name=True
        )  # tries without site_info if with site_info fails

    # (re-)determine doped defect name and store in metadata, regardless of whether folder name is
    # recognised:
    if "full_unrelaxed_defect_name" not in defect_entry.calculation_metadata:
        defect_entry.calculation_metadata["full_unrelaxed_defect_name"] = (
            f"{get_defect_name_from_entry(defect_entry, relaxed=False)}_"
            f"{'+' if charge_state > 0 else ''}{charge_state}"
        )

    if formatted_defect_name is not None:
        defect_entry.name = defect_name_w_charge_state
    else:  # otherwise use default doped name
        defect_entry.name = defect_entry.calculation_metadata["full_unrelaxed_defect_name"]


def defect_from_structures(
    bulk_supercell: Structure,
    defect_supercell: Structure,
    return_all_info: bool = False,
    skip_atom_mapping_check: bool = False,
    **kwargs,
) -> Defect | tuple[Defect, PeriodicSite, PeriodicSite, int | None, int | None, Structure, Structure]:
    """
    Auto-determines the defect type and defect site from the supplied bulk and
    defect structures, and returns a corresponding ``Defect`` object with the
    defect site in the primitive structure.

    If ``return_all_info`` is set to true, then also returns:

    - `relaxed` defect site in the defect supercell
    - the defect site in the bulk supercell
    - defect site index in the defect supercell
    - bulk site index (index of defect site in bulk supercell)
    - guessed initial defect structure (before relaxation)
    - 'unrelaxed defect structure' (also before relaxation, but with
      interstitials at their final `relaxed` positions, and all bulk atoms at
      their unrelaxed positions).

    Args:
        bulk_supercell (Structure):
            Bulk supercell structure.
        defect_supercell (Structure):
            Defect structure to use for identifying the defect site and type.
        return_all_info (bool):
            If ``True``, returns additional python objects related to the
            site-matching, listed above. (Default = False)
        skip_atom_mapping_check (bool):
            If ``True``, skips the atom mapping check which ensures that the
            bulk and defect supercell lattice definitions are matched
            (important for accurate defect site determination and charge
            corrections). Can be used to speed up parsing when you are sure
            the cell definitions match (e.g. both supercells were generated
            with ``doped``). Default is ``False``.
        **kwargs:
            Keyword arguments to pass to ``get_equiv_frac_coords_in_primitive``
            (such as ``symprec``, ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor``, ``verbose``) and/or
            ``Defect`` initialization (such as ``oxi_state``, ``multiplicity``,
            ``symprec``, ``dist_tol_factor``). Mainly intended for cases where
            fast site matching and ``Defect`` creation are desired (e.g. when
            analysing MD trajectories of defects), where providing these
            parameters can greatly speed up parsing.
            Setting ``oxi_state='N/A'`` and ``multiplicity=1`` will skip their
            auto-determination and accelerate parsing, if these properties are
            not required.

    Returns:
        defect (Defect):
            ``doped`` ``Defect`` object.

        If ``return_all_info`` is True, then also:

        defect_site (Site):
            ``pymatgen`` ``Site`` object of the `relaxed` defect site in the
            defect supercell.
        defect_site_in_bulk (Site):
            ``pymatgen`` ``Site`` object of the defect site in the bulk
            supercell (i.e. unrelaxed vacancy/substitution site, or final
            `relaxed` interstitial site for interstitials).
        defect_site_index (int):
            Index of defect site in defect supercell (None for vacancies)
        bulk_site_index (int):
            Index of defect site in bulk supercell (None for interstitials)
        guessed_initial_defect_structure (Structure):
            ``pymatgen`` ``Structure`` object of the guessed initial defect
            structure.
        unrelaxed_defect_structure (Structure):
            ``pymatgen`` ``Structure`` object of the unrelaxed defect
            structure.
    """
    try:  # Try automatic defect site detection -- this gives us the "unrelaxed" defect structure
        (
            defect_type,
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_type_site_idxs_and_unrelaxed_structure(bulk_supercell, defect_supercell)

    except RuntimeError as exc:
        check_atom_mapping_far_from_defect(
            bulk_supercell,
            defect_supercell,
            guess_defect_position(defect_supercell),
            coords_are_cartesian=True,
        )
        raise RuntimeError(
            f"Could not identify {defect_type} defect site in defect structure. Please check that your "
            f"defect supercells are reasonable, and that they match the bulk supercell. If so, "
            f"and this error is not resolved, please report this issue to the developers."
        ) from exc

    if defect_type == "vacancy":
        site_in_bulk = defect_site_in_bulk = defect_site = bulk_supercell[bulk_site_idx]
    elif defect_type == "substitution":
        defect_site = defect_supercell[defect_site_idx]
        site_in_bulk = bulk_supercell[bulk_site_idx]  # this is with orig (substituted) element
        defect_site_in_bulk = PeriodicSite(
            defect_site.species, site_in_bulk.frac_coords, site_in_bulk.lattice
        )
    else:  # interstitial
        site_in_bulk = defect_site_in_bulk = defect_site = defect_supercell[defect_site_idx]

    if not skip_atom_mapping_check:
        check_atom_mapping_far_from_defect(
            bulk_supercell, defect_supercell, defect_site_in_bulk.frac_coords
        )
        # Note: This function checks (and warns, if necessary) for large mismatches between defect and bulk
        # supercells, where a common case is a symmetry-equivalent bulk supercell but with a different
        # basis/definition for the atomic positions (discussion:
        # doped.readthedocs.io/en/latest/Troubleshooting.html#mis-matching-bulk-and-defect-supercells )
        # In theory, we could use orient_s2_like_s1 with allow_subset to shift the defect cell to match
        # the (different definition) bulk cell, tracking the site matches, and accounting for the site
        # matches properly with the charge corrections. But, beyond being a lot of work to allow the
        # unnecessary (and usually easily fixed) case of mismatching supercells, which can also lead to
        # other issues, it would require different definitions of 'defect supercell sites' (e.g. for a
        # vacancy with a mismatching supercell definition, the supercell site should be the exact atom site
        # in the bulk supercell, but this is now entirely different from the defect supercell). Also, the
        # choice of matching orientation for the bulk supercell (and thus defect site) can become arbitrary
        # in these situations, where there are many possible defect cell translations etc which match the
        # bulk cell...

    if unrelaxed_defect_structure:
        if defect_type == "interstitial":
            # get closest Voronoi site in bulk supercell to final interstitial site as this is likely
            # the _initial_ interstitial site
            closest_node_frac_coords = min(
                [site.frac_coords for site in get_voronoi_nodes(bulk_supercell)],
                key=lambda node: defect_site.distance_and_image_from_frac_coords(node)[0],
            )
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()
            int_site = guessed_initial_defect_structure[defect_site_idx]
            guessed_initial_defect_structure.remove_sites([defect_site_idx])
            guessed_initial_defect_structure.insert(
                defect_site_idx,  # Place defect at same position as in DFT calculation
                int_site.species_string,
                closest_node_frac_coords,
                coords_are_cartesian=False,
                validate_proximity=True,
            )
            # if guessed initial site is sufficiently close to the relaxed site, then use it as
            # "defect_site_in_bulk", otherwise use the relaxed site:
            if defect_site_in_bulk.distance_and_image_from_frac_coords(closest_node_frac_coords)[0] < 1:
                defect_site_in_bulk = guessed_initial_defect_structure[defect_site_idx]

        else:
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()

    else:
        warnings.warn(
            "Cannot determine the unrelaxed `initial_defect_structure`. Please ensure the "
            "`initial_defect_structure` is indeed unrelaxed."
        )

    # get defect site in primitive structure, for Defect generation:
    primitive_structure = get_primitive_structure(bulk_supercell, symprec=kwargs.get("symprec") or 0.01)
    equiv_frac_coords_in_prim = get_equiv_frac_coords_in_primitive(
        (defect_site if defect_type == "interstitial" else defect_site_in_bulk).frac_coords,
        primitive_structure,
        bulk_supercell,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["symprec", "dist_tol_factor", "fixed_symprec_and_dist_tol_factor", "verbose"]
        },  # allowed kwargs for ``get_equiv_frac_coords_in_primitive``
    )
    equiv_frac_coords_in_prim = sorted(equiv_frac_coords_in_prim, key=_frac_coords_sort_func)
    equiv_defect_sites_in_prim = [
        PeriodicSite(
            defect_site_in_bulk.species,
            frac_coords_in_prim,
            primitive_structure.lattice,
            coords_are_cartesian=False,
        )
        for frac_coords_in_prim in equiv_frac_coords_in_prim
    ]

    if defect_type != "interstitial":  # ensure exact matches to Defect.structure (primitive) sites:
        for defect_site_in_prim in equiv_defect_sites_in_prim:
            bulk_site_in_prim = deepcopy(defect_site_in_prim)
            bulk_site_in_prim.species = site_in_bulk.species
            bulk_site_in_prim = get_matching_site(bulk_site_in_prim, primitive_structure)
            defect_site_in_prim.frac_coords = bulk_site_in_prim.frac_coords

        # also drop unsupported Defect() kwargs for non-interstitial defects:
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["dist_tol_factor", "fixed_symprec_and_dist_tol_factor", "verbose"]
        }

    for_monty_defect = {  # initialise doped Defect object, needs to use defect site in bulk (which for
        # substitutions differs from defect_site)
        "@module": "doped.core",
        "@class": defect_type.capitalize(),
        "structure": primitive_structure,
        "site": equiv_defect_sites_in_prim[0],
        "equivalent_sites": equiv_defect_sites_in_prim,
        **kwargs,
    }  # define the Defect object in the primitive structure, matching the approach for generation
    defect = MontyDecoder().process_decoded(for_monty_defect)

    if not return_all_info:
        return defect

    return (
        defect,
        defect_site,
        defect_site_in_bulk,
        defect_site_idx,
        bulk_site_idx,
        guessed_initial_defect_structure,
        unrelaxed_defect_structure,
    )


def defect_and_info_from_structures(
    bulk_supercell: Structure,
    defect_supercell: Structure,
    skip_atom_mapping_check: bool = False,
    initial_defect_structure_path: PathLike | None = None,
    **kwargs,
) -> tuple[Defect, PeriodicSite, dict]:
    """
    Generates a corresponding ``Defect`` object from the supplied bulk and
    defect supercells (using ``defect_from_structures``), and returns the
    ``Defect`` object, the `relaxed` defect site in the defect supercell, and a
    dictionary of calculation metadata (including the defect site in the bulk
    supercell, defect site indices in the defect and bulk supercells, the
    guessed initial defect structure, and the unrelaxed defect structure).

    Args:
        bulk_supercell (Structure):
            Bulk supercell structure.
        defect_supercell (Structure):
            Defect structure to use for identifying the defect site and type.
        skip_atom_mapping_check (bool):
            If ``True``, skips the atom mapping check which ensures that the
            bulk and defect supercell lattice definitions are matched
            (important for accurate defect site determination and charge
            corrections). Can be used to speed up parsing when you are sure
            the cell definitions match (e.g. both supercells were generated
            with ``doped``). Default is ``False``.
        initial_defect_structure_path (PathLike):
            Path to the initial/unrelaxed defect structure. Only recommended
            for use if structure matching with the relaxed defect structure(s)
            fails (rare). Default is ``None``.
        **kwargs:
            Keyword arguments to pass to ``get_equiv_frac_coords_in_primitive``
            (such as ``symprec``, ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor``, ``verbose``) and/or
            ``Defect`` initialization (such as ``oxi_state``, ``multiplicity``,
            ``symprec``, ``dist_tol_factor``). Mainly intended for cases where
            fast site matching and ``Defect`` creation are desired (e.g. when
            analysing MD trajectories of defects), where providing these
            parameters can greatly speed up parsing.
            Setting ``oxi_state='N/A'`` and ``multiplicity=1`` will skip their
            auto-determination and accelerate parsing, if these properties are
            not required.

    Returns:
        tuple[Defect, PeriodicSite, dict]:
            defect (Defect):
                ``doped`` ``Defect`` object.
            defect_site (Site):
                ``pymatgen`` ``Site`` object of the `relaxed` defect site in
                the defect supercell.
            defect_structure_metadata (dict):
                Dictionary containing metadata about the defect structure,
                including:

                - ``guessed_initial_defect_structure``: The guessed initial
                  defect structure (before relaxation).
                - ``guessed_defect_displacement``: Displacement from the
                  guessed initial defect site to the final `relaxed` site
                  (``None`` for vacancies).
                - ``defect_site_index``: Index of the defect site in the defect
                  supercell (``None`` for vacancies).
                - ``bulk_site_index``: Index of the defect site in the bulk
                  supercell (``None`` for interstitials).
                - ``unrelaxed_defect_structure``: The unrelaxed defect
                  structure (similar to ``guessed_initial_defect_structure``,
                  but with interstitials at their final `relaxed` positions,
                  and all bulk atoms at their unrelaxed positions).
                - ``bulk_site``: The defect site in the bulk supercell (i.e.
                  unrelaxed vacancy/substitution site, or final `relaxed` site
                  for interstitials).
    """
    defect_structure_metadata = {}

    # identify defect site, structural information, and create defect object:
    # Can specify initial defect structure (to help find the defect site if we have a very distorted
    # final structure), but regardless try using the final structure (from defect OUTCAR) first:
    try:
        (
            defect,
            defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
            defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
            # w/interstitials (if guessed initial site is sufficiently close to the relaxed site, then
            # it is used here, otherwise the actual relaxed site is used)
            defect_site_index,
            bulk_site_index,
            guessed_initial_defect_structure,
            unrelaxed_defect_structure,
        ) = defect_from_structures(
            bulk_supercell,
            defect_supercell,
            skip_atom_mapping_check=skip_atom_mapping_check,
            return_all_info=True,
            **kwargs,
        )

    except RuntimeError:
        if not initial_defect_structure_path:
            raise

        defect_structure_for_ID = Poscar.from_file(initial_defect_structure_path).structure.copy()
        (
            defect,
            defect_site_in_initial_struct,
            defect_site_in_bulk,  # bulk site for vac/sub, relaxed defect site w/interstitials
            defect_site_index,  # in this initial_defect_structure
            bulk_site_index,
            guessed_initial_defect_structure,
            unrelaxed_defect_structure,
        ) = defect_from_structures(
            bulk_supercell,
            defect_structure_for_ID,
            skip_atom_mapping_check=skip_atom_mapping_check,
            return_all_info=True,
            **kwargs,
        )

        # then try get defect_site in final structure:
        # need to check that it's the correct defect site and hasn't been reordered/changed compared to
        # the initial_defect_structure used here -> check same element and distance reasonable:
        defect_site = defect_site_in_initial_struct

        if defect.defect_type != core.DefectType.Vacancy:
            final_defect_site = defect_supercell[defect_site_index]
            if (
                defect_site_in_initial_struct.specie.symbol == final_defect_site.specie.symbol
            ) and final_defect_site.distance(defect_site_in_initial_struct) < 2:
                defect_site = final_defect_site

    defect_structure_metadata["guessed_initial_defect_structure"] = guessed_initial_defect_structure
    defect_structure_metadata["defect_site_index"] = defect_site_index
    defect_structure_metadata["bulk_site_index"] = bulk_site_index

    # add displacement from (guessed) initial site to final defect site:
    if defect_site_index is not None:  # not a vacancy
        guessed_initial_site = guessed_initial_defect_structure[defect_site_index]
        guessed_displacement = defect_site.distance(guessed_initial_site)
        defect_structure_metadata["guessed_initial_defect_site"] = guessed_initial_site
        defect_structure_metadata["guessed_defect_displacement"] = guessed_displacement
    else:  # vacancy
        defect_structure_metadata["guessed_initial_defect_site"] = bulk_supercell[bulk_site_index]
        defect_structure_metadata["guessed_defect_displacement"] = None  # type: ignore

    defect_structure_metadata["unrelaxed_defect_structure"] = unrelaxed_defect_structure
    if bulk_site_index is None:  # interstitial
        defect_structure_metadata["bulk_site"] = defect_site_in_bulk
    else:
        defect_structure_metadata["bulk_site"] = bulk_supercell[bulk_site_index]

    return (
        defect,
        defect_site,
        defect_structure_metadata,
    )


def guess_defect_position(defect_supercell: Structure) -> np.ndarray[float]:
    """
    Guess the position (in Cartesian coordinates) of a defect in an input
    defect supercell, without a bulk/reference supercell.

    This is achieved by computing cosine dissimilarities between site SOAP
    vectors (and the mean SOAP vectors for each species) and then determining
    the centre of mass of sites, weighted by the squared cosine
    dissimilarities. For accurate defect site determination, the
    ``defect_from_structure`` function (or underlying code) is preferred. These
    coordinates are unlikely to `directly` match the defect position
    (especially in the presence of random noise), but should provide a pretty
    good estimate in most cases. If the defect is an extrinsic interstitial /
    substitution, then this will identify the exact defect site.

    Args:
        defect_supercell (Structure):
            Defect supercell structure.

    Returns:
        np.ndarray[float]:
            Guessed position of the defect in **Cartesian** coordinates.
    """

    # Note from profiling: This function is pretty fast (e.g. ~25 s for ~1000 frames of a ~100-atom
    # supercell on SK's 2021 MacBook Pro), but the main bottleneck is SOAP vector creation,
    # if we ever needed to accelerate
    def cos_dissimilarity(vec1, vec2):
        return 1 - cosine_similarity(vec1, vec2)

    # if there is only one site of a particular element in the defect supercell, then we guess it as the
    # defect site (extrinsic substitution/interstitial):
    i_elt_dict = {
        i: _parse_site_species_str(site, wout_charge=True) for i, site in enumerate(defect_supercell.sites)
    }
    for elt in defect_supercell.composition.elements:
        if list(i_elt_dict.values()).count(elt.symbol) == 1:
            return defect_supercell.sites[list(i_elt_dict.values()).index(elt.symbol)].coords

    soap_vecs = [site_vec.vec for site_vec in get_site_vecs(defect_supercell)]
    elt_mean_soap_vec_dict = {
        elt.symbol: np.mean(
            [soap_vec for i, soap_vec in enumerate(soap_vecs) if i_elt_dict[i] == elt.symbol],
            axis=0,
        )
        for elt in defect_supercell.composition.elements
    }
    cos_dissimilarities = [
        cos_dissimilarity(soap_vecs[i], elt_mean_soap_vec_dict[i_elt]) for i, i_elt in i_elt_dict.items()
    ]

    rel_cos_dissimilarities = np.zeros(len(defect_supercell))
    for elt in elt_mean_soap_vec_dict:
        indices = [i for i, i_elt in i_elt_dict.items() if i_elt == elt]
        avg_cos_dissimilarity = np.mean([cos_dissimilarities[i] for i in indices])
        rel_cos_dissimilarities[indices] = np.array(cos_dissimilarities)[indices] / avg_cos_dissimilarity

    largest_outlier = defect_supercell.sites[
        np.where(rel_cos_dissimilarities == np.max(rel_cos_dissimilarities))[0][0]
    ]
    cos_diss_frac_coords_dict = {np.max(rel_cos_dissimilarities): largest_outlier.frac_coords}
    for i, site in enumerate(defect_supercell.sites):
        if not np.all(site.frac_coords == largest_outlier.frac_coords):
            image = largest_outlier.distance_and_image(site)[1]
            cos_diss_frac_coords_dict[rel_cos_dissimilarities[i]] = site.frac_coords + image

    cos_diss_coords_dict = dict(
        zip(
            cos_diss_frac_coords_dict.keys(),
            defect_supercell.lattice.get_cartesian_coords(
                np.array(list(cos_diss_frac_coords_dict.values()))
            ),
            strict=False,
        )
    )
    return np.average(  # weighted centre of mass
        np.array(list(cos_diss_coords_dict.values())),
        axis=0,
        weights=np.array(list(cos_diss_coords_dict.keys())) ** 2,
    )


def defect_name_from_structures(bulk_supercell: Structure, defect_supercell: Structure, **kwargs) -> str:
    """
    Get the doped/SnB defect name using the bulk and defect structures.

    Args:
        bulk_supercell (Structure):
            Bulk (pristine) structure.
        defect_supercell (Structure):
            Defect structure.
        **kwargs:
            Keyword arguments to pass to ``defect_from_structures`` (such as
            ``oxi_state``, ``multiplicity``, ``symprec``, ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor``, ``verbose``).

    Returns:
        str: Defect name.
    """
    # set oxi_state and multiplicity to avoid wasting time trying to auto-determine when unnecessary here
    default_init_kwargs = {"oxi_state": "Undetermined", "multiplicity": 1}
    default_init_kwargs.update(kwargs)
    defect = defect_from_structures(
        bulk_supercell, defect_supercell, return_all_info=False, **default_init_kwargs  # type: ignore
    )
    assert isinstance(defect, Defect)  # mypy typing

    # note that if the symm_op approach fails for any reason here, the defect-supercell expansion
    # approach will only be valid if the defect structure is a diagonal expansion of the primitive...

    return get_defect_name_from_defect(defect)


def defect_entry_from_paths(
    defect_path: PathLike,
    bulk_path: PathLike,
    dielectric: float | np.ndarray | list | None = None,
    charge_state: int | None = None,
    skip_corrections: bool = False,
    error_tolerance: float = 0.05,
    bulk_band_gap_vr: PathLike | Vasprun | None = None,
    **kwargs,
) -> DefectEntry:
    """
    Parse the defect calculation outputs in ``defect_path`` and return the
    parsed ``DefectEntry`` object.

    By default, the ``DefectEntry.name`` attribute (later used to label the
    defects in plots) is set to the defect_path folder name (if it is a
    recognised defect name), else it is set to the default ``doped`` name for
    that defect (using the estimated `unrelaxed` defect structure, for the
    point group and neighbour distances).

    Note that the bulk and defect supercells should have the same definitions /
    basis sets (for site-matching and finite-size charge corrections to work
    appropriately).

    Args:
        defect_path (PathLike):
            Path to defect supercell folder (containing at least
            ``vasprun.xml(.gz)``).
        bulk_path (PathLike):
            Path to bulk supercell folder (containing at least
            ``vasprun.xml(.gz)``).
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constant (ionic + static contributions), in the
            same xyz Cartesian basis as the supercell calculations (likely but
            not necessarily the same as the raw output of a VASP dielectric
            calculation, if an oddly-defined primitive cell is used). If not
            provided, charge corrections cannot be computed and so
            ``skip_corrections`` will be set to ``True``. See
            https://doped.readthedocs.io/en/latest/GGA_workflow_tutorial.html#dielectric-constant
            for information on calculating and converging the dielectric
            constant.
        charge_state (int):
            Charge state of defect. If not provided, will be automatically
            determined from the defect calculation outputs.
        skip_corrections (bool):
            Whether to skip the calculation and application of finite-size
            charge corrections to the defect energy (not recommended in most
            cases). Default is ``False``.
        error_tolerance (float):
            If the estimated error in the defect charge correction, based on
            the variance of the potential in the sampling region is greater
            than this value (in eV), then a warning is raised. Default is 0.05
            eV.
        bulk_band_gap_vr (PathLike or Vasprun):
            Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` ``Vasprun``
            object, from which to determine the bulk band gap and band edge
            positions. If the VBM/CBM occur at `k`-points which are not
            included in the bulk supercell calculation, then this parameter
            should be used to provide the output of a bulk bandstructure
            calculation so that these are correctly determined.
            Alternatively, you can edit/add the ``"band_gap"`` and ``"vbm"``
            entries in ``self.defect_entry.calculation_metadata`` to match the
            correct (eigen)values.
            If None, will use ``DefectEntry.calculation_metadata["bulk_path"]``
            (i.e. the bulk supercell calculation output).

            Note that the ``"band_gap"`` and ``"vbm"`` values should only
            affect the reference for the Fermi level values output by ``doped``
            (as this VBM eigenvalue is used as the zero reference), thus
            affecting the position of the band edges in the defect formation
            energy plots and doping window / dopability limit functions, and
            the reference of the reported Fermi levels.
        **kwargs:
            Keyword arguments to pass to ``DefectParser()`` methods
            (``load_FNV_data()``, ``load_eFNV_data()``,
            ``load_bulk_gap_data()``), ``point_symmetry_from_defect_entry()``
            or ``defect_and_info_from_structures``, including
            ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
            ``mpid``, ``api_key``, ``oxi_state``, ``multiplicity``,
            ``angle_tolerance``, ``user_charges``,
            ``initial_defect_structure_path`` etc (see their docstrings).
            Note that ``bulk_symprec`` can be supplied as the ``symprec`` value
            to use for determining equivalent sites (and thus defect
            multiplicities / unrelaxed site symmetries), while an input
            ``symprec`` value will be used for determining `relaxed` site
            symmetries.

    Returns:
        Parsed ``DefectEntry`` object.
    """
    dp = DefectParser.from_paths(
        defect_path,
        bulk_path,
        dielectric=dielectric,
        charge_state=charge_state,
        skip_corrections=skip_corrections,
        error_tolerance=error_tolerance,
        bulk_band_gap_vr=bulk_band_gap_vr,
        **kwargs,
    )
    return dp.defect_entry


class DefectsParser:
    def __init__(
        self,
        output_path: PathLike = ".",
        dielectric: float | np.ndarray | list | None = None,
        subfolder: PathLike | None = None,
        bulk_path: PathLike | None = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        bulk_band_gap_vr: PathLike | Vasprun | None = None,
        processes: int | None = None,
        json_filename: PathLike | bool | None = None,
        parse_projected_eigen: bool | None = None,
        **kwargs,
    ):
        r"""
        A class for rapidly parsing multiple VASP defect supercell calculations
        for a given host (bulk) material.

        Loops over calculation directories in ``output_path`` (likely the same
        ``output_path`` used with ``DefectsSet`` for file generation in
        ``doped.vasp``) and parses the defect calculations into a dictionary
        of: ``{defect_name: DefectEntry}``, where the ``defect_name`` is set to
        the defect calculation folder name (`if it is a recognised defect
        name`), else it is set to the default ``doped`` name for that defect
        (using the estimated `unrelaxed` defect structure, for the point group
        and neighbour distances). By default, searches for folders in
        ``output_path`` with ``subfolder`` containing ``vasprun.xml(.gz)``
        files, and tries to parse them as ``DefectEntry``\s.

        By default, tries multiprocessing to speed up defect parsing, which can
        be controlled with ``processes``. If parsing hangs, this may be due to
        memory issues, in which case you should manually reduce ``processes``
        (e.g. <=4).

        Defect charge states are automatically determined from the defect
        calculation outputs if ``POTCAR``\s are set up with ``pymatgen`` (see
        docs Installation page), or if that fails, using the defect folder name
        (must end in "_+X" or "_-X" where +/-X is the defect charge state).

        Uses the (single) ``DefectParser`` class to parse the individual defect
        calculations. Note that the bulk and defect supercells should have the
        same definitions/basis sets (for site-matching and finite-size charge
        corrections to work appropriately).

        Args:
            output_path (PathLike):
                Path to the output directory containing the defect calculation
                folders (likely the same ``output_path`` used with
                ``DefectsSet`` for file generation in ``doped.vasp``).
                Default is current directory.
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Total dielectric constant (ionic + static contributions), in
                the same xyz Cartesian basis as the supercell calculations
                (likely but not necessarily the same as the raw output of a
                VASP dielectric calculation, if an oddly-defined primitive cell
                is used). If not provided, charge corrections cannot be
                computed and so ``skip_corrections`` will be set to ``True``.
                See https://doped.readthedocs.io/en/latest/GGA_workflow_tutorial.html#dielectric-constant
                for information on calculating and converging the dielectric
                constant.
            subfolder (PathLike):
                Name of subfolder(s) within each defect calculation folder (in
                the ``output_path`` directory) containing the VASP calculation
                files to parse (e.g. ``vasp_ncl``, ``vasp_std``, ``vasp_gam``
                etc.). If not specified, ``doped`` checks first for
                ``vasp_ncl``, ``vasp_std``, ``vasp_gam`` subfolders with
                calculation outputs (``vasprun.xml(.gz)`` files) and uses the
                highest level VASP type (ncl > std > gam) found as
                ``subfolder``, otherwise uses the defect calculation folder
                itself with no subfolder (set ``subfolder = "."`` to enforce
                this).
            bulk_path (PathLike):
                Path to bulk supercell reference calculation folder. If not
                specified, searches for folder with name "X_bulk" in the
                ``output_path`` directory (matching the default ``doped`` name
                for the bulk supercell reference folder). Can be the full path,
                or the relative path from the ``output_path`` directory.
            skip_corrections (bool):
                Whether to skip the calculation & application of finite-size
                charge corrections to the defect energies (not recommended in
                most cases). Default is ``False``.
            error_tolerance (float):
                If the estimated error in any charge correction, based on the
                variance of the potential in the sampling region, is greater
                than this value (in eV), then a warning is raised. Default is
                0.05 eV. Note that this warning is skipped for defects which
                are predicted to not be stable for any Fermi level in the band
                gap (based on all parsed defects here), or are predicted to be
                shallow (perturbed host) states according to eigenvalue
                analysis and only be stable for Fermi levels within a small
                window to a band edge (taken as the smaller of
                ``error_tolerance`` or 10% of the band gap, by default, or can
                be set by a ``shallow_charge_stability_tolerance = X`` keyword
                argument).
            bulk_band_gap_vr (PathLike or Vasprun):
                Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen``
                ``Vasprun`` object, from which to determine the bulk band gap
                and band edge positions. If the VBM/CBM occur at `k`-points
                which are not included in the bulk supercell calculation, then
                this parameter should be used to provide the output of a bulk
                bandstructure calculation so that these are correctly
                determined. Alternatively, you can edit the ``"band_gap"`` and
                ``"vbm"`` entries in ``self.defect_entry.calculation_metadata``
                to match the correct (eigen)values. If ``None``, will use
                ``DefectEntry.calculation_metadata["bulk_path"]``
                (i.e. the bulk supercell calculation output).

                Note that the ``"band_gap"`` and ``"vbm"`` values should only
                affect the reference for the Fermi level values output by
                ``doped`` (as this VBM eigenvalue is used as the zero
                reference), thus affecting the position of the band edges in
                the defect formation energy plots and doping window /
                dopability limit functions, and the reference of the reported
                Fermi levels.
            processes (int):
                Number of processes to use for multiprocessing for expedited
                parsing. If not set, defaults to one less than the number of
                CPUs available. Set to 1 for no multiprocessing.
            json_filename (PathLike):
                Filename to save the parsed defect entries dict
                (``DefectsParser.defect_dict``) to in ``output_path``, to avoid
                having to re-parse defects when later analysing further and
                aiding calculation provenance. Can be reloaded using the
                ``loadfn`` function from ``monty.serialization`` (and then
                input to ``DefectThermodynamics`` etc.). If ``None`` (default),
                set as ``{Host Chemical Formula}_defect_dict.json.gz``.
                If ``False``, no json file is saved.
            parse_projected_eigen (bool):
                Whether to parse the projected eigenvalues & magnetization from
                the bulk and defect calculations (so
                ``DefectEntry.get_eigenvalue_analysis()`` can then be used with
                no further parsing, and magnetization values can be pulled for
                SOC / non-collinear magnetism calculations). Will initially try
                to load orbital projections from ``vasprun.xml(.gz)`` files
                (slightly slower but more accurate), or failing that from
                ``PROCAR(.gz)`` files if present in the bulk/defect
                directories. Parsing this data can increase total parsing time
                by anywhere from ~5-25%, so set to ``False`` if parsing speed
                is crucial.
                Default is ``None``, which will attempt to load this data but
                with no warning if it fails (otherwise if ``True`` a warning
                will be printed).
            **kwargs:
                Keyword arguments to pass to ``DefectParser()`` methods
                (``load_FNV_data()``, ``load_eFNV_data()``,
                ``load_bulk_gap_data()``),
                ``point_symmetry_from_defect_entry()`` or
                ``defect_and_info_from_structures``, including
                ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
                ``mpid``, ``api_key``, ``oxi_state``, ``multiplicity``,
                ``angle_tolerance``, ``user_charges``,
                ``initial_defect_structure_path`` etc. (see their docstrings);
                or for controlling shallow defect charge correction error
                warnings (see ``error_tolerance`` description) with
                ``shallow_charge_stability_tolerance``.
                Note that ``bulk_symprec`` can be supplied as the ``symprec``
                value to use for determining equivalent sites (and thus defect
                multiplicities / unrelaxed site symmetries), while an input
                ``symprec`` value will be used for determining `relaxed` site
                symmetries.

        Attributes:
            defect_dict (dict):
                Dictionary of parsed defect calculations in the format:
                ``{"defect_name": DefectEntry}`` where the defect_name is set
                to the defect calculation folder name (`if it is a recognised
                defect name`), else it is set to the default ``doped`` name for
                that defect (using the estimated `unrelaxed` defect structure,
                for the point group and neighbour distances).
        """
        self.output_path = output_path
        self.dielectric = dielectric
        self.skip_corrections = skip_corrections
        self.error_tolerance = error_tolerance
        self.bulk_path = bulk_path
        self.subfolder = subfolder
        if bulk_band_gap_vr and not isinstance(bulk_band_gap_vr, Vasprun):
            self.bulk_band_gap_vr = get_vasprun(bulk_band_gap_vr, parse_projected_eigen=False)
        else:
            self.bulk_band_gap_vr = bulk_band_gap_vr
        self.processes = processes
        self.json_filename = json_filename
        self.parse_projected_eigen = parse_projected_eigen
        self.bulk_vr = None  # loaded later
        self.kwargs = kwargs

        # get folders for parsing:
        self.defect_folders, self.output_path, self.subfolder, self.bulk_path = (
            _get_calculation_folders_for_parsing(self.output_path, self.subfolder, self.bulk_path)
        )

        pbar = tqdm(total=len(self.defect_folders), desc="Parsing bulk reference calculation")
        # parse bulk calculation:
        self.bulk_vr, self.bulk_procar = _parse_vr_and_poss_procar(
            output_path=self.bulk_path,
            parse_projected_eigen=self.parse_projected_eigen,
            label="bulk",
            parse_procar=True,
        )
        self.parse_projected_eigen = any(
            i is not None for i in [self.bulk_vr.projected_eigenvalues, self.bulk_procar]
        )

        # try parsing the bulk oxidation states first, for later assigning defect "oxi_state"s (i.e.
        # fully ionised charge states):
        pbar.set_description("Guessing oxidation states in bulk structure")
        self._bulk_oxi_states: Structure | Composition | dict | bool = False
        if bulk_struct_w_oxi := guess_and_set_oxi_states_with_timeout(
            self.bulk_vr.final_structure, break_early_if_expensive=True
        ):
            self.bulk_vr.final_structure = self._bulk_oxi_states = bulk_struct_w_oxi

        self.defect_dict = {}
        self.bulk_corrections_data = {  # so we only load and parse bulk data once
            "bulk_locpot_dict": None,
            "bulk_site_potentials": None,
        }
        parsed_defect_entries: list[DefectEntry] = []
        parsing_warnings: list[str] = []

        # set up multiprocessing:
        mp = get_mp_context()  # https://github.com/python/cpython/pull/100229
        if self.processes is None:  # only multiprocess as much as makes sense, if only few defect folders:
            self.processes = min(max(1, mp.cpu_count() - 1), len(self.defect_folders) - 1)

        if self.processes <= 1:  # no multiprocessing
            for folder in self.defect_folders:
                parsed_defect_entry, processed_warnings_string = self._parse_defect_and_handle_warnings(
                    folder, pbar=pbar
                )
                parsing_warnings.append(processed_warnings_string)  # parsing warnings/errors
                parsed_defect_entries.append(parsed_defect_entry)  # None if failed parsing

        else:  # otherwise multiprocessing:
            # here we try to parse one charged defect first, to check if dielectric and charge corrections
            # are correctly set, and loading the bulk reference data for charge corrections for efficiency,
            # before then using multiprocessing for the rest of the defect folders, with the same settings:
            charged_defect_folder = None  # find a charged defect folder to parse first
            for possible_charged_defect_folder in self.defect_folders:
                with contextlib.suppress(Exception):
                    if abs(int(possible_charged_defect_folder[-1])) > 0:  # likely charged defect
                        charged_defect_folder = possible_charged_defect_folder

            try:
                if charged_defect_folder is not None:
                    # will throw warnings if dielectric is None / charge corrections not possible,
                    # and set self.skip_corrections appropriately
                    parsed_defect_entry, processed_warnings_string = (
                        self._parse_defect_and_handle_warnings(charged_defect_folder, pbar=pbar)
                    )
                    parsing_warnings.append(processed_warnings_string)  # parsing warnings/errors
                    parsed_defect_entries.append(parsed_defect_entry)  # None if failed parsing

                # also load the other bulk corrections data if possible:
                for k, v in self.bulk_corrections_data.items():
                    if v is None:
                        with contextlib.suppress(Exception):
                            if k == "bulk_locpot_dict":
                                self.bulk_corrections_data[k] = _get_bulk_locpot_dict(
                                    self.bulk_path, quiet=True
                                )
                            elif k == "bulk_site_potentials":
                                self.bulk_corrections_data[k] = _get_bulk_site_potentials(
                                    self.bulk_path,
                                    quiet=True,
                                    total_energy=[
                                        self.bulk_vr.final_energy,
                                        self.bulk_vr.ionic_steps[-1]["electronic_steps"][-1]["e_0_energy"],
                                    ],
                                )

                folders_to_process = [
                    folder for folder in self.defect_folders if folder != charged_defect_folder
                ]
                pbar.set_description("Setting up multiprocessing")
                if self.processes > 1:
                    with pool_manager(self.processes) as pool:  # parsed_defect_entry, warnings
                        pbar.set_description(
                            f"Parsing {folders_to_process[0]}/{self.subfolder}".replace("/.", "")
                        )
                        for parsed_defect_entry, processed_warnings_string in pool.imap_unordered(
                            self._parse_defect_and_handle_warnings, folders_to_process
                        ):
                            pbar.update()
                            if parsed_defect_entry is not None:
                                defect_folder = _get_defect_folder(parsed_defect_entry, self.subfolder)
                                pbar.set_description(
                                    f"Parsed {defect_folder}/{self.subfolder}".replace("/.", "")
                                )
                            parsing_warnings.append(processed_warnings_string)  # parsing warnings/errors
                            parsed_defect_entries.append(parsed_defect_entry)  # None if failed parsing

            except Exception as exc:
                pbar.close()
                raise exc

            finally:
                pbar.close()

        _format_and_raise_parsing_warnings(  # format and raise any parsing warnings
            parsing_warnings, bulk_path=self.bulk_path, subfolder=self.subfolder
        )

        parsed_defect_entries = [
            i for i in parsed_defect_entries if i is not None
        ]  # remove None (failed parsing)
        if not parsed_defect_entries:
            subfolder_string = f" and `subfolder`: '{self.subfolder}'" if self.subfolder != "." else ""
            raise ValueError(
                f"No defect calculations in `output_path` '{self.output_path}' were successfully parsed, "
                f"using `bulk_path`: {self.bulk_path}{subfolder_string}. Please check the correct "
                f"defect/bulk paths and subfolder are being set, and that the folder structure is as "
                f"expected (see `DefectsParser` docstring)."
            )

        self.defect_dict = _name_parsed_defect_entries(parsed_defect_entries, subfolder=self.subfolder)

        # handle (and warn) any charge correction errors or calculation parameter mismatches:
        # TODO: Add these to a separate defect_parsing_checks function, and add in dimer detection with
        #  info message about checking triplet states
        self._handle_charge_correction_errors(self.error_tolerance, **kwargs)
        _warn_calculation_mismatches(self.defect_dict)  # warn any mismatching defect/bulk calc parameters

        if self.json_filename is not False:  # save to json unless json_filename is False:
            if self.json_filename is None:
                formula = next(
                    iter(self.defect_dict.values())
                ).defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
                self.json_filename = f"{formula}_defect_dict.json.gz"

            assert isinstance(self.json_filename, str)  # typing
            dumpfn(self.defect_dict, os.path.join(self.output_path, self.json_filename))

    def _parse_single_defect(self, defect_folder: str) -> DefectEntry | None:
        """
        Parse a single defect calculation at
        ``{self.output_path}/{defect_folder}/{self.subfolder}``, using
        ``DefectParser.from_paths()``.

        Args:
            defect_folder (str):
                The defect folder to parse in ``self.output_path`` (and using
                ``self.subfolder``), with ``DefectParser.from_paths()``.

        Returns:
            DefectEntry | None:
                The parsed ``DefectEntry`` object, or ``None`` if parsing
                failed.
        """
        try:
            self.kwargs.update(self.bulk_corrections_data)  # update with bulk corrections data
            assert isinstance(self.subfolder, str)  # typing, converted to str by this point
            dp = DefectParser.from_paths(
                defect_path=os.path.join(self.output_path, defect_folder, self.subfolder),
                bulk_path=self.bulk_path,
                bulk_vr=self.bulk_vr,
                bulk_procar=self.bulk_procar,
                dielectric=self.dielectric,
                skip_corrections=self.skip_corrections,
                error_tolerance=self.error_tolerance,
                bulk_band_gap_vr=self.bulk_band_gap_vr,
                oxi_state=self.kwargs.get("oxi_state") if self._bulk_oxi_states else "Undetermined",
                parse_projected_eigen=self.parse_projected_eigen,
                **self.kwargs,
            )

            if dp.skip_corrections and dp.defect_entry.charge_state != 0 and self.dielectric is None:
                self.skip_corrections = dp.skip_corrections  # set skip_corrections to True if
                # dielectric is None and there are charged defects present (shows dielectric warning once)

            for bulk_correction_data_key in [
                "bulk_locpot_dict",
                "bulk_site_potentials",
            ]:
                if (
                    dp.defect_entry.calculation_metadata.get(bulk_correction_data_key) is not None
                    and self.bulk_corrections_data.get(bulk_correction_data_key) is None
                ):  # if not already set, update
                    self.bulk_corrections_data[bulk_correction_data_key] = (
                        dp.defect_entry.calculation_metadata[bulk_correction_data_key]
                    )

        except Exception as exc:
            warnings.warn(
                f"Parsing failed for "
                f"{defect_folder if self.subfolder == '.' else f'{defect_folder}/{self.subfolder}'}, "
                f"got error: {exc!r}"
            )
            return None

        return dp.defect_entry

    def _parse_defect_and_handle_warnings(self, defect_folder: str, pbar: tqdm | None = None) -> tuple:
        """
        Process defect and catch warnings along the way, so we can print which
        warnings came from which defect together at the end, in a summarised
        output.

        Args:
            defect_folder (str):
                The defect folder to parse in ``self.output_path`` (and using
                ``self.subfolder``), with ``_parse_single_defect``.
            pbar (tqdm):
                ``tqdm`` progress bar to update with parsing progress.

        Returns:
            tuple: (parsed_defect_entry, warnings_string)
        """
        if pbar:  # set tqdm progress bar description to defect folder being parsed:
            pbar.set_description(f"Parsing {defect_folder}/{self.subfolder}".replace("/.", ""))

        with warnings.catch_warnings(record=True) as captured_warnings:
            parsed_defect_entry = self._parse_single_defect(defect_folder)

        ignore_messages = [
            "Estimated error",
            "There are mismatching",
            "The KPOINTS",
            "The POTCAR",
        ]  # collectively warned later

        def _check_ignored_message_in_warning(warning_message):
            if hasattr(warning_message, "args"):
                return any(warning_message.args[0].startswith(i) for i in ignore_messages)
            return any(warning_message.startswith(i) for i in ignore_messages)

        warnings_string = "\n\n".join(
            str(warning.message)
            for warning in captured_warnings
            if not _check_ignored_message_in_warning(warning.message)
        )

        defect_path = (
            parsed_defect_entry.calculation_metadata.get("defect_path", "N/A")
            if parsed_defect_entry is not None
            else f"{defect_folder}/{self.subfolder}"
        )
        processed_warnings_string = _process_parsing_warnings(warnings_string, defect_folder, defect_path)

        if pbar:
            pbar.update()

        return parsed_defect_entry, processed_warnings_string

    def _handle_charge_correction_errors(self, error_tolerance: float, **kwargs) -> None:
        """
        Check for charge correction errors and warn if they exceed the error
        tolerance.

        Args:
            error_tolerance (float):
                The error tolerance threshold for charge corrections (in eV),
                used to decide whether to trigger a warning.
            **kwargs:
                Additional keyword arguments, such as
                ``shallow_charge_stability_tolerance``.
        """
        FNV_correction_errors: list[tuple[str, float]] = []
        eFNV_correction_errors: list[tuple[str, float]] = []
        defect_thermo = self.get_defect_thermodynamics(check_compatibility=False, skip_dos_check=True)

        for name, defect_entry in self.defect_dict.items():
            # first check if it's a stable defect:
            fermi_stability_window = defect_thermo._get_in_gap_fermi_level_stability_window(defect_entry)

            if fermi_stability_window < 0 or (  # Note we avoid the prune_to_stable_entries() method here
                defect_entry.is_shallow  # as this would require two ``DefectThermodynamics`` inits...
                and fermi_stability_window
                < kwargs.get(
                    "shallow_charge_stability_tolerance",
                    min(error_tolerance, defect_thermo.band_gap * 0.1 if defect_thermo.band_gap else 0.05),
                )
            ):
                continue  # no charge correction warnings for unstable charge states

            for correction_type, correction_error_list in [
                ("freysoldt", FNV_correction_errors),
                ("kumagai", eFNV_correction_errors),
            ]:
                if (
                    defect_entry.corrections_metadata.get(f"{correction_type}_charge_correction_error", 0)
                    > error_tolerance
                ):
                    correction_error_list.append(
                        (
                            name,
                            defect_entry.corrections_metadata[
                                f"{correction_type}_charge_correction_error"
                            ],
                        )
                    )

        def _call_multiple_corrections_tolerance_warning(correction_errors, type="FNV"):
            long_name = "Freysoldt" if type == "FNV" else "Kumagai"
            if error_tolerance >= 0.01:  # if greater than 10 meV, round energy values to meV:
                error_tol_string = f"{error_tolerance:.3f}"
                correction_errors_string = "\n".join(
                    f"{name}: {error:.3f} eV" for name, error in correction_errors
                )
            else:  # else give in scientific notation:
                error_tol_string = f"{error_tolerance:.2e}"
                correction_errors_string = "\n".join(
                    f"{name}: {error:.2e} eV" for name, error in correction_errors
                )

            warnings.warn(
                f"Estimated error in the {long_name} ({type}) charge correction for certain defects is "
                f"greater than the `error_tolerance` (= {error_tol_string} eV):"
                f"\n{correction_errors_string}\n"
                f"You may want to check the accuracy of the corrections by plotting the site potential "
                f"differences (using `defect_entry.get_{long_name.lower()}_correction()` with "
                f"`plot=True`). Large errors are often due to unstable or shallow defect charge states "
                f"(which can't be accurately modelled with the supercell approach). If these errors are "
                f"not acceptable, you may need to use a larger supercell for more accurate energies."
            )

        for correction_errors, type in [
            (FNV_correction_errors, "FNV"),
            (eFNV_correction_errors, "eFNV"),
        ]:
            if correction_errors:
                _call_multiple_corrections_tolerance_warning(correction_errors, type=type)

        # check if same type of charge correction was used in each case or not:
        if (
            len(
                {
                    k
                    for defect_entry in self.defect_dict.values()
                    for k in defect_entry.corrections
                    if k.endswith("_charge_correction")
                }
            )
            > 1
        ):
            warnings.warn(
                "Beware: The Freysoldt (FNV) charge correction scheme has been used for some defects, "
                "while the Kumagai (eFNV) scheme has been used for others. For _isotropic_ materials, "
                "this should be fine, and the results should be the same regardless (assuming a "
                "relatively well-converged supercell size), while for _anisotropic_ materials this could "
                "lead to some quantitative inaccuracies. You can use the "
                "`DefectThermodynamics.get_formation_energies()` method to print out the calculated "
                "charge corrections for all defects, and/or visualise the charge corrections using "
                "`defect_entry.get_freysoldt_correction`/`get_kumagai_correction` with `plot=True` to "
                "check."
            )
        # note that we also check if multiple charge corrections have been applied to the same defect
        # within the charge correction functions (with self._check_if_multiple_finite_size_corrections())

    def get_defect_thermodynamics(
        self,
        chempots: dict | None = None,
        el_refs: dict | None = None,
        vbm: float | None = None,
        band_gap: float | None = None,
        dist_tol: float = 1.5,
        check_compatibility: bool = True,
        bulk_dos: FermiDos | None = None,
        skip_dos_check: bool = False,
        **kwargs,
    ) -> DefectThermodynamics:
        r"""
        Generates a ``DefectThermodynamics`` object from the parsed
        ``DefectEntry`` objects in ``self.defect_dict``\, which can then be
        used to analyse and plot the defect thermodynamics (formation energies,
        transition levels, concentrations etc).

        Note that the ``DefectEntry.name`` attributes (rather than the
        ``defect_name`` key in the ``defect_dict``) are used to label the
        defects in plots.

        See the ``DefectThermodynamics`` and accompanying methods docstrings in
        ``doped.thermodynamics`` for more.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the
                defect formation energies. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format
                generated by ``doped``\'s chemical potential parsing functions
                (see tutorials)) which allows easy analysis over a range of
                chemical potentials -- where limit(s) (chemical potential
                limit(s)) to analyse/plot can later be chosen using the
                ``limits`` argument.

                Alternatively this can be a dictionary of chemical potentials
                for a single limit, in the format:
                ``{element symbol: chemical potential}``. If manually
                specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the
                elemental phases in order to show the formal (relative)
                chemical potentials above the formation energy plot, in which
                case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the
                absolute (DFT) chemical potentials should be given.

                If ``None`` (default), sets all chemical potentials to zero.
                Chemical potentials can also be supplied later in each analysis
                function. (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical
                potentials in the format:
                ``{element symbol: reference energy}`` (to determine the formal
                chemical potentials, when ``chempots`` has been manually
                specified as ``{element symbol: chemical potential}``).
                Unnecessary if ``chempots`` is provided in format generated by
                ``doped`` (see tutorials).

                If ``None`` (default), sets all elemental reference energies to
                zero. Reference energies can also be supplied later in each
                analysis function, or set using
                ``DefectThermodynamics.el_refs = ...`` (with the same input
                options).
            vbm (float):
                VBM eigenvalue to use as Fermi level reference point for
                analysis. If ``None`` (default), will use ``"vbm"`` from the
                ``calculation_metadata`` dict attributes of the parsed
                ``DefectEntry`` objects, which by default is taken from the
                bulk supercell VBM (unless ``bulk_band_gap_vr`` is set during
                parsing). Note that ``vbm`` should only affect the reference
                for the Fermi level values output by ``doped`` (as this VBM
                eigenvalue is used as the zero reference), thus affecting the
                position of the band edges in the defect formation energy plots
                and doping window / dopability limit functions, and the
                reference of the reported Fermi levels.
            band_gap (float):
                Band gap of the host, to use for analysis.
                If ``None`` (default), will use "band_gap" from the
                ``calculation_metadata`` dict attributes of the parsed
                ``DefectEntry`` objects.
            dist_tol (float):
                Threshold for the closest distance (in Å) between equivalent
                defect sites, for different species of the same defect type,
                to be grouped together (for plotting, transition level analysis
                and defect concentration calculations). For the most part, if
                the minimum distance between equivalent defect sites is less
                than ``dist_tol``, then they will be grouped together,
                otherwise treated as separate defects.
                See ``plot()`` and ``get_fermi_level_and_concentrations()``
                docstrings for more information.
                (Default: 1.5)
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each
                defect entry (i.e. that all reference bulk energies are the
                same).
                (Default: True)
            bulk_dos (FermiDos or Vasprun or PathLike):
                ``pymatgen`` ``FermiDos`` for the bulk electronic density of
                states (DOS), for calculating Fermi level positions and
                defect/carrier concentrations. Alternatively, can be a
                ``pymatgen`` ``Vasprun`` object or path to the
                ``vasprun.xml(.gz)`` output of a bulk DOS calculation in VASP.
                Can also be provided later when using
                ``get_equilibrium_fermi_level()``,
                ``get_fermi_level_and_concentrations`` etc, or set using
                ``DefectThermodynamics.bulk_dos = ...`` (with the same input
                options).

                Usually this is a static calculation with the `primitive` cell
                of the bulk material, with relatively dense `k`-point sampling
                (especially for materials with disperse band edges) to ensure
                an accurately-converged DOS and thus Fermi level. Using large
                ``NEDOS`` (>3000) and ``ISMEAR = -5`` (tetrahedron smearing)
                are recommended for best convergence (wrt `k`-point sampling)
                in VASP. Consistent functional settings should be used for the
                bulk DOS and defect supercell calculations. See
                https://doped.readthedocs.io/en/latest/Tips.html#density-of-states-dos-calculations
                (Default: None)
            skip_dos_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                the defect entries VBM by >0.05 eV. Should only be used when
                the reason for this difference is known/acceptable.
                (Default: False)
            **kwargs:
                Additional keyword arguments to pass to the
                ``DefectThermodynamics`` constructor.

        Returns:
            ``doped`` ``DefectThermodynamics`` object
        """
        if not self.defect_dict or self.defect_dict is None:
            raise ValueError(
                "No defects found in `defect_dict`. DefectThermodynamics object can only be generated "
                "when defects have been parsed and are present as `DefectEntry`s in "
                "`DefectsParser.defect_dict`."
            )

        return DefectThermodynamics(
            list(self.defect_dict.values()),
            chempots=chempots,
            el_refs=el_refs,
            vbm=vbm,
            band_gap=band_gap,
            dist_tol=dist_tol,
            check_compatibility=check_compatibility,
            bulk_dos=bulk_dos,
            skip_dos_check=skip_dos_check,
            **kwargs,
        )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectsParser`` object.
        """
        formula = next(
            iter(self.defect_dict.values())
        ).defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectsParser for bulk composition {formula}, with {len(self.defect_dict)} parsed "
            f"defect entries in self.defect_dict. Available attributes:\n{properties}\n\n"
            f"Available methods:\n{methods}"
        )


def _get_calculation_folders_for_parsing(
    output_path: PathLike = ".",
    subfolder: PathLike | None = None,
    bulk_path: PathLike | None = None,
) -> tuple[list[str], str, str, str]:
    """
    Get calculation folders for parsing.

    Args:
        output_path (PathLike):
            Path to the output directory containing the calculation folders to
            be parsed. Default is current directory (".").
        subfolder (PathLike | None):
            Name of subfolder(s) within each calculation folder (in the
            ``output_path`` directory) from which to parse. If not specified
            (default), ``doped`` checks first for ``vasp_ncl``, ``vasp_std``,
            ``vasp_gam`` subfolders with calculation outputs
            (``vasprun.xml(.gz)`` files) and uses the highest level VASP type
            (ncl > std > gam) found as ``subfolder``, otherwise uses the
            defect calculation folder itself with no subfolder (set
            ``subfolder = "."`` to enforce this).
        bulk_path (PathLike | None):
            Path to bulk reference calculation folder. If not specified,
            searches for folder with "bulk" in the name in the ``output_path``
            directory (matching the default ``doped`` name for the bulk
            reference folder). Can be the full path, or the relative path from
            the ``output_path`` directory.

    Returns:
        tuple[list[str], PathLike, PathLike, PathLike]:
            List of calculation folders for parsing, output path, subfolder,
            and bulk path (the last three of which are the input arguments
            which may have been updated within this function).
    """
    out_root = Path(output_path).resolve()
    user_set_subfolder = subfolder is not None

    def _get_calc_files_df(root: Path) -> pd.DataFrame:
        """
        Get a DataFrame of calculation output files in folders under ``root``,
        matching the ``_CALC_OUTPUT_MASK`` filter, recursively, ignoring hidden
        files and folders.
        """
        files_df = _dataframe_of_files(root)  # dataframe of files in folders under ``out_root``
        pattern = "|".join(map(re.escape, _CALC_OUTPUT_MASK))  # regex filter pattern for output files
        return (
            files_df[files_df["filename"].str.contains(pattern, regex=True, na=False)]
            if not files_df.empty
            else pd.DataFrame()
        )

    calc_files_df = _get_calc_files_df(out_root)  # DataFrame of calculation output files
    if calc_files_df.empty:  # user may have specified defect sub-folder directly, so check one level up
        parent_root = out_root.parent
        calc_files_df = _get_calc_files_df(parent_root)
        files_not_found_error = FileNotFoundError(
            f"No calculation folders with any of {_CALC_OUTPUT_MASK} in filenames found under "
            f"{out_root}."
        )
        if calc_files_df.empty:  # no calculation output files found
            raise files_not_found_error

        possible_defect_folders = [  # candidate defect folders
            g
            for g in calc_files_df["folder_in_root"].unique()
            if out_root.name in g  # only the specific defect directory specified
            or _BULK_FOLDER_PATTERN in g.lower()  # or a bulk directory, for later
            or (bulk_path and str(bulk_path).lower() in g.lower())
        ]
        if not possible_defect_folders:
            raise files_not_found_error
        out_root = parent_root  # shift context to parent directory

    else:
        possible_defect_folders = calc_files_df["folder_in_root"].unique().tolist()

    subfolder = (
        _determine_subfolder(calc_files_df, possible_defect_folders) if subfolder is None else subfolder
    )

    possible_bulk_folders = [  # candidate bulk folders
        g
        for g in possible_defect_folders
        if _BULK_FOLDER_PATTERN in str(g).lower()
        or (bulk_path and str(g).lower() == str(bulk_path).lower())
    ]
    defect_folders = [
        d  # update candidate defect calculation folders, based on bulk calculation folder(s) and subfolder
        for d in possible_defect_folders
        if d not in possible_bulk_folders and (subfolder == "." or (out_root / d / subfolder).is_dir())
    ]

    bulk_path = _resolve_bulk_path(out_root, possible_bulk_folders, bulk_path)  # resolve bulk path
    bulk_path = _append_subfolder_if_needed(bulk_path, subfolder, user_set_subfolder)

    return defect_folders, str(out_root), str(subfolder), str(bulk_path)


def _dataframe_of_files(root: Path) -> pd.DataFrame:
    """
    Get a dataframe with one row per file under *root*.

    Args:
        root (Path):
            Path to the root directory.
    """
    rows: list[dict[str, Any]] = []
    for f in root.rglob("*"):  # recursively find all files under root, ignoring hidden folders/files
        if f.is_file():
            relative_to_root_parts = f.relative_to(root).parts
            if (
                any(part.startswith(".") for part in relative_to_root_parts)
                or len(relative_to_root_parts) < 2
            ):  # ignore hidden files and folders, and files in root directory itself
                continue
            rows.append(
                {
                    "filename": f.name,
                    "full_path": f,
                    "folder_path": f.parent,
                    "folder_in_root": f.relative_to(root).parts[0],
                }
            )
    return pd.DataFrame(rows)


def _determine_subfolder(files_df: pd.DataFrame, defect_folders: list[str]) -> str:
    """
    Pick the highest-priority calculation subfolder name, or "." if none found.

    Args:
        files_df (pd.DataFrame):
            DataFrame with one row per file in folders under ``out_root``.
        defect_folders (list[str]):
            List of defect calculation folders (in ``out_root``).

    Returns:
        str:
            The highest-priority calculation subfolder name, or "." if none
            found.
    """
    defect_folders_df = files_df[files_df["folder_in_root"].isin(defect_folders)]
    for subfolder in _SUBFOLDER_PRIORITY:
        if any(subfolder in p.name for p in defect_folders_df["folder_path"].unique()):
            return subfolder
    return "."


def _resolve_bulk_path(
    out_root: Path, possible_bulk_folders: list[str], bulk_path: PathLike | None
) -> Path:
    """
    Return absolute Path to bulk folder (may contain subfolder later).

    Args:
        out_root (Path):
            Path to the output directory.
        possible_bulk_folders (list[str]):
            List of possible bulk calculation folders (in ``out_root``).
        bulk_path (str | None):
            User-provided explicit path to the bulk calculation directory.
    """
    if bulk_path is None:
        if len(possible_bulk_folders) == 1:
            return out_root / possible_bulk_folders[0]  # only one possible bulk folder, so return it

        suffix_bulk = [
            d for d in possible_bulk_folders if str(d).lower().endswith(f"_{_BULK_FOLDER_PATTERN}")
        ]
        if len(suffix_bulk) == 1:
            return out_root / next(iter(suffix_bulk))  # only one possible bulk folder, so return it

        raise ValueError(
            f"Could not determine bulk supercell calculation folder in {out_root}, found "
            f"{len(possible_bulk_folders)} folders containing any of {_CALC_OUTPUT_MASK} in filenames (in "
            f"subfolders) and '{_BULK_FOLDER_PATTERN}' in the folder name. Please specify `bulk_path` "
            f"manually."
        )

    bulk_path = Path(bulk_path)
    if bulk_path and not bulk_path.is_absolute():
        bulk_path = out_root / bulk_path  # make relative path absolute
    if bulk_path and not bulk_path.is_dir():
        raise FileNotFoundError(f"Could not find bulk supercell calculation folder at '{bulk_path}'!")

    return bulk_path


def _append_subfolder_if_needed(bulk_path: Path, subfolder: PathLike, user_set: bool) -> Path:
    """
    Ensure ``bulk_path`` actually contains calculation files; dive into
    ``subfolder`` if needed.

    Args:
        bulk_path (Path):
            Path to the bulk calculation directory.
        subfolder (str):
            Subfolder with calculation output files.
        user_set (bool):
            Whether the subfolder was explicitly set by the user.

    Returns:
        Path:
            Path to the bulk calculation directory, with subfolder if needed.
    """
    if (bulk_path / subfolder).is_dir() and any(
        k in f.name for k in _CALC_OUTPUT_MASK for f in (bulk_path / subfolder).iterdir()
    ):  # subfolder contains calculation output files, so add to bulk path
        return bulk_path / subfolder

    if not any(k in f.name for k in _CALC_OUTPUT_MASK for f in bulk_path.iterdir()):  # no output files
        possible_bulk_subfolders = [
            p
            for p in bulk_path.iterdir()
            if p.is_dir() and any(k in f.name for k in _CALC_OUTPUT_MASK for f in p.iterdir())
        ]
        if len(possible_bulk_subfolders) == 1 and not user_set:
            # if only one subfolder with calculation outputs, and `subfolder` not explicitly set, use this:
            return possible_bulk_subfolders[0].resolve()

        raise FileNotFoundError(
            f"No files with any of {_CALC_OUTPUT_MASK} in names found under {bulk_path} (subfolder "
            f"{subfolder}). Please ensure bulk supercell calculation files are present and/or specify "
            f"`bulk_path` manually."
        )
    return bulk_path


def _process_parsing_warnings(
    warnings_string: str = "",
    defect_folder: str = "",
    defect_path: str = "N/A",
) -> str:
    """
    Process any warnings from parsing.

    Args:
        warnings_string (str):
            String containing warnings from parsing, to be processed.
        defect_folder (str):
            Name of the defect folder being parsed, for formatting the warning
            message.
        defect_path (str):
            Path to the defect calculation directory, for formatting the
            warning message. Default is "N/A".

    Returns:
        str:
            Processed warnings string, formatted for clarity and readability.
            If there are no warnings or exceptions, returns an empty string.
    """
    if warnings_string:
        split_warnings = warnings_string.split("\n\n")
        if "Parsing failed for " not in warnings_string or len(split_warnings) > 1:
            location = f" at {defect_path}" if defect_path != "N/A" else ""  # let's ride the vibration
            return (  # either only warnings (no exceptions), or warning(s) + exception
                f"Warning(s) encountered when parsing {defect_folder}{location}:\n\n{warnings_string}"
            )

    return warnings_string  # if exception, return as is, or "" if no warnings


def _format_and_raise_parsing_warnings(
    parsing_warnings: list[str], bulk_path: str = "bulk", subfolder: str = "."
) -> None:
    """
    Process and display parsing warnings in an organized manner, grouping
    duplicate warnings/errors.

    Args:
        parsing_warnings (list[str]):
            List of warning/error strings from defect calculation parsing.
        bulk_path (str):
            Path to the bulk calculation directory (just for formatted error /
            warning messages). Default is "bulk".
        subfolder (str):
            Subfolder of the defect calculation directory (just for formatted
            error / warning messages). Default is ".".
    """
    parsing_warnings = [warning for warning in parsing_warnings if warning]  # remove empty strings
    if not parsing_warnings:
        return

    split_parsing_warnings = [warning.split("\n\n") for warning in parsing_warnings]

    def _mention_bulk_path_subfolder_for_correction_warnings(warning: str) -> str:
        if "defect & bulk" in warning or "defect or bulk" in warning:
            # charge correction file warning, print subfolder and bulk_path:
            if subfolder == ".":
                warning += f"\n(using bulk path: {bulk_path} and without defect subfolders)"
            else:
                warning += f"\n(using bulk path {bulk_path} and {subfolder} defect subfolders)"

        return warning

    split_parsing_warnings = [
        [_mention_bulk_path_subfolder_for_correction_warnings(warning) for warning in warning_list]
        for warning_list in split_parsing_warnings
    ]
    flattened_warnings_list = [
        warning for warning_list in split_parsing_warnings for warning in warning_list
    ]
    duplicate_warnings: dict[str, list[str]] = {
        warning: []
        for warning in set(flattened_warnings_list)
        if flattened_warnings_list.count(warning) > 1 and "Parsing failed for " not in warning
    }
    new_parsing_warnings = []
    parsing_errors_dict: dict[str, list[str]] = {
        message.split("got error: ")[1]: []
        for message in set(flattened_warnings_list)
        if "Parsing failed for " in message
    }
    multiple_files_warning_dict: dict[str, list[tuple]] = {
        "vasprun.xml": [],
        "OUTCAR": [],
        "LOCPOT": [],
    }

    for warnings_list in split_parsing_warnings:
        failed_warnings = [
            warning_message
            for warning_message in warnings_list
            if "Parsing failed for " in warning_message
        ]
        if failed_warnings:
            defect_name = failed_warnings[0].split("Parsing failed for ")[1].split(", got ")[0]
            error = failed_warnings[0].split("got error: ")[1]
            parsing_errors_dict[error].append(defect_name)
        elif "Warning(s) encountered" in warnings_list[0]:
            defect_name = warnings_list[0].split("when parsing ")[1].split(" at")[0]
        else:
            defect_name = None

        new_warnings_list = []
        for warning in warnings_list:
            if warning.startswith("Multiple"):
                file_type = warning.split("`")[1]
                directory = warning.split("directory: ")[1].split(". Using")[0]
                chosen_file = warning.split("Using ")[1].split(" to")[0]
                multiple_files_warning_dict[file_type].append((directory, chosen_file))

            elif warning in duplicate_warnings:
                duplicate_warnings[warning].append(defect_name or "N/A")

            else:
                new_warnings_list.append(warning)

        if [  # if we still have other warnings, keep them for parsing_warnings list
            warning
            for warning in new_warnings_list
            if "Warning(s) encountered" not in warning and "Parsing failed for " not in warning
        ]:
            new_parsing_warnings.append(
                "\n".join(
                    [warning for warning in new_warnings_list if "Parsing failed for " not in warning]
                )
            )

    for error, defect_list in parsing_errors_dict.items():
        if defect_list:
            if len(set(defect_list)) > 1:
                warnings.warn(f"Parsing failed for defects: {defect_list} with the same error:\n{error}")
            else:
                warnings.warn(f"Parsing failed for defect {defect_list[0]} with error:\n{error}")

    for file_type, directory_file_list in multiple_files_warning_dict.items():
        if directory_file_list:
            joined_info_string = "\n".join(
                [f"{directory}: {file}" for directory, file in directory_file_list]
            )
            warnings.warn(
                f"Multiple `{file_type}` files found in certain defect directories:\n"
                f"(directory: chosen file for parsing):\n"
                f"{joined_info_string}\n"
                f"{file_type} files are used to {_vasp_file_parsing_action_dict[file_type]}"
            )

    if new_parsing_warnings:
        warnings.warn("\n\n".join(new_parsing_warnings))

    for warning, defect_name_list in duplicate_warnings.items():
        # remove None and don't warn if later encountered parsing error (already warned)
        defect_set = {defect_name for defect_name in defect_name_list if defect_name}
        if defect_set:
            warnings.warn(f"Defects: {defect_set} each encountered the same warning:\n{warning}")


def _get_defect_folder(entry: DefectEntry, subfolder: str = ".") -> str:
    """
    Get the defect folder name from which a ``DefectEntry`` object was parsed.

    Args:
        entry (DefectEntry):
            The defect entry to get the folder name from.
        subfolder (str):
            The subfolder of the defect calculation directory.

    Returns:
        str:
            The defect folder name.
    """
    return (
        entry.calculation_metadata["defect_path"]
        .replace("/.", "")
        .split("/")[-1 if subfolder == "." else -2]
    )


def _name_parsed_defect_entries(
    parsed_defect_entries: list[DefectEntry], subfolder: str = "."
) -> dict[str, DefectEntry]:
    """
    Format parsed defect entries, including naming and sorting, handling any
    duplicates and renaming appropriately.

    Args:
        parsed_defect_entries (list[DefectEntry]):
            List of parsed defect entries to format.
        subfolder (str):
            Defect calculation subfolder name.

    Returns:
        dict[str, DefectEntry]:
            Formatted dictionary of defect entries.
    """
    # sort input entries for deterministic naming:
    parsed_defect_entries = sort_defect_entries(parsed_defect_entries)

    # check if there are duplicate entries in the parsed defect entries, warn and remove:
    energy_entries_dict: dict[float, list[DefectEntry]] = {}  # {energy: [defect_entry]}
    for defect_entry in parsed_defect_entries:  # find duplicates by comparing supercell energies
        if defect_entry.sc_entry_energy in energy_entries_dict:
            energy_entries_dict[defect_entry.sc_entry_energy].append(defect_entry)
        else:
            energy_entries_dict[defect_entry.sc_entry_energy] = [defect_entry]

    for energy, entries_list in energy_entries_dict.items():
        if len(entries_list) > 1:  # more than one entry with the same energy
            # sort any duplicates by name length, name, folder length, folder (shorter preferred)
            energy_entries_dict[energy] = sorted(
                entries_list,
                key=lambda x: (
                    len(x.name),
                    x.name,
                    len(_get_defect_folder(x, subfolder)),
                    _get_defect_folder(x, subfolder),
                ),
            )

    if any(len(entries_list) > 1 for entries_list in energy_entries_dict.values()):
        duplicate_entry_names_folders_string = "\n".join(
            "["
            + ", ".join(f"{entry.name} ({_get_defect_folder(entry, subfolder)})" for entry in entries_list)
            + "]"
            for entries_list in energy_entries_dict.values()
            if len(entries_list) > 1
        )
        warnings.warn(
            f"The following parsed defect entries were found to be duplicates (exact same defect "
            f"supercell energies). The first of each duplicate group shown will be kept and the "
            f"other duplicate entries omitted:\n{duplicate_entry_names_folders_string}"
        )
    parsed_defect_entries = [next(iter(entries_list)) for entries_list in energy_entries_dict.values()]

    # get any defect entries in parsed_defect_entries that share the same name (without charge):
    # first get any entries with duplicate names:
    entries_to_rename = [
        defect_entry
        for defect_entry in parsed_defect_entries
        if len(
            [
                defect_entry
                for other_defect_entry in parsed_defect_entries
                if defect_entry.name == other_defect_entry.name
            ]
        )
        > 1
    ]
    # then get all entries with the same name(s), ignoring charge state (in case e.g. only duplicate
    # for one charge state etc):
    entries_to_rename = [
        defect_entry
        for defect_entry in parsed_defect_entries
        if any(
            defect_entry.name.rsplit("_", 1)[0] == other_defect_entry.name.rsplit("_", 1)[0]
            for other_defect_entry in entries_to_rename
        )
    ]

    # Create initial defect_dict with non-duplicate entries
    defect_dict = {
        defect_entry.name: defect_entry
        for defect_entry in parsed_defect_entries
        if defect_entry not in entries_to_rename
    }

    with contextlib.suppress(AttributeError, TypeError):  # sort by supercell frac cooords,
        # to aid deterministic naming:
        entries_to_rename.sort(key=lambda x: _frac_coords_sort_func(_get_defect_supercell_frac_coords(x)))

    new_named_defect_entries_dict = name_defect_entries(entries_to_rename)
    # set name attribute: (these are names without charges!)
    for defect_name_wout_charge, defect_entry in new_named_defect_entries_dict.items():
        defect_entry.name = (
            f"{defect_name_wout_charge}_{'+' if defect_entry.charge_state > 0 else ''}"
            f"{defect_entry.charge_state}"
        )

    if duplicate_names := [  # if any duplicate names, crash (and burn, b...)
        defect_entry.name for defect_entry in entries_to_rename if defect_entry.name in defect_dict
    ]:
        raise ValueError(
            f"Some defect entries have the same name, due to mixing of doped-named and unnamed "
            f"defect folders. This would cause defect entries to be overwritten. Please check "
            f"your defect folder names in `output_path`!\nDuplicate defect names:\n"
            f"{duplicate_names}"
        )

    defect_dict.update(
        {defect_entry.name: defect_entry for defect_entry in new_named_defect_entries_dict.values()}
    )

    return sort_defect_entries(defect_dict)


def _warn_calculation_mismatches(defect_dict: dict[str, DefectEntry]) -> None:
    """
    Generic handler for mismatching calculation parameters, stored in
    ``DefectEntry.calculation_metadata``.
    """
    # key = mismatch key, value = dict with transform of DefectEntry.calculation_metadata[mismatch key],
    # and message format function:
    mismatch_dict: dict[str, dict] = {
        "mismatching_INCAR_tags": {
            "transform": set,
            "message": lambda lst: (
                "'Defects: (INCAR tag, value in defect calculation, value in bulk calculation))':\n"
                f"{_format_mismatching_incar_warning(lst)}\n"
                "In general, the same INCAR settings should be used in all final calculations for these "
                "tags which can affect energies!"
            ),
        },
        "mismatching_KPOINTS": {
            "transform": lambda v: v,  # no change
            "message": lambda lst: (
                "(defect kpoints, bulk kpoints)):\n" + "\n".join(f"{n}: {m}" for n, m in lst) + "\n"
                "In general, the same KPOINTS settings should be used for all final calculations for "
                "accurate results!"
            ),
        },
        "mismatching_POTCAR_symbols": {
            "transform": lambda v: v,
            "message": lambda lst: (
                "(defect POTCARs, bulk POTCARs)):\n" + "\n".join(f"{n}: {m}" for n, m in lst) + "\n"
                "In general, the same POTCAR settings should be used for all calculations for accurate "
                "results!"
            ),
        },
    }

    for mismatch_key, mismatch_spec in mismatch_dict.items():
        mismatch_object = mismatch_key.split("_")[1]  # "mismatching_INCAR_tags" -> "INCAR" (for message)
        if mismatch_object == "INCAR":
            mismatch_object = "INCAR tags"
        elif mismatch_object == "POTCAR":
            mismatch_object = "POTCAR symbols"  # otherwise "KPOINTS" stays as is

        mismatches = [
            (name, mismatch_spec["transform"](entry.calculation_metadata[mismatch_key]))
            for name, entry in defect_dict.items()
            if entry.calculation_metadata.get(mismatch_key, False)
        ]
        if not mismatches:
            continue

        # sort by number of items then by name, descending, then warn
        mismatches.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)

        warnings.warn(
            f"There are mismatching {mismatch_object} for (some of) your defect and bulk calculations "
            f"which are likely to cause errors in the parsed results (energies). Found the following "
            f"differences:\n(in the format: {mismatch_spec['message'](mismatches)})"
        )


def _parse_vr_and_poss_procar(
    output_path: PathLike,
    parse_projected_eigen: bool | None = None,
    label: str = "bulk",
    parse_procar: bool = True,
):
    procar = None
    failed_eig_parsing_warning_message = (
        f"Could not parse eigenvalue data from vasprun.xml.gz files in {label} folder at {output_path}"
    )

    vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", output_path)
    if multiple:
        _multiple_files_warning("vasprun.xml", output_path, vr_path, dir_type=label)

    try:
        vr = get_vasprun(
            vr_path,
            parse_projected_eigen=parse_projected_eigen is not False,
            parse_eigen=(parse_projected_eigen is not False or label == "bulk"),
        )  # vr.eigenvalues not needed for defects except for vr-only eigenvalue analysis
    except Exception as vr_exc:
        vr = get_vasprun(vr_path, parse_projected_eigen=False, parse_eigen=label == "bulk")
        failed_eig_parsing_warning_message += f", got error:\n{vr_exc}"

        if parse_procar:
            procar_path, multiple = _get_output_files_and_check_if_multiple("PROCAR", output_path)
            if multiple:
                _multiple_files_warning("PROCAR", output_path, procar_path, dir_type=label)
            if "PROCAR" in procar_path and parse_projected_eigen is not False:
                try:
                    procar = get_procar(procar_path)

                except Exception as procar_exc:
                    failed_eig_parsing_warning_message += (
                        f"\nThen got the following error when attempting to parse projected eigenvalues "
                        f"from the defect PROCAR(.gz):\n{procar_exc}"
                    )

    if vr.projected_eigenvalues is None and procar is None and parse_projected_eigen is True:
        # only warn if parse_projected_eigen is set to True (not None)
        warnings.warn(failed_eig_parsing_warning_message)

    return vr, procar if parse_procar else vr


class DefectParser:
    def __init__(
        self,
        defect_entry: DefectEntry,
        defect_vr: Vasprun | None = None,
        bulk_vr: Vasprun | None = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        parse_projected_eigen: bool | None = None,
        **kwargs,
    ):
        """
        Create a ``DefectParser`` object, which has methods for parsing the
        results of defect supercell calculations.

        Direct initialisation with ``DefectParser()`` is typically not
        recommended. Rather ``DefectParser.from_paths()`` or
        ``defect_entry_from_paths()`` are preferred as shown in the ``doped``
        parsing tutorials.

        Args:
            defect_entry (DefectEntry):
                doped ``DefectEntry``
            defect_vr (Vasprun):
                ``pymatgen`` ``Vasprun`` object for the defect supercell
                calculation.
            bulk_vr (Vasprun):
                ``pymatgen`` ``Vasprun`` object for the reference bulk
                supercell calculation.
            skip_corrections (bool):
                Whether to skip calculation and application of finite-size
                charge corrections to the defect energy (not recommended in
                most cases). Default is ``False``.
            error_tolerance (float):
                If the estimated error in the defect charge correction, based
                on the variance of the potential in the sampling region is
                greater than this value (in eV), then a warning is raised.
                Default is 0.05 eV.
            parse_projected_eigen (bool):
                Whether to parse the projected eigenvalues & magnetization from
                the bulk and defect calculations (so
                ``DefectEntry.get_eigenvalue_analysis()`` can then be used with
                no further parsing, and magnetization values can be pulled for
                SOC / non-collinear magnetism calculations). Will initially try
                to load orbital projections from ``vasprun.xml(.gz)`` files
                (slightly slower but more accurate), or failing that from
                ``PROCAR(.gz)`` files if present in the bulk/defect
                directories. Parsing this data can increase total parsing time
                by anywhere from ~5-25%, so set to ``False`` if parsing speed
                is crucial.
                Default is ``None``, which will attempt to load this data but
                with no warning if it fails (otherwise if ``True`` a warning
                will be printed).
            **kwargs:
                Keyword arguments to pass to ``DefectParser()`` methods
                (``load_FNV_data()``, ``load_eFNV_data()``,
                ``load_bulk_gap_data()``),
                ``point_symmetry_from_defect_entry()`` or
                ``defect_and_info_from_structures``, including
                ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
                ``mpid``, ``api_key``, ``oxi_state``, ``multiplicity``,
                ``angle_tolerance``, ``user_charges``,
                ``initial_defect_structure_path`` etc (see their docstrings).
                Primarily used by ``DefectsParser`` to expedite parsing by
                avoiding reloading bulk data for each defect. Note that
                ``bulk_symprec`` can be supplied as the ``symprec`` value to
                use for determining equivalent sites (and thus defect
                multiplicities / unrelaxed site symmetries), while an input
                ``symprec`` value will be used for determining `relaxed` site
                symmetries.
        """
        self.defect_entry: DefectEntry = defect_entry
        self.defect_vr = defect_vr
        self.bulk_vr = bulk_vr
        self.skip_corrections = skip_corrections
        self.error_tolerance = error_tolerance
        self.kwargs = kwargs or {}
        self.parse_projected_eigen = parse_projected_eigen

    @classmethod
    def from_paths(
        cls,
        defect_path: PathLike,
        bulk_path: PathLike | None = None,
        bulk_vr: Vasprun | None = None,
        bulk_procar: Procar | None = None,
        dielectric: float | np.ndarray | list | None = None,
        charge_state: int | None = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        bulk_band_gap_vr: PathLike | Vasprun | None = None,
        parse_projected_eigen: bool | None = None,
        **kwargs,
    ):
        """
        Parse the defect calculation outputs in ``defect_path`` and return the
        ``DefectParser`` object. By default, the
        ``DefectParser.defect_entry.name`` attribute (later used to label
        defects in plots) is set to the defect_path folder name (if it is a
        recognised defect name), else it is set to the default `doped`` name
        for that defect (using the estimated `unrelaxed` defect structure, for
        the point group and neighbour distances).

        Note that the bulk and defect supercells should have the same
        definitions/basis sets (for site-matching and finite-size charge
        corrections to work appropriately).

        Args:
            defect_path (PathLike):
                Path to defect supercell folder (containing at least
                ``vasprun.xml(.gz)``).
            bulk_path (PathLike):
                Path to bulk supercell folder (containing at least
                ``vasprun.xml(.gz)``). Not required if ``bulk_vr`` is provided.
            bulk_vr (Vasprun):
                ``pymatgen`` ``Vasprun`` object for the reference bulk
                supercell calculation, if already loaded (can be supplied to
                expedite parsing). Default is ``None``.
            bulk_procar (Procar):
                ``pymatgen`` ``Procar`` object, for the reference bulk
                supercell calculation if already loaded (can be supplied to
                expedite parsing). Default is ``None``.
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Total dielectric constant (ionic + static contributions), in
                the same xyz Cartesian basis as the supercell calculations
                (likely but not necessarily the same as the raw output of a
                VASP dielectric calculation, if an oddly-defined primitive cell
                is used). If not provided, charge corrections cannot be
                computed and so ``skip_corrections`` will be set to ``True``.
                See https://doped.readthedocs.io/en/latest/GGA_workflow_tutorial.html#dielectric-constant
                for information on calculating and converging the dielectric
                constant.
            charge_state (int):
                Charge state of defect. If not provided, will be automatically
                determined from defect calculation outputs, or if that fails,
                using the defect folder name (must end in "_+X" or "_-X" where
                +/-X is the defect charge state).
            skip_corrections (bool):
                Whether to skip the calculation and application of finite-size
                charge corrections to the defect energy (not recommended in
                most cases). Default = ``False``.
            error_tolerance (float):
                If the estimated error in the defect charge correction, based
                on the variance of the potential in the sampling region, is
                greater than this value (in eV), then a warning is raised.
                Default is 0.05 eV.
            bulk_band_gap_vr (PathLike or Vasprun):
                Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen``
                ``Vasprun`` object, from which to determine the bulk band gap
                and band edge positions. If the VBM/CBM occur at `k`-points
                which are not included in the bulk supercell calculation, then
                this parameter should be used to provide the output of a bulk
                bandstructure calculation so that these are correctly
                determined. Alternatively, you can edit the ``"band_gap"`` and
                ``"vbm"`` entries in ``self.defect_entry.calculation_metadata``
                to match the correct (eigen)values.
                If ``None``, will use
                ``DefectEntry.calculation_metadata["bulk_path"]`` (i.e. the
                bulk supercell calculation output).

                Note that the ``"band_gap"`` and ``"vbm"`` values should only
                affect the reference for the Fermi level values output by
                ``doped`` (as this VBM eigenvalue is used as the zero
                reference), thus affecting the position of the band edges in
                the defect formation energy plots and doping window /
                dopability limit functions, and the reference of the reported
                Fermi levels.
            parse_projected_eigen (bool):
                Whether to parse the projected eigenvalues & magnetization from
                the bulk and defect calculations (so
                ``DefectEntry.get_eigenvalue_analysis()`` can then be used with
                no further parsing, and magnetization values can be pulled for
                SOC / non-collinear magnetism calculations). Will initially try
                to load orbital projections from ``vasprun.xml(.gz)`` files
                (slightly slower but more accurate), or failing that from
                ``PROCAR(.gz)`` files if present in the bulk/defect
                directories. Parsing this data can increase total parsing time
                by anywhere from ~5-25%, so set to ``False`` if parsing speed
                is crucial.
                Default is ``None``, which will attempt to load this data but
                with no warning if it fails (otherwise if ``True`` a warning
                will be printed).
            **kwargs:
                Keyword arguments to pass to ``DefectParser()`` methods
                (``load_FNV_data()``, ``load_eFNV_data()``,
                ``load_bulk_gap_data()``),
                ``point_symmetry_from_defect_entry()`` or
                ``defect_and_info_from_structures``, including
                ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
                ``mpid``, ``api_key``, ``oxi_state``, ``multiplicity``,
                ``angle_tolerance``, ``user_charges``,
                ``initial_defect_structure_path`` etc (see their docstrings).
                Primarily used by ``DefectsParser`` to expedite parsing by
                avoiding reloading bulk data for each defect. Note that
                ``bulk_symprec`` can be supplied as the ``symprec`` value to
                use for determining equivalent sites (and thus defect
                multiplicities / unrelaxed site symmetries), while an input
                ``symprec`` value will be used for determining `relaxed` site
                symmetries.

        Return:
            ``DefectParser`` object.
        """
        _ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings

        calculation_metadata = {
            "bulk_path": os.path.abspath(bulk_path) if bulk_path else "bulk Vasprun supplied",
            "defect_path": os.path.abspath(defect_path),
        }

        if bulk_path is not None and bulk_vr is None:  # add bulk simple properties
            parsed_bulk_vasp_objs = _parse_vr_and_poss_procar(  # (bulk_vr, bulk_procar) if parse_procar
                output_path=bulk_path,  # else just bulk_vr
                parse_projected_eigen=parse_projected_eigen,
                label="bulk",
                parse_procar=bulk_procar is None,
            )
            bulk_vr, bulk_procar = (
                parsed_bulk_vasp_objs if len(parsed_bulk_vasp_objs) == 2 else (parsed_bulk_vasp_objs, None)
            )
            parse_projected_eigen = bulk_vr.projected_eigenvalues is not None or bulk_procar is not None

        elif bulk_vr is None:
            raise ValueError("Either `bulk_path` or `bulk_vr` must be provided!")
        bulk_supercell = bulk_vr.final_structure.copy()

        # add defect simple properties
        defect_vr, defect_procar = _parse_vr_and_poss_procar(
            defect_path, parse_projected_eigen=parse_projected_eigen, label="defect", parse_procar=True
        )
        parse_projected_eigen = defect_procar is not None or defect_vr.projected_eigenvalues is not None

        possible_defect_name = os.path.basename(
            defect_path.rstrip("/.").rstrip("/")  # remove any trailing slashes to ensure correct name
        )  # set equal to folder name
        if "vasp" in possible_defect_name:  # get parent directory name:
            possible_defect_name = os.path.basename(os.path.dirname(defect_path))

        try:
            parsed_charge_state: int = total_charge_from_vasprun(defect_vr, charge_state)
        except RuntimeError as orig_exc:  # auto charge guessing failed and charge_state not provided,
            # try to determine from folder name -- must have "-" or "+" at end of name for this
            try:
                charge_state_suffix = possible_defect_name.rsplit("_", 1)[-1]
                if charge_state_suffix[0] not in ["-", "+"]:
                    raise ValueError(
                        f"Could not guess charge state from folder name ({possible_defect_name}), must "
                        f"end in '_+X' or '_-X' where +/-X is the charge state."
                    )

                parsed_charge_state = int(charge_state_suffix)
                if abs(parsed_charge_state) >= 7:
                    raise ValueError(
                        f"Guessed charge state from folder name was {parsed_charge_state:+} which is "
                        f"almost certainly unphysical"
                    )
            except Exception as next_exc:
                raise orig_exc from next_exc

        # parse spin degeneracy now, before proj eigenvalues/magnetization are cut (for SOC/NCL calcs):
        degeneracy_factors = {
            "spin degeneracy": spin_degeneracy_from_vasprun(defect_vr, charge_state=parsed_charge_state)
            / spin_degeneracy_from_vasprun(bulk_vr, charge_state=0)
        }

        if dielectric is None and not skip_corrections and parsed_charge_state != 0:
            warnings.warn(
                "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
                "corrections, but none was provided, so charge corrections will be skipped "
                "(`skip_corrections = True`). Formation energies and transition levels of charged "
                "defects will likely be very inaccurate without charge corrections!"
            )
            skip_corrections = True

        if dielectric is not None:
            dielectric = _convert_dielectric_to_tensor(dielectric)
            calculation_metadata["dielectric"] = dielectric

        # Add defect structure to calculation_metadata, so it can be pulled later on (e.g. for eFNV)
        defect_structure = defect_vr.final_structure.copy()
        calculation_metadata["defect_structure"] = defect_structure

        # check if the bulk and defect supercells are the same size:
        if not np.isclose(defect_structure.volume, bulk_supercell.volume, rtol=1e-2):
            warnings.warn(
                f"The defect and bulk supercells are not the same size, having volumes of "
                f"{defect_structure.volume:.1f} and {bulk_supercell.volume:.1f} Å^3 respectively. This "
                f"may cause errors in parsing and/or output energies. In most cases (unless looking at "
                f"extremely high doping concentrations) the same fixed supercell (ISIF = 2) should be "
                f"used for both the defect and bulk calculations! (i.e. assuming the dilute limit)"
            )

        (
            defect,
            defect_site,
            defect_structure_metadata,
        ) = defect_and_info_from_structures(
            bulk_supercell,
            defect_structure.copy(),
            **{
                k.replace("bulk_", ""): v
                for k, v in kwargs.items()
                if k
                in [
                    "oxi_state",
                    "multiplicity",
                    "symprec",
                    "bulk_symprec",  # for interstitial multiplicities; changed to "symprec"
                    "dist_tol_factor",  # for interstitial multiplicities
                    "angle_tolerance",
                    "user_charges",
                    "initial_defect_structure_path",
                    "fixed_symprec_and_dist_tol_factor",
                    "verbose",
                ]
            },
        )
        calculation_metadata.update(defect_structure_metadata)  # add defect structure metadata

        # ComputedEntry.parameters keys have random order when using Vasprun.get_computed_entry(), which is
        # fine but shows file differences in git diffs, so sort them to avoid this (just for easier
        # tracking for SK, allow it fam)
        sc_entry = defect_vr.get_computed_entry()
        bulk_entry = bulk_vr.get_computed_entry()
        for computed_entry in [sc_entry, bulk_entry]:
            computed_entry.parameters = dict(sorted(computed_entry.parameters.items()))

        defect_entry = DefectEntry(
            # pmg attributes:
            defect=defect,  # this corresponds to _unrelaxed_ defect
            charge_state=parsed_charge_state,
            sc_entry=sc_entry,
            sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
            bulk_entry=bulk_entry,
            # doped attributes:
            name=possible_defect_name,  # set later, so set now to avoid guessing in ``__post_init__()``
            defect_supercell_site=defect_site,  # _relaxed_ defect site
            defect_supercell=defect_vr.final_structure,
            bulk_supercell=bulk_vr.final_structure,
            calculation_metadata=calculation_metadata,
            degeneracy_factors=degeneracy_factors,
        )
        # get orientational degeneracy
        point_symm_and_periodicity_breaking = point_symmetry_from_defect_entry(
            defect_entry,
            relaxed=True,
            verbose=kwargs.get("verbose", False),
            return_periodicity_breaking=True,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["symprec", "dist_tol_factor", "fixed_symprec_and_dist_tol_factor"]
            },
        )
        assert isinstance(point_symm_and_periodicity_breaking, tuple)  # typing (tuple returned)
        relaxed_point_group, periodicity_breaking = point_symm_and_periodicity_breaking
        bulk_site_point_group = point_symmetry_from_defect_entry(
            defect_entry,
            relaxed=False,
            **{
                k.replace("bulk_", ""): v
                for k, v in kwargs.items()
                if k in ["bulk_symprec", "dist_tol_factor", "fixed_symprec_and_dist_tol_factor", "verbose"]
            },
        )  # same symprec used w/interstitial multiplicity for consistency
        assert isinstance(bulk_site_point_group, str)  # typing (str returned)
        with contextlib.suppress(ValueError):
            defect_entry.degeneracy_factors["orientational degeneracy"] = get_orientational_degeneracy(
                relaxed_point_group=relaxed_point_group,
                bulk_site_point_group=bulk_site_point_group,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in [
                        "symprec",
                        "bulk_symprec",
                        "dist_tol_factor",
                        "fixed_symprec_and_dist_tol_factor",
                        "verbose",
                    ]
                },
            )
        defect_entry.calculation_metadata["relaxed point symmetry"] = relaxed_point_group
        defect_entry.calculation_metadata["bulk site symmetry"] = bulk_site_point_group
        defect_entry.calculation_metadata["periodicity_breaking_supercell"] = periodicity_breaking

        check_and_set_defect_entry_name(defect_entry, possible_defect_name)

        dp = cls(
            defect_entry,
            defect_vr=defect_vr,
            bulk_vr=bulk_vr,
            skip_corrections=skip_corrections,
            error_tolerance=error_tolerance,
            parse_projected_eigen=parse_projected_eigen,
            **kwargs,
        )

        if parse_projected_eigen is not False:
            try:
                dp.defect_entry._load_and_parse_eigenvalue_data(
                    bulk_vr=bulk_vr,
                    bulk_procar=bulk_procar,
                    defect_vr=defect_vr,
                    defect_procar=defect_procar,
                )
            except Exception as exc:
                if parse_projected_eigen is True:  # otherwise no warning
                    warnings.warn(f"Projected eigenvalues/orbitals parsing failed with error: {exc!r}")

                # these are removed in _load_and_parse_eigenvalue_data, but in case it fails:
                defect_vr.projected_eigenvalues = None  # no longer needed, delete to reduce memory demand
                defect_vr.projected_magnetisation = (
                    None  # no longer needed, delete to reduce memory demand
                )
                defect_vr.eigenvalues = None  # no longer needed, delete to reduce memory demand

        dp.load_and_check_calculation_metadata()  # Load standard defect metadata
        dp.load_bulk_gap_data(bulk_band_gap_vr=bulk_band_gap_vr)  # Load band gap data

        if not skip_corrections and defect_entry.charge_state != 0:
            # no finite-size charge corrections by default for neutral defects
            skip_corrections = dp._check_and_load_appropriate_charge_correction()

        if not skip_corrections and defect_entry.charge_state != 0:
            try:
                dp.apply_corrections()
            except Exception as exc:
                warnings.warn(
                    f"Got this error message when attempting to apply finite-size charge corrections:"
                    f"\n{exc}\n"
                    f"-> Charge corrections will not be applied for this defect."
                )

            # check that charge corrections are not negative
            summed_corrections = sum(
                val
                for key, val in dp.defect_entry.corrections.items()
                if any(i in key.lower() for i in ["freysoldt", "kumagai", "fnv", "charge"])
            )
            if summed_corrections < -0.08:
                # usually unphysical for _isotropic_ dielectrics (suggests over-delocalised charge,
                # affecting the potential alignment)
                # how anisotropic is the dielectric?
                how_aniso = np.diag(
                    (dielectric - np.mean(np.diag(dielectric))) / np.mean(np.diag(dielectric))
                )
                if np.allclose(how_aniso, 0, atol=0.05):
                    warnings.warn(
                        f"The calculated finite-size charge corrections for defect at {defect_path} and "
                        f"bulk at {bulk_path} sum to a _negative_ value of {summed_corrections:.3f}. For "
                        f"relatively isotropic dielectrics (as is the case here) this is usually "
                        f"unphyical, and can indicate 'false charge state' behaviour (with the supercell "
                        f"charge occupying the band edge states and not localised at the defect), "
                        f"affecting the potential alignment, or some error/mismatch in the defect and "
                        f"bulk calculations. If this defect species is not stable in the formation "
                        f"energy diagram then this warning can usually be ignored, but if it is, "
                        f"you should double-check your calculations and parsed results!"
                    )

        return dp

    def _check_and_load_appropriate_charge_correction(self):
        skip_corrections = False
        dielectric = self.defect_entry.calculation_metadata["dielectric"]
        bulk_path = self.defect_entry.calculation_metadata["bulk_path"]
        defect_path = self.defect_entry.calculation_metadata["defect_path"]

        # determine charge correction to use, based on what output files are available (`LOCPOT`s or
        # `OUTCAR`s), and whether the supplied dielectric is isotropic or not
        def _check_folder_for_file_match(folder, filename):
            return any(
                filename.lower() in folder_filename.lower() for folder_filename in os.listdir(folder)
            )

        # check if dielectric (3x3 matrix) has diagonal elements that differ by more than 20%
        isotropic_dielectric = all(np.isclose(i, dielectric[0, 0], rtol=0.2) for i in np.diag(dielectric))

        # regardless, try parsing OUTCAR files first (quickest, more robust for cases where defect
        # charge is localised somewhat off the (auto-determined) defect site (e.g. split-interstitials
        # etc) and also works regardless of isotropic/anisotropic)
        if _check_folder_for_file_match(defect_path, "OUTCAR") and _check_folder_for_file_match(
            bulk_path, "OUTCAR"
        ):
            try:
                self.load_eFNV_data()
            except Exception as kumagai_exc:
                if _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
                    bulk_path, "LOCPOT"
                ):
                    try:
                        if not isotropic_dielectric:
                            # convert anisotropic dielectric to harmonic mean of the diagonal:
                            # (this is a better approximation than the pymatgen default of the
                            # standard arithmetic mean of the diagonal)
                            self.defect_entry.calculation_metadata["dielectric"] = (
                                _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                            )
                        self.load_FNV_data()
                        if not isotropic_dielectric:
                            warnings.warn(
                                _aniso_dielectric_but_outcar_problem_warning
                                + "in the defect or bulk folder were unable to be parsed, giving the "
                                "following error message:"
                                + f"\n{kumagai_exc}\n"
                                + _aniso_dielectric_but_using_locpot_warning
                            )
                    except Exception as freysoldt_exc:
                        warnings.warn(
                            f"Got this error message when attempting to parse defect & bulk `OUTCAR` "
                            f"files to compute the Kumagai (eFNV) charge correction:"
                            f"\n{kumagai_exc}\n"
                            f"Then got this error message when attempting to parse defect & bulk "
                            f"`LOCPOT` files to compute the Freysoldt (FNV) charge correction:"
                            f"\n{freysoldt_exc}\n"
                            f"-> Charge corrections will not be applied for this defect."
                        )
                        if not isotropic_dielectric:
                            # reset dielectric to original anisotropic value if FNV failed as well:
                            self.defect_entry.calculation_metadata["dielectric"] = dielectric
                        skip_corrections = True

                else:
                    warnings.warn(
                        f"`OUTCAR` files (needed to compute the Kumagai eFNV charge correction for "
                        f"_anisotropic_ and isotropic systems) in the defect or bulk folder were unable "
                        f"to be parsed, giving the following error message:"
                        f"\n{kumagai_exc}\n"
                        f"-> Charge corrections will not be applied for this defect."
                    )
                    skip_corrections = True

        elif _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
            bulk_path, "LOCPOT"
        ):
            try:
                if not isotropic_dielectric:
                    # convert anisotropic dielectric to harmonic mean of the diagonal:
                    # (this is a better approximation than the pymatgen default of the
                    # standard arithmetic mean of the diagonal)
                    self.defect_entry.calculation_metadata["dielectric"] = (
                        _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                    )
                self.load_FNV_data()
                if not isotropic_dielectric:
                    warnings.warn(
                        _aniso_dielectric_but_outcar_problem_warning
                        + "are missing from the defect or bulk folder.\n"
                        + _aniso_dielectric_but_using_locpot_warning
                    )
            except Exception as freysoldt_exc:
                warnings.warn(
                    f"Got this error message when attempting to parse defect & bulk `LOCPOT` files to "
                    f"compute the Freysoldt (FNV) charge correction:"
                    f"\n{freysoldt_exc}\n"
                    f"-> Charge corrections will not be applied for this defect."
                )
                if not isotropic_dielectric:
                    # reset dielectric to original anisotropic value if FNV failed as well:
                    self.defect_entry.calculation_metadata["dielectric"] = dielectric
                skip_corrections = True

        else:
            if int(self.defect_entry.charge_state) != 0:
                warnings.warn(
                    "`LOCPOT` or `OUTCAR` files are missing from the defect or bulk folder. "
                    "These are needed to perform the finite-size charge corrections. "
                    "Charge corrections will not be applied for this defect."
                )
                skip_corrections = True

        return skip_corrections

    def load_FNV_data(self, bulk_locpot_dict: dict | None = None):
        """
        Load metadata required for performing Freysoldt correction (i.e.
        ``LOCPOT`` planar-averaged potential dictionary).

        Requires "bulk_path" and "defect_path" to be present in
        ``DefectEntry.calculation_metadata``, and VASP ``LOCPOT`` files to be
        present in these directories. Can read compressed "LOCPOT.gz" files.
        The ``bulk_locpot_dict`` can be supplied if already parsed, for
        expedited parsing of multiple defects.

        Saves the ``bulk_locpot_dict`` and ``defect_locpot_dict`` dictionaries
        (containing the planar-averaged electrostatic potentials along each
        axis direction) to the ``DefectEntry.calculation_metadata`` dict, for
        use with ``DefectEntry.get_freysoldt_correction()``.

        Args:
            bulk_locpot_dict (dict):
                Planar-averaged potential dictionary for bulk supercell, if
                already parsed. If ``None`` (default), will try to load from
                the ``LOCPOT(.gz)`` file in
                ``defect_entry.calculation_metadata["bulk_path"]``.

        Returns:
            ``bulk_locpot_dict`` for reuse in parsing other defect entries.
        """
        if not self.defect_entry.charge_state:
            # no charge correction if charge is zero
            return None

        bulk_locpot_dict = (
            bulk_locpot_dict
            or self.kwargs.get("bulk_locpot_dict", None)
            or _get_bulk_locpot_dict(self.defect_entry.calculation_metadata["bulk_path"])
        )

        defect_locpot_path, multiple = _get_output_files_and_check_if_multiple(
            "LOCPOT", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            _multiple_files_warning(
                "LOCPOT",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_locpot_path,
                dir_type="defect",
            )
        defect_locpot = get_locpot(defect_locpot_path)
        defect_locpot_dict = {str(k): defect_locpot.get_average_along_axis(k) for k in [0, 1, 2]}

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_locpot_dict": bulk_locpot_dict,
                "defect_locpot_dict": defect_locpot_dict,
            }
        )

        return bulk_locpot_dict

    def load_eFNV_data(self, bulk_site_potentials: list | None = None):
        """
        Load metadata required for performing Kumagai correction (i.e. atomic
        site potentials from the ``OUTCAR`` files).

        Requires "bulk_path" and "defect_path" to be present in
        ``DefectEntry.calculation_metadata``, and ``VASP`` ``OUTCAR`` files to
        be present in these directories. Can read compressed ``OUTCAR.gz``
        files. The bulk_site_potentials can be supplied if already parsed, for
        expedited parsing of multiple defects.

        Saves the ``bulk_site_potentials`` and ``defect_site_potentials`` lists
        (containing the atomic site electrostatic potentials, from
        ``-1*np.array(Outcar.electrostatic_potential)``) to
        ``DefectEntry.calculation_metadata``, for use with
        ``DefectEntry.get_kumagai_correction()``.

        Args:
            bulk_site_potentials (list):
                Atomic site potentials for the bulk supercell, if already
                parsed. If ``None`` (default), will load from ``OUTCAR(.gz)``
                file in ``defect_entry.calculation_metadata["bulk_path"]``.

        Returns:
            ``bulk_site_potentials`` to reuse in parsing other defect entries.
        """
        if not self.defect_entry.charge_state:
            # don't need to load outcars if charge is zero
            return None

        bulk_site_potentials = bulk_site_potentials or self.kwargs.get("bulk_site_potentials", None)

        def _get_total_energies(computed_entry=None, vr=None):
            """
            Get the total energies from the defect entry or vasprun.
            """
            energies = [
                (
                    computed_entry.energy
                    if computed_entry
                    else None(vr.ionic_steps[-1]["electronic_steps"][-1]["e_0_energy"] if vr else None)
                ),
            ]
            return [energy for energy in energies if energy is not None]

        if bulk_site_potentials is None:
            bulk_site_potentials = _get_bulk_site_potentials(
                self.defect_entry.calculation_metadata["bulk_path"],
                total_energy=_get_total_energies(self.defect_entry.bulk_entry, self.bulk_vr),
            )

        defect_outcar_path, multiple = _get_output_files_and_check_if_multiple(
            "OUTCAR", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            _multiple_files_warning(
                "OUTCAR",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_outcar_path,
                dir_type="defect",
            )
        defect_site_potentials = get_core_potentials_from_outcar(
            defect_outcar_path,
            dir_type="defect",
            total_energy=_get_total_energies(self.defect_entry.sc_entry, self.defect_vr),
        )

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_site_potentials": bulk_site_potentials,
                "defect_site_potentials": defect_site_potentials,
            }
        )

        return bulk_site_potentials

    def load_and_check_calculation_metadata(self):
        """
        Pull metadata about the defect supercell calculations from the outputs,
        and check if the defect and bulk supercell calculations settings are
        compatible.
        """
        for attr in ["bulk_vr", "defect_vr"]:
            if not getattr(self, attr, None):
                label = attr.split("_")[0]  # "bulk" or "defect"
                setattr(
                    self,
                    attr,
                    _parse_vr_and_poss_procar(
                        output_path=self.defect_entry.calculation_metadata[f"{label}_path"],
                        parse_projected_eigen=False,  # not needed for DefectEntry metadata
                        label=label,  # "bulk" or "defect"
                        parse_procar=False,
                    ),
                )

        def _get_vr_dict_without_proj_eigenvalues(vr):
            attributes_to_cut = ["projected_eigenvalues", "projected_magnetisation"]
            orig_values = {}
            for attribute in attributes_to_cut:
                orig_values[attribute] = getattr(vr, attribute)
                setattr(vr, attribute, None)

            vr_dict = vr.as_dict()  # only call once
            vr_dict_wout_proj = {  # projected eigenvalue data might be present, but not needed (v slow
                # and data-heavy)
                **{k: v for k, v in vr_dict.items() if k != "output"},
                "output": {k: v for k, v in vr_dict["output"].items() if k not in attributes_to_cut},
            }
            for attribute in attributes_to_cut:
                vr_dict_wout_proj["output"][attribute] = None
                setattr(vr, attribute, orig_values[attribute])  # reset to original value

            return vr_dict_wout_proj

        run_metadata = {
            # incars need to be as dict without module keys otherwise not JSONable:
            "defect_incar": {k: v for k, v in self.defect_vr.incar.as_dict().items() if "@" not in k},
            "bulk_incar": {k: v for k, v in self.bulk_vr.incar.as_dict().items() if "@" not in k},
            "defect_kpoints": self.defect_vr.kpoints,
            "bulk_kpoints": self.bulk_vr.kpoints,
            "defect_actual_kpoints": self.defect_vr.actual_kpoints,
            "bulk_actual_kpoints": self.bulk_vr.actual_kpoints,
            "defect_potcar_symbols": self.defect_vr.potcar_spec,
            "bulk_potcar_symbols": self.bulk_vr.potcar_spec,
            "defect_vasprun_dict": _get_vr_dict_without_proj_eigenvalues(self.defect_vr),
            "bulk_vasprun_dict": _get_vr_dict_without_proj_eigenvalues(self.bulk_vr),
        }

        incar_mismatches = _compare_incar_tags(
            run_metadata["bulk_incar"],
            run_metadata["defect_incar"],
        )
        self.defect_entry.calculation_metadata["mismatching_INCAR_tags"] = (
            incar_mismatches if not (isinstance(incar_mismatches, bool)) else False
        )
        potcar_mismatches = _compare_potcar_symbols(
            run_metadata["bulk_potcar_symbols"],
            run_metadata["defect_potcar_symbols"],
        )
        self.defect_entry.calculation_metadata["mismatching_POTCAR_symbols"] = (
            potcar_mismatches if not (isinstance(potcar_mismatches, bool)) else False
        )
        kpoint_mismatches = _compare_kpoints(
            run_metadata["bulk_actual_kpoints"],
            run_metadata["defect_actual_kpoints"],
            run_metadata["bulk_kpoints"],
            run_metadata["defect_kpoints"],
        )
        self.defect_entry.calculation_metadata["mismatching_KPOINTS"] = (
            kpoint_mismatches if not (isinstance(kpoint_mismatches, bool)) else False
        )
        self.defect_entry.calculation_metadata.update({"run_metadata": run_metadata.copy()})

    def load_bulk_gap_data(
        self,
        bulk_band_gap_vr: PathLike | Vasprun | None = None,
        use_MP: bool = False,
        mpid: str | None = None,
        api_key: str | None = None,
    ):
        r"""
        Load the ``"band_gap"``, ``"vbm"`` and ``"cbm"`` values for the parsed
        ``DefectEntry``\s.

        If ``bulk_band_gap_vr`` is provided, then these values are parsed from
        it, else taken from the parsed bulk supercell calculation.

        ``"band_gap"`` and ``"vbm"`` are used by default when generating
        ``DefectThermodynamics`` objects, to be used in plotting & analysis.

        Alternatively, one can specify query the Materials Project (MP)
        database for the bulk gap data, using ``use_MP = True``, in which case
        the MP entry with the lowest number ID and composition matching the
        bulk will be used, or the MP ID (``mpid``) of the bulk material to use
        can be specified. This is not recommended as it will correspond to a
        severely-underestimated GGA DFT bandgap!

        Args:
            bulk_band_gap_vr (PathLike or Vasprun):
                Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen``
                ``Vasprun`` object, from which to determine the bulk band gap
                and band edge positions. If the VBM/CBM occur at `k`-points
                which are not included in the bulk supercell calculation, then
                this parameter should be used to provide the output of a bulk
                bandstructure calculation so that these are correctly
                determined. Alternatively, you can edit the ``"band_gap"`` and
                ``"vbm"`` entries in ``self.defect_entry.calculation_metadata``
                to match the correct (eigen)values.
                If ``None``, will use
                ``DefectEntry.calculation_metadata["bulk_path"]``
                (i.e. the bulk supercell calculation output).

                Note that the ``"band_gap"`` and ``"vbm"`` values should only
                affect the reference for the Fermi level values output by
                ``doped`` (as this VBM eigenvalue is used as the zero
                reference), thus affecting the position of the band edges in
                the defect formation energy plots and doping window /
                dopability limit functions, and the reference of the reported
                Fermi levels.
            use_MP (bool):
                If True, will query the Materials Project database for the bulk
                gap data.
            mpid (str):
                If provided, will query the Materials Project database for the
                bulk gap data, using this Materials Project ID.
            api_key (str):
                Materials API key to access database.
        """
        if not self.bulk_vr:
            self.bulk_vr = _parse_vr_and_poss_procar(
                output_path=self.defect_entry.calculation_metadata["bulk_path"],
                parse_projected_eigen=self.parse_projected_eigen,
                label="bulk",
                parse_procar=False,
            )

        bulk_sc_structure = self.bulk_vr.initial_structure
        band_gap, cbm, vbm, _ = self.bulk_vr.eigenvalue_band_properties
        gap_calculation_metadata = {}

        use_MP = use_MP or self.kwargs.get("use_MP", False)
        mpid = mpid or self.kwargs.get("mpid", None)
        api_key = api_key or self.kwargs.get("api_key", None)

        if use_MP and mpid is None:
            try:
                with MPRester(api_key=api_key) as mpr:
                    tmp_mplist = mpr.get_entries_in_chemsys(list(bulk_sc_structure.symbol_set))
                mplist = [
                    mp_ent.entry_id
                    for mp_ent in tmp_mplist
                    if mp_ent.composition.reduced_composition
                    == bulk_sc_structure.composition.reduced_composition
                ]
            except Exception as exc:
                raise ValueError(
                    f"Error with querying MPRester for {bulk_sc_structure.composition.reduced_formula}:"
                ) from exc

            mpid_fit_list = []
            for trial_mpid in mplist:
                with MPRester(api_key=api_key) as mpr:
                    mpstruct = mpr.get_structure_by_material_id(trial_mpid)
                if StructureMatcher_scan_stol(
                    bulk_sc_structure,
                    mpstruct,
                    func_name="fit",
                    primitive_cell=True,
                    scale=False,
                    attempt_supercell=True,
                    allow_subset=False,
                ):
                    mpid_fit_list.append(trial_mpid)

            if len(mpid_fit_list) == 1:
                mpid = mpid_fit_list[0]
                print(f"Single mp-id found for bulk structure:{mpid}.")
            elif len(mpid_fit_list) > 1:
                num_mpid_list = [int(mpid.split("-")[1]) for mpid in mpid_fit_list]
                num_mpid_list.sort()
                mpid = f"mp-{num_mpid_list[0]!s}"
                print(
                    f"Multiple mp-ids found for bulk structure:{mpid_fit_list}. Will use lowest "
                    f"number mpid for bulk band structure = {mpid}."
                )
            else:
                print(
                    "Could not find bulk structure in MP database after tying the following "
                    f"list:\n{mplist}"
                )
                mpid = None

        if mpid is not None:
            print(f"Using user-provided mp-id for bulk structure: {mpid}.")
            with MPRester(api_key=api_key) as mpr:
                bs = mpr.get_bandstructure_by_material_id(mpid)
            if bs:
                cbm = bs.get_cbm()["energy"]
                vbm = bs.get_vbm()["energy"]
                band_gap = bs.get_band_gap()["energy"]
                gap_calculation_metadata["MP_gga_BScalc_data"] = bs.get_band_gap().copy()

        if (vbm is None or band_gap is None or cbm is None or not bulk_band_gap_vr) and (
            mpid and band_gap is None
        ):
            warnings.warn(
                f"MPID {mpid} was provided, but no bandstructure entry currently exists for it. "
                f"Reverting to use of bulk supercell calculation for band edge extrema."
            )
            gap_calculation_metadata["MP_gga_BScalc_data"] = None  # to signal no MP BS is used

        if bulk_band_gap_vr:
            if not isinstance(bulk_band_gap_vr, Vasprun):
                bulk_band_gap_vr = get_vasprun(bulk_band_gap_vr, parse_projected_eigen=False)

            band_gap, cbm, vbm, _ = bulk_band_gap_vr.eigenvalue_band_properties

        gap_calculation_metadata.update(
            {
                "cbm": cbm,
                "vbm": vbm,
                "band_gap": band_gap,
            }
        )
        if mpid is not None:
            gap_calculation_metadata["mpid"] = mpid

        self.defect_entry.calculation_metadata.update(gap_calculation_metadata)

    def apply_corrections(self):
        """
        Get and apply defect corrections, and warn if likely to be
        inappropriate.
        """
        if not self.defect_entry.charge_state:  # no charge correction if charge is zero
            return

        # try run Kumagai (eFNV) correction if required info available:
        if (
            self.defect_entry.calculation_metadata.get("bulk_site_potentials", None) is not None
            and self.defect_entry.calculation_metadata.get("defect_site_potentials", None) is not None
        ):
            self.defect_entry.get_kumagai_correction(verbose=False, error_tolerance=self.error_tolerance)

        elif self.defect_entry.calculation_metadata.get(
            "bulk_locpot_dict"
        ) and self.defect_entry.calculation_metadata.get("defect_locpot_dict"):
            self.defect_entry.get_freysoldt_correction(verbose=False, error_tolerance=self.error_tolerance)

        else:
            raise ValueError(
                "No charge correction performed! Missing required metadata in "
                "defect_entry.calculation_metadata ('bulk/defect_site_potentials' for Kumagai ("
                "eFNV) correction, or 'bulk/defect_locpot_dict' for Freysoldt (FNV) correction) -- these "
                "are loaded with either the load_eFNV_data() or load_FNV_data() methods for "
                "DefectParser."
            )

        if (
            self.defect_entry.charge_state != 0
            and (not self.defect_entry.corrections or sum(self.defect_entry.corrections.values())) == 0
        ):
            warnings.warn(
                f"No charge correction computed for {self.defect_entry.name} with charge"
                f" {self.defect_entry.charge_state:+}, indicating problems with the required data for "
                f"the charge correction (i.e. dielectric constant, LOCPOT files for Freysoldt "
                f"correction, OUTCAR (with ICORELEVEL = 0) for Kumagai correction etc)."
            )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectParser`` object.
        """
        formula = self.bulk_vr.final_structure.composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectParser for bulk composition {formula}. "
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )
