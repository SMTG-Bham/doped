"""
Helper functions for parsing VASP supercell defect calculations.
"""
import os
import warnings

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Vasprun
from pymatgen.util.coord import pbc_diff

from doped import _ignore_pmg_warnings


def find_archived_fname(fname, raise_error=True):
    """
    Find a suitable filename, taking account of possible use of compression
    software.
    """
    if os.path.exists(fname):
        return fname
    # Check for archive files
    for ext in [".gz", ".xz", ".bz", ".lzma"]:
        if os.path.exists(fname + ext):
            return fname + ext
    if raise_error:
        raise FileNotFoundError
    return None


def get_vasprun(vasprun_path, **kwargs):
    """
    Read the vasprun.xml(.gz) file as a pymatgen Vasprun object.
    """
    vasprun_path = str(vasprun_path)  # convert to string if Path object
    warnings.filterwarnings(
        "ignore", category=UnknownPotcarWarning
    )  # Ignore unknown POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings(
        "ignore", message="No POTCAR file with matching TITEL fields"
    )  # `message` only needs to match start of message
    try:
        vasprun = Vasprun(find_archived_fname(vasprun_path), **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"vasprun.xml or compressed version (.gz/.xz/.bz/.lzma) not found at {vasprun_path}("
            f".gz/.xz/.bz/.lzma). Needed for parsing calculation output!"
        ) from None
    return vasprun


def get_locpot(locpot_path):
    """
    Read the LOCPOT(.gz) file as a pymatgen Locpot object.
    """
    locpot_path = str(locpot_path)  # convert to string if Path object
    try:
        locpot = Locpot.from_file(find_archived_fname(locpot_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LOCPOT or compressed version not found at (.gz/.xz/.bz/.lzma) not found at {locpot_path}("
            f".gz/.xz/.bz/.lzma). Needed for calculating the Freysoldt (FNV) image charge correction!"
        ) from None
    return locpot


def get_outcar(outcar_path):
    """
    Read the OUTCAR(.gz) file as a pymatgen Outcar object.
    """
    outcar_path = str(outcar_path)  # convert to string if Path object
    try:
        outcar = Outcar(find_archived_fname(outcar_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"OUTCAR file not found at {outcar_path}. Needed for calculating the Kumagai (eFNV) "
            f"image charge correction."
        ) from None
    return outcar


def _get_output_files_and_check_if_multiple(output_file="vasprun.xml", path="."):
    """
    Search for all files with filenames matching `output_file`, case-
    insensitive.

    Returns (output file path, Multiple?) where Multiple is True if multiple
    matching files are found.
    """
    files = os.listdir(path)
    output_files = [filename for filename in files if output_file.lower() in filename.lower()]
    # sort by direct match to {output_file}, direct match to {output_file}.gz, then alphabetically:
    if output_files := sorted(
        output_files,
        key=lambda x: (x == output_file, x == f"{output_file}.gz", x),
        reverse=True,
    ):
        output_path = os.path.join(path, output_files[0])
        return (output_path, True) if len(output_files) > 1 else (output_path, False)
    return path, False  # so when `get_X()` is called, it will raise an informative FileNotFoundError


def get_defect_type_and_composition_diff(bulk, defect):
    """
    Get the difference in composition between a bulk structure and a defect
    structure.

    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for
    extrinsic species and code efficiency/robustness improvements.
    """
    bulk_comp = bulk.composition.get_el_amt_dict()
    defect_comp = defect.composition.get_el_amt_dict()

    composition_diff = {
        element: int(defect_amount - bulk_comp.get(element, 0))
        for element, defect_amount in defect_comp.items()
        if int(defect_amount - bulk_comp.get(element, 0)) != 0
    }

    if len(composition_diff) == 1 and list(composition_diff.values())[0] == 1:
        defect_type = "interstitial"
    elif len(composition_diff) == 1 and list(composition_diff.values())[0] == -1:
        defect_type = "vacancy"
    elif len(composition_diff) == 2:
        defect_type = "substitution"
    else:
        raise RuntimeError(
            "Could not determine defect type from composition difference of bulk and defect structures."
        )

    return defect_type, composition_diff


def get_defect_site_idxs_and_unrelaxed_structure(
    bulk, defect, defect_type, composition_diff, unique_tolerance=1
):
    """
    Get the defect site and unrelaxed structure, where "unrelaxed structure"
    corresponds to the pristine defect supercell structure for
    vacancies/substitutions, and the pristine bulk structure with the _final_
    relaxed interstitial site for interstitials.

    Initially contributed by Dr. Alex Ganose (@ Imperial Chemistry) and
    refactored for extrinsic species and code efficiency/robustness improvements.

    Returns:
        bulk_site_idx: index of the site in the bulk structure that corresponds
            to the defect site in the defect structure
        defect_site_idx: index of the defect site in the defect structure
        unrelaxed_defect_structure: pristine defect supercell structure for
            vacancies/substitutions (i.e. pristine bulk with unrelaxed vacancy/
            substitution), or the pristine bulk structure with the _final_
            relaxed interstitial site for interstitials.
    """

    def process_substitution(bulk, defect, composition_diff):
        old_species = _get_species_from_composition_diff(composition_diff, -1)
        new_species = _get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, _bulk_new_species_idx = get_coords_and_idx_of_species(bulk, new_species)
        defect_new_species_coords, defect_new_species_idx = get_coords_and_idx_of_species(
            defect, new_species
        )

        if bulk_new_species_coords.size > 0:  # intrinsic substitution
            # find coords of new species in defect structure, taking into account periodic boundaries
            defect_site_arg_idx = find_nearest_coords(
                bulk_new_species_coords[:, None],
                defect_new_species_coords,
                bulk.lattice.matrix,
                defect_type="substitution",
                searched_structure="defect",
            )

        else:  # extrinsic substitution
            defect_site_arg_idx = 0

        # Get the coords and site index of the defect that was used in the VASP calculation
        defect_coords = defect_new_species_coords[defect_site_arg_idx]  # frac coords of defect site
        defect_site_idx = defect_new_species_idx[defect_site_arg_idx]

        # now find the closest old_species site in the bulk structure to the defect site
        # again, make sure to use periodic boundaries
        bulk_old_species_coords, _bulk_old_species_idx = get_coords_and_idx_of_species(bulk, old_species)

        bulk_site_arg_idx = find_nearest_coords(
            bulk_old_species_coords,
            defect_coords,
            bulk.lattice.matrix,
            defect_type="substitution",
            searched_structure="bulk",
        )

        # currently, original_site_idx is indexed with respect to the old species only.
        # need to get the index in the full structure:
        unrelaxed_defect_structure, bulk_site_idx = _remove_and_insert_species_from_bulk(
            bulk,
            bulk_old_species_coords,
            bulk_site_arg_idx,
            new_species,
            defect_site_idx,
            defect_type="substitution",
            searched_structure="bulk",
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    def process_vacancy(bulk, defect, composition_diff):
        old_species = _get_species_from_composition_diff(composition_diff, -1)
        bulk_old_species_coords, _bulk_old_species_idx = get_coords_and_idx_of_species(bulk, old_species)
        defect_old_species_coords, _defect_old_species_idx = get_coords_and_idx_of_species(
            defect, old_species
        )

        bulk_site_arg_idx = find_nearest_coords(
            bulk_old_species_coords[:, None],
            defect_old_species_coords,
            bulk.lattice.matrix,
            defect_type="vacancy",
            searched_structure="bulk",
        )

        # currently, original_site_idx is indexed with respect to the old species only.
        # need to get the index in the full structure:
        defect_site_idx = None
        unrelaxed_defect_structure, bulk_site_idx = _remove_and_insert_species_from_bulk(
            bulk,
            bulk_old_species_coords,
            bulk_site_arg_idx,
            new_species=None,
            defect_site_idx=defect_site_idx,
            defect_type="vacancy",
            searched_structure="bulk",
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    def process_interstitial(bulk, defect, composition_diff):
        new_species = _get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, _bulk_new_species_idx = get_coords_and_idx_of_species(bulk, new_species)
        defect_new_species_coords, defect_new_species_idx = get_coords_and_idx_of_species(
            defect, new_species
        )

        if bulk_new_species_coords.size > 0:  # intrinsic interstitial
            defect_site_arg_idx = find_nearest_coords(
                bulk_new_species_coords[:, None],
                defect_new_species_coords,
                bulk.lattice.matrix,
                defect_type="interstitial",
                searched_structure="defect",
            )

        else:  # extrinsic interstitial
            defect_site_arg_idx = 0

        # Get the coords and site index of the defect that was used in the VASP calculation
        defect_site_coords = defect_new_species_coords[defect_site_arg_idx]  # frac coords of defect site
        defect_site_idx = defect_new_species_idx[defect_site_arg_idx]

        # currently, original_site_idx is indexed with respect to the old species only.
        # need to get the index in the full structure:
        unrelaxed_defect_structure, bulk_site_idx = _remove_and_insert_species_from_bulk(
            bulk,
            coords=defect_site_coords,
            site_arg_idx=None,
            new_species=new_species,
            defect_site_idx=defect_site_idx,
            defect_type="interstitial",
            searched_structure="defect",
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    handlers = {
        "substitution": process_substitution,
        "vacancy": process_vacancy,
        "interstitial": process_interstitial,
    }

    if defect_type not in handlers:
        raise ValueError(f"Invalid defect type: {defect_type}")

    return handlers[defect_type](bulk, defect, composition_diff)


def _get_species_from_composition_diff(composition_diff, el_change):
    """
    Get the species corresponding to the given change in composition.
    """
    return [el for el, amt in composition_diff.items() if amt == el_change][0]


def get_coords_and_idx_of_species(structure, species_name):
    """
    Get arrays of the coordinates and indices of the given species in the
    structure.
    """
    coords = []
    idx = []
    for i, site in enumerate(structure):
        if site.specie.name == species_name:
            coords.append(site.frac_coords)
            idx.append(i)

    return np.array(coords), np.array(idx)


def find_nearest_coords(
    bulk_coords,
    target_coords,
    bulk_lattice_matrix,
    defect_type="substitution",
    searched_structure="bulk",
    unique_tolerance=1,
):
    """
    Find the nearest coords in bulk_coords to target_coords.
    """
    distance_matrix = np.linalg.norm(
        np.dot(pbc_diff(bulk_coords, target_coords), bulk_lattice_matrix), axis=-1
    )
    site_matches = distance_matrix.argmin(axis=0 if defect_type == "vacancy" else -1)

    def _site_matching_failure_error(defect_type, searched_structure):
        raise RuntimeError(
            f"Could not uniquely determine site of {defect_type} in {searched_structure} "
            f"structure. Remember the bulk and defect supercells should have the same "
            f"definitions/basis sets for site-matching (parsing) to be possible."
        )

    if len(site_matches.shape) == 1:
        if len(np.unique(site_matches)) != len(site_matches):
            _site_matching_failure_error(defect_type, searched_structure)

        return list(
            set(np.arange(max(bulk_coords.shape[0], target_coords.shape[0]), dtype=int))
            - set(site_matches)
        )[0]

    if len(site_matches.shape) == 0:
        # if there are any other matches with a distance within unique_tolerance of the located site
        # then unique matching failed
        if len(distance_matrix[distance_matrix < distance_matrix[site_matches] * unique_tolerance]) > 1:
            _site_matching_failure_error(defect_type, searched_structure)

        return site_matches
    return None


def _remove_and_insert_species_from_bulk(
    bulk,
    coords,
    site_arg_idx,
    new_species,
    defect_site_idx,
    defect_type="substitution",
    searched_structure="bulk",
    unique_tolerance=1,
):
    # currently, original_site_idx is indexed with respect to the old species only.
    # need to get the index in the full structure:
    unrelaxed_defect_structure = bulk.copy()  # create unrelaxed defect structure
    bulk_coords = np.array([s.frac_coords for s in bulk])
    bulk_site_idx = None

    if site_arg_idx is not None:
        bulk_site_idx = find_nearest_coords(
            bulk_coords,
            coords[site_arg_idx],
            bulk.lattice.matrix,
            defect_type=defect_type,
            searched_structure=searched_structure,
            unique_tolerance=unique_tolerance,
        )
        unrelaxed_defect_structure.remove_sites([bulk_site_idx])
        defect_coords = bulk_coords[bulk_site_idx]

    else:
        defect_coords = coords

    # Place defect in same location as output from DFT
    if defect_site_idx is not None:
        unrelaxed_defect_structure.insert(defect_site_idx, new_species, defect_coords)

    return unrelaxed_defect_structure, bulk_site_idx


def check_atom_mapping_far_from_defect(bulk, defect, defect_coords):
    """
    Check the displacement of atoms far from the determined defect site, and
    warn the user if they are large (often indicates a mismatch between the
    bulk and defect supercell definitions).
    """
    # suppress pydefect INFO messages
    import logging

    from vise import user_settings

    user_settings.logger.setLevel(logging.CRITICAL)
    from pydefect.cli.vasp.make_efnv_correction import calc_max_sphere_radius

    # vise suppresses `UserWarning`s, so need to reset
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`")
    warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")
    _ignore_pmg_warnings()

    far_from_defect_disps = {site.specie.name: [] for site in bulk}

    wigner_seitz_radius = calc_max_sphere_radius(bulk.lattice.matrix)

    bulk_sites_outside_or_at_wigner_radius = [
        site
        for site in bulk
        if site.distance_and_image_from_frac_coords(defect_coords)[0]
        > np.max((wigner_seitz_radius - 1, 1))
    ]

    bulk_species_coord_dict = {}
    for species in bulk.composition.elements:  # avoid recomputing coords for each site
        bulk_species_coords, _bulk_new_species_idx = get_coords_and_idx_of_species(
            bulk_sites_outside_or_at_wigner_radius, species.name
        )
        bulk_species_coord_dict[species.name] = bulk_species_coords

    for site in defect:
        if site.distance_and_image_from_frac_coords(defect_coords)[0] > wigner_seitz_radius:
            bulk_site_arg_idx = find_nearest_coords(  # get closest site in bulk to defect site
                bulk_species_coord_dict[site.specie.name],
                site.frac_coords,
                bulk.lattice.matrix,
                defect_type="substitution",
                searched_structure="bulk",
            )
            far_from_defect_disps[site.specie.name].append(
                site.distance_and_image_from_frac_coords(
                    bulk_species_coord_dict[site.specie.name][bulk_site_arg_idx]
                )[0]
            )

    if far_from_defect_large_disps := {
        specie: list for specie, list in far_from_defect_disps.items() if list and np.mean(list) > 0.5
    }:
        warnings.warn(
            f"Detected atoms far from the defect site (>{wigner_seitz_radius:.2f} Å) with major "
            f"displacements (>0.5 Å) in the defect supercell. This likely indicates a mismatch "
            f"between the bulk and defect supercell definitions or an unconverged supercell size, "
            f"both of which will likely cause errors in parsing. The mean displacement of the "
            f"following species, at sites far from the determined defect position, is >0.5 Å: "
            f"{list(far_from_defect_large_disps.keys())}, with displacements (Å): "
            f"{far_from_defect_large_disps}"
        )


def get_site_mapping_indices(structure_a: Structure, structure_b: Structure, threshold=2.0):
    """
    Reset the position of a partially relaxed structure to its unrelaxed
    positions.

    The template structure may have a different species ordering to the
    `input_structure`.
    """
    ## Generate a site matching table between the input and the template
    min_dist_with_index = []
    all_input_fcoords = [list(site.frac_coords.round(3)) for site in structure_a]
    all_template_fcoords = [list(site.frac_coords.round(3)) for site in structure_b]

    for species in structure_a.composition.elements:
        input_fcoords = [
            list(site.frac_coords.round(3))
            for site in structure_a
            if site.species.elements[0].symbol == species.symbol
        ]
        template_fcoords = [
            list(site.frac_coords.round(3))
            for site in structure_b
            if site.species.elements[0].symbol == species.symbol
        ]

        dmat = structure_a.lattice.get_all_distances(input_fcoords, template_fcoords)
        for index, coords in enumerate(all_input_fcoords):
            if coords in input_fcoords:
                dists = dmat[input_fcoords.index(coords)]
                current_dist = dists.min()
                template_fcoord = template_fcoords[dists.argmin()]
                template_index = all_template_fcoords.index(template_fcoord)
                min_dist_with_index.append(
                    [
                        current_dist,
                        index,
                        template_index,
                    ]
                )

                if current_dist > threshold:
                    site_a = structure_a[index]
                    site_b = structure_b[template_index]
                    warnings.warn(
                        f"Large site displacement {current_dist:.2f} Å detected when matching atomic "
                        f"sites: {site_a} -> {site_b}."
                    )

    return min_dist_with_index


def reorder_s1_like_s2(s1_structure: Structure, s2_structure: Structure, threshold=5.0):
    """
    Reorder the atoms of a (relaxed) structure, s1, to match the ordering of
    the atoms in s2_structure.

    s1/s2 structures may have a different species orderings.

    Previously used to ensure correct site matching when pulling site
    potentials for the eFNV Kumagai correction, though no longer used for this
    purpose. If threshold is set to a low value, it will raise a warning if
    there is a large site displacement detected.
    """
    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mapping_indices(s2_structure, s1_structure, threshold=threshold)

    # Reorder s1_structure so that it matches the ordering of s2_structure
    reordered_sites = [s1_structure[tmp[2]] for tmp in mapping]

    # avoid warning about selective_dynamics properties (can happen if user explicitly set "T T T" (or
    # otherwise) for the bulk):
    warnings.filterwarnings("ignore", message="Not all sites have property")

    new_structure = Structure.from_sites(reordered_sites)

    assert len(new_structure) == len(s1_structure)

    return new_structure


def _compare_potcar_symbols(
    bulk_potcar_symbols, defect_potcar_symbols, bulk_name="bulk", defect_name="defect"
):
    """
    Check all POTCAR symbols in the bulk are the same in the defect
    calculation.
    """
    for symbol in bulk_potcar_symbols:
        if symbol["titel"] not in [symbol["titel"] for symbol in defect_potcar_symbols]:
            warnings.warn(
                f"The POTCAR symbols for your {bulk_name} and {defect_name} calculations do not match, "
                f"which is likely to cause severe errors in the parsed results. Found the following "
                f"symbol in the {bulk_name} calculation:"
                f"\n{symbol['titel']}\n"
                f"but not in the {defect_name} calculation:"
                f"\n{[symbol['titel'] for symbol in defect_potcar_symbols]}\n"
                f"The same POTCAR settings should be used for all calculations for accurate results!"
            )
            return False
    return True


def _compare_kpoints(bulk_actual_kpoints, defect_actual_kpoints, bulk_name="bulk", defect_name="defect"):
    """
    Check bulk and defect KPOINTS are the same, using the
    Vasprun.actual_kpoints lists (i.e. the VASP IBZKPTs essentially).
    """
    # sort kpoints, in case same KPOINTS just different ordering:
    sorted_bulk_kpoints = sorted(np.array(bulk_actual_kpoints), key=tuple)
    sorted_defect_kpoints = sorted(np.array(defect_actual_kpoints), key=tuple)

    if not np.allclose(sorted_bulk_kpoints, sorted_defect_kpoints):
        warnings.warn(
            f"The KPOINTS for your {bulk_name} and {defect_name} calculations do not match, which is "
            f"likely to cause errors in the parsed results. Found the following KPOINTS in the "
            f"{bulk_name} calculation:"
            f"\n{sorted_bulk_kpoints}\n"
            f"and in the {defect_name} calculation:"
            f"\n{sorted_defect_kpoints}\n"
            f"The same KPOINTS settings should be used for all final calculations for accurate results!"
        )
        return False

    return True


def _compare_incar_tags(
    bulk_incar_dict,
    defect_incar_dict,
    fatal_incar_mismatch_tags=None,
    bulk_name="bulk",
    defect_name="defect",
):
    """
    Check bulk and defect INCAR tags (that can affect energies) are the same.
    """
    if fatal_incar_mismatch_tags is None:
        fatal_incar_mismatch_tags = {  # dict of tags that can affect energies and their defaults
            "AEXX": 0.25,  # default 0.25
            "ENCUT": 0,
            "LREAL": False,  # default False
            "HFSCREEN": 0,  # default 0 (None)
            "GGA": "PE",  # default PE
            "LHFCALC": False,  # default False
            "ADDGRID": False,  # default False
            "ISIF": 2,
            "LASPH": False,  # default False
            "PREC": "Normal",  # default Normal
            "PRECFOCK": "Normal",  # default Normal
            "LDAU": False,  # default False
        }

    def _compare_incar_vals(val1, val2):
        if isinstance(val1, str):
            return val1.split()[0].lower() == val2.split()[0].lower()
        if isinstance(val1, float):
            return np.isclose(val1, val2, rtol=1e-3)

        return val1 == val2

    mismatch_list = []
    for key, val in bulk_incar_dict.items():
        if key in fatal_incar_mismatch_tags:
            defect_val = defect_incar_dict.get(key, fatal_incar_mismatch_tags[key])
            if not _compare_incar_vals(val, defect_val):
                mismatch_list.append((key, val, defect_val))

    # get any missing keys:
    defect_incar_keys_not_in_bulk = set(defect_incar_dict.keys()) - set(bulk_incar_dict.keys())

    for key in defect_incar_keys_not_in_bulk:
        if key in fatal_incar_mismatch_tags and not _compare_incar_vals(
            defect_incar_dict[key], fatal_incar_mismatch_tags[key]
        ):
            mismatch_list.append((key, fatal_incar_mismatch_tags[key], defect_incar_dict[key]))

    if mismatch_list:
        # compare to defaults:
        warnings.warn(
            f"There are mismatching INCAR tags for your {bulk_name} and {defect_name} calculations which "
            f"are likely to cause errors in the parsed results (energies). Found the following "
            f"differences:\n"
            f"(in the format: (INCAR tag, value in {bulk_name} calculation, value in {defect_name} "
            f"calculation)):"
            f"\n{mismatch_list}\n"
            f"The same INCAR settings should be used in all final calculations for these tags which can "
            f"affect energies!"
        )
        return False
    return True
