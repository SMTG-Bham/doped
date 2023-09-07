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
    warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
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
    if os.path.exists(outcar_path) and os.path.isfile(outcar_path):
        outcar = Outcar(outcar_path)
    else:
        raise FileNotFoundError(
            f"OUTCAR file not found at {outcar_path}. Needed for calculating the Kumagai (eFNV) "
            f"image charge correction."
        )
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

    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for
    extrinsic species and code efficiency/robustness improvements.
    """

    def get_species_from_composition_diff(composition_diff, el_change):
        return [el for el, amt in composition_diff.items() if amt == el_change][0]

    def get_coords_and_idx(structure, species_name):
        coords = np.array([site.frac_coords for site in structure if site.specie.name == species_name])
        idx = np.array([structure.index(site) for site in structure if site.specie.name == species_name])
        return coords, idx

    def find_nearest_species(
        bulk_coords,
        target_coords,
        bulk_lattice_matrix,
        defect_type="substitution",
        searched_structure="bulk",
        unique_tolerance=1,
    ):
        distance_matrix = np.linalg.norm(
            np.dot(pbc_diff(bulk_coords, target_coords), bulk_lattice_matrix), axis=-1
        )
        site_matches = distance_matrix.argmin(axis=0 if defect_type == "vacancy" else -1)

        if len(site_matches.shape) == 1:
            if len(np.unique(site_matches)) != len(site_matches):
                raise RuntimeError(
                    f"Could not uniquely determine site of {defect_type} in {searched_structure} structure"
                )

            return list(
                set(np.arange(max(bulk_coords.shape[0], target_coords.shape[0]), dtype=int))
                - set(site_matches)
            )[0]

        if len(site_matches.shape) == 0:
            # # if there are any other matches with a distance within unique_tolerance of the located
            # # site then unique matching failed
            if (
                len(distance_matrix[distance_matrix < distance_matrix[site_matches] * unique_tolerance])
                > 1
            ):
                raise RuntimeError(
                    f"Could not uniquely determine site of {defect_type} in {searched_structure} structure"
                )

            return site_matches
        return None

    def remove_and_insert_species_from_bulk(
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
            bulk_site_idx = find_nearest_species(
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

    def process_substitution(bulk, defect, composition_diff):
        old_species = get_species_from_composition_diff(composition_diff, -1)
        new_species = get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, _bulk_new_species_idx = get_coords_and_idx(bulk, new_species)
        defect_new_species_coords, defect_new_species_idx = get_coords_and_idx(defect, new_species)

        if bulk_new_species_coords.size > 0:  # intrinsic substitution
            # find coords of new species in defect structure, taking into account periodic boundaries
            defect_site_arg_idx = find_nearest_species(
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
        bulk_old_species_coords, _bulk_old_species_idx = get_coords_and_idx(bulk, old_species)

        bulk_site_arg_idx = find_nearest_species(
            bulk_old_species_coords,
            defect_coords,
            bulk.lattice.matrix,
            defect_type="substitution",
            searched_structure="bulk",
        )

        # currently, original_site_idx is indexed with respect to the old species only.
        # need to get the index in the full structure:
        unrelaxed_defect_structure, bulk_site_idx = remove_and_insert_species_from_bulk(
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
        old_species = get_species_from_composition_diff(composition_diff, -1)
        bulk_old_species_coords, _bulk_old_species_idx = get_coords_and_idx(bulk, old_species)
        defect_old_species_coords, _defect_old_species_idx = get_coords_and_idx(defect, old_species)

        bulk_site_arg_idx = find_nearest_species(
            bulk_old_species_coords[:, None],
            defect_old_species_coords,
            bulk.lattice.matrix,
            defect_type="vacancy",
            searched_structure="bulk",
        )

        # currently, original_site_idx is indexed with respect to the old species only.
        # need to get the index in the full structure:
        defect_site_idx = None
        unrelaxed_defect_structure, bulk_site_idx = remove_and_insert_species_from_bulk(
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
        new_species = get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, _bulk_new_species_idx = get_coords_and_idx(bulk, new_species)
        defect_new_species_coords, defect_new_species_idx = get_coords_and_idx(defect, new_species)

        if bulk_new_species_coords.size > 0:  # intrinsic interstitial
            defect_site_arg_idx = find_nearest_species(
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
        unrelaxed_defect_structure, bulk_site_idx = remove_and_insert_species_from_bulk(
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


def get_site_mapping_indices(structure_a: Structure, structure_b: Structure, threshold=2.0):
    """
    Reset the position of a partially relaxed structure to its unrelaxed
    positions.

    The template structure may have a different species ordering to the
    `input_structure`.
    """
    ## Generate a site matching table between the input and the template
    input_fcoords = [site.frac_coords for site in structure_a]
    template_fcoords = [site.frac_coords for site in structure_b]

    dmat = structure_a.lattice.get_all_distances(input_fcoords, template_fcoords)
    min_dist_with_index = []
    for index in range(len(input_fcoords)):
        dists = dmat[index]
        template_index = dists.argmin()
        current_dist = dists.min()
        min_dist_with_index.append(
            [
                current_dist,
                index,
                template_index,
            ]
        )

        if current_dist > threshold:
            sitea = structure_a[index]
            siteb = structure_b[template_index]
            warnings.warn(
                f"Large site displacement {current_dist:.4f} detected when matching atomic sites:"
                f" {sitea}-> {siteb}."
            )
    return min_dist_with_index


def reorder_s1_like_s2(s1_structure: Structure, s2_structure: Structure, threshold=2.0):
    """
    Reorder the atoms of a (relaxed) structure, s1, to match the ordering of
    the atoms in s2_structure.

    s1/s2 structures may have a different species orderings.
    """
    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mapping_indices(s2_structure, s1_structure, threshold=threshold)

    # Reorder s1_structure so that it matches the ordering of s2_structure
    reordered_sites = [s1_structure[tmp[2]] for tmp in mapping]
    new_structure = Structure.from_sites(reordered_sites)

    assert len(new_structure) == len(s1_structure)

    return new_structure
