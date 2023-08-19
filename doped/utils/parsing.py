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


def get_vasprun(vasprun_path, **kwargs):
    """
    Read the vasprun.xml(.gz) file as a pymatgen Vasprun object.
    """
    vasprun_path = str(vasprun_path)  # convert to string if Path object
    warnings.filterwarnings(
        "ignore", category=UnknownPotcarWarning
    )  # Ignore POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
    if os.path.exists(vasprun_path) and os.path.isfile(vasprun_path):
        vasprun = Vasprun(vasprun_path, **kwargs)
    else:
        raise FileNotFoundError(
            f"vasprun.xml file not found at {vasprun_path}. Needed for parsing calculation output."
        )
    return vasprun


def get_locpot(locpot_path):
    """
    Read the LOCPOT(.gz) file as a pymatgen Locpot object.
    """
    locpot_path = str(locpot_path)  # convert to string if Path object
    if os.path.exists(locpot_path) and os.path.isfile(locpot_path):
        locpot = Locpot.from_file(locpot_path)
    else:
        raise FileNotFoundError(
            f"LOCPOT file not found at {locpot_path}. Needed for calculating the Freysoldt (FNV) "
            f"image charge correction."
        )
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
    # sort by, direct match to output_file, direct match to output_file with .gz extension,
    # then alphabetically:
    output_files = sorted(
        output_files,
        key=lambda x: (
            x == output_file,
            x == output_file + ".gz",
            x,
        ),
        reverse=True,
    )
    if output_files:
        output_path = os.path.join(path, output_files[0])
        return (output_path, True) if len(output_files) > 1 else (output_path, False)
    return path, False  # so when `get_X()` is called, it will raise an informative FileNotFoundError


def get_defect_type_and_composition_diff(bulk, defect):
    """
    Get the difference in composition between a bulk structure and a defect
    structure.

    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for
    extrinsic species.
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
    Get the defect site and unrelaxed structure.

    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for
    extrinsic species.
    """
    if defect_type == "substitution":
        old_species = [el for el, amt in composition_diff.items() if amt == -1][0]
        new_species = [el for el, amt in composition_diff.items() if amt == 1][0]

        bulk_new_species_coords = np.array(
            [site.frac_coords for site in bulk if site.specie.name == new_species]
        )
        defect_new_species_coords = np.array(
            [site.frac_coords for site in defect if site.specie.name == new_species]
        )
        defect_new_species_idx = np.array(
            [defect.index(site) for site in defect if site.specie.name == new_species]
        )

        if bulk_new_species_coords.size > 0:  # intrinsic substitution
            # find coords of new species in defect structure, taking into account periodic
            # boundaries
            distance_matrix = np.linalg.norm(
                pbc_diff(bulk_new_species_coords[:, None], defect_new_species_coords),
                axis=2,
            )
            site_matches = distance_matrix.argmin(axis=1)

            if len(np.unique(site_matches)) != len(site_matches):
                raise RuntimeError("Could not uniquely determine site of new species in defect structure")

            defect_site_idx = list(
                set(np.arange(len(defect_new_species_coords), dtype=int)) - set(site_matches)
            )[0]

        else:  # extrinsic substitution
            defect_site_idx = 0

        defect_coords = defect_new_species_coords[defect_site_idx]

        # Get the site index of the defect that was used in the VASP calculation
        defect_site_idx = defect_new_species_idx[defect_site_idx]

        # now find the closest old_species site in the bulk structure to the defect site
        # again, make sure to use periodic boundaries
        bulk_old_species_coords = np.array(
            [site.frac_coords for site in bulk if site.specie.name == old_species]
        )
        distances = np.linalg.norm(pbc_diff(bulk_old_species_coords, defect_coords), axis=1)
        original_site_idx = distances.argmin()

        # if there are any other matches with a distance within unique_tolerance of the located
        # site then unique matching failed
        if len(distances[distances < distances[original_site_idx] * unique_tolerance]) > 1:
            raise RuntimeError("Could not uniquely determine site of old species in bulk structure")

        # currently, original_site_idx is indexed with respect to the old species only.
        # Need to get the index in the full structure
        bulk_coords = np.array([s.frac_coords for s in bulk])
        bulk_site_idx = np.linalg.norm(
            pbc_diff(bulk_coords, bulk_old_species_coords[original_site_idx]), axis=1
        ).argmin()

        # create unrelaxed defect structure
        unrelaxed_defect_structure = bulk.copy()
        unrelaxed_defect_structure.remove_sites([bulk_site_idx])
        # Place defect in same location as output from DFT
        unrelaxed_defect_structure.insert(defect_site_idx, new_species, bulk_coords[bulk_site_idx])

    elif defect_type == "vacancy":
        old_species = list(composition_diff.keys())[0]

        bulk_old_species_coords = np.array(
            [site.frac_coords for site in bulk if site.specie.name == old_species]
        )
        defect_old_species_coords = np.array(
            [site.frac_coords for site in defect if site.specie.name == old_species]
        )

        # make sure to do take into account periodic boundaries
        distance_matrix = np.linalg.norm(
            pbc_diff(bulk_old_species_coords[:, None], defect_old_species_coords),
            axis=2,
        )
        site_matches = distance_matrix.argmin(axis=0)

        if len(np.unique(site_matches)) != len(site_matches):
            raise RuntimeError("Could not uniquely determine site of vacancy in defect structure")

        original_site_idx = list(
            set(np.arange(len(bulk_old_species_coords), dtype=int)) - set(site_matches)
        )[0]

        # currently, original_site_idx is indexed with respect to the old species only.
        # Need to get the index in the full structure
        bulk_coords = np.array([s.frac_coords for s in bulk])
        bulk_site_idx = np.linalg.norm(
            pbc_diff(bulk_coords, bulk_old_species_coords[original_site_idx]), axis=1
        ).argmin()

        # create unrelaxed defect structure
        unrelaxed_defect_structure = bulk.copy()
        unrelaxed_defect_structure.remove_sites([bulk_site_idx])
        defect_site_idx = None

    elif defect_type == "interstitial":
        new_species = list(composition_diff.keys())[0]

        bulk_new_species_coords = np.array(
            [site.frac_coords for site in bulk if site.specie.name == new_species]
        )
        defect_new_species_coords = np.array(
            [site.frac_coords for site in defect if site.specie.name == new_species]
        )
        defect_new_species_idx = np.array(
            [defect.index(site) for site in defect if site.specie.name == new_species]
        )

        if bulk_new_species_coords.size > 0:  # intrinsic interstitial
            # make sure to take into account periodic boundaries
            distance_matrix = np.linalg.norm(
                pbc_diff(bulk_new_species_coords[:, None], defect_new_species_coords),
                axis=2,
            )
            site_matches = distance_matrix.argmin(axis=1)

            if len(np.unique(site_matches)) != len(site_matches):
                raise RuntimeError("Could not uniquely determine site of interstitial in defect structure")

            defect_site_idx = list(
                set(np.arange(len(defect_new_species_coords), dtype=int)) - set(site_matches)
            )[0]

        else:  # extrinsic interstitial
            defect_site_idx = 0

        defect_site_coords = defect_new_species_coords[defect_site_idx]

        # Get the site index of the defect that was used in the VASP calculation
        defect_site_idx = defect_new_species_idx[defect_site_idx]

        # create unrelaxed defect structure
        unrelaxed_defect_structure = bulk.copy()
        # Place defect in same location as output from DFT
        unrelaxed_defect_structure.insert(defect_site_idx, new_species, defect_site_coords)
        bulk_site_idx = None

    else:
        raise ValueError(f"Invalid defect type: {defect_type}")

    return (
        bulk_site_idx,
        defect_site_idx,
        unrelaxed_defect_structure,
    )


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


def reorder_unrelaxed_structure(
    unrelaxed_structure: Structure, initial_relax_structure: Structure, threshold=2.0
):
    """
    Reset the position of a partially relaxed structure to its unrelaxed
    positions.

    The template structure may have a different species ordering to the
    `input_structure`.
    """
    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mapping_indices(initial_relax_structure, unrelaxed_structure, threshold=threshold)

    # Reorder the unrelaxed_structure so it matches the ordering of the initial_relax_structure (
    # from the actual calculation)
    reordered_sites = [unrelaxed_structure[tmp[2]] for tmp in mapping]
    new_structure = Structure.from_sites(reordered_sites)

    assert len(new_structure) == len(unrelaxed_structure)

    return new_structure
