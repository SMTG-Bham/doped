"""
Helper functions for parsing VASP supercell defect calculations.
"""

import contextlib
import itertools
import logging
import os
import re
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Optional, Union

import numpy as np
from monty.serialization import loadfn
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.inputs import POTCAR_STATS_PATH, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Procar, Vasprun, _parse_vasp_array
from pymatgen.util.coord import pbc_diff
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry


@lru_cache(maxsize=1000)  # cache POTCAR generation to speed up generation and writing
def _get_potcar_summary_stats() -> dict:
    return loadfn(POTCAR_STATS_PATH)


@contextlib.contextmanager
def suppress_logging(level=logging.CRITICAL):
    """
    Context manager to catch and suppress logging messages.
    """
    previous_level = logging.root.manager.disable  # store the current logging level
    logging.disable(level)  # disable logging at the specified level
    try:
        yield
    finally:
        logging.disable(previous_level)  # restore the original logging level


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


# has to be defined as staticmethod to be consistent with usage in pymatgen, alternatively could make
# fake custom class:
@staticmethod  # type: ignore[misc]
def parse_projected_eigen_no_mag(elem):
    """
    Parse the projected eigenvalues from a ``Vasprun`` object (used during
    initialisation), but excluding the projected magnetisation for efficiency.

    This is a modified version of ``_parse_projected_eigen``
    from ``pymatgen.io.vasp.outputs.Vasprun``, which skips
    parsing of the projected magnetisation in order to expedite
    parsing in ``doped``, as well as some small adjustments to
    maximise efficiency.
    """
    root = elem.find("array/set")
    proj_eigen = defaultdict(list)
    sets = root.findall("set")

    for s in sets:
        spin = int(re.match(r"spin(\d+)", s.attrib["comment"])[1])
        if spin == 1 or (spin == 2 and len(sets) == 2):
            spin_key = Spin.up if spin == 1 else Spin.down
            proj_eigen[spin_key] = np.array(
                [[_parse_vasp_array(sss) for sss in ss.findall("set")] for ss in s.findall("set")]
            )

    # here we _could_ round to 3 decimal places (and ensure rounding 0.0005 up to 0.001) to be _mostly_
    # consistent with PROCAR values (still not 100% the same as e.g. 0.00047 will be rounded to 0.0005
    # in vasprun, but 0.000 in PROCAR), but this is _reducing_ the accuracy so better not to do this,
    # and accept that PROCAR results may not be as numerically robust
    # proj_eigen = {k: np.round(v+0.00001, 3) for k, v in proj_eigen.items()}
    proj_mag = None
    elem.clear()
    return proj_eigen, proj_mag


Vasprun._parse_projected_eigen = parse_projected_eigen_no_mag  # skip parsing of proj magnetisation


def get_vasprun(vasprun_path: PathLike, **kwargs):
    """
    Read the ``vasprun.xml(.gz)`` file as a ``pymatgen`` ``Vasprun`` object.
    """
    vasprun_path = str(vasprun_path)  # convert to string if Path object
    warnings.filterwarnings(
        "ignore", category=UnknownPotcarWarning
    )  # Ignore unknown POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings(
        "ignore", message="No POTCAR file with matching TITEL fields"
    )  # `message` only needs to match start of message
    default_kwargs = {"parse_dos": False}
    default_kwargs.update(kwargs)
    try:
        vasprun = Vasprun(find_archived_fname(vasprun_path), **default_kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"vasprun.xml not found at {vasprun_path}(.gz/.xz/.bz/.lzma). Needed for parsing calculation "
            f"output!"
        ) from exc
    return vasprun


def get_locpot(locpot_path: PathLike):
    """
    Read the ``LOCPOT(.gz)`` file as a ``pymatgen`` ``Locpot`` object.
    """
    locpot_path = str(locpot_path)  # convert to string if Path object
    try:
        locpot = Locpot.from_file(find_archived_fname(locpot_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LOCPOT file not found at {locpot_path}(.gz/.xz/.bz/.lzma). Needed for calculating the "
            f"Freysoldt (FNV) image charge correction!"
        ) from None
    return locpot


def get_outcar(outcar_path: PathLike):
    """
    Read the ``OUTCAR(.gz)`` file as a ``pymatgen`` ``Outcar`` object.
    """
    outcar_path = str(outcar_path)  # convert to string if Path object
    try:
        outcar = Outcar(find_archived_fname(outcar_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"OUTCAR file not found at {outcar_path}(.gz/.xz/.bz/.lzma). Needed for calculating the "
            f"Kumagai (eFNV) image charge correction."
        ) from None
    return outcar


def get_procar(procar_path: PathLike):
    """
    Read the ``PROCAR(.gz)`` file as an ``easyunfold`` ``Procar`` object (if
    ``easyunfold`` installed), else a ``pymatgen`` ``Procar`` object (doesn't
    support SOC).

    If ``easyunfold`` installed, the ``Procar`` will be parsed with
    ``easyunfold`` and then the ``proj_data`` attribute will be converted
    to a ``data`` attribute (to be compatible with ``pydefect``, which uses
    the ``pymatgen`` format).
    """
    try:
        procar_path = find_archived_fname(str(procar_path))  # convert to string if Path object
    except FileNotFoundError:
        raise FileNotFoundError(f"PROCAR file not found at {procar_path}(.gz/.xz/.bz/.lzma)!") from None

    easyunfold_installed = True  # first try loading with easyunfold
    try:
        from easyunfold.procar import Procar as EasyunfoldProcar
    except ImportError:
        easyunfold_installed = False

    if easyunfold_installed:
        procar = EasyunfoldProcar(procar_path, normalise=False)
        if procar._is_soc:
            procar.data = {Spin.up: procar.proj_data[0]}
        else:
            procar.data = {Spin.up: procar.proj_data[0], Spin.down: procar.proj_data[1]}
        del procar.proj_data  # reduce space
    else:
        try:  # try parsing with ``pymatgen`` instead, but doesn't support SOC!
            procar = Procar(procar_path)
        except IndexError as exc:  # SOC error
            raise ValueError(
                "PROCAR from a SOC calculation was provided, but `easyunfold` is not installed and "
                "`pymatgen` does not support SOC PROCAR parsing! Please install `easyunfold` with `pip "
                "install easyunfold`."
            ) from exc

    return procar


def _get_output_files_and_check_if_multiple(output_file: PathLike = "vasprun.xml", path: PathLike = "."):
    """
    Search for all files with filenames matching ``output_file``, case-
    insensitive.

    Returns (output file path, Multiple?) where ``Multiple`` is
    ``True`` if multiple matching files are found.

    Args:
        output_file (PathLike):
            The filename to search for (case-insensitive).
            Should be either ``vasprun.xml``, ``OUTCAR``,
            ``LOCPOT`` or ``PROCAR``.
        path (PathLike): The path to the directory to search in.
    """
    if output_file.lower() == "vasprun.xml":
        search_patterns = ["vasprun", ".xml"]
    else:
        search_patterns = [output_file.lower()]

    files = os.listdir(path)
    output_files = [
        filename
        for filename in files
        if all(i in filename.lower() for i in search_patterns) and not filename.startswith(".")
    ]
    # sort by direct match to {output_file}, direct match to {output_file}.gz, then alphabetically:
    if output_files := sorted(
        output_files,
        key=lambda x: (x == output_file, x == f"{output_file}.gz", x),
        reverse=True,
    ):
        output_path = os.path.join(path, output_files[0])
        return (output_path, True) if len(output_files) > 1 else (output_path, False)
    return (
        os.path.join(path, output_file),
        False,
    )  # so `get_X()` will raise an informative FileNotFoundError


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

    if len(composition_diff) == 1 and next(iter(composition_diff.values())) == 1:
        defect_type = "interstitial"
    elif len(composition_diff) == 1 and next(iter(composition_diff.values())) == -1:
        defect_type = "vacancy"
    elif len(composition_diff) == 2:
        defect_type = "substitution"
    else:
        raise RuntimeError(
            f"Could not determine defect type from composition difference of bulk ({bulk_comp}) and "
            f"defect ({defect_comp}) structures."
        )

    return defect_type, composition_diff


def get_defect_site_idxs_and_unrelaxed_structure(
    bulk, defect, defect_type, composition_diff, unique_tolerance=1
):
    """
    Get the defect site and unrelaxed structure, where 'unrelaxed structure'
    corresponds to the pristine defect supercell structure for vacancies /
    substitutions, and the pristine bulk structure with the `final` relaxed
    interstitial site for interstitials.

    Initially contributed by Dr. Alex Ganose (@ Imperial Chemistry) and
    refactored for extrinsic species and code efficiency/robustness improvements.

    Returns:
        bulk_site_idx:
            Index of the site in the bulk structure that corresponds
            to the defect site in the defect structure
        defect_site_idx:
            Index of the defect site in the defect structure
        unrelaxed_defect_structure:
            Pristine defect supercell structure for vacancies/substitutions
            (i.e. pristine bulk with unrelaxed vacancy/substitution), or the
            pristine bulk structure with the `final` relaxed interstitial
            site for interstitials.
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
    return next(el for el, amt in composition_diff.items() if amt == el_change)


def get_coords_and_idx_of_species(structure, species_name):
    """
    Get arrays of the coordinates and indices of the given species in the
    structure.
    """
    coords = []
    idx = []
    for i, site in enumerate(structure):
        if site.specie.symbol == species_name:
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

        return next(
            iter(
                set(np.arange(max(bulk_coords.shape[0], target_coords.shape[0]), dtype=int))
                - set(site_matches)
            )
        )

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
    orig_simplefilter = warnings.simplefilter
    warnings.simplefilter = lambda *args, **kwargs: None  # monkey-patch to avoid vise warning suppression

    # suppress pydefect INFO messages
    import logging

    try:
        from vise import user_settings

        user_settings.logger.setLevel(logging.CRITICAL)
        from pydefect.cli.vasp.make_efnv_correction import calc_max_sphere_radius

    except ImportError:  # can't check as vise/pydefect not installed. Not critical so just return
        return

    warnings.simplefilter = orig_simplefilter  # reset to original

    far_from_defect_disps = {site.specie.symbol: [] for site in bulk}

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
                bulk_species_coord_dict[site.specie.symbol],
                site.frac_coords,
                bulk.lattice.matrix,
                defect_type="substitution",
                searched_structure="bulk",
            )
            far_from_defect_disps[site.specie.symbol].append(
                round(
                    site.distance_and_image_from_frac_coords(
                        bulk_species_coord_dict[site.specie.symbol][bulk_site_arg_idx]
                    )[0],
                    2,
                )
            )

    if far_from_defect_large_disps := {
        specie: list for specie, list in far_from_defect_disps.items() if list and np.mean(list) > 0.5
    }:
        warnings.warn(
            f"Detected atoms far from the defect site (>{wigner_seitz_radius:.2f} Å) with major "
            f"displacements (>0.5 Å) in the defect supercell. This likely indicates a mismatch "
            f"between the bulk and defect supercell definitions or an unconverged supercell size, "
            f"both of which could cause errors in parsing. The mean displacement of the following "
            f"species, at sites far from the determined defect position, is >0.5 Å: "
            f"{list(far_from_defect_large_disps.keys())}, with displacements (Å): "
            f"{far_from_defect_large_disps}"
        )


def get_site_mapping_indices(
    structure_a: Structure, structure_b: Structure, threshold: float = 2.0, dists_only: bool = False
):
    """
    Get the site mapping indices between two structures, based on the
    fractional coordinates of the sites.

    The template structure may have a different species ordering to the
    ``input_structure``.

    NOTE: This assumes that both structures have the same lattice definitions
    (i.e. that they match, and aren't rigidly translated/rotated with respect
    to each other), which is mostly the case unless we have a mismatching
    defect/bulk supercell (in which case the ``check_atom_mapping_far_from_defect``
    warning should be thrown anyway during parsing). Currently this function
    is only used for analysing site displacements in the ``displacements`` module
    so this is fine (user will already have been warned at this point if there is a
    possible mismatch).

    Args:
        structure_a (Structure):
            The input structure.
        structure_b (Structure):
            The template structure.
        threshold (float):
            If the distance between a pair of matched sites is larger than this,
            then a warning will be thrown. Default is 2.0 Å.
        dists_only (bool):
            Whether to return only the distances between matched sites, rather
            than a list of lists containing the distance, index in structure_a
            and index in structure_b. Default is False.

    Returns:
        list:
            A list of lists containing the distance, index in structure_a and
            index in structure_b for each matched site. If ``dists_only`` is
            ``True``, then only the distances between matched sites are returned.
    """
    ## Generate a site matching table between the input and the template
    min_dist_with_index = []
    all_input_fcoords = [list(site.frac_coords) for site in structure_a]
    all_template_fcoords = [list(site.frac_coords) for site in structure_b]

    for species in structure_a.composition.elements:
        input_fcoords = [
            list(site.frac_coords) for site in structure_a if site.specie.symbol == species.symbol
        ]
        template_fcoords = [
            list(site.frac_coords) for site in structure_b if site.specie.symbol == species.symbol
        ]

        dmat = structure_a.lattice.get_all_distances(input_fcoords, template_fcoords)
        for index, coords in enumerate(all_input_fcoords):
            if coords in input_fcoords:
                dists = dmat[input_fcoords.index(coords)]
                current_dist = dists.min()
                if dists_only:
                    min_dist_with_index.append(current_dist)
                else:
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


def reorder_s1_like_s2(s1_structure: Structure, s2_structure: Structure, threshold=5.0) -> Structure:
    """
    Reorder the atoms of a (relaxed) structure, s1, to match the ordering of
    the atoms in s2_structure.

    s1/s2 structures may have a different species orderings.

    Previously used to ensure correct site matching when pulling site
    potentials for the eFNV Kumagai correction, though no longer used for this
    purpose. If threshold is set to a low value, it will raise a warning if
    there is a large site displacement detected.

    NOTE: This assumes that both structures have the same lattice definitions
    (i.e. that they match, and aren't rigidly translated/rotated with respect
    to each other), which is mostly the case unless we have a mismatching
    defect/bulk supercell (in which case the ``check_atom_mapping_far_from_defect``
    warning should be thrown anyway during parsing). Currently this function
    is no longer used, but if it is reintroduced at any point, this point should
    be noted!

    Args:
        s1_structure (Structure):
            The input structure.
        s2_structure (Structure):
            The template structure.
        threshold (float):
            If the distance between a pair of matched sites is larger than this,
            then a warning will be thrown. Default is 5.0 Å.

    Returns:
        Structure:
            The reordered structure.
    """
    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mapping_indices(s2_structure, s1_structure, threshold=threshold)

    # Reorder s1_structure so that it matches the ordering of s2_structure
    reordered_sites = [s1_structure[tmp[2]] for tmp in mapping]

    # avoid warning about selective_dynamics properties (can happen if user explicitly set "T T T" (or
    # otherwise) for the bulk):
    warnings.filterwarnings("ignore", message="Not all sites have property")

    new_structure = Structure.from_sites(reordered_sites)

    if len(new_structure) != len(s1_structure):
        raise ValueError("Structure reordering failed: structures have different number of sites?")

    return new_structure


def _compare_potcar_symbols(
    bulk_potcar_symbols, defect_potcar_symbols, bulk_name="bulk", defect_name="defect"
):
    """
    Check all POTCAR symbols in the bulk are the same in the defect
    calculation.

    Returns True if the symbols match, otherwise returns a list of the symbols
    for the bulk and defect calculations.
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
            return [bulk_potcar_symbols, defect_potcar_symbols]
    return True


def _compare_kpoints(
    bulk_actual_kpoints,
    defect_actual_kpoints,
    bulk_kpoints=None,
    defect_kpoints=None,
    bulk_name="bulk",
    defect_name="defect",
):
    """
    Check bulk and defect KPOINTS are the same, using the
    Vasprun.actual_kpoints lists (i.e. the VASP IBZKPTs essentially).

    Returns True if the KPOINTS match, otherwise returns a list of the KPOINTS
    for the bulk and defect calculations.
    """
    # sort kpoints, in case same KPOINTS just different ordering:
    sorted_bulk_kpoints = sorted(np.array(bulk_actual_kpoints), key=tuple)
    sorted_defect_kpoints = sorted(np.array(defect_actual_kpoints), key=tuple)

    actual_kpoints_eq = len(sorted_bulk_kpoints) == len(sorted_defect_kpoints) and np.allclose(
        sorted_bulk_kpoints, sorted_defect_kpoints
    )
    # if different symmetry settings used (e.g. for bulk), actual_kpoints can differ but are the same
    # input kpoints, which we assume is fine:
    kpoints_eq = bulk_kpoints.kpts == defect_kpoints.kpts if bulk_kpoints and defect_kpoints else True

    if not (actual_kpoints_eq or kpoints_eq):
        warnings.warn(
            f"The KPOINTS for your {bulk_name} and {defect_name} calculations do not match, which is "
            f"likely to cause errors in the parsed results. Found the following KPOINTS in the "
            f"{bulk_name} calculation:"
            f"\n{[list(kpoints) for kpoints in sorted_bulk_kpoints]}\n"  # list more readable than array
            f"and in the {defect_name} calculation:"
            f"\n{[list(kpoints) for kpoints in sorted_defect_kpoints]}\n"
            f"In general, the same KPOINTS settings should be used for all final calculations for "
            f"accurate results!"
        )
        return [
            [list(kpoints) for kpoints in sorted_bulk_kpoints],
            [list(kpoints) for kpoints in sorted_defect_kpoints],
        ]

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

    Returns True if no mismatching tags are found, otherwise returns a list of
    the mismatching tags.
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
            "NKRED": 1,  # default 1
        }

    def _compare_incar_vals(val1, val2):
        if isinstance(val1, str):
            return val1.split()[0].lower() == str(val2).split()[0].lower()
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
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
            f"In general, the same INCAR settings should be used in all final calculations for these tags "
            f"which can affect energies!"
        )
        return mismatch_list
    return True


def get_magnetization_from_vasprun(vasprun: Vasprun) -> Union[int, float]:
    """
    Determine the magnetization (number of spin-up vs spin-down electrons) from
    a ``Vasprun`` object.

    Args:
        vasprun (Vasprun):
            The ``Vasprun`` object from which to extract the total magnetization.

    Returns:
        int or float: The total magnetization of the system.
    """
    # in theory should be able to use vasprun.idos (integrated dos), but this
    # doesn't show spin-polarisation / account for NELECT changes from neutral
    # apparently

    eigenvalues_and_occs = vasprun.eigenvalues
    kweights = vasprun.actual_kpoints_weights

    # first check if it's even a spin-polarised calculation:
    if len(eigenvalues_and_occs) == 1 or not vasprun.is_spin:
        return 0  # non-spin polarised or SOC calculation
    # (can't pull SOC magnetization this way and either way isn't needed/desired for magnetization value
    # in ``VaspBandEdgeProperties``)

    # product of the sum of occupations over all bands, times the k-point weights:
    n_spin_up = np.sum(eigenvalues_and_occs[Spin.up][:, :, 1].sum(axis=1) * kweights)
    n_spin_down = np.sum(eigenvalues_and_occs[Spin.down][:, :, 1].sum(axis=1) * kweights)

    return n_spin_up - n_spin_down


def get_nelect_from_vasprun(vasprun: Vasprun) -> Union[int, float]:
    """
    Determine the number of electrons (``NELECT``) from a ``Vasprun`` object.

    Args:
        vasprun (Vasprun):
            The ``Vasprun`` object from which to extract ``NELECT``.

    Returns:
        int or float: The number of electrons in the system.
    """
    # in theory should be able to use vasprun.idos (integrated dos), but this
    # doesn't show spin-polarisation / account for NELECT changes from neutral
    # apparently

    eigenvalues_and_occs = vasprun.eigenvalues
    kweights = vasprun.actual_kpoints_weights

    # product of the sum of occupations over all bands, times the k-point weights:
    nelect = np.sum(eigenvalues_and_occs[Spin.up][:, :, 1].sum(axis=1) * kweights)
    if len(eigenvalues_and_occs) > 1:
        nelect += np.sum(eigenvalues_and_occs[Spin.down][:, :, 1].sum(axis=1) * kweights)
    elif not vasprun.parameters.get("LNONCOLLINEAR", False):
        nelect *= 2  # non-spin-polarised or SOC calc

    return round(nelect, 2)


def get_neutral_nelect_from_vasprun(vasprun: Vasprun, skip_potcar_init: bool = False) -> Union[int, float]:
    """
    Determine the number of electrons (``NELECT``) from a ``Vasprun`` object,
    corresponding to a neutral charge state for the structure.

    Args:
        vasprun (Vasprun):
            The ``Vasprun`` object from which to extract ``NELECT``.
        skip_potcar_init (bool):
            Whether to skip the initialisation of the ``POTCAR`` statistics
            (i.e. the auto-charge determination) and instead try to reverse
            engineer ``NELECT`` using the ``DefectDictSet``.

    Returns:
        int or float: The number of electrons in the system for a neutral
        charge state.
    """
    nelect = None
    if not skip_potcar_init:
        with contextlib.suppress(Exception):  # try determine charge without POTCARs first:
            grouped_symbols = [list(group) for key, group in itertools.groupby(vasprun.atomic_symbols)]
            potcar_summary_stats = _get_potcar_summary_stats()

            for trial_functional in ["PBE_64", "PBE_54", "PBE_52", "PBE", potcar_summary_stats.keys()]:
                if all(
                    potcar_summary_stats[trial_functional].get(
                        vasprun.potcar_spec[i]["titel"].replace(" ", ""), False
                    )
                    for i in range(len(grouped_symbols))
                ):
                    break

            nelect = sum(  # this is always the NELECT for the bulk
                np.array([len(i) for i in grouped_symbols])
                * np.array(
                    [
                        potcar_summary_stats[trial_functional][
                            vasprun.potcar_spec[i]["titel"].replace(" ", "")
                        ][0]["ZVAL"]
                        for i in range(len(grouped_symbols))
                    ]
                )
            )

    if nelect is not None:
        return nelect

    # else try reverse engineer NELECT using DefectDictSet
    from doped.vasp import DefectDictSet

    potcar_symbols = [titel.split()[1] for titel in vasprun.potcar_symbols]
    potcar_settings = {symbol.split("_")[0]: symbol for symbol in potcar_symbols}
    with warnings.catch_warnings():  # ignore POTCAR warnings if not available
        warnings.simplefilter("ignore", UserWarning)
        return DefectDictSet(
            vasprun.structures[-1],
            charge_state=0,
            user_potcar_settings=potcar_settings,
        ).nelect


def get_interstitial_site_and_orientational_degeneracy(
    interstitial_defect_entry: DefectEntry, dist_tol: float = 0.15
) -> int:
    """
    Get the combined site and orientational degeneracy of an interstitial
    defect entry.

    The standard approach of using ``_get_equiv_sites()`` for interstitial
    site multiplicity and then ``point_symmetry_from_defect_entry()`` &
    ``get_orientational_degeneracy`` for symmetry/orientational degeneracy
    is preferred (as used in the ``DefectParser`` code), but alternatively
    this function can be used to compute the product of the site and
    orientational degeneracies.

    This is done by determining the number of equivalent sites in the bulk
    supercell for the given interstitial site (from defect_supercell_site),
    which gives the combined site and orientational degeneracy `if` there
    was no relaxation of the bulk lattice atoms. This matches the true
    combined degeneracy in most cases, except for split-interstitial type
    defects etc, where this would give an artificially high degeneracy
    (as, for example, the interstitial site is automatically assigned to
    one of the split-interstitial atoms and not the midpoint, giving a
    doubled degeneracy as it considers the two split-interstitial sites as
    two separate (degenerate) interstitial sites, instead of one).
    This is counteracted by dividing by the number of sites which are present
    in the defect supercell (within a distance tolerance of dist_tol in Å)
    with the same species, ensuring none of the predicted `different`
    equivalent sites are actually `included` in the defect structure.

    Args:
        interstitial_defect_entry: DefectEntry object for the interstitial
            defect.
        dist_tol: distance tolerance in Å for determining equivalent sites.

    Returns:
        combined site and orientational degeneracy of the interstitial defect
        entry (int).
    """
    from doped.utils.symmetry import _get_all_equiv_sites

    if interstitial_defect_entry.bulk_entry is None:
        raise ValueError(
            "bulk_entry must be set for interstitial_defect_entry to determine the site and orientational "
            "degeneracies! (i.e. must be a parsed DefectEntry)"
        )
    equiv_sites = _get_all_equiv_sites(
        _get_defect_supercell_bulk_site_coords(interstitial_defect_entry),
        _get_bulk_supercell(interstitial_defect_entry),
    )
    equiv_sites_array = np.array([site.frac_coords for site in equiv_sites])
    defect_supercell_sites_of_same_species_array = np.array(
        [
            site.frac_coords
            for site in _get_defect_supercell(interstitial_defect_entry)
            if site.specie.symbol == interstitial_defect_entry.defect.site.specie.symbol
        ]
    )

    distance_matrix = np.linalg.norm(
        np.dot(
            pbc_diff(defect_supercell_sites_of_same_species_array[:, None], equiv_sites_array),
            _get_bulk_supercell(interstitial_defect_entry).lattice.matrix,
        ),
        axis=-1,
    )

    return len(equiv_sites) // len(distance_matrix[distance_matrix < dist_tol])


def get_orientational_degeneracy(
    defect_entry: Optional[DefectEntry] = None,
    relaxed_point_group: Optional[str] = None,
    bulk_site_point_group: Optional[str] = None,
    bulk_symm_ops: Optional[list] = None,
    defect_symm_ops: Optional[list] = None,
    symprec: float = 0.1,
) -> float:
    r"""
    Get the orientational degeneracy factor for a given `relaxed` DefectEntry,
    by supplying either the DefectEntry object or the bulk-site & relaxed
    defect point group symbols (e.g. "Td", "C3v" etc).

    If a DefectEntry is supplied (and the point group symbols are not),
    this is computed by determining the `relaxed` defect point symmetry and the
    (unrelaxed) bulk site symmetry, and then getting the ratio of
    their point group orders (equivalent to the ratio of partition
    functions or number of symmetry operations (i.e. degeneracy)).

    For interstitials, the bulk site symmetry corresponds to the
    point symmetry of the interstitial site with `no relaxation
    of the host structure`, while for vacancies/substitutions it is
    simply the symmetry of their corresponding bulk site.
    This corresponds to the point symmetry of ``DefectEntry.defect``,
    or ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``.

    Note: This tries to use the defect_entry.defect_supercell to determine
    the `relaxed` site symmetry. However, it should be noted that this is not
    guaranteed to work in all cases; namely for non-diagonal supercell
    expansions, or sometimes for non-scalar supercell expansion matrices
    (e.g. a 2x1x2 expansion)(particularly with high-symmetry materials)
    which can mess up the periodicity of the cell. ``doped`` tries to automatically
    check if this is the case, and will warn you if so.

    This can also be checked by using this function on your doped `generated` defects:

    .. code-block:: python

        from doped.generation import get_defect_name_from_entry
        for defect_name, defect_entry in defect_gen.items():
            print(defect_name, get_defect_name_from_entry(defect_entry, relaxed=False),
                  get_defect_name_from_entry(defect_entry), "\n")

    And if the point symmetries match in each case, then using this function on your
    parsed `relaxed` DefectEntry objects should correctly determine the final relaxed
    defect symmetry (and orientational degeneracy) - otherwise periodicity-breaking
    prevents this.

    If periodicity-breaking prevents auto-symmetry determination, you can manually
    determine the relaxed defect and bulk-site point symmetries, and/or orientational
    degeneracy, from visualising the structures (e.g. using VESTA)(can use
    ``get_orientational_degeneracy`` to obtain the corresponding orientational
    degeneracy factor for given defect/bulk-site point symmetries) and setting the
    corresponding values in the
    ``calculation_metadata['relaxed point symmetry']/['bulk site symmetry']`` and/or
    ``degeneracy_factors['orientational degeneracy']`` attributes.
    Note that the bulk-site point symmetry corresponds to that of ``DefectEntry.defect``,
    or equivalently ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``,
    which for vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `final relaxed` interstitial
    site when placed in the (unrelaxed) bulk structure.
    The degeneracy factor is used in the calculation of defect/carrier concentrations
    and Fermi level behaviour (see e.g. https://doi.org/10.1039/D2FD00043A &
    https://doi.org/10.1039/D3CS00432E).

    Args:
        defect_entry (DefectEntry): DefectEntry object. (Default = None)
        relaxed_point_group (str): Point group symmetry (e.g. "Td", "C3v" etc)
            of the `relaxed` defect structure, if already calculated / manually
            determined. Default is None (automatically calculated by doped).
        bulk_site_point_group (str): Point group symmetry (e.g. "Td", "C3v" etc)
            of the defect site in the bulk, if already calculated / manually
            determined. For vacancies/substitutions, this should match the site
            symmetry label from ``doped`` when generating the defect, while for
            interstitials it should be the point symmetry of the `final relaxed`
            interstitial site, when placed in the bulk structure.
            Default is None (automatically calculated by doped).
        bulk_symm_ops (list):
            List of symmetry operations of the defect_entry.bulk_supercell
            structure (used in determining the `unrelaxed` bulk site symmetry), to
            avoid re-calculating. Default is None (recalculates).
        defect_symm_ops (list):
            List of symmetry operations of the defect_entry.defect_supercell
            structure (used in determining the `relaxed` point symmetry), to
            avoid re-calculating. Default is None (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib`` to use when determining point
            symmetries and thus orientational degeneracies. Default is ``0.1``
            which matches that used by the ``Materials Project`` and is larger
            than the ``pymatgen`` default of ``0.01`` to account for residual
            structural noise in relaxed defect supercells.
            You may want to adjust for your system (e.g. if there are very slight
            octahedral distortions etc.).

    Returns:
        float: orientational degeneracy factor for the defect.
    """
    from doped.utils.symmetry import group_order_from_schoenflies, point_symmetry_from_defect_entry

    if defect_entry is None:
        if relaxed_point_group is None or bulk_site_point_group is None:
            raise ValueError(
                "Either the DefectEntry or both defect and bulk site point group symbols must be "
                "provided for doped to determine the orientational degeneracy! "
            )

    elif defect_entry.bulk_entry is None:
        raise ValueError(
            "bulk_entry must be set for defect_entry to determine the (relaxed) orientational degeneracy! "
            "(i.e. must be a parsed DefectEntry)"
        )

    if relaxed_point_group is None:
        # this will throw warning if auto-detected that supercell breaks trans symmetry
        relaxed_point_group = point_symmetry_from_defect_entry(
            defect_entry,  # type: ignore
            symm_ops=defect_symm_ops,  # defect not bulk symm_ops
            symprec=symprec,
            relaxed=True,  # relaxed
        )

    if bulk_site_point_group is None:
        bulk_site_point_group = point_symmetry_from_defect_entry(
            defect_entry,  # type: ignore
            symm_ops=bulk_symm_ops,  # bulk not defect symm_ops
            symprec=symprec,  # same symprec as relaxed_point_group for consistency
            relaxed=False,  # unrelaxed
        )

    # actually fine for split-vacancies (e.g. Ke's V_Sb in Sb2O5), or antisite-swaps etc:
    # (so avoid warning for now; user will be warned anyway if symmetry determination failing)
    # if orientational_degeneracy < 1 and not (
    #     defect_type == DefectType.Interstitial
    #     or (isinstance(defect_type, str) and defect_type.lower() == "interstitial")
    # ):
    #     raise ValueError(
    #         f"From the input/determined point symmetries, an orientational degeneracy factor of "
    #         f"{orientational_degeneracy} is predicted, which is less than 1, which is not reasonable "
    #         f"for vacancies/substitutions, indicating an error in the symmetry determination!"
    #     )

    return group_order_from_schoenflies(bulk_site_point_group) / group_order_from_schoenflies(
        relaxed_point_group
    )


def _get_bulk_supercell(defect_entry: DefectEntry):
    if hasattr(defect_entry, "bulk_supercell") and defect_entry.bulk_supercell:
        return defect_entry.bulk_supercell

    if (
        hasattr(defect_entry, "bulk_entry")
        and defect_entry.bulk_entry
        and hasattr(defect_entry.bulk_entry, "structure")
        and defect_entry.bulk_entry.structure
    ):
        return defect_entry.bulk_entry.structure

    return None


def _get_defect_supercell(defect_entry: DefectEntry):
    if hasattr(defect_entry, "defect_supercell") and defect_entry.defect_supercell:
        return defect_entry.defect_supercell

    if (
        hasattr(defect_entry, "sc_entry")
        and defect_entry.sc_entry
        and hasattr(defect_entry.sc_entry, "structure")
        and defect_entry.sc_entry.structure
    ):
        return defect_entry.sc_entry.structure

    return None


def _get_unrelaxed_defect_structure(defect_entry: DefectEntry):
    if (
        hasattr(defect_entry, "calculation_metadata")
        and defect_entry.calculation_metadata
        and "unrelaxed_defect_structure" in defect_entry.calculation_metadata
    ):
        return defect_entry.calculation_metadata["unrelaxed_defect_structure"]

    bulk_supercell = _get_bulk_supercell(defect_entry)

    if bulk_supercell is not None:
        from doped.analysis import defect_from_structures

        (
            _defect,
            _defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
            _defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
            # w/interstitials
            _defect_site_index,
            _bulk_site_index,
            _guessed_initial_defect_structure,
            unrelaxed_defect_structure,
            _bulk_voronoi_node_dict,
        ) = defect_from_structures(
            bulk_supercell,
            _get_defect_supercell(defect_entry),
            return_all_info=True,
            oxi_state="Undefined",  # don't need oxidation states for this
        )
        return unrelaxed_defect_structure

    return None


def _get_defect_supercell_bulk_site_coords(defect_entry: DefectEntry, relaxed=True):
    sc_defect_frac_coords = defect_entry.sc_defect_frac_coords
    site = None

    if not relaxed:
        site = _get_defect_supercell_site(defect_entry, relaxed=False)
    if sc_defect_frac_coords is None and site is None:
        site = _get_defect_supercell_site(defect_entry)
    if site is not None:
        sc_defect_frac_coords = site.frac_coords

    return sc_defect_frac_coords


def _get_defect_supercell_site(defect_entry: DefectEntry, relaxed=True):
    if not relaxed:
        if (  # noqa: SIM102
            hasattr(defect_entry, "calculation_metadata") and defect_entry.calculation_metadata
        ):
            if site := defect_entry.calculation_metadata.get("bulk_site"):
                return site

        # otherwise need to reparse info:
        from doped.analysis import defect_from_structures

        bulk_supercell = _get_bulk_supercell(defect_entry)
        defect_supercell = _get_defect_supercell(defect_entry)

        (
            _defect,
            _defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
            defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
            # w/interstitials
            defect_site_index,
            bulk_site_index,
            guessed_initial_defect_structure,
            unrelaxed_defect_structure,
            _bulk_voronoi_node_dict,
        ) = defect_from_structures(
            bulk_supercell,
            defect_supercell,
            return_all_info=True,
            oxi_state="Undefined",  # don't need oxidation states for this
        )

        # update any missing calculation_metadata:
        defect_entry.calculation_metadata["guessed_initial_defect_structure"] = (
            defect_entry.calculation_metadata.get(
                "guessed_initial_defect_structure", guessed_initial_defect_structure
            )
        )
        defect_entry.calculation_metadata["defect_site_index"] = defect_entry.calculation_metadata.get(
            "defect_site_index", defect_site_index
        )
        defect_entry.calculation_metadata["bulk_site_index"] = defect_entry.calculation_metadata.get(
            "bulk_site_index", bulk_site_index
        )
        defect_entry.calculation_metadata["unrelaxed_defect_structure"] = (
            defect_entry.calculation_metadata.get("unrelaxed_defect_structure", unrelaxed_defect_structure)
        )

        # add displacement from (guessed) initial site to final defect site:
        if defect_site_index is not None:  # not a vacancy
            guessed_initial_site = guessed_initial_defect_structure[defect_site_index]
            final_site = defect_supercell[defect_site_index]
            guessed_displacement = final_site.distance(guessed_initial_site)
            defect_entry.calculation_metadata["guessed_initial_defect_site"] = (
                defect_entry.calculation_metadata.get("guessed_initial_defect_site", guessed_initial_site)
            )
            defect_entry.calculation_metadata["guessed_defect_displacement"] = (
                defect_entry.calculation_metadata.get("guessed_defect_displacement", guessed_displacement)
            )
            defect_entry.calculation_metadata["bulk_site_index"] = defect_entry.calculation_metadata.get(
                "bulk_site_index", bulk_site_index
            )
        else:  # vacancy
            defect_entry.calculation_metadata["guessed_initial_defect_site"] = (
                defect_entry.calculation_metadata.get(
                    "guessed_initial_defect_site", bulk_supercell[bulk_site_index]
                )
            )
            defect_entry.calculation_metadata[
                "guessed_defect_displacement"
            ] = defect_entry.calculation_metadata.get(
                "guessed_defect_displacement", None
            )  # type: ignore

        if bulk_site_index is None:  # interstitial
            defect_entry.calculation_metadata["bulk_site"] = defect_entry.calculation_metadata.get(
                "bulk_site", defect_site_in_bulk
            )
        else:
            defect_entry.calculation_metadata["bulk_site"] = defect_entry.calculation_metadata.get(
                "bulk_site", bulk_supercell[bulk_site_index]
            )

        return defect_entry.calculation_metadata["bulk_site"]

    if hasattr(defect_entry, "defect_supercell_site") and defect_entry.defect_supercell_site:
        return defect_entry.defect_supercell_site

    if defect_entry.sc_defect_frac_coords is not None:
        return PeriodicSite(
            defect_entry.defect.site.species,
            defect_entry.sc_defect_frac_coords,
            _get_defect_supercell(defect_entry).lattice,
        )

    return None


def simple_spin_degeneracy_from_charge(structure, charge_state: int = 0) -> int:
    """
    Get the defect spin degeneracy from the supercell and charge state,
    assuming either simple singlet (S=0) or doublet (S=1/2) behaviour.

    Even-electron defects are assumed to have a singlet ground state, and odd-
    electron defects are assumed to have a doublet ground state.
    """
    total_Z = int(sum(Element(elt).Z * num for elt, num in structure.composition.as_dict().items()))
    return int((total_Z + charge_state) % 2 + 1)


def _defect_spin_degeneracy_from_vasprun(defect_vr: Vasprun, charge_state: int = 0) -> int:
    """
    Get the defect spin degeneracy from the vasprun output, assuming either
    singlet (S=0) or doublet (S=1/2) behaviour.

    Even-electron defects are assumed to have a singlet ground state, and odd-
    electron defects are assumed to have a doublet ground state.
    """
    return simple_spin_degeneracy_from_charge(defect_vr.final_structure, charge_state)


def defect_charge_from_vasprun(defect_vr: Vasprun, charge_state: Optional[int]) -> int:
    """
    Determine the defect charge state from the defect vasprun, and compare to
    the manually-set charge state if provided.

    Args:
        defect_vr (Vasprun):
            Defect ``pymatgen`` ``Vasprun`` object.
        charge_state (int):
            Manually-set charge state for the defect, to check if it matches
            the auto-determined charge state.

    Returns:
        int: The auto-determined defect charge state.
    """
    auto_charge = None

    try:
        if defect_vr.incar.get("NELECT") is None:
            auto_charge = 0  # neutral defect if NELECT not specified

        else:
            defect_nelect = defect_vr.parameters.get("NELECT")
            neutral_defect_nelect = get_neutral_nelect_from_vasprun(defect_vr)

            auto_charge = -1 * (defect_nelect - neutral_defect_nelect)

            if auto_charge is None or abs(auto_charge) >= 10:
                neutral_defect_nelect = get_neutral_nelect_from_vasprun(defect_vr, skip_potcar_init=True)
                try:
                    auto_charge = -1 * (defect_nelect - neutral_defect_nelect)

                except Exception as e:
                    auto_charge = None
                    if charge_state is None:
                        raise RuntimeError(
                            "Defect charge cannot be automatically determined as POTCARs have not been "
                            "setup with pymatgen (see Step 2 at "
                            "https://github.com/SMTG-Bham/doped#installation). Please specify defect "
                            "charge manually using the `charge_state` argument, or set up POTCARs with "
                            "pymatgen."
                        ) from e

            if auto_charge is not None and abs(auto_charge) >= 10:  # crazy charge state predicted
                raise RuntimeError(
                    f"Auto-determined defect charge q={int(auto_charge):+} is unreasonably large. "
                    f"Please specify defect charge manually using the `charge` argument."
                )

        if (
            charge_state is not None
            and auto_charge is not None
            and int(charge_state) != int(auto_charge)
            and abs(auto_charge) < 5
        ):
            warnings.warn(
                f"Auto-determined defect charge q={int(auto_charge):+} does not match specified charge "
                f"q={int(charge_state):+}. Will continue with specified charge_state, but beware!"
            )

        if charge_state is None and auto_charge is not None:
            charge_state = auto_charge

    except Exception as e:
        if charge_state is None:
            raise e

    if charge_state is None:
        raise RuntimeError(
            "Defect charge could not be automatically determined from the defect calculation outputs. "
            "Please manually specify defect charge using the `charge_state` argument."
        )

    return charge_state


def _get_bulk_locpot_dict(bulk_path, quiet=False):
    bulk_locpot_path, multiple = _get_output_files_and_check_if_multiple("LOCPOT", bulk_path)
    if multiple and not quiet:
        _multiple_files_warning(
            "LOCPOT",
            bulk_path,
            bulk_locpot_path,
            dir_type="bulk",
        )
    bulk_locpot = get_locpot(bulk_locpot_path)
    return {str(k): bulk_locpot.get_average_along_axis(k) for k in [0, 1, 2]}


def _get_bulk_site_potentials(bulk_path, quiet=False):
    from doped.corrections import _raise_incomplete_outcar_error  # avoid circular import

    bulk_outcar_path, multiple = _get_output_files_and_check_if_multiple("OUTCAR", bulk_path)
    if multiple and not quiet:
        _multiple_files_warning(
            "OUTCAR",
            bulk_path,
            bulk_outcar_path,
            dir_type="bulk",
        )
    bulk_outcar = get_outcar(bulk_outcar_path)

    if bulk_outcar.electrostatic_potential is None:
        _raise_incomplete_outcar_error(bulk_outcar_path, dir_type="bulk")

    return -1 * np.array(bulk_outcar.electrostatic_potential)


def _update_defect_entry_charge_corrections(defect_entry, charge_correction_type):
    meta = defect_entry.calculation_metadata[f"{charge_correction_type}_meta"]
    corr = (
        meta[f"{charge_correction_type}_electrostatic"]
        + meta[f"{charge_correction_type}_potential_alignment_correction"]
    )
    defect_entry.corrections.update({f"{charge_correction_type}_charge_correction": corr})


_vasp_file_parsing_action_dict = {
    "vasprun.xml": "parse the calculation energy and metadata.",
    "OUTCAR": "parse core levels and compute the Kumagai (eFNV) image charge correction.",
    "LOCPOT": "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
}


def _multiple_files_warning(file_type, directory, chosen_filepath, action=None, dir_type="bulk"):
    filename = os.path.basename(chosen_filepath)
    if action is None:
        action = _vasp_file_parsing_action_dict[file_type]
    warnings.warn(
        f"Multiple `{file_type}` files found in {dir_type} directory: {directory}. Using {filename} to "
        f"{action}"
    )
