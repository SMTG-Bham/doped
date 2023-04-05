# coding: utf-8
"""
Parses the computed data from VASP defect calculations.
"""
# from __future__ import unicode_literals
from __future__ import division

import glob
import logging
import os
import warnings

import numpy as np
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects.core import (
    DefectEntry,
    Interstitial,
    Substitution,
    Vacancy,
)
from pymatgen.analysis.defects.defect_compatibility import DefectCompatibility
from pymatgen.analysis.defects.utils import TopographyAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Potcar, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Poscar, Vasprun
from pymatgen.util.coord import pbc_diff

from doped.pycdt.core import _chemical_potentials

angstrom = "\u212B"  # unicode symbol for angstrom to print in strings

# globally ignore these POTCAR warnings
warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
warnings.filterwarnings("ignore", message="Ignoring unknown variable type")

# until updated from pymatgen==2022.7.25 :
warnings.filterwarnings(
    "ignore", message="Using `tqdm.autonotebook.tqdm` in notebook mode"
)
warnings.filterwarnings(
    "ignore", message="`np.int` is a deprecated alias for the builtin `int`"
)
warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")


def custom_formatwarning(msg, *args, **kwargs):
    """Reformat warnings to just print the warning message"""
    return f"{msg}\n"


warnings.formatwarning = custom_formatwarning


def convert_cd_to_de(cd, b_cse):
    """
    As of pymatgen v2.0, ComputedDefect objects were deprecated in favor
    of DefectEntry objects in pymatgen.analysis.defects.core
    This function takes a ComputedDefect (either as a dict or object) and
    converts it into a DefectEntry object in order to handle legacy
    PyCDT creation within the current paradigm of PyCDT.

    :param cd (dict or ComputedDefect object): ComputedDefect as an object or as a dictionary
    :params b_cse (dict or ComputedStructureEntry object): ComputedStructureEntry of bulk entry
        associated with the ComputedDefect.
    :return: de (DefectEntry): Resulting DefectEntry object
    """
    if isinstance(cd, dict):
        cd = cd.as_dict()
    if isinstance(b_cse, dict):
        b_cse = b_cse.as_dict()

    bulk_sc_structure = Structure.from_dict(b_cse["structure"])

    # modify defect_site as required for Defect object, confirming site exists in bulk structure
    site_cls = cd["site"]
    defect_site = PeriodicSite.from_dict(site_cls)
    def_nom = cd["name"].lower()
    if "sub_" in def_nom or "as_" in def_nom:
        # modify site object for substitution site of Defect object
        site_cls["species"][0]["element"] = cd["name"].split("_")[2]
        defect_site = PeriodicSite.from_dict(site_cls)

    poss_deflist = sorted(
        bulk_sc_structure.get_sites_in_sphere(
            defect_site.coords, 0.2, include_index=True
        ),
        key=lambda x: x[1],
    )
    if len(poss_deflist) != 1:
        raise ValueError(
            "ComputedDefect to DefectEntry conversion failed. "
            "Could not determine periodic site position in bulk supercell."
        )

    # create defect object
    if "vac_" in def_nom:
        defect_obj = Vacancy(bulk_sc_structure, defect_site, charge=cd["charge"])
    elif "as_" in def_nom or "sub_" in def_nom:
        defect_obj = Substitution(bulk_sc_structure, defect_site, charge=cd["charge"])
    elif "int_" in def_nom:
        defect_obj = Interstitial(bulk_sc_structure, defect_site, charge=cd["charge"])
    else:
        raise ValueError(f"Could not recognize defect type for {cd['name']}")

    # assign proper energy and parameter metadata
    uncorrected_energy = cd["entry"]["energy"] - b_cse["energy"]
    def_path = os.path.split(cd["entry"]["data"]["locpot_path"])[0]
    bulk_path = os.path.split(b_cse["data"]["locpot_path"])[0]
    p = {
        "defect_path": def_path,
        "bulk_path": bulk_path,
        "encut": cd["entry"]["data"]["encut"],
    }

    de = DefectEntry(defect_obj, uncorrected_energy, parameters=p)

    return de


def get_vasprun(vasprun_path, **kwargs):
    """Read the vasprun.xml(.gz) file as a pymatgen Vasprun object"""
    vasprun_path = str(vasprun_path)  # convert to string if Path object
    warnings.filterwarnings(
        "ignore", category=UnknownPotcarWarning
    )  # Ignore POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings(
        "ignore", message="No POTCAR file with matching TITEL fields"
    )
    if os.path.exists(vasprun_path):
        vasprun = Vasprun(vasprun_path)
        read_vasprun_path = vasprun_path
    elif os.path.exists(vasprun_path + ".gz", **kwargs):
        vasprun = Vasprun(vasprun_path + ".gz", **kwargs)
        read_vasprun_path = vasprun_path + ".gz"
    else:
        raise FileNotFoundError(
            f"""vasprun.xml(.gz) not found at {vasprun_path}(.gz). Needed for parsing defect 
            calculations."""
        )
    return vasprun, read_vasprun_path


def get_locpot(locpot_path):
    """Read the LOCPOT(.gz) file as a pymatgen Locpot object"""
    locpot_path = str(locpot_path)  # convert to string if Path object
    if os.path.exists(locpot_path):
        locpot = Locpot.from_file(locpot_path)
    elif os.path.exists(locpot_path + ".gz"):
        locpot = Locpot.from_file(locpot_path + ".gz")
    else:
        raise FileNotFoundError(
            f"""LOCPOT(.gz) not found at {locpot_path}(.gz). Needed for calculating the 
            Freysoldt (FNV) image charge corrections."""
        )
    return locpot


def get_outcar(outcar_path):
    """Read the OUTCAR(.gz) file as a pymatgen Outcar object"""
    outcar_path = str(outcar_path)  # convert to string if Path object
    if os.path.exists(outcar_path):
        outcar = Outcar(outcar_path)
    elif os.path.exists(outcar_path + ".gz"):
        outcar = Outcar(outcar_path + ".gz")
    else:
        raise FileNotFoundError(
            f"""OUTCAR(.gz) not found at {outcar_path}(.gz). Needed for calculating the Kumagai (
            eFNV) image charge corrections."""
        )
    return outcar


def get_defect_type_and_composition_diff(bulk, defect):
    """Get the difference in composition between a bulk structure and a defect structure.
    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for extrinsic species
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
            "Could not determine defect type from composition difference of bulk "
            "and defect structures."
        )

    return defect_type, composition_diff


def get_defect_site_idxs_and_unrelaxed_structure(
    bulk, defect, defect_type, composition_diff, unique_tolerance=1
):
    """Get the defect site and unrelaxed structure.
    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for extrinsic species
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
                raise RuntimeError(
                    "Could not uniquely determine site of new species in defect structure"
                )

            defect_site_idx = list(
                set(np.arange(len(defect_new_species_coords), dtype=int))
                - set(site_matches)
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
        distances = np.linalg.norm(
            pbc_diff(bulk_old_species_coords, defect_coords), axis=1
        )
        original_site_idx = distances.argmin()

        # if there are any other matches with a distance within unique_tolerance of the located
        # site then unique matching failed
        if (
            len(distances[distances < distances[original_site_idx] * unique_tolerance])
            > 1
        ):
            raise RuntimeError(
                "Could not uniquely determine site of old species in bulk structure"
            )

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
        unrelaxed_defect_structure.insert(
            defect_site_idx, new_species, bulk_coords[bulk_site_idx]
        )

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
            raise RuntimeError(
                "Could not uniquely determine site of vacancy in defect structure"
            )

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
                raise RuntimeError(
                    "Could not uniquely determine site of interstitial in defect structure"
                )

            defect_site_idx = list(
                set(np.arange(len(defect_new_species_coords), dtype=int))
                - set(site_matches)
            )[0]

        else:  # extrinsic interstitial
            defect_site_idx = 0

        defect_site_coords = defect_new_species_coords[defect_site_idx]

        # Get the site index of the defect that was used in the VASP calculation
        defect_site_idx = defect_new_species_idx[defect_site_idx]

        # create unrelaxed defect structure
        unrelaxed_defect_structure = bulk.copy()
        # Place defect in same location as output from DFT
        unrelaxed_defect_structure.insert(
            defect_site_idx, new_species, defect_site_coords
        )
        bulk_site_idx = None

    else:
        raise ValueError(f"Invalid defect type: {defect_type}")

    return (
        bulk_site_idx,
        defect_site_idx,
        unrelaxed_defect_structure,
    )


class SingleDefectParser:
    # TODO: Given this is our recommended workflow, should streamline this similar to in
    #  `vasp_input.py`, where it tries to run the freysoldt/kumagai loader (chosen by an
    #  optional imput parameter, if not set check if OUTCAR/LOCPOT present), throw warning if not
    #  successful, then `get_stdrd_metadata()`, `get_bulk_gap_data()` and `run_compatibility()` (
    #  again throwing warnings for any of these that don't work) – so replacing 4+ function calls
    #  with just one.
    # TODO: Should also add check the bulk and defect KPOINTS/INCAR/POTCAR/POSCAR (size) settings
    #  are compatible, and throw warning if not.
    # TODO: Add `check_defects_compatibility()` function that checks the bulk and defect
    #  KPOINTS/INCAR/POTCAR/POSCAR (size) settings for all defects in the supplied defect_dict
    #  are compatible, if not throw warnings and say what the differences are. Should recommend
    #  using this in the example notebook if a user has parsed the defects individually (rather
    #  than with the single looping function described below):
    # TODO: Add a function that loops over all the defects in a directory (with `defect_dir = .`,
    #  and `subfolder = vasp_ncl` options) and parses them all, returning a dictionary of defect
    #  entries, with the defect name as the key. (i.e. doing the loop in the example notebook).
    #  Show both this function and the individual function calls in the example notebook. Benefit
    #  of this one is that we can then auto-run `check_defects_compatibility()` at the end of
    #  parsing the full defects dict.
    def __init__(
        self,
        defect_entry,
        compatibility=DefectCompatibility(
            plnr_avg_var_tol=0.01,
            plnr_avg_minmax_tol=0.3,
            atomic_site_var_tol=0.025,
            atomic_site_minmax_tol=0.3,
            tot_relax_tol=5.0,
            defect_tot_relax_tol=5.0,
            use_bandfilling=False,  # don't include bandfilling by default
            use_bandedgeshift=False,  # don't include band edge shift by default
        ),
        defect_vr=None,
        bulk_vr=None,
    ):
        """
        Parse a defect object using features that resemble that of a standard
        DefectBuilder object (emmet), but without the requirement of atomate.
        Also allows for use of DefectCompatibility object within pymatgen

        :param defect_entry (DefectEntry): DefectEntry of interest (using the bulk supercell as
        bulk_structure)
            NOTE: to make use of methods within the class, bulk_path and and defect_path
            must exist within the defect_entry parameters class.
        :param compatibility (DefectCompatibility): Compatibility class instance for
            performing charge correction compatibility analysis on defect entry.
        :param defect_vr (Vasprun):
        :param bulk_vr (Vasprun):

        """
        self.defect_entry = defect_entry
        self.compatibility = compatibility
        self.defect_vr = defect_vr
        self.bulk_vr = bulk_vr

    @staticmethod
    def from_paths(
        path_to_defect,
        path_to_bulk,
        dielectric,
        defect_charge,
        mpid=None,
        compatibility=DefectCompatibility(
            plnr_avg_var_tol=0.01,
            plnr_avg_minmax_tol=0.3,
            atomic_site_var_tol=0.025,
            atomic_site_minmax_tol=0.3,
            tot_relax_tol=5.0,
            defect_tot_relax_tol=5.0,
            use_bandfilling=False,  # don't include bandfilling by default
            use_bandedgeshift=False,  # don't include band edge shift by default
        ),
        initial_defect_structure=None,
    ):
        """
        Identify defect object based on file paths. Minimal parsing performing for
        instantiating the SingleDefectParser class.

        Args:
        path_to_defect (str): path to defect folder of interest (with vasprun.xml(.gz))
        path_to_bulk (str): path to bulk folder of interest (with vasprun.xml(.gz))
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
        defect_charge (int):  charge of defect
        mpid (str):  Materials Project ID of bulk structure
        compatibility (DefectCompatibility): Compatibility class instance for
            performing charge correction compatibility analysis on defect entry.
        initial_defect_structure (str):  Path to the unrelaxed defect structure,
            if structure matching with the relaxed defect structure(s) fails.

        Return:
            Instance of the SingleDefectParser class.
        """
        # check if dielectric in required 3x3 matrix format
        if not isinstance(dielectric, (float, int)):
            dielectric = np.array(dielectric)
            if dielectric.shape == (3,):
                dielectric = np.diag(dielectric)
            elif dielectric.shape != (3, 3):
                raise ValueError(
                    f"Dielectric constant must be a float/int or a 3x1 matrix or 3x3 matrix, "
                    f"got type {type(dielectric)} and shape {dielectric.shape}"
                )

        parameters = {
            "bulk_path": path_to_bulk,
            "defect_path": path_to_defect,
            "dielectric": dielectric,
            "mpid": mpid,
        }

        # add bulk simple properties
        bulk_vr, bulk_vr_path = get_vasprun(os.path.join(path_to_bulk, "vasprun.xml"))
        bulk_energy = bulk_vr.final_energy
        bulk_sc_structure = bulk_vr.initial_structure.copy()

        # add defect simple properties
        defect_vr, defect_vr_path = get_vasprun(
            os.path.join(path_to_defect, "vasprun.xml")
        )
        defect_energy = defect_vr.final_energy
        # Can specify initial defect structure (to help PyCDT find the defect site if
        # multiple relaxations were required, else use from defect relaxation OUTCAR:
        if initial_defect_structure:
            initial_defect_structure = Poscar.from_file(
                initial_defect_structure
            ).structure.copy()
        else:
            initial_defect_structure = defect_vr.initial_structure.copy()

        # Add initial defect structure to parameters, so it can be pulled later on
        # (eg. for Kumagai loader)
        parameters["initial_defect_structure"] = initial_defect_structure

        # identify defect site, structural information, and create defect object
        try:
            def_type, comp_diff = get_defect_type_and_composition_diff(
                bulk_sc_structure, initial_defect_structure
            )
        except RuntimeError as exc:
            raise ValueError(
                "Could not identify defect type from number of sites in structure: "
                f"{len(bulk_sc_structure)} in bulk vs. {len(initial_defect_structure)} in defect?"
            ) from exc

        bulk_site_idx = None
        defect_site_idx = None
        unrelaxed_defect_structure = None
        # Try automatic defect site detection - this gives us the "unrelaxed" defect structure
        try:
            (
                bulk_site_idx,
                defect_site_idx,
                unrelaxed_defect_structure,
            ) = get_defect_site_idxs_and_unrelaxed_structure(
                bulk_sc_structure, initial_defect_structure, def_type, comp_diff
            )
        except RuntimeError as exc:
            # if auto site-matching failed, try use transformation.json
            # The goal is to find the `defect_site_idx` or `defect_site_idx` based on the
            # tranformation.
            transformation_path = os.path.join(path_to_defect, "transformation.json")
            if not os.path.exists(transformation_path):  # try next folder up
                orig_transformation_path = transformation_path
                transformation_path = os.path.join(
                    os.path.dirname(os.path.normpath(path_to_defect)),
                    "transformation.json",
                )
                if os.path.exists(transformation_path):
                    print(
                        f"No transformation file found at {orig_transformation_path}, but found "
                        f"one at {transformation_path}. Using this for defect parsing."
                    )

            if os.path.exists(transformation_path):
                tf = loadfn(transformation_path)
                site = tf["defect_supercell_site"]
                if def_type == "vacancy":
                    poss_deflist = sorted(
                        bulk_sc_structure.get_sites_in_sphere(
                            site.coords, 0.2, include_index=True
                        ),
                        key=lambda x: x[1],
                    )
                    searched = "bulk_supercell"
                    if poss_deflist:
                        bulk_site_idx = poss_deflist[0][2]
                else:
                    poss_deflist = sorted(
                        initial_defect_structure.get_sites_in_sphere(
                            site.coords, 2.5, include_index=True
                        ),
                        key=lambda x: x[1],
                    )
                    searched = "initial_defect_structure"
                    if poss_deflist:
                        defect_site_idx = poss_deflist[0][2]
                if not poss_deflist:
                    raise ValueError(
                        f"{transformation_path} specified defect site {site}, but could not find "
                        f"it in {searched}. Abandoning parsing."
                    ) from exc
                if poss_deflist[0][1] > 1:
                    site_matched_defect = poss_deflist[0]  # pymatgen Neighbor object
                    offsite_warning = (
                        f"Site-matching has determined {site_matched_defect.species} at "
                        f"{site_matched_defect.coords} as the defect site, located "
                        f"{site_matched_defect.nn_distance:.2f} {angstrom} from its initial "
                        f"position. This may incur small errors in the charge correction."
                    )
                    warnings.warn(message=offsite_warning)

            else:
                raise RuntimeError(
                    f"Could not identify {def_type} defect site in defect structure. "
                    f"Try supplying the initial defect structure to "
                    f"SingleDefectParser.from_paths(), or making sure the doped "
                    f"transformation.json files are in the defect directory."
                ) from exc

        if def_type == "vacancy":
            defect_site = bulk_sc_structure[bulk_site_idx]
        else:
            if unrelaxed_defect_structure:
                defect_site = unrelaxed_defect_structure[defect_site_idx]
            else:
                defect_site = initial_defect_structure[defect_site_idx]

        if unrelaxed_defect_structure:
            if def_type == "interstitial":
                # get closest Voronoi site in bulk supercell to final interstitial site as this is
                # likely to be the initial interstitial site
                try:
                    struc_and_node_dict = loadfn("./bulk_voronoi_nodes.json")
                    if not StructureMatcher(
                        stol=0.05,
                        primitive_cell=False,
                        scale=False,
                        attempt_supercell=False,
                        allow_subset=False,
                    ).fit(struc_and_node_dict["bulk_supercell"], bulk_sc_structure):
                        warnings.warn(
                            "Previous bulk_voronoi_nodes.json detected, but does not "
                            "match current bulk supercell. Recalculating Voronoi nodes."
                        )
                        raise FileNotFoundError

                    voronoi_frac_coords = struc_and_node_dict["Voronoi nodes"]

                except FileNotFoundError:  # first time parsing
                    topography = TopographyAnalyzer(
                        bulk_sc_structure,
                        bulk_sc_structure.symbol_set,
                        [],
                        check_volume=False,
                    )
                    topography.cluster_nodes()
                    topography.remove_collisions()
                    voronoi_frac_coords = [
                        site.frac_coords for site in topography.vnodes
                    ]
                    struc_and_node_dict = {
                        "bulk_supercell": bulk_sc_structure,
                        "Voronoi nodes": voronoi_frac_coords,
                    }
                    dumpfn(
                        struc_and_node_dict, "./bulk_voronoi_nodes.json"
                    )  # for efficient
                    # parsing of multiple defects at once
                    print(
                        "Saving parsed Voronoi sites (for interstitial site-matching) to "
                        "bulk_voronoi_sites.json to speed up future parsing."
                    )

                closest_node_frac_coords = min(
                    voronoi_frac_coords,
                    key=lambda node: defect_site.distance_and_image_from_frac_coords(
                        node
                    )[0],
                )
                int_site = unrelaxed_defect_structure[defect_site_idx]
                unrelaxed_defect_structure.remove_sites([defect_site_idx])
                unrelaxed_defect_structure.insert(
                    defect_site_idx,  # Place defect at same position as in DFT calculation
                    int_site.species_string,
                    closest_node_frac_coords,
                    coords_are_cartesian=False,
                    validate_proximity=True,
                )
                defect_site = unrelaxed_defect_structure[defect_site_idx]

            # Use the unrelaxed_defect_structure to fix the initial defect structure
            initial_defect_structure = reorder_unrelaxed_structure(
                unrelaxed_defect_structure, initial_defect_structure
            )
            parameters["initial_defect_structure"] = initial_defect_structure
            parameters["unrelaxed_defect_structure"] = unrelaxed_defect_structure
        else:
            warnings.warn(
                "Cannot determine the unrelaxed `initial_defect_structure`. Please ensure the "
                "`initial_defect_structure` is indeed unrelaxed."
            )

        for_monty_defect = {
            "@module": "pymatgen.analysis.defects.core",
            "@class": def_type.capitalize(),
            "charge": defect_charge,
            "structure": bulk_sc_structure,
            "defect_site": defect_site,
        }
        defect = MontyDecoder().process_decoded(for_monty_defect)

        if unrelaxed_defect_structure:
            # only do StructureMatcher test if unrelaxed structure exists
            test_defect_structure = defect.generate_defect_structure()
            if not StructureMatcher(
                stol=0.25,
                primitive_cell=False,
                scale=False,
                attempt_supercell=False,
                allow_subset=False,
            ).fit(test_defect_structure, unrelaxed_defect_structure):
                # NOTE: this does not insure that cartesian coordinates or indexing are identical
                raise ValueError(
                    "Error in defect object matching! Unrelaxed structure (1st below) "
                    "does not match pymatgen defect.generate_defect_structure() "
                    f"(2nd below):\n{unrelaxed_defect_structure}"
                    f"\n{test_defect_structure}"
                )

        defect_entry = DefectEntry(
            defect, defect_energy - bulk_energy, corrections={}, parameters=parameters
        )

        return SingleDefectParser(
            defect_entry,
            compatibility=compatibility,
            defect_vr=defect_vr,
            bulk_vr=bulk_vr,
        )

    def freysoldt_loader(self, bulk_locpot=None):
        """
        Load metadata required for performing Freysoldt correction
        requires "bulk_path" and "defect_path" to be loaded to DefectEntry parameters dict.
        Can read gunzipped "LOCPOT.gz" files as well.

        Args:
            bulk_locpot (Locpot): Add bulk Locpot object for expedited parsing.
                If None, will load from file path variable bulk_path
        Return:
            bulk_locpot object for reuse by another defect entry (for expedited parsing)
        """
        if not self.defect_entry.charge:
            # dont need to load locpots if charge is zero
            return None

        if not bulk_locpot:
            bulk_locpot_path = os.path.join(
                self.defect_entry.parameters["bulk_path"], "LOCPOT"
            )
            bulk_locpot = get_locpot(bulk_locpot_path)

        def_locpot_path = os.path.join(
            self.defect_entry.parameters["defect_path"], "LOCPOT"
        )
        def_locpot = get_locpot(def_locpot_path)

        axis_grid = [def_locpot.get_axis_grid(i) for i in range(3)]
        bulk_planar_averages = [bulk_locpot.get_average_along_axis(i) for i in range(3)]
        defect_planar_averages = [
            def_locpot.get_average_along_axis(i) for i in range(3)
        ]

        self.defect_entry.parameters.update(
            {
                "axis_grid": axis_grid,
                "bulk_planar_averages": bulk_planar_averages,
                "defect_planar_averages": defect_planar_averages,
                "defect_frac_sc_coords": self.defect_entry.site.frac_coords,
            }
        )
        if "unrelaxed_defect_structure" in self.defect_entry.parameters:
            self.defect_entry.parameters.update(
                {
                    "initial_defect_structure": self.defect_entry.parameters[
                        "unrelaxed_defect_structure"
                    ],
                }
            )

        return bulk_locpot

    # noinspection DuplicatedCode
    def kumagai_loader(self, bulk_outcar=None):
        """
        Load metadata required for performing Kumagai correction
        requires "bulk_path" and "defect_path" to be loaded to DefectEntry parameters dict.

        Args:
            bulk_outcar (Outcar): Add bulk Outcar object for expedited parsing.
                If None, will load from file path variable bulk_path
        Return:
            bulk_outcar object for reuse by another defect entry (for expedited parsing)
        """
        if not self.defect_entry.charge:
            # dont need to load outcars if charge is zero
            return None

        if not bulk_outcar:
            bulk_outcar_path = os.path.join(
                self.defect_entry.parameters["bulk_path"], "OUTCAR"
            )
            bulk_outcar = get_outcar(bulk_outcar_path)
        else:
            bulk_outcar_path = "`bulk_outcar`"

        def_outcar_path = os.path.join(
            self.defect_entry.parameters["defect_path"], "OUTCAR"
        )
        def_outcar = get_outcar(def_outcar_path)

        bulk_atomic_site_averages = bulk_outcar.electrostatic_potential
        defect_atomic_site_averages = def_outcar.electrostatic_potential
        if not bulk_atomic_site_averages:
            raise ValueError(
                f"Unable to parse atomic core potentials from bulk `OUTCAR` at "
                f"{bulk_outcar_path}. This can happen if `ICORELEVEL` was not set "
                f"to 0 (= default) in the `INCAR`, or if the calculation was "
                f"finished prematurely with a `STOPCAR`. The Kumagai charge "
                f"correction cannot be computed without this data!"
            )

        if not defect_atomic_site_averages:
            raise ValueError(
                f"Unable to parse atomic core potentials from defect `OUTCAR` at "
                f"{def_outcar_path}. This can happen if `ICORELEVEL` was not set "
                f"to 0 (= default) in the `INCAR`, or if the calculation was "
                f"finished prematurely with a `STOPCAR`. The Kumagai charge "
                f"correction cannot be computed without this data!"
            )

        bulk_structure = self.defect_entry.bulk_structure
        bulksites = [site.frac_coords for site in bulk_structure]

        defect_structure = self.defect_entry.parameters["initial_defect_structure"]
        initsites = [site.frac_coords for site in defect_structure]

        distmatrix = bulk_structure.lattice.get_all_distances(
            bulksites, initsites
        )  # first index of this list is bulk index
        min_dist_with_index = [
            [
                min(distmatrix[bulk_index]),
                int(bulk_index),
                int(distmatrix[bulk_index].argmin()),
            ]
            for bulk_index in range(len(distmatrix))
        ]  # list of [min dist, bulk ind, defect ind]

        site_matching_indices = []
        if isinstance(self.defect_entry.defect, (Vacancy, Interstitial)):
            for mindist, bulk_index, defect_index in min_dist_with_index:
                if mindist < 0.5:
                    site_matching_indices.append([bulk_index, defect_index])

        elif isinstance(self.defect_entry.defect, Substitution):
            for mindist, bulk_index, defect_index in min_dist_with_index:
                species_match = (
                    bulk_structure[bulk_index].specie
                    == defect_structure[defect_index].specie
                )
                if mindist < 0.5 and species_match:
                    site_matching_indices.append([bulk_index, defect_index])

        self.defect_entry.parameters.update(
            {
                "bulk_atomic_site_averages": bulk_atomic_site_averages,
                "defect_atomic_site_averages": defect_atomic_site_averages,
                "site_matching_indices": site_matching_indices,
                "defect_frac_sc_coords": self.defect_entry.site.frac_coords,
            }
        )

        return bulk_outcar

    def get_stdrd_metadata(self):
        if not self.bulk_vr:
            path_to_bulk = self.defect_entry.parameters["bulk_path"]
            self.bulk_vr, bulk_vr_path = get_vasprun(
                os.path.join(path_to_bulk, "vasprun.xml")
            )

        if not self.defect_vr:
            path_to_defect = self.defect_entry.parameters["defect_path"]
            self.defect_vr, defect_vr_path = get_vasprun(
                os.path.join(path_to_defect, "vasprun.xml")
            )

        # standard bulk metadata
        bulk_energy = self.bulk_vr.final_energy
        bulk_sc_structure = self.bulk_vr.initial_structure
        self.defect_entry.parameters.update(
            {"bulk_energy": bulk_energy, "bulk_sc_structure": bulk_sc_structure}
        )

        # standard run metadata
        run_metadata = {}
        run_metadata.update(
            {
                "defect_incar": self.defect_vr.incar,
                "bulk_incar": self.bulk_vr.incar,
                "defect_kpoints": self.defect_vr.kpoints,
                "bulk_kpoints": self.bulk_vr.kpoints,
            }
        )
        run_metadata.update(
            {
                "incar_calctype_summary": {
                    k: self.defect_vr.incar.get(k, None)
                    if self.defect_vr.incar.get(k) not in ["None", "False", False]
                    else None
                    for k in [
                        "LHFCALC",
                        "HFSCREEN",
                        "IVDW",
                        "LUSE_VDW",
                        "LDAU",
                        "METAGGA",
                    ]
                }
            }
        )
        run_metadata.update(
            {
                "potcar_summary": {
                    "pot_spec": [
                        potelt["titel"] for potelt in self.defect_vr.potcar_spec
                    ],
                    "pot_labels": self.defect_vr.potcar_spec,
                    "pot_type": self.defect_vr.run_type,
                }
            }
        )

        self.defect_entry.parameters.update({"run_metadata": run_metadata.copy()})

        # standard defect run metadata
        self.defect_entry.parameters.update(
            {
                "final_defect_structure": self.defect_vr.final_structure,
                "defect_energy": self.defect_vr.final_energy,
            }
        )

        # grab defect energy and eigenvalue information for band filling and localization analysis
        eigenvalues = {
            spincls.value: eigdict.copy()
            for spincls, eigdict in self.defect_vr.eigenvalues.items()
        }
        kpoint_weights = self.defect_vr.actual_kpoints_weights[:]
        self.defect_entry.parameters.update(
            {"eigenvalues": eigenvalues, "kpoint_weights": kpoint_weights}
        )

    def get_bulk_gap_data(self, no_MP=True, actual_bulk_path=None):
        """Get bulk gap data from Materials Project or from local OUTCAR file.

        Args:
            no_MP (bool): If True, will not query MP for bulk gap data. (Default: True)
            actual_bulk_path (str): Path to bulk OUTCAR file for determining the band gap. If
                the VBM/CBM occur at reciprocal space points not included in the bulk supercell
                calculation, you should use this tag to point to a bulk bandstructure calculation
                instead. If None, will use self.defect_entry.parameters["bulk_path"].
        """
        if not self.bulk_vr:
            path_to_bulk = self.defect_entry.parameters["bulk_path"]
            self.bulk_vr, bulk_vr_path = get_vasprun(
                os.path.join(path_to_bulk, "vasprun.xml")
            )

        bulk_sc_structure = self.bulk_vr.initial_structure
        mpid = self.defect_entry.parameters["mpid"]

        if not mpid and not no_MP:
            try:
                with MPRester() as mp:
                    tmp_mplist = mp.get_entries_in_chemsys(
                        list(bulk_sc_structure.symbol_set)
                    )
                mplist = [
                    ment.entry_id
                    for ment in tmp_mplist
                    if ment.composition.reduced_composition
                    == bulk_sc_structure.composition.reduced_composition
                ]
            except Exception as exc:
                raise ValueError(
                    f"Error with querying MPRester for"
                    f" {bulk_sc_structure.composition.reduced_formula}:"
                ) from exc

            mpid_fit_list = []
            for trial_mpid in mplist:
                with MPRester() as mp:
                    mpstruct = mp.get_structure_by_material_id(trial_mpid)
                if StructureMatcher(
                    primitive_cell=True,
                    scale=False,
                    attempt_supercell=True,
                    allow_subset=False,
                ).fit(bulk_sc_structure, mpstruct):
                    mpid_fit_list.append(trial_mpid)

            if len(mpid_fit_list) == 1:
                mpid = mpid_fit_list[0]
                print(f"Single mp-id found for bulk structure:{mpid}.")
            elif len(mpid_fit_list) > 1:
                num_mpid_list = [int(mp.split("" - "")[1]) for mp in mpid_fit_list]
                num_mpid_list.sort()
                mpid = "mp-" + str(num_mpid_list[0])
                print(
                    f"Multiple mp-ids found for bulk structure:{mpid_fit_list}\nWill use lowest "
                    f"number mpid for bulk band structure = {mpid}."
                )
            else:
                print(
                    "Could not find bulk structure in MP database after tying the following "
                    f"list:\n{mplist}"
                )
                mpid = None

        vbm, cbm, bandgap = None, None, None
        gap_parameters = {}
        if mpid is not None and not no_MP:
            print(f"Using user-provided mp-id for bulk structure: {mpid}.")
            with MPRester() as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if bs:
                cbm = bs.get_cbm()["energy"]
                vbm = bs.get_vbm()["energy"]
                bandgap = bs.get_band_gap()["energy"]
                gap_parameters.update(
                    {"MP_gga_BScalc_data": bs.get_band_gap().copy()}
                )  # contains gap kpt transition

        if (
            vbm is None
            or bandgap is None
            or cbm is None
            or no_MP
            or not actual_bulk_path
        ):
            if mpid and bandgap is None:
                print(
                    f"WARNING: Mpid {mpid} was provided, but no bandstructure entry currently "
                    "exists for it. \nReverting to use of bulk supercell calculation for band "
                    "edge extrema."
                )
            if mpid and no_MP:
                print(
                    f"Mpid {mpid} was provided, but `no_MP` flag was set to True. \n"
                    "Reverting to use of bulk supercell calculation for band edge extrema."
                )

            gap_parameters.update(
                {"MP_gga_BScalc_data": None}
            )  # to signal no MP BS is used
            bandgap, cbm, vbm, _ = self.bulk_vr.eigenvalue_band_properties

        if actual_bulk_path:
            print(f"Using actual bulk path: {actual_bulk_path}")
            actual_bulk_vr, actual_bulk_vr_path = get_vasprun(
                os.path.join(actual_bulk_path, "vasprun.xml")
            )
            bandgap, cbm, vbm, _ = actual_bulk_vr.eigenvalue_band_properties

        gap_parameters.update({"mpid": mpid, "cbm": cbm, "vbm": vbm, "gap": bandgap})
        self.defect_entry.parameters.update(gap_parameters)

    def run_compatibility(self):
        # Set potalign so pymatgen can calculate bandfilling for 'neutral' defects
        # (possible for resonant dopants etc.)
        if (
            self.defect_entry.charge == 0
            and "potalign" not in self.defect_entry.parameters
        ):
            self.defect_entry.parameters["potalign"] = 0

        self.defect_entry = self.compatibility.process_entry(self.defect_entry)

        if "delocalization_meta" in self.defect_entry.parameters:
            delocalization_meta = self.defect_entry.parameters["delocalization_meta"]
            if (
                "plnr_avg" in delocalization_meta
                and not delocalization_meta["plnr_avg"]["is_compatible"]
            ) or (
                "atomic_site" in delocalization_meta
                and not delocalization_meta["atomic_site"]["is_compatible"]
            ):
                specific_delocalized_warning = f"""
Delocalization analysis has indicated that {self.defect_entry.name}
with charge {self.defect_entry.charge} may not be compatible with the chosen charge correction."""
                general_delocalization_warning = """
Note: Defects throwing a "delocalization analysis" warning may require a larger supercell for
accurate total energies. Recommended to look at the correction plots (i.e. run 
`get_correction_freysoldt(DefectEntry,...,plot=True)` from
`doped.pycdt.corrections.finite_size_charge_correction`) to visually determine if the charge 
correction scheme is still appropriate (replace 'freysoldt' with 'kumagai' if using anisotropic 
correction). You can also change the DefectCompatibility() tolerance settings via the 
`compatibility` parameter in `SingleDefectParser.from_paths()`."""
                warnings.warn(message=specific_delocalized_warning)
                warnings.warn(
                    message=general_delocalization_warning
                )  # should only print once

        if "num_hole_vbm" in self.defect_entry.parameters:
            if (
                self.compatibility.free_chg_cutoff
                < self.defect_entry.parameters["num_hole_vbm"]
            ) or (
                self.compatibility.free_chg_cutoff
                < self.defect_entry.parameters["num_elec_cbm"]
            ):
                num_holes = self.defect_entry.parameters["num_hole_vbm"]
                num_electrons = self.defect_entry.parameters["num_elec_cbm"]
                warnings.warn(
                    f"Eigenvalue analysis has determined that `num_hole_vbm` (= {num_holes}) or "
                    f"`num_elec_cbm` (= {num_electrons}) is significant (>2.1) for "
                    f"{self.defect_entry.name} with charge {self.defect_entry.charge}, indicating "
                    f"that there are many free charges in this defect supercell calculation and "
                    f"so the defect charge correction is unlikely to be accurate."
                )
                if "freysoldt_meta" in self.defect_entry.parameters:
                    frey_meta = self.defect_entry.parameters["freysoldt_meta"]
                    frey_corr = (
                        frey_meta["freysoldt_electrostatic"]
                        + frey_meta["freysoldt_potential_alignment_correction"]
                    )
                    self.defect_entry.corrections.update(
                        {"charge_correction": frey_corr}
                    )
                elif "kumagai_meta" in self.defect_entry.parameters:
                    kumagai_meta = self.defect_entry.parameters["kumagai_meta"]
                    kumagai_corr = (
                        kumagai_meta["kumagai_electrostatic"]
                        + kumagai_meta["kumagai_potential_alignment_correction"]
                    )
                    self.defect_entry.corrections.update(
                        {"charge_correction": kumagai_corr}
                    )

        if (
            self.defect_entry.charge != 0
            and self.defect_entry.corrections.get("charge_correction", None) is None
        ):
            warnings.warn(
                f"No charge correction computed for {self.defect_entry.name} with "
                f"charge {self.defect_entry.charge}, indicating problems with the "
                f"required data for the charge correction (i.e. dielectric constant, "
                f"LOCPOT files for Freysoldt correction, OUTCAR (with ICORELEVEL = 0) "
                f"for Kumagai correction etc)."
            )


class PostProcess:
    def __init__(self, root_fldr, mpid=None, mapi_key=None):
        """
        Post processing object for charged point-defect calculations.

        Args:
            root_fldr (str): path (relative) to directory
                in which data of charged point-defect calculations for
                a particular system are to be found;
            mpid (str): Materials Project ID of bulk structure;
                format "mp-X", where X is an integer;
            mapi_key (str): Materials API key to access database.

        """
        self._root_fldr = root_fldr
        self._mpid = mpid
        self._mapi_key = mapi_key
        self._substitution_species = set()

    def parse_defect_calculations(self):
        """
        Parses the defect calculations as DefectEntry objects,
        from a PyCDT root_fldr file structure.
        Charge correction is missing in the first run.
        """
        logger = logging.getLogger(__name__)
        parsed_defects = []
        subfolders = glob.glob(os.path.join(self._root_fldr, "vac_*"))
        subfolders += glob.glob(os.path.join(self._root_fldr, "as_*"))
        subfolders += glob.glob(os.path.join(self._root_fldr, "sub_*"))
        subfolders += glob.glob(os.path.join(self._root_fldr, "inter_*"))

        def get_vr_and_check_locpot(fldr):
            vr_file = os.path.join(fldr, "vasprun.xml")
            if not (os.path.exists(vr_file) or os.path.exists(vr_file + ".gz")):
                logger.warning("{} doesn't exit".format(vr_file))
                error_msg = ": Failure, vasprun.xml doesn't exist."
                return (None, error_msg)  # Further processing is not useful

            try:
                vr, vr_path = get_vasprun(vr_file, parse_potcar_file=False)
            except:
                logger.warning("Couldn't parse {}".format(vr_file))
                error_msg = ": Failure, couldn't parse vasprun.xml file."
                return (None, error_msg)

            if not vr.converged:
                logger.warning("Vasp calculation at {} not converged".format(fldr))
                error_msg = ": Failure, Vasp calculation not converged."
                return (None, error_msg)  # Further processing is not useful

            # Check if locpot exists
            locpot_file = os.path.join(fldr, "LOCPOT")
            if not (os.path.exists(locpot_file) or os.path.exists(locpot_file + ".gz")):
                logger.warning("{} doesn't exit".format(locpot_file))
                error_msg = ": Failure, LOCPOT doesn't exist"
                return (None, error_msg)  # Further processing is not useful

            return (vr, None)

        def get_encut_from_potcar(fldr):
            potcar_file = os.path.join(fldr, "POTCAR")
            if not os.path.exists(potcar_file):
                logger.warning("Not POTCAR in {} to parse ENCUT".format(fldr))
                error_msg = ": Failure, No POTCAR file."
                return (None, error_msg)  # Further processing is not useful

            try:
                potcar = Potcar.from_file(potcar_file)
            except:
                logger.warning("Couldn't parse {}".format(potcar_file))
                error_msg = ": Failure, couldn't read POTCAR file."
                return (None, error_msg)

            encut = max(ptcr_sngl.enmax for ptcr_sngl in potcar)
            return (encut, None)

        # get bulk entry information first
        fldr = os.path.join(self._root_fldr, "bulk")
        vr, error_msg = get_vr_and_check_locpot(fldr)
        if error_msg:
            logger.error("Abandoning parsing of the calculations")
            return {}
        bulk_energy = vr.final_energy
        bulk_sc_struct = vr.final_structure
        try:
            encut = vr.incar["ENCUT"]
        except:  # ENCUT not specified in INCAR. Read from POTCAR
            encut, error_msg = get_encut_from_potcar(fldr)
            if error_msg:
                logger.error("Abandoning parsing of the calculations")
                return {}

        trans_dict = loadfn(os.path.join(fldr, "transformation.json"), cls=MontyDecoder)
        supercell_size = trans_dict["supercell"]

        bulk_file_path = fldr
        bulk_entry = ComputedStructureEntry(
            bulk_sc_struct,
            bulk_energy,
            data={
                "bulk_path": bulk_file_path,
                "encut": encut,
                "supercell_size": supercell_size,
            },
        )

        # get defect entry information
        for fldr in subfolders:
            fldr_name = os.path.split(fldr)[1]
            chrg_fldrs = glob.glob(os.path.join(fldr, "charge*"))
            for chrg_fldr in chrg_fldrs:
                trans_dict = loadfn(
                    os.path.join(chrg_fldr, "transformation.json"), cls=MontyDecoder
                )
                chrg = trans_dict["charge"]
                vr, error_msg = get_vr_and_check_locpot(chrg_fldr)
                if error_msg:
                    logger.warning("Parsing the rest of the calculations")
                    continue
                if (
                    "substitution_specie" in trans_dict
                    and trans_dict["substitution_specie"]
                    not in bulk_sc_struct.symbol_set
                ):
                    self._substitution_species.add(trans_dict["substitution_specie"])
                elif (
                    "inter" in trans_dict["defect_type"]
                    and trans_dict["defect_site"].specie.symbol
                    not in bulk_sc_struct.symbol_set
                ):
                    # added because extrinsic interstitials don't have
                    # "substitution_specie" character...
                    trans_dict["substitution_specie"] = trans_dict[
                        "defect_site"
                    ].specie.symbol
                    self._substitution_species.add(
                        trans_dict["defect_site"].specie.symbol
                    )

                defect_type = trans_dict.get("defect_type", None)
                energy = vr.final_energy
                try:
                    encut = vr.incar["ENCUT"]
                except:  # ENCUT not specified in INCAR. Read from POTCAR
                    encut, error_msg = get_encut_from_potcar(chrg_fldr)
                    if error_msg:
                        logger.warning(
                            "Not able to determine ENCUT " "in {}".format(fldr_name)
                        )
                        logger.warning("Parsing the rest of the " "calculations")
                        continue

                comp_data = {
                    "bulk_path": bulk_file_path,
                    "defect_path": chrg_fldr,
                    "encut": encut,
                    "fldr_name": fldr_name,
                    "supercell_size": supercell_size,
                }
                if "substitution_specie" in trans_dict:
                    comp_data["substitution_specie"] = trans_dict["substitution_specie"]

                # create Defect Object as dict, then load to DefectEntry object
                defect_dict = {
                    "structure": bulk_sc_struct,
                    "charge": chrg,
                    "@module": "pymatgen.analysis.defects.core",
                }
                defect_site = trans_dict["defect_supercell_site"]
                if "vac_" in defect_type:
                    defect_dict["@class"] = "Vacancy"
                elif "as_" in defect_type or "sub_" in defect_type:
                    defect_dict["@class"] = "Substitution"
                    substitution_specie = trans_dict["substitution_specie"]
                    defect_site = PeriodicSite(
                        substitution_specie,
                        defect_site.frac_coords,
                        defect_site.lattice,
                        coords_are_cartesian=False,
                    )
                elif "int_" in defect_type:
                    defect_dict["@class"] = "Interstitial"
                else:
                    raise ValueError(
                        "defect type {} not recognized...".format(defect_type)
                    )

                defect_dict.update({"defect_site": defect_site})
                defect = MontyDecoder().process_decoded(defect_dict)
                parsed_defects.append(
                    DefectEntry(defect, energy - bulk_energy, parameters=comp_data)
                )

        try:
            parsed_defects_data = {}
            parsed_defects_data["bulk_entry"] = bulk_entry
            parsed_defects_data["defects"] = parsed_defects
            return parsed_defects_data
        except:
            return {}  # Return Null dict due to failure

    def get_vbm_bandgap(self):
        """
        Returns the valence band maximum (float) of the structure with
        MP-ID mpid.

        Args:
            mpid (str): MP-ID for which the valence band maximum is to
                be fetched from the Materials Project database
        """
        logger = logging.getLogger(__name__)
        vbm, bandgap = None, None

        if self._mpid is not None:
            with MPRester(api_key=self._mapi_key) as mp:
                bs = mp.get_bandstructure_by_material_id(self._mpid)
            if bs:
                vbm = bs.get_vbm()["energy"]
                bandgap = bs.get_band_gap()["energy"]

        if vbm is None or bandgap is None:
            if self._mpid:
                logger.warning(
                    "Mpid {} was provided, but no bandstructure entry currently exists for it. "
                    "Reverting to use of bulk calculation.".format(self._mpid)
                )
            else:
                logger.warning(
                    "No mp-id provided, will fetch CBM/VBM details from the "
                    "bulk calculation."
                )
            logger.warning(
                "This may not be appropriate if the VBM/CBM occur at reciprocal points "
                "not included in the bulk calculation."
            )
            vr, vr_path = get_vasprun(
                os.path.join(self._root_fldr, "bulk", "vasprun.xml"),
                parse_potcar_file=False,
            )
            bandgap = vr.eigenvalue_band_properties[0]
            vbm = vr.eigenvalue_band_properties[2]

        return (vbm, bandgap)

    def get_chempot_limits(self):
        """
        Returns atomic chempots from bulk_composition based on data in
        the materials project database. This is abstractly handled in the
        ChemPotAnalyzer

        Note to user: If personal phase diagram desired,
            option exists in the pycdt.core.chemical_potentials to setup,
            run and parse personal phase diagrams for purposes of chemical potentials
        """
        logger = logging.getLogger(__name__)

        if self._mpid:
            cpa = chemical_potentials.MPChemPotAnalyzer(
                mpid=self._mpid,
                sub_species=self._substitution_species,
                mapi_key=self._mapi_key,
            )
        else:
            bulkvr, bulkvr_path = get_vasprun(
                os.path.join(self._root_fldr, "bulk", "vasprun.xml"),
                parse_potcar_file=False,
            )
            if not bulkvr:
                msg = "Could not fetch computed entry for atomic chempots!"
                logger.warning(msg)
                raise ValueError(msg)
            cpa = chemical_potentials.MPChemPotAnalyzer(
                bulk_ce=bulkvr.get_computed_entry(),
                sub_species=self._substitution_species,
                mapi_key=self._mapi_key,
            )

        chem_lims = cpa.analyze_GGA_chempots()

        return chem_lims

    def parse_dielectric_calculation(self):
        """
        Parses the "vasprun.xml(.gz)" file in subdirectory "dielectric" of
        root directory root_fldr and returns the average of the trace
        of the dielectric tensor.

        Args:
            root_fldr (str):
                root directory where subdirectory "dielectric" is expected
        Returns:
            eps (float):
                average of the trace of the dielectric tensor
        """

        try:
            vr, vr_path = get_vasprun(
                os.path.join(self._root_fldr, "dielectric", "vasprun.xml"),
                parse_potcar_file=False,
            )
        except:
            logging.getLogger(__name__).warning("Parsing Dielectric calculation failed")
            return None

        eps_ion = vr.epsilon_ionic
        eps_stat = vr.epsilon_static

        eps = []
        for i in range(len(eps_ion)):
            eps.append([e[0] + e[1] for e in zip(eps_ion[i], eps_stat[i])])

        return eps

    def compile_all(self):
        """
        Run to get all post processing objects as dictionary

        note: still need to implement
            1) ability for substitutional atomic chempots
            2) incorporated charge corrections for defects
        """
        output = self.parse_defect_calculations()
        output["epsilon"] = self.parse_dielectric_calculation()
        output["mu_range"] = self.get_chempot_limits()
        vbm, gap = self.get_vbm_bandgap()
        output["vbm"] = vbm
        output["gap"] = gap

        return output


def get_site_mapping_indices(
    structure_a: Structure, structure_b: Structure, threshold=2.0
):
    """
    Reset the position of a partially relaxed structure to its unrelaxed positions.
    The template structure may have a different species ordering to the `input_structure`.
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
    Reset the position of a partially relaxed structure to its unrelaxed positions.
    The template structure may have a different species ordering to the `input_structure`.
    """

    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mapping_indices(
        initial_relax_structure, unrelaxed_structure, threshold=threshold
    )

    # Reorder the unrelaxed_structure so it matches the ordering of the initial_relax_structure (
    # from the actual calculation)
    reordered_sites = [unrelaxed_structure[tmp[2]] for tmp in mapping]
    new_structure = Structure.from_sites(reordered_sites)

    assert len(new_structure) == len(unrelaxed_structure)

    return new_structure
