"""
Parses the computed data from VASP defect calculations.
"""

import glob
import os
import warnings

import numpy as np
from monty.json import MontyDecoder
from monty.serialization import loadfn
from pymatgen.analysis.defects.core import DefectEntry, Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.defect_compatibility import DefectCompatibility
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Potcar, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Vasprun
from pymatgen.util.coord import pbc_diff

from doped import _ignore_pmg_warnings
from doped.pycdt.core import _chemical_potentials

_ignore_pmg_warnings()


def _custom_formatwarning(msg, *args, **kwargs):
    """
    Reformat warnings to just print the warning message.
    """
    return f"{msg}\n"


warnings.formatwarning = _custom_formatwarning


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
        if len(output_files) > 1:
            return output_path, True
        else:
            return output_path, False
    else:
        return path, False  # so when `get_X()` is called, it will raise an
        # informative FileNotFoundError


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
            "Could not determine defect type from composition difference of bulk " "and defect structures."
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


class SingleDefectParser:
    _delocalization_warning_printed = False  # class variable
    # ensures the verbose delocalization analysis warning is only printed once. Needs to be done
    # this way because the current workflow is to create a `SingleDefectParser` object for each
    # defect, and then warning originates from the `run_compatibility()` method of different
    # `SingleDefectParser` instances, so warnings detects each instance as a different source and
    # prints the warning multiple times. When we move to a single function call for all defects
    # (as described above), this can be removed.

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
        Also allows for use of DefectCompatibility object within pymatgen.

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

    @classmethod
    def from_paths(
        self,
        defect_path,
        bulk_path,
        dielectric,
        charge=None,
        initial_defect_structure=None,
        skip_corrections=False,
        bulk_bandgap_path=None,
        **kwargs,
    ):
        """
        Parse the defect calculation outputs in `defect_path` and return the
        parsed `DefectEntry` object.

        Args:
        defect_path (str): path to defect folder of interest (with vasprun.xml(.gz))
        bulk_path (str): path to bulk folder of interest (with vasprun.xml(.gz))
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
        charge (int): charge of defect. If not provided, will be automatically determined
            from the defect calculation outputs (requires POTCARs to be set up with `pymatgen`).
        initial_defect_structure (str):  Path to the unrelaxed defect structure,
            if structure matching with the relaxed defect structure(s) fails.
        skip_corrections (bool): Whether to skip the calculation and application of finite-size
            charge corrections to the defect energy.
        bulk_bandgap_path (str): Path to bulk OUTCAR file for determining the band gap. If the
            VBM/CBM occur at reciprocal space points not included in the bulk supercell calculation,
            you should use this tag to point to a bulk bandstructure calculation instead. If None,
            will use self.defect_entry.parameters["bulk_path"].

        Return:
            Parsed `DefectEntry` object.
        """
        from doped.analysis import defect_entry_from_paths

        kwargs.update({"return SingleDefectParser": True})
        return defect_entry_from_paths(
            defect_path,
            bulk_path,
            dielectric,
            charge=charge,
            initial_defect_structure=initial_defect_structure,
            skip_corrections=skip_corrections,
            bulk_bandgap_path=bulk_bandgap_path,
            **kwargs,
        )

    def freysoldt_loader(self, bulk_locpot=None):
        """
        Load metadata required for performing Freysoldt correction requires
        "bulk_path" and "defect_path" to be loaded to DefectEntry parameters
        dict. Can read gunzipped "LOCPOT.gz" files as well.

        Args:
            bulk_locpot (Locpot): Add bulk Locpot object for expedited parsing.
                If None, will load from file path variable bulk_path
        Return:
            bulk_locpot object for reuse by another defect entry (for expedited parsing)
        """
        if not self.defect_entry.charge:
            # dont need to load locpots if charge is zero
            return

        if not bulk_locpot:
            bulk_locpot_path, multiple = _get_output_files_and_check_if_multiple(
                "LOCPOT", self.defect_entry.parameters["bulk_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `LOCPOT` files found in bulk directory: "
                    f"{self.defect_entry.parameters['bulk_path']}. Using {bulk_locpot_path} to "
                    f"parse the electrostatic potential and compute the Freysoldt (FNV) charge "
                    f"correction."
                )
            bulk_locpot = get_locpot(bulk_locpot_path)

        defect_locpot_path, multiple = _get_output_files_and_check_if_multiple(
            "LOCPOT", self.defect_entry.parameters["defect_path"]
        )
        if multiple:
            warnings.warn(
                f"Multiple `LOCPOT` files found in defect directory: "
                f"{self.defect_entry.parameters['defect_path']}. Using {defect_locpot_path} to "
                f"parse the electrostatic potential and compute the Freysoldt (FNV) charge "
                f"correction."
            )
        defect_locpot = get_locpot(defect_locpot_path)

        axis_grid = [defect_locpot.get_axis_grid(i) for i in range(3)]
        bulk_planar_averages = [bulk_locpot.get_average_along_axis(i) for i in range(3)]
        defect_planar_averages = [defect_locpot.get_average_along_axis(i) for i in range(3)]

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
                    "initial_defect_structure": self.defect_entry.parameters["unrelaxed_defect_structure"],
                }
            )

    def kumagai_loader(self, bulk_outcar=None):
        """
        Load metadata required for performing Kumagai correction requires
        "bulk_path" and "defect_path" to be loaded to DefectEntry parameters
        dict.

        Args:
            bulk_outcar (Outcar): Add bulk Outcar object for expedited parsing.
                If None, will load from file path variable bulk_path
        Return:
            bulk_outcar object for reuse by another defect entry (for expedited parsing)
        """
        if not self.defect_entry.charge:
            # dont need to load outcars if charge is zero
            return

        if not bulk_outcar:
            bulk_outcar_path, multiple = _get_output_files_and_check_if_multiple(
                "OUTCAR", self.defect_entry.parameters["bulk_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `OUTCAR` files found in bulk directory: "
                    f"{self.defect_entry.parameters['bulk_path']}. Using {bulk_outcar_path} to "
                    f"parse core levels and compute the Kumagai (eFNV) image charge correction."
                )
            bulk_outcar = get_outcar(bulk_outcar_path)
        else:
            bulk_outcar_path = bulk_outcar.filename

        defect_outcar_path, multiple = _get_output_files_and_check_if_multiple(
            "OUTCAR", self.defect_entry.parameters["defect_path"]
        )
        if multiple:
            warnings.warn(
                f"Multiple `OUTCAR` files found in defect directory: "
                f"{self.defect_entry.parameters['defect_path']}. Using {defect_outcar_path} to "
                f"parse core levels and compute the Kumagai (eFNV) image charge correction."
            )
        defect_outcar = get_outcar(defect_outcar_path)

        bulk_atomic_site_averages = bulk_outcar.electrostatic_potential
        defect_atomic_site_averages = defect_outcar.electrostatic_potential
        if not bulk_atomic_site_averages:
            raise ValueError(
                f"Unable to parse atomic core potentials from bulk `OUTCAR` at "
                f"{bulk_outcar_path}. This can happen if `ICORELEVEL` was not set to 0 (= "
                f"default) in the `INCAR`, or if the calculation was finished prematurely with a "
                f"`STOPCAR`. The Kumagai charge correction cannot be computed without this data!"
            )

        if not defect_atomic_site_averages:
            raise ValueError(
                f"Unable to parse atomic core potentials from defect `OUTCAR` at "
                f"{defect_outcar_path}. This can happen if `ICORELEVEL` was not set to 0 (= "
                f"default) in the `INCAR`, or if the calculation was finished prematurely with a "
                f"`STOPCAR`. The Kumagai charge correction cannot be computed without this data!"
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
                species_match = bulk_structure[bulk_index].specie == defect_structure[defect_index].specie
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

    def get_stdrd_metadata(self):
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.parameters["bulk_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in bulk directory: "
                    f"{self.defect_entry.parameters['bulk_path']}. Using {bulk_vr_path} to "
                    f"parse the calculation energy and metadata."
                )
            self.bulk_vr = get_vasprun(bulk_vr_path)

        if not self.defect_vr:
            defect_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.parameters["defect_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in defect directory: "
                    f"{self.defect_entry.parameters['defect_path']}. Using {defect_vr_path} to "
                    f"parse the calculation energy and metadata."
                )
            self.defect_vr = get_vasprun(defect_vr_path)

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
                    "pot_spec": [potelt["titel"] for potelt in self.defect_vr.potcar_spec],
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
            spincls.value: eigdict.copy() for spincls, eigdict in self.defect_vr.eigenvalues.items()
        }
        kpoint_weights = self.defect_vr.actual_kpoints_weights[:]
        self.defect_entry.parameters.update({"eigenvalues": eigenvalues, "kpoint_weights": kpoint_weights})

    def get_bulk_gap_data(self, bulk_bandgap_path=None, use_MP=False, mpid=None, api_key=None):
        """
        Get bulk gap data from bulk OUTCAR file, or OUTCAR located at
        `actual_bulk_path`.

        Alternatively, one can specify query the Materials Project (MP) database for the bulk gap
        data, using `use_MP = True`, in which case the MP entry with the lowest number ID and
        composition matching the bulk will be used, or  the MP ID (mpid) of the bulk material to
        use can be specified. This is not recommended as it will correspond to a
        severely-underestimated GGA DFT bandgap!

        Args:
            bulk_bandgap_path (str): Path to bulk OUTCAR file for determining the band gap. If
                the VBM/CBM occur at reciprocal space points not included in the bulk supercell
                calculation, you should use this tag to point to a bulk bandstructure calculation
                instead. If None, will use self.defect_entry.parameters["bulk_path"].
            use_MP (bool): If True, will query the Materials Project database for the bulk gap
                data.
            mpid (str): If provided, will query the Materials Project database for the bulk gap
                data, using this Materials Project ID.
            api_key (str): Materials API key to access database.
        """
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.parameters["bulk_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in bulk directory: "
                    f"{self.defect_entry.parameters['bulk_path']}. Using {bulk_vr_path} to "
                    f"parse the calculation energy and metadata."
                )
            self.bulk_vr = get_vasprun(bulk_vr_path)

        bulk_sc_structure = self.bulk_vr.initial_structure

        vbm, cbm, bandgap = None, None, None
        gap_parameters = {}

        if use_MP and mpid is None:
            try:
                with MPRester(api_key=api_key) as mp:
                    tmp_mplist = mp.get_entries_in_chemsys(list(bulk_sc_structure.symbol_set))
                mplist = [
                    mp_ent.entry_id
                    for mp_ent in tmp_mplist
                    if mp_ent.composition.reduced_composition
                    == bulk_sc_structure.composition.reduced_composition
                ]
            except Exception as exc:
                raise ValueError(
                    f"Error with querying MPRester for"
                    f" {bulk_sc_structure.composition.reduced_formula}:"
                ) from exc

            mpid_fit_list = []
            for trial_mpid in mplist:
                with MPRester(api_key=api_key) as mp:
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
            with MPRester(api_key=api_key) as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if bs:
                cbm = bs.get_cbm()["energy"]
                vbm = bs.get_vbm()["energy"]
                bandgap = bs.get_band_gap()["energy"]
                gap_parameters.update(
                    {"MP_gga_BScalc_data": bs.get_band_gap().copy()}
                )  # contains gap kpt transition

        if vbm is None or bandgap is None or cbm is None or not bulk_bandgap_path:
            if mpid and bandgap is None:
                print(
                    f"WARNING: Mpid {mpid} was provided, but no bandstructure entry currently "
                    "exists for it. \nReverting to use of bulk supercell calculation for band "
                    "edge extrema."
                )

            gap_parameters.update({"MP_gga_BScalc_data": None})  # to signal no MP BS is used
            bandgap, cbm, vbm, _ = self.bulk_vr.eigenvalue_band_properties

        if bulk_bandgap_path:
            print(f"Using actual bulk path: {bulk_bandgap_path}")
            actual_bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", bulk_bandgap_path
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in specified directory: "
                    f"{bulk_bandgap_path}. Using {actual_bulk_vr_path} to  parse the calculation "
                    f"energy and metadata."
                )
            actual_bulk_vr = get_vasprun(actual_bulk_vr_path)
            bandgap, cbm, vbm, _ = actual_bulk_vr.eigenvalue_band_properties

        gap_parameters.update({"mpid": mpid, "cbm": cbm, "vbm": vbm, "gap": bandgap})
        self.defect_entry.parameters.update(gap_parameters)

    def run_compatibility(self):
        # Set potalign so pymatgen can calculate bandfilling for 'neutral' defects
        # (possible for resonant dopants etc.)
        if self.defect_entry.charge == 0 and "potalign" not in self.defect_entry.parameters:
            self.defect_entry.parameters["potalign"] = 0

        self.defect_entry = self.compatibility.process_entry(self.defect_entry)

        if "delocalization_meta" in self.defect_entry.parameters:
            delocalization_meta = self.defect_entry.parameters["delocalization_meta"]
            if (
                "plnr_avg" in delocalization_meta and not delocalization_meta["plnr_avg"]["is_compatible"]
            ) or (
                "atomic_site" in delocalization_meta
                and not delocalization_meta["atomic_site"]["is_compatible"]
            ):
                specific_delocalized_warning = (
                    f"Delocalization analysis has indicated that {self.defect_entry.name} with "
                    f"charge {self.defect_entry.charge:+} may not be compatible with the chosen "
                    f"charge correction."
                )
                general_delocalization_warning = """
Note: Defects throwing a "delocalization analysis" warning may require a larger supercell for
accurate total energies. Recommended to look at the correction plots (i.e. run
`get_correction_freysoldt(DefectEntry,...,plot=True)` from
`doped.corrections`) to visually determine if the charge
correction scheme is still appropriate (replace 'freysoldt' with 'kumagai' if using anisotropic
correction). You can also change the DefectCompatibility() tolerance settings via the
`compatibility` parameter in `SingleDefectParser.from_paths()`."""
                warnings.warn(message=specific_delocalized_warning)
                if not self._delocalization_warning_printed:
                    warnings.warn(message=general_delocalization_warning)  # should only print once
                    SingleDefectParser._delocalization_warning_printed = True  # don't print again

        if "num_hole_vbm" in self.defect_entry.parameters and (
            (self.compatibility.free_chg_cutoff < self.defect_entry.parameters["num_hole_vbm"])
            or (self.compatibility.free_chg_cutoff < self.defect_entry.parameters["num_elec_cbm"])
        ):
            num_holes = self.defect_entry.parameters["num_hole_vbm"]
            num_electrons = self.defect_entry.parameters["num_elec_cbm"]
            warnings.warn(
                f"Eigenvalue analysis has determined that `num_hole_vbm` (= {num_holes}) or "
                f"`num_elec_cbm` (= {num_electrons}) is significant (>2.1) for "
                f"{self.defect_entry.name} with charge {self.defect_entry.charge}:+, "
                f"indicating that there are many free charges in this defect supercell "
                f"calculation and so the defect charge correction is unlikely to be accurate."
            )
            if "freysoldt_meta" in self.defect_entry.parameters:
                frey_meta = self.defect_entry.parameters["freysoldt_meta"]
                frey_corr = (
                    frey_meta["freysoldt_electrostatic"]
                    + frey_meta["freysoldt_potential_alignment_correction"]
                )
                self.defect_entry.corrections.update({"charge_correction": frey_corr})
            elif "kumagai_meta" in self.defect_entry.parameters:
                kumagai_meta = self.defect_entry.parameters["kumagai_meta"]
                kumagai_corr = (
                    kumagai_meta["kumagai_electrostatic"]
                    + kumagai_meta["kumagai_potential_alignment_correction"]
                )
                self.defect_entry.corrections.update({"charge_correction": kumagai_corr})

        if (
            self.defect_entry.charge != 0
            and self.defect_entry.corrections.get("charge_correction", None) is None
        ):
            warnings.warn(
                f"No charge correction computed for {self.defect_entry.name} with "
                f"charge {self.defect_entry.charge:+}, indicating problems with the "
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
        Parses the defect calculations as DefectEntry objects, from a PyCDT
        root_fldr file structure.

        Charge correction is missing in the first run.
        """
        parsed_defects = []
        subfolders = glob.glob(os.path.join(self._root_fldr, "vac_*"))
        subfolders += glob.glob(os.path.join(self._root_fldr, "as_*"))
        subfolders += glob.glob(os.path.join(self._root_fldr, "sub_*"))
        subfolders += glob.glob(os.path.join(self._root_fldr, "inter_*"))

        def get_vr_and_check_locpot(fldr):
            vr_file = os.path.join(fldr, "vasprun.xml")
            if not (os.path.exists(vr_file) or os.path.exists(vr_file + ".gz")):
                warnings.warn(f"{vr_file} doesn't exit")
                error_msg = ": Failure, vasprun.xml doesn't exist."
                return (None, error_msg)  # Further processing is not useful

            try:
                vr = get_vasprun(vr_file, parse_potcar_file=False)
            except:
                warnings.warn(f"Couldn't parse {vr_file}")
                error_msg = ": Failure, couldn't parse vasprun.xml file."
                return (None, error_msg)

            if not vr.converged:
                warnings.warn(f"Vasp calculation at {fldr} not converged")
                error_msg = ": Failure, Vasp calculation not converged."
                return (None, error_msg)  # Further processing is not useful

            # Check if locpot exists
            locpot_file = os.path.join(fldr, "LOCPOT")
            if not (os.path.exists(locpot_file) or os.path.exists(locpot_file + ".gz")):
                warnings.warn(f"{locpot_file} doesn't exit")
                error_msg = ": Failure, LOCPOT doesn't exist"
                return (None, error_msg)  # Further processing is not useful

            return (vr, None)

        def get_encut_from_potcar(fldr):
            potcar_file = os.path.join(fldr, "POTCAR")
            if not os.path.exists(potcar_file):
                warnings.warn(f"Not POTCAR in {fldr} to parse ENCUT")
                error_msg = ": Failure, No POTCAR file."
                return (None, error_msg)  # Further processing is not useful

            try:
                potcar = Potcar.from_file(potcar_file)
            except:
                warnings.warn(f"Couldn't parse {potcar_file}")
                error_msg = ": Failure, couldn't read POTCAR file."
                return (None, error_msg)

            encut = max(ptcr_sngl.enmax for ptcr_sngl in potcar)
            return (encut, None)

        # get bulk entry information first
        fldr = os.path.join(self._root_fldr, "bulk")
        vr, error_msg = get_vr_and_check_locpot(fldr)
        if error_msg:
            raise ValueError(f"Abandoning parsing of the calculations: {error_msg}")
        bulk_energy = vr.final_energy
        bulk_sc_struct = vr.final_structure
        try:
            encut = vr.incar["ENCUT"]
        except:  # ENCUT not specified in INCAR. Read from POTCAR
            encut, error_msg = get_encut_from_potcar(fldr)
            if error_msg:
                raise ValueError(f"Abandoning parsing of the calculations: {error_msg}")

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
                trans_dict = loadfn(os.path.join(chrg_fldr, "transformation.json"), cls=MontyDecoder)
                chrg = trans_dict["charge"]
                vr, error_msg = get_vr_and_check_locpot(chrg_fldr)
                if error_msg:
                    warnings.warn("Parsing the rest of the calculations")
                    continue
                if (
                    "substitution_specie" in trans_dict
                    and trans_dict["substitution_specie"] not in bulk_sc_struct.symbol_set
                ):
                    self._substitution_species.add(trans_dict["substitution_specie"])
                elif (
                    "inter" in trans_dict["defect_type"]
                    and trans_dict["defect_site"].specie.symbol not in bulk_sc_struct.symbol_set
                ):
                    # added because extrinsic interstitials don't have
                    # "substitution_specie" character...
                    trans_dict["substitution_specie"] = trans_dict["defect_site"].specie.symbol
                    self._substitution_species.add(trans_dict["defect_site"].specie.symbol)

                defect_type = trans_dict.get("defect_type", None)
                energy = vr.final_energy
                try:
                    encut = vr.incar["ENCUT"]
                except:  # ENCUT not specified in INCAR. Read from POTCAR
                    encut, error_msg = get_encut_from_potcar(chrg_fldr)
                    if error_msg:
                        warnings.warn("Not able to determine ENCUT " "in {}".format(fldr_name))
                        warnings.warn("Parsing the rest of the " "calculations")
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
                    raise ValueError(f"defect type {defect_type} not recognized...")

                defect_dict.update({"defect_site": defect_site})
                defect = MontyDecoder().process_decoded(defect_dict)
                parsed_defects.append(DefectEntry(defect, energy - bulk_energy, parameters=comp_data))

        try:
            parsed_defects_data = {}
            parsed_defects_data["bulk_entry"] = bulk_entry
            parsed_defects_data["defects"] = parsed_defects
            return parsed_defects_data
        except:
            return {}  # Return Null dict due to failure

    def get_vbm_bandgap(self):
        """
        Returns the valence band maximum (float) of the structure with MP-ID
        mpid.

        Args:
            mpid (str): MP-ID for which the valence band maximum is to
                be fetched from the Materials Project database
        """
        vbm, bandgap = None, None

        if self._mpid is not None:
            with MPRester(api_key=self._mapi_key) as mp:
                bs = mp.get_bandstructure_by_material_id(self._mpid)
            if bs:
                vbm = bs.get_vbm()["energy"]
                bandgap = bs.get_band_gap()["energy"]

        if vbm is None or bandgap is None:
            if self._mpid:
                warnings.warn(
                    "Mpid {} was provided, but no bandstructure entry currently exists for it. "
                    "Reverting to use of bulk calculation.".format(self._mpid)
                )
            else:
                warnings.warn(
                    "No mp-id provided, will fetch CBM/VBM details from the " "bulk calculation."
                )
            warnings.warn(
                "This may not be appropriate if the VBM/CBM occur at reciprocal points "
                "not included in the bulk calculation."
            )
            vr = get_vasprun(
                os.path.join(self._root_fldr, "bulk", "vasprun.xml"),
                parse_potcar_file=False,
            )
            bandgap = vr.eigenvalue_band_properties[0]
            vbm = vr.eigenvalue_band_properties[2]

        return (vbm, bandgap)

    def get_chempot_limits(self):
        """
        Returns atomic chempots from bulk_composition based on data in the
        materials project database. This is abstractly handled in the
        ChemPotAnalyzer.

        Note to user: If personal phase diagram desired,
            option exists in the pycdt.core.chemical_potentials to setup,
            run and parse personal phase diagrams for purposes of chemical potentials
        """
        if self._mpid:
            cpa = _chemical_potentials.MPChemPotAnalyzer(
                mpid=self._mpid,
                sub_species=self._substitution_species,
                mapi_key=self._mapi_key,
            )
        else:
            bulkvr = get_vasprun(
                os.path.join(self._root_fldr, "bulk", "vasprun.xml"),
                parse_potcar_file=False,
            )
            if not bulkvr:
                msg = "Could not fetch computed entry for atomic chempots!"
                warnings.warn(msg)
                raise ValueError(msg)
            cpa = _chemical_potentials.MPChemPotAnalyzer(
                bulk_ce=bulkvr.get_computed_entry(),
                sub_species=self._substitution_species,
                mapi_key=self._mapi_key,
            )

        chem_lims = cpa.analyze_GGA_chempots()

        return chem_lims

    def parse_dielectric_calculation(self):
        """
        Parses the "vasprun.xml(.gz)" file in subdirectory "dielectric" of root
        directory root_fldr and returns the average of the trace of the
        dielectric tensor.

        Args:
            root_fldr (str):
                root directory where subdirectory "dielectric" is expected
        Returns:
            eps (float):
                average of the trace of the dielectric tensor
        """
        try:
            vr = get_vasprun(
                os.path.join(self._root_fldr, "dielectric", "vasprun.xml"),
                parse_potcar_file=False,
            )
        except:
            warnings.warn("Parsing Dielectric calculation failed")
            return None

        eps_ion = vr.epsilon_ionic
        eps_stat = vr.epsilon_static

        eps = []
        for i in range(len(eps_ion)):
            eps.append([e[0] + e[1] for e in zip(eps_ion[i], eps_stat[i])])

        return eps

    def compile_all(self):
        """
        Run to get all post processing objects as dictionary.

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
