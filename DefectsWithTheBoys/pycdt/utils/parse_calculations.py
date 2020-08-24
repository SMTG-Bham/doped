# coding: utf-8
"""
Parses the computed data from VASP defect calculations.
"""
# from __future__ import unicode_literals
from __future__ import division

__author__ = "Bharat Medasani, Nils Zimmermann, Danny Broberg"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Seán Kavanagh"
__email__ = "sean.kavanagh.19@ucl.ac.uk"
__date__ = "May 19, 2020"

import glob
import logging
import os
import warnings

import numpy as np
from monty.json import MontyDecoder
from monty.serialization import loadfn
from pymatgen.analysis.defects.core import Vacancy, Substitution, Interstitial, DefectEntry
from pymatgen.analysis.defects.defect_compatibility import DefectCompatibility
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Potcar, BadPotcarWarning
from pymatgen.io.vasp.outputs import Vasprun, Locpot, Outcar, Poscar

from DefectsWithTheBoys.pycdt.core.chemical_potentials import MPChemPotAnalyzer


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
    if type(cd) != dict:
        cd = cd.as_dict()
    if type(b_cse) != dict:
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
        bulk_sc_structure.get_sites_in_sphere(defect_site.coords, 0.1, include_index=True),
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
        raise ValueError("Could not recognize defect type for {}".format(cd["name"]))

    # assign proper energy and parameter metadata
    uncorrected_energy = cd["entry"]["energy"] - b_cse["energy"]
    def_path = os.path.split(cd["entry"]["data"]["locpot_path"])[0]
    bulk_path = os.path.split(b_cse["data"]["locpot_path"])[0]
    p = {"defect_path": def_path, "bulk_path": bulk_path, "encut": cd["entry"]["data"]["encut"]}

    de = DefectEntry(defect_obj, uncorrected_energy, parameters=p)

    return de


def get_vasprun(vasprun_path, **kwargs):
    """ Read the vasprun.xml(.gz) file as a pymatgen Locpot object """
    warnings.filterwarnings(
        "ignore", category=BadPotcarWarning
    )  # Ignore POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
    if os.path.exists(vasprun_path):
        vasprun = Vasprun(vasprun_path)
    elif os.path.exists(vasprun_path + ".gz", **kwargs):
        vasprun = Vasprun(vasprun_path + ".gz", **kwargs)
    else:
        raise FileNotFoundError(
            f"""Well I can't fucking find a vasprun.xml(.gz) at {vasprun_path}(.gz).
                   You sure there's one there pal? I need it to parse the calculation results"""
        )
    return vasprun


def get_locpot(locpot_path):
    """ Read the LOCPOT(.gz) file as a pymatgen Locpot object """
    if os.path.exists(locpot_path):
        locpot = Locpot.from_file(locpot_path)
    elif os.path.exists(locpot_path + ".gz"):
        locpot = Locpot.from_file(locpot_path + ".gz")
    else:
        raise FileNotFoundError(
            f"""Well I can't fucking find a LOCPOT(.gz) at {locpot_path}(.gz).
                   You sure there's one there pal? I need it to get the Freysoldt correction"""
        )
    return locpot


class SingleDefectParser:
    def __init__(
        self,
        defect_entry,
        compatibility=DefectCompatibility(plnr_avg_var_tol=0.0002),
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
            performing compatibility analysis on defect entry.
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
        compatibility=DefectCompatibility(plnr_avg_var_tol=0.0002),
        initial_defect_structure=None,
    ):
        """
        Identify defect object based on file paths. Minimal parsing performing for
        instantiating the SingleDefectParser class.

        :param path_to_defect (str): path to defect file of interest (with vasprun.xml(.gz))
        :param path_to_bulk (str): path to bulk file of interest (with vasprun.xml(.gz))
        :param dielectric (float or 3x3 matrix): ionic + static contributions to dielectric constant
        :param defect_charge (int):
        :param mpid (str):
        :param compatibility (DefectCompatibility): Compatibility class instance for
            performing compatibility analysis on defect entry.

        Return:
            Instance of the SingleDefectParser class.
        """
        parameters = {
            "bulk_path": path_to_bulk,
            "defect_path": path_to_defect,
            "dielectric": dielectric,
            "mpid": mpid,
        }

        # add bulk simple properties
        bulk_vr = get_vasprun(os.path.join(path_to_bulk, "vasprun.xml"))
        bulk_energy = bulk_vr.final_energy
        bulk_sc_structure = bulk_vr.initial_structure.copy()

        # add defect simple properties
        defect_vr = get_vasprun(os.path.join(path_to_defect, "vasprun.xml"))
        defect_energy = defect_vr.final_energy
        # Can specify initial defect structure (to help PyCDT find the defect site if
        # multiple relaxations were required, else use from defect relaxation OUTCAR:
        if initial_defect_structure:
            initial_defect_structure = Poscar.from_file(initial_defect_structure).structure.copy()
        else:
            initial_defect_structure = defect_vr.initial_structure.copy()

        # identify defect site, structural information, and create defect object
        num_ids = len(initial_defect_structure)
        num_bulk = len(bulk_sc_structure)
        if num_ids == num_bulk - 1:
            defect_type = "Vacancy"
        elif num_ids == num_bulk + 1:
            defect_type = "Interstitial"
        elif num_ids == num_bulk:
            defect_type = "Substitution"
        else:
            raise ValueError(
                "Could not identify defect type just from number of sites in structure: "
                "{} in bulk vs. {} in defect?".format(num_ids, num_bulk)
            )

        defect_index_sc_coords = None
        transformation_path = os.path.join(path_to_defect, "transformation.json")
        if os.path.exists(transformation_path):
            tf = loadfn(transformation_path)
            site = tf["defect_supercell_site"]
            if defect_type == "Vacancy":
                poss_deflist = sorted(
                    bulk_sc_structure.get_sites_in_sphere(site.coords, 0.15, include_index=True),
                    key=lambda x: x[1],
                )
            else:
                poss_deflist = sorted(
                    initial_defect_structure.get_sites_in_sphere(
                        site.coords, 0.15, include_index=True
                    ),
                    key=lambda x: x[1],
                )
            if not poss_deflist:
                raise ValueError(
                    "{} specified defect site {}, but could not find it in bulk_supercell."
                    " Abandoning parsing".format(transformation_path, site)
                )
            defect_index_sc_coords = poss_deflist[0][2]
        else:
            print(
                "No transformation file exists at {}.\nCalculating defect index manually"
                " (proceed with caution)".format(transformation_path)
            )

        # IF not transformation file exists, the defect_index_sc_coords will not be identified in
        # previous routine, proceed by identifying the defect site through a comparison of bulk
        # sites and initial defect structure sites.
        # WARNING: this can cause issues if initial_defect_structure is slightly different than
        # bulk_sc_structure (as a result of multiple relaxation steps, for example)
        # noinspection DuplicatedCode
        if defect_index_sc_coords is None:
            bulksites = [site.frac_coords for site in bulk_sc_structure]
            initsites = [site.frac_coords for site in initial_defect_structure]
            distmatrix = initial_defect_structure.lattice.get_all_distances(bulksites, initsites)
            min_dist_with_index = [
                [min(distmatrix[bulk_index]), int(bulk_index), int(distmatrix[bulk_index].argmin())]
                for bulk_index in range(len(distmatrix))
            ]  # list of [min dist, bulk ind, defect ind]

            site_matching_indices = []
            poss_defect = []
            if defect_type in ["Vacancy", "Interstitial"]:
                for mindist, bulk_index, defect_index in min_dist_with_index:
                    if mindist < 0.1:
                        site_matching_indices.append([bulk_index, defect_index])
                    elif defect_type == "Vacancy":
                        poss_defect.append([bulk_index, bulksites[bulk_index][:]])

                if defect_type == "Interstitial":
                    poss_defect = [
                        [ind, fc[:]]
                        for ind, fc in enumerate(initsites)
                        if ind not in np.array(site_matching_indices)[:, 1]
                    ]

            elif defect_type == "Substitution":
                # noinspection DuplicatedCode
                for mindist, bulk_index, defect_index in min_dist_with_index:
                    species_match = (
                        bulk_sc_structure[bulk_index].specie
                        == initial_defect_structure[defect_index].specie
                    )
                    if mindist < 0.1 and species_match:
                        site_matching_indices.append([bulk_index, defect_index])

                    elif not species_match:
                        poss_defect.append([defect_index, initsites[defect_index][:]])

            if len(poss_defect) == 1:
                defect_index_sc_coords = poss_defect[0][0]
            else:
                raise ValueError(
                    "Found {} possible defect sites when matching bulk and "
                    "defect structure".format(len(poss_defect))
                )

            if len(set(np.array(site_matching_indices)[:, 0])) != len(
                set(np.array(site_matching_indices)[:, 1])
            ):
                raise ValueError(
                    "Error occurred in site_matching routine. Double counting of site matching "
                    "occurred:{}\nAbandoning structure parsing.".format(site_matching_indices)
                )

        if defect_type == "Vacancy":
            defect_site = bulk_sc_structure[defect_index_sc_coords]
        else:
            defect_site = initial_defect_structure[defect_index_sc_coords]

        for_monty_defect = {
            "@module": "pymatgen.analysis.defects.core",
            "@class": defect_type,
            "charge": defect_charge,
            "structure": bulk_sc_structure,
            "defect_site": defect_site,
        }
        defect = MontyDecoder().process_decoded(for_monty_defect)
        test_defect_structure = defect.generate_defect_structure()
        if not StructureMatcher(
            stol=0.5, primitive_cell=False, scale=False, attempt_supercell=False, allow_subset=False
        ).fit(test_defect_structure, defect_vr.initial_structure):
            # NOTE: this does not insure that cartesian coordinates or indexing are identical
            # Note: I've changed stol to 0.5 to fix matching for defects that move the fuck about yo
            if not StructureMatcher(
                stol=0.5,
                primitive_cell=False,
                scale=False,
                attempt_supercell=False,
                allow_subset=False,
            ).fit(test_defect_structure, initial_defect_structure):
                # defect_vr.initial_structure may not match test_defect_structure if we've had
                # significant relaxation prior to final defect energy calculation, so test against
                # initial_defect_structure (which may have been specified in function arguments)
                raise ValueError("Error in defect object matching!")

        defect_entry = DefectEntry(
            defect, defect_energy - bulk_energy, corrections={}, parameters=parameters
        )

        return SingleDefectParser(
            defect_entry, compatibility=compatibility, defect_vr=defect_vr, bulk_vr=bulk_vr
        )

    def freysoldt_loader(self, bulk_locpot=None):
        """Load metadata required for performing Freysoldt correction
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
            bulk_locpot_path = os.path.join(self.defect_entry.parameters["bulk_path"], "LOCPOT")
            bulk_locpot = get_locpot(bulk_locpot_path)

        def_locpot_path = os.path.join(self.defect_entry.parameters["defect_path"], "LOCPOT")
        def_locpot = get_locpot(def_locpot_path)

        axis_grid = [def_locpot.get_axis_grid(i) for i in range(3)]
        bulk_planar_averages = [bulk_locpot.get_average_along_axis(i) for i in range(3)]
        defect_planar_averages = [def_locpot.get_average_along_axis(i) for i in range(3)]

        defect_frac_sc_coords = self.defect_entry.site.frac_coords

        self.defect_entry.parameters.update(
            {
                "axis_grid": axis_grid,
                "bulk_planar_averages": bulk_planar_averages,
                "defect_planar_averages": defect_planar_averages,
                "initial_defect_structure": def_locpot.structure,
                "defect_frac_sc_coords": defect_frac_sc_coords,
            }
        )

        return bulk_locpot

    # noinspection DuplicatedCode
    def kumagai_loader(self, bulk_outcar=None):
        """Load metadata required for performing Kumagai correction
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
            bulk_outcar_path = os.path.join(self.defect_entry.parameters["bulk_path"], "OUTCAR")
            bulk_outcar = Outcar(bulk_outcar_path)

        def_outcar_path = os.path.join(self.defect_entry.parameters["defect_path"], "OUTCAR")
        def_outcar = Outcar(def_outcar_path)

        bulk_atomic_site_averages = bulk_outcar.electrostatic_potential
        defect_atomic_site_averages = def_outcar.electrostatic_potential

        if os.path.exists(os.path.join(self.defect_entry.parameters["bulk_path"], "POSCAR")):
            bulk_sc_structure = Poscar.from_file(
                os.path.join(self.defect_entry.parameters["bulk_path"], "POSCAR")
            ).structure
        elif self.bulk_vr:
            bulk_sc_structure = self.bulk_vr.initial_structure.copy()
        else:
            bulk_sc_structure = get_vasprun(
                os.path.join(self.defect_entry.parameters["bulk_path"], "vasprun.xml")
            ).initial_structure.copy()

        if self.defect_entry.site:
            defect_frac_sc_coords = self.defect_entry.site.frac_coords
            poss_deflist = sorted(
                bulk_sc_structure.get_sites_in_sphere(
                    self.defect_entry.site.coords, 0.15, include_index=True
                ),
                key=lambda x: x[1],
            )
            defect_index_sc_coords = poss_deflist[0][2]
        # May add this bit if it's still suggesting several possible defect sites

        if "initial_defect_structure" in self.defect_entry.parameters:
            initial_defect_structure = self.defect_entry.parameters["initial_defect_structure"]
        elif os.path.exists(os.path.join(self.defect_entry.parameters["defect_path"], "POSCAR")):
            initial_defect_structure = Poscar.from_file(
                os.path.join(self.defect_entry.parameters["defect_path"], "POSCAR")
            ).structure
        elif self.defect_vr:
            initial_defect_structure = self.defect_vr.initial_structure
        else:
            initial_defect_structure = get_vasprun(
                os.path.join(self.defect_entry.parameters["defect_path"], "vasprun.xml")
            ).initial_structure

        bulksites = [site.frac_coords for site in bulk_sc_structure]
        initsites = [site.frac_coords for site in initial_defect_structure]
        distmatrix = initial_defect_structure.lattice.get_all_distances(
            bulksites, initsites
        )  # first index of this list is bulk index
        min_dist_with_index = [
            [min(distmatrix[bulk_index]), int(bulk_index), int(distmatrix[bulk_index].argmin())]
            for bulk_index in range(len(distmatrix))
        ]  # list of [min dist, bulk ind, defect ind]

        site_matching_indices = []
        poss_defect = []
        if isinstance(self.defect_entry.defect, (Vacancy, Interstitial)):
            for mindist, bulk_index, defect_index in min_dist_with_index:
                if mindist < 0.1:
                    site_matching_indices.append([bulk_index, defect_index])
                elif (
                    isinstance(self.defect_entry.defect, Vacancy) and defect_index_sc_coords is None
                ):
                    poss_defect.append([bulk_index, bulksites[bulk_index][:]])

            if (
                isinstance(self.defect_entry.defect, Interstitial)
                and defect_index_sc_coords is None
            ):
                poss_defect = [
                    [ind, fc[:]]
                    for ind, fc in enumerate(initsites)
                    if ind not in np.array(site_matching_indices)[:, 1]
                ]

        elif isinstance(self.defect_entry.defect, Substitution):
            # noinspection DuplicatedCode
            for mindist, bulk_index, defect_index in min_dist_with_index:
                species_match = (
                    bulk_sc_structure[bulk_index].specie
                    == initial_defect_structure[defect_index].specie
                )
                if mindist < 0.1 and species_match:
                    site_matching_indices.append([bulk_index, defect_index])

                elif not species_match and defect_index_sc_coords is None:
                    poss_defect.append([defect_index, initsites[defect_index][:]])

        if defect_index_sc_coords is None:
            if len(poss_defect) == 1:
                defect_index_sc_coords = poss_defect[0][0]
                defect_frac_sc_coords = poss_defect[0][1]
            else:
                raise ValueError(
                    "Found {} possible defect sites when matching bulk and "
                    "defect structure".format(len(poss_defect))
                )

        if len(set(np.array(site_matching_indices)[:, 0])) != len(
            set(np.array(site_matching_indices)[:, 1])
        ):
            raise ValueError(
                "Error occurred in site_matching routine. Double counting of site matching "
                "occurred:{}\nAdvising against Kumagai parsing.".format(site_matching_indices)
            )

        # user Wigner-Seitz radius for sampling radius
        wz = initial_defect_structure.lattice.get_wigner_seitz_cell()
        dist = []
        for facet in wz:
            midpt = np.mean(np.array(facet), axis=0)
            dist.append(np.linalg.norm(midpt))
        sampling_radius = min(dist)

        self.defect_entry.parameters.update(
            {
                "bulk_atomic_site_averages": bulk_atomic_site_averages,
                "defect_atomic_site_averages": defect_atomic_site_averages,
                "initial_defect_structure": initial_defect_structure,
                "site_matching_indices": site_matching_indices,
                "sampling_radius": sampling_radius,
                "defect_frac_sc_coords": defect_frac_sc_coords,
                "defect_index_sc_coords": defect_index_sc_coords,
            }
        )

        return bulk_outcar

    def get_stdrd_metadata(self):

        if not self.bulk_vr:
            path_to_bulk = self.defect_entry.parameters["bulk_path"]
            self.bulk_vr = get_vasprun(os.path.join(path_to_bulk, "vasprun.xml"))

        if not self.defect_vr:
            path_to_defect = self.defect_entry.parameters["defect_path"]
            self.defect_vr = get_vasprun(os.path.join(path_to_defect, "vasprun.xml"))

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
                    for k in ["LHFCALC", "HFSCREEN", "IVDW", "LUSE_VDW", "LDAU", "METAGGA"]
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
                "initial_defect_structure": self.defect_vr.initial_structure,
                "defect_energy": self.defect_vr.final_energy,
            }
        )

        # grab defect energy and eigenvalue information for band filling and localization analysis
        eigenvalues = {
            spincls.value: eigdict.copy() for spincls, eigdict in self.defect_vr.eigenvalues.items()
        }
        kpoint_weights = self.defect_vr.actual_kpoints_weights[:]
        self.defect_entry.parameters.update(
            {"eigenvalues": eigenvalues, "kpoint_weights": kpoint_weights}
        )

    def get_bulk_gap_data(self, no_MP=False, actual_bulk_path=None):

        if not self.bulk_vr:
            path_to_bulk = self.defect_entry.parameters["bulk_path"]
            self.bulk_vr = get_vasprun(os.path.join(path_to_bulk, "vasprun.xml"))

        bulk_sc_structure = self.bulk_vr.initial_structure
        mpid = self.defect_entry.parameters["mpid"]

        if not mpid and not no_MP:
            try:
                with MPRester() as mp:
                    tmp_mplist = mp.get_entries_in_chemsys(list(bulk_sc_structure.symbol_set))
                mplist = [
                    ment.entry_id
                    for ment in tmp_mplist
                    if ment.composition.reduced_composition
                    == bulk_sc_structure.composition.reduced_composition
                ]
            except:
                raise ValueError(
                    "Error with querying MPRester for {}"
                    "".format(bulk_sc_structure.composition.reduced_formula)
                )

            mpid_fit_list = []
            for trial_mpid in mplist:
                with MPRester() as mp:
                    mpstruct = mp.get_structure_by_material_id(trial_mpid)
                if StructureMatcher(
                    primitive_cell=True, scale=False, attempt_supercell=True, allow_subset=False
                ).fit(bulk_sc_structure, mpstruct):
                    mpid_fit_list.append(trial_mpid)

            if len(mpid_fit_list) == 1:
                mpid = mpid_fit_list[0]
                print("Single mp-id found for bulk structure:{}.".format(mpid))
            elif len(mpid_fit_list) > 1:
                num_mpid_list = [int(mp.split("" - "")[1]) for mp in mpid_fit_list]
                num_mpid_list.sort()
                mpid = "mp-" + str(num_mpid_list[0])
                print(
                    "Multiple mp-ids found for bulk structure:{}\nWill use lowest number mpid "
                    "for bulk band structure = {}.".format(str(mpid_fit_list), mpid)
                )
            else:
                print(
                    "Could not find bulk structure in MP database after tying the "
                    "following list:\n{}".format(mplist)
                )
                mpid = None
        else:
            print("Manually fed mpid = {}".format(mpid))

        vbm, cbm, bandgap = None, None, None
        gap_parameters = {}
        if mpid is not None and not no_MP:
            # TODO: NEED to be smarter about use of +U or HSE etc in MP gga band structure
            #  calculations...
            with MPRester() as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if bs:
                cbm = bs.get_cbm()["energy"]
                vbm = bs.get_vbm()["energy"]
                bandgap = bs.get_band_gap()["energy"]
                gap_parameters.update(
                    {"MP_gga_BScalc_data": bs.get_band_gap().copy()}
                )  # contains gap kpt transition

        if vbm is None or bandgap is None or cbm is None or no_MP or not actual_bulk_path:
            if mpid and bandgap is None:
                print(
                    "WARNING: Mpid {} was provided, but no bandstructure entry currently exists "
                    "for it. \n"
                    "Reverting to use of bulk supercell calculation for band edge extrema.".format(
                        mpid
                    )
                )
            if mpid and no_MP:
                print(
                    "Mpid {} was provided, but we've decided to say:\n"
                    "'Thanks, but no thanks' to the Materials Project.\n"
                    "Reverting to use of bulk supercell calculation for band edge extrema.".format(
                        mpid
                    )
                )
            else:
                print(
                    "WARNING: No mp-id provided, will fetch CBM/VBM details from the "
                    "bulk calculation."
                )
                print(
                    "Note that it would be better to " "perform real band structure calculation..."
                )

            gap_parameters.update({"MP_gga_BScalc_data": None})  # to signal no MP BS is used
            bandgap, cbm, vbm, _ = self.bulk_vr.eigenvalue_band_properties

        # Note that we've modified
        # pycdt.utils.parse_calculations.SingleDefectParser.get_bulk_gap_data() to have
        # the option to say 'no_MP', where it will use the bulk calculation bandgap, regardless
        # of whether a shit GGA bandgap
        # for our material exists in the MP database.
        # We've also added actual_bulk optional argument to
        # pycdt.utils.parse_calculations.SingleDefectParser.get_bulk_gap_data(), in case we're
        # looking at defect complexes (so the SingleDefectParser 'bulk' is actually the original
        # point defect, with the wrong pmg-computed gap)

        if actual_bulk_path:
            print(f"Using actual bulk path: {actual_bulk_path}")
            actual_bulk_vr = get_vasprun(os.path.join(actual_bulk_path, "vasprun.xml"))
            bandgap, cbm, vbm, _ = actual_bulk_vr.eigenvalue_band_properties

        gap_parameters.update({"mpid": mpid, "cbm": cbm, "vbm": vbm, "gap": bandgap})
        self.defect_entry.parameters.update(gap_parameters)

    def run_compatibility(self):
        self.defect_entry = self.compatibility.process_entry(self.defect_entry)
        if not self.defect_entry.parameters["is_compatible"]:
            warnings.warn(
                f"""
Delocalization analysis has indicated that {self.defect_entry.name}
with charge {self.defect_entry.charge} may not be compatible with the chosen charge correction
scheme, and may require a larger supercell for accurate calculation of the energy. Recommended to
look at the correction plots (i.e. run `get_correction_freysoldt(DefectEntry,...,plot=True)` from
`DefectsWithTheBoys.pycdt.corrections.finite_size_charge_correction` to visually determine if
charge correction scheme still appropriate, then `sdp.compatibility.perform_freysoldt(DefectEntry)`
if you're happy (replace 'freysoldt' with 'kumagai' if using anisotropic correction.
You can also change the DefectCompatibility() tolerance settings via the `compatibility` parameter
in `SingleDefectParser.from_paths()`.
Watch out that if `num_hole_vbm` or `num_elec_cbm` are greater than the free_chg_cutoff (default
2.1), charge correction will not be applied.
""",
                stacklevel=2,
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
                vr = get_vasprun(vr_file, parse_potcar_file=False)
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
            data={"bulk_path": bulk_file_path, "encut": encut, "supercell_size": supercell_size},
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
                        logger.warning("Not able to determine ENCUT " "in {}".format(fldr_name))
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
                    raise ValueError("defect type {} not recognized...".format(defect_type))

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
                    "No mp-id provided, will fetch CBM/VBM details from the " "bulk calculation."
                )
            logger.warning(
                "Note that it would be better to " "perform real band structure calculation..."
            )
            vr = get_vasprun(
                os.path.join(self._root_fldr, "bulk", "vasprun.xml"), parse_potcar_file=False
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
            cpa = MPChemPotAnalyzer(
                mpid=self._mpid, sub_species=self._substitution_species, mapi_key=self._mapi_key
            )
        else:
            bulkvr = get_vasprun(
                os.path.join(self._root_fldr, "bulk", "vasprun.xml"), parse_potcar_file=False
            )
            if not bulkvr:
                msg = "Could not fetch computed entry for atomic chempots!"
                logger.warning(msg)
                raise ValueError(msg)
            cpa = MPChemPotAnalyzer(
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
            vr = get_vasprun(
                os.path.join(self._root_fldr, "dielectric", "vasprun.xml"), parse_potcar_file=False
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
