# coding: utf-8
"""
A class for performing analysis of chemical potentials with the grand
canonical linear programming approach
"""
from __future__ import division

import copy
import logging
import os

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Element, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester


def get_mp_chempots_from_dpd(dpd):
    """
    Grab Materials Project chemical potentials from a pymatgen DefectPhaseDiagram object
    """
    print("Retrieving chemical potentials from MP database using dpd object...")
    bulk_energy = 0.0
    if ("bulk_energy" in dpd.entries[0].parameters.keys()) and (
        "bulk_sc_structure" in dpd.entries[0].parameters.keys()
    ):
        try:
            bulk_struct = dpd.entries[0].parameters["bulk_sc_structure"].copy()
            if not isinstance(bulk_struct, Structure):
                bulk_struct = Structure.from_dict(bulk_struct)
            bulk_energy = dpd.entries[0].parameters["bulk_energy"]
        except:
            print(
                "Failure in grabbing bulk energy and structure for chemical potential parsing."
            )
    if not bulk_energy:
        print(
            "Grabbing chemical potentials without analyzing stability of bulk structure. "
            "Ignore any flags raised about stability of structure"
        )
        bulk_energy = 0.0
        bulk_struct = dpd.entries[0].defect.bulk_structure.copy()

    bulk_ce = ComputedStructureEntry(bulk_struct, bulk_energy)
    bulk_elt_set = list(bulk_struct.symbol_set)

    sub_species = []
    for entry in dpd.entries:
        def_site = entry.defect.site.specie.symbol
        if def_site not in bulk_elt_set:
            sub_species.append(def_site)

    sub_species = set(sub_species)
    print("Bulk symbols = {}, Sub symbols = {}".format(bulk_elt_set, sub_species))
    mp_cpa = MPChemPotAnalyzer(bulk_ce=bulk_ce, sub_species=sub_species)

    return mp_cpa.analyze_GGA_chempots()


class ChemPotAnalyzer:
    """
    Post processing for atomic chemical potentials used in defect
    calculations.
    """

    def __init__(self, **kwargs):
        """
        Args:
            bulk_ce: Pymatgen ComputedStructureEntry object for
                bulk entry / supercell
        """
        self.bulk_ce = kwargs.get("bulk_ce", None)

    def get_chempots_from_pd(self, pd):
        logger = logging.getLogger(__name__)

        if not self.bulk_ce:
            msg = (
                "No bulk entry supplied. "
                "Cannot compute atomic chempots without knowing the bulk entry of interest."
            )
            logger.warning(msg)
            raise ValueError(msg)

        bulk_composition = self.bulk_ce.composition
        redcomp = bulk_composition.reduced_composition
        # append bulk_ce to phase diagram, if not present
        entries = pd.all_entries
        if not any(
            [
                (
                    ent.composition == self.bulk_ce.composition
                    and ent.energy == self.bulk_ce.energy
                )
                for ent in entries
            ]
        ):
            entries.append(
                PDEntry(
                    self.bulk_ce.composition,
                    self.bulk_ce.energy,
                    attribute="Bulk Material",
                )
            )
            pd = PhaseDiagram(entries)

        chem_lims = pd.get_all_chempots(redcomp)

        return chem_lims

    def diff_bulk_sub_phases(self, face_list, sub_el=None):
        # method for pulling out phases within a facet of a phase diagram
        # which may include a substitutional element...
        # face_list is an array of phases in a facet
        # sub_el is the element to look out for within the face_list array
        blk = []
        sub_spcs = []
        for face in face_list:
            if sub_el:
                if sub_el in face:
                    sub_spcs.append(face)
                else:
                    blk.append(face)
            else:
                blk.append(face)
        blk.sort()
        sub_spcs.sort()
        blknom = "-".join(blk)
        subnom = "-".join(sub_spcs)
        return blk, blknom, subnom


class MPChemPotAnalyzer(ChemPotAnalyzer):
    """
    Post processing for atomic chemical potentials by querying MP database

    Makes use of Materials Project pre-computed data to generate
    needed information for chem pots in different growth conditions.

    WARNING: If you plan to use this method, then you better be sure you are
    using the same settings as MP (same INCAR, POTCARs etc.)
    """

    def __init__(self, **kwargs):
        """
        Args:
            bulk_ce: Pymatgen ComputedStructureEntry object for
                bulk entry / supercell
            sub_species (set): set of elemental species that are extrinsic to structure.
                Default is no subs included
            entries (dict): a dict of pymatgen ComputedEntry objects to build relevant phase diagram
                The dict contains two keys: 'bulk_derived', and 'subs_set', each contains a list
                of ComputedEntry objects
                'bulk_derived' list only has compositions containing elements from the bulk (
                un-defective) composition
                'subs_set' list has compositions which contain at least one element that is not
                in the bulk composition
            mpid (str): Materials Project ID of bulk structure (not required, can use bulk_ce
            instead);
                format "mp-X", where X is an integer;
            mapi_key (str): Materials API key to access database
                (if not in ~/.pmgrc.yaml already)
        """
        super(self.__class__, self).__init__(**kwargs)
        self.sub_species = kwargs.get("sub_species", set())
        self.entries = kwargs.get("entries", {})
        self.mpid = kwargs.get("mpid", None)
        self.mapi_key = kwargs.get("mapi_key", None)

    def analyze_GGA_chempots(self, full_sub_approach=False):
        """
        For calculating GGA-PBE atomic chemical potentials by using
            Materials Project pre-computed data

        Args:
            full_sub_approach: generate chemical potentials by looking at
                full phase diagram (setting to True is NOT recommended
                if subs_species set has more than one element in it...)

        This code retrieves atomic chempots from Materials
        Project (MP) entries by making use of the pymatgen
        phase diagram (PD) object and computed entries from the MP
        database. There are debug notes that are made based on the stability of
        the structure of interest with respect to the phase diagram generated from MP

        NOTE on 'full_sub_approach':
            The default approach for substitutional elements (full_sub_approach = False)
            is to only consider facets which have a maximum of 1 composition with
            the extrinsic species present
            (see PyCDT paper for chemical potential methodology DOI: 10.1016/j.cpc.2018.01.004).

            This default approach speeds up analysis when analyzing several substitutional
            species at the same time.

            If you prefer to consider the full phase diagram (not recommended
            when you have more than 2 substitutional defects), then set
            full_sub_approach to True.
        """
        logger = logging.getLogger(__name__)

        # gather entries
        self.get_mp_entries(full_sub_approach=full_sub_approach)

        # figure out how system should be treated for chemical potentials
        # based on phase diagram
        entry_list = self.entries["bulk_derived"]
        pd = PhaseDiagram(entry_list)

        decomp_en = round(
            pd.get_decomp_and_e_above_hull(self.bulk_ce, allow_negative=True)[1], 4
        )

        stable_composition_exists = False
        for i in pd.stable_entries:
            if i.composition.reduced_composition == self.redcomp:
                stable_composition_exists = True

        if (decomp_en <= 0) and stable_composition_exists:
            logger.debug(
                "Bulk Computed Entry found to be stable with respect "
                "to MP Phase Diagram (e_above_hull = {} eV/atom).".format(decomp_en)
            )
        elif (decomp_en <= 0) and not stable_composition_exists:
            logger.info(
                "Bulk Computed Entry found to be stable with respect "
                "to MP Phase Diagram (e_above_hull = {} eV/atom).\n"
                "However, no stable entry with this composition exists "
                "in the MP database!\nPlease consider submitting the "
                "POSCAR to the MP xtaltoolkit, so future users will "
                "know about this structure:"
                " https://materialsproject.org/#apps/xtaltoolkit\n"
                "Manually inserting structure into phase diagram and "
                "proceeding as normal.".format(decomp_en)
            )
            entry_list.append(self.bulk_ce)
        elif stable_composition_exists:
            logger.warning(
                "Bulk Computed Entry not stable with respect to MP "
                "Phase Diagram (e_above_hull = {} eV/atom), but found "
                "stable MP composition to exist.\nProducing chemical "
                "potentials with respect to stable phase.".format(decomp_en)
            )
        else:
            logger.warning(
                "Bulk Computed Entry not stable with respect to MP "
                "Phase Diagram (e_above_hull = {} eV/atom) and no "
                "stable structure with this composition exists in the "
                "MP database.\nProceeding with atomic chemical "
                "potentials according to composition position within "
                "phase diagram.".format(decomp_en)
            )

        pd = PhaseDiagram(entry_list)
        chem_lims = self.get_chempots_from_pd(pd)
        logger.debug("Bulk Chemical potential facets: {}".format(chem_lims.keys()))

        if not full_sub_approach:
            # NOTE if full_sub_approach was True, then all the sub_entries
            # would be ported into the bulk_derived list
            finchem_lims = {}  # this will be final chem_lims dictionary
            for key in chem_lims.keys():
                face_list = key.split("-")
                blk, blknom, subnom = self.diff_bulk_sub_phases(face_list)
                finchem_lims[blknom] = {}
                finchem_lims[blknom] = chem_lims[key]

            # Now add single elements to extend the phase diagram,
            # adding new additions to chemical potentials ONLY for the cases
            # where the phases in equilibria are those from the bulk phase
            # diagram. This is essentially the assumption that the majority of
            # the elements in the total composition will be from the native
            # species present rather than the sub species (a good approximation)
            for sub_el in self.sub_species:
                sub_specie_entries = entry_list[:]
                for entry in self.entries["subs_set"][sub_el]:
                    sub_specie_entries.append(entry)

                pd = PhaseDiagram(sub_specie_entries)
                chem_lims = self.get_chempots_from_pd(pd)

                for key in chem_lims.keys():
                    face_list = key.split("-")
                    blk, blknom, subnom = self.diff_bulk_sub_phases(
                        face_list, sub_el=sub_el
                    )
                    # if number of facets from bulk phase diagram is
                    # equal to bulk species then full_sub_approach says this
                    # can be grouped with rest of structures
                    if len(blk) == len(self.bulk_species_symbol):
                        if blknom not in finchem_lims.keys():
                            finchem_lims[blknom] = chem_lims[key]
                        else:
                            finchem_lims[blknom][Element(sub_el)] = chem_lims[key][
                                Element(sub_el)
                            ]
                        if "name-append" not in finchem_lims[blknom].keys():
                            finchem_lims[blknom]["name-append"] = subnom
                        else:
                            finchem_lims[blknom]["name-append"] += "-" + subnom
                    else:
                        # if chem pots determined by two (or more) sub-specie
                        # containing phases, skip this facet!
                        continue
            # run a check to make sure all facets dominantly defined by bulk species
            overdependent_chempot = False
            facets_to_delete = []
            for facet_name, cps in finchem_lims.items():
                cp_key_num = (
                    (len(cps.keys()) - 1) if "name-append" in cps else len(cps.keys())
                )
                if cp_key_num != (
                    len(self.bulk_species_symbol) + len(self.sub_species)
                ):
                    facets_to_delete.append(facet_name)
                    logger.info(
                        "Not using facet {} because insufficient number of bulk facets for "
                        "bulk set {} with sub_species set {}. (only dependent on {})."
                        "".format(
                            facet_name,
                            self.bulk_species_symbol,
                            self.sub_species,
                            cps.get("name-append"),
                        )
                    )
            if len(facets_to_delete) == len(finchem_lims):
                overdependent_chempot = True
                logger.warning(
                    "Determined chemical potentials to be over dependent"
                    " on a substitutional specie. Needing to revert to full_sub_approach. If "
                    "multiple sub species exist this could take a while/break the code..."
                )
            else:
                finchem_lims = {
                    k: v for k, v in finchem_lims.items() if k not in facets_to_delete
                }

            if not overdependent_chempot:
                chem_lims = {}
                for orig_facet, fc_cp_dict in finchem_lims.items():
                    if "name-append" not in fc_cp_dict:
                        facet_nom = orig_facet
                    else:
                        full_facet_list = orig_facet.split("-")
                        full_facet_list.extend(fc_cp_dict["name-append"].split("-"))
                        full_facet_list.sort()
                        facet_nom = "-".join(full_facet_list)
                    chem_lims[facet_nom] = {
                        k: v for k, v in fc_cp_dict.items() if k != "name-append"
                    }
            else:
                # This is for when overdetermined chempots occur, forcing the full_sub_approach
                # to happen
                for sub, subentries in self.entries["subs_set"].items():
                    for subentry in subentries:
                        entry_list.append(subentry)
                pd = PhaseDiagram(entry_list)
                chem_lims = self.get_chempots_from_pd(pd)

        return chem_lims

    def get_chempots_from_composition(self, bulk_composition, full_sub_approach=False):
        """
        A simple method for getting GGA-PBE chemical potentials JUST
        from the composition information (Note: this only works if the
        composition already exists in the MP database)

        Args:
            bulk_composition : Composition of bulk as a pymatgen Composition
                object. This and mapi_key are only actual required input for
                generating set of chemical potentials from Materials Project
                database
            full_sub_approach : Include sub_species in query, sub entries can
                be found in self.entries['subs_set']
        """
        logger = logging.getLogger(__name__)

        redcomp = bulk_composition.reduced_composition
        self.bulk_species_symbol = [s.symbol for s in redcomp.elements]

        if not self.entries:
            if (
                full_sub_approach
            ):  # this can be time consuming if several sub species exist
                species_symbols = self.bulk_species_symbol[:]
                for sub_el in self.sub_species:
                    species_symbols.append(sub_el)

                with MPRester(api_key=self.mapi_key) as mp:
                    self.entries["bulk_derived"] = mp.get_entries_in_chemsys(
                        species_symbols
                    )

                self.entries["subs_set"] = {sub_el: [] for sub_el in self.sub_species}
                for entry in self.entries["bulk_derived"]:
                    for sub_el in self.sub_species:
                        if sub_el in entry.composition:
                            self.entries["subs_set"][sub_el].append(entry)

            else:
                with MPRester(api_key=self.mapi_key) as mp:
                    self.entries["bulk_derived"] = mp.get_entries_in_chemsys(
                        self.bulk_species_symbol
                    )

        pd = PhaseDiagram(self.entries["bulk_derived"])
        chem_lims = pd.get_all_chempots(redcomp)

        return chem_lims

    def get_mp_entries(self, full_sub_approach=False):
        """
        This queries MP database for computed entries according to
        input bulk and sub elements of interest

        Args:
            mpid (str): Structure id of the system in the MP databse.
            mapi_key (str): Materials API key to access database
                (if not in ~/.pmgrc.yaml already)
        """
        logger = logging.getLogger(__name__)

        if self.bulk_ce:
            self.bulk_species_symbol = [
                s.symbol for s in self.bulk_ce.composition.elements
            ]
            self.redcomp = self.bulk_ce.composition.reduced_composition
            bce_override = True
        elif self.mpid:
            with MPRester(api_key=self.mapi_key) as mp:
                self.bulk_ce = mp.get_entry_by_material_id(self.mpid)
            self.bulk_species_symbol = [
                s.symbol for s in self.bulk_ce.composition.elements
            ]
            self.redcomp = self.bulk_ce.composition.reduced_composition
            bce_override = False
        else:
            msg = (
                "No bulk entry OR mpid supplied. "
                "Cannot compute atomic chempots without know the bulk entry of interest."
            )
            logger.warning(msg)
            raise ValueError(msg)

        if full_sub_approach:  # this can be time consuming if several sub species exist
            species_symbols = self.bulk_species_symbol[:]
            for sub_el in self.sub_species:
                species_symbols.append(sub_el)

            with MPRester(api_key=self.mapi_key) as mp:
                self.entries["bulk_derived"] = mp.get_entries_in_chemsys(
                    species_symbols
                )

            self.entries["subs_set"] = {sub_el: [] for sub_el in self.sub_species}
            for entry in self.entries["bulk_derived"]:
                for sub_el in self.sub_species:
                    if sub_el in entry.composition:
                        self.entries["subs_set"][sub_el].append(entry)

        else:  # this is recommended approach for running sub species seperately (assumes subs
            # are in dilute concentrations)
            with MPRester(api_key=self.mapi_key) as mp:
                self.entries["bulk_derived"] = mp.get_entries_in_chemsys(
                    self.bulk_species_symbol
                )
                if self.mpid and bce_override:  # overriding bulk_ce if mp-id is given.
                    self.bulk_ce = mp.get_entry_by_material_id(self.mpid)
            if not self.entries:
                msg = (
                    "Could not fetch bulk entries for atomic chempots!"
                    "MPRester query error."
                )
                logger.warning(msg)
                raise ValueError(msg)

            # now compile substitution entries
            self.entries["subs_set"] = dict()
            bulk_entry_set = [entry.entry_id for entry in self.entries["bulk_derived"]]
            for sub_el in self.sub_species:
                els = self.bulk_species_symbol + [sub_el]
                with MPRester(api_key=self.mapi_key) as mp:
                    sub_entry_set = mp.get_entries_in_chemsys(els)
                if not sub_entry_set:
                    msg = (
                        "Could not fetch sub entries for {} atomic chempots! "
                        "Encountered MPRester query error".format(sub_el)
                    )
                    logger.warning(msg)
                    raise ValueError(msg)

                fin_sub_entry_set = []
                for entry in sub_entry_set:
                    if entry.entry_id not in bulk_entry_set:
                        fin_sub_entry_set.append(entry)
                # All entries apart from the bulk entry set
                self.entries["subs_set"][sub_el] = fin_sub_entry_set


class UserChemPotAnalyzer(ChemPotAnalyzer):
    """
    Post processing for atomic chemical potentials based on user computed
    phase diagram entries (possibly supplemented with MP database entries)
    """

    def __init__(self, **kwargs):
        """
        Args:
            bulk_ce: Pymatgen ComputedStructureEntry object for bulk entry
                or supercell
            path_base (str): the base path where the 'PhaseDiagram' folder
                exists defaults to the local folder
            sub_species (set): set of elemental species that are extrinsic
                to structure. Default is no subs included
            entries (dict): pymatgen ComputedEntry objects to build phase
                diagram The dict contains two keys: 'bulk_derived', and
                'subs_set', each contains a list of computed entries
                bulk_derived entries only have a composition containing
                elements from the set of elements in the bulk phase
                subs_set contains elements that are extrinsic to the
                structure of interest
            mapi_key (str): Materials API key to access database
                (if not in ~/.pmgrc.yaml already)
        """
        super(self.__class__, self).__init__(**kwargs)
        self.path_base = kwargs.get("path_base", ".")
        self.sub_species = kwargs.get("sub_species", set())
        self.entries = kwargs.get("entries", {})
        self.mapi_key = kwargs.get("mapi_key", None)

    def read_phase_diagram_and_chempots(
        self, full_sub_approach=False, include_mp_entries=True
    ):
        """
        Once phase diagram has been set up and run by user (in a folder
        called "PhaseDiagram"), this method parses and prints the chemical
        potentials based on the computed entries. The methodology is
        basically identical to that in the analyze_GGA_chempots method.

        Will supplement unfinished entries with MP database entries
        unless no_mp_entries is set to False

        Args:
            full_sub_approach: same attribute as described at length in
                the analyze_GGA_chempots method. Basically, the user can
                set this to True if they want to mix extrinsic species
                in the phase diagram

            include_mp_entries: if set to True, extra entries from
                Materials Project will be added to phase diagram
                according to phases that are stable in the Materials
                Project database

        """
        pdfile = os.path.join(self.path_base, "PhaseDiagram")
        if not os.path.exists(pdfile):
            print("Phase diagram file does not exist at ", pdfile)
            return

        # this is where we read computed entries into a list for parsing...
        # NOTE TO USER: If not running with VASP need to use another
        # pymatgen functionality for importing computed entries below...
        personal_entry_list = []
        for structfile in os.listdir(pdfile):
            if os.path.exists(
                os.path.join(pdfile, structfile, "vasprun.xml")
            ) or os.path.exists(os.path.join(pdfile, structfile, "vasprun.xml.gz")):
                try:
                    print("loading ", structfile)
                    from doped.pycdt.utils.parse_calculations import get_vasprun

                    vr, vr_path = get_vasprun(
                        os.path.join(pdfile, structfile, "vasprun.xml")
                    )
                    vr_entry = vr.get_computed_entry()
                    pdentry = PDEntry(
                        vr_entry.composition, vr_entry.energy, attribute=structfile
                    )
                    personal_entry_list.append(pdentry)
                except:
                    print("Could not load ", structfile)

            else:
                print("No vasprun.xml(.gz) found in ", structfile)

        # add bulk computed entry to phase diagram, and see if it is stable
        if not self.bulk_ce:
            vr_path = os.path.join(self.path_base, "bulk", "vasprun.xml")
            if os.path.exists(vr_path):
                print("loading bulk computed entry")
                from doped.pycdt.utils.parse_calculations import get_vasprun

                bulkvr, bulkvr_path = get_vasprun(vr_path)
                bulkvr_entry = bulkvr.get_computed_entry()
                self.bulk_ce = PDEntry(
                    bulkvr_entry.composition, bulkvr_entry.energy, attribute=bulkvr_path
                )
            else:
                print(
                    "No bulk entry given locally. Phase diagram "
                    + "calculations cannot be set up without this"
                )
                return

        self.bulk_composition = self.bulk_ce.composition
        self.redcomp = self.bulk_composition.reduced_composition
        self.bulk_species_symbol = [s.symbol for s in self.bulk_ce.composition.elements]

        # Supplement entries to phase diagram with those from MP database
        if include_mp_entries:
            mpcpa = MPChemPotAnalyzer(
                bulk_ce=self.bulk_ce,
                sub_species=self.sub_species,
                mapi_key=self.mapi_key,
            )
            tempcl = mpcpa.analyze_GGA_chempots(
                full_sub_approach=full_sub_approach
            )  # Use MPentries

            curr_pd = PhaseDiagram(
                list(
                    set().union(
                        mpcpa.entries["bulk_derived"], mpcpa.entries["subs_set"]
                    )
                )
            )
            stable_idlist = {
                i.composition.reduced_composition: [i.energy_per_atom, i.entry_id, i]
                for i in curr_pd.stable_entries
            }
            for mpcomp, mplist in stable_idlist.items():
                matched = False
                for pe in personal_entry_list:
                    if pe.composition.reduced_composition == mpcomp:
                        # #USER: uncomment this if you want additional stable phases of identical
                        # composition included in your phase diagram
                        # if personalentry.energy_per_atom > mplist[0]:
                        #     print('Adding entry from MP-database:',mpcomp,'(entry-id:',mplist[1])
                        #     personal_entry_list.append(mplist[2])
                        matched = True
                if not matched:
                    print(
                        "Adding entry from MP-database:",
                        mpcomp,
                        "(entry-id:",
                        mplist[1],
                    )
                    personal_entry_list.append(mplist[2])
        else:
            # personal_entry_list.append(self.bulk_ce)
            # if you dont have entries for elemental corners of phase diagram then code breaks
            # manually inserting entries with energies of zero for competeness...USER DO NOT USE
            # THIS
            eltcount = {elt: 0 for elt in set(self.bulk_ce.composition.elements)}
            for pentry in personal_entry_list:
                if (
                    pentry.is_element
                    and pentry.composition.elements[0] in eltcount.keys()
                ):
                    eltcount[pentry.composition.elements[0]] += 1
            for elt, eltnum in eltcount.items():
                if not eltnum:
                    s = Structure(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [elt],
                        [[0, 0, 0]],
                    )
                    eltentry = ComputedStructureEntry(s, 0.0)
                    print(
                        "USER! Note that you have added a fake "
                        + str(elt)
                        + " structure to prevent from breaking the "
                        "Phase Diagram Analyzer.\n As a result DO NOT trust the chemical "
                        "potential results for regions "
                        "of phase diagram that involve the element " + str(elt)
                    )
                    personal_entry_list.append(
                        PDEntry(eltentry.composition, eltentry.energy)
                    )

        personal_entry_list.append(
            PDEntry(
                self.bulk_ce.composition, self.bulk_ce.energy, attribute="Bulk Material"
            )
        )
        unique_entries = list(
            set(personal_entry_list)
        )  # remove duplicate entries if present

        # compute chemical potentials
        if full_sub_approach:
            pd = PhaseDiagram(unique_entries)
            chem_lims = self.get_chempots_from_pd(pd)
        else:
            # first separate out the bulk associated elements from those of substitutional elements
            entry_list = []
            sub_associated_entry_list = []
            for localentry in personal_entry_list:
                bulk_associated = True
                for elt in localentry.composition.elements:
                    if elt not in self.bulk_composition.elements:
                        bulk_associated = False

                if bulk_associated:
                    entry_list.append(localentry)
                else:
                    sub_associated_entry_list.append(localentry)

            # now iterate through and collect chemical potentials
            pd = PhaseDiagram(entry_list)
            chem_lims = self.get_chempots_from_pd(pd)

            finchem_lims = {}  # this will be final chem_lims dictionary
            for key in chem_lims.keys():
                face_list = key.split("-")
                blk, blknom, subnom = self.diff_bulk_sub_phases(face_list)
                finchem_lims[blknom] = {}
                finchem_lims[blknom] = chem_lims[key]

            # Now consider adding single elements to extend the phase diagram,
            # adding new additions to chemical potentials ONLY for the cases
            # where the phases in equilibria are those from the bulk phase
            # diagram. This is essentially the assumption that the majority of
            # the elements in the total composition will be from the native
            # species present rather than the sub species (a good approximation)
            for sub_el in self.sub_species:
                sub_specie_entries = entry_list[:]
                for entry in sub_associated_entry_list:
                    if sub_el in entry.composition.elements:
                        sub_specie_entries.append(entry)

                pd = PhaseDiagram(sub_specie_entries)
                chem_lims = self.get_chempots_from_pd(pd)

                for key in chem_lims.keys():
                    face_list = key.split("-")
                    blk, blknom, subnom = self.diff_bulk_sub_phases(
                        face_list, sub_el=sub_el
                    )
                    # if one less than number of bulk species then can be
                    # grouped with rest of structures
                    if len(blk) == len(self.bulk_species_symbol):
                        if blknom not in finchem_lims.keys():
                            finchem_lims[blknom] = chem_lims[key]
                        else:
                            finchem_lims[blknom][sub_el] = chem_lims[key][sub_el]
                        if "name-append" not in finchem_lims[blknom].keys():
                            finchem_lims[blknom]["name-append"] = subnom
                        else:
                            finchem_lims[blknom]["name-append"] += "-" + subnom
                    else:
                        # if chem pots determined by two (or more) sub-specie
                        # containing phases, skip this facet!
                        continue

            # run a check to make sure all facets dominantly defined by bulk species
            overdependent_chempot = False
            facets_to_delete = []
            for facet_name, cps in finchem_lims.items():
                cp_key_num = (
                    (len(cps.keys()) - 1) if "name-append" in cps else len(cps.keys())
                )
                if cp_key_num != (
                    len(self.bulk_species_symbol) + len(self.sub_species)
                ):
                    facets_to_delete.append(facet_name)
                    print(
                        "Not using facet {} because insufficient number of bulk facets for "
                        "bulk set {} with sub_species set {}. (only dependent on {})."
                        "".format(
                            facet_name,
                            self.bulk_species_symbol,
                            self.sub_species,
                            cps.get("name-append"),
                        )
                    )
            if len(facets_to_delete) == len(finchem_lims):
                overdependent_chempot = True
                print(
                    "Determined chemical potentials to be over dependent"
                    " on a substitutional specie. Needing to revert to full_sub_approach. If "
                    "multiple sub species exist this could take a while/break the code..."
                )
            else:
                finchem_lims = {
                    k: v for k, v in finchem_lims.items() if k not in facets_to_delete
                }

            if not overdependent_chempot:
                chem_lims = {}
                for orig_facet, fc_cp_dict in finchem_lims.items():
                    if "name-append" not in fc_cp_dict:
                        facet_nom = orig_facet
                    else:
                        full_facet_list = orig_facet.split("-")
                        full_facet_list.extend(fc_cp_dict["name-append"].split("-"))
                        full_facet_list.sort()
                        facet_nom = "-".join(full_facet_list)
                    chem_lims[facet_nom] = {
                        k: v for k, v in fc_cp_dict.items() if k != "name-append"
                    }
            else:
                # This is for when overdetermined chempots occur, forcing the full_sub_approach
                # to happen
                for sub, subentries in self.entries["subs_set"].items():
                    for subentry in subentries:
                        entry_list.append(subentry)
                pd = PhaseDiagram(entry_list)
                chem_lims = self.get_chempots_from_pd(pd)

        self.phase_diagram = pd
        chem_lims = {
            "facets": chem_lims,
            "elemental_refs": {
                elt: ent.energy_per_atom
                for elt, ent in self.phase_diagram.el_refs.items()
            },
            "facets_wrt_elt_refs": {},
        }
        for facet, chempot_dict in chem_lims["facets"].items():
            rel_chempot_dict = copy.deepcopy(chempot_dict)
            for elt, chempot_energy in rel_chempot_dict.items():
                rel_chempot_dict[elt] -= chem_lims["elemental_refs"][elt]
            chem_lims["facets_wrt_elt_refs"].update({facet: rel_chempot_dict})
        return chem_lims


class UserChemPotInputGenerator(object):
    """
    For setting up phase diagram for user, based on structures that exist in the MP database
    """

    def __init__(
        self, bulk_composition, sub_species=set(), path_base=".", mapi_key=None
    ):
        """
        Args:
            bulk_composition : Composition of bulk as a pymatgen Composition
                object. This and mapi_key are only actual required input for
                generating set of chemical potentials from Materials Project
                database
            sub_species : set of elemental species that are extrinsic to
                structure defaults to No substitutions needed.
            path_base (str): the base path where the 'PhaseDiagram' folder should be created
                defaults to the local folder
            mapi_key (str): Materials API key to access database
                (if not in ~/.pmgrc.yaml already)
        """
        self.bulk_composition = bulk_composition
        self.bulk_species_symbol = [s.symbol for s in bulk_composition.elements]
        self.redcomp = bulk_composition.reduced_composition
        self.sub_species = sub_species
        self.path_base = path_base
        self.mapi_key = mapi_key
        self.MPC = MPChemPotAnalyzer(sub_species=sub_species)

    def setup_phase_diagram_calculations(
        self,
        full_phase_diagram=False,
        energy_above_hull=0,
        struct_fmt="poscar",
        write_files=False,
        overwrite=False,
        include_elements=True,
        full_sub_approach=True,
    ):
        """
        This method allows for setting up local phase diagram calculations so a user can calculate
        chemical potentials on a level of interest beyond PBE-GGA/GGA+U
        Method is to pull the MP phase diagram and use PBE-GGA level data to decide which phases
        need to be computed

        full_phase_diagram flag has two options:
            False: set up the structures/phases which are stable in GGA phase diagram and are
            relevant for defining
                    the chemical potentials (exist to define the facets adjacent to composition
                    of interest)
            True:  set up the full phase diagram according to all the entries in the MP database
            with elements of interest

        entry_above_hull: allows for a range of energies above hull for each composition being
        set up
                default is 0, meaning just the PBE-GGA ground state phases are set up. If you set
                value to 0.5 then all
                phases within 0.5 eV/atom of PBE-GGA ground state hull will be set up etc.

        struct_fmt: is file format you want structure to be written as. Options are “cif”,
        “poscar”, “cssr”, and “json”
                    Defaults to "poscar"

        write_files: write the calculation folders and structure files to the PhaseDiagram
        folder. Defaults to False.

        overwrite: write files even if PhaseDiagram folder already exists. Defaults to False.

        include_elements: set up all elemental reference phases as well (necessary to properly
            determine absolute chemical potentials, zero-reference elemental energies etc.),
            regardless of whether the elemental phase is a local facet adjacent to the MP
            GGA-calculated stable phase space for the composition of interest. Defaults to True.

        full_sub_approach : Include sub_species in query, sub entries can then be found in
            self.MPC.entries['subs_set']

        """

        # while GGA chem pots won't be used here; use this method for quickly gathering phase
        # diagram object entries AND to find phases of interest if you just want to re-calculate
        # local facets
        if full_sub_approach and self.sub_species:
            MPgga_muvals = self.MPC.get_chempots_from_composition(
                self.bulk_composition, full_sub_approach=True
            )
        else:
            MPgga_muvals = self.MPC.get_chempots_from_composition(self.bulk_composition)

        if full_phase_diagram:
            setupphases = set(
                [localentry.name for localentry in self.MPC.entries["bulk_derived"]]
            )  # all elements in
            # phase diagram
        else:
            if (
                len(self.bulk_composition) == 2
            ):  # neccessary because binary species have chempots
                # written as "A-rich, B-rich"
                setupphases = set(
                    [
                        phase.split("_")[0]
                        for facet in MPgga_muvals.keys()
                        for phase in facet.split("-")
                    ]
                )
            else:
                setupphases = set(
                    [
                        phase
                        for facet in MPgga_muvals.keys()
                        for phase in facet.split("-")
                    ]
                )  # just local facets

            if (
                include_elements
            ):  # add elemental reference phases to structures to setup
                for localentry in self.MPC.entries["bulk_derived"]:
                    if localentry.is_element:
                        setupphases.add(
                            localentry.name
                        )  # set.add() only adds entry if not
                        # already in set

        structures_to_setup = (
            {}
        )  # this will be a list of structure objects which need to be
        # setup locally

        # create phase diagram object for analyzing PBE-GGA energetics of structures computed in
        # MP database
        full_structure_entries = [struct for struct in self.MPC.entries["bulk_derived"]]
        pd = PhaseDiagram(full_structure_entries)

        for entry in full_structure_entries:
            if (entry.name in setupphases) and (
                pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]
                <= energy_above_hull
            ):
                with MPRester(api_key=self.mapi_key) as mp:
                    localstruct = mp.get_structure_by_material_id(entry.entry_id)

                # Name to two significant figures
                name = (
                    str(entry.name)
                    + "_EaH="
                    + f"{pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]:.2g}"
                )
                if (
                    name in structures_to_setup.keys()
                ):  # Is 2 sig. figures rounding to same
                    # value for two entries?
                    name = (
                        str(entry.name)
                        + "_EaH="
                        + f"{pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]:.3g}"
                    )
                try:
                    sg_symbol = localstruct.get_space_group_info()[0]
                except TypeError:
                    try:
                        sg_symbol = localstruct.get_space_group_info(symprec=1e-1)[0]
                    except:
                        try:
                            sg_symbol = localstruct.get_space_group_info(symprec=1e0)[0]
                        except:
                            sg_symbol = None

                structures_to_setup[name] = {
                    "Structure": localstruct,
                    "Energy above Hull": pd.get_decomp_and_e_above_hull(
                        entry, allow_negative=True
                    )[1],
                    "MP Entry ID": entry.entry_id,
                    "Space Group": sg_symbol,
                }

        # Set up structure files locally if desired
        if not write_files:
            print("Returning chempot structures, but not making POSCAR files.")
        else:
            if (
                os.path.exists(os.path.join(self.path_base, "PhaseDiagram"))
                and not overwrite
            ):
                print(
                    "PhaseDiagram folder already exists! Set overwrite = True if you want me to overwrite it"
                )
            else:
                if (
                    os.path.exists(os.path.join(self.path_base, "PhaseDiagram"))
                    and overwrite
                ):
                    print(
                        "PhaseDiagram folder already exists, but f@ck it we're gonna (over)write files"
                    )
                if not os.path.exists(os.path.join(self.path_base, "PhaseDiagram")):
                    os.makedirs(os.path.join(self.path_base, "PhaseDiagram"))
                for localname in structures_to_setup.keys():
                    filename = os.path.join(self.path_base, "PhaseDiagram", localname)
                    if not os.path.isdir(filename):
                        os.makedirs(filename)
                    if struct_fmt == "poscar":
                        outputname = "POSCAR"
                    else:
                        outputname = "structfile"
                    structures_to_setup[localname]["Structure"].to(
                        fmt=struct_fmt, filename=os.path.join(filename, outputname)
                    )
                    # NOTE TO USER. Can use pymatgen here to setup additional calculation
                    # files if interested...

        return structures_to_setup
