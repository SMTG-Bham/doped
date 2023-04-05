# coding: utf-8
from __future__ import division

"""
Code to generate charged defects structure files.
"""


import abc
import re
from tabulate import tabulate

from monty.serialization import dumpfn
from pymatgen.analysis.defects.core import Interstitial
from pymatgen.analysis.defects.generators import (
    InterstitialGenerator,
    SimpleChargeGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
    VoronoiInterstitialGenerator,
)
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator as VIRE
from pymatgen.core.periodic_table import Element, Specie, get_el_sp
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_optimized_sc_scale(inp_struct, final_site_no):
    """
    Get the optimal scaling to generate supercells with atoms less than
    the final_site_no.
    """

    if final_site_no < len(inp_struct.sites):
        final_site_no = len(inp_struct.sites)

    dictio = {}
    result = []
    for k1 in range(1, 6):
        for k2 in range(1, 6):
            for k3 in range(1, 6):
                struct = inp_struct.copy()
                struct.make_supercell([k1, k2, k3])
                if len(struct.sites) > final_site_no:
                    continue

                min_dist = 1000.0
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            try:
                                distance = struct.get_distance(0, 0, (a, b, c))
                            except:
                                print(a, b, c)
                                raise
                            if distance < min_dist and distance > 0.00001:
                                min_dist = distance
                min_dist = round(min_dist, 3)
                if min_dist in dictio:
                    if dictio[min_dist]["num_sites"] > struct.num_sites:
                        dictio[min_dist]["num_sites"] = struct.num_sites
                        dictio[min_dist]["supercell"] = [k1, k2, k3]
                else:
                    dictio[min_dist] = {}
                    dictio[min_dist]["num_sites"] = struct.num_sites
                    dictio[min_dist]["supercell"] = [k1, k2, k3]
    min_dist = -1.0
    biggest = None
    for c in dictio:
        if c > min_dist:
            biggest = dictio[c]["supercell"]
            min_dist = c
    if biggest is None or min_dist < 0.0:
        raise RuntimeError("could not find any supercell scaling vector")
    return biggest


class DefectCharger:
    __metaclass__ = abc.ABCMeta
    """
    Abstract base class to define the properties of a defect charge generator
    """

    def __init__(self, structure):
        pass

    @abc.abstractmethod
    def get_charges(self, defect_type, site_specie=None, sub_specie=None):
        """
        Based on the type of defect, site and substitution (if any) species
        the defect charge states are generated.
        Args:
            defect_type (str): Options are vacancy, antisite, substitution,
                               and interstitial
            site_specie: Specie on the host lattice site
                         For interstitials, use this
            sub_specie: Specie that is replacing the site specie.
                        For antisites and substitution defects

        """
        raise NotImplementedError


class DefectChargerSemiconductor(DefectCharger):
    """
    Determine oxidation states from bond valence method (unless oxidation states specified)
    Use BVM or specified oxidation states for site oxidation preference
    Use possible oxidation state list for sub_site elements
    """

    def __init__(self, structure, min_max_oxi=None, oxi_states=None):
        """
        Charge assignment based on the oxidation states referenced from
        semiconductor database. Targetted materials are shallow and some
        wideband semiconductors. For these systems, antisites are common and
        their charge assignment for antisites follows vacancies
        Args: structure
            pymatgen structure object to determine the oxidation states
            min_max_oxi: any user specified min/max oxidation ranges for elements in structure
            oxi_states: any user specified oxidation states of elements in structure
        """
        min_max_oxi = min_max_oxi if min_max_oxi is not None else {}
        oxi_states = oxi_states if oxi_states is not None else {}

        struct_species = structure.types_of_specie
        if (len(struct_species) == 1) and struct_species[
            0
        ].symbol not in oxi_states.keys():
            oxi_states[struct_species[0].symbol] = 0
        else:
            vir = VIRE(structure)
            for elt, oxi in vir.valences.items():
                strip_key = "".join([s for s in elt if s.isalpha()])
                if strip_key not in oxi_states.keys():
                    oxi_states[strip_key] = oxi
        self.oxi_states = oxi_states

        self.min_max_oxi_bulk = [0, 0]
        for s in struct_species:
            if isinstance(s, Specie):
                el = s.element
            elif isinstance(s, Element):
                el = s
            else:
                continue
            if el.symbol not in min_max_oxi.keys():
                max_oxi = max(el.common_oxidation_states)
                min_oxi = min(el.common_oxidation_states)
                min_max_oxi[el.symbol] = [min_oxi, max_oxi]
            if min_max_oxi[el.symbol][0] < self.min_max_oxi_bulk[0]:
                self.min_max_oxi_bulk[0] = min_max_oxi[el.symbol][0]
            if min_max_oxi[el.symbol][1] > self.min_max_oxi_bulk[1]:
                self.min_max_oxi_bulk[1] = min_max_oxi[el.symbol][1]
        self.min_max_oxi = min_max_oxi

    def get_charges(self, defect_type, site_specie, sub_specie=None):
        """
        Based on the type of defect, site and substitution (if any) species
        the defect charge states are generated.
        Args:
            defect_type (str): Options are vacancy, antisite, substitution,
                               and interstitial
            site_specie (str): Specie on the host lattice site
                         Use this for interstitials as well
            sub_specie (str): Specie that is replacing the site specie.
                        At present used for substitution and antisite defects
        """
        if site_specie not in self.min_max_oxi.keys():
            max_oxi = max(Element(site_specie).common_oxidation_states)
            min_oxi = min(Element(site_specie).common_oxidation_states)
            self.min_max_oxi[site_specie] = [min_oxi, max_oxi]
        if sub_specie:
            if sub_specie not in self.min_max_oxi.keys():
                max_oxi = max(Element(sub_specie).common_oxidation_states)
                min_oxi = min(Element(sub_specie).common_oxidation_states)
                self.min_max_oxi[sub_specie] = [min_oxi, max_oxi]
        if defect_type == "vacancy":
            site_oxi = self.oxi_states[site_specie]
            if site_oxi:
                return list(range(-abs(site_oxi), abs(site_oxi) + 1))
            else:
                min_oxi = self.min_max_oxi[site_specie][0]
                max_oxi = self.min_max_oxi[site_specie][1]
                if abs(min_oxi) < abs(max_oxi):
                    return list(range(-abs(max_oxi), abs(max_oxi) + 1))
                else:
                    return list(range(-abs(min_oxi), abs(min_oxi) + 1))
        elif defect_type == "antisite":
            return list(range(self.min_max_oxi_bulk[0], self.min_max_oxi_bulk[1] - 1))
        elif defect_type == "substitution":
            oxi_sub_set = Element(sub_specie).common_oxidation_states
            oxi_site = self.oxi_states[site_specie]
            min_max_oxi_bulk_sub = [
                min(min(oxi_sub_set) - oxi_site, -1),
                max(max(oxi_sub_set) - oxi_site, 1),
            ]
            if (min_max_oxi_bulk_sub[1] - min_max_oxi_bulk_sub[0]) > 2:
                if min_max_oxi_bulk_sub[1] > 2:
                    return list(
                        range(min_max_oxi_bulk_sub[0], min_max_oxi_bulk_sub[1] - 2)
                    )  # if range exists, less likely to be a higher charge
                else:
                    return list(
                        range(min_max_oxi_bulk_sub[0], 2)
                    )  # extend to 2 if upper range does not already include this
            else:
                return list(
                    range(min_max_oxi_bulk_sub[0] - 1, min_max_oxi_bulk_sub[1] + 2)
                )  # likely upper bound is 0, so extend to 2
        elif defect_type == "interstitial":
            if self.min_max_oxi[site_specie][0] > 0:
                min_oxi = 0
            else:
                min_oxi = self.min_max_oxi[site_specie][0]
            if self.min_max_oxi[site_specie][1] < 0:
                max_oxi = 0
            else:
                max_oxi = self.min_max_oxi[site_specie][1]
            return list(range(min_oxi, max_oxi + 1))
        else:
            raise ValueError("Defect type not understood")


class DefectChargerInsulator(DefectCharger):
    """
    Conservative charge assignment based on the oxidation statess determined
    by bond valence. Targetted materials are wideband semiconductors and
    insulators. AxBy where A is cation and B is anion will have charge
    assignments {A: [0:y], B:[-x:0]}. For these systems, antisites typically
    have very high formation energies and are ignored.
    """

    def __init__(self, structure):
        """
        Conservative defect charge generator based on the oxidation statess
        determined by bond valence. Targetted materials are wideband
        semiconductors and insulators. AxBy where A is cation and B is
        anion will have charge assignments {A: [0:y], B:[-x:0]}. For these
        systems, antisites typically have very high formation energies and
        are ignored.
        Args:
            structure: pymatgen structure object
        """
        struct_species = structure.types_of_specie
        if len(struct_species) == 1:
            oxi_states = {struct_species[0].symbol: 0}
        else:
            vir = VIRE(structure)
            oxi_states = vir.valences
        self.oxi_states = {}
        for key, val in oxi_states.items():
            strip_key = "".join([s for s in key if s.isalpha()])
            self.oxi_states[strip_key] = val

        self.min_max_oxi = {}
        for s in struct_species:
            if isinstance(s, Specie):
                el = s.element
            elif isinstance(s, Element):
                el = s
            else:
                continue
            max_oxi = max(el.common_oxidation_states)
            min_oxi = min(el.common_oxidation_states)
            self.min_max_oxi[el.symbol] = (min_oxi, max_oxi)

    def get_charges(self, defect_type, site_specie=None, sub_specie=None):
        """
        Based on the type of defect, site and substitution (if any) species
        the defect charge states are generated.
        Args:
            defect_type (str): Options are vacancy, antisite, substitution,
                               and interstitial
            site_specie: Specie on the host lattice site
                         For interstitials, use this
            sub_specie: Specie that is replacing the site specie.
                        For antisites and substitution defects
        """
        if defect_type == "vacancy":
            vac_symbol = get_el_sp(site_specie).symbol
            vac_oxi_state = self.oxi_states[vac_symbol]
            if vac_oxi_state < 0:
                min_oxi = max(vac_oxi_state, self.min_max_oxi[vac_symbol][0])
                max_oxi = 0
            elif vac_oxi_state > 0:
                max_oxi = min(vac_oxi_state, self.min_max_oxi[vac_symbol][1])
                min_oxi = 0
            else:  # most probably single element
                oxi_states = get_el_sp(site_specie).common_oxidation_states
                min_oxi = min(oxi_states)
                max_oxi = max(oxi_states)
            return [-c for c in range(min_oxi, max_oxi + 1)]

        elif defect_type == "antisite":
            vac_symbol = get_el_sp(site_specie).symbol
            vac_oxi_state = self.oxi_states[vac_symbol]
            as_symbol = get_el_sp(sub_specie).symbol
            if vac_oxi_state > 0:
                oxi_max = max(self.min_max_oxi[as_symbol][1], 0)
                oxi_min = 0
            else:
                oxi_max = 0
                oxi_min = min(self.min_max_oxi[as_symbol][0], 0)
            return [c - vac_oxi_state for c in range(oxi_min, oxi_max + 1)]

        elif defect_type == "substitution":
            site_specie = get_el_sp(site_specie)
            sub_specie = get_el_sp(sub_specie)
            vac_symbol = site_specie.symbol
            vac_oxi_state = self.oxi_states[vac_symbol]

            max_oxi_sub = max(sub_specie.common_oxidation_states)
            min_oxi_sub = min(sub_specie.common_oxidation_states)
            if vac_oxi_state > 0:
                if max_oxi_sub < 0:
                    raise ValueError("Substitution seems not possible")
                else:
                    if max_oxi_sub > vac_oxi_state:
                        return list(range(max_oxi_sub - vac_oxi_state + 1))
                    else:
                        return [max_oxi_sub - vac_oxi_state]
            else:
                if min_oxi_sub > 0:
                    raise ValueError("Substitution seems not possible")
                else:
                    if min_oxi_sub < vac_oxi_state:
                        return list(range(min_oxi_sub - vac_oxi_state, 1))
                    else:
                        return [min_oxi_sub - vac_oxi_state]

        elif defect_type == "interstitial":
            site_specie = get_el_sp(site_specie)
            min_oxi = min(min(site_specie.common_oxidation_states), 0)
            max_oxi = max(max(site_specie.common_oxidation_states), 0)

            return list(range(min_oxi, max_oxi + 1))


class DefectChargerIonic(DefectCharger):
    """
    Charge assignments based on values expected purely from ionic theory, range to zero.
    Simple but good for first guesses.
    """

    def __init__(self, structure):
        """
        Args:
            structure: pymatgen structure object
        """
        struct_species = structure.types_of_specie
        if len(struct_species) == 1:
            oxi_states = {struct_species[0].symbol: 0}
        else:
            vir = VIRE(structure)
            oxi_states = vir.valences
        self.oxi_states = {}
        for key, val in oxi_states.items():
            strip_key = "".join([s for s in key if s.isalpha()])
            self.oxi_states[strip_key] = val

    def get_charges(self, defect_type, site_specie=None, sub_specie=None):
        """
        Based on the type of defect, site and substitution (if any) species
        the defect charge states are generated.
        Args:
            defect_type (str): Options are vacancy, antisite, substitution,
                               and interstitial
            site_specie: Specie on the host lattice site
                         For interstitials, use this
            sub_specie: Specie that is replacing the site specie.
                        For antisites and substitution defects
        """
        if defect_type == "vacancy":
            vac_symbol = get_el_sp(site_specie).symbol
            vac_oxi_state = self.oxi_states[vac_symbol]
            if vac_oxi_state == 0:
                return [-1, 0, 1]
            else:
                minval = min(-vac_oxi_state, 0)
                maxval = max(-vac_oxi_state, 0)
                return [c for c in range(minval - 1, maxval + 2)]

        elif defect_type in ["antisite", "substitution"]:
            # TODO: may cause some weird states for substitutions. Worth updating in future.
            vac_symbol = get_el_sp(site_specie).symbol
            vac_oxi_state = self.oxi_states[vac_symbol]
            as_symbol = get_el_sp(sub_specie).symbol
            as_oxi_state = self.oxi_states[as_symbol]
            expected_oxi = as_oxi_state - vac_oxi_state
            if expected_oxi == 0:
                return [-1, 0, 1]
            else:
                minval = min(expected_oxi, 0)
                maxval = max(expected_oxi, 0)
                return [c for c in range(minval - 1, maxval + 2)]

        elif defect_type == "interstitial":
            return [-1, 0, 1]


class DefectChargerUserCustom(DefectCharger):
    """
        NOTE from developers:
            This code is for full control over charging approach of defects
            but will be replaced with a future version.
            Code has no unit test and will not be maintained
            going forward (as of 12/15/2017).
            However, we are keeping function here to allow for
            current users to make use of it...

    Determine oxidation states from bond valence method
    (unless oxidation states specified)
    Then ask user what charges they want
    """

    def __init__(self, structure, oxi_states=None):
        """
        Does initial Valence bond method to initialize oxidation states for user help
        Overwridden by oxi_state specified by
        Args: structure
            pymatgen structure object to determine the oxidation states
            min_max_oxi: any user specified min/max oxidation ranges for elements in structure
            oxi_states: any user specified oxidation states of elements in structure
        """
        oxi_states = oxi_states if oxi_states is not None else {}
        struct_species = structure.types_of_specie
        if (len(struct_species) == 1) and struct_species[
            0
        ].symbol not in oxi_states.keys():
            oxi_states[struct_species[0].symbol] = 0
        else:
            vir = VIRE(structure)
            for elt, oxi in vir.valences.items():
                strip_key = "".join([s for s in elt if s.isalpha()])
                if strip_key not in oxi_states.keys():
                    oxi_states[strip_key] = oxi
        self.oxi_states = oxi_states
        print(
            "\nThis is Full-User Charge Generation Mode.\n"
            "Options are: (1) Range mode (input min and max of each each type of defect) "
            "or (2) Individual mode (input each charge state you want for each defect)\n"
            "\nWhen finished with specific defect, press ENTER to continue."
        )
        rng_mod = raw_input("Please specify Range (R) or Individual (I) Mode:")
        if "R" == rng_mod.upper()[0]:
            self.rangemode = True
        else:
            self.rangemode = False

    def get_charges(self, defect_type, site_specie=None, sub_specie=None):
        """
        Based on the type of defect, site and substitution (if any) species
        the defect charge states are generated.
        Args:
            defect_type (str): Options are vacancy, antisite, substitution,
                               and interstitial
            site_specie: Specie on the host lattice site
                         For interstitials, use this
            sub_specie: Specie that is replacing the site specie.
                        For antisites and substitution defects
        """
        if site_specie in self.oxi_states.keys():
            sitechg = self.oxi_states[site_specie]
        else:
            sitechg = False

        if sub_specie:
            if sub_specie in self.oxi_states.keys():
                subchg = self.oxi_states[sub_specie]
            else:
                subchg = False

        def get_users_charges():
            tmpchgs = raw_input("What charges would you like? : ")
            chgs = [int(c) for c in tmpchgs.split()]
            if self.rangemode:
                return list(range(chgs[0], chgs[-1] + 1))
            else:
                return list(chgs)

        if defect_type == "vacancy":
            if not sitechg:
                print(
                    site_specie,
                    defect_type,
                    "charge suggestion unknown (specify oxidation states to get suggestion)",
                )
            else:
                print(
                    site_specie,
                    defect_type,
                    "has charge =",
                    -sitechg,
                    "according to Simple Ionic Theory",
                )
        elif defect_type in ["antisite", "substitution"]:
            nom = sub_specie + "_on_" + site_specie
            if not sitechg or not subchg:
                print(
                    nom,
                    defect_type,
                    "charge suggestion unknown (specify oxidation states to get suggestion)",
                )
            else:
                print(
                    nom,
                    defect_type,
                    "has charge = ",
                    subchg - sitechg,
                    "according to Simple Ionic Theory",
                )
        elif defect_type == "interstitial":
            if not sitechg:
                print(
                    site_specie,
                    defect_type,
                    "charge suggestion unknown (specify oxidation states to get suggestion)",
                )
            else:
                print(
                    site_specie,
                    defect_type,
                    "has charge = ",
                    sitechg,
                    "according to Simple Ionic Theory",
                )
        outchgs = get_users_charges()
        print("    Charges generated:", outchgs)
        return outchgs


class ChargedDefectsStructures(object):
    """
    A class to generate charged defective structures for use in first
    principles supercell formalism. The standard defects such as antisites
    and vacancies are generated.  Interstitial finding is also implemented
    (optional).
    """

    def __init__(
        self,
        structure,
        max_min_oxi=None,
        substitutions=None,
        oxi_states=None,
        cellmax=128,
        antisites_flag=True,
        include_interstitials=True,
        interstitial_elements=None,
        intersites=None,
        standardized=False,
        struct_type="semiconductor",
    ):
        """
        Args:
            structure (Structure):
                the bulk structure.
            max_min_oxi (dict):
                The minimal and maximum oxidation state of each element as a
                dict. For instance {"O":(-2,0)}. If not given, the oxi-states
                of pymatgen are considered.
            substitutions (dict):
                The allowed substitutions of elements as a dict. If not given,
                intrinsic defects are computed. If given, intrinsic (e.g.,
                anti-sites) and extrinsic are considered explicitly specified.
                Example: {"Co":["Zn","Mn"]} means Co sites can be substituted
                by Mn or Zn.
            oxi_states (dict):
                The oxidation state of the elements in the compound e.g.
                {"Fe":2,"O":-2}. If not given, the oxidation state of each
                site is computed with bond valence sum. WARNING: Bond-valence
                method can fail for mixed-valence compounds.
            cellmax (int):
                Maximum number of atoms allowed in the supercell.
            antisites_flag (bool):
                If False, don't generate antisites.
            include_interstitials (bool):
                If true, do generate interstitial defect configurations
                (default: False).
            interstitial_elements ([str]):
                List of strings containing symbols of the elements that are
                to be considered for interstitial sites.  The default (None)
                triggers self-interstitial generation,
                given that include_interstitials is True.
            intersites ([PeriodicSite]):
                A list of PeriodicSites in the bulk structure on which we put
                interstitials.  Note that you still have to set flag
                include_interstitials to True in order to make use of this
                manual way of providing interstitial sites.
                If this is used, then no additional interstitials are generated
                beyond the list that is provided in intersites.
            standardized (bool):
                If True, use the primitive standard structure as unit cell
                for generating the defect configurations (default is False).
                The primitive standard structure is obtained from the
                SpacegroupAnalyzer class with a symprec of 0.01.
            struct_type (string):
                Options are 'semiconductor' and 'insulator'. If semiconductor
                is selected, charge states based on database of semiconductors
                is used to assign defect charges. For insulators, defect
                charges are conservatively assigned.
        """
        max_min_oxi = max_min_oxi if max_min_oxi is not None else {}
        substitutions = substitutions if substitutions is not None else {}
        oxi_states = oxi_states if oxi_states is not None else {}
        interstitial_elements = (
            interstitial_elements if interstitial_elements is not None else []
        )
        intersites = intersites if intersites is not None else []

        self.defects = []
        self.cellmax = cellmax
        self.substitutions = {}
        self.struct_type = struct_type
        for key, val in substitutions.items():
            self.substitutions[key] = val

        spa = SpacegroupAnalyzer(structure, symprec=1e-2)
        prim_struct = spa.get_primitive_standard_structure()
        if standardized:
            self.struct = prim_struct
        else:
            self.struct = structure

        struct_species = self.struct.types_of_specie
        if self.struct_type == "semiconductor":
            self.defect_charger = DefectChargerSemiconductor(
                self.struct, min_max_oxi=max_min_oxi
            )
        elif self.struct_type == "insulator":
            self.defect_charger = DefectChargerInsulator(self.struct)
        elif self.struct_type == "manual":
            self.defect_charger = DefectChargerUserCustom(
                self.struct, oxi_states=oxi_states
            )
        elif self.struct_type == "ionic":
            self.defect_charger = DefectChargerIonic(self.struct)
        else:
            raise NotImplementedError

        if include_interstitials and interstitial_elements:
            for elem_str in interstitial_elements:
                if not Element.is_valid_symbol(elem_str):
                    raise ValueError(f"invalid interstitial element {elem_str}")

        sc_scale = get_optimized_sc_scale(self.struct, cellmax)
        self.defects = {}
        sc = self.struct.copy()
        sc.make_supercell(sc_scale)
        self.defects["bulk"] = {
            "name": "bulk",
            "supercell": {"size": sc_scale, "structure": sc},
        }

        # If interstitials are provided as a list of PeriodicSites,
        # make sure that the lattice has not changed.
        if include_interstitials and intersites:
            for intersite in intersites:  # list of PeriodicSite objects
                if intersite.lattice != self.struct.lattice:
                    raise RuntimeError(
                        "Discrepancy between lattices"
                        " underlying the input interstitials and"
                        " the bulk structure; possibly because of"
                        " standardizing the input structure."
                    )

        vacancies = []
        as_defs = []
        sub_defs = []

        VG = VacancyGenerator(self.struct)
        print("Setting up vacancies")
        for i, vac in enumerate(VG):
            vac_site = vac.site
            vac_symbol = vac.site.specie.symbol
            vac_sc = vac.generate_defect_structure(sc_scale)

            # create a trivial defect structure to find where supercell transformation moves the lattice
            struct_for_defect_site = Structure(
                vac.bulk_structure.copy().lattice,
                [vac.site.specie],
                [vac.site.frac_coords],
                to_unit_cell=True,
                coords_are_cartesian=False,
            )
            struct_for_defect_site.make_supercell(sc_scale)
            vac_sc_site = struct_for_defect_site[0]

            charges_vac = self.defect_charger.get_charges("vacancy", vac_symbol)

            for c in SimpleChargeGenerator(vac):
                vacancies.append(
                    {
                        "name": f"vac_{i+1}_{vac_symbol}",
                        "unique_site": vac_site,
                        "bulk_supercell_site": vac_sc_site,
                        "defect_type": "vacancy",
                        "site_specie": vac_symbol,
                        "site_multiplicity": vac.multiplicity,
                        "supercell": {"size": sc_scale, "structure": vac_sc},
                        "charges": charges_vac,
                        "Possible_KV_Charge": c.charge,
                    }
                )

        if antisites_flag:
            print("Setting up antisites")
            for as_specie in set(struct_species):
                SG = SubstitutionGenerator(self.struct, as_specie)
                for i, sub in enumerate(SG):
                    as_symbol = as_specie.symbol
                    as_sc = sub.generate_defect_structure(sc_scale)

                    # create a trivial defect structure to find where supercell transformation
                    # moves the defect
                    struct_for_defect_site = Structure(
                        sub.bulk_structure.copy().lattice,
                        [sub.site.specie],
                        [sub.site.frac_coords],
                        to_unit_cell=True,
                        coords_are_cartesian=False,
                    )
                    struct_for_defect_site.make_supercell(sc_scale)
                    as_sc_site = struct_for_defect_site[0]

                    # get bulk_site (non sc)
                    poss_deflist = sorted(
                        sub.bulk_structure.get_sites_in_sphere(
                            sub.site.coords, 0.01, include_index=True
                        ),
                        key=lambda x: x[1],
                    )
                    if not len(poss_deflist):
                        raise ValueError(
                            f"Could not find substitution site inside bulk structure for {sub.name}?"
                        )
                    defindex = poss_deflist[0][2]
                    as_site = sub.bulk_structure[defindex]
                    vac_symbol = as_site.specie

                    charges_as = self.defect_charger.get_charges(
                        "antisite", vac_symbol, as_symbol
                    )

                    for c in SimpleChargeGenerator(sub):
                        as_defs.append(
                            {
                                "name": f"as_{i+1}_{as_symbol}_on_{vac_symbol}",
                                "unique_site": as_site,
                                "bulk_supercell_site": as_sc_site,
                                "defect_type": "antisite",
                                "site_specie": vac_symbol,
                                "substituting_specie": as_symbol,
                                "site_multiplicity": sub.multiplicity,
                                "supercell": {"size": sc_scale, "structure": as_sc},
                                "charges": charges_as,
                                "Possible_KV_Charge": c.charge,
                            }
                        )

        for vac_symbol, subspecie_list in self.substitutions.items():
            for subspecie_symbol in subspecie_list:
                SG = SubstitutionGenerator(self.struct, subspecie_symbol)
                for i, sub in enumerate(SG):
                    sub_symbol = sub.site.specie.symbol

                    # get bulk_site (non sc)
                    poss_deflist = sorted(
                        sub.bulk_structure.get_sites_in_sphere(
                            sub.site.coords, 0.1, include_index=True
                        ),
                        key=lambda x: x[1],
                    )
                    if not len(poss_deflist):
                        raise ValueError(
                            f"Could not find substitution site inside bulk structure for {sub.name}?"
                        )
                    defindex = poss_deflist[0][2]
                    sub_site = self.struct[defindex]
                    this_vac_symbol = sub_site.specie.symbol

                    if (sub_symbol != subspecie_symbol) or (
                        this_vac_symbol != vac_symbol
                    ):
                        continue
                    else:
                        sub_sc = sub.generate_defect_structure(sc_scale)

                        # create a trivial defect structure to find where supercell
                        # transformation moves the defect
                        struct_for_defect_site = Structure(
                            sub.bulk_structure.copy().lattice,
                            [sub.site.specie],
                            [sub.site.frac_coords],
                            to_unit_cell=True,
                            coords_are_cartesian=False,
                        )
                        struct_for_defect_site.make_supercell(sc_scale)
                        sub_sc_site = struct_for_defect_site[0]

                        charges_sub = self.defect_charger.get_charges(
                            "substitution", vac_symbol, subspecie_symbol
                        )
                        for c in SimpleChargeGenerator(sub):
                            sub_defs.append(
                                {
                                    "name": f"sub_{i+1}_{subspecie_symbol}_on_{vac_symbol}",
                                    "unique_site": sub_site,
                                    "bulk_supercell_site": sub_sc_site,
                                    "defect_type": "substitution",
                                    "site_specie": vac_symbol,
                                    "substitution_specie": subspecie_symbol,
                                    "site_multiplicity": sub.multiplicity,
                                    "supercell": {
                                        "size": sc_scale,
                                        "structure": sub_sc,
                                    },
                                    "charges": charges_sub,
                                    "Possible_KV_Charge": c.charge,
                                }
                            )

        self.defects["vacancies"] = vacancies
        self.defects["substitutions"] = sub_defs
        self.defects["substitutions"] += as_defs

        if include_interstitials:
            interstitials = []

            if interstitial_elements:
                inter_elems = interstitial_elements
            else:
                inter_elems = [elem.symbol for elem in self.struct.composition.elements]
            if len(inter_elems) == 0:
                raise RuntimeError("empty element list for interstitials")

            if intersites:
                print("Setting up interstitials from intersites")
                # manual specification of interstitials
                for i, intersite in enumerate(intersites):
                    elt = intersite.specie
                    name = f"inter_{i+1}_{elt}"

                    if intersite.lattice != self.struct.lattice:
                        err_msg = "Lattice matching error occurs between provided interstitial and the bulk structure."
                        if standardized:
                            err_msg += (
                                "\nLikely because the standardized flag was used. Turn this flag off or reset "
                                "your interstitial PeriodicSite to match the standardized form of the bulk structure."
                            )
                        raise ValueError(err_msg)
                    else:
                        intersite_object = Interstitial(self.struct, intersite)

                    # create a trivial defect structure to find where supercell transformation
                    # moves the defect site
                    struct_for_defect_site = Structure(
                        intersite_object.bulk_structure.copy().lattice,
                        [intersite_object.site.specie],
                        [intersite_object.site.frac_coords],
                        to_unit_cell=True,
                        coords_are_cartesian=False,
                    )
                    struct_for_defect_site.make_supercell(sc_scale)
                    site_sc = struct_for_defect_site[0]

                    sc_with_inter = intersite_object.generate_defect_structure(sc_scale)
                    charges_inter = self.defect_charger.get_charges("interstitial", elt)

                    for c in SimpleChargeGenerator(intersite_object):
                        interstitials.append(
                            {
                                "name": name,
                                "unique_site": intersite_object.site,
                                "bulk_supercell_site": site_sc,
                                "defect_type": "interstitial",
                                "site_specie": intersite_object.site.specie.symbol,
                                "site_multiplicity": intersite_object.multiplicity,
                                "supercell": {
                                    "size": sc_scale,
                                    "structure": sc_with_inter,
                                },
                                "charges": charges_inter,
                                "Possible_KV_Charge": c.charge,
                            }
                        )

            else:
                print(
                    "Searching for Voronoi interstitial sites (this can take a little while)"
                )
                # the use of O here is completely arbitrary
                IG = VoronoiInterstitialGenerator(self.struct, "O")
                sites = []
                for i in IG:
                    sites.append(i)

                inters = []
                for el in inter_elems:
                    for s in sites:
                        d = PeriodicSite(
                            el, s.site.frac_coords, s.bulk_structure.copy().lattice
                        )
                        inters.append(d)

                print("Found the interstital sites, setting up interstitials")
                for i, intersite in enumerate(inters):
                    elt = intersite.specie
                    if intersite.lattice != self.struct.lattice:
                        err_msg = (
                            "Lattice matching error occurs between provided interstitial "
                            "and the bulk structure."
                        )
                        raise ValueError(err_msg)
                    else:
                        intersite_object = Interstitial(self.struct, intersite)

                    name = f"inter_{i+1}_{elt}"

                    # create a trivial defect structure to find where supercell transformation
                    # moves the defect site
                    struct_for_defect_site = Structure(
                        intersite_object.bulk_structure.copy().lattice,
                        [intersite_object.site.specie],
                        [intersite_object.site.frac_coords],
                        to_unit_cell=True,
                        coords_are_cartesian=False,
                    )
                    struct_for_defect_site.make_supercell(sc_scale)
                    site_sc = struct_for_defect_site[0]

                    sc_with_inter = intersite_object.generate_defect_structure(sc_scale)
                    charges_inter = self.defect_charger.get_charges("interstitial", elt)

                    for c in SimpleChargeGenerator(intersite_object):
                        interstitials.append(
                            {
                                "name": name,
                                "unique_site": intersite_object.site,
                                "bulk_supercell_site": site_sc,
                                "defect_type": "interstitial",
                                "site_specie": intersite_object.site.specie.symbol,
                                "site_multiplicity": intersite_object.multiplicity,
                                "supercell": {
                                    "size": sc_scale,
                                    "structure": sc_with_inter,
                                },
                                "charges": charges_inter,
                                "Possible_KV_Charge": c.charge,
                            }
                        )

            self.defects["interstitials"] = interstitials

        print("Defects generated:")
        for key, defect_list in self.defects.items():
            if key != "bulk" and len(defect_list) > 0:
                table = []
                for single_defect_dict in defect_list:
                    header = [
                        key.capitalize(),
                        "Potential Charge States",
                        "Supercell Site " "Multiplicity",
                    ]
                    row = [
                        single_defect_dict["name"],
                        single_defect_dict["charges"],
                        single_defect_dict["site_multiplicity"],
                    ]
                    table.append(row)
                print(
                    tabulate(
                        table,
                        headers=header,
                        stralign="left",
                        numalign="left",
                    ),
                    "\n",
                )

    def to(self, outfile):
        dumpfn(self.defects, outfile)

    def get_n_defects_of_type(self, defect_type):
        """
        Get the number of defects of the given type.
        Args:
            defect_type (str): defect type (vacancies,
                interstitials, substitutions).
        Returns:
            n_defects (int): number of defects of given type.
        """
        try:
            return len(self.defects[defect_type])
        except:
            return 0

    def get_ith_supercell_of_defect_type(self, i, defect_type):
        """
        Get the (i-1)-th defect supercell with the given defect type.
        Args:
            i (int): index (starting from 0) of target supercell.
            defect_type (str): defect type (vacancies,
                interstitials, substitutions).
        Returns:
            sc (Structure): copy of the defect supercell.
        """
        return self.defects[defect_type][i]["supercell"]["structure"].copy()
