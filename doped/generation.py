"""
Code to generate Defect objects and supercell structures for ab-initio calculations.
"""
import warnings
from tabulate import tabulate
from typing import Optional, List, Dict, Union
from itertools import chain
from tqdm import tqdm

from monty.json import MontyDecoder
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.analysis.defects.generators import (
    VacancyGenerator,
    AntiSiteGenerator,
    SubstitutionGenerator,
    InterstitialGenerator,
    VoronoiInterstitialGenerator,
)
from shakenbreak.input import _get_defect_name_from_obj, _update_defect_dict


# TODO: Should have option to provide the bulk supercell to use, and generate from this, in case ppl want to directly
#  compare to calculations with the same supercell before etc
# TODO: For specifying interstitial sites, will want to be able to specify as either primitive or supercell coords in
#  this case, so will need functions for transforming between primitive and supercell defect structures (will want this
#  for defect parsing as well). Defectivator has functions that do some of this. This will be tricky (SSX trickay you
#  might say) for relaxed interstitials -> get symmetry-equivalent positions of relaxed interstitial position in
#  unrelaxed bulk (easy pal, tf you mean 'tricky'??)

# Only include interstitials flag if interstitial generation is still slow, but doubt it?


def get_defect_entry_from_defect(
    defect: Defect,
    defect_supercell: Structure,
    charge_state: int,
    dummy_species: DummySpecies = DummySpecies("X"),
):
    """
    Generate DefectEntry object from a Defect object.
    This is used to describe a Defect with a specified simulation cell.
    """
    defect_entry_structure = (
        defect_supercell.copy()
    )  # duplicate the structure so we don't edit the input Structure
    # Dummy species (used to keep track of the defect coords in the supercell)
    # Find its fractional coordinates & remove it from the supercell
    dummy_site = [
        site
        for site in defect_entry_structure
        if site.species.elements[0].symbol == dummy_species.symbol
    ][0]
    sc_defect_frac_coords = dummy_site.frac_coords
    defect_entry_structure.remove(dummy_site)

    computed_structure_entry = ComputedStructureEntry(
        structure=defect_entry_structure,
        energy=0.0,  # needs to be set, so set to 0.0
    )
    return DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=computed_structure_entry,
        sc_defect_frac_coords=sc_defect_frac_coords,
    )


class DefectsGenerator:
    def __init__(
        self,
        structure: Structure,
        extrinsic: Union[List, Dict] = {},
        interstitial_coords: List = [],
        **kwargs,
    ):
        """
        Generates pymatgen DefectEntry objects for defects in the input host structure.
        By default, generates all intrinsic defects, but extrinsic defects (impurities)
        can also be created using the `extrinsic` argument.

        Interstitial sites are generated using Voronoi tesselation by default (found
        to be the most reliable), however these can also be manually specified using
        the `interstitial_coords` argument.

        Supercells are generated for each defect using the pymatgen `get_supercell_structure()`
        method, with `doped` default settings of `min_length = 10` (minimum supercell length
        of 10 Å) and `min_atoms = 50` (minimum 50 atoms in supercell). If a different supercell
        is desired, this can be controlled by specifying keyword arguments in DefectsGenerator(),
        which are passed to the `get_supercell_structure()` method.

        Generated defect entries are named (by setting the `DefectEntry.name` attribute) as
        "{DefectEntry.defect.name}_m{DefectEntry.defect.multiplicity}_{charge}" for interstitials
        and "{DefectEntry.defect.name}_s{DefectEntry.defect.defect_site_index}_{charge}" for
        vacancies and antisites/substitutions. The labels "a", "b", "c"... will be appended for
        defects with multiple inequivalent sites.
        # TODO: This 👆 is the current ShakeNBreak default, but will be updated!

        # TODO: Mention how charge states are generated, and how to modify, as shown in the example notebook.
        # Also show how to remove certain defects from the dictionary? Mightn't be worth the space for this though

        Args:
            structure (Structure):
                Structure of the host material (as a pymatgen Structure object).
                If this is not the primitive unit cell, it will be reduced to the
                primitive cell for defect generation, before supercell generation.
            extrinsic (Union[List, Dict]):
                List or dict of elements to be used for extrinsic defect generation
                (i.e. dopants/impurities). If a list is provided, all possible
                substitutional defects for each extrinsic element will be generated.
                If a dict is provided, the keys should be the host elements to be
                substituted, and the values the extrinsic element(s) to substitute in;
                as a string or list.
                In both cases, all possible extrinsic interstitials are generated.
            interstitial_coords (List):
                List of fractional coordinates (in the primitive cell) to use as
                interstitial defect sites. Default (when interstitial_coords not
                specified) is to automatically generate interstitial sites using
                Voronoi tesselation.
            **kwargs:
                Keyword arguments to be passed to the `get_supercell_structure()` method.
        """
        self.defects = {}  # {defect_type: [Defect, ...]}
        self.defect_entries = {}  # {defect_name: DefectEntry}
        pbar = tqdm(total=100)  # tqdm progress bar. 100% is completion
        pbar.set_description(f"Getting primitive structure")

        # Check if input structure is primitive. If not, reduce and warn
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        prim_struct = sga.get_primitive_standard_structure()
        if len(prim_struct) != len(structure):
            print(
                f"Input structure (N(atoms) = {len(structure)}) is not the primitive unit cell. "
                f"Reducing to primitive cell (N(atoms) = {len(prim_struct)}) for defect generation."
            )
        self.primitive_structure = prim_struct
        pbar.update(5)  # 5% of progress bar

        # Generate defects
        # Vacancies:
        pbar.set_description(f"Generating vacancies")
        vac_generator_obj = VacancyGenerator()
        vac_generator = vac_generator_obj.generate(self.primitive_structure)
        self.defects["vacancies"] = [vacancy for vacancy in vac_generator]
        pbar.update(10)  # 15% of progress bar

        # Antisites:
        pbar.set_description(f"Generating antisites")
        antisite_generator_obj = AntiSiteGenerator()
        as_generator = antisite_generator_obj.generate(self.primitive_structure)
        self.defects["antisites"] = [antisite for antisite in as_generator]
        pbar.update(10)  # 25% of progress bar

        # Substitutions:
        substitution_generator_obj = SubstitutionGenerator()
        if isinstance(extrinsic, list):  # substitute all host elements
            substitutions = {
                str(el): extrinsic
                for el in self.primitive_structure.composition.elements
            }
        elif isinstance(extrinsic, dict):  # substitute only specified host elements
            substitutions = extrinsic
        else:
            warnings.warn(
                f"Invalid `extrinsic` defect input. Got type {type(extrinsic)}, but list or dict required."
                f"No extrinsic defects will be generated."
            )
            substitutions = {}

        if substitutions:
            pbar.set_description(f"Generating substitutions")
            sub_generator = substitution_generator_obj.generate(
                self.primitive_structure, substitution=substitutions
            )
            self.defects["substitutions"] = [
                substitution for substitution in sub_generator
            ]
        pbar.update(10)  # 35% of progress bar

        # Interstitials:
        # determine which, if any, extrinsic elements are present:
        pbar.set_description(f"Generating interstitials")
        # previous generators add oxidation states, but messes with interstitial generators, so remove oxi states:
        self.primitive_structure.remove_oxidation_states()
        if isinstance(extrinsic, list):
            extrinsic_elements = extrinsic
        elif isinstance(
            extrinsic, dict
        ):  # dict of host: extrinsic elements, as lists or strings
            # convert to flattened list of extrinsic elements:
            extrinsic_elements = list(
                chain(*[i if isinstance(i, list) else [i] for i in extrinsic.values()])
            )
            extrinsic_elements = list(
                set(extrinsic_elements)
            )  # get only unique elements
        else:
            extrinsic_elements = []

        if interstitial_coords:
            # For the moment, this assumes interstitial_sites
            insertions = {
                str(el): interstitial_coords
                for el in self.primitive_structure.composition.elements
                + extrinsic_elements
            }
            interstitial_generator_obj = InterstitialGenerator()
            interstitial_generator = interstitial_generator_obj.generate(
                self.primitive_structure, insertions=insertions
            )
            self.defects["interstitials"] = [
                interstitial for interstitial in interstitial_generator
            ]

        else:
            # Generate interstitial sites using Voronoi tesselation
            voronoi_interstitial_generator_obj = VoronoiInterstitialGenerator()
            voronoi_interstitial_generator = (
                voronoi_interstitial_generator_obj.generate(
                    self.primitive_structure,
                    insert_species=[
                        str(el)
                        for el in self.primitive_structure.composition.elements
                        + extrinsic_elements
                    ],
                )
            )
            self.defects["interstitials"] = [
                interstitial for interstitial in voronoi_interstitial_generator
            ]

        pbar.update(
            30
        )  # 65% of progress bar, generating interstitials typically takes the longest

        # Generate supercell once, so this isn't redundantly rerun for each defect, and ensures the same supercell is
        # used for each defect and bulk calculation
        pbar.set_description(f"Generating simulation supercell")
        self.supercell_matrix = get_sc_fromstruct(
            self.primitive_structure,
            min_atoms=kwargs.get("min_atoms", 50),
            max_atoms=kwargs.get("max_atoms", 240),  # same as current pymatgen default
            min_length=kwargs.get("min_length", 10),  # same as current pymatgen default
            force_diagonal=kwargs.get(
                "force_diagonal", False
            ),  # same as current pymatgen default
        )
        pbar.update(20)  # 85% of progress bar

        # Generate DefectEntry objects:
        pbar.set_description(f"Generating DefectEntry objects")
        num_defects = sum([len(defect_list) for defect_list in self.defects.values()])
        for defect_type, defect_list in self.defects.items():
            defect_naming_dict = {}
            for defect in defect_list:
                defect_name_wout_charge = (
                    None  # determine for the first charge state only
                )
                defect_supercell = defect.get_supercell_structure(
                    sc_mat=self.supercell_matrix,
                    dummy_species="X",  # keep track of the defect frac coords in the supercell
                )
                # set defect charge states: currently from +/-1 to defect oxi state; TODO: Update!!
                if defect.oxi_state >= 0:
                    charge_states = [*range(-1, int(defect.oxi_state) + 1)]
                else:
                    charge_states = [*range(int(defect.oxi_state), 1 + 1)]
                for charge in charge_states:
                    # TODO: Will be updated to our chosen charge generation algorithm!
                    defect_entry = get_defect_entry_from_defect(
                        defect,
                        defect_supercell,
                        charge,
                        dummy_species=DummySpecies("X"),
                    )
                    if (
                        defect_name_wout_charge is None
                    ):  # determine for the first charge state only
                        defect_name_wout_charge = _get_defect_name_from_obj(defect)
                        defect_name_wout_charge = _update_defect_dict(
                            defect_entry, defect_name_wout_charge, defect_naming_dict
                        )
                    defect_name = defect_name_wout_charge + f"_{'+' if charge > 0 else ''}{charge}"
                    defect_entry.name = defect_name  # set name attribute
                    self.defect_entries[defect_name] = defect_entry
                pbar.update(
                    (1 / num_defects) * (pbar.total - pbar.n)
                )  # 100% of progress bar
        pbar.close()

        print("Defect Entries generated:")
        for defect_class, defect_list in self.defects.items():
            table = []
            header = [
                defect_class.capitalize(),
                "Charge States",
                "Unit Cell Coords",
                "Site Multiplicity (Unit Cell)",
            ]
            defect_type = defect_list[0].defect_type
            matching_defect_types = {
                defect_entry_name: defect_entry
                for defect_entry_name, defect_entry in self.defect_entries.items()
                if defect_entry.defect.defect_type == defect_type
            }
            matching_type_names_wout_charge = list(
                set(
                    [
                        defect_entry_name.rsplit("_", 1)[0]
                        for defect_entry_name in matching_defect_types
                    ]
                )
            )
            # sort to match order of appearance of elements in the primitive structure composition:
            sorted_defect_names = sorted(
                matching_type_names_wout_charge,
                key=lambda s: [
                    s.find(sub) if s.find(sub) != -1 else float("inf")
                    for sub in [
                        el.symbol for el in self.primitive_structure.composition.elements
                    ]
                ],
            )
            for defect_name in sorted_defect_names:
                charges = [
                    int(name.rsplit("_", 1)[1])
                    for name in self.defect_entries
                    if name.startswith(defect_name + "_")
                ]  # so e.g. Te_i_m1 doesn't match with Te_i_m1b
                neutral_defect_entry = self.defect_entries[defect_name + "_0"]  # neutral has no +/- sign
                frac_coords_string = ",".join(f"{x:.2f}" for x in neutral_defect_entry.defect.site.frac_coords)
                row = [defect_name, charges, f"[{frac_coords_string}]", neutral_defect_entry.defect.multiplicity]
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

    def as_dict(self):
        """
        JSON-serializable dict representation of DefectsGenerator
        """
        json_dict = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "defects": self.defects,
            "defect_entries": self.defect_entries,
            "primitive_structure": self.primitive_structure,
            "supercell_matrix": self.supercell_matrix,
        }

        return json_dict

    @classmethod
    def from_dict(cls, d):
        """
        Reconstructs DefectsGenerator object from a dict representation
        created using DefectsGenerator.as_dict().

        Args:
            d (dict): dict representation of DefectsGenerator.

        Returns:
            DefectsGenerator object
        """
        d_decoded = MontyDecoder().process_decoded(d)  # decode dict
        defects_generator = cls.__new__(cls)  # Create new DefectsGenerator object without invoking __init__

        # Manually set object attributes
        defects_generator.defects = d_decoded["defects"]
        defects_generator.defect_entries = d_decoded["defect_entries"]
        defects_generator.primitive_structure = d_decoded["primitive_structure"]
        defects_generator.supercell_matrix = d_decoded["supercell_matrix"]

        return defects_generator

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the defect_entries dictionary attribute,
        if the attribute doesn't exist in DefectsGenerator.
        """
        return getattr(self.defect_entries, attr)
