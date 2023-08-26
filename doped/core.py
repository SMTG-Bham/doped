"""
Core functions and classes for defects in doped.
"""

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core, thermo
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

# TODO: Need to set the str and repr functions for these to give an informative output! Same for our
#  parsing functions/classes


@dataclass
class DefectEntry(thermo.DefectEntry):
    """
    Subclass of pymatgen.analysis.defects.thermo.DefectEntry with additional
    attributes used by doped.

    Core Attributes:
        defect:
            doped/pymatgen defect object corresponding to the defect in the entry.
        charge_state:
            Charge state of the defect.
        sc_entry:
            pymatgen ComputedStructureEntry for the _defect_ supercell.
        sc_defect_frac_coords:
            The fractional coordinates of the defect in the supercell.
            If None, structures attributes of the LOCPOT file are used to
            automatically determine the defect location.
        bulk_entry:
            pymatgen ComputedEntry for the bulk supercell reference. Required
            for calculating the defect formation energy.
        corrections:
            A dictionary of energy corrections which are summed and added to
            the defect formation energy.
        corrections_metadata:
            A dictionary that acts as a generic container for storing information
            about how the corrections were calculated. Only used for debugging
            and plotting purposes.

    Parsing Attributes:
        calculation_metadata:
            A dictionary of calculation parameters and data, used to perform
            charge corrections and compute formation energies.

    Generation Attributes:
        name:
            The doped-generated name of the defect entry. See docstrings of
            DefectsGenerator for the doped naming algorithm.
        conventional_structure:
            Conventional cell structure of the host according to the Bilbao
            Crystallographic Server (BCS) definition, used to determine defect
            site Wyckoff labels and multiplicities.
        conv_cell_frac_coords:
            Fractional coordinates of the defect in the conventional cell.
        equiv_conv_cell_frac_coords:
            Symmetry-equivalent defect positions in fractional coordinates of
            the conventional cell.
        _BilbaoCS_conv_cell_vector_mapping:
            A vector mapping the lattice vectors of the spglib-defined
            conventional cell to that of the Bilbao Crystallographic Server
            definition (for most space groups the definitions are the same).
        wyckoff:
            Wyckoff label of the defect site.
        charge_state_guessing_log:
            A log of the input & computed values used to determine charge state
            probabilities.
        defect_supercell:
            pymatgen Structure object of the defect supercell.
        defect_supercell_site:
            pymatgen PeriodicSite object of the defect in the defect supercell.
        equivalent_supercell_sites:
            List of pymatgen PeriodicSite objects of symmetry-equivalent defect
            sites in the defect supercell.
        bulk_supercell:
            pymatgen Structure object of the bulk (pristine, defect-free) supercell.
    """

    # core attributes:
    defect: "Defect"
    charge_state: int
    sc_entry: ComputedStructureEntry
    corrections: Dict[str, float] = field(default_factory=dict)
    corrections_metadata: Dict[str, Any] = field(default_factory=dict)
    sc_defect_frac_coords: Optional[Tuple[float, float, float]] = None
    bulk_entry: Optional[ComputedStructureEntry] = None
    entry_id: Optional[str] = None

    # doped attributes:
    name: str = ""
    calculation_metadata: Dict = field(default_factory=dict)
    conventional_structure: Optional[Structure] = None
    conv_cell_frac_coords: Optional[np.ndarray] = None
    equiv_conv_cell_frac_coords: List[np.ndarray] = field(default_factory=list)
    _BilbaoCS_conv_cell_vector_mapping: List[int] = field(default_factory=lambda: [0, 1, 2])
    wyckoff: Optional[str] = None
    charge_state_guessing_log: Dict = field(default_factory=dict)
    defect_supercell: Optional[Structure] = None
    defect_supercell_site: Optional[PeriodicSite] = None  # TODO: Should be able to refactor SnB to use
    # this, in the from_structures approach, and just show general doped workflow on the docs and
    # from_structures, and mention can also do other input options. Also add `from_structures` method to
    # doped DefectEntry??
    equivalent_supercell_sites: List[PeriodicSite] = field(default_factory=list)
    bulk_supercell: Optional[Structure] = None

    def __post_init__(self):
        """
        Post-initialization method, using super() and self.defect.
        """
        super().__post_init__()
        self.name: str = self.defect.name if not self.name else self.name

    @property  # bug fix for now, will PR and delete later
    def corrected_energy(self) -> float:
        """
        The energy of the defect entry with _all_ corrections applied.
        """
        return self.sc_entry.energy + sum(self.corrections.values())

    def to_json(self, filename: Optional[str] = None):
        """
        Save the DefectEntry object to a json file, which can be reloaded with
        the DefectEntry.from_json() class method.

        Args:
            filename (str):
                Filename to save json file as. If None, the filename will
                be set as "{DefectEntry.name}.json".
        """
        if filename is None:
            filename = f"{self.name}.json"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: str):
        """
        Load a DefectEntry object from a json file.

        Args:
            filename (str):
                Filename of json file to load DefectEntry from.

        Returns:
            DefectEntry object
        """
        return loadfn(filename)


class Defect(core.Defect):
    """
    Doped Defect object.
    """

    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: Optional[int] = None,
        oxi_state: Optional[float] = None,
        equivalent_sites: Optional[List[PeriodicSite]] = None,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
        user_charges: Optional[List[int]] = None,
        **doped_kwargs,
    ):
        """
        Subclass of pymatgen.analysis.defects.core.Defect with additional
        attributes and methods used by doped.

        Args:
            structure:
                The structure in which to create the defect. Typically
                the primitive structure of the host crystal.
            site: The defect site in the structure.
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
            equivalent_sites:
                A list of equivalent sites for the defect in the structure.
            symprec: Tolerance for symmetry finding.
            angle_tolerance: Angle tolerance for symmetry finding.
            user_charges:
                User specified charge states. If specified, ``get_charge_states``
                will return this list. If ``None`` or empty list the charge
                states will be determined automatically.
            **doped_kwargs:
                Additional keyword arguments to define doped-specific attributes
                (listed below), in the form ``doped_attribute_name=value``.
                (e.g. ``wyckoff = "4a"``).
        """
        super().__init__(
            structure=structure,
            site=site.to_unit_cell(),  # ensure mapped to unit cell
            multiplicity=multiplicity,
            oxi_state=oxi_state,
            equivalent_sites=[site.to_unit_cell() for site in equivalent_sites]
            if equivalent_sites is not None
            else None,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            user_charges=user_charges,
        )  # core attributes

        self.conventional_structure: Optional[Structure] = doped_kwargs.get("conventional_structure", None)
        self.conv_cell_frac_coords: Optional[np.ndarray] = doped_kwargs.get("conv_cell_frac_coords", None)
        self.equiv_conv_cell_frac_coords: List[np.ndarray] = doped_kwargs.get(
            "equiv_conv_cell_frac_coords", []
        )
        self._BilbaoCS_conv_cell_vector_mapping: List[int] = doped_kwargs.get(
            "_BilbaoCS_conv_cell_vector_mapping", [0, 1, 2]
        )
        self.wyckoff: Optional[str] = doped_kwargs.get("wyckoff", None)

    @classmethod
    def _from_pmg_defect(cls, defect: core.Defect, **doped_kwargs) -> "Defect":
        """
        Create a doped Defect from a pymatgen Defect object.

        Args:
            defect:
                pymatgen Defect object.
            **doped_kwargs:
                Additional keyword arguments to define doped-specific attributes
                (see class docstring).
        """
        # get doped kwargs from defect attributes, if defined:
        for doped_attr in [
            "conventional_structure",
            "conv_cell_frac_coords",
            "equiv_conv_cell_frac_coords",
            "_BilbaoCS_conv_cell_vector_mapping",
            "wyckoff",
        ]:
            if (
                hasattr(defect, doped_attr)
                and getattr(defect, doped_attr) is not None
                and doped_attr not in doped_kwargs
            ):
                doped_kwargs[doped_attr] = getattr(defect, doped_attr)

        return cls(
            structure=defect.structure,
            site=defect.site.to_unit_cell(),  # ensure mapped to unit cell
            multiplicity=defect.multiplicity,
            oxi_state=defect.oxi_state,
            equivalent_sites=[site.to_unit_cell() for site in defect.equivalent_sites]
            if defect.equivalent_sites is not None
            else None,
            symprec=defect.symprec,
            angle_tolerance=defect.angle_tolerance,
            user_charges=defect.user_charges,
            **doped_kwargs,
        )

    def get_supercell_structure(
        self,
        sc_mat: Optional[np.ndarray] = None,
        min_atoms: int = 50,  # different to current pymatgen default (80)
        max_atoms: int = 500,  # different to current pymatgen default (240)
        min_length: float = 10.0,  # same as current pymatgen default
        force_diagonal: bool = False,  # same as current pymatgen default
        dummy_species: Optional[str] = None,
        target_frac_coords: Optional[np.ndarray] = None,
        return_sites: bool = False,
    ) -> Structure:
        """
        Generate the supercell for a defect.

        Redefined from the parent class to allow the use of target_frac_coords
        to place the defect at the closest equivalent site to the target
        fractional coordinates in the supercell, while keeping the supercell
        fixed (to avoid any issues with defect parsing).
        Also returns information about equivalent defect sites in the supercell.

        Args:
            sc_mat:
                Transformation matrix of self.structure to create the supercell.
                If None, then automatically determined by `CubicSupercellAnalyzer`.
            target_frac_coords:
                If set, the defect will be placed at the closest equivalent site to
                these fractional coordinates (using self.equivalent_sites).
            return_sites:
                If True, returns a tuple of the defect supercell, defect supercell
                site and list of equivalent supercell sites.
            dummy_species:
                Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms:
                Maximum number of atoms allowed in the generated supercell (if sc_mat is None).
            min_atoms:
                Minimum number of atoms allowed in the generated supercell (if sc_mat is None).
            min_length:
                Minimum length of the generated supercell lattice vectors (if sc_mat is None).
            force_diagonal:
                If True, generate a supercell with a diagonal transformation matrix
                (if sc_mat is None).

        Returns:
            The defect supercell structure. If return_sites is True, also returns
            the defect supercell site and list of equivalent supercell sites.
        """
        if sc_mat is None:
            sc_mat = get_sc_fromstruct(
                self.structure,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )

        sites = self.equivalent_sites or [self.site]
        structure_w_all_defect_sites = Structure.from_sites(
            [PeriodicSite("X", site.frac_coords, self.structure.lattice) for site in sites]
        )
        sc_structure_w_all_defect_sites = structure_w_all_defect_sites * sc_mat
        equiv_sites = [
            PeriodicSite(self.site.specie, sc_x_site.frac_coords, sc_x_site.lattice).to_unit_cell()
            for sc_x_site in sc_structure_w_all_defect_sites
        ]

        if target_frac_coords is None:
            sc_structure = self.structure * sc_mat
            sc_mat_inv = np.linalg.inv(sc_mat)
            sc_pos = np.dot(self.site.frac_coords, sc_mat_inv)
            sc_site = PeriodicSite(self.site.specie, sc_pos, sc_structure.lattice).to_unit_cell()

        else:
            # sort by distance from target_frac_coords, then by magnitude of fractional coordinates:
            sc_site = sorted(
                equiv_sites,
                key=lambda site: (
                    round(
                        np.linalg.norm(site.frac_coords - np.array(target_frac_coords)),
                        4,
                    ),
                    round(np.linalg.norm(site.frac_coords), 4),
                    round(np.abs(site.frac_coords[0]), 4),
                    round(np.abs(site.frac_coords[1]), 4),
                    round(np.abs(site.frac_coords[2]), 4),
                ),
            )[0]

        sc_defect = self.__class__(
            structure=self.structure * sc_mat, site=sc_site, oxi_state=self.oxi_state
        )
        sc_defect_struct = sc_defect.defect_structure
        sc_defect_struct.remove_oxidation_states()

        # also remove oxidation states from sites:
        def _remove_site_oxi_state(site):
            """
            Remove site oxidation state in-place.

            Same method as Structure.remove_oxidation_states().
            """
            new_sp: Dict[Element, float] = collections.defaultdict(float)
            for el, occu in site.species.items():
                sym = el.symbol
                new_sp[Element(sym)] += occu
            site.species = Composition(new_sp)

        _remove_site_oxi_state(sc_site)
        for site in equiv_sites:
            _remove_site_oxi_state(site)

        if dummy_species is not None:
            sc_defect_struct.insert(len(self.structure * sc_mat), dummy_species, sc_site.frac_coords)

        return (
            (
                sc_defect_struct,
                sc_site,
                equiv_sites,
            )
            if return_sites
            else sc_defect_struct
        )

    def as_dict(self):
        """
        JSON-serializable dict representation of Defect.

        Needs to be redefined because attributes not explicitly specified in
        subclasses, which is required for monty functions.
        """
        return {"@module": type(self).__module__, "@class": type(self).__name__, **self.__dict__}

    def to_json(self, filename: Optional[str] = None):
        """
        Save the Defect object to a json file, which can be reloaded with the
        Defect.from_json() class method.

        Args:
            filename (str):
                Filename to save json file as. If None, the filename will
                be set as "{Defect.name}.json".
        """
        if filename is None:
            filename = f"{self.name}.json"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: str):
        """
        Load a Defect object from a json file.

        Args:
            filename (str):
                Filename of json file to load Defect from.

        Returns:
            Defect object
        """
        return loadfn(filename)


class Vacancy(core.Vacancy, Defect):
    def __init__(self, *args, **kwargs):
        """
        Subclass of pymatgen.analysis.defects.core.Vacancy with additional
        attributes and methods used by doped.
        """
        super().__init__(*args, **kwargs)


class Substitution(core.Substitution, Defect):
    def __init__(self, *args, **kwargs):
        """
        Subclass of pymatgen.analysis.defects.core.Substitution with additional
        attributes and methods used by doped.
        """
        super().__init__(*args, **kwargs)


class Interstitial(core.Interstitial, Defect):
    def __init__(self, *args, **kwargs):
        """
        Subclass of pymatgen.analysis.defects.core.Interstitial with additional
        attributes and methods used by doped.
        """
        super().__init__(*args, **kwargs)
