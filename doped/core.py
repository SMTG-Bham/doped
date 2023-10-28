"""
Core functions and classes for defects in doped.
"""


import collections
import contextlib
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core, supercells, thermo
from pymatgen.analysis.defects.utils import CorrectionResult
from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from scipy.stats import sem

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
        if not self.name:
            # try get using doped functions:
            try:
                from doped.generation import get_defect_name_from_defect

                name_wout_charge = get_defect_name_from_defect(self.defect)
            except Exception:
                name_wout_charge = self.defect.name

            self.name: str = (
                f"{name_wout_charge}_{'+' if self.charge_state > 0 else ''}{self.charge_state}"
            )

    def _check_if_multiple_finite_size_corrections(self):
        """
        Checks that there is no double counting of finite-size charge
        corrections, in the defect_entry.corrections dict.
        """
        matching_finite_size_correction_keys = {
            key
            for key in self.corrections
            if any(x in key for x in ["FNV", "freysoldt", "Freysoldt", "Kumagai", "kumagai"])
        }
        if len(matching_finite_size_correction_keys) > 1:
            warnings.warn(
                f"It appears there are multiple finite-size charge corrections in the corrections dict "
                f"attribute for defect {self.name}. These are:"
                f"\n{matching_finite_size_correction_keys}."
                f"\nPlease ensure there is no double counting / duplication of energy corrections!"
            )

    @property
    def corrected_energy(self) -> float:
        """
        The energy of the defect entry with _all_ corrections applied.
        """
        self._check_if_multiple_finite_size_corrections()
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

    def as_dict(self) -> dict:
        """
        Returns a dictionary representation of the DefectEntry object.
        """
        # ignore warning about oxidation states not summing to Structure charge:
        warnings.filterwarnings("ignore", message=".*unset_charge.*")

        return asdict(self)

    def _check_correction_error_and_return_output(
        self,
        correction_output,
        correction_error,
        return_correction_error=False,
        type="FNV",
        error_tolerance=0.05,
    ):
        if return_correction_error:
            return correction_output, correction_error

        if (
            correction_error > error_tolerance
        ):  # greater than 50 meV error in charge correction, warn the user
            warnings.warn(
                f"Estimated error in the {'Freysoldt (FNV)' if type == 'FNV' else 'Kumagai (eFNV)'} "
                f"charge correction for defect {self.name} is {correction_error:.3f} eV (i.e. which is "
                f"than the `error_tolerance`: {error_tolerance:.3f} eV). You may want to check the "
                f"accuracy of the correction by plotting the site potential differences (using "
                f"`defect_entry.get_{'freysoldt' if type == 'FNV' else 'kumagai'}_correction()` with "
                f"`plot=True`). If this error is not acceptable (and this charge state is reasonable), "
                f"you likely need to use a larger supercell for the defect calculations."
            )

        return correction_output

    def get_freysoldt_correction(
        self,
        dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
        defect_locpot: Optional[Union[str, Locpot, dict]] = None,
        bulk_locpot: Optional[Union[str, Locpot, dict]] = None,
        plot: bool = False,
        filename: Optional[str] = None,
        axis=None,
        return_correction_error: bool = False,
        error_tolerance: float = 0.05,
        **kwargs,
    ) -> CorrectionResult:
        """
        Compute the _isotropic_ Freysoldt (FNV) correction for the
        defect_entry.

        The correction is added to the `defect_entry.corrections` dictionary
        (to be used in following formation energy calculations).
        If this correction is used, please cite Freysoldt's
        original paper; 10.1103/PhysRevLett.102.016402.

        Args:
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Total dielectric constant of the host compound (including both
                ionic and (high-frequency) electronic contributions). If None,
                then the dielectric constant is taken from the `defect_entry`
                `calculation_metadata` if available.
            defect_locpot:
                Path to the output VASP LOCPOT file from the defect supercell
                calculation, or the corresponding pymatgen Locpot object, or
                a dictionary of the planar-averaged potential in the form:
                {i: Locpot.get_average_along_axis(i) for i in [0,1,2]}.
                If None, will try to use `defect_locpot` from the
                `defect_entry` `calculation_metadata` if available.
            bulk_locpot:
                Path to the output VASP LOCPOT file from the bulk supercell
                calculation, or the corresponding pymatgen Locpot object, or
                a dictionary of the planar-averaged potential in the form:
                {i: Locpot.get_average_along_axis(i) for i in [0,1,2]}.
                If None, will try to use `bulk_locpot` from the
                `defect_entry` `calculation_metadata` if available.
            plot (bool):
                Whether to plot the FNV electrostatic potential plots (for
                manually checking the behaviour of the charge correction here).
            filename (str):
                Filename to save the FNV electrostatic potential plots to.
                If None, plots are not saved.
            axis (int or None):
                If int, then the FNV electrostatic potential plot along the
                specified axis (0, 1, 2 for a, b, c) will be plotted. Note that
                the output charge correction is still that for _all_ axes.
                If None, then all three axes are plotted.
            return_correction_error (bool):
                If True, also returns the average standard deviation of the
                planar-averaged potential difference times the defect charge
                (which gives an estimate of the error range of the correction
                energy). Default is False.
            error_tolerance (float):
                If the estimated error in the charge correction is greater than
                this value (in eV), then a warning is raised. (default: 0.05 eV)
            **kwargs:
                Additional kwargs to pass to
                pymatgen.analysis.defects.corrections.freysoldt.get_freysoldt_correction
                (e.g. energy_cutoff, mad_tol, q_model, step).

        Returns:
            CorrectionResults (summary of the corrections applied and metadata), and
            the matplotlib figure object (or axis object if axis specified) if `plot`
            or `saved` is True.
        """
        from doped.utils.corrections import get_freysoldt_correction

        if dielectric is None:
            dielectric = self.calculation_metadata.get("dielectric", None)
        if dielectric is None:
            raise ValueError(
                "No dielectric constant provided, either as a function argument or in "
                "defect_entry.calculation_metadata."
            )

        fnv_correction_output = get_freysoldt_correction(
            defect_entry=self,
            dielectric=dielectric,
            defect_locpot=defect_locpot,
            bulk_locpot=bulk_locpot,
            plot=plot,
            filename=filename,
            axis=axis,
            **kwargs,
        )
        correction = fnv_correction_output if not plot and filename is None else fnv_correction_output[0]
        self.corrections.update({"freysoldt_charge_correction": correction.correction_energy})
        self._check_if_multiple_finite_size_corrections()
        self.corrections_metadata.update({"freysoldt_charge_correction": correction.metadata.copy()})

        # check accuracy of correction:
        correction_error = np.mean(
            [
                np.sqrt(
                    correction.metadata["plot_data"][i]["pot_corr_uncertainty_md"]["stats"]["variance"]
                )
                for i in [0, 1, 2]
            ]
        ) * abs(self.charge_state)

        return self._check_correction_error_and_return_output(
            fnv_correction_output,
            correction_error,
            return_correction_error,
            type="FNV",
            error_tolerance=error_tolerance,
        )

    def get_kumagai_correction(
        self,
        dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
        defect_outcar: Optional[Union[str, Outcar]] = None,
        bulk_outcar: Optional[Union[str, Outcar]] = None,
        plot: bool = False,
        filename: Optional[str] = None,
        return_correction_error: bool = False,
        error_tolerance: float = 0.05,
        **kwargs,
    ):
        """
        Compute the Kumagai (eFNV) finite-size charge correction for the
        defect_entry. Compatible with both isotropic/cubic and anisotropic
        systems.

        The correction is added to the `defect_entry.corrections` dictionary
        (to be used in following formation energy calculations).
        If this correction is used, please cite the Kumagai & Oba paper:
        10.1103/PhysRevB.89.195205

        Args:
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Total dielectric constant of the host compound (including both
                ionic and (high-frequency) electronic contributions). If None,
                then the dielectric constant is taken from the `defect_entry`
                `calculation_metadata` if available.
            defect_outcar:
                Path to the output VASP OUTCAR file from the defect supercell
                calculation, or the corresponding pymatgen Outcar object.
                If None, will try to use the `defect_supercell_site_potentials`
                from the `defect_entry` `calculation_metadata` if available.
            bulk_outcar:
                Path to the output VASP OUTCAR file from the bulk supercell
                calculation, or the corresponding pymatgen Outcar object.
                If None, will try to use the `bulk_supercell_site_potentials`
                from the `defect_entry` `calculation_metadata` if available.
            plot (bool):
                Whether to plot the Kumagai site potential plots (for
                manually checking the behaviour of the charge correction here).
            filename (str):
                Filename to save the Kumagai site potential plots to.
                If None, plots are not saved.
            return_correction_error (bool):
                If True, also returns the standard error of the mean of the
                sampled site potential differences times the defect charge
                (which gives an estimate of the error range of the correction
                energy). Default is False.
            error_tolerance (float):
                If the estimated error in the charge correction is greater than
                this value (in eV), then a warning is raised. (default: 0.05 eV)
            **kwargs:
                Additional kwargs to pass to
                pydefect.corrections.efnv_correction.ExtendedFnvCorrection
                (e.g. charge, defect_region_radius, defect_coords).

        Returns:
            CorrectionResults (summary of the corrections applied and metadata), and
            the matplotlib figure object if `plot` or `saved` is True.
        """
        from doped.utils.corrections import get_kumagai_correction

        if dielectric is None:
            dielectric = self.calculation_metadata.get("dielectric", None)
        if dielectric is None:
            raise ValueError(
                "No dielectric constant provided, either as a function argument or in "
                "defect_entry.calculation_metadata."
            )

        efnv_correction_output = get_kumagai_correction(
            defect_entry=self,
            dielectric=dielectric,
            defect_outcar=defect_outcar,
            bulk_outcar=bulk_outcar,
            plot=plot,
            filename=filename,
            **kwargs,
        )
        correction = efnv_correction_output if not plot and filename is None else efnv_correction_output[0]
        self.corrections.update({"kumagai_charge_correction": correction.correction_energy})
        self._check_if_multiple_finite_size_corrections()
        self.corrections_metadata.update({"kumagai_charge_correction": correction.metadata.copy()})

        # check accuracy of correction:
        efnv_corr_obj = correction.metadata["pydefect_ExtendedFnvCorrection"]
        sampled_pot_diff_array = np.array(
            [s.diff_pot for s in efnv_corr_obj.sites if s.distance > efnv_corr_obj.defect_region_radius]
        )

        # correction energy error can be estimated from standard error of the mean:
        correction_error = sem(sampled_pot_diff_array) * abs(self.charge_state)
        return self._check_correction_error_and_return_output(
            efnv_correction_output,
            correction_error,
            return_correction_error,
            type="eFNV",
            error_tolerance=error_tolerance,
        )


def _guess_and_set_struct_oxi_states(structure, try_without_max_sites=False, queue=None):
    """
    Tries to guess (and set) the oxidation states of the input structure.
    """
    if try_without_max_sites:
        with contextlib.suppress(Exception):
            structure.add_oxidation_state_by_guess()
            # check all oxidation states are whole numbers:
            if all(specie.oxi_state.is_integer() for specie in structure.species):
                if queue is not None:
                    queue.put(structure)
                return

    # else try to use the reduced cell since oxidation state assignment scales poorly with system size:
    try:
        structure.add_oxidation_state_by_guess(max_sites=-1)
        # check oxi_states assigned and not all zero
        if all(specie.oxi_state == 0 for specie in structure.species) or not all(
            specie.oxi_state.is_integer() for specie in structure.species
        ):
            structure.add_oxidation_state_by_guess()
    except Exception:
        structure.add_oxidation_state_by_guess()

    if queue is not None:
        queue.put(structure)


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
            oxi_state=0,  # set oxi_state in more efficient and robust way below (crashes for large
            # input structures)
            equivalent_sites=[site.to_unit_cell() for site in equivalent_sites]
            if equivalent_sites is not None
            else None,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            user_charges=user_charges,
        )  # core attributes

        if oxi_state is None:
            self._set_oxi_state()
        else:
            self.oxi_state = oxi_state

        self.conventional_structure: Optional[Structure] = doped_kwargs.get("conventional_structure", None)
        self.conv_cell_frac_coords: Optional[np.ndarray] = doped_kwargs.get("conv_cell_frac_coords", None)
        self.equiv_conv_cell_frac_coords: List[np.ndarray] = doped_kwargs.get(
            "equiv_conv_cell_frac_coords", []
        )
        self._BilbaoCS_conv_cell_vector_mapping: List[int] = doped_kwargs.get(
            "_BilbaoCS_conv_cell_vector_mapping", [0, 1, 2]
        )
        self.wyckoff: Optional[str] = doped_kwargs.get("wyckoff", None)

    def _set_oxi_state(self):
        # only try guessing bulk oxi states if not already set:
        if not (
            all(hasattr(site.specie, "oxi_state") for site in self.structure.sites)
            and all(isinstance(site.specie.oxi_state, (int, float)) for site in self.structure.sites)
        ):
            _guess_and_set_struct_oxi_states(self.structure)

        self.oxi_state = self._guess_oxi_state()

    @classmethod
    def _from_pmg_defect(cls, defect: core.Defect, bulk_oxi_states=False, **doped_kwargs) -> "Defect":
        """
        Create a doped Defect from a pymatgen Defect object.

        Args:
            defect:
                pymatgen Defect object.
            bulk_oxi_states:
                Either a dict of bulk oxidation states to use, or a boolean. If True,
                re-guesses the oxidation state of the defect (ignoring the pymatgen
                Defect oxi_state attribute), otherwise uses the already-set oxi_state
                (default = 0). Used in doped defect generation to make defect setup
                more robust and efficient (particularly for odd input structures,
                such as defect supercells etc).
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

        if isinstance(bulk_oxi_states, dict):
            # set oxidation states, as these are removed in pymatgen defect generation
            defect.structure.add_oxidation_state_by_element(bulk_oxi_states)

        return cls(
            structure=defect.structure,
            site=defect.site.to_unit_cell(),  # ensure mapped to unit cell
            multiplicity=defect.multiplicity,
            oxi_state=defect.oxi_state if not bulk_oxi_states else None,
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
            sc_mat = supercells.get_sc_fromstruct(
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

        sorted_sc_defect_struct = sc_defect_struct.get_sorted_structure()  # ensure proper sorting

        return (
            (
                sorted_sc_defect_struct,
                sc_site,
                equiv_sites,
            )
            if return_sites
            else sorted_sc_defect_struct
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


def doped_defect_from_pmg_defect(defect: core.Defect, bulk_oxi_states=False, **doped_kwargs):
    """
    Create the corresponding doped Defect (Vacancy, Interstitial, Substitution)
    from an input pymatgen Defect object.

    Args:
        defect:
            pymatgen Defect object.
        bulk_oxi_states:
            Either a dict of bulk oxidation states to use, or a boolean. If True,
            re-guesses the oxidation state of the defect (ignoring the pymatgen
            Defect oxi_state attribute), otherwise uses the already-set oxi_state
            (default = 0). Used in doped defect generation to make defect setup
            more robust and efficient (particularly for odd input structures,
            such as defect supercells etc).
        **doped_kwargs:
            Additional keyword arguments to define doped-specific attributes
            (see class docstring).
    """
    # determine defect type:
    if isinstance(defect, core.Vacancy):
        defect_type = Vacancy
    elif isinstance(defect, core.Substitution):
        defect_type = Substitution
    elif isinstance(defect, core.Interstitial):
        defect_type = Interstitial
    else:
        raise TypeError(
            f"Input defect must be a pymatgen Vacancy, Substitution or Interstitial object, "
            f"not {type(defect)}."
        )

    return defect_type._from_pmg_defect(defect, bulk_oxi_states=bulk_oxi_states, **doped_kwargs)


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
