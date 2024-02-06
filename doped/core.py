"""
Core functions and classes for defects in doped.
"""


import collections
import contextlib
import warnings
from dataclasses import asdict, dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core, thermo
from pymatgen.analysis.defects.utils import CorrectionResult
from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from scipy.stats import sem

from doped.utils.displacements import _plot_site_displacements


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
            ``pymatgen`` ``ComputedStructureEntry`` for the `defect` supercell.
        sc_defect_frac_coords:
            The fractional coordinates of the defect in the supercell.
        bulk_entry:
            ``pymatgen`` ``ComputedEntry`` for the bulk supercell reference. Required
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
        degeneracy_factors:
            A dictionary of degeneracy factors contributing to the total degeneracy
            of the defect species (such as spin and configurational degeneracy etc).
            This is an important factor in the defect concentration equation (see
            discussion in doi.org/10.1039/D2FD00043A and doi.org/10.1039/D3CS00432E),
            and so affects the output of the defect concentration / Fermi level
            functions. This can be edited by the user if the doped defaults are not
            appropriate (e.g. doped assumes singlet (S=0) state for even-electron
            defects and doublet (S=1/2) state for odd-electron defects, which is
            typically the case but can have triplets (S=1) or other multiplets for
            e.g. bipolarons, quantum / d-orbital / magnetic defects etc).

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
    bulk_entry: Optional[ComputedEntry] = None
    entry_id: Optional[str] = None

    # doped attributes:
    name: str = ""
    calculation_metadata: Dict = field(default_factory=dict)
    degeneracy_factors: Dict = field(default_factory=dict)
    conventional_structure: Optional[Structure] = None
    conv_cell_frac_coords: Optional[np.ndarray] = None
    equiv_conv_cell_frac_coords: List[np.ndarray] = field(default_factory=list)
    _BilbaoCS_conv_cell_vector_mapping: List[int] = field(default_factory=lambda: [0, 1, 2])
    wyckoff: Optional[str] = None
    charge_state_guessing_log: Dict = field(default_factory=dict)
    defect_supercell: Optional[Structure] = None
    defect_supercell_site: Optional[PeriodicSite] = None  # TODO: Add `from_structures` method to
    # doped DefectEntry?? (Yeah would prob be useful function to have for porting over stuff from other
    # codes etc)
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
        The energy of the defect entry with `all` corrections applied.
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
            if isinstance(correction_output, tuple):  # correction_output may be a tuple, so amalgamate:
                return (*correction_output, correction_error)
            return correction_output, correction_error

        if (
            correction_error > error_tolerance
        ):  # greater than 50 meV error in charge correction, warn the user
            warnings.warn(
                f"Estimated error in the {'Freysoldt (FNV)' if type == 'FNV' else 'Kumagai (eFNV)'} "
                f"charge correction for defect {self.name} is {correction_error:.3f} eV (i.e. which is "
                f"greater than the `error_tolerance`: {error_tolerance:.3f} eV). You may want to check "
                f"the accuracy of the correction by plotting the site potential differences (using "
                f"`defect_entry.get_{'freysoldt' if type == 'FNV' else 'kumagai'}_correction()` with "
                f"`plot=True`). Large errors are often due to unstable or shallow defect charge states ("
                f"which can't be accurately modelled with the supercell approach). If this error is not "
                f"acceptable, you may need to use a larger supercell for more accurate energies."
            )  # TODO: Link docs mention of shallow defects / false charge states here when ready

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
        Compute the `isotropic` Freysoldt (FNV) correction for the
        defect_entry.

        The correction is added to the ``defect_entry.corrections`` dictionary
        (to be used in following formation energy calculations).
        If this correction is used, please cite Freysoldt's
        original paper; 10.1103/PhysRevLett.102.016402.

        Args:
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Total dielectric constant of the host compound (including both
                ionic and (high-frequency) electronic contributions), in the
                same xyz Cartesian basis as the supercell calculations. If None,
                then the dielectric constant is taken from the ``defect_entry``
                ``calculation_metadata`` if available.
            defect_locpot:
                Path to the output VASP LOCPOT file from the defect supercell
                calculation, or the corresponding pymatgen Locpot object, or
                a dictionary of the planar-averaged potential in the form:
                {i: Locpot.get_average_along_axis(i) for i in [0,1,2]}.
                If None, will try to use ``defect_locpot`` from the
                ``defect_entry`` ``calculation_metadata`` if available.
            bulk_locpot:
                Path to the output VASP LOCPOT file from the bulk supercell
                calculation, or the corresponding pymatgen Locpot object, or
                a dictionary of the planar-averaged potential in the form:
                {i: Locpot.get_average_along_axis(i) for i in [0,1,2]}.
                If None, will try to use ``bulk_locpot`` from the
                ``defect_entry`` ``calculation_metadata`` if available.
            plot (bool):
                Whether to plot the FNV electrostatic potential plots (for
                manually checking the behaviour of the charge correction here).
            filename (str):
                Filename to save the FNV electrostatic potential plots to.
                If None, plots are not saved.
            axis (int or None):
                If int, then the FNV electrostatic potential plot along the
                specified axis (0, 1, 2 for a, b, c) will be plotted. Note that
                the output charge correction is still that for `all` axes.
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
            the matplotlib figure object (or axis object if axis specified) if ``plot``
            is True, and the estimated charge correction error if
            ``return_correction_error`` is True.
        """
        from doped.corrections import get_freysoldt_correction

        if dielectric is None:
            dielectric = self.calculation_metadata.get("dielectric")
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
        self.corrections_metadata.update({"freysoldt_charge_correction_error": correction_error})

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
        defect_region_radius: Optional[float] = None,
        excluded_indices: Optional[List[int]] = None,
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

        The correction is added to the ``defect_entry.corrections`` dictionary
        (to be used in following formation energy calculations).
        If this correction is used, please cite the Kumagai & Oba paper:
        10.1103/PhysRevB.89.195205

        Typically for reasonably well-converged supercell sizes, the default
        ``defect_region_radius`` works perfectly well. However, for certain materials
        at small/intermediate supercell sizes, you may want to adjust this (and/or
        ``excluded_indices``) to ensure the best sampling of the plateau region away
        from the defect position - ``doped`` should throw a warning in these cases
        (about the correction error being above the default tolerance (50 meV)).
        For example, with layered materials, the defect charge is often localised
        to one layer, so we may want to adjust ``defect_region_radius`` and/or
        ``excluded_indices`` to ensure that only sites in other layers are used for
        the sampling region (plateau) - see example on doped docs.

        Args:
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Total dielectric constant of the host compound (including both
                ionic and (high-frequency) electronic contributions), in the
                same xyz Cartesian basis as the supercell calculations. If None,
                then the dielectric constant is taken from the ``defect_entry``
                ``calculation_metadata`` if available.
            defect_region_radius (float):
                Radius of the defect region (in Å). Sites outside the defect
                region are used for sampling the electrostatic potential far
                from the defect (to obtain the potential alignment).
                If None (default), uses the Wigner-Seitz radius of the supercell.
            excluded_indices (list):
                List of site indices (in the defect supercell) to exclude from
                the site potential sampling in the correction calculation/plot.
                If None (default), no sites are excluded.
            defect_outcar (str or Outcar):
                Path to the output VASP OUTCAR file from the defect supercell
                calculation, or the corresponding pymatgen Outcar object.
                If None, will try to use the ``defect_supercell_site_potentials``
                from the ``defect_entry`` ``calculation_metadata`` if available.
            bulk_outcar (str or Outcar):
                Path to the output VASP OUTCAR file from the bulk supercell
                calculation, or the corresponding pymatgen Outcar object.
                If None, will try to use the ``bulk_supercell_site_potentials``
                from the ``defect_entry`` ``calculation_metadata`` if available.
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
            the matplotlib figure object if ``plot`` is True, and the estimated charge
            correction error if ``return_correction_error`` is True.
        """
        from doped.corrections import get_kumagai_correction

        if dielectric is None:
            dielectric = self.calculation_metadata.get("dielectric")
        if dielectric is None:
            raise ValueError(
                "No dielectric constant provided, either as a function argument or in "
                "defect_entry.calculation_metadata."
            )

        efnv_correction_output = get_kumagai_correction(
            defect_entry=self,
            dielectric=dielectric,
            defect_region_radius=defect_region_radius,
            excluded_indices=excluded_indices,
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
        self.corrections_metadata.update({"kumagai_charge_correction_error": correction_error})
        return self._check_correction_error_and_return_output(
            efnv_correction_output,
            correction_error,
            return_correction_error,
            type="eFNV",
            error_tolerance=error_tolerance,
        )

    def _get_chempot_term(self, chemical_potentials=None):
        chemical_potentials = chemical_potentials or {}

        return sum(
            chem_pot * -self.defect.element_changes[Element(el)]
            for el, chem_pot in chemical_potentials.items()
            if Element(el) in self.defect.element_changes
        )

    def formation_energy(
        self,
        chempots: Optional[dict] = None,
        facet: Optional[str] = None,
        vbm: Optional[float] = None,
        fermi_level: float = 0,
    ) -> float:
        """
        Compute the formation energy for the DefectEntry at a given chemical
        potential limit and fermi_level.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energy. This can have the doped form of:
                {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using ``facet``.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                If None (default), sets all chemical potentials to 0 (inaccurate
                formation energies!)
            facet (str):
                The phase diagram facet (chemical potential limit) to use for
                calculating the formation energy. Can be:

                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                - None (default), if ``chempots`` corresponds to a single chemical
                  potential limit - otherwise will use the first chemical potential
                  limit in the doped chempots dict.
            vbm (float):
                VBM eigenvalue in the bulk supercell, to use as Fermi level reference
                point for calculating formation energy. If None (default), will use
                "vbm" from the calculation_metadata dict attribute if present.
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM. Default is 0 (i.e. the VBM).

        Returns:
            Formation energy value (float)
        """
        if chempots is None:
            _no_chempots_warning("Formation energies (and concentrations)")

        dft_chempots = _get_dft_chempots(chempots, facet)
        chempot_correction = self._get_chempot_term(dft_chempots)
        formation_energy = self.get_ediff() + chempot_correction

        if vbm is not None:
            formation_energy += self.charge_state * (vbm + fermi_level)
        elif "vbm" in self.calculation_metadata:
            formation_energy += self.charge_state * (self.calculation_metadata["vbm"] + fermi_level)
        else:
            warnings.warn(
                "VBM eigenvalue was not set, and is not present in DefectEntry.calculation_metadata. "
                "Formation energy will be inaccurate!"
            )

        return formation_energy

    def equilibrium_concentration(
        self,
        chempots: Optional[dict] = None,
        facet: Optional[str] = None,
        temperature: float = 300,
        fermi_level: float = 0,
        vbm: Optional[float] = None,
        per_site: bool = False,
    ) -> float:
        """
        Compute the `equilibrium` concentration (in cm^-3) for the DefectEntry
        at a given chemical potential limit, fermi_level and temperature,
        assuming the dilute limit approximation.

        Note that these are the `equilibrium` defect concentrations!
        DefectThermodynamics.get_quenched_fermi_level_and_concentrations() can
        instead be used to calculate the Fermi level and defect concentrations
        for a material grown/annealed at higher temperatures and then cooled
        (quenched) to room/operating temperature (where defect concentrations
        are assumed to remain fixed) - this is known as the frozen defect
        approach and is typically the most valid approximation (see its
        docstring for more information, and discussion in 10.1039/D3CS00432E).

        The degeneracy/multiplicity factor "g" is an important parameter in the defect
        concentration equation (see discussion in doi.org/10.1039/D2FD00043A and
        doi.org/10.1039/D3CS00432E), affecting the final concentration by up to 2 orders
        of magnitude. This factor is taken from the product of the
        defect_entry.defect.multiplicity and defect_entry.degeneracy_factors attributes.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energy (and thus concentration). This can have the doped form of:
                {"facets": [{'facet': [chempot_dict]}]}
                (the format generated by doped's chemical potential parsing functions
                (see tutorials)) and specific facets (chemical potential limits) can
                then be chosen using ``facet``.
                Alternatively, can be a dictionary of **DFT**/absolute chemical
                potentials (not formal chemical potentials!), in the format:
                {element symbol: chemical potential}.
                If None (default), sets all chemical potentials to 0 (inaccurate
                formation energies and concentrations!)
            facet (str):
                The phase diagram facet (chemical potential limit) to use for
                calculating the formation energy and thus concentration. Can be:

                - "X-rich"/"X-poor" where X is an element in the system, in which
                  case the most X-rich/poor facet will be used (e.g. "Li-rich").
                - A key in the (self.)chempots["facets"] dictionary, if the chempots
                  dict is in the doped format (see chemical potentials tutorial).
                - None (default), if ``chempots`` corresponds to a single chemical
                  potential limit - otherwise will use the first chemical potential
                  limit in the doped chempots dict.
            temperature (float):
                Temperature in Kelvin at which to calculate the equilibrium concentration.
            vbm (float):
                VBM eigenvalue in the bulk supercell, to use as Fermi level reference
                point for calculating formation energy. If None (default), will use
                "vbm" from the calculation_metadata dict attribute if present.
            fermi_level (float):
                Value corresponding to the electron chemical potential,
                referenced to the VBM. Default is 0 (i.e. the VBM).
            per_site (bool):
                Whether to return the concentration as fractional concentration per site,
                rather than the default of per cm^3. (default: False)

        Returns:
            Concentration in cm^-3 (or as fractional per site, if per_site = True) (float)
        """
        if "spin degeneracy" not in self.degeneracy_factors:
            warnings.warn(
                "'spin degeneracy' is not defined in the DefectEntry degeneracy_factors attribute. "
                "This factor contributes to the degeneracy term 'g' in the defect concentration equation "
                "(N_X = N*g*exp(-E/kT)) and is automatically computed when parsing with doped "
                "(see discussion in doi.org/10.1039/D2FD00043A and doi.org/10.1039/D3CS00432E). This will "
                "affect the computed defect concentration / Fermi level!\n"
                "To avoid this, you can (re-)parse your defect(s) with doped, or manually set "
                "'spin degeneracy' in the degeneracy_factors attribute(s) - usually 2 for odd-electron "
                "defect species and 1 for even-electron)."
            )

        if (
            "orientational degeneracy" not in self.degeneracy_factors
            and self.defect.defect_type != core.DefectType.Interstitial
        ):
            warnings.warn(
                "'orientational degeneracy' is not defined in the DefectEntry degeneracy_factors "
                "attribute (for this vacancy/substitution defect). This factor contributes to the "
                "degeneracy term 'g' in the defect concentration equation (N_X = N*g*exp(-E/kT) - see "
                "discussion in doi.org/10.1039/D2FD00043A and doi.org/10.1039/D3CS00432E) and is "
                "automatically computed when parsing with doped if possible (if the defect supercell "
                "doesn't break the host periodicity). This will affect the computed defect concentrations "
                "/ Fermi level!\n"
                "To avoid this, you can (re-)parse your defects with doped (if not tried already), or "
                "manually set 'orientational degeneracy' in the degeneracy_factors attribute(s)."
            )

        formation_energy = self.formation_energy(  # if chempots is None, this will throw warning
            chempots=chempots, facet=facet, vbm=vbm, fermi_level=fermi_level
        )
        from scipy.constants import value as constants_value

        exp_factor = np.exp(
            -formation_energy / (constants_value("Boltzmann constant in eV/K") * temperature)
        )
        degeneracy_factor = (
            reduce(lambda x, y: x * y, self.degeneracy_factors.values()) if self.degeneracy_factors else 1
        )
        if per_site:
            return exp_factor * degeneracy_factor

        volume_in_cm3 = self.defect.structure.volume * 1e-24  # convert volume in Å^3 to cm^3

        return self.defect.multiplicity * degeneracy_factor * exp_factor / volume_in_cm3

    def __repr__(self):
        """
        Returns a string representation of the DefectEntry object.
        """
        from doped.utils.parsing import _get_bulk_supercell

        bulk_supercell = _get_bulk_supercell(self)
        if bulk_supercell is not None:
            formula = bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        else:
            formula = self.defect.structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
        attrs = {k for k in vars(self) if not k.startswith("_")}
        methods = {k for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")}
        return (
            f"doped DefectEntry: {self.name}, with bulk composition: {formula} and defect: "
            f"{self.defect.name}. Available attributes:\n{attrs}\n\nAvailable methods:\n{methods}"
        )

    def __eq__(self, other):
        """
        Determine whether two DefectEntry objects are equal, by comparing
        self.name and self.sc_entry.
        """
        return self.name == other.name and self.sc_entry == other.sc_entry

    def plot_site_displacements(
        self,
        separated_by_direction: Optional[bool] = True,
        use_plotly: Optional[bool] = False,
        mpl_style_file: Optional[str] = "",
    ):
        """
        Plot the site displacements as a function of distance from the defect
        site.

        Args:
            separated_by_direction (bool):
                Whether to plot the site displacements separated by the
                x, y and z directions (True) or all together (False).
            use_plotly (bool):
                Whether to use plotly (True) or matplotlib (False).
            mpl_style_file (str):
                Path to a matplotlib style file to use for the plot. If None,
                uses the default doped style file.
        """
        return _plot_site_displacements(
            defect_entry=self,
            separated_by_direction=separated_by_direction,
            use_plotly=use_plotly,
            style_file=mpl_style_file,
        )


def _no_chempots_warning(property="Formation energies (and concentrations)"):
    warnings.warn(
        f"No chemical potentials supplied, so using 0 for all chemical potentials. {property} will likely "
        f"be highly inaccurate!"
    )


def _get_dft_chempots(chempots, facet):
    """
    Parse the DFT chempots from the input chempots and facet.
    """
    from doped.thermodynamics import _parse_chempots, _parse_facet

    chempots, _el_refs = _parse_chempots(chempots)
    if chempots is not None:
        facet = _parse_facet(chempots, facet)
        if facet is None:
            facet = list(chempots["facets"].keys())[0]
            if "User" not in facet:
                warnings.warn(
                    f"No facet (chemical potential limit) specified! Using {facet} for computing the "
                    f"formation energy"
                )

    elif facet is not None:
        warnings.warn(
            "You have specified a facet (chemical potential limit) but no chemical potentials "
            "(`chempots`) were supplied, so `facet` will be ignored."
        )

    if facet is not None and chempots is not None:
        return chempots["facets"][facet]

    return chempots


def _guess_and_set_struct_oxi_states(structure, try_without_max_sites=False, queue=None):
    """
    Tries to guess (and set) the oxidation states of the input structure.
    """
    if try_without_max_sites:
        with contextlib.suppress(Exception):
            structure.add_oxidation_state_by_guess()
            # check all oxidation states are whole numbers:
            if all(np.isclose(int(specie.oxi_state), specie.oxi_state) for specie in structure.species):
                if queue is not None:
                    queue.put(structure)
                return

    # else try to use the reduced cell since oxidation state assignment scales poorly with system size:
    try:
        attempt = 0
        structure.add_oxidation_state_by_guess(max_sites=-1)
        # check oxi_states assigned and not all zero:
        while (
            attempt < 3
            and all(specie.oxi_state == 0 for specie in structure.species)
            or not all(np.isclose(int(specie.oxi_state), specie.oxi_state) for specie in structure.species)
        ):
            attempt += 1
            if attempt == 1:
                structure.add_oxidation_state_by_guess(max_sites=-1, all_oxi_states=True)
            elif attempt == 2:
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
                the primitive structure of the host crystal for defect
                generation, and/or the calculation supercell for defect
                parsing.
            site: The defect site in the structure.
            multiplicity: The multiplicity of the defect in the structure.
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
            oxi_state=None if bulk_oxi_states else defect.oxi_state,
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
        target_frac_coords: Optional[np.ndarray] = None,
        return_sites: bool = False,
        min_image_distance: float = 10.0,  # same as current pymatgen default
        min_atoms: int = 50,  # different to current pymatgen default (80)
        force_cubic: bool = False,
        force_diagonal: bool = False,  # same as current pymatgen default
        ideal_threshold: float = 0.1,
        min_length: Optional[float] = None,  # same as current pymatgen default, kept for compatibility
        dummy_species: Optional[str] = None,
    ) -> Structure:
        """
        Generate the simulation supercell for a defect.

        Redefined from the parent class to allow the use of ``target_frac_coords``
        to place the defect at the closest equivalent site to the target
        fractional coordinates in the supercell, while keeping the supercell
        fixed (to avoid any issues with defect parsing).
        Also returns information about equivalent defect sites in the supercell.

        If ``sc_mat`` is None, then the supercell is generated automatically
        using the ``doped`` algorithm described in the ``get_ideal_supercell_matrix``
        function docstring in ``doped.generation``.

        Args:
            sc_mat (3x3 matrix):
                Transformation matrix of ``self.structure`` to create the supercell.
                If None, then automatically computed using ``get_ideal_supercell_matrix``
                from ``doped.generation``.
            target_frac_coords (3x1 matrix):
                If set, the defect will be placed at the closest equivalent site to
                these fractional coordinates (using self.equivalent_sites).
            return_sites (bool):
                If True, returns a tuple of the defect supercell, defect supercell
                site and list of equivalent supercell sites.
            dummy_species (str):
                Dummy species to highlight the defect position (for visualizing vacancies).
            min_image_distance (float):
                Minimum image distance in Å of the generated supercell (i.e. minimum
                distance between periodic images of atoms/sites in the lattice),
                if ``sc_mat`` is None.
                (Default = 10.0)
            min_atoms (int):
                Minimum number of atoms allowed in the generated supercell, if ``sc_mat``
                is None.
                (Default = 50)
            force_cubic (bool):
                Enforce usage of ``CubicSupercellTransformation`` from
                ``pymatgen`` for supercell generation (if ``sc_mat`` is None).
                (Default = False)
            force_diagonal (bool):
                If True, return a transformation with a diagonal
                transformation matrix (if ``sc_mat`` is None).
                (Default = False)
            ideal_threshold (float):
                Threshold for increasing supercell size (beyond that which satisfies
                ``min_image_distance`` and `min_atoms``) to achieve an ideal
                supercell matrix (i.e. a diagonal expansion of the primitive or
                conventional cell). Supercells up to ``1 + perfect_cell_threshold``
                times larger (rounded up) are trialled, and will instead be
                returned if they yield an ideal transformation matrix (if ``sc_mat``
                is None).
                (Default = 0.1; i.e. 10% larger than the minimum size)
            min_length (float):
                Same as ``min_image_distance`` (kept for compatibility).

        Returns:
            The defect supercell structure. If ``return_sites`` is True, also returns
            the defect supercell site and list of equivalent supercell sites.
        """
        if sc_mat is None:
            if min_length is not None:
                min_image_distance = min_length

            from doped.generation import get_ideal_supercell_matrix

            sc_mat = get_ideal_supercell_matrix(
                self.structure,
                min_image_distance=min_image_distance,
                min_atoms=min_atoms,
                ideal_threshold=ideal_threshold,
                force_cubic=force_cubic,
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

    def __repr__(self) -> str:
        """
        String representation of a vacancy defect.
        """
        frac_coords_string = ",".join(f"{x:.3f}" for x in self.site.frac_coords)
        return f"{self.name} vacancy defect at site [{frac_coords_string}] in structure"


class Substitution(core.Substitution, Defect):
    def __init__(self, *args, **kwargs):
        """
        Subclass of pymatgen.analysis.defects.core.Substitution with additional
        attributes and methods used by doped.
        """
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """
        String representation of a substitutional defect.
        """
        frac_coords_string = ",".join(f"{x:.3f}" for x in self.site.frac_coords)
        return f"{self.name} substitution defect at site [{frac_coords_string}] in structure"


class Interstitial(core.Interstitial, Defect):
    def __init__(self, *args, **kwargs):
        """
        Subclass of pymatgen.analysis.defects.core.Interstitial with additional
        attributes and methods used by doped.
        """
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """
        String representation of an interstitial defect.
        """
        frac_coords_string = ",".join(f"{x:.3f}" for x in self.site.frac_coords)
        return f"{self.name} interstitial defect at site [{frac_coords_string}] in structure"
