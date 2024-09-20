"""
Module for solving the Fermi level and defect/carrier concentrations self-
consistently over various parameter spaces including chemical potentials,
temperatures, and effective dopant concentrations, under various thermodynamic
constraints.

This is done using both doped and py-sc-fermi. The aim of this module is to
implement a unified interface for both "backends".
"""

import contextlib
import importlib.util
import warnings
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd
from monty.json import MSONable
from pymatgen.core.periodic_table import Element
from pymatgen.electronic_structure.dos import FermiDos, Spin
from pymatgen.io.vasp import Vasprun
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from doped.core import _get_dft_chempots
from doped.thermodynamics import _add_effective_dopant_concentration, _parse_chempots, _parse_limit

if TYPE_CHECKING:
    from pymatgen.util.typing import PathLike

    from doped.thermodynamics import DefectThermodynamics

    with contextlib.suppress(ImportError):
        from py_sc_fermi.defect_species import DefectSpecies
        from py_sc_fermi.defect_system import DefectSystem
        from py_sc_fermi.dos import DOS


def _get_label_and_charge(name: str) -> tuple[str, int]:
    """
    Extracts the label and charge from a defect name string.

    Args:
        name (str): Name of the defect.

    Returns:
        tuple: A tuple containing the label and charge.
    """
    last_underscore = name.rfind("_")
    label = name[:last_underscore] if last_underscore != -1 else name
    charge_str = name[last_underscore + 1 :] if last_underscore != -1 else None

    charge = 0  # Initialize charge with a default value
    if charge_str is not None:
        with contextlib.suppress(ValueError):
            charge = int(charge_str)

    return label, charge


# TODO: Update DefectThermodynamics docstrings after `py-sc-fermi` interface written, to point toward it
#  for more advanced analysis


def get_py_sc_fermi_dos_from_fermi_dos(
    fermi_dos: FermiDos,
    vbm: Optional[float] = None,
    nelect: Optional[int] = None,
    bandgap: Optional[float] = None,
) -> "DOS":
    """
    Given an input ``pymatgen`` ``FermiDos`` object, return a corresponding
    ``py-sc-fermi`` ``DOS`` object (which can then be used with the ``py-sc-
    fermi`` ``FermiSolver`` backend).

    Args:
        fermi_dos (FermiDos):
            ``pymatgen`` ``FermiDos`` object to convert to ``py-sc-fermi``
            ``DOS``.
        vbm (float):
            The valence band maximum (VBM) eigenvalue in eV. If not provided,
            the VBM will be taken from the FermiDos object. When this function
            is used internally in ``doped``, the ``DefectThermodynamics.vbm``
            attribute is used.
        nelect (int):
            The total number of electrons in the system. If not provided, the
            number of electrons will be taken from the FermiDos object (which
            usually takes this value from the ``vasprun.xml(.gz)`` when parsing).
        bandgap (float):
            Band gap of the system in eV. If not provided, the band gap will be
            taken from the FermiDos object. When this function is used internally
            in ``doped``, the ``DefectThermodynamics.band_gap`` attribute is used.

    Returns:
        DOS: A ``py-sc-fermi`` ``DOS`` object.
    """
    try:
        from py_sc_fermi.dos import DOS
    except ImportError as exc:
        raise ImportError("py-sc-fermi must be installed to use this function!") from exc

    densities = fermi_dos.densities
    if vbm is None:  # tol 1e-4 is lowest possible, as VASP rounds to 4 dp:
        vbm = fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1]

    edos = fermi_dos.energies - vbm
    if len(densities) == 2:
        dos = np.array([densities[Spin.up], densities[Spin.down]])
        spin_pol = True
    else:
        dos = np.array(densities[Spin.up])
        spin_pol = False

    if nelect is None:
        # this requires the input dos to be a FermiDos. NELECT could be calculated alternatively
        # by integrating the tdos of a ``pymatgen`` ``Dos`` object, but this isn't expected to be
        # a common use case and using parsed NELECT from vasprun.xml(.gz) is more reliable
        nelect = fermi_dos.nelecs
    if bandgap is None:
        bandgap = fermi_dos.get_gap(tol=1e-4, abs_tol=True)

    return DOS(dos=dos, edos=edos, nelect=nelect, bandgap=bandgap, spin_polarised=spin_pol)


class FermiSolver(MSONable):
    def __init__(
        self,
        defect_thermodynamics: "DefectThermodynamics",
        bulk_dos: Optional[Union[FermiDos, Vasprun, "PathLike"]] = None,
        chempots: Optional[dict] = None,
        el_refs: Optional[dict] = None,
        backend: str = "doped",
        skip_check: bool = False,
    ):
        r"""
        Class to calculate the Fermi level, defect and carrier concentrations
        under various conditions, using the input ``DefectThermodynamics``
        object.

        This constructor initializes a ``FermiSolver`` object, setting up the
        necessary attributes, which includes loading the bulk density of states
        (DOS) data from the provided VASP output.

        If using the ``py-sc-fermi`` backend (currently required for the
        ``free_defects`` and ``fix_charge_states`` options in the scanning
        functions), please cite the code paper:
        Squires et al., (2023). Journal of Open Source Software, 8(82), 4962
        https://doi.org/10.21105/joss.04962

        Args:
            defect_thermodynamics (DefectThermodynamics):
                A ``DefectThermodynamics`` object, providing access to defect
                formation energies and other related thermodynamic properties.
            bulk_dos (FermiDos or Vasprun or PathLike):
                Either a path to the ``vasprun.xml(.gz)`` output of a bulk DOS
                calculation in VASP, a ``pymatgen`` ``Vasprun`` object or a
                ``pymatgen`` ``FermiDos`` for the bulk electronic DOS, for
                calculating carrier concentrations.
                If not provided, uses ``DefectThermodynamics.bulk_dos`` if present.

                Usually this is a static calculation with the `primitive` cell of
                the bulk material, with relatively dense `k`-point sampling
                (especially for materials with disperse band edges) to ensure an
                accurately-converged DOS and thus Fermi level. ``ISMEAR = -5``
                (tetrahedron smearing) is usually recommended for best convergence
                wrt `k`-point sampling. Consistent functional settings should be
                used for the bulk DOS and defect supercell calculations.

                Note that the ``DefectThermodynamics.bulk_dos`` will be set to match
                this input, if provided.
            chempots (Optional[dict]):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus concentrations and Fermi level), under
                different chemical environments.

                If ``None`` (default), will use ``DefectThermodynamics.chempots``
                (= 0 for all chemical potentials by default, if not set).
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}], ...}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.

                If provided here, then ``defect_thermodynamics.chempots`` is set to this
                input.

                Chemical potentials can also be supplied later in each analysis function, or
                set using ``FermiSolver.defect_thermodynamics.chempots = ...`` or
                ``DefectThermodynamics.chempots = ...`` (with the same input options).
            el_refs (Optional[dict]):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in ``DefectThermodynamics`` in the format generated by
                ``doped`` (see tutorials), or if ``DefectThermodynamics.el_refs`` has been
                set.

                If provided here, then ``defect_thermodynamics.el_refs`` is set to this
                input.

                Elemental reference energies can also be supplied later in each analysis
                function, or set using ``FermiSolver.defect_thermodynamics.el_refs = ...``
                or ``DefectThermodynamics.el_refs = ...`` (with the same input options).
                Default is ``None``.
            backend (Optional[str]):
                The code backend to use for the thermodynamic calculations, which
                can be either ``"doped"`` or ``"py-sc-fermi"``. ``"py-sc-fermi"``
                allows the use of ``free_defects`` for advanced constrained defect
                equilibria (i.e. mobile defects, see advanced thermodynamics tutorial),
                while ``"doped"`` is usually (but not always) quicker. Default is
                ``doped``, but will attempt to switch to ``py-sc-fermi`` if required
                (and installed).
            skip_check (bool):
                Whether to skip the warning about the DOS VBM differing from
                ``DefectThermodynamics.vbm`` by >0.05 eV. Should only be used when the
                reason for this difference is known/acceptable. Default is ``False``.

        Key attributes:
            defect_thermodynamics (DefectThermodynamics):
                The ``DefectThermodynamics`` object used for the thermodynamic
                calculations.
            backend (str):
                The code backend used for the thermodynamic calculations (``"doped"``
                or ``"py-sc-fermi"``).
            volume (float):
                Volume of the unit cell in the bulk DOS calculation (stored in
                ``self.defect_thermodynamics.bulk_dos``).
            skip_check (bool):
                Whether to skip the warning about the DOS VBM differing from the defect
                entries VBM by >0.05 eV. Should only be used when the reason for this
                difference is known/acceptable.
            multiplicity_scaling (int):
                Scaling factor to account for the difference in volume between the
                defect supercells and the bulk DOS calculation cell, when using the
                ``py-sc-fermi`` backend.
            py_sc_fermi_dos (DOS):
                A ``py-sc-fermi`` ``DOS`` object, generated from the input
                ``FermiDos`` object, for use with the ``py-sc-fermi`` backend.
        """
        # Note: In theory multiprocessing could be introduced to make thermodynamic calculations
        # over large grids faster, but with the current code there seems to be issues with thread
        # locking / synchronisation (which shouldn't be necessary...). Worth keeping in mind if
        # needed in future.
        self.defect_thermodynamics = defect_thermodynamics
        self.skip_check = skip_check
        if bulk_dos is not None:
            self.defect_thermodynamics.bulk_dos = self.defect_thermodynamics._parse_fermi_dos(
                bulk_dos, skip_check=self.skip_check
            )
        if self.defect_thermodynamics.bulk_dos is None:
            raise ValueError(
                "No bulk DOS calculation (`bulk_dos`) provided or previously parsed to "
                "`DefectThermodynamics.bulk_dos`, which is required for calculating carrier "
                "concentrations and solving for Fermi level position."
            )
        self.volume = self.defect_thermodynamics.bulk_dos.volume

        if "fermi" in backend.lower():
            if bool(importlib.util.find_spec("py_sc_fermi")):
                self.backend = "py-sc-fermi"
            else:  # py-sc-fermi explicitly chosen but not installed
                raise ImportError(
                    "py-sc-fermi is not installed, so only the doped FermiSolver backend is available."
                )
        elif backend.lower() == "doped":
            self.backend = "doped"
        else:
            raise ValueError(f"Unrecognised `backend`: {backend}")

        self.multiplicity_scaling = 1
        self.py_sc_fermi_dos = None
        self._DefectSystem = self._DefectSpecies = self._DefectChargeState = self._DOS = None

        if self.backend == "py-sc-fermi":
            self._activate_py_sc_fermi_backend()

        # Parse chemical potentials, either using input values (after formatting them in the doped format)
        # or using the class attributes if set:
        self.defect_thermodynamics._chempots, self.defect_thermodynamics._el_refs = _parse_chempots(
            chempots or self.defect_thermodynamics.chempots, el_refs or self.defect_thermodynamics.el_refs
        )

        if self.defect_thermodynamics.chempots is None:
            raise ValueError(
                "You must supply a chemical potentials dictionary or have them present in the "
                "DefectThermodynamics object."
            )

    def _activate_py_sc_fermi_backend(self, error_message: Optional[str] = None):
        try:
            from py_sc_fermi.defect_charge_state import DefectChargeState
            from py_sc_fermi.defect_species import DefectSpecies
            from py_sc_fermi.defect_system import DefectSystem
            from py_sc_fermi.dos import DOS
        except ImportError as exc:  # py-sc-fermi activation attempted but not installed
            finishing_message = (
                ", but py-sc-fermi is not installed, so only the doped FermiSolver "
                "backend is available!"
            )
            message = (
                error_message + finishing_message
                if error_message
                else "The py-sc-fermi backend was attempted to be activated" + finishing_message
            )
            raise ImportError(message) from exc

        self._DefectSystem = DefectSystem
        self._DefectSpecies = DefectSpecies
        self._DefectChargeState = DefectChargeState
        self._DOS = DOS

        if isinstance(self.defect_thermodynamics.bulk_dos, FermiDos):
            self.py_sc_fermi_dos = get_py_sc_fermi_dos_from_fermi_dos(
                self.defect_thermodynamics.bulk_dos,
                vbm=self.defect_thermodynamics.vbm,
                bandgap=self.defect_thermodynamics.band_gap,
            )

        ms = (
            next(iter(self.defect_thermodynamics.defect_entries.values())).sc_entry.structure.volume
            / self.volume
        )
        if not np.isclose(ms, round(ms), atol=3e-2):  # check multiplicity scaling is almost an integer
            warnings.warn(
                f"The detected volume ratio ({ms:.3f}) between the defect supercells and the DOS "
                f"calculation cell is non-integer, indicating that they correspond to (slightly) "
                f"different unit cell volumes. This can cause quantitative errors of a similar "
                f"relative magnitude in the predicted defect/carrier concentrations!"
            )
        self.multiplicity_scaling = round(ms)

    def _check_required_backend_and_error(self, required_backend: str):
        """
        Check if the attributes needed for the ``required_backend`` backend are
        set, and throw an error message if not.

        Args:
            required_backend (str):
                Backend choice ("doped" or "py-sc-fermi") required.
        """
        raise_error = False
        if required_backend.lower() == "doped":
            if not isinstance(self.defect_thermodynamics.bulk_dos, FermiDos):
                raise_error = True
        elif self._DOS is None:
            raise_error = True

        if raise_error:
            raise RuntimeError(
                f"This function is only supported for the {required_backend} backend, but you are "
                f"using the {self.backend} backend!"
            )

    def _get_fermi_level_and_carriers(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: Optional[dict[str, float]] = None,
        temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the equilibrium Fermi level and carrier concentrations under
        a given chemical potential regime and temperature.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the equilibrium
                Fermi level position. Here, this should be a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in ``self.defect_thermodynamics.el_refs``
                then it is the formal chemical potentials (i.e. relative to the elemental
                reference energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if ``chempots`` was
                provided to ``self.defect_thermodynamics`` in the format generated by
                ``doped``).
                (Default: None)
            temperature (float):
                The temperature at which to solve for the Fermi level
                and carrier concentrations, in Kelvin.
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.

        Returns:
            tuple[float, float, float]: A tuple containing:
                - The Fermi level (float) in eV.
                - The electron concentration (float) in cm^-3.
                - The hole concentration (float) in cm^-3.
        """
        self._check_required_backend_and_error("doped")
        fermi_level, electrons, holes = self.defect_thermodynamics.get_equilibrium_fermi_level(  # type: ignore
            chempots=single_chempot_dict,
            el_refs=el_refs,
            temperature=temperature,
            return_concs=True,
            effective_dopant_concentration=effective_dopant_concentration,
        )  # use already-set bulk dos
        return fermi_level, electrons, holes

    def _get_single_chempot_dict(
        self, limit: Optional[str] = None, chempots: Optional[dict] = None, el_refs: Optional[dict] = None
    ) -> tuple[dict[str, float], Any]:
        """
        Get the chemical potentials for a single limit (``limit``) from the
        ``chempots`` (or ``self.defect_thermodynamics.chempots``) dictionary.

        Returns a `single` chemical potential dictionary for the specified limit.
        """
        chempots, el_refs = self.defect_thermodynamics._get_chempots(
            chempots, el_refs
        )  # returns self.defect_thermodynamics.chempots if chempots is None
        if chempots is None:
            raise ValueError(
                "No chemical potentials supplied or present in self.defect_thermodynamics.chempots!"
            )
        limit = _parse_limit(chempots, limit)

        if (
            limit not in chempots["limits_wrt_el_refs"]
            and "User Chemical Potentials" not in chempots["limits_wrt_el_refs"]
        ):
            raise ValueError(
                f"Limit '{limit}' not found in the chemical potentials dictionary! You must specify "
                f"an appropriate limit or provide an appropriate chempots dictionary."
            )

        return chempots["limits_wrt_el_refs"][limit or "User Chemical Potentials"], el_refs

    def equilibrium_solve(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: Optional[dict[str, float]] = None,
        temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
        append_chempots: bool = True,
        fixed_defects: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate the Fermi level and defect/carrier concentrations under
        `thermodynamic equilibrium`, given a set of chemical potentials and a
        temperature.

        Typically not intended for direct usage, as the same functionality
        is provided by
        ``DefectThermodynamics.get_equilibrium_concentrations/fermi_level()``.
        Rather this is used internally to provide a unified interface for both
        backends, within the ``scan_...`` functions.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the equilibrium
                Fermi level position and defect/carrier concentrations. Here, this
                should be a dictionary of chemical potentials for a single limit
                (``limit``), in the format: ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in ``self.defect_thermodynamics.el_refs``
                then it is the formal chemical potentials (i.e. relative to the elemental
                reference energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if ``chempots`` was
                provided to ``self.defect_thermodynamics`` in the format generated by
                ``doped``).
                (Default: None)
            temperature (float):
                The temperature at which to solve for defect concentrations, in Kelvin.
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            append_chempots (bool):
                Whether to append the chemical potentials (and effective dopant
                concentration, if provided) to the results ``DataFrame``.
                Default is ``True``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.

        Returns:
        # TODO: Check this output format matches for both backends!
            pd.DataFrame:
                A DataFrame containing the defect and carrier concentrations, as well
                as the self-consistent Fermi level and additional properties such as
                electron and hole concentrations. The columns include:
                    - "Fermi Level": The self-consistent Fermi level in eV.
                    - "Electrons (cm^-3)": The electron concentration.
                    - "Holes (cm^-3)": The hole concentration.
                    - "Temperature": The temperature at which the calculation was performed.
                    - "Dopant (cm^-3)": The dopant concentration, if applicable.
                    - "Defect": The defect type.
                    - "Concentration (cm^-3)": The concentration of the defect in cm^-3.
                Additional columns may include concentrations for specific defects and carriers.

        Returns:
                    pd.DataFrame: A DataFrame containing the calculated defect and carrier concentrations,
                    along with the self-consistent Fermi level. The DataFrame also includes the provided
                    chemical potentials as additional columns.
        """
        py_sc_fermi_required = fixed_defects is not None
        if py_sc_fermi_required and self._DOS is None:
            self._activate_py_sc_fermi_backend(
                error_message="The `fix_charge_states` and `free_defects` options are only supported "
                "for the py-sc-fermi backend"
            )

        if self.backend == "doped" and not py_sc_fermi_required:
            fermi_level, electrons, holes = self._get_fermi_level_and_carriers(
                single_chempot_dict=single_chempot_dict,
                el_refs=el_refs,
                temperature=temperature,
                effective_dopant_concentration=effective_dopant_concentration,
            )
            concentrations = self.defect_thermodynamics.get_equilibrium_concentrations(
                chempots=single_chempot_dict,
                el_refs=el_refs,
                fermi_level=fermi_level,
                temperature=temperature,
                per_charge=False,
                per_site=False,
                skip_formatting=False,
            )
            concentrations = _add_effective_dopant_concentration(
                concentrations, effective_dopant_concentration
            )
            new_columns = {
                "Fermi Level": fermi_level,
                "Electrons (cm^-3)": electrons,
                "Holes (cm^-3)": holes,
                "Temperature": temperature,
            }
            for column, value in new_columns.items():
                concentrations[column] = value
            excluded_columns = ["Defect", "Charge", "Charge State Population"]
            for column in concentrations.columns.difference(excluded_columns):
                concentrations[column] = concentrations[column].astype(float)

        else:  # py-sc-fermi backend:
            defect_system = self._generate_defect_system(
                single_chempot_dict=single_chempot_dict,
                el_refs=el_refs,
                temperature=temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                fixed_defects=fixed_defects,
            )

            with np.errstate(all="ignore"):
                conc_dict = defect_system.concentration_dict()
            data = []

            for k, v in conc_dict.items():
                if k not in ["Fermi Energy", "n0", "p0", "Dopant"]:
                    row = {
                        "Temperature": defect_system.temperature,
                        "Fermi Level": conc_dict["Fermi Energy"],
                        "Holes (cm^-3)": conc_dict["p0"],
                        "Electrons (cm^-3)": conc_dict["n0"],
                    }
                    if "Dopant" in conc_dict:
                        row["Dopant (cm^-3)"] = conc_dict["Dopant"]
                    row.update({"Defect": k, "Concentration (cm^-3)": v})
                    data.append(row)

            concentrations = pd.DataFrame(data)
            concentrations = concentrations.set_index("Defect", drop=True)

        if append_chempots:
            for key, value in single_chempot_dict.items():
                concentrations[f"μ_{key}"] = value
            if effective_dopant_concentration:
                concentrations["Dopant (cm^-3)"] = effective_dopant_concentration

        return concentrations

    def pseudo_equilibrium_solve(
        self,
        annealing_temperature: float,
        single_chempot_dict: dict[str, float],
        el_refs: Optional[dict[str, float]] = None,
        quenched_temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
        append_chempots: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the self-consistent Fermi level and corresponding
        carrier/defect calculations under pseudo-equilibrium conditions given a
        set of elemental chemical potentials, a quenched temperature, and an
        annealing temperature.

        Typically not intended for direct usage, as the same functionality
        is provided by
        ``DefectThermodynamics.get_quenched_fermi_level_and_concentrations()``.
        Rather this is used internally to provide a unified interface for both
        backends, within the ``scan_...`` functions.

        'Pseudo-equilibrium' here refers to the use of frozen defect and dilute
        limit approximations under the constraint of charge neutrality (i.e.
        constrained equilibrium). According to the 'frozen defect' approximation,
        we typically expect defect concentrations to reach equilibrium during
        annealing/crystal growth (at elevated temperatures), but `not` upon
        quenching (i.e. at room/operating temperature) where we expect kinetic
        inhibition of defect annhiliation and hence non-equilibrium defect
        concentrations / Fermi level.
        Typically this is approximated by computing the equilibrium Fermi level and
        defect concentrations at the annealing temperature, and then assuming the
        total concentration of each defect is fixed to this value, but that the
        relative populations of defect charge states (and the Fermi level) can
        re-equilibrate at the lower (room) temperature. See discussion in
        https://doi.org/10.1039/D3CS00432E (brief),
        https://doi.org/10.1016/j.cpc.2019.06.017 (detailed) and
        ``doped``/``py-sc-fermi`` tutorials for more information.
        In certain cases (such as Li-ion battery materials or extremely slow charge
        capture/emission), these approximations may have to be adjusted such that some
        defects/charge states are considered fixed and some are allowed to
        re-equilibrate (e.g. highly mobile Li vacancies/interstitials). Modelling
        these specific cases can be achieved using the ``free_defects`` and/or
        ``fix_charge_states`` options, as demonstrated in:
        https://doped.readthedocs.io/en/latest/fermisolver_tutorial.html

        This function works by calculating the self-consistent Fermi level and total
        concentration of each defect at the annealing temperature, then fixing the
        total concentrations to these values and re-calculating the self-consistent
        (constrained equilibrium) Fermi level and relative charge state concentrations
        under this constraint at the quenched/operating temperature. If using the
        ``"py-sc-fermi"`` backend, then you can optionally fix the concentrations of
        individual defect `charge states` (rather than fixing total defect concentrations)
        using ``fix_charge_states=True``, and/or optionally specify ``free_defects`` to
        exclude from high-temperature concentration fixing (e.g. highly mobile defects).

        Note that the bulk DOS calculation should be well-converged with respect to
        k-points for accurate Fermi level predictions!

        The degeneracy/multiplicity factor "g" is an important parameter in the defect
        concentration equation and thus Fermi level calculation (see discussion in
        https://doi.org/10.1039/D2FD00043A and https://doi.org/10.1039/D3CS00432E),
        affecting the final concentration by up to 2 orders of magnitude. This factor
        is taken from the product of the ``defect_entry.defect.multiplicity`` and
        ``defect_entry.degeneracy_factors`` attributes.

        Args:
            annealing_temperature (float):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to the
                highest temperature during annealing/synthesis of the material (at
                which we assume equilibrium defect concentrations) within the frozen
                defect approach.
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the
                pseudo-equilibrium Fermi level position and defect/carrier
                concentrations. Here, this should be a dictionary of chemical
                potentials for a single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in ``self.defect_thermodynamics.el_refs``
                then it is the formal chemical potentials (i.e. relative to the elemental
                reference energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if ``chempots`` was
                provided to ``self.defect_thermodynamics`` in the format generated by
                ``doped``).
                (Default: None)
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.
                # TODO: Allow matching of substring
            append_chempots (bool):
                Whether to append the chemical potentials (and effective dopant
                concentration, if provided) to the results ``DataFrame``.
                Default is ``True``.

        Returns: # TODO: Check output format for both backends!
            pd.DataFrame:
                A ``DataFrame`` containing the defect and carrier concentrations
                under pseudo-equilibrium conditions, along with the self-consistent
                Fermi level.
                The columns include:
                    - "Fermi Level": The self-consistent Fermi level.
                    - "Electrons (cm^-3)": The electron concentration.
                    - "Holes (cm^-3)": The hole concentration.
                    - "Annealing Temperature": The annealing temperature.
                    - "Quenched Temperature": The quenched temperature.
                    - "Dopant (cm^-3)": The dopant concentration, if applicable.
                    - "Defect": The defect type.
                    - "Concentration (cm^-3)": The concentration of the defect in cm^-3.

                Additional columns may include concentrations for specific defects
                and other relevant data.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated defect and carrier concentrations
            using the pseudo-equilibrium approach, along with the provided chemical potentials
            as additional columns.
        """
        py_sc_fermi_required = fix_charge_states or free_defects or fixed_defects is not None
        if py_sc_fermi_required and self._DOS is None:
            self._activate_py_sc_fermi_backend(
                error_message="The `fix_charge_states` and `free_defects` options are only supported "
                "for the py-sc-fermi backend"
            )

        if self.backend == "doped" and not py_sc_fermi_required:
            (
                fermi_level,
                electrons,
                holes,
                concentrations,
            ) = self.defect_thermodynamics.get_quenched_fermi_level_and_concentrations(  # type: ignore
                chempots=single_chempot_dict,
                el_refs=el_refs,
                annealing_temperature=annealing_temperature,
                quenched_temperature=quenched_temperature,
                effective_dopant_concentration=effective_dopant_concentration,
            )  # use already-set bulk dos
            concentrations = concentrations.drop(
                columns=[
                    "Charge",
                    "Charge State Population",
                    "Concentration (cm^-3)",
                    "Formation Energy (eV)",
                ],
            )

            new_columns = {
                "Fermi Level": fermi_level,
                "Electrons (cm^-3)": electrons,
                "Holes (cm^-3)": holes,
                "Annealing Temperature": annealing_temperature,
                "Quenched Temperature": quenched_temperature,
            }

            for column, value in new_columns.items():
                concentrations[column] = value

            trimmed_concentrations_sub_duplicates = concentrations.drop_duplicates()
            excluded_columns = ["Defect"]
            for column in concentrations.columns.difference(excluded_columns):
                concentrations[column] = concentrations[column].astype(float)

            concentrations = trimmed_concentrations_sub_duplicates.rename(
                columns={"Total Concentration (cm^-3)": "Concentration (cm^-3)"},
            )
            concentrations = concentrations.set_index("Defect", drop=True)

        else:  # py-sc-fermi
            defect_system = self._generate_annealed_defect_system(
                annealing_temperature=annealing_temperature,
                single_chempot_dict=single_chempot_dict,
                el_refs=el_refs,
                quenched_temperature=quenched_temperature,
                effective_dopant_concentration=effective_dopant_concentration,
                free_defects=free_defects,
                fixed_defects=fixed_defects,
                fix_charge_states=fix_charge_states,
            )

            with np.errstate(all="ignore"):
                conc_dict = defect_system.concentration_dict()

            data = []
            for k, v in conc_dict.items():
                if k not in ["Fermi Energy", "n0", "p0"]:
                    row = {
                        "Annealing Temperature": annealing_temperature,
                        "Quenched Temperature": quenched_temperature,
                        "Fermi Level": conc_dict["Fermi Energy"],
                        "Holes (cm^-3)": conc_dict["p0"],
                        "Electrons (cm^-3)": conc_dict["n0"],
                    }
                    row.update({"Defect": k, "Concentration (cm^-3)": v})
                    if "Dopant" in conc_dict:
                        row["Dopant (cm^-3)"] = conc_dict["Dopant"]
                    row.update({"Defect": k, "Concentration (cm^-3)": v})
                    data.append(row)

            concentrations = pd.DataFrame(data)
            concentrations = concentrations.set_index("Defect", drop=True)

        if append_chempots:
            for key, value in single_chempot_dict.items():
                concentrations[f"μ_{key}"] = value
            if effective_dopant_concentration:
                concentrations["Dopant (cm^-3)"] = effective_dopant_concentration

        return concentrations

    def scan_temperature(
        self,
        annealing_temperature_range: Optional[Union[float, list[float]]] = None,
        quenched_temperature_range: Union[float, list[float]] = 300,
        temperature_range: Union[float, list[float]] = 300,
        chempots: Optional[dict[str, float]] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict[str, float]] = None,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        r"""
        Scan over a range of temperatures and solve for the defect
        concentrations, carrier concentrations, and Fermi level at each
        temperature / annealing & quenched temperature pair.

        If ``annealing_temperature_range`` (and ``quenched_temperature_range``;
        just 300 K by default) are specified, then the frozen defect approximation
        is employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature_range`` is
        specified, then the Fermi level and defect/carrier concentrations are
        calculated assuming thermodynamic equilibrium at each temperature.

        Args:
            annealing_temperature_range (Optional[Union[float, list[float]]]):
                Temperature range in Kelvin at which to calculate the high
                temperature (fixed) total defect concentrations, which should
                correspond to the highest temperature during annealing/synthesis
                of the material (at which we assume equilibrium defect
                concentrations) within the frozen defect approach. Default is ``None``
                (uses ``temperature_range`` under thermodynamic equilibrium).
            quenched_temperature_range (Union[float, list[float]]):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is just 300 K.
            temperature_range (Union[float, list[float]]):
                Temperature range to solve over, under thermodynamic equilibrium
                (if ``annealing_temperature_range`` is not specified).
                Defaults to just 300 K.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus concentrations). If ``None`` (default),
                will use ``self.defect_thermodynamics.chempots`` (= 0 for all
                chemical potentials by default).
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}], ...}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.

                Note that you can also set ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input options)
                to set the default chemical potentials for all calculations.
                (Default: None)
            limit (str):
                The chemical potential limit for which to calculate the Fermi level
                positions and defect/carrier concentrations. Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical potential
                  limit - otherwise will use the first chemical potential limit in the
                  ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.defect_thermodynamics.)chempots["limits"]``
                  dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).

                Note that you can also set ``FermiSolver.defect_thermodynamics.el_refs = ...``
                or ``DefectThermodynamics.el_refs = ...`` (with the same input options)
                to set the default elemental reference energies for all calculations.
                (Default: None)
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations.
        """
        # Ensure temperature ranges are lists:
        if isinstance(temperature_range, float):
            temperature_range = [temperature_range]
        if annealing_temperature_range is not None and isinstance(annealing_temperature_range, float):
            annealing_temperature_range = [annealing_temperature_range]
        if isinstance(quenched_temperature_range, float):
            quenched_temperature_range = [quenched_temperature_range]

        single_chempot_dict, el_refs = self._get_single_chempot_dict(limit, chempots, el_refs)

        if annealing_temperature_range is not None:
            return pd.concat(
                [
                    self.pseudo_equilibrium_solve(
                        single_chempot_dict=single_chempot_dict,
                        el_refs=el_refs,
                        quenched_temperature=quench_temp,
                        annealing_temperature=anneal_temp,
                        effective_dopant_concentration=effective_dopant_concentration,
                        fix_charge_states=fix_charge_states,
                        free_defects=free_defects,
                        fixed_defects=fixed_defects,
                    )
                    for quench_temp, anneal_temp in tqdm(
                        product(quenched_temperature_range, annealing_temperature_range)
                    )
                ]
            )

        return pd.concat(
            [
                self.equilibrium_solve(
                    single_chempot_dict=single_chempot_dict,
                    el_refs=el_refs,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    fixed_defects=fixed_defects,
                )
                for temperature in tqdm(temperature_range)
            ]
        )

    def scan_dopant_concentration(
        self,
        effective_dopant_concentration_range: Union[float, list[float]],
        chempots: Optional[dict[str, float]] = None,
        limit: Optional[str] = None,
        el_refs: Optional[dict[str, float]] = None,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        r"""
        Calculate the defect concentrations under a range of effective dopant
        concentrations.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            effective_dopant_concentration_range (Union[float, list[float]]):
                The range of effective dopant concentrations to solver over.
                This can be a single value or a list of values representing
                different concentrations.
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies (and thus concentrations). If ``None`` (default),
                will use ``self.defect_thermodynamics.chempots`` (= 0 for all
                chemical potentials by default).
                This can have the form of ``{"limits": [{'limit': [chempot_dict]}], ...}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials)) and specific limits (chemical potential
                limits) can then be chosen using ``limit``.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (``limit``), in the format:
                ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases,
                in which case it is the formal chemical potentials (i.e. relative to the
                elemental references) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.

                Note that you can also set ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input options)
                to set the default chemical potentials for all calculations.
                (Default: None)
            limit (str):
                The chemical potential limit for which to calculate the Fermi level
                positions and defect/carrier concentrations. Can be either:

                - ``None``, if ``chempots`` corresponds to a single chemical potential
                  limit - otherwise will use the first chemical potential limit in the
                  ``chempots`` dict.
                - ``"X-rich"/"X-poor"`` where X is an element in the system, in which
                  case the most X-rich/poor limit will be used (e.g. "Li-rich").
                - A key in the ``(self.defect_thermodynamics.)chempots["limits"]``
                  dictionary.

                The latter two options can only be used if ``chempots`` is in the
                ``doped`` format (see chemical potentials tutorial).
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided/present in format generated by ``doped`` (see tutorials).

                Note that you can also set ``FermiSolver.defect_thermodynamics.el_refs = ...``
                or ``DefectThermodynamics.el_refs = ...`` (with the same input options)
                to set the default elemental reference energies for all calculations.
                (Default: None)
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the defect and carrier concentrations for each
                effective dopant concentration. Each row represents the concentrations
                for a different dopant concentration.
        """
        if isinstance(effective_dopant_concentration_range, float):
            effective_dopant_concentration_range = [effective_dopant_concentration_range]

        single_chempot_dict, el_refs = self._get_single_chempot_dict(limit, chempots, el_refs)

        if annealing_temperature is not None:
            return pd.concat(
                [
                    self.pseudo_equilibrium_solve(
                        single_chempot_dict=single_chempot_dict,
                        el_refs=el_refs,
                        quenched_temperature=quenched_temperature,
                        annealing_temperature=annealing_temperature,
                        effective_dopant_concentration=effective_dopant_concentration,
                        fix_charge_states=fix_charge_states,
                        free_defects=free_defects,
                        fixed_defects=fixed_defects,
                    )
                    for effective_dopant_concentration in tqdm(effective_dopant_concentration_range)
                ]
            )

        return pd.concat(
            [
                self.equilibrium_solve(
                    single_chempot_dict=single_chempot_dict,
                    el_refs=el_refs,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    fixed_defects=fixed_defects,
                )
                for effective_dopant_concentration in tqdm(effective_dopant_concentration_range)
            ]
        )

    def interpolate_chempots(
        self,
        n_points: int,
        chempots: Optional[Union[list[dict], dict]] = None,
        limits: Optional[list[str]] = None,
        el_refs: Optional[dict[str, float]] = None,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Interpolate between two sets of chemical potentials and solve for the
        defect concentrations and Fermi level at each interpolated point.
        Chemical potentials can be interpolated between two sets of chemical
        potentials or between two limits.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            n_points (int):
                The number of points to generate between chemical potential
                end points.
            chempots (Optional[list[dict]]):
                The chemical potentials to interpolate between. This can be
                either a list containing two dictionaries, each representing
                a set of chemical potentials for a single limit (in the format:
                ``{element symbol: chemical potential}``) to interpolate between,
                or can be a single chemical potentials dictionary in the
                ``doped`` format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- in which case ``limits`` must be specified to pick the
                end-points to interpolate between.

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.
                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input
                options) to set the default chemical potentials for all calculations.

                If manually specifying chemical potentials with a list of two
                dictionaries, you can also set the ``el_refs`` option with the DFT
                reference energies of the elemental phases if desired, in which case
                it is the formal chemical potentials (i.e. relative to the elemental
                references) that should be given here, otherwise the absolute (DFT)
                chemical potentials should be given.
            limits (Optional[list[str]]):
                The chemical potential limits to interpolate between, as a list
                containing two strings. Each string should be in the format
                ``"X-rich"/"X-poor"``, where X is an element in the system, or
                be a key in the ``(self.defect_thermodynamics.)chempots["limits"]``
                dictionary.

                If not provided, ``chempots`` must be specified as a list of two
                single chemical potential dictionaries for single limits.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``[{element symbol: chemical potential}, ...]``). Unnecessary if
                ``chempots`` is provided/present in format generated by ``doped``
                (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``).

                Note that you can also set ``FermiSolver.defect_thermodynamics.el_refs = ...``
                or ``DefectThermodynamics.el_refs = ...`` (with the same input options)
                to set the default elemental reference energies for all calculations.
                (Default: None)
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the defect and carrier concentrations for
                each interpolated set of chemical potentials. Each row represents the
                concentrations for a different interpolated point.
        """
        if isinstance(chempots, list):  # should be two single chempot dictionaries
            if len(chempots) != 2:
                raise ValueError(
                    f"If `chempots` is a list, it must contain two dictionaries representing the starting "
                    f"and ending chemical potentials. The provided list has {len(chempots)} entries!"
                )
            single_chempot_dict_1, single_chempot_dict_2 = chempots

        else:  # should be a dictionary in the ``doped`` format or ``None``:
            chempots, el_refs = self.defect_thermodynamics._get_chempots(
                chempots, el_refs
            )  # returns self.defect_thermodynamics.chempots if chempots is None
            if chempots is None:
                raise ValueError(
                    "No chemical potentials supplied or present in self.defect_thermodynamics.chempots!"
                )

            if limits is None or len(limits) != 2:
                raise ValueError(
                    f"If `chempots` is not provided as a list, then `limits` must be a list containing "
                    f"two strings representing the chemical potential limits to interpolate between. "
                    f"The provided `limits` is: {limits}."
                )

            assert isinstance(chempots, dict)  # typing
            single_chempot_dict_1, el_refs = self._get_single_chempot_dict(limits[0], chempots, el_refs)
            single_chempot_dict_2, el_refs = self._get_single_chempot_dict(limits[1], chempots, el_refs)

        interpolated_chempots = self._get_interpolated_chempots(
            single_chempot_dict_1, single_chempot_dict_2, n_points
        )

        if annealing_temperature is not None:
            return pd.concat(
                [
                    self.pseudo_equilibrium_solve(
                        single_chempot_dict=single_chempot_dict,
                        el_refs=el_refs,
                        quenched_temperature=quenched_temperature,
                        annealing_temperature=annealing_temperature,
                        effective_dopant_concentration=effective_dopant_concentration,
                        fix_charge_states=fix_charge_states,
                        free_defects=free_defects,
                        fixed_defects=fixed_defects,
                    )
                    for single_chempot_dict in tqdm(interpolated_chempots)
                ]
            )

        return pd.concat(
            [
                self.equilibrium_solve(
                    single_chempot_dict=single_chempot_dict,
                    el_refs=el_refs,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                )
                for single_chempot_dict in tqdm(interpolated_chempots)
            ]
        )

    def _get_interpolated_chempots(
        self,
        chempot_start: dict,
        chempot_end: dict,
        n_points: int,
    ) -> list:
        """
        Generate a set of interpolated chemical potentials between two points.

        Here, these should be dictionaries of chemical potentials for `single`
        limits, in the format: ``{element symbol: chemical potential}``.
        If ``el_refs`` is provided (to the parent function) or set in
        ``self.defect_thermodynamics.el_refs``, then it is the formal chemical
        potentials (i.e. relative to the elemental reference energies) that
        should be used here, otherwise the absolute (DFT) chemical potentials
        should be used.

        Args:
            chempot_start (dict):
                A dictionary representing the starting chemical potentials.
            chempot_end (dict):
                A dictionary representing the ending chemical potentials.
            n_points (int):
                The number of interpolated points to generate, `including`
                the start and end points.

        Returns:
            list:
                A list of dictionaries, where each dictionary contains a `single`
                set of interpolated chemical potentials. The length of the list
                corresponds to `n_points`, and each dictionary corresponds to an
                interpolated state between the starting and ending chemical potentials.
        """
        return [
            {
                key: chempot_start[key] + (chempot_end[key] - chempot_start[key]) * i / (n_points - 1)
                for key in chempot_start
            }
            for i in range(n_points)
        ]

    def scan_chempots(
        self,
        chempots: Union[list[dict[str, float]], dict[str, dict]],
        limits: Optional[list[str]] = None,
        el_refs: Optional[dict[str, float]] = None,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Scan over a range of chemical potentials and solve for the defect
        concentrations and Fermi level at each set of chemical potentials.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            chempots (Optional[Union[list[dict], dict]]):
                The chemical potentials to scan over. This can be either a list
                containing dictionaries of a set of chemical potentials for a
                `single` limit (in the format: ``{element symbol: chemical potential}``),
                or can be a single chemical potentials dictionary in the
                ``doped`` format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- in which case ``limits`` can be specified to pick the chemical
                potential limits to scan over (otherwise scans over all limits in the
                ``chempots`` dictionary).

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.
                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...``
                or ``DefectThermodynamics.chempots = ...`` (with the same input
                options) to set the default chemical potentials for all calculations.

                If manually specifying chemical potentials with a list of dictionaries,
                you can also set the ``el_refs`` option with the DFT reference energies
                of the elemental phases if desired, in which case it is the formal
                chemical potentials (i.e. relative to the elemental references) that
                should be given here, otherwise the absolute (DFT) chemical potentials
                should be given.
            limits (Optional[list[str]]):
                The chemical potential limits to scan over, as a list of strings, if
                ``chempots`` was provided / is present in the ``doped`` format. Each
                string should be in the format ``"X-rich"/"X-poor"``, where X is an
                element in the system, or be a key in the
                ``(self.defect_thermodynamics.)chempots["limits"]`` dictionary.

                If ``None`` (default) and ``chempots`` is in the ``doped`` format
                (rather than a list of single chemical potential limits), will scan
                over all limits in the ``chempots`` dictionary.
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``[{element symbol: chemical potential}, ...]``). Unnecessary if
                ``chempots`` is provided/present in format generated by ``doped``
                (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``).

                Note that you can also set ``FermiSolver.defect_thermodynamics.el_refs = ...``
                or ``DefectThermodynamics.el_refs = ...`` (with the same input options)
                to set the default elemental reference energies for all calculations.
                (Default: None)
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing defect and carrier concentrations for each
                set of chemical potentials. Each row corresponds to a different set
                of chemical potentials.
        """
        if isinstance(chempots, dict):  # should be a dictionary in the ``doped`` format or ``None``:
            chempots, el_refs = self.defect_thermodynamics._get_chempots(
                chempots, el_refs
            )  # returns self.defect_thermodynamics.chempots if chempots is None
            if chempots is None:
                raise ValueError(
                    "No chemical potentials supplied or present in self.defect_thermodynamics.chempots!"
                )
            assert isinstance(chempots, dict)  # typing
            chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)

            if limits is None:
                limits = list(chempots["limits_wrt_el_refs"].keys())
            elif not isinstance(limits, list):
                raise ValueError(
                    "`limits` must be either a list of limits (as strings) or `None` for `scan_chempots`!"
                )

            chempots = [self._get_single_chempot_dict(limit, chempots, el_refs)[0] for limit in limits]

        if annealing_temperature is not None:
            return pd.concat(
                [
                    self.pseudo_equilibrium_solve(
                        single_chempot_dict=single_chempot_dict,
                        quenched_temperature=quenched_temperature,
                        annealing_temperature=annealing_temperature,
                        effective_dopant_concentration=effective_dopant_concentration,
                        fix_charge_states=fix_charge_states,
                        free_defects=free_defects,
                        fixed_defects=fixed_defects,
                    )
                    for single_chempot_dict in chempots
                ]
            )

        return pd.concat(
            [
                self.equilibrium_solve(
                    single_chempot_dict=single_chempot_dict,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    fixed_defects=fixed_defects,
                )
                for single_chempot_dict in chempots
            ]
        )

    def scan_chemical_potential_grid(
        self,
        chempots: Optional[dict] = None,
        n_points: int = 10,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        r"""
        Given a ``doped``-formatted chemical potential dictionary, generate a
        ``ChemicalPotentialGrid`` object and calculate the Fermi level
        positions and defect/carrier concentrations at the grid points.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the ``doped``
                format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials).

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.

                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...`` or
                ``DefectThermodynamics.chempots = ...`` to set the default chemical
                potentials for all calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update the
                elemental reference energies for any reason.
            n_points (int):
                The number of points to generate along each axis of the grid.
                The actual number of grid points may be less, as points outside
                the convex hull are excluded. Default is 10.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame: A DataFrame containing the Fermi level solutions at the grid
            points, based on the provided chemical potentials and conditions.
        """
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)
        grid = ChemicalPotentialGrid(chempots).get_grid(n_points)

        if annealing_temperature is not None:
            return pd.concat(
                [
                    self.pseudo_equilibrium_solve(
                        single_chempot_dict={
                            k.replace("μ_", ""): v for k, v in chempot_series.to_dict().items()
                        },
                        el_refs=el_refs,
                        annealing_temperature=annealing_temperature,
                        quenched_temperature=quenched_temperature,
                        effective_dopant_concentration=effective_dopant_concentration,
                        fix_charge_states=fix_charge_states,
                        free_defects=free_defects,
                        fixed_defects=fixed_defects,
                    )
                    for _idx, chempot_series in tqdm(grid.iterrows())
                ]
            )

        return pd.concat(
            [
                self.equilibrium_solve(
                    single_chempot_dict={
                        k.replace("μ_", ""): v for k, v in chempot_series.to_dict().items()
                    },
                    el_refs=el_refs,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                )
                for _idx, chempot_series in tqdm(grid.iterrows())
            ]
        )

    def _parse_and_check_grid_like_chempots(self, chempots: Optional[dict] = None) -> tuple[dict, dict]:
        r"""
        Parse a dictionary of chemical potentials for the chemical potential
        scanning functions, checking that it is in the correct format.

        Args:
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the ``doped``
                format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials).

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.

                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...`` or
                ``DefectThermodynamics.chempots = ...`` to set the default chemical
                potentials for all calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update the
                elemental reference energies for any reason.

        Returns:
            dict:
                The chemical potentials in the correct format, along with the
                elemental reference energies.
        """
        # returns self.defect_thermodynamics.chempots if chempots is None:
        chempots, el_refs = self.defect_thermodynamics._get_chempots(chempots)

        if chempots is None:
            raise ValueError(
                "No chemical potentials supplied or present in `self.defect_thermodynamics.chempots`!"
            )
        if len(chempots["limits"]) == 1:
            raise ValueError(
                "Only one chemical potential limit is present in "
                "`chempots`/`self.defect_thermodynamics.chempots`, which makes no sense for a chemical "
                "potential grid scan (with `scan_chemical_potential_grid`/`min_max_X`/`scan_chempots`)!"
            )

        return chempots, el_refs

    def min_max_X(
        self,
        target: str,
        min_or_max: str = "max",
        chempots: Optional[dict] = None,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        tolerance: float = 0.01,
        n_points: int = 10,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        r"""
        Search for the chemical potentials that minimize or maximize a target
        variable, such as electron concentration, within a specified tolerance.

        This function iterates over a grid of chemical potentials and "zooms in" on
        the chemical potential that either minimizes or maximizes the target variable.
        The process continues until the change in the target variable is less than
        the specified tolerance.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            target (str):
                The target variable to minimize or maximize, e.g., "Electrons (cm^-3)".
                # TODO: Allow this to match a substring of the column name.
            min_or_max (str):
                Specify whether to "minimize" ("min") or "maximize" ("max"; default)
                the target variable.
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the ``doped``
                format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials).

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.

                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...`` or
                ``DefectThermodynamics.chempots = ...`` to set the default chemical
                potentials for all calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update the
                elemental reference energies for any reason.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            tolerance (float):
                The convergence criterion for the target variable. The search
                stops when the target value change is less than this value.
                Defaults to ``0.01``.
            n_points (int):
                The number of points to generate along each axis of the grid for
                the initial search. Defaults to ``10``.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the results of the minimization or
                maximization process, including the optimal chemical potentials and
                the corresponding values of the target variable.

        Raises:
            ValueError:
                If neither ``chempots`` nor ``self.chempots`` is provided, or if
                ``min_or_max`` is not ``"minimize"/"min"`` or ``"maximize"/"max"``.
        """
        # Determine the dimensionality of the chemical potential space
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)
        n_chempots = len(el_refs)

        # Call the appropriate method based on dimensionality
        if n_chempots == 2:
            return self._min_max_X_line(
                target,
                min_or_max,
                chempots,
                annealing_temperature,
                quenched_temperature,
                temperature,
                tolerance,
                n_points,
                effective_dopant_concentration,
                fix_charge_states,
                fixed_defects,
                free_defects,
            )
        return self._min_max_X_grid(
            target,
            min_or_max,
            chempots,
            annealing_temperature,
            quenched_temperature,
            temperature,
            tolerance,
            n_points,
            effective_dopant_concentration,
            fix_charge_states,
            fixed_defects,
            free_defects,
        )

    def _min_max_X_line(
        self,
        target: str,
        min_or_max: str,
        chempots: dict,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        tolerance: float = 0.01,
        n_points: int = 100,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        r"""
        Search for the chemical potentials that minimize or maximize a target
        variable, such as electron concentration, within a specified tolerance.

        This function iterates over a grid of chemical potentials and "zooms in" on
        the chemical potential that either minimizes or maximizes the target variable.
        The process continues until the change in the target variable is less than
        the specified tolerance.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            target (str):
                The target variable to minimize or maximize, e.g., "Electrons (cm^-3)".
                # TODO: Allow this to match a substring of the column name.
            min_or_max (str):
                Specify whether to "minimize" ("min") or "maximize" ("max"; default)
                the target variable.
            chempots (dict):
                Dictionary of chemical potentials to scan over, in the ``doped``
                format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials).

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.

                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...`` or
                ``DefectThermodynamics.chempots = ...`` to set the default chemical
                potentials for all calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update the
                elemental reference energies for any reason.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            tolerance (float):
                The convergence criterion for the target variable. The search
                stops when the target value change is less than this value.
                Defaults to ``0.01``.
            n_points (int):
                The number of points to generate along each axis of the grid for
                the initial search. Defaults to ``10``.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the results of the minimization or
                maximization process, including the optimal chemical potentials and
                the corresponding values of the target variable.

        Raises:
            ValueError:
                If neither ``chempots`` nor ``self.chempots`` is provided, or if
                ``min_or_max`` is not ``"minimize"/"min"`` or ``"maximize"/"max"``.
        """
        el_refs = chempots["elemental_refs"]
        chempots_label = list(el_refs.keys())  # Assuming 1D space, focus on one label.
        chempots_labels = [f"μ_{label}" for label in chempots_label]

        # Initial line scan: Rich to Poor limit
        rich = self._get_single_chempot_dict(f"{chempots_label[0]}-rich")
        poor = self._get_single_chempot_dict(f"{chempots_label[0]}-poor")
        starting_line = self._get_interpolated_chempots(rich[0], poor[0], n_points)

        previous_value = None

        while True:
            # Calculate results based on the given temperature conditions
            if annealing_temperature is not None:
                results_df = pd.concat(
                    [
                        self.pseudo_equilibrium_solve(
                            single_chempot_dict={
                                k.replace("μ_", ""): v for k, v in chempot_series.items()
                            },
                            el_refs=el_refs,
                            annealing_temperature=annealing_temperature,
                            quenched_temperature=quenched_temperature,
                            effective_dopant_concentration=effective_dopant_concentration,
                            fix_charge_states=fix_charge_states,
                            free_defects=free_defects,
                            fixed_defects=fixed_defects,
                        )
                        for chempot_series in tqdm(starting_line)
                    ]
                )
            else:
                results_df = pd.concat(
                    [
                        self.equilibrium_solve(
                            single_chempot_dict={
                                k.replace("μ_", ""): v for k, v in chempot_series.items()
                            },
                            el_refs=el_refs,
                            temperature=temperature,
                            effective_dopant_concentration=effective_dopant_concentration,
                        )
                        for chempot_series in tqdm(starting_line)
                    ]
                )

            if target in results_df.columns:
                if "min" in min_or_max:
                    target_chempot = results_df[results_df[target] == results_df[target].min()][
                        chempots_labels
                    ]
                    target_dataframe = results_df[results_df[target] == results_df[target].min()]
                elif "max" in min_or_max:
                    target_chempot = results_df[results_df[target] == results_df[target].max()][
                        chempots_labels
                    ]
                    target_dataframe = results_df[results_df[target] == results_df[target].max()]
                current_value = (
                    results_df[target].min() if "min" in min_or_max else results_df[target].max()
                )

            else:
                # Filter the DataFrame for the specific defect
                filtered_df = results_df[results_df.index == target]
                # Find the row where "Concentration (cm^-3)" is at its minimum or maximum
                if "min" in min_or_max:
                    min_value = filtered_df["Concentration (cm^-3)"].min()
                    target_chempot = results_df.loc[results_df["Concentration (cm^-3)"] == min_value][
                        chempots_labels
                    ]
                    target_dataframe = results_df[
                        results_df[chempots_labels].eq(target_chempot.iloc[0]).all(axis=1)
                    ]

                elif "max" in min_or_max:
                    max_value = filtered_df["Concentration (cm^-3)"].max()
                    target_chempot = results_df.loc[results_df["Concentration (cm^-3)"] == max_value][
                        chempots_labels
                    ]
                    target_dataframe = results_df[
                        results_df[chempots_labels].eq(target_chempot.iloc[0]).all(axis=1)
                    ]
                current_value = (
                    filtered_df["Concentration (cm^-3)"].min()
                    if "min" in min_or_max
                    else filtered_df["Concentration (cm^-3)"].max()
                )

            # Check if the change in the target value is less than the tolerance
            if (
                previous_value is not None
                and abs((current_value - previous_value) / previous_value) < tolerance
            ):
                break
            previous_value = current_value
            target_chempot = target_chempot.drop_duplicates(ignore_index=True)

            starting_line = self._get_interpolated_chempots(
                chempot_start={
                    k: v - 0.1 for k, v in target_chempot.iloc[0].items() if k in chempots_labels
                },
                chempot_end={
                    k: v + 0.1 for k, v in target_chempot.iloc[0].items() if k in chempots_labels
                },
                n_points=100,
            )

            return target_dataframe

    def _min_max_X_grid(
        self,
        target: str,
        min_or_max: str = "max",
        chempots: Optional[dict] = None,
        annealing_temperature: Optional[float] = None,
        quenched_temperature: float = 300,
        temperature: float = 300,
        tolerance: float = 0.01,
        n_points: int = 10,
        effective_dopant_concentration: Optional[float] = None,
        fix_charge_states: bool = False,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        r"""
        Search for the chemical potentials that minimize or maximize a target
        variable, such as electron concentration, within a specified tolerance.

        This function iterates over a grid of chemical potentials and "zooms in" on
        the chemical potential that either minimizes or maximizes the target variable.
        The process continues until the change in the target variable is less than
        the specified tolerance.

        If ``annealing_temperature`` (and ``quenched_temperature``; 300 K by
        default) are specified, then the frozen defect approximation is
        employed, whereby total defect concentrations are calculated at the
        elevated annealing temperature, then fixed at these values (unless
        ``free_defects`` or ``fix_charge_states`` are specified) and the Fermi
        level and relative charge state populations are recalculated at the
        quenched temperature. Otherwise, if only ``temperature`` is specified,
        then the Fermi level and defect/carrier concentrations are calculated
        assuming thermodynamic equilibrium at that temperature.

        Args:
            target (str):
                The target variable to minimize or maximize, e.g., "Electrons (cm^-3)".
                # TODO: Allow this to match a substring of the column name.
            min_or_max (str):
                Specify whether to "minimize" ("min") or "maximize" ("max"; default)
                the target variable.
            chempots (Optional[dict]):
                Dictionary of chemical potentials to scan over, in the ``doped``
                format (i.e. ``{"limits": [{'limit': [chempot_dict]}], ...}``)
                -- the format generated by ``doped``\'s chemical potential parsing
                functions (see tutorials).

                If ``None`` (default), will use ``self.defect_thermodynamics.chempots``.

                Note that you can also set
                ``FermiSolver.defect_thermodynamics.chempots = ...`` or
                ``DefectThermodynamics.chempots = ...`` to set the default chemical
                potentials for all calculations, and you can set
                ``FermiSolver.defect_thermodynamics.el_refs = ...`` or
                ``DefectThermodynamics.el_refs = ...`` if you want to update the
                elemental reference energies for any reason.
            annealing_temperature (Optional[float]):
                Temperature in Kelvin at which to calculate the high temperature
                (fixed) total defect concentrations, which should correspond to
                the highest temperature during annealing/synthesis of the material
                (at which we assume equilibrium defect concentrations) within the
                frozen defect approach. Default is ``None`` (uses ``temperature``
                under thermodynamic equilibrium).
            quenched_temperature (float):
                Temperature in Kelvin at which to calculate the self-consistent
                (constrained equilibrium) Fermi level and carrier concentrations,
                given the fixed total concentrations, which should correspond to
                operating temperature of the material (typically room temperature).
                Default is 300 K.
            temperature (float):
                The temperature at which to solve for defect concentrations
                and Fermi level, under thermodynamic equilibrium (if
                ``annealing_temperature`` is not specified).
                Defaults to 300 K.
            tolerance (float):
                The convergence criterion for the target variable. The search
                stops when the target value change is less than this value.
                Defaults to ``0.01``.
            n_points (int):
                The number of points to generate along each axis of the grid for
                the initial search. Defaults to ``10``.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or
                impurity in the material. This value is included in the charge
                neutrality condition to analyze the Fermi level and doping
                response under hypothetical doping conditions.
                A positive value corresponds to donor doping, while a negative
                value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no additional extrinsic
                dopant.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``) upon quenching. Not expected to be
                physically sensible in most cases. Defaults to ``False``.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected
                to be "frozen-in" upon quenching. Defaults to ``None``.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the results of the minimization or
                maximization process, including the optimal chemical potentials and
                the corresponding values of the target variable.

        Raises:
            ValueError:
                If neither ``chempots`` nor ``self.chempots`` is provided, or if
                ``min_or_max`` is not ``"minimize"/"min"`` or ``"maximize"/"max"``.
        """
        chempots, el_refs = self._parse_and_check_grid_like_chempots(chempots)
        starting_grid = ChemicalPotentialGrid(chempots)
        current_vertices = starting_grid.vertices
        chempots_labels = list(current_vertices.columns)
        previous_value = None

        while True:
            if annealing_temperature is not None:
                results_df = pd.concat(
                    [
                        self.pseudo_equilibrium_solve(
                            single_chempot_dict={
                                k.replace("μ_", ""): v for k, v in chempot_series.to_dict().items()
                            },
                            el_refs=el_refs,
                            annealing_temperature=annealing_temperature,
                            quenched_temperature=quenched_temperature,
                            effective_dopant_concentration=effective_dopant_concentration,
                            fix_charge_states=fix_charge_states,
                            free_defects=free_defects,
                            fixed_defects=fixed_defects,
                        )
                        for _idx, chempot_series in tqdm(starting_grid.get_grid(n_points).iterrows())
                    ]
                )
            else:
                results_df = pd.concat(
                    [
                        self.equilibrium_solve(
                            single_chempot_dict={
                                k.replace("μ_", ""): v for k, v in chempot_series.to_dict().items()
                            },
                            el_refs=el_refs,
                            temperature=temperature,
                            effective_dopant_concentration=effective_dopant_concentration,
                        )
                        for _idx, chempot_series in tqdm(starting_grid.get_grid(n_points).iterrows())
                    ]
                )

            # Find chemical potentials value where target is lowest or highest
            if target in results_df.columns:
                if "min" in min_or_max:
                    target_chempot = results_df[results_df[target] == results_df[target].min()][
                        chempots_labels
                    ]
                    target_dataframe = results_df[results_df[target] == results_df[target].min()]
                elif "max" in min_or_max:
                    target_chempot = results_df[results_df[target] == results_df[target].max()][
                        chempots_labels
                    ]
                    target_dataframe = results_df[results_df[target] == results_df[target].max()]
                current_value = (
                    results_df[target].min() if "min" in min_or_max else results_df[target].max()
                )

            else:
                # Filter the DataFrame for the specific defect
                filtered_df = results_df[results_df.index == target]
                # Find the row where "Concentration (cm^-3)" is at its minimum or maximum
                if "min" in min_or_max:
                    min_value = filtered_df["Concentration (cm^-3)"].min()
                    target_chempot = results_df.loc[results_df["Concentration (cm^-3)"] == min_value][
                        chempots_labels
                    ]
                    target_dataframe = results_df[
                        results_df[chempots_labels].eq(target_chempot.iloc[0]).all(axis=1)
                    ]

                elif "max" in min_or_max:
                    max_value = filtered_df["Concentration (cm^-3)"].max()
                    target_chempot = results_df.loc[results_df["Concentration (cm^-3)"] == max_value][
                        chempots_labels
                    ]
                    target_dataframe = results_df[
                        results_df[chempots_labels].eq(target_chempot.iloc[0]).all(axis=1)
                    ]
                current_value = (
                    filtered_df["Concentration (cm^-3)"].min()
                    if "min" in min_or_max
                    else filtered_df["Concentration (cm^-3)"].max()
                )

            # Check if the change in the target value is less than the tolerance
            if (
                previous_value is not None
                and abs((current_value - previous_value) / previous_value) < tolerance
            ):
                break
            previous_value = current_value
            target_chempot = target_chempot.drop_duplicates(ignore_index=True)

            new_vertices = [  # get midpoint between current vertices and target_chempot
                (current_vertices + row[1]) / 2 for row in target_chempot.iterrows()
            ]
            # Generate a new grid around the target_chempot that
            # does not go outside the bounds of the starting grid
            new_vertices_df = pd.DataFrame(new_vertices[0], columns=chempots_labels)
            starting_grid = ChemicalPotentialGrid(new_vertices_df.to_dict("index"))

        return target_dataframe

    def _generate_dopant_for_py_sc_fermi(self, effective_dopant_concentration: float) -> "DefectSpecies":
        """
        Generate a dopant defect charge state object, for use with the ``py-sc-
        fermi`` functions.

        This method creates a defect charge state object representing an
        arbitrary dopant or impurity in the material, used to include in the
        charge neutrality condition and analyze the Fermi level/doping
        response under hypothetical doping conditions.

        Args:
            effective_dopant_concentration (float):
                The fixed concentration of the dopant or impurity in the
                material, specified in cm^-3. A positive value indicates
                donor doping (positive defect charge state), while a negative
                value indicates acceptor doping (negative defect charge state).

        Returns:
            DefectSpecies:
                An instance of the ``DefectSpecies`` class, representing the
                generated dopant with the specified charge state and concentration.

        Raises:
            ValueError:
                If ``effective_dopant_concentration`` is zero or if there is an
                issue with generating the dopant.
        """
        self._check_required_backend_and_error("py-sc-fermi")
        assert self._DefectChargeState
        assert self._DefectSpecies
        dopant = self._DefectChargeState(
            charge=np.sign(effective_dopant_concentration),
            fixed_concentration=abs(effective_dopant_concentration) / 1e24 * self.volume,
            degeneracy=1,
        )
        return self._DefectSpecies(
            nsites=1, charge_states={np.sign(effective_dopant_concentration): dopant}, name="Dopant"
        )

    def _generate_defect_system(
        self,
        single_chempot_dict: dict[str, float],
        el_refs: Optional[dict[str, float]] = None,
        temperature: float = 300,
        effective_dopant_concentration: Optional[float] = None,
        fixed_defects: Optional[dict[str, float]] = None,
    ) -> "DefectSystem":
        """
        Generates a ``DefectSystem`` object from ``self.defect_thermodynamics``
        and a set of chemical potentials.

        This method constructs a ``DefectSystem`` object, which encompasses all
        relevant defect species and their properties under the given conditions,
        including temperature, chemical potentials, and an optional dopant
        concentration.

        Args:
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the equilibrium
                Fermi level position and defect/carrier concentrations. Here, this
                should be a dictionary of chemical potentials for a single limit
                (``limit``), in the format: ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in ``self.defect_thermodynamics.el_refs``
                then it is the formal chemical potentials (i.e. relative to the elemental
                reference energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if ``chempots`` was
                provided to ``self.defect_thermodynamics`` in the format generated by
                ``doped``).
                (Default: None)
            temperature (float):
                The temperature at which to perform the calculations, in Kelvin.
                Defaults to 300 K.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant or impurity
                in the material. This value is included in the charge neutrality
                condition to analyze the Fermi level and doping response under
                hypothetical doping conditions. A positive value corresponds to donor
                doping, while a negative value corresponds to acceptor doping.
                Defaults to ``None``, corresponding to no extrinsic dopant.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.

        Returns:
            DefectSystem:
                An initialized ``DefectSystem`` object, containing the defect species
                with their charge states, formation energies, and degeneracies, as well
                as the density of states (DOS), volume, and temperature of the system.
        """
        self._check_required_backend_and_error("py-sc-fermi")
        assert self._DefectSpecies
        assert self._DefectSystem
        entries = list(self.defect_thermodynamics.defect_entries.values())
        entries = sorted(entries, key=lambda x: x.name)
        labels = {_get_label_and_charge(entry.name)[0] for entry in entries}
        defect_species: dict[str, Any] = {
            label: {"charge_states": {}, "nsites": None, "name": label} for label in labels
        }
        dft_chempots = _get_dft_chempots(single_chempot_dict, el_refs)

        for entry in entries:
            label, charge = _get_label_and_charge(entry.name)
            defect_species[label]["nsites"] = entry.defect.multiplicity / self.multiplicity_scaling

            formation_energy = self.defect_thermodynamics.get_formation_energy(
                entry, chempots=dft_chempots, fermi_level=0
            )
            total_degeneracy = np.prod(list(entry.degeneracy_factors.values()))
            defect_species[label]["charge_states"][charge] = {
                "charge": charge,
                "energy": formation_energy,
                "degeneracy": total_degeneracy,
            }

        all_defect_species = [self._DefectSpecies.from_dict(v) for k, v in defect_species.items()]
        if effective_dopant_concentration is not None:
            dopant = self._generate_dopant_for_py_sc_fermi(effective_dopant_concentration)
            all_defect_species.append(dopant)

        if fixed_defects is not None:
            for k, v in fixed_defects.items():
                if k.split("_")[-1].strip("+-").isdigit():
                    q = int(k.split("_")[-1])
                    defect_name = "_".join(k.split("_")[:-1])
                    next(d for d in all_defect_species if d.name == defect_name).charge_states[
                        q
                    ].fix_concentration(v / 1e24 * self.volume)
                else:
                    next(d for d in all_defect_species if d.name == k).fix_concentration(
                        v / 1e24 * self.volume
                    )

        return self._DefectSystem(
            defect_species=all_defect_species,
            dos=self.py_sc_fermi_dos,
            volume=self.volume,
            temperature=temperature,
            convergence_tolerance=1e-20,
        )

    def _generate_annealed_defect_system(
        self,
        annealing_temperature: float,
        single_chempot_dict: dict[str, float],
        el_refs: Optional[dict[str, float]] = None,
        quenched_temperature: float = 300,
        fix_charge_states: bool = False,
        effective_dopant_concentration: Optional[float] = None,
        fixed_defects: Optional[dict[str, float]] = None,
        free_defects: Optional[list[str]] = None,
    ) -> "DefectSystem":
        """
        Generate a ``py-sc-fermi`` ``DefectSystem`` object that has defect
        concentrations fixed to the values determined at a high temperature
        (``annealing_temperature``), and then set to a lower temperature
        (``quenched_temperature``).

        This method creates a defect system where defect concentrations are
        initially calculated at an annealing temperature and then "frozen" as
        the system is cooled to a lower quenched temperature. It can optionally
        fix the concentrations of individual defect charge states or allow
        charge states to vary while keeping total defect concentrations fixed.

        Args:
            annealing_temperature (float):
                The higher temperature (in Kelvin) at which the system is annealed
                to set initial defect concentrations.
            single_chempot_dict (dict[str, float]):
                Dictionary of chemical potentials to use for calculating the equilibrium
                Fermi level position and defect/carrier concentrations. Here, this
                should be a dictionary of chemical potentials for a single limit
                (``limit``), in the format: ``{element symbol: chemical potential}``.
                If ``el_refs`` is provided or set in ``self.defect_thermodynamics.el_refs``
                then it is the formal chemical potentials (i.e. relative to the elemental
                reference energies) that should be given here, otherwise the absolute
                (DFT) chemical potentials should be given.
            el_refs (dict[str, float]):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}``. Unnecessary if
                ``self.defect_thermodynamics.el_refs`` is set (i.e. if ``chempots`` was
                provided to ``self.defect_thermodynamics`` in the format generated by
                ``doped``).
                (Default: None)
            quenched_temperature (float):
                The lower temperature (in Kelvin) to which the system is quenched.
                Defaults to 300 K.
            fix_charge_states (bool):
                Whether to fix the concentrations of individual defect charge states
                (``True``) or allow charge states to vary while keeping total defect
                concentrations fixed (``False``). Defaults to ``False``.
            effective_dopant_concentration (Optional[float]):
                The fixed concentration (in cm^-3) of an arbitrary dopant/impurity in
                the material. A positive value indicates donor doping, while a negative
                value indicates acceptor doping. Defaults to ``None``, corresponding to
                no extrinsic dopant.
            fixed_defects (Optional[dict[str, float]]):
                A dictionary of defect concentrations to fix at the quenched temperature,
                in the format: ``{defect_name: concentration}``. Concentrations should be
                given in cm^-3. The this can be used to fix the concentrations of specific
                defects regardless of the chemical potentials, or anneal-quench procedure
                (e.g. to simulate the effect of a fixed impurity concentration).
                If a fixed-concentration of a specific charge state is desired,
                the defect name should be formatted as ``"defect_name_charge"``.
                I.e. ``"v_O_+2"`` for a doubly positively charged oxygen vacancy.
                Defaults to ``None``.
            free_defects (Optional[list[str]]):
                A list of defects to be excluded from high-temperature concentration
                fixing, useful for highly mobile defects that are not expected to be
                "frozen-in." Defaults to ``None``.

        Returns:
            DefectSystem: A low-temperature defect system (`quenched_temperature`)
            with defect concentrations fixed to high-temperature (`annealing_temperature`) values.
        """
        self._check_required_backend_and_error("py-sc-fermi")
        if free_defects is None:
            free_defects = []

        defect_system = self._generate_defect_system(
            single_chempot_dict=single_chempot_dict,  # chempots handled in _generate_defect_system()
            el_refs=el_refs,
            temperature=annealing_temperature,
            effective_dopant_concentration=effective_dopant_concentration,
        )
        initial_conc_dict = defect_system.concentration_dict()  # concentrations at initial temperature

        # Exclude the free_defects, carrier concentrations and Fermi level from fixing
        all_free_defects = ["Fermi Energy", "n0", "p0"]
        all_free_defects.extend(free_defects)

        # Get the fixed concentrations of non-exceptional defects
        decomposed_conc_dict = defect_system.concentration_dict(decomposed=True)
        additional_data = {}
        for k, v in decomposed_conc_dict.items():
            if k not in all_free_defects:
                for k1, v1 in v.items():
                    additional_data[k + "_" + str(k1)] = v1
        initial_conc_dict.update(additional_data)

        fixed_concs = {k: v for k, v in initial_conc_dict.items() if k not in all_free_defects}

        # Apply the fixed concentrations
        for defect_species in defect_system.defect_species:
            if fix_charge_states:
                for k, v in defect_species.charge_states.items():
                    key = f"{defect_species.name}_{int(k)}"
                    if key in list(fixed_concs.keys()):
                        v.fix_concentration(fixed_concs[key] / 1e24 * defect_system.volume)

            elif defect_species.name in fixed_concs:
                defect_species.fix_concentration(
                    fixed_concs[defect_species.name] / 1e24 * defect_system.volume
                )

        fixed_charge_state_names = []
        fixed_species_names = []
        if fixed_defects is not None:
            for k, v in fixed_defects.items():
                if k.split("_")[-1].strip("+-").isdigit():
                    q = int(k.split("_")[-1])
                    defect_name = "_".join(k.split("_")[:-1])
                    next(d for d in defect_system.defect_species if d.name == defect_name).charge_states[
                        q
                    ].fix_concentration(v / 1e24 * self.volume)
                    fixed_charge_state_names.append(k)
                    if v > fixed_concs[defect_name]:
                        warnings.warn(
                            f"""Fixed concentration of {k} ({v}) is higher than
                            the total concentration of ({fixed_concs[defect_name]})
                            at the annealing temperature. Adjusting the total
                            concentration of {defect_name} to {v}. Check that
                            this is the behavior you expect."""
                        )
                        defect_system.defect_species_by_name(defect_name).fix_concentration(
                            v / 1e24 * self.volume
                        )
                else:
                    next(d for d in defect_system.defect_species if d.name == k).fix_concentration(
                        v / 1e24 * self.volume
                    )
                fixed_species_names.append(k)
            target_system = deepcopy(defect_system)
            target_system.temperature = quenched_temperature
            return target_system

        target_system = deepcopy(defect_system)
        target_system.temperature = quenched_temperature
        return target_system


class ChemicalPotentialGrid:
    """
    A class to represent a grid in chemical potential space and to perform
    operations such as generating a grid within the convex hull of given
    vertices (chemical potential limits).

    This class provides methods for handling and manipulating chemical
    potential data, including the creation of a grid that spans a specified
    chemical potential space.
    """

    def __init__(self, chempots: dict[str, Any]):
        r"""
        Initializes the ``ChemicalPotentialGrid`` with chemical potential data.

        This constructor takes a dictionary of chemical potentials and sets up
        the initial vertices of the grid.

        Args:
            chempots (dict):
                Dictionary of chemical potentials for the grid. This can have
                the form of ``{"limits": [{'limit': [chempot_dict]}], ...}``
                (the format generated by ``doped``\'s chemical potential parsing
                functions), or alternatively can be a dictionary of the form
                ``{'limit': [chempot_dict, ...]}`` (i.e. matching the format of
                ``chempots["limits_wrt_el_refs"]`` from the ``doped`` ``chempots``
                dict) where the keys are the limit names (e.g. "Cd-CdTe", "Cd-rich"
                etc) and the values are dictionaries of a single chemical potential
                limit in the format: ``{element symbol: chemical potential}``.

                If ``chempots`` in the ``doped`` format is supplied, then the
                chemical potentials `with respect to the elemental reference
                energies` will be used (i.e. ``chempots["limits_wrt_el_refs"]``)!
        """
        unformatted_chempots_dict = chempots.get("limits_wrt_el_refs", chempots)
        test_elt = Element("H")
        formatted_chempots_dict = {
            limit: {
                f"μ_{k}" if test_elt.is_valid_symbol(k) else k: v
                for (k, v) in unformatted_chempots_subdict.items()
            }
            for limit, unformatted_chempots_subdict in unformatted_chempots_dict.items()
        }

        self.vertices = pd.DataFrame.from_dict(formatted_chempots_dict, orient="index")

    def get_grid(self, n_points: int = 100) -> pd.DataFrame:
        """
        Generates a grid within the convex hull of the vertices and
        interpolates the dependent variable values.

        This method creates a grid of points that spans the chemical potential
        space defined by the vertices. It ensures that the generated points lie
        within the convex hull of the provided vertices and interpolates the
        chemical potential values at these points.

        Args:
            n_points (int):
                The number of points to generate along each axis of the grid.
                Note that this may not always be the final number of points in the grid,
                as points lying outside the convex hull are excluded.
                Default is 100.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the points within the convex hull,
                along with their corresponding interpolated chemical potential values.
                Each row represents a point in the grid with associated chemical
                potential values.
        """
        return self.grid_from_dataframe(self.vertices, n_points)

    @staticmethod
    def grid_from_dataframe(mu_dataframe: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
        r"""
        Generates a grid within the convex hull of the vertices.

        This method creates a grid of points within the convex hull
        defined by the input ``DataFrame``\.
        It interpolates the values of chemical potentials over this
        grid, ensuring that all generated points lie within the convex
        hull of the given vertices.

        Args:
            mu_dataframe (pd.DataFrame):
                A ``DataFrame`` containing the chemical potential data,
                with the last column representing the dependent variable
                and the preceding columns representing the independent
                variables.
            n_points (int):
                The number of points to generate along each axis of the
                grid. Note that this may not always be the final number
                of points in the grid, as points lying outside the convex
                hull are excluded. Defaults to 100.

        Returns:
            pd.DataFrame:
                A ``DataFrame`` containing the points within the convex
                hull along with their corresponding interpolated values of
                the dependent variable. Each row represents a point in the
                grid.
        """
        dependent_variable = mu_dataframe.columns[-1]
        dependent_var = mu_dataframe[dependent_variable].to_numpy()
        independent_vars = mu_dataframe.drop(columns=dependent_variable)

        n_dims = independent_vars.shape[1]  # Get the number of independent variables (dimensions)

        # Get the convex hull of the vertices
        hull = ConvexHull(independent_vars.values)

        # Create a dense grid that covers the entire range of the vertices
        grid_ranges = [
            np.linspace(independent_vars.iloc[:, i].min(), independent_vars.iloc[:, i].max(), n_points)
            for i in range(n_dims)
        ]
        grid = np.meshgrid(*grid_ranges, indexing="ij")  # Create N-dimensional grid
        grid_points = np.vstack([g.ravel() for g in grid]).T  # Flatten the grid to points

        # Delaunay triangulation to get points inside the convex hull
        delaunay = Delaunay(hull.points[hull.vertices])
        inside_hull = delaunay.find_simplex(grid_points) >= 0
        points_inside = grid_points[inside_hull]

        # Interpolate the values to get the dependent chemical potential
        values_inside = griddata(independent_vars.values, dependent_var, points_inside, method="linear")

        # Combine points with their corresponding interpolated values
        grid_with_values = np.hstack((points_inside, values_inside.reshape(-1, 1)))

        # Add vertices to the grid
        grid_with_values = np.vstack((grid_with_values, mu_dataframe.to_numpy()))

        return pd.DataFrame(
            grid_with_values,
            columns=[*list(independent_vars.columns), dependent_variable],
        )
