from typing import Optional, Union, List
from warnings import warn
from itertools import product
from copy import deepcopy
import numpy as np
from joblib import Parallel, delayed

import pandas as pd
from monty.json import MSONable
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.io.vasp import Vasprun

from doped.thermodynamics import DefectThermodynamics
from doped.utils.parsing import get_neutral_nelect_from_vasprun

try:
    from py_sc_fermi.defect_system import DefectSystem
    from py_sc_fermi.defect_species import DefectSpecies
    from py_sc_fermi.dos import DOS
except ImportError:
    warn("py-sc-fermi not installed, will only be able to use doped backend")


def _get_label_and_charge(name: str) -> tuple:
    """
    Extracts the label and charge from a defect name string.

    Args:
        name (str): Name of the defect.

    Returns:
        tuple: A tuple containing the label and charge.
    """
    last_underscore = name.rfind("_")
    label = name[:last_underscore] if last_underscore != -1 else name
    charge = name[last_underscore + 1 :] if last_underscore != -1 else None
    return label, int(charge)


class FermiSolver(MSONable):
    """class to manage the generation of DefectSystems from DefectPhaseDiagrams"""

    def __init__(self, defect_thermodynamics: DefectThermodynamics, bulk_dos_vr: str):
        self.defect_thermodynamics = defect_thermodynamics
        self.bulk_dos = bulk_dos_vr

    def equilibrium_solve(self, chempots: dict[str, float], temperature: float) -> None:
        """not implemented in the base class, implemented in the derived class"""
        raise NotImplementedError(
            """This method is implemented in the derived class, 
            use FermiSolverDoped or FermiSolverPyScFermi instead."""
        )

    def pseudo_equilibrium_solve(
        self, chempots: dict[str, float], quenched_temperature: float, annealing_temperature: float
    ) -> None:
        """not implemented in the base class, implemented in the derived class"""
        raise NotImplementedError(
            """This method is implemented in the derived class, 
            use FermiSolverDoped or FermiSolverPyScFermi instead."""
        )

    def scan_temperature(
        self, chempots: dict[str, float], temperature_range: List[float], processes: int = 1, **kwargs
    ) -> pd.DataFrame:
        """Scan over a range of temperatures to calculate the defect concentrations
        under thermodynamic equilibrium given a set of elemental chemical potentials.

        Args:
            chempots (dict): chemical potentials to solve
            temperature_range (List): range of temperatures to solve over
            processes (int): number of processes to use for parallel processing

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
            and the self consistent Fermi energy
        """
        all_data = Parallel(n_jobs=processes)(
            delayed(self.equilibrium_solve)(chempots, temperature, **kwargs)
            for temperature in temperature_range
        )
        return pd.concat(all_data)

    def scan_anneal_and_quench(
        self,
        chempots: dict[str, float],
        quenching_temperatures: Union[float, List[float]],
        annealing_temperatures: Union[float, List[float]],
        processes: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Scan over a range of quenching and annealing temperatures to calculate the defect concentrations
        under pseudo equilibrium given a set of elemental chemical potentials.

        Args:
            chempots (dict): chemical potentials to solve
            quenching_temperatures (float or List): quenching temperatures to solve over
            annealing_temperatures (float or List): annealing temperatures to solve over
            processes (int): number of processes to use for parallel processing

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
              and the self consistent Fermi energy
        """
        quenching_temperatures = list(quenching_temperatures)
        annealing_temperatures = list(annealing_temperatures)

        all_data = Parallel(n_jobs=processes)(
            delayed(self.pseudo_equilibrium_solve)(
                chempots,
                quenched_temperature=quenched_temperature,
                annealing_temperature=anneal_temperature,
                **kwargs,
            )
            for quenched_temperature, anneal_temperature in product(
                quenching_temperatures, annealing_temperatures
            )
        )
        return pd.concat(all_data)

    def _solve_and_append_chemical_potentials(
        self, chempots: dict[str, float], temperature: float, **kwargs
    ):
        """Solve for the defect concentrations at a given temperature and chemical potentials.

        Args:
            chempots (dict): Chemical potentials for the elements.
            temperature (float): Temperature in K.

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
              and the self consistent Fermi energy
        """
        df = self.equilibrium_solve(chempots, temperature)
        for key, value in chempots.items():
            df[key] = value
        return df

    def _solve_and_append_chemical_potentials_pseudo(
        self, chem_pot, quench_temperature, anneal_temperature
    ):
        df = self.pseudo_equilibrium_solve(chem_pot, quench_temperature, anneal_temperature)
        for key, value in chem_pot.items():
            df[key] = value
        return df

    def interpolate_chemical_potentials(
        self,
        chem_pot_start: dict,
        chem_pot_end: dict,
        n_points: int,
        temperature: float = 300.0,
        annealing_temperatures: Optional[Union[float, List]] = None,
        quenching_temperatures: Optional[Union[float, List]] = None,
        processes: int = 1,
    ) -> pd.DataFrame:
        """Linearly interpolates between two dictionaries of chemical potentials
        and returns a DataFrame containing all the concentration data. We can
        then optionally saves the data to a csv.

        Args:
            chem_pot_start (dict): Dictionary of starting chemical potentials.
            chem_pot_end (dict): Dictionary of ending chemical potentials.
            n_points (int): Number of points in the linear interpolation
            temperature (float): temperature for final the Fermi level solver
            annealing_temperatures (float or List): annealing temperatures to solve over
            quenching_temperatures (float or List): quenching temperatures to solve over
            processes (int): number of processes to use for parallel processing

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different
            chemical potentials in long format.
        """

        interpolated_chem_pots, interpolation = self._get_interpolated_chempots(
            chem_pot_start, chem_pot_end, n_points
        )
        concentrations = []

        if annealing_temperatures is None and quenching_temperatures is None:
            concentrations = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials)(chem_pot, temperature)
                for chem_pot in interpolated_chem_pots
            )
            concentrations = pd.concat(concentrations)
            return concentrations

        elif annealing_temperatures is not None and quenching_temperatures is not None:
            if not isinstance(annealing_temperatures, list):
                annealing_temperatures = [annealing_temperatures]
            if not isinstance(quenching_temperatures, list):
                quenching_temperatures = [quenching_temperatures]

            concentrations = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials_pseudo)(
                    chem_pot, quench_temperature, anneal_temperature
                )
                for quench_temperature, anneal_temperature in product(
                    quenching_temperatures, annealing_temperatures
                )
                for chem_pot in interpolated_chem_pots
            )
            concentrations = pd.concat(concentrations)
            return concentrations

        else:
            raise ValueError("You must specify both annealing and quenching temperatures, or neither.")

    def _get_interpolated_chempots(
        self, start: dict[str, float], end: dict[str, float], n_points: float
    ) -> tuple[list[dict], np.ndarray]:
        """Linearly interpolates between two dictionaries of chemical potentials
        and returns a List of dictionaries of the interpolated chemical potentials.

        Args:
            start (dict): Dictionary of starting chemical potentials.
            end (dict): Dictionary of ending chemical potentials.
            n_points (int): Number of points in the linear interpolation

        Returns:
            List[dict]: List of dictionaries of the interpolated chemical potentials.
        """
        interpolated_chem_pots = []
        interpolation = np.linspace(0, 1, n_points)

        for t in interpolation:
            chem_pot_interpolated = {
                element: (1 - t) * start + t * end
                for element, (start, end) in zip(start.keys(), zip(start.values(), end.values()))
            }
            interpolated_chem_pots.append(chem_pot_interpolated)

        return interpolated_chem_pots, interpolation


class FermiSolverDoped(FermiSolver):
    """class to to calculate the self-consistent Fermi level using the doped backend

    Args:
        defect_thermodynamics (DefectThermodynamics): The DefectThermodynamics object
          to use for the calculations.
        bulk_dos_vr (str): The path to the vasprun.xml file containing the bulk DOS.
    """

    def __init__(self, defect_thermodynamics: DefectThermodynamics, bulk_dos_vr: str):
        """initialize the FermiSolverDoped object"""
        super().__init__(defect_thermodynamics, bulk_dos_vr)
        bulk_dos_vr = Vasprun(self.bulk_dos)
        self.bulk_dos = FermiDos(
            bulk_dos_vr.complete_dos, nelecs=get_neutral_nelect_from_vasprun(bulk_dos_vr)
        )

    def _get_fermi_level_and_carriers(
        self, chempots: dict[str, float], temperature: float
    ) -> tuple[float, float, float]:
        """Calculate the Fermi level and carrier concentrations under a given
        chemical potential regime and temperature.

        Args:
            chempots (dict[str, float]) chemical potentials to solve at
            temperature (float): temperature in to solve at

        Returns:
            float: fermi level
            float: electron concentration
            float: hole concentration
        """
        fermi_level, electrons, holes = self.defect_thermodynamics.get_equilibrium_fermi_level(
            bulk_dos_vr=self.bulk_dos,
            chempots=chempots,
            facet=None,
            temperature=temperature,
            return_concs=True,
        )
        return fermi_level, electrons, holes

    def equilibrium_solve(self, chempots: dict[str, float], temperature: float) -> pd.DataFrame:
        """calculate the defect concentrations under thermodynamic equilibrium
        given a set of elemental chemical potentials and a temperature.

        Args:
            chempots (dict): chemical potentials to solve
            temperature (float): temperature to solve

        returns:
           pd.DataFrame: DataFrame containing defect and carrier concentrations
              and the self consistent Fermi energy
        """
        fermi_level, electrons, holes = self._get_fermi_level_and_carriers(
            chempots=chempots, temperature=temperature
        )
        concentrations = self.defect_thermodynamics.get_equilibrium_concentrations(
            chempots=chempots,
            fermi_level=fermi_level,
            temperature=temperature,
            per_charge=False,
            per_site=False,
            skip_formatting=False,
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
        return concentrations

    def pseudo_equilibrium_solve(
        self, chempots: dict[str, float], quenched_temperature: float, annealing_temperature: float
    ) -> pd.DataFrame:
        """calculate the defect concentrations under pseudo equilibrium given a set of elemental
        chemical potentials, a quenching temperature and an annealing temperature.

        Args:
            chempots (dict): chemical potentials to solve
            quenched_temperature (float): temperature to quench to
            annealing_temperature (float): temperature to anneal at

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
                and the self consistent Fermi energy
        """
        (
            fermi_level,
            electrons,
            holes,
            concentrations,
        ) = self.defect_thermodynamics.get_quenched_fermi_level_and_concentrations(
            bulk_dos_vr=self.bulk_dos,
            chempots=chempots,
            facet=None,
            annealing_temperature=annealing_temperature,
            quenched_temperature=quenched_temperature,
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

        concentrations.drop(
            columns=[
                "Charge",
                "Charge State Population",
                "Concentration (cm^-3)",
                "Formation Energy (eV)",
            ],
            inplace=True,
        )
        concentrations.drop_duplicates(inplace=True)
        excluded_columns = ["Defect"]
        for column in concentrations.columns.difference(excluded_columns):
            concentrations[column] = concentrations[column].astype(float)

        concentrations.rename(
            columns={"Total Concentration (cm^-3)": "Concentration (cm^-3)"}, inplace=True
        )
        concentrations.set_index("Defect", inplace=True, drop=True)
        return concentrations


class FermiSolverPyScFermi(FermiSolver):
    """class to to calculate the self-consistent Fermi level using the py-sc-fermi backend

    This class is a wrapper around the py-sc-fermi package.

    Args:
        defect_thermodynamics (DefectThermodynamics): The DefectThermodynamics object
          to use for the calculations.
        bulk_dos_vr (str): The path to the vasprun.xml file containing the bulk DOS.
    """

    def __init__(
        self, defect_thermodynamics: DefectThermodynamics, bulk_dos_vr: str, multiplicity_scaling=1.0
    ):
        """initialize the FermiSolverPyScFermi object"""
        super().__init__(defect_thermodynamics, bulk_dos_vr)
        vr = Vasprun(self.bulk_dos)
        self.bulk_dos = DOS.from_vasprun(self.bulk_dos, nelect=vr.parameters["NELECT"])
        self.volume = vr.final_structure.volume
        self.multiplicity_scaling = multiplicity_scaling

    def _generate_defect_system(
        self, temperature: float, chemical_potentials: dict[str, float]
    ) -> DefectSystem:
        """
        Generates a DefectSystem object from the DefectPhaseDiagram and a set
        of chemical potentials.

        Args:
            temperature (float): Temperature in K.
            chemical_potentials (dict): Chemical potentials for the elements.

        Returns:
            DefectSystem: The initialized DefectSystem.
        """
        entries = sorted(self.defect_thermodynamics.defect_entries, key=lambda x: x.name)
        labels = {_get_label_and_charge(entry.name)[0] for entry in entries}
        defect_species = {label: {"charge_states": {}, "nsites": None, "name": label} for label in labels}

        for entry in entries:
            label, charge = _get_label_and_charge(entry.name)
            defect_species[label]["nsites"] = entry.defect.multiplicity / self.multiplicity_scaling

            formation_energy = self.defect_thermodynamics.get_formation_energy(
                entry, chempots=chemical_potentials, fermi_level=0
            )
            total_degeneracy = np.prod(list(entry.degeneracy_factors.values()))
            defect_species[label]["charge_states"][charge] = {
                "charge": charge,
                "energy": formation_energy,
                "degeneracy": total_degeneracy,
            }

        all_defect_species = [DefectSpecies.from_dict(v) for k, v in defect_species.items()]

        return DefectSystem(
            defect_species=all_defect_species,
            dos=self.bulk_dos,
            volume=self.volume,
            temperature=temperature,
            convergence_tolerance=1e-20,
        )

    def defect_system_from_chemical_potentials(
        self, chemical_potentials: dict[str, float], temperature: float = 300.0
    ) -> DefectSystem:
        """
        Generates a DefectSystem object from a set of chemical potentials.

        Args:
            chemical_potentials (dict): Chemical potentials for the elements.
            temperature (float): Temperature in K.

        Returns:
            DefectSystem: The initialized DefectSystem.
        """
        defect_system = self._generate_defect_system(
            temperature=temperature, chemical_potentials=chemical_potentials
        )
        return defect_system

    def equilibrium_solve(self, chempots: dict[str, float], temperature: float) -> pd.DataFrame:
        """
        Solve for the defect concentrations at a given temperature and chemical potentials.

        Args:
            chempots (dict): Chemical potentials for the elements.
            temperature (float): Temperature in K.

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
            and the self consistent Fermi energy
        """
        defect_system = self.defect_system_from_chemical_potentials(
            chemical_potentials=chempots, temperature=temperature
        )
        conc_dict = defect_system.concentration_dict()
        data = []

        for k, v in conc_dict.items():
            if k not in ["Fermi Energy", "n0", "p0"]:
                row = {
                    "Temperature": defect_system.temperature,
                    "Fermi Level": conc_dict["Fermi Energy"],
                    "Holes (cm^-3)": conc_dict["p0"],
                    "Electrons (cm^-3)": conc_dict["n0"],
                }
                row.update({"Defect": k, "Concentration (cm^-3)": v})
                data.append(row)

        df = pd.DataFrame(data)
        df.set_index("Defect", inplace=True, drop=True)
        return df

    def pseudo_equilibrium_solve(
        self, chempots: dict[str, float], quenched_temperature: float, annealing_temperature: float
    ):
        """
        Solve for the defect concentrations at a given quenching and annealing temperature
        and chemical potentials.

        Args:
            chempots (dict): Chemical potentials for the elements.
            quenched_temperature (float): Temperature to quench to.
            annealing_temperature (float): Temperature to anneal at.

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
                and the self consistent Fermi energy
        """
        defect_system = self.generate_annealed_defect_system(
            chemical_potentials=chempots,
            quenched_temperature=quenched_temperature,
            annealing_temperature=annealing_temperature,
        )
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
                data.append(row)

        df = pd.DataFrame(data)
        df.set_index("Defect", inplace=True, drop=True)
        return df

    def generate_annealed_defect_system(
        self,
        chemical_potentials,
        quenched_temperature,
        annealing_temperature,
        fix_defect_species=True,
        exceptions=[],
    ) -> DefectSystem:
        """generate a py-sc-fermi DefectSystem object that has defect concentrations
        fixed to the values determined at a high temperature (annealing_temperature),
        and then set to a lower temperature (quenching_temperature)

        Args:
            mu (Dict[str, float]): set of chemical potentials used to generate the
              DefectSystem
            annealing_temperature (float): high temperature at which to generate
              the initial DefectSystem for concentration fixing
            target_temperature (float, optional): The low temperature at which to
              generate the final DefectSystem. Defaults to 300.0.
            fix_defect_species (bool): if annealing temperature is set, this sets
              whether the concentrations of the py-sc-fermi DefectSpecies are fixed
              to their high temperature values, or whether the DefectChargeStates
              are fixed. If in doubt, leave as default value.
            exceptions (List): if annealing_temperature is set, this lists the
              defects to be excluded from the high-temperature concentration fixing
              this may be important in systems with highly mobile defects that are
              not expected to be "frozen-in"

        Returns:
            DefectSystem: a low temperature defect system (target_temperature),
              with defect concentrations fixed to high temperature
              (annealing_temperature) values.

        """

        # Calculate concentrations at initial temperature
        defect_system = self.defect_system_from_chemical_potentials(
            chemical_potentials=chemical_potentials, temperature=annealing_temperature
        )
        initial_conc_dict = defect_system.concentration_dict()

        # Exclude the exceptions, carrier concentrations and
        # Fermi energy from fixing
        all_exceptions = ["Fermi Energy", "n0", "p0"]
        all_exceptions.extend(exceptions)

        # Get the fixed concentrations of non-exceptional defects
        decomposed_conc_dict = defect_system.concentration_dict(decomposed=True)
        additional_data = {}
        for k, v in decomposed_conc_dict.items():
            if k not in all_exceptions:
                for k1, v1 in v.items():
                    additional_data.update({k + "_" + str(k1): v1})
        initial_conc_dict.update(additional_data)

        fixed_concs = {k: v for k, v in initial_conc_dict.items() if k not in all_exceptions}

        # Apply the fixed concentrations
        for defect_species in defect_system.defect_species:
            if fix_defect_species == True:
                if defect_species.name in fixed_concs:
                    defect_species.fix_concentration(
                        fixed_concs[defect_species.name] / 1e24 * defect_system.volume
                    )
            elif fix_defect_species == False:
                for k, v in defect_species.charge_states.items():
                    key = f"{defect_species.name}_{int(k)}"
                    if key in List(fixed_concs.keys()):
                        v.fix_concentration(fixed_concs[key] / 1e24 * defect_system.volume)

        target_system = deepcopy(defect_system)
        target_system.temperature = quenched_temperature
        return target_system
