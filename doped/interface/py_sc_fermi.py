from dataclasses import dataclass
from doped.utils.legacy_pmg.thermodynamics import DefectPhaseDiagram
from copy import deepcopy
from typing import List, Dict, Union
import pandas as pd
import numpy as np

try:
    from py_sc_fermi.dos import DOS
    from py_sc_fermi.defect_system import DefectSystem
    from py_sc_fermi.defect_species import DefectSpecies
except ImportError:
    raise ImportError(
        "Please install py-sc-fermi via `pip install py-sc-fermi` to use this functionality."
    )


def _get_label_and_charge(name: str) -> tuple:
    """Extracts the label and charge from a defect name string.

    Args:
        name (str): Name of the defect.

    Returns:
        tuple: A tuple containing the label and charge.
    """
    last_underscore = name.rfind("_")
    label = name[:last_underscore] if last_underscore != -1 else name
    charge = name[last_underscore + 1 :] if last_underscore != -1 else None
    return label, int(charge)


@dataclass
class FermiSolver:
    defect_phase_diagram: DefectPhaseDiagram
    bulk_vasprun: str
    base_defect_system: DefectSystem = None  # Placeholder for the base DefectSystem

    def __post_init__(self) -> None:
        """Initializes additional attributes after dataclass instantiation."""
        self.bulk_dos = DOS.from_vasprun(self.bulk_vasprun)
        self.volume = self.defect_phase_diagram.entries[0].defect.structure.volume

    def initialize_base_defect_system(self, temperature: float, chemical_potentials: dict) -> None:
        self.base_defect_system = self._generate_defect_system(temperature, chemical_potentials)

    def _defect_picker(self, name: str):  # Type hint should represent the actual type of the return value
        """
        Get defect entry from doped name.

        Args:
        name (str): Name of the defect to find in the defect phase diagram entries.

        Returns:
        Entry: The first entry matching the given name.
        """
        defect = next(
            e for e in self.defect_phase_diagram.entries if e.name == name
        )  # Use next with a generator to get the first matching entry
        return defect

    def _generate_defect_system(self, temperature: float, chemical_potentials: dict) -> DefectSystem:
        """Generates a DefectSystem object from the DefectPhaseDiagram and a set of chemical potentials.

        Args:
            temperature (float): Temperature in K.
            chemical_potentials (dict): Chemical potentials for the elements.

        Returns:
            DefectSystem: The initialized DefectSystem.
        """
        entries = sorted(self.defect_phase_diagram.entries, key=lambda x: x.name)
        labels = {_get_label_and_charge(entry.name)[0] for entry in entries}
        defect_species = {label: {"charge_states": {}, "nsites": None, "name": label} for label in labels}

        for entry in entries:
            label, charge = _get_label_and_charge(entry.name)
            defect_species[label]["nsites"] = 1  # Assumes one site for the defect

            formation_energy = self.defect_phase_diagram._formation_energy(
                entry, chemical_potentials=chemical_potentials, fermi_level=0
            )

            defect_species[label]["charge_states"][charge] = {
                "charge": charge,
                "energy": formation_energy,
                "degeneracy": entry.defect.multiplicity,
            }

        all_defect_species = [DefectSpecies.from_dict(v) for k, v in defect_species.items()]

        return DefectSystem(
            defect_species=all_defect_species,
            dos=self.bulk_dos,
            volume=self.volume,
            temperature=temperature,
        )

    def update_defect_system_energies(self, chemical_potentials: dict) -> DefectSystem:
        """Updates the energies of the DefectSystem with a new set of chemical potentials."""
        if self.base_defect_system is None:
            raise ValueError("Base defect system has not been initialized.")

        updated_defect_system = deepcopy(self.base_defect_system)

        for ds in updated_defect_system.defect_species:
            for charge, charge_state in ds.charge_states.items():
                full_name = f"{ds.name}_{charge}"
                corresponding_entry = self._defect_picker(full_name)

                new_energy = self.defect_phase_diagram._formation_energy(
                    corresponding_entry, chemical_potentials=chemical_potentials, fermi_level=0
                )
                charge_state._energy = new_energy

        return updated_defect_system

    def update_defect_system_temperature(self, temperature: float) -> DefectSystem:
        """Updates the temperature of the defect system.
        
        Args:
            temperature (float): temperature to recalculate the defect system at.

        Returns:
            DefectSystem: a new DefectSystem object with the updated temperature.
        """
        if self.base_defect_system is None:
            raise ValueError("Base defect system has not been initialized.")

        updated_defect_system = deepcopy(self.base_defect_system)
        updated_defect_system.temperature = temperature
        return updated_defect_system

    def scan_temperature_and_save(
        self, 
        temp_range: List[float], 
        fix_concentration_temp: float = None, 
        level: str = "DefectChargeState", 
        exceptions: List[str] = None
    ) -> pd.DataFrame:
        """
        Scans a range of temperatures and saves the concentration_dict() to a DataFrame.

        Args:
            temp_range (List[float]): List of temperatures to scan.
            fix_concentration_temp (float, optional): The temperature at which to fix concentrations. Defaults to None.
            level (str, optional): The level at which to fix concentrations. Defaults to "DefectChargeState".
            exceptions (List[str], optional): List of species or charge states to exclude from fixing. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different temperatures in long format.
        """
        if self.base_defect_system is None:
            raise ValueError("Base defect system has not been initialized.")

        concentration_data = []
        fermi_level_data = []

        for temp in temp_range:
            if fix_concentration_temp:
                conc_dict = self.fix_concentrations_and_recalculate(
                    initial_temp=fix_concentration_temp, 
                    target_temp=temp, 
                    level=level, 
                    exceptions=exceptions
                )
            else:
                updated_system = self.update_defect_system_temperature(temp)
                conc_dict = updated_system.concentration_dict()

            for name, value in conc_dict.items():
                if name != "Fermi Energy":
                    concentration_data.append(
                        {"Temperature": temp, "Defect": name, "Concentration": value}
                    )
                else:
                    fermi_level_data.append({"Temperature": temp, "Fermi Level": value})

        concentration_df = pd.DataFrame(concentration_data)
        fermi_level_df = pd.DataFrame(fermi_level_data)
        return pd.merge(concentration_df, fermi_level_df, on="Temperature")

    def scan_chemical_potentials_and_save(
        self, chem_pot_start: dict, chem_pot_end: dict, n_points: int
    ) -> pd.DataFrame:
        """Scans a range of chemical potentials and saves the concentration_dict() to a DataFrame.

        Args:
            chem_pot_start (dict): Dictionary of starting chemical potentials.
            chem_pot_end (dict): Dictionary of ending chemical potentials.
            n_points (int): Number of points to scan.
        
        Returns:
            pd.DataFrame: DataFrame containing concentrations at different chemical potentials in long format.
        """
        if self.base_defect_system is None:
            raise ValueError("Base defect system has not been initialized.")

        concentration_data = []
        fermi_level_data = []

        for t in np.linspace(0, 1, n_points):
            chem_pot_interpolated = {
                element: (1 - t) * start + t * end
                for element, (start, end) in zip(
                    chem_pot_start.keys(), zip(chem_pot_start.values(), chem_pot_end.values())
                )
            }

            updated_system = self.update_defect_system_energies(chem_pot_interpolated)
            conc_dict = updated_system.concentration_dict()

            for name, value in conc_dict.items():
                if name != "Fermi Energy":
                    concentration_data.append(
                        {
                            "Interpolation_Parameter": t,
                            "Defect": name,
                            "Concentration": value,
                            **chem_pot_interpolated,
                        }
                    )
                else:
                    fermi_level_data.append(
                        {"Interpolation_Parameter": t, "Fermi Level": value, **chem_pot_interpolated}
                    )

        concentration_df = pd.DataFrame(concentration_data)
        fermi_level_df = pd.DataFrame(fermi_level_data)

        return pd.merge(
            concentration_df,
            fermi_level_df,
            on=["Interpolation_Parameter"] + list(chem_pot_interpolated.keys()),
        )

    def fix_concentrations_and_recalculate(
        self, 
        initial_temp: float, 
        target_temp: float, 
        level: str, 
        exceptions: List[str] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]: 
        """
        Calculates concentrations at an initial temperature, fixes them,
        and then recalculates at a target temperature.

        Args:
            initial_temp (float): The initial temperature.
            target_temp (float): The target temperature.
            level (str): 'DefectSpecies' or 'DefectChargeState' to specify the level to fix.
            exceptions (List[str]): List of DefectSpecies or DefectChargeStates to be excluded from fixing.

        Returns:
            dict: The concentrations at the target temperature with fixed concentrations.
        """
        if level not in ["DefectSpecies", "DefectChargeState"]:
            raise ValueError("level must be either 'DefectSpecies' or 'DefectChargeState'")

        # Calculate concentrations at initial temperature
        initial_system = self.update_defect_system_temperature(initial_temp)
        initial_conc_dict = initial_system.concentration_dict()

        # Exclude the exceptions and Fermi energy from fixing
        if exceptions is None:
            exceptions = []
        exceptions.extend(["Fermi Energy", "n0", "p0"])

        decomposed_conc_dict = initial_system.concentration_dict(decomposed=True)
        additional_data = {}
        for k, v in decomposed_conc_dict.items():
            if k not in exceptions:
                for k1, v1 in v.items():
                    additional_data.update({k + "_" + str(k1): v1})

        initial_conc_dict.update(additional_data)
        fixed_concs = {k: v for k, v in initial_conc_dict.items() if k not in exceptions}

        # Update the system at target temperature
        target_system = self.update_defect_system_temperature(target_temp)

        # Apply the fixed concentrations
        for defect_species in target_system.defect_species:
            if level == "DefectSpecies":
                if defect_species.name in fixed_concs:
                    defect_species.fix_concentration(fixed_concs[defect_species.name] / 1e24 * self.volume)
            elif level == "DefectChargeState":
                for k, v in defect_species.charge_states.items():
                    key = f"{defect_species.name}_{int(k)}"
                    if key in [k for k in fixed_concs.keys()]:
                        v.fix_concentration(fixed_concs[key] / 1e24 * self.volume)

        # Recalculate the concentrations
        recalculated_conc_dict = target_system.concentration_dict()

        return recalculated_conc_dict
