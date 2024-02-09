from typing import Optional
from warnings import warn
from itertools import product

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


class FermiSolver(MSONable):
    """class to manage the generation of DefectSystems from DefectPhaseDiagrams"""

    def __init__(self, defect_thermodynamics: DefectThermodynamics, dos: str):
        self.defect_thermodynamics = defect_thermodynamics
        bulk_dos_vr = Vasprun(dos)
        self.bulk_dos = FermiDos(
            bulk_dos_vr.complete_dos, nelecs=get_neutral_nelect_from_vasprun(bulk_dos_vr)
        )
        # self.volume = self.bulk_dos.volume

    def equilibrium_solve(
        self, chempots, temperature, per_charge=False, per_site=False, skip_formatting=False
    ):
        """calculate the defect concentrations at equilibrium"""
        raise NotImplementedError("This method should be implemented in the derived class")

    def pseudo_equilibrium_solve(self, chempots, quenched_temperature, annealing_temperature):
        """calculate the defect concentrations at equilibrium"""
        raise NotImplementedError("This method should be implemented in the derived class")

    def scan_temperature(self, chempots, temperature_range, **kwargs):
        """Scan over a range of temperatures to calculate the defect concentrations"""
        all_data = []
        for temperature in temperature_range:
            data = self.equilibrium_solve(chempots, temperature, **kwargs)
            all_data.append(data)
        return pd.concat(all_data)

    def scan_anneal_and_quench(self, chempots, quenching_temperatures, annealing_temperatures, **kwargs):
        """Scan over a range of temperatures to calculate the defect concentrations"""
        all_data = []
        for quenched_temperature, anneal_temperature in product(
            quenching_temperatures, annealing_temperatures
        ):
            data = self.pseudo_equilibrium_solve(
                chempots,
                quenched_temperature=quenched_temperature,
                annealing_temperature=anneal_temperature,
                **kwargs,
            )
            all_data.append(data)
        return pd.concat(all_data)


class FermiSolverDoped(FermiSolver):
    """class to to calculate the Fermi level using the doped backend"""

    def __init__(self, defect_thermodynamics: DefectThermodynamics, dos: str):
        super().__init__(defect_thermodynamics, dos)

    def get_fermi_level_and_carriers(self, chempots, temperature):
        """Calculate the Fermi level using the doped backend"""
        fermi_level, electrons, holes = self.defect_thermodynamics.get_equilibrium_fermi_level(
            bulk_dos_vr=self.bulk_dos,
            chempots=chempots,
            facet=None,
            temperature=temperature,
            return_concs=True,
        )
        return fermi_level, electrons, holes

    def equilibrium_solve(
        self, chempots, temperature, per_charge=False, per_site=False, skip_formatting=False
    ):
        """Calculate the Fermi level using the doped backend"""
        fermi_level, electrons, holes = self.get_fermi_level_and_carriers(chempots, temperature)
        concentrations = self.defect_thermodynamics.get_equilibrium_concentrations(
            chempots=chempots,
            fermi_level=fermi_level,
            temperature=temperature,
            per_charge=per_charge,
            per_site=per_site,
            skip_formatting=skip_formatting,
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

    def pseudo_equilibrium_solve(self, chempots, quenched_temperature, annealing_temperature):

        print(annealing_temperature, quenched_temperature)
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

        excluded_columns = ["Defect", "Charge", "Charge State Population"]
        for column in concentrations.columns.difference(excluded_columns):
            concentrations[column] = concentrations[column].astype(float)

        return concentrations
