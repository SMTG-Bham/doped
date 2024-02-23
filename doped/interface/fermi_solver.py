from typing import Optional, Union, List
from warnings import warn
from itertools import product
from copy import deepcopy
import numpy as np
from joblib import Parallel, delayed
from typing import Dict, Any

from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata
import pandas as pd
from monty.json import MSONable
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.io.vasp import Vasprun

from doped.thermodynamics import DefectThermodynamics
from doped.utils.parsing import get_neutral_nelect_from_vasprun

try:
    from py_sc_fermi.defect_system import DefectSystem
    from py_sc_fermi.defect_species import DefectSpecies
    from py_sc_fermi.defect_charge_state import DefectChargeState
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

    def equilibrium_solve(
        self,
        chempots: dict[str, float],
        temperature: float,
        effective_dopant_concentration: Optional[float] = None,
    ) -> None:
        """not implemented in the base class, implemented in the derived class"""
        raise NotImplementedError(
            """This method is implemented in the derived class, 
            use FermiSolverDoped or FermiSolverPyScFermi instead."""
        )

    def pseudo_equilibrium_solve(
        self,
        chempots: dict[str, float],
        quenched_temperature: float,
        annealing_temperature: float,
        effective_dopant_concentration: Optional[float] = None,
    ) -> None:
        """not implemented in the base class, implemented in the derived class"""
        raise NotImplementedError(
            """This method is implemented in the derived class, 
            use FermiSolverDoped or FermiSolverPyScFermi instead."""
        )

    def scan_temperature(
        self,
        chempots: dict[str, float],
        temperature_range: Union[float, List[float]],
        annealing_temperature_range: Optional[Union[float, List[float]]] = None,
        quenching_temperature_range: Optional[Union[float, List[float]]] = None,
        effective_dopant_concentration: float = None,
        exceptions: Optional[List] = None,
        processes: int = 1,
    ) -> pd.DataFrame:
        
        if annealing_temperature_range is not None and quenching_temperature_range is not None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials_pseudo)(
                    chempots=chempots,
                    quenched_temperature=quench_temp,
                    annealing_temperature=anneal_temp,
                    effective_dopant_concentration=effective_dopant_concentration,
                    exceptions=exceptions,
                )
                for quench_temp, anneal_temp in product(
                    quenching_temperature_range, annealing_temperature_range
                )
            )
            return pd.concat(all_data)
        
        elif annealing_temperature_range is None and quenching_temperature_range is None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials)(
                    chempots=chempots,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    exceptions=exceptions,
                )
                for temperature in temperature_range
            )
            return pd.concat(all_data)
        
        else:
            raise ValueError(
                "You must specify both annealing and quenching temperature, or just temperature."
            )
        
    def scan_chemical_potentials(self, chempots: list[dict[str, float]], 
                                 temperature: float, 
                                 annealing_temperature: float, 
                                 quenching_temperature: float, 
                                 effective_dopant_concentration: Optional[float], 
                                 exceptions: list,
                                 processes: 1) -> pd.DataFrame:
        

        if annealing_temperature is not None and quenching_temperature is not None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials_pseudo)(
                    chempots=chempots,
                    quenched_temperature=quenching_temperature,
                    annealing_temperature=annealing_temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    exceptions=exceptions,
                )
            )
            return pd.concat(all_data)
        
        elif annealing_temperature is None and quenching_temperature is None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials)(
                    chempots=chempots,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    exceptions=exceptions,
                )
            )
            return pd.concat(all_data)
        
        else:
            raise ValueError(
                "You must specify both annealing and quenching temperature, or just temperature."
            )


    def scan_dopant_concentration(
        self,
        chempots: dict[str, float],
        temperature: float,
        effective_dopant_concentration_range: Union[float, List[float]],
        annealing_temperature: Optional[float] = None,
        quenching_temperature: Optional[float] = None,
        exceptions: Optional[List] = None,
        processes: int = 1,
    ) -> pd.DataFrame:
        
        if annealing_temperature is not None and quenching_temperature is not None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials_pseudo)(
                    chempots=chempots,
                    quenched_temperature=quenching_temperature,
                    annealing_temperature=annealing_temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    exceptions=exceptions,
                )
            for effective_dopant_concentration in effective_dopant_concentration_range)
            return pd.concat(all_data)
        
        elif annealing_temperature is None and quenching_temperature is None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials)(
                    chempots=chempots,
                    temperature=temperature,
                    effective_dopant_concentration=effective_dopant_concentration,
                    exceptions=exceptions,
                )
                for effective_dopant_concentration in effective_dopant_concentration_range
            )
            return pd.concat(all_data)
        
        else:
            raise ValueError(
                "You must specify both annealing and quenching temperature, or just temperature."
            )
        
    def interpolate_chemical_potentials(
        self,
        chem_pot_start: dict,
        chem_pot_end: dict,
        n_points: int,
        temperature = 300.0,
        annealing_temperature: Optional[float] = None, 
        quenching_temperature: Optional[float] = None,
        effective_dopant_concentration: Optional[float] = None,
        exceptions: Optional[List] = None,
        processes: int = 1,
    ) -> pd.DataFrame:
        
        interpolated_chem_pots = self._get_interpolated_chempots(chem_pot_start, chem_pot_end, n_points)
        
        if annealing_temperature is not None and quenching_temperature is not None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials_pseudo)(
                    chempots=chem_pots,
                    quenched_temperature=quenching_temperature,
                    annealing_temperature=annealing_temperature,
                    # effective_dopant_concentration=effective_dopant_concentration,
                    # exceptions=exceptions,
                )
            for chem_pots in interpolated_chem_pots)
            return pd.concat(all_data)
        
        elif annealing_temperature is None and quenching_temperature is None:
            all_data = Parallel(n_jobs=processes)(
                delayed(self._solve_and_append_chemical_potentials)(
                    chempots=chem_pots,
                    temperature=temperature,
                    # effective_dopant_concentration=effective_dopant_concentration,
                    # exceptions=exceptions,
                )
            for chem_pots in interpolated_chem_pots)
            return pd.concat(all_data)
        
        else:
            raise ValueError(
                "You must specify both annealing and quenching temperature, or just temperature."
            )
        
    def _get_interpolated_chempots(self, chem_pot_start, chem_pot_end, n_points):
        """
        Generate a set of interpolated chemical potentials between two points.

        Args:
            chem_pot_start (dict): The starting chemical potentials.
            chem_pot_end (dict): The ending chemical potentials.
            n_points (int): The number of points to generate.

        Returns:
            list: A list of dictionaries containing the interpolated chemical potentials.
        """
        interpolated_chem_pots = []
        for i in range(n_points):
            chem_pots = {}
            for key in chem_pot_start:
                chem_pots[key] = chem_pot_start[key] + (chem_pot_end[key] - chem_pot_start[key]) * i / (n_points - 1)
            interpolated_chem_pots.append(chem_pots)
        return interpolated_chem_pots


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
        self, chempots, quenched_temperature, annealing_temperature
    ):
        df = self.pseudo_equilibrium_solve(chempots, quenched_temperature, annealing_temperature)
        for key, value in chempots.items():
            df[key] = value
        return df


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

    def _generate_dopant(self, effective_dopant_concentration: float) -> DefectSpecies:
        """Generate a dopant defect charge state object.

        Args:
            effective_dopant_concentration (float): The effective dopant concentration.

        Returns:
            DefectChargeState: The initialized DefectChargeState.
        """

        if effective_dopant_concentration is not None:
            if effective_dopant_concentration > 0:
                charge = 1
                effective_dopant_concentration = abs(effective_dopant_concentration) / 1e24 * self.volume
            elif effective_dopant_concentration < 0:
                charge = -1
                effective_dopant_concentration = abs(effective_dopant_concentration) / 1e24 * self.volume
        dopant = DefectChargeState(
            charge=charge, fixed_concentration=effective_dopant_concentration, degeneracy=1
        )
        return DefectSpecies(nsites=1, charge_states={charge: dopant}, name="Dopant")

    def generate_defect_system(
        self,
        temperature: float,
        chemical_potentials: dict[str, float],
        effective_dopant_concentration: Optional[float] = None,
    ) -> DefectSystem:
        """
        Generates a DefectSystem object from the DefectPhaseDiagram and a set
        of chemical potentials.

        Args:
            temperature (float): Temperature in K.
            chemical_potentials (dict): Chemical potentials for the elements.
            effective_dopant_concentration (float): The effective dopant concentration.

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
        if effective_dopant_concentration is not None:
            dopant = self._generate_dopant(effective_dopant_concentration)
            all_defect_species.append(dopant)

        return DefectSystem(
            defect_species=all_defect_species,
            dos=self.bulk_dos,
            volume=self.volume,
            temperature=temperature,
            convergence_tolerance=1e-20,
        )

    def equilibrium_solve(
        self,
        chempots: dict[str, float],
        temperature: float,
        effective_dopant_concentration: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Solve for the defect concentrations at a given temperature and chemical potentials.

        Args:
            chempots (dict): Chemical potentials for the elements.
            temperature (float): Temperature in K.

        Returns:
            pd.DataFrame: DataFrame containing defect and carrier concentrations
            and the self consistent Fermi energy
        """
        defect_system = self.generate_defect_system(
            chemical_potentials=chempots,
            temperature=temperature,
            effective_dopant_concentration=effective_dopant_concentration,
        )
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
                    row.update({"Dopant (cm^-3)": conc_dict["Dopant"]})
                row.update({"Defect": k, "Concentration (cm^-3)": v})
                data.append(row)

        df = pd.DataFrame(data)
        df.set_index("Defect", inplace=True, drop=True)
        return df

    def pseudo_equilibrium_solve(
        self,
        chempots: dict[str, float],
        quenched_temperature: float,
        annealing_temperature: float,
        effective_dopant_concentration: Optional[float] = None,
        **kwargs,
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
            effective_dopant_concentration=effective_dopant_concentration,
            **kwargs,
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
                if "Dopant" in conc_dict:
                    row.update({"Dopant (cm^-3)": conc_dict["Dopant"]})
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
        effective_dopant_concentration=None,
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
        defect_system = self.generate_defect_system(
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

        if effective_dopant_concentration is not None:
            dopant = self._generate_dopant(effective_dopant_concentration)
            defect_system.defect_species.append(dopant)

        target_system = deepcopy(defect_system)
        target_system.temperature = quenched_temperature
        return target_system


class ChemicalPotentialGrid:
    """
    A class to represent a grid of chemical potentials and to perform
    operations such as generating a grid within the convex hull of given vertices.
    """

    def __init__(self, chemical_potentials: Dict[str, Any]) -> None:
        """
        Initializes the ChemicalPotentialGrid with chemical potential data.

        Parameters:
        chemical_potentials (Dict[str, Any]): A dictionary containing chemical
                                               potential information.
        """
        self.chemical_potentials = chemical_potentials
        self.vertices = pd.DataFrame.from_dict(chemical_potentials["facets"], orient="index")

    def get_grid(self, dependent_variable: str, n_points: int = 100) -> pd.DataFrame:
        """
        Generates a grid within the convex hull of the vertices and interpolates
        the dependent variable values.

        Parameters:
        dependent_variable (str): The name of the column in vertices representing the dependent variable.
        n_points (int): The number of points to generate along each axis of the grid.

        Returns:
        pd.DataFrame: A DataFrame of points within the convex hull with their corresponding
                    interpolated dependent variable values.
        """
        # Exclude the dependent variable from the vertices
        independent_vars = self.vertices.drop(columns=dependent_variable)
        dependent_var = self.vertices[dependent_variable].values

        # Generate the complex number for grid spacing
        complex_num = complex(0, n_points)

        # Get the convex hull of the vertices
        hull = ConvexHull(independent_vars.values)

        # Create a dense grid that covers the entire range of the vertices
        x_min, y_min = independent_vars.min(axis=0)
        x_max, y_max = independent_vars.max(axis=0)
        grid_x, grid_y = np.mgrid[x_min:x_max:complex_num, y_min:y_max:complex_num]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        # Delaunay triangulation to get points inside the hull
        delaunay = Delaunay(hull.points[hull.vertices])
        inside_hull = delaunay.find_simplex(grid_points) >= 0
        points_inside = grid_points[inside_hull]

        # Interpolate the values to get the dependent chemical potential
        values_inside = griddata(independent_vars.values, dependent_var, points_inside, method="linear")

        # Combine points with their corresponding interpolated values
        grid_with_values = np.hstack((points_inside, values_inside.reshape(-1, 1)))

        # Convert to DataFrame and set column names
        grid_df = pd.DataFrame(
            grid_with_values, columns=list(independent_vars.columns) + [dependent_variable]
        )

        return grid_df
