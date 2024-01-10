from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pymatgen.io.vasp.outputs import Vasprun
from scipy.spatial import ConvexHull, Delaunay

from doped.thermodynamics import DefectThermodynamics

try:
    from py_sc_fermi.defect_species import DefectSpecies
    from py_sc_fermi.defect_system import DefectSystem
    from py_sc_fermi.dos import DOS
except ImportError:
    raise ImportError(
        "Please install py-sc-fermi via `pip install py-sc-fermi` to use this functionality."
    )


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


@dataclass
class FermiSolver:
    """class to manage the generation of DefectSystems from DefectPhaseDiagrams"""

    defect_phase_diagram: DefectThermodynamics
    dos_vasprun: str
    volume: Optional[float] = None

    def __post_init__(self) -> None:
        self.bulk_dos = DOS.from_vasprun(self.dos_vasprun)

        if self.volume is None:
            vr = Vasprun(self.dos_vasprun)
            self.volume = vr.final_structure.volume

    def _defect_picker(self, name: str):  # -> DefectEntry?
        """
        Get defect entry from doped name.

        Args:
        name (str): Name of the defect to find in the defect phase diagram entries.

        Returns:
        Entry: The first entry matching the given name.
        """
        defect = next(
            e for e in self.defect_phase_diagram.defect_entries if e.name == name
        )  # Use next with a generator to get the first matching entry
        return defect

    def _generate_defect_system(self, temperature: float, chemical_potentials: dict) -> DefectSystem:
        """
        Generates a DefectSystem object from the DefectPhaseDiagram and a set
        of chemical potentials.

        Args:
            temperature (float): Temperature in K.
            chemical_potentials (dict): Chemical potentials for the elements.

        Returns:
            DefectSystem: The initialized DefectSystem.
        """
        entries = sorted(self.defect_phase_diagram.defect_entries, key=lambda x: x.name)
        labels = {_get_label_and_charge(entry.name)[0] for entry in entries}
        defect_species = {label: {"charge_states": {}, "nsites": None, "name": label} for label in labels}

        for entry in entries:
            label, charge = _get_label_and_charge(entry.name)
            defect_species[label]["nsites"] = entry.defect.multiplicity  # Assumes one site for the defect

            formation_energy = self.defect_phase_diagram.get_formation_energy(
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
        )

    def defect_system_from_chemical_potentials(
        self, chemical_potentials: dict, temperature: float = 300.0
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

    def update_defect_system_temperature(
        self, defect_system: DefectSystem, temperature: float
    ) -> DefectSystem:
        """
        Generates a new DefectSystem object with a new temperature, generated
        from an existing DefectSystem.

        Args:
            defect_system (DefectSystem): DefectSystem to update.
            temperature (float): Temperature in K.

        Returns:
            DefectSystem: The updated DefectSystem.
        """
        nu_defect_system = deepcopy(defect_system)
        nu_defect_system.temperature = temperature
        return nu_defect_system

    def scan_annealing_temperature(
        self,
        chemical_potentials: dict,
        temperature: float,
        annealing_temperature_range: List[float],
        fix_defect_species: bool = True,
        exceptions: List[str] = [],
        file_name: str = None,
        cpus: int = 1,
        suppress_warnings: bool = True,
    ) -> pd.DataFrame:
        """
        Scans a range of temperatures and returns the concentration_dict() as a
        DataFrame.

        Args:
            chemical_potentials (dict): Chemical potentials for the elements.
            temperature (float): Temperature in K.
            annealing_temperature_range (List[float]): List of annealing temperatures
                to scan.
            fix_defect_species (bool): if annealing temperature is set, this sets
              whether the concentrations of the py-sc-fermi DefectSpecies are fixed
              to their high temperature values, or whether the DefectChargeStates
              are fixed. If in doubt, leave as default value.
            exceptions (list): if annealing_temperature is set, this lists the
              defects to be excluded from the high-temperature concentration fixing
              this may be important in systems with highly mobile defects that are
              not expected to be "frozen-in"
            file name (str): if set, will save a csv file containing results to
              `file_name`
            cpus (int): set to >1 to calculate defect concentrations in parallel

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different
            temperatures in long format.
        """

        # generate a base defect system, and then make one at each temperature in temp_range
        defect_system = self.defect_system_from_chemical_potentials(chemical_potentials)

        # if annealing_temperature is set, generate a defect system at the
        # annealing temperature
        with Pool(processes=cpus) as pool:
            defect_systems = pool.map(
                self.generate_annealed_defect_system,
                [
                    {
                        "initial_system": defect_system,
                        "target_temperature": temperature,
                        "annealing_temperature": annealing_temperature,
                        "fix_defect_species": fix_defect_species,
                        "exceptions": exceptions,
                    }
                    for annealing_temperature in annealing_temperature_range
                ],
            )

        for defect_system in defect_systems:
            defect_system.report()

        # get the concentrations at each temperature
        with Pool(processes=cpus) as pool:
            results = pool.map(self._get_concentrations, defect_systems)

        all_concentration_data = [
            {**data, "Anneal Temperature": anneal} 
            for (concentration_data_list, _), anneal in zip(results, annealing_temperature_range) 
            for data in concentration_data_list
        ]

        all_fermi_level_data = [
            {**data, "Anneal Temperature": anneal} 
            for (_, fermi_level_data_list), anneal in zip(results, annealing_temperature_range) 
            for data in fermi_level_data_list
        ]

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)

        all_data = pd.merge(concentration_df, fermi_level_df, on=["Temperature", "Anneal Temperature"])

        if file_name is not None:
            all_data.to_csv(file_name, index=False)

        return all_data

    def scan_temperature(
        self,
        chemical_potentials: dict,
        temperature_range: List[float],
        annealing_temperature: Optional[float] = None,
        fix_defect_species: bool = True,
        exceptions: List[str] = [],
        file_name: str = None,
        cpus: int = 1,
        suppress_warnings: bool = True,
    ) -> pd.DataFrame:
        """
        Scans a range of temperatures and returns the concentration_dict() as a
        DataFrame.

        Args:
            temp_range (List[float]): List of temperatures to scan.
            annealing_temperature (float): if set, this will carry out a preliminary
              high temperature fermi-energy solution, and fix the defect concentrations
              to the high temperature values before recalculating at lower T
            fix_defect_species (bool): if annealing temperature is set, this sets
              whether the concentrations of the py-sc-fermi DefectSpecies are fixed
              to their high temperature values, or whether the DefectChargeStates
              are fixed. If in doubt, leave as default value.
            exceptions (list): if annealing_temperature is set, this lists the
              defects to be excluded from the high-temperature concentration fixing
              this may be important in systems with highly mobile defects that are
              not expected to be "frozen-in"
            file name (str): if set, will save a csv file containing results to
              `file_name`
            cpus (int): set to >1 to calculate defect concentrations in parallel

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different
            temperatures in long format.
        """

        # generate a base defect system, and then make one at each temperature in temp_range
        defect_system = self.defect_system_from_chemical_potentials(chemical_potentials)
        defect_systems = [
            self.update_defect_system_temperature(defect_system, temp) for temp in temperature_range
        ]

        # if annealing_temperature is set, generate a defect system at the
        # annealing temperature
        if annealing_temperature is not None:
            with Pool(processes=cpus) as pool:
                defect_systems = pool.map(
                    self.generate_annealed_defect_system,
                    [
                        {
                            "initial_system": defect_system,
                            "target_temperature": temp,
                            "annealing_temperature": annealing_temperature,
                            "fix_defect_species": fix_defect_species,
                            "exceptions": exceptions,
                        }
                        for temp in temperature_range
                    ],
                )

        # get the concentrations at each temperature
        with Pool(processes=cpus) as pool:
            results = pool.map(self._get_concentrations, defect_systems)

        all_concentration_data = []
        all_fermi_level_data = []

        for concentration_data, fermi_level_data in results:
            all_concentration_data.extend(concentration_data)
            all_fermi_level_data.extend(fermi_level_data)

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)

        all_data = pd.merge(concentration_df, fermi_level_df, on=["Temperature"])

        if file_name is not None:
            all_data.to_csv(file_name, index=False)

        return all_data

    @staticmethod
    def _get_concentrations(defect_system: DefectSystem) -> Dict[str, float]:
        """
        Returns the concentration_dict() of a DefectSystem.

        Args:
            defect_system (DefectSystem): DefectSystem to get concentrations from.

        Returns:
            Dict[str, float]: Dictionary of concentrations.
        """
        conc_dict = defect_system.concentration_dict()

        concentration_data = []
        fermi_level_data = []

        for name, value in conc_dict.items():
            if name != "Fermi Energy":
                concentration_data.append(
                    {
                        "Defect": name,
                        "Concentration": value,
                        "Temperature": defect_system.temperature,
                    }
                )
            else:
                fermi_level_data.append(
                    {
                        "Fermi Level": value,
                        "Temperature": defect_system.temperature,
                    }
                )

        return concentration_data, fermi_level_data

    def interpolate_chemical_potentials(
        self,
        chem_pot_start: dict,
        chem_pot_end: dict,
        n_points: int,
        temperature: float = 300.0,
        annealing_temperature: Optional[float] = None,
        fix_defect_species: bool = True,
        exceptions: List[str] = [],
        file_name: str = None,
        cpus: int = 1,
        suppress_warnings: bool = True,
    ) -> pd.DataFrame:
        """Linearly interpolates between two dictionaries of chemical potentials
        and returns a dataframe containing all the concentration data. We can
        then optionally saves the data to a csv.

        Args:
            chem_pot_start (dict): Dictionary of starting chemical potentials.
            chem_pot_end (dict): Dictionary of ending chemical potentials.
            n_points (int): Number of points in the linear interpolation
            temperature (float): temperature for final the Fermi level solver
            anneal_temperature (float): if set, this will carry out a preliminary
              high temperature fermi-energy solution, and fix the defect concentrations
              to the high temperature values before recalculating at lower T
            fix_defect_species (bool): if annealing temperature is set, this sets
              whether the concentrations of the py-sc-fermi DefectSpecies are fixed
              to their high temperature values, or whether the DefectChargeStates
              are fixed. If in doubt, leave as default value.
            exceptions (list): if annealing_temperature is set, this lists the
              defects to be excluded from the high-temperature concentration fixing
              this may be important in systems with highly mobile defects that are
              not expected to be "frozen-in"
            file name (str): if set, will save a csv file containing results to
              `file_name`
            cpus (int): set to >1 to calculate defect concentrations in parallel

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different
            chemical potentials in long format.
        """
        all_concentration_data = []
        all_fermi_level_data = []
        interpolated_chem_pots = []
        interpolation = np.linspace(0, 1, n_points)

        for t in interpolation:
            chem_pot_interpolated = {
                element: (1 - t) * start + t * end
                for element, (start, end) in zip(
                    chem_pot_start.keys(), zip(chem_pot_start.values(), chem_pot_end.values())
                )
            }
            interpolated_chem_pots.append(chem_pot_interpolated)

        defect_systems = [
            self.defect_system_from_chemical_potentials(mu, temperature) for mu in interpolated_chem_pots
        ]

        if annealing_temperature is not None:
            with Pool(processes=cpus) as pool:
                defect_systems = pool.map(
                    self.generate_annealed_defect_system,
                    [
                        {
                            "initial_system": defect_system,
                            "target_temperature": temperature,
                            "annealing_temperature": annealing_temperature,
                            "fix_defect_species": fix_defect_species,
                            "exceptions": exceptions,
                        }
                        for defect_system in defect_systems
                    ],
                )

        with Pool(processes=cpus) as pool:
            results = pool.map(self._get_concentrations, defect_systems)

        for r, mu, t in zip(results, interpolated_chem_pots, interpolation):
            [d.update({"interpolation": t} | mu) for d in r[0]]
            [d.update({"interpolation": t} | mu) for d in r[1]]
            all_concentration_data.extend(r[0])
            all_fermi_level_data.extend(r[1])

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)

        all_data = pd.merge(
            concentration_df,
            fermi_level_df,
            on=["Temperature", *list(chem_pot_start.keys()), "interpolation"],
        )

        if file_name is not None:
            all_data.to_csv(file_name, index=False)

        return all_data

    @staticmethod
    def generate_annealed_defect_system(args: dict) -> DefectSystem:
        """generate a py-sc-fermi DefectSystem object that has defect concentrations
        fixed to the values determined at a high temperature (annealing_temperature),
        and then set to a lower temperature (target_temperature)

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
            exceptions (list): if annealing_temperature is set, this lists the
              defects to be excluded from the high-temperature concentration fixing
              this may be important in systems with highly mobile defects that are
              not expected to be "frozen-in"

        Returns:
            DefectSystem: a low temperature defect system (target_temperature),
              with defect concentrations fixed to high temperature
              (annealing_temperature) values.

        """

        # Calculate concentrations at initial temperature
        defect_system = args["initial_system"]
        defect_system.temperature = args["annealing_temperature"]
        initial_conc_dict = defect_system.concentration_dict()

        # Exclude the exceptions and Fermi energy from fixing
        exceptions = ["Fermi Energy", "n0", "p0"]
        exceptions.extend(args["exceptions"])

        # Get the fixed concentrations of non-exceptional defects
        decomposed_conc_dict = args["initial_system"].concentration_dict(decomposed=True)
        additional_data = {}
        for k, v in decomposed_conc_dict.items():
            if k not in exceptions:
                for k1, v1 in v.items():
                    additional_data.update({k + "_" + str(k1): v1})

        initial_conc_dict.update(additional_data)

        fixed_concs = {k: v for k, v in initial_conc_dict.items() if k not in exceptions}

        # Apply the fixed concentrations
        for defect_species in defect_system.defect_species:
            if args["fix_defect_species"] == True:
                if defect_species.name in fixed_concs:
                    defect_species.fix_concentration(
                        fixed_concs[defect_species.name] / 1e24 * defect_system.volume
                    )
            elif args["fix_defect_species"] == False:
                for k, v in defect_species.charge_states.items():
                    key = f"{defect_species.name}_{int(k)}"
                    if key in list(fixed_concs.keys()):
                        v.fix_concentration(fixed_concs[key] / 1e24 * defect_system.volume)

        target_system = deepcopy(defect_system)
        target_system.temperature = args["target_temperature"]
        return target_system

    def chempot_grid(self, chemical_potentials, num_points=10, num_points_along_edge=5):
        """
        Generate a grid of chemical potentials.

        Args:
            chemical_potenials ([type]): [description]
        """
        return ChemicalPotentialGrid(
            chemical_potentials, num_points_along_edge=num_points_along_edge, num_points=num_points
        ).get_grid()

    def grid_solve(
        self,
        chempot_grid: pd.DataFrame,
        temperature: float = 300.0,
        annealing_temperature: float = None,
        fix_defect_species: bool = True,
        exceptions: Optional[list[str]] = [],
        file_name: Optional[str] = None,
        cpus: int = 1,
        suppress_warnings: bool = True,
    ) -> pd.DataFrame:
        """Solve for all chemical potentials in a grid in n-dimensional
        chemical potential space.

        Args:
            chempot_grid (pd.DataFrame): n-dimensional chemical potential grid
              in a pandas DataFrame format
            temperature (float): temperature for final the Fermi level solver
            annealing_temperature (float): if set, this will carry out a preliminary
              high temperature fermi-energy solution, and fix the defect concentrations
              to the high temperature values before recalculating at lower T
            fix_defect_species (bool): if annealing temperature is set, this sets
              whether the concentrations of the py-sc-fermi DefectSpecies are fixed
              to their high temperature values, or whether the DefectChargeStates
              are fixed. If in doubt, leave as default value.
            exceptions (list): if annealing_temperature is set, this lists the
              defects to be excluded from the high-temperature concentration fixing
              this may be important in systems with highly mobile defects that are
              not expected to be "frozen-in"
            file name (str): if set, will save a csv file containing results to
              `file_name`
            cpus (int): set to >1 to calculate defect concentrations in parallel

        Returns:
            pd.DataFrame: DataFrame containing all the defect concentrations at
              each chemical potential set in the DataFrame
        """

        chemical_potentials = chempot_grid.copy()
        chemical_potentials.drop(["is_vertex", "facet"], axis=1, inplace=True)

        defect_systems = [
            self.defect_system_from_chemical_potentials(row.to_dict(), temperature=temperature)
            for _i, row in chemical_potentials.iterrows()
        ]

        if annealing_temperature is not None:
            with Pool(processes=cpus) as pool:
                defect_systems = pool.map(
                    self.generate_annealed_defect_system,
                    [
                        {
                            "initial_system": defect_system,
                            "target_temperature": temperature,
                            "annealing_temperature": annealing_temperature,
                            "fix_defect_species": fix_defect_species,
                            "exceptions": exceptions,
                        }
                        for defect_system in defect_systems
                    ],
                )

        with Pool(processes=cpus) as pool:
            results = pool.map(self._get_concentrations, defect_systems)

        all_concentration_data = []
        all_fermi_level_data = []
        for r, t in zip(results, chempot_grid.iterrows()):
            [d.update(t[1]) for d in r[0]]
            [d.update(t[1]) for d in r[1]]
            all_concentration_data.extend(r[0])
            all_fermi_level_data.extend(r[1])

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)

        all_data = pd.merge(
            concentration_df, fermi_level_df, on=[*list(chempot_grid.columns), "Temperature"]
        )

        if file_name is not None:
            all_data.to_csv(file_name, index=False)

        return all_data


@dataclass
class ChemicalPotentialGrid:
    chemical_potentials: dict
    num_points_along_edge: int = 5
    num_points: int = 10
    vertices: np.ndarray = field(init=False, default=None)
    grid_points: np.ndarray = field(init=False, default=None)
    internal_grid_points: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.vertices = self._initialize_vertices()
        self._generate_grid_from_vertices()
        self._generate_internal_grid()

    def _initialize_vertices(self):
        vertices = [
            np.array(list(facet.values())) for facet in self.chemical_potentials["facets"].values()
        ]
        return np.array(vertices)

    def _generate_grid_from_vertices(self):
        try:
            hull = ConvexHull(self.vertices)
        except:
            print("QhullError encountered. Trying to joggle the points.")
            self.vertices += np.random.uniform(-1e-12, 1e-12, self.vertices.shape)
            hull = ConvexHull(self.vertices)

        grid_points = []
        distance_covered = 0

        total_perimeter = sum(
            np.linalg.norm(self.vertices[simplex[i]] - self.vertices[simplex[j]])
            for simplex in hull.simplices
            for i in range(len(simplex))
            for j in range(i + 1, len(simplex))
        )

        spacing = total_perimeter / self.num_points_along_edge

        for simplex in hull.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    point1 = self.vertices[simplex[i]]
                    point2 = self.vertices[simplex[j]]
                    edge_length = np.linalg.norm(point2 - point1)

                    num_points_on_edge = int(
                        np.floor((distance_covered + edge_length) / spacing)
                        - np.floor(distance_covered / spacing)
                    )

                    if num_points_on_edge > 0:
                        edge_points = np.linspace(point1, point2, num_points_on_edge + 2)[1:-1]
                        grid_points.extend(edge_points)

                    distance_covered += edge_length

        self.grid_points = np.array(grid_points)

    def _generate_internal_grid(self):
        all_points = np.vstack([self.grid_points, self.vertices])
        tri = Delaunay(all_points)
        internal_grid_points = []

        for simplex in tri.simplices:
            simplex_vertices = all_points[simplex]
            dim = len(simplex)
            barycentric_coords = self._regular_barycentric_coords(self.num_points, dim)

            interpolated_points = np.dot(barycentric_coords, simplex_vertices)
            internal_grid_points.append(interpolated_points)

        self.internal_grid_points = np.vstack(internal_grid_points)

    def _regular_barycentric_coords(self, num_divisions: int, dim: int) -> np.ndarray:
        barycentric_coords = []
        for coords in np.ndindex((num_divisions + 1,) * dim):
            if sum(coords) == num_divisions:
                barycentric_coords.append(np.array(coords) / num_divisions)
        return np.array(barycentric_coords)

    def get_grid(self) -> pd.DataFrame:
        elements = list(self.chemical_potentials["facets"].values())[0].keys()
        grid_df = pd.DataFrame(self.grid_points, columns=elements)
        internal_grid_df = pd.DataFrame(self.internal_grid_points, columns=elements)
        vertices_df = pd.DataFrame(self.vertices, columns=elements)

        grid_df["is_vertex"] = [False] * len(self.grid_points)
        internal_grid_df["is_vertex"] = [False] * len(self.internal_grid_points)
        vertices_df["is_vertex"] = [True] * len(self.vertices)
        vertices_df["facet"] = self.chemical_potentials["facets"].keys()

        df = pd.concat([grid_df, internal_grid_df, vertices_df], ignore_index=True)
        grid_df = df.drop_duplicates(
            subset=[*list(list(self.chemical_potentials["facets"].values())[0].keys()), "is_vertex"]
        )

        return grid_df
