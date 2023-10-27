from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from doped.utils.legacy_pmg.thermodynamics import DefectPhaseDiagram

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
    defect_phase_diagram: DefectPhaseDiagram
    bulk_vasprun: str

    def __post_init__(self) -> None:
        """
        Initializes additional attributes after dataclass instantiation.
        """
        self.bulk_dos = DOS.from_vasprun(self.bulk_vasprun)
        self.volume = self.defect_phase_diagram.entries[0].defect.structure.volume

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
        """
        Generates a DefectSystem object from the DefectPhaseDiagram and a set
        of chemical potentials.

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

    def defect_system_from_chemical_potentials(
        self, chemical_potentials: dict, temperature: float = 300.0
    ) -> DefectSystem:
        """
        Updates the energies of the DefectSystem with a new set of chemical
        potentials.
        """
        defect_system = self._generate_defect_system(
            temperature=temperature, chemical_potentials=chemical_potentials
        )
        return defect_system

    def update_defect_system_temperature(
        self, defect_system: DefectSystem, temperature: float
    ) -> DefectSystem:
        nu_defect_system = deepcopy(defect_system)
        nu_defect_system.temperature = temperature
        return nu_defect_system

    def scan_temperature_and_save(
        self,
        chemical_potentials: dict,
        temperature_range: List[float],
        anneal_temperature: float = None,
        level: str = "DefectSpecies",
        exceptions: List[str] = [],
    ) -> pd.DataFrame:
        """
        Scans a range of temperatures and returns the concentration_dict() as a
        DataFrame.

        Args:
            temp_range (List[float]): List of temperatures to scan.
            fix_concentration_temp (float, optional): The temperature at which to fix concentrations. Defaults to None.
            level (str, optional): The level at which to fix concentrations. Defaults to "DefectSpecies".
            exceptions (List[str], optional): List of species or charge states to exclude from fixing. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different temperatures in long format.
        """
        all_concentration_data = []
        all_fermi_level_data = []

        defect_system = self.defect_system_from_chemical_potentials(chemical_potentials)

        for temp in temperature_range:
            if anneal_temperature is None:
                updated_system = self.update_defect_system_temperature(defect_system, temp)
            else:
                updated_system = self.generate_annealed_defect_system(
                    mu=chemical_potentials,
                    annealing_temperature=anneal_temperature,
                    target_temperature=temp,
                    level=level,
                    exceptions=exceptions,
                )

            concentration_data, fermi_level_data = self._get_concentrations(updated_system)
            all_concentration_data.extend(concentration_data)
            all_fermi_level_data.extend(fermi_level_data)

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)
        return pd.merge(concentration_df, fermi_level_df, on="Temperature")

    def _get_concentrations(self, defect_system: DefectSystem):
        """
        _summary_.

        Args:
            defect_system (DefectSystem): _description_

        Returns:
            _type_: _description_
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

    def interpolate_chemical_potentials_and_save(
        self,
        chem_pot_start: dict,
        chem_pot_end: dict,
        n_points: int,
        temp: float = 300.0,
        anneal_temp: Optional[float] = None,
        level: str = "DefectSpecies",
        exceptions: List[str] = [],
    ) -> pd.DataFrame:
        """
        Scans a range of chemical potentials and returns the
        concentration_dict() as a DataFrame.

        Args:
            chem_pot_start (dict): Dictionary of starting chemical potentials.
            chem_pot_end (dict): Dictionary of ending chemical potentials.
            n_points (int): Number of points to scan.

        Returns:
            pd.DataFrame: DataFrame containing concentrations at different chemical potentials in long format.
        """
        all_concentration_data = []
        all_fermi_level_data = []

        for t in np.linspace(0, 1, n_points):
            chem_pot_interpolated = {
                element: (1 - t) * start + t * end
                for element, (start, end) in zip(
                    chem_pot_start.keys(), zip(chem_pot_start.values(), chem_pot_end.values())
                )
            }

            if anneal_temp is None:
                updated_system = self.defect_system_from_chemical_potentials(
                    chem_pot_interpolated, temperature=temp
                )
            else:
                updated_system = self.generate_annealed_defect_system(
                    mu=chem_pot_interpolated,
                    annealing_temperature=anneal_temp,
                    target_temperature=temp,
                    level=level,
                    exceptions=exceptions,
                )

            concentration_data, fermi_level_data = self._get_concentrations(updated_system)
            [d.update({"Interpolation_Parameter": t}) for d in concentration_data]
            [d.update({"Interpolation_Parameter": t}) for d in fermi_level_data]
            [d.update(chem_pot_interpolated) for d in concentration_data]
            all_concentration_data.extend(concentration_data)
            all_fermi_level_data.extend(fermi_level_data)

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)

        return pd.merge(concentration_df, fermi_level_df, on=["Interpolation_Parameter"])

    def generate_annealed_defect_system(
        self,
        mu: Dict[str, float],
        annealing_temperature: float,
        target_temperature: float = 300.0,
        level: str = "DefectSpecies",
        exceptions: List[str] = [],
    ) -> DefectSystem:
        if level not in ["DefectSpecies", "DefectChargeState"]:
            raise ValueError("level must be either 'DefectSpecies' or 'DefectChargeState'")

        # Calculate concentrations at initial temperature
        initial_system = self.defect_system_from_chemical_potentials(mu, temperature=annealing_temperature)
        initial_conc_dict = initial_system.concentration_dict()

        # Exclude the exceptions and Fermi energy from fixing
        exceptions.extend(["Fermi Energy", "n0", "p0"])

        # Get the fixed concentrations of non-exceptional defects
        decomposed_conc_dict = initial_system.concentration_dict(decomposed=True)
        additional_data = {}
        for k, v in decomposed_conc_dict.items():
            if k not in exceptions:
                for k1, v1 in v.items():
                    additional_data.update({k + "_" + str(k1): v1})

        initial_conc_dict.update(additional_data)

        fixed_concs = {k: v for k, v in initial_conc_dict.items() if k not in exceptions}

        # Apply the fixed concentrations
        for defect_species in initial_system.defect_species:
            if level == "DefectSpecies":
                if defect_species.name in fixed_concs:
                    defect_species.fix_concentration(fixed_concs[defect_species.name] / 1e24 * self.volume)
            elif level == "DefectChargeState":
                for k, v in defect_species.charge_states.items():
                    key = f"{defect_species.name}_{int(k)}"
                    if key in list(fixed_concs.keys()):
                        v.fix_concentration(fixed_concs[key] / 1e24 * self.volume)

        target_system = self.update_defect_system_temperature(initial_system, target_temperature)
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
        chempot_grid,
        temperature=300.0,
        anneal_temperature=None,
        level="DefectSpecies",
        exceptions=[],
    ):
        """
        Args:
            chempot_grid (_type_): _description_.

        Returns:
            _type_: _description_
        """
        all_concentration_data = []
        all_fermi_level_data = []
        for _i, row in tqdm(chempot_grid.iterrows()):
            row = row.drop(["is_vertex", "facet"])
            chemical_potentials = row.to_dict()

            if anneal_temperature:
                live_system = self.generate_annealed_defect_system(
                    chemical_potentials, anneal_temperature, temperature, level, exceptions
                )

            else:
                live_system = self.defect_system_from_chemical_potentials(
                    chemical_potentials, temperature=temperature
                )

            concentration_data, fermi_level_data = self._get_concentrations(live_system)
            [d.update(chemical_potentials) for d in concentration_data]
            [d.update(chemical_potentials) for d in fermi_level_data]
            all_concentration_data.extend(concentration_data)
            all_fermi_level_data.extend(fermi_level_data)

        concentration_df = pd.DataFrame(all_concentration_data)
        fermi_level_df = pd.DataFrame(all_fermi_level_data)

        return pd.merge(
            concentration_df,
            fermi_level_df,
            on=["Temperature", *list(chemical_potentials.keys())],
        )


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
