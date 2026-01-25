"""
Code to generate and parse GPAW defect calculations.
"""

import os
from typing import Optional, Union, Dict, Any

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry

from doped.core import DefectEntry
from doped.utils.parsing import _get_defect_supercell, _get_bulk_supercell
from doped.analysis import defect_from_structures

class GPAWDefectRelaxSet:
    """
    Class for generating input files (Python scripts) for GPAW defect relaxation.
    """

    def __init__(
        self,
        defect_entry: Union[DefectEntry, Structure],
        charge_state: Optional[int] = None,
        gpaw_settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            defect_entry (DefectEntry, Structure):
                doped/pymatgen DefectEntry or Structure object.
            charge_state (int):
                Charge state of the defect. Overrides DefectEntry.charge_state.
            gpaw_settings (dict):
                Dictionary of GPAW settings (mode, xc, kpts, etc.).
                Example:
                {
                    "mode": {"name": "pw", "ecut": 300},
                    "xc": "PBE",
                    "kpts": {"size": (2, 2, 2), "gamma": True},
                    "txt": "gpaw_output.txt"
                }
        """
        self.defect_entry = defect_entry
        self.charge_state = charge_state
        if self.charge_state is None and isinstance(self.defect_entry, DefectEntry):
             self.charge_state = self.defect_entry.charge_state

        self.gpaw_settings = gpaw_settings or {}

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)

    def write_input(
        self,
        output_path: str,
        filename: str = "relax.py",
        make_dir_if_not_present: bool = True,
    ):
        """
        Writes the input files (structure and script) to a directory.
        """
        if make_dir_if_not_present:
            os.makedirs(output_path, exist_ok=True)
        
        # Write structure to a file
        structure_filename = "structure.cif"
        self.defect_supercell.to(filename=os.path.join(output_path, structure_filename), fmt="cif")
        
        # Generate Python script
        script_content = self._generate_script(structure_filename)
        
        with open(os.path.join(output_path, filename), "w") as f:
            f.write(script_content)

    def _generate_script(self, structure_filename: str) -> str:
        """
        Generates the content of the GPAW script.
        """
        
        settings = self.gpaw_settings.copy()

        # Extract known parameters
        mode_params = settings.pop("mode", {"name": "pw", "ecut": 400})
        xc = settings.pop("xc", "PBE")
        kpts = settings.pop("kpts", {"size": (1, 1, 1), "gamma": True})
        txt = settings.pop("txt", "gpaw_output.txt")
        convergence = settings.pop("convergence", {})
        
        # Determine charge
        charge = self.charge_state or 0
        
        # Determine spinpol (default True for defects if not specified)
        spinpol = settings.pop("spinpol", True)
        
        # Relaxation params
        fmax = settings.pop("fmax", 0.05)

        # Prepare mode string
        if isinstance(mode_params, dict):
            name = mode_params.pop("name", "pw")
            args = ", ".join([f"{k}={v!r}" for k, v in mode_params.items()])
            mode_str = f"{name.upper()}({args})"
        else:
            mode_str = repr(mode_params)
            
        # Prepare other settings
        other_kwargs = ""
        if settings:
            other_kwargs = ",\n    " + ",\n    ".join([f"{k}={v!r}" for k, v in settings.items()])

        script = f"""
from ase.io import read
from gpaw import GPAW, PW
from ase.optimize import BFGS
import json

# Read structure
atoms = read('{structure_filename}')

# Setup calculator
calc = GPAW(
    mode={mode_str},
    xc='{xc}',
    kpts={kpts},
    txt='{txt}',
    convergence={convergence},
    charge={charge},
    spinpol={spinpol}{other_kwargs}
)

atoms.calc = calc

print("Starting calculation...")
energy = atoms.get_potential_energy()
print(f"Initial Energy: {{energy}} eV")

# Relaxation
dyn = BFGS(atoms, trajectory='relax.traj')
dyn.run(fmax={fmax})

# Save the final state
calc.write('relaxed.gpw')

print(f"Final Energy: {{atoms.get_potential_energy()}} eV")
"""
        return script

def get_gpaw_site_potentials(gpw_file: str) -> np.ndarray:
    """
    Extracts atomic site potentials from a GPAW .gpw file.
    In GPAW, we can get the electrostatic potential on a grid and then
    average it at atomic positions.
    """
    from gpaw import GPAW
    calc = GPAW(gpw_file)
    atoms = calc.get_atoms()
    
    # Get electrostatic potential on grid
    v_ext = calc.get_electrostatic_potential()
    
    # Grid info
    gd = calc.hamiltonian.finegd
    
    site_potentials = []
    for atom in atoms:
        # Get position in grid coordinates
        indices = gd.get_nearest_grid_point(atom.position)
        val = v_ext[tuple(indices % gd.N_c)]
        site_potentials.append(val)
        
    return np.array(site_potentials)

def get_gpaw_planar_averaged_potential(gpw_file: str) -> Dict[str, np.ndarray]:
    """
    Extracts planar-averaged potential from a GPAW .gpw file.
    Needed for Freysoldt (FNV) correction.
    """
    from gpaw import GPAW
    calc = GPAW(gpw_file)
    v_ext = calc.get_electrostatic_potential()
    
    # Grid info
    gd = calc.hamiltonian.finegd
    
    planar_averages = {}
    for i in range(3):
        # Average over the other two dimensions
        axes = [0, 1, 2]
        axes.remove(i)
        planar_averages[str(i)] = v_ext.mean(axis=tuple(axes))
        
    return planar_averages

class GPAWParser:
    """
    Parser for GPAW calculations to interface with doped.
    """
    def __init__(self, gpw_file: str):
        """
        Args:
            gpw_file (str): Path to GPAW .gpw file.
        """
        from gpaw import GPAW
        self.gpw_file = gpw_file
        self.calc = GPAW(gpw_file)
        self.atoms = self.calc.get_atoms()
        self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        self.energy = self.calc.get_potential_energy()

    def get_computed_structure_entry(self) -> ComputedStructureEntry:
        """
        Returns a ComputedStructureEntry for the calculation.
        """
        return ComputedStructureEntry(self.structure, self.energy)

    def get_computed_entry(self) -> ComputedEntry:
        """
        Returns a ComputedEntry for the calculation.
        """
        return ComputedEntry(self.structure.composition, self.energy)

    def get_site_potentials(self) -> np.ndarray:
        """
        Returns atomic site potentials.
        """
        return get_gpaw_site_potentials(self.gpw_file)

    def get_locpot_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns planar-averaged potential dictionary.
        """
        return get_gpaw_planar_averaged_potential(self.gpw_file)

    def get_eigenvalue_properties(self) -> tuple:
        """
        Returns (band_gap, cbm, vbm, efermi).
        """
        # Basic implementation
        efermi = self.calc.get_fermi_level()
        # GPAW can give eigenvalues for each k-point and spin
        # This is a simplification to get VBM/CBM
        energies = []
        for s in range(self.calc.get_number_of_spins()):
            for k in range(len(self.calc.get_ibz_k_points())):
                energies.extend(self.calc.get_eigenvalues(kpt=k, spin=s))
        
        energies = sorted(energies)
        # Identify VBM and CBM based on efermi
        vbm = max([e for e in energies if e <= efermi]) if any(e <= efermi for e in energies) else efermi
        cbm = min([e for e in energies if e > efermi]) if any(e > efermi for e in energies) else efermi
        band_gap = cbm - vbm
        
        return band_gap, cbm, vbm, efermi

def get_gpaw_defect_entry(
    defect_path: str, 
    bulk_path: str, 
    dielectric: Optional[Union[float, np.ndarray]] = None,
    charge_state: int = 0
) -> DefectEntry:
    """
    Convenience function to create a DefectEntry from GPAW directories.
    Assumes 'relaxed.gpw' exists in both directories.
    """
    defect_parser = GPAWParser(os.path.join(defect_path, "relaxed.gpw"))
    bulk_parser = GPAWParser(os.path.join(bulk_path, "relaxed.gpw"))

    # Identify defect
    defect = defect_from_structures(bulk_parser.structure, defect_parser.structure)

    # Band edge data
    band_gap, cbm, vbm, efermi = bulk_parser.get_eigenvalue_properties()

    defect_entry = DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=defect_parser.get_computed_structure_entry(),
        bulk_entry=bulk_parser.get_computed_structure_entry(),
        sc_defect_frac_coords=defect.site.frac_coords,
        defect_supercell=defect_parser.structure,
        bulk_supercell=bulk_parser.structure,
        defect_supercell_site=defect.site,
        calculation_metadata={
            "bulk_path": bulk_path,
            "defect_path": defect_path,
            "dielectric": dielectric,
            "bulk_site_potentials": bulk_parser.get_site_potentials(),
            "defect_site_potentials": defect_parser.get_site_potentials(),
            "bulk_locpot_dict": bulk_parser.get_locpot_dict(),
            "defect_locpot_dict": defect_parser.get_locpot_dict(),
            "vbm": vbm,
            "band_gap": band_gap,
            "cbm": cbm,
            "efermi": efermi,
        }
    )

    return defect_entry

class GPAWDefectsParser:
    """
    Class for rapidly parsing multiple GPAW defect supercell calculations.
    """
    def __init__(
        self,
        output_path: str = ".",
        bulk_path: Optional[str] = None,
        dielectric: Optional[Union[float, np.ndarray]] = None,
        subfolder: Optional[str] = None,
    ):
        """
        Args:
            output_path (str): Path to directory containing defect folders.
            bulk_path (str): Path to bulk reference folder.
            dielectric (float or matrix): Dielectric constant for corrections.
            subfolder (str): Optional subfolder within each defect folder.
        """
        self.output_path = output_path
        self.dielectric = dielectric
        self.subfolder = subfolder
        
        if bulk_path is None:
            # Try to find bulk folder
            folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
            bulk_folders = [f for f in folders if "bulk" in f.lower()]
            if not bulk_folders:
                raise ValueError("Could not find bulk folder. Please specify bulk_path.")
            self.bulk_path = os.path.join(output_path, bulk_folders[0])
        else:
            self.bulk_path = bulk_path

    def parse_all(self) -> Dict[str, DefectEntry]:
        """
        Parses all defect folders in output_path.
        """
        defect_dict = {}
        folders = [f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))]
        
        # Exclude bulk folder
        defect_folders = [f for f in folders if os.path.abspath(os.path.join(self.output_path, f)) != os.path.abspath(self.bulk_path)]
        
        for folder in defect_folders:
            defect_dir = os.path.join(self.output_path, folder)
            if self.subfolder:
                defect_dir = os.path.join(defect_dir, self.subfolder)
            
            if not os.path.exists(os.path.join(defect_dir, "relaxed.gpw")):
                continue
            
            print(f"Parsing {folder}...")
            # Try to guess charge state from folder name
            charge_state = 0
            try:
                if "_" in folder:
                    suffix = folder.rsplit("_", 1)[-1]
                    if suffix.startswith(("+", "-")):
                        charge_state = int(suffix)
            except Exception:
                pass
            
            try:
                defect_entry = get_gpaw_defect_entry(
                    defect_dir, 
                    self.bulk_path, 
                    dielectric=self.dielectric, 
                    charge_state=charge_state
                )
                
                # Apply Kumagai correction if possible
                if self.dielectric is not None and charge_state != 0:
                    try:
                        defect_entry.get_kumagai_correction()
                    except Exception as e:
                        print(f"Warning: Kumagai correction failed for {folder}: {e}")
                
                defect_dict[defect_entry.name] = defect_entry
            except Exception as e:
                print(f"Failed to parse {folder}: {e}")
                
        return defect_dict