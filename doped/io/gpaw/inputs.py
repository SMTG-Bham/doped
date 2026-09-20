"""
Code to generate GPAW defect calculation input files.
"""

import copy
import os
from typing import Any, Literal

from pymatgen.core.structure import Structure

from doped.core import DefectEntry, _get_defect_supercell


class GPAWDefectRelaxSet:
    """
    Class for generating input files (Python scripts) for GPAW defect
    relaxation.
    """

    def __init__(
        self,
        defect_entry: DefectEntry | Structure,
        charge_state: int | None = None,
        gpaw_settings: dict[str, Any] | None = None,
        calculation_type: Literal["relax", "singlepoint"] = "relax",
        **kwargs,
    ):
        """
        Args:
            defect_entry (DefectEntry, Structure):
                doped/pymatgen DefectEntry or Structure object.
            charge_state (int):
                Charge state of the defect. Overrides DefectEntry.charge_state.
            gpaw_settings (dict):
                Dictionary of GPAW settings. Defaults used if not specified:
                - "mode": {"name": "pw", "ecut": 400}
                - "xc": "PBE"
                - "kpts": {"size": (1, 1, 1), "gamma": True}
                - "txt": "gpaw_output.txt"
                - "spinpol": True
                - "fmax": 0.05
                - "optimizer": "BFGS"
                - "initial_magnetic_moments": None
            calculation_type (str):
                Type of calculation script to generate. Supported values are
                "relax" (default) and "singlepoint".
            **kwargs:
                Additional keyword arguments.
        """
        self.defect_entry = defect_entry
        self.charge_state = charge_state
        if self.charge_state is None:
            self.charge_state = kwargs.get("charge")  # Catch it if passed as kwarg
        if self.charge_state is None and isinstance(self.defect_entry, DefectEntry):
            self.charge_state = self.defect_entry.charge_state

        if calculation_type not in {"relax", "singlepoint"}:
            raise ValueError("calculation_type must be 'relax' or 'singlepoint'")

        self.gpaw_settings = gpaw_settings or {}
        self.calculation_type = calculation_type
        self.kwargs = kwargs

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)

    def write_input(
        self,
        output_path: str,
        filename: str | None = None,
        make_dir_if_not_present: bool = True,
    ):
        """
        Writes the input files (structure and script) to a directory.
        """
        if make_dir_if_not_present:
            os.makedirs(output_path, exist_ok=True)
        elif not os.path.isdir(output_path):
            raise FileNotFoundError(f"Output directory does not exist: {output_path}")

        if filename is None:
            filename = f"{self.calculation_type}.py"

        # Write structure to a file
        structure_filename = "structure.cif"

        from pymatgen.io.cif import CifWriter

        # Do not use symprec arg inside CifWriter. It reduces supercells to primitives.
        writer = CifWriter(self.defect_supercell)
        writer.write_file(os.path.join(output_path, structure_filename))

        # Generate Python script
        script_content = self._generate_script(structure_filename)

        with open(os.path.join(output_path, filename), "w") as f:
            f.write(script_content)

    def _generate_script(self, structure_filename: str) -> str:
        """
        Generates the content of the GPAW script.
        """
        settings = copy.deepcopy(self.gpaw_settings)

        # Extract known parameters
        mode_params = settings.pop("mode", {"name": "pw", "ecut": 400})
        xc = settings.pop("xc", "PBE")
        kpts = settings.pop("kpts", {"size": (1, 1, 1), "gamma": True})
        txt = settings.pop("txt", "gpaw_output.txt")
        convergence = settings.pop("convergence", {})
        optimizer = settings.pop("optimizer", "BFGS")
        initial_magnetic_moments = settings.pop("initial_magnetic_moments", None)

        magnetic_moments_line = ""
        if initial_magnetic_moments is not None:
            if len(initial_magnetic_moments) != len(self.defect_supercell):
                raise ValueError(
                    "initial_magnetic_moments must contain one value per atom "
                    f"({len(self.defect_supercell)} values required)"
                )
            magnetic_moments_line = (
                f"atoms.set_initial_magnetic_moments({list(initial_magnetic_moments)!r})\n"
            )

        # Determine charge
        charge = self.charge_state or 0

        # Determine spinpol (default True for defects if not specified)
        spinpol = settings.pop("spinpol", True)

        # Relaxation params
        fmax = settings.pop("fmax", 0.05)

        supported_optimizers = {"BFGS", "FIRE", "LBFGS", "QuasiNewton"}
        if self.calculation_type == "relax" and optimizer not in supported_optimizers:
            raise ValueError(
                f"Unsupported optimizer {optimizer!r}. Choose one of: {sorted(supported_optimizers)}"
            )

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

        optimizer_import = (
            f"from ase.optimize import {optimizer}\n" if self.calculation_type == "relax" else ""
        )
        if self.calculation_type == "relax":
            calculation_block = f"""
# Relaxation
dyn = {optimizer}(atoms, trajectory='relax.traj')
dyn.run(fmax={fmax})

# Save the final state
energy = atoms.get_potential_energy()
calc.write('relaxed.gpw')
"""
        else:
            calculation_block = """
# Static single-point calculation
energy = atoms.get_potential_energy()
calc.write('singlepoint.gpw')
"""

        return f"""
from ase.io import read
from gpaw import GPAW, PW, LCAO, FD
{optimizer_import}

# Read structure
atoms = read('{structure_filename}')
{magnetic_moments_line}

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
{calculation_block}
print(f"Final Energy: {{energy}} eV")
"""
