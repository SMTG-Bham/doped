"""
Code to generate and parse GPAW defect calculations.
"""

import copy
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import RegularGridInterpolator

from doped.analysis import defect_from_structures
from doped.core import Defect, DefectEntry
from doped.utils.parsing import _get_defect_supercell

_GPAW_OUTPUT_PRIORITY = (
    "relaxed.gpw.gz",
    "relaxed.gpw",
    "singlepoint.gpw.gz",
    "singlepoint.gpw",
    "final.gpw.gz",
    "final.gpw",
)


def _find_gpaw_output(
    output_path: str | os.PathLike,
    subfolder: str | os.PathLike | None = None,
) -> str:
    """
    Find a GPAW restart file in a calculation directory.

    If multiple restart files are present, standard relaxation, single-point,
    and final-state filenames are preferred in that order.
    """
    calc_path = Path(output_path)
    if calc_path.is_file():
        if calc_path.name.lower().endswith((".gpw", ".gpw.gz")):
            return str(calc_path)
        raise ValueError(f"GPAW output must be a '.gpw' or '.gpw.gz' file: {calc_path}")

    if subfolder is not None and subfolder != ".":
        calc_path /= Path(subfolder)

    if not calc_path.is_dir():
        raise FileNotFoundError(f"GPAW calculation directory not found: {calc_path}")

    gpw_files = [
        path
        for path in calc_path.iterdir()
        if path.is_file() and path.name.lower().endswith((".gpw", ".gpw.gz"))
    ]
    files_by_name = {path.name.lower(): path for path in gpw_files}
    for preferred_name in _GPAW_OUTPUT_PRIORITY:
        if preferred_name in files_by_name:
            return str(files_by_name[preferred_name])

    if len(gpw_files) == 1:
        return str(gpw_files[0])
    if not gpw_files:
        raise FileNotFoundError(f"No '.gpw' or '.gpw.gz' file found in: {calc_path}")

    filenames = ", ".join(sorted(path.name for path in gpw_files))
    raise ValueError(
        f"Multiple GPAW output files found in {calc_path}, with no preferred filename: {filenames}"
    )


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
                - "legacy_gpaw": True
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
        legacy_gpaw = settings.pop("legacy_gpaw", True)
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
calc.write('relaxed.gpw.gz')
"""
        else:
            calculation_block = """
# Static single-point calculation
energy = atoms.get_potential_energy()
calc.write('singlepoint.gpw.gz')
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
    legacy_gpaw={legacy_gpaw},
    spinpol={spinpol}{other_kwargs}
)

atoms.calc = calc

print("Starting calculation...")
{calculation_block}
print(f"Final Energy: {{energy}} eV")
"""


def _get_site_potentials_from_calc(calc, beta_bohr: float = 1.5) -> np.ndarray:
    """
    Helper to extract site potentials from a GPAW calculator using Gaussian
    spherical averaging in reciprocal space.
    """
    atoms = calc.get_atoms()
    v_ext = calc.get_electrostatic_potential()  # 3D grid in eV natively
    nx, ny, nz = v_ext.shape

    # Setup reciprocal lattice and broadening
    ang_to_bohr = 1.8897259886
    reci_cell = atoms.cell.reciprocal() * 2 * np.pi

    dgx = np.linalg.norm(reci_cell[0]) / ang_to_bohr
    dgy = np.linalg.norm(reci_cell[1]) / ang_to_bohr
    dgz = np.linalg.norm(reci_cell[2]) / ang_to_bohr

    gx = np.roll(np.arange(-nx // 2, nx // 2, 1, dtype=int), int(nx // 2)) * dgx
    gy = np.roll(np.arange(-ny // 2, ny // 2, 1, dtype=int), int(ny // 2)) * dgy
    gz = np.roll(np.arange(-nz // 2, nz // 2, 1, dtype=int), int(nz // 2)) * dgz

    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    g2 = Gx**2 + Gy**2 + Gz**2

    # Gaussian averaging via FFT
    gaussian = np.exp(-0.5 * (beta_bohr**2) * g2)

    v_G = np.fft.fftn(v_ext)
    v_G *= gaussian
    smoothed_potential = np.real(np.fft.ifftn(v_G))

    # Robust Parsing Logic
    xpoints = np.linspace(0.0, 1.0, nx, endpoint=False)
    ypoints = np.linspace(0.0, 1.0, ny, endpoint=False)
    zpoints = np.linspace(0.0, 1.0, nz, endpoint=False)

    # pad the grid with periodic images so (cubic) interpolation works at cell boundaries:
    xpoints_padded = np.concatenate([xpoints[-1:] - 1.0, xpoints, xpoints[:1] + 1.0])
    ypoints_padded = np.concatenate([ypoints[-1:] - 1.0, ypoints, ypoints[:1] + 1.0])
    zpoints_padded = np.concatenate([zpoints[-1:] - 1.0, zpoints, zpoints[:1] + 1.0])

    padded = np.concatenate(
        [smoothed_potential[-1:, :, :], smoothed_potential, smoothed_potential[:1, :, :]], axis=0
    )
    padded = np.concatenate([padded[:, -1:, :], padded, padded[:, :1, :]], axis=1)
    padded = np.concatenate([padded[:, :, -1:], padded, padded[:, :, :1]], axis=2)

    interpolator = RegularGridInterpolator(
        (xpoints_padded, ypoints_padded, zpoints_padded),
        padded,
        method="cubic",
        bounds_error=True,
    )

    atomic_site_potentials = np.zeros(len(atoms))
    for i, frac in enumerate(atoms.get_scaled_positions()):
        # Need to use fractional coordinates modulo 1.0 to interpolate the potentials
        atomic_site_potentials[i] = float(interpolator(frac % 1.0)[0])

    return atomic_site_potentials


def _get_planar_averaged_potential_from_calc(calc) -> dict[str, np.ndarray]:
    """
    Helper to extract planar-averaged potentials from a GPAW calculator.
    """
    v_ext = calc.get_electrostatic_potential()
    planar_averages = {}
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        planar_averages[str(i)] = v_ext.mean(axis=tuple(axes))

    return planar_averages


def get_gpaw_site_potentials(
    gpw_file: str | os.PathLike,
    legacy_gpaw: bool = True,
) -> np.ndarray:
    """
    Extracts atomic site potentials from a ``GPAW`` ``.gpw(.gz)`` file.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(gpw_file)
    calc = GPAW(gpw_file, legacy_gpaw=legacy_gpaw)
    site_potentials = _get_site_potentials_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    if hasattr(calc, "atoms") and calc.atoms:
        calc.atoms.calc = None

    return site_potentials


def get_gpaw_planar_averaged_potential(
    gpw_file: str | os.PathLike,
    legacy_gpaw: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts planar-averaged potential from a ``GPAW`` ``.gpw(.gz)`` file.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(gpw_file)
    calc = GPAW(gpw_file, legacy_gpaw=legacy_gpaw)
    planar_averages = _get_planar_averaged_potential_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    return planar_averages


class GPAWParser:
    """
    Parser for GPAW calculations to interface with doped.

    Note:
        The Kumagai (eFNV) finite-size charge correction is applied by default
        during parsing, as it is generally preferred. However, the standard
        Freysoldt (FNV) correction is also fully supported. If preferred, users
        can manually apply it to the parsed defects using:
        `defect_entry.get_freysoldt_correction()`
    """

    def __init__(
        self,
        gpw_file: str | os.PathLike,
        legacy_gpaw: bool = True,
    ):
        """
        Args:
            gpw_file (str): Path to ``GPAW`` ``.gpw(.gz)`` file.
            legacy_gpaw (bool):
                Whether to use GPAW's legacy calculator implementation. Defaults
                to ``True``.
        """
        from gpaw import GPAW

        self.gpw_file = _find_gpaw_output(gpw_file)
        self.calc = GPAW(self.gpw_file, legacy_gpaw=legacy_gpaw)
        self.atoms = self.calc.get_atoms()
        self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        self.energy = self.calc.get_potential_energy()

        # Pull charge directly from calculation parameters
        try:
            self.charge = self.calc.parameters.get("charge", None)
        except Exception:
            self.charge = None

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
        return _get_site_potentials_from_calc(self.calc)

    def get_locpot_dict(self) -> dict[str, np.ndarray]:
        """
        Returns planar-averaged potential dictionary.
        """
        return _get_planar_averaged_potential_from_calc(self.calc)

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

    def close(self):
        """
        Closes the underlying GPAW calculator.
        """
        if hasattr(self.calc, "close"):
            self.calc.close()

        # Break reference cycle
        if self.atoms:
            self.atoms.calc = None
        self.calc = None
        self.atoms = None


def _get_gpaw_bulk_data(bulk_parser: GPAWParser, bulk_path: str | os.PathLike) -> dict[str, Any]:
    """
    Parse reusable bulk reference data once.
    """
    band_gap, cbm, vbm, efermi = bulk_parser.get_eigenvalue_properties()
    return {
        "bulk_entry": bulk_parser.get_computed_structure_entry(),
        "bulk_site_potentials": bulk_parser.get_site_potentials(),
        "bulk_locpot_dict": bulk_parser.get_locpot_dict(),
        "bulk_path": str(bulk_path),
        "vbm": vbm,
        "band_gap": band_gap,
        "cbm": cbm,
        "efermi": efermi,
    }


def _get_gpaw_defect_entry_from_parsers(
    defect_parser: GPAWParser,
    bulk_parser: GPAWParser,
    defect_path: str | os.PathLike,
    dielectric: float | np.ndarray | None,
    charge_state: int,
    bulk_data: dict[str, Any],
) -> DefectEntry:
    """
    Build a defect entry from already-open GPAW parsers.
    """
    defect = defect_from_structures(
        defect_supercell=defect_parser.structure,
        bulk_supercell=bulk_parser.structure,
        _parameter_order_warn=False,
    )
    assert isinstance(defect, Defect)  # typing

    return DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=defect_parser.get_computed_structure_entry(),
        bulk_entry=bulk_data["bulk_entry"],
        sc_defect_frac_coords=defect.site.frac_coords,
        defect_supercell=defect_parser.structure,
        bulk_supercell=bulk_parser.structure,
        defect_supercell_site=defect.site,
        calculation_metadata={
            "bulk_path": bulk_data["bulk_path"],
            "defect_path": str(defect_path),
            "dielectric": dielectric,
            "bulk_site_potentials": bulk_data["bulk_site_potentials"],
            "defect_site_potentials": defect_parser.get_site_potentials(),
            "bulk_locpot_dict": bulk_data["bulk_locpot_dict"],
            "defect_locpot_dict": defect_parser.get_locpot_dict(),
            "vbm": bulk_data["vbm"],
            "band_gap": bulk_data["band_gap"],
            "cbm": bulk_data["cbm"],
            "efermi": bulk_data["efermi"],
        },
    )


def get_gpaw_defect_entry(
    defect_path: str | os.PathLike,
    bulk_path: str | os.PathLike,
    dielectric: float | np.ndarray | None = None,
    charge_state: int = 0,
    bulk_parser: GPAWParser | None = None,
    legacy_gpaw: bool = True,
) -> DefectEntry:
    """
    Create a defect entry from GPAW output files or directories.
    """
    defect_parser = GPAWParser(defect_path, legacy_gpaw=legacy_gpaw)
    close_bulk = bulk_parser is None
    if bulk_parser is None:
        bulk_parser = GPAWParser(bulk_path, legacy_gpaw=legacy_gpaw)

    try:
        bulk_data = _get_gpaw_bulk_data(bulk_parser, bulk_path)
        return _get_gpaw_defect_entry_from_parsers(
            defect_parser=defect_parser,
            bulk_parser=bulk_parser,
            defect_path=defect_path,
            dielectric=dielectric,
            charge_state=charge_state,
            bulk_data=bulk_data,
        )
    finally:
        defect_parser.close()
        if close_bulk:
            bulk_parser.close()


class GPAWDefectsParser:
    """
    Class for rapidly parsing multiple GPAW defect supercell calculations.
    """

    def __init__(
        self,
        output_path: str | os.PathLike = ".",
        dielectric: float | np.ndarray | None = None,
        subfolder: str | os.PathLike | None = None,
        bulk_path: str | os.PathLike | None = None,
        legacy_gpaw: bool = True,
    ):
        """
        Args:
            output_path (str): Path to directory containing defect folders.
            dielectric (float or matrix): Dielectric constant for corrections.
            subfolder (str): Optional subfolder within each defect folder.
            bulk_path (str): Path to bulk reference folder.
            legacy_gpaw (bool): Whether to use GPAW's legacy calculator.

        Attributes:
            defect_dict (dict): Parsed defect entries keyed by calculation folder name.
        """
        self.output_path = str(output_path)
        self.dielectric = dielectric
        self.subfolder = subfolder
        self.legacy_gpaw = legacy_gpaw

        if bulk_path is None:
            # Try to find bulk folder
            folders = [
                f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))
            ]
            bulk_folders = [f for f in folders if "bulk" in f.lower()]
            if not bulk_folders:
                raise ValueError("Could not find bulk folder. Please specify bulk_path.")
            bulk_folder = sorted(bulk_folders, key=lambda name: (name.lower() != "bulk", name))[0]
            self.bulk_path = os.path.join(self.output_path, bulk_folder)
        else:
            bulk_path = os.fspath(bulk_path)
            self.bulk_path = (
                bulk_path if os.path.isabs(bulk_path) else os.path.join(self.output_path, bulk_path)
            )

        self.defect_dict = self._parse_all()

    @staticmethod
    def _get_charge_state(folder: str, parsed_charge: int | None) -> int:
        """
        Use the GPAW charge, falling back to a signed folder-name component.
        """
        if parsed_charge is not None:
            return int(parsed_charge)
        for component in reversed(folder.split("_")):
            if component.startswith(("+", "-")):
                try:
                    return int(component)
                except ValueError:
                    pass
        return 0

    def _parse_all(self) -> dict[str, DefectEntry]:
        """
        Parse all GPAW defect calculations during initialisation.
        """
        defect_dict = {}
        folders = [
            f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))
        ]

        # Exclude bulk folder
        defect_folders = [
            f
            for f in folders
            if os.path.abspath(os.path.join(self.output_path, f)) != os.path.abspath(self.bulk_path)
        ]

        bulk_parser = GPAWParser(self.bulk_path, legacy_gpaw=self.legacy_gpaw)
        try:
            bulk_data = _get_gpaw_bulk_data(bulk_parser, self.bulk_path)
            for folder in defect_folders:
                defect_dir = os.path.join(self.output_path, folder)
                try:
                    gpw_file = _find_gpaw_output(defect_dir, self.subfolder)
                except FileNotFoundError:
                    continue
                except ValueError as exc:
                    print(f"Failed to parse {folder}: {exc}")
                    continue

                print(f"Parsing {folder}...")
                defect_parser = None
                try:
                    defect_parser = GPAWParser(gpw_file, legacy_gpaw=self.legacy_gpaw)
                    charge_state = self._get_charge_state(folder, defect_parser.charge)
                    defect_entry = _get_gpaw_defect_entry_from_parsers(
                        defect_parser=defect_parser,
                        bulk_parser=bulk_parser,
                        defect_path=os.path.dirname(gpw_file),
                        dielectric=self.dielectric,
                        charge_state=charge_state,
                        bulk_data=bulk_data,
                    )

                    if self.dielectric is not None and charge_state != 0:
                        try:
                            defect_entry.get_kumagai_correction()
                        except Exception as exc:
                            print(f"Warning: Kumagai correction failed for {folder}: {exc}")

                    defect_entry.name = folder
                    defect_dict[folder] = defect_entry
                except Exception as exc:
                    print(f"Failed to parse {folder}: {exc}")
                finally:
                    if defect_parser is not None:
                        defect_parser.close()
        finally:
            bulk_parser.close()

        return defect_dict
