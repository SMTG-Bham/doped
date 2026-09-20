"""
Code to generate GPAW defect calculation input files.

GPAW support is experimental. Features which are not implemented, and so
unavailable with GPAW:

- :class:`DefectRelaxSet` is a standalone class, rather than a
  :class:`~doped.io.inputs.DefectsSetBase` subclass, so the ``DefectsSet``
  workflow (per-defect input sets for a full ``DefectsGenerator`` output,
  folder-structure writing, rattling, provenance serialisation) is
  unavailable, as is ``DefectsGenerator``-driven input writing.
- The competing phase input-set functions
  (``get_kpoint_convergence_sets()``, ``get_relaxation_sets()``,
  ``get_singlepoint_sets()``, ``write_input_sets()`` and the corresponding
  ``write_*_files()``), so ``CompetingPhases(..., calculator="gpaw")`` is
  not supported.
- Default calculation parameters live hard-coded in
  ``DefectRelaxSet._generate_script()``, rather than in data files
  alongside this module (as with ``doped/io/vasp/VASP_sets``), so they
  cannot be inspected or overridden as a set.

See the GPAW tracking issue.
"""

import copy
import os
from typing import Any, Literal

from pymatgen.core.structure import Structure
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry, _get_bulk_supercell, _get_defect_supercell
from doped.io.inputs import DefectsSetBase

# ``doped`` accesses the competing phase functions below directly on this backend module (and
# ``write_input_sets`` is the documented protocol name a user would reach for), so they are intercepted
# here to fail informatively -- e.g. for ``CompetingPhases(..., calculator="gpaw")`` -- rather than with
# an obscure ``AttributeError`` (see the module docstring, and ``doped.io.gpaw.outputs``):
_UNIMPLEMENTED_BACKEND_ATTRS = (
    "get_kpoint_convergence_sets",
    "get_relaxation_sets",
    "get_singlepoint_sets",
    "write_input_sets",
    "write_kpoint_convergence_files",
    "write_relaxation_files",
    "write_singlepoint_files",
)


def __getattr__(name: str) -> Any:
    """
    Raise an informative error for the ``doped.io`` backend protocol
    functions/classes which are not implemented for GPAW (see the module
    docstring).
    """
    if name in _UNIMPLEMENTED_BACKEND_ATTRS:
        raise NotImplementedError(
            f"`{__name__}.{name}` is not implemented; GPAW support in `doped` is experimental, and "
            f"does not yet cover competing phase input generation. Defect supercell inputs are "
            f"generated as usual, with `doped.io.gpaw.inputs.DefectsSet`. See the GPAW tracking issue."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class DefectRelaxSet:
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
        # note that unrecognised ``**kwargs`` are silently swallowed here (only ``charge`` is consumed,
        # above), rather than raising as ``doped``'s VASP input sets do. See the GPAW tracking issue:
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

        # Prepare mode string; note that the ``mode`` sub-parameters are written through unvalidated, so
        # arguments which the GPAW mode classes do not accept give a script which fails at runtime. In
        # particular ``LCAO(basis=...)`` raises ``TypeError`` in GPAW, so generated LCAO scripts are
        # invalid (and ``tests/test_gpaw.py`` asserts that string). See the GPAW tracking issue:
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


class DefectsSet(DefectsSetBase):
    r"""
    Generate GPAW calculation input files for all defect supercells in a
    :class:`~doped.generation.DefectsGenerator` output (or any set of
    |DefectEntry|\ s), in the ``<defect name>/`` folder structure.

    The calculator-agnostic orchestration (naming, folder structure,
    multiprocessed writing and provenance serialisation) comes from
    :class:`~doped.io.inputs.DefectsSetBase`; this only builds and writes the
    per-defect :class:`DefectRelaxSet`\ s.
    """

    _input_set_name = "DefectRelaxSet"

    def __init__(
        self,
        defect_entries,
        gpaw_settings: dict[str, Any] | None = None,
        calculation_type: Literal["relax", "singlepoint"] = "relax",
        **kwargs,
    ):
        r"""
        Args:
            defect_entries (|DefectsGenerator|, dict/list of |DefectEntry|\ s, or |DefectEntry|):
                The defect entries for which to generate GPAW calculation
                inputs; see :class:`~doped.io.inputs.DefectsSetBase`.
            gpaw_settings (dict):
                ``GPAW`` settings for the generated calculation scripts; see
                :class:`DefectRelaxSet`. Default is ``None``.
            calculation_type (str):
                Type of calculation script to generate, ``"relax"`` (default)
                or ``"singlepoint"``.
            **kwargs:
                Additional keyword arguments for :class:`DefectRelaxSet`.
        """
        self.gpaw_settings = gpaw_settings
        self.calculation_type = calculation_type
        super().__init__(defect_entries, **kwargs)

    def _defect_input_set(self, defect_entry: DefectEntry) -> DefectRelaxSet:
        """
        Build the :class:`DefectRelaxSet` for a single defect entry.
        """
        return DefectRelaxSet(
            defect_entry=defect_entry,
            charge_state=defect_entry.charge_state,
            gpaw_settings=self.gpaw_settings,
            calculation_type=self.calculation_type,
            **self.kwargs,
        )

    @staticmethod
    def _write_defect(args: tuple) -> None:
        """
        Write the GPAW input files for a single defect (and, for the last
        defect, the reference bulk supercell).
        """
        defect_species, defect_input_set, output_path, bulk, write_kwargs = args
        defect_input_set.write_input(os.path.join(output_path, defect_species), **dict(write_kwargs))

        if bulk:  # write the neutral bulk reference once, with the same settings
            bulk_folder = bulk if isinstance(bulk, str) else "bulk"
            DefectRelaxSet(
                _get_bulk_supercell(defect_input_set.defect_entry),
                charge_state=0,
                gpaw_settings=defect_input_set.gpaw_settings,
                calculation_type=defect_input_set.calculation_type,
            ).write_input(os.path.join(output_path, bulk_folder), **dict(write_kwargs))

    def write_files(  # type: ignore[override]  # narrows the base signature's ``**kwargs``
        self,
        output_path: PathLike = ".",
        bulk: bool | str = True,
        processes: int | None = None,
        **kwargs,
    ):
        """
        Write GPAW input files (a calculation script and ``structure.cif``) to
        ``<output_path>/<defect name>/`` for every defect entry.

        Args:
            output_path (PathLike):
                Folder in which to create the defect calculation folders.
                Default is the current directory (".").
            bulk (bool, str):
                Whether to also write inputs for the reference bulk supercell
                calculation; a string is used as the folder name (default
                ``"bulk"``). Default is ``True``.
            processes (int):
                Number of processes to use for multiprocessed file writing.
                Default (``None``) sets this automatically.
            **kwargs:
                Additional keyword arguments for
                :meth:`DefectRelaxSet.write_input`.
        """
        super().write_files(output_path=output_path, bulk=bulk, processes=processes, **kwargs)
