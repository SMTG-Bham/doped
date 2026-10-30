"""
Code to generate GPAW defect calculation input files.

GPAW support is experimental. Features which are not implemented, and so
unavailable with GPAW:

- The competing phase input-set functions
  (``get_kpoint_convergence_sets()``, ``get_relaxation_sets()``,
  ``get_singlepoint_sets()``, ``write_input_sets()`` and the corresponding
  ``write_*_files()``), so ``CompetingPhases(..., calculator="gpaw")`` is
  not supported.
- Default calculation parameters live hard-coded in
  ``DefectDictSet._generate_script()``, rather than in data files
  alongside this module (as with ``doped/io/vasp/VASP_sets``), so they
  cannot be inspected or overridden as a set.

See the GPAW tracking issue.
"""

import copy
import os
import warnings
from typing import Any, Literal

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry, _get_bulk_supercell, _get_defect_supercell
from doped.io.inputs import DefectsSetBase, _get_bulk_folder_name

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


def _gpaw_settings_to_dicts(gpaw_settings: dict[str, Any] | None) -> dict[str, Any]:
    """
    Convert GPAW parameter objects in ``gpaw_settings`` (e.g. ``PW(400)`` or
    ``Davidson(3)``) to their ``todict()`` forms (as ASE does when saving
    calculator parameters), so that they can be written to the generated
    scripts and serialised to JSON.
    """
    return {k: v.todict() if hasattr(v, "todict") else v for k, v in (gpaw_settings or {}).items()}


class DefectDictSet:
    """
    Class for generating the input files (``structure.cif`` and a GPAW Python
    script) for a single defect supercell calculation with GPAW.
    """

    def __init__(
        self,
        structure: Structure,
        charge_state: int = 0,
        gpaw_settings: dict[str, Any] | None = None,
        calculation_type: Literal["relax", "singlepoint"] = "relax",
    ):
        """
        Args:
            structure (|Structure|):
                ``pymatgen`` |Structure| object of the defect supercell.
            charge_state (int):
                Charge state of the defect, which sets GPAW's ``charge``
                parameter. Default is 0.
            gpaw_settings (dict):
                Dictionary of GPAW settings. GPAW parameter objects (e.g.
                ``PW(400)``) are converted to their ``todict()`` forms.
                Defaults used if not specified:
                - "mode": {"name": "pw", "ecut": 400}
                - "xc": "PBE"
                - "kpts": {"size": (1, 1, 1), "gamma": True}
                - "txt": "gpaw_output.txt"
                - "spinpol": True
                - "fmax": 0.05
                - "optimizer": "BFGS" (any ``ase.optimize`` optimizer)
                - "initial_magnetic_moments": None; one value per site in a
                  list matching the structure order, or a ``{element: moment}``
                  dict (unlisted elements set to zero)
            calculation_type (str):
                Type of calculation script to generate. Supported values are
                "relax" (default) and "singlepoint". The relaxation script
                (``relax.py``) continues from the last frame of ``relax.traj``
                if present (e.g. when resubmitting a timed-out calculation),
                otherwise starting from ``structure.cif``.
        """
        if "charge" in (gpaw_settings or {}):  # would be repeated in the generated ``GPAW()`` call
            raise ValueError("Set the charge with `charge_state`, not gpaw_settings['charge'].")
        if calculation_type not in {"relax", "singlepoint"}:
            raise ValueError("calculation_type must be 'relax' or 'singlepoint'")

        self.structure = structure
        self.charge_state = charge_state
        self.gpaw_settings = _gpaw_settings_to_dicts(gpaw_settings)
        self.calculation_type = calculation_type

    def write_input(
        self,
        output_path: str,
        filename: str | None = None,
        make_dir_if_not_present: bool = True,
    ):
        """
        Writes the input files (structure and script) to a directory.
        """
        # Generate the Python script first, so that invalid settings raise before anything is written
        structure_filename = "structure.cif"
        script_content = self._generate_script(structure_filename)

        if make_dir_if_not_present:
            os.makedirs(output_path, exist_ok=True)
        elif not os.path.isdir(output_path):
            raise FileNotFoundError(f"Output directory does not exist: {output_path}")

        if filename is None:
            filename = f"{self.calculation_type}.py"

        trajectory = os.path.join(output_path, "relax.traj")
        if self.calculation_type == "relax" and os.path.exists(trajectory):
            warnings.warn(
                f"{trajectory} already exists, so the relaxation will continue from its last frame (if "
                f"any) rather than the newly-written {structure_filename}. Delete it to start from "
                f"{structure_filename}."
            )

        # Write structure to a file
        writer = CifWriter(self.structure)  # no symprec arg -- reduces supercells to primitives
        writer.write_file(os.path.join(output_path, structure_filename))

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
        txt = os.fspath(txt) if isinstance(txt, os.PathLike) else txt  # e.g. ``Path``, as a plain string
        convergence = settings.pop("convergence", {})
        optimizer = settings.pop("optimizer", "BFGS")
        initial_magnetic_moments = settings.pop("initial_magnetic_moments", None)

        magnetic_moments_line = ""
        if isinstance(initial_magnetic_moments, dict):  # {element: moment}, expanded for this supercell
            initial_magnetic_moments = [
                initial_magnetic_moments.get(site.specie.symbol, 0.0) for site in self.structure
            ]
        if initial_magnetic_moments is not None:
            if len(initial_magnetic_moments) != len(self.structure):
                raise ValueError(
                    "initial_magnetic_moments must contain one value per atom "
                    f"({len(self.structure)} values required)"
                )
            magnetic_moments_line = (
                f"atoms.set_initial_magnetic_moments({list(initial_magnetic_moments)!r})\n"
            )

        spinpol = settings.pop("spinpol", True)

        # Relaxation params
        fmax = settings.pop("fmax", 0.05)

        # ``basis`` belongs to GPAW, not to the mode classes (none of ``PW``/``LCAO``/``FD`` accept it).
        # Accept the existing nested form as well as the native top-level setting:
        if isinstance(mode_params, dict):
            name = mode_params.pop("name", "pw")
            if "basis" in mode_params:
                basis = mode_params.pop("basis")
                if "basis" in settings and settings["basis"] != basis:
                    raise ValueError(
                        f"Conflicting basis settings: {basis!r} in gpaw_settings['mode'] and "
                        f"{settings['basis']!r} in gpaw_settings['basis']. Set the basis in one place "
                        "only (preferably gpaw_settings['basis'])."
                    )
                settings["basis"] = basis
            args = ", ".join([f"{k}={v!r}" for k, v in mode_params.items()])
            mode_str = f"{name.upper()}({args})"
        else:
            mode_str = repr(mode_params)

        # Prepare other settings
        other_kwargs = ""
        if settings:
            other_kwargs = ",\n    " + ",\n    ".join([f"{k}={v!r}" for k, v in settings.items()])

        if self.calculation_type == "relax":
            optimizer_import = f"from ase.optimize import {optimizer}\n"
            # Continue from the last frame of a previous (e.g. timed-out) relaxation if present:
            read_structure = (
                "frames = read('relax.traj', ':') if os.path.isfile('relax.traj') and "
                # runs interrupted before the first step may leave empty or header-only trajectories,
                # depending on the ASE version:
                "os.path.getsize('relax.traj') else []\n"
                f"atoms = frames[-1] if frames else read('{structure_filename}')"
            )
            calculation_block = f"""
# Relaxation
dyn = {optimizer}(atoms, trajectory='relax.traj', append_trajectory=True)
dyn.run(fmax={fmax})

# Save the final state
energy = atoms.get_potential_energy()
calc.write('relaxed.gpw')
"""
        else:
            optimizer_import = ""
            read_structure = f"atoms = read('{structure_filename}')"
            calculation_block = """
# Static single-point calculation
energy = atoms.get_potential_energy()
calc.write('singlepoint.gpw')
"""

        return f"""
import os

from ase.io import read
from gpaw import GPAW, PW, LCAO, FD
{optimizer_import}

# Read structure
{read_structure}
{magnetic_moments_line}

# Setup calculator
calc = GPAW(
    mode={mode_str},
    xc={xc!r},
    kpts={kpts},
    txt={txt!r},
    convergence={convergence},
    charge={self.charge_state},
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
    |DefectsGenerator| output (or any set of |DefectEntry|\ s), in the
    ``<defect name>/`` folder structure.

    The calculator-agnostic orchestration (naming, folder structure,
    multiprocessed writing and provenance serialisation) comes from
    :class:`~doped.io.inputs.DefectsSetBase`; this only builds and writes the
    per-defect :class:`DefectDictSet`\ s.
    """

    _input_set_name = "DefectDictSet"

    def __init__(
        self,
        defect_entries,
        gpaw_settings: dict[str, Any] | None = None,
        calculation_type: Literal["relax", "singlepoint"] = "relax",
    ):
        r"""
        Args:
            defect_entries (|DefectsGenerator|, dict/list of |DefectEntry|\ s, or |DefectEntry|):
                The defect entries for which to generate GPAW calculation
                inputs; see :class:`~doped.io.inputs.DefectsSetBase`.
            gpaw_settings (dict):
                ``GPAW`` settings for the generated calculation scripts; see
                :class:`DefectDictSet`. If using ``initial_magnetic_moments``,
                it must be given as a per-element dict (``{element: moment}``).
                Default is ``None``.
            calculation_type (str):
                Type of calculation script to generate, ``"relax"`` (default)
                or ``"singlepoint"``.
        """
        initial_magnetic_moments = (gpaw_settings or {}).get("initial_magnetic_moments")
        if initial_magnetic_moments is not None and not isinstance(initial_magnetic_moments, dict):
            raise ValueError(
                "`initial_magnetic_moments` must be a {element: moment} dict with `DefectsSet`. Use "
                "`DefectDictSet` directly if you wish to use site-resolved initial magnetic moments."
            )

        self.gpaw_settings = _gpaw_settings_to_dicts(gpaw_settings)  # JSON-serialisable, for provenance
        self.calculation_type = calculation_type
        super().__init__(defect_entries)

    def _defect_input_set(self, defect_entry: DefectEntry) -> DefectDictSet:
        """
        Build the :class:`DefectDictSet` for a single defect entry.
        """
        return DefectDictSet(
            _get_defect_supercell(defect_entry),
            charge_state=defect_entry.charge_state,
            gpaw_settings=self.gpaw_settings,
            calculation_type=self.calculation_type,
        )

    @staticmethod
    def _write_defect(args: tuple) -> None:
        """
        Write the GPAW input files for a single defect.
        """
        defect_species, defect_input_set, output_path, _bulk, write_kwargs = args
        defect_input_set.write_input(os.path.join(output_path, defect_species), **dict(write_kwargs))

    def write_files(  # type: ignore[override]  # narrows the base signature's ``bulk`` (to a bool)
        self,
        output_path: PathLike = ".",
        bulk: bool = True,
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
            bulk (bool):
                Whether to also write inputs for the (singlepoint) reference
                bulk supercell calculation, to ``{formula}_bulk``. Default is
                ``True``.
            processes (int):
                Number of processes to use for multiprocessed file writing.
                Default (``None``) sets this automatically.
            **kwargs:
                Additional keyword arguments for
                :meth:`DefectDictSet.write_input`.
        """
        super().write_files(output_path=output_path, bulk=False, processes=processes, **kwargs)
        if bulk:  # write the neutral bulk reference once, with the same settings
            bulk_supercell = _get_bulk_supercell(next(iter(self.defect_entries.values())))
            DefectDictSet(
                bulk_supercell,
                gpaw_settings=self.gpaw_settings,
                calculation_type="singlepoint",  # static reference (nothing to relax)
            ).write_input(os.path.join(output_path, _get_bulk_folder_name(bulk_supercell)), **kwargs)
