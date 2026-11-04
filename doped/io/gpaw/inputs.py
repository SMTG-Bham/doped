"""
Code to generate GPAW defect calculation input files.

GPAW support is experimental. Features which are not implemented, and so
unavailable with GPAW:

- The competing phase input-set functions
  (``get_kpoint_convergence_sets()``, ``get_relaxation_sets()``,
  ``get_singlepoint_sets()``, ``write_input_sets()`` and the corresponding
  ``write_*_files()``), so ``CompetingPhases(..., calculator="gpaw")`` is
  not supported.

See the GPAW tracking issue.
"""

import os
import warnings
from typing import Any, Literal

from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.util.typing import PathLike

from doped.core import DefectEntry, _get_bulk_supercell, _get_defect_supercell
from doped.io.inputs import DefectsSetBase, _get_bulk_folder_name

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_gpaw_settings = loadfn(os.path.join(_MODULE_DIR, "GPAW_sets/GPAW_DefectSet.yaml"))
"""
Default GPAW settings for defect and bulk supercell calculations, from
|GPAW_DefectSet.yaml|.
"""

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
    calculator parameters), and paths (e.g. for ``txt``) to strings, so that
    they can be written to the generated scripts and serialised to JSON.
    """
    return {
        key: (
            value.todict()
            if hasattr(value, "todict")
            else os.fspath(value)
            if isinstance(value, os.PathLike)
            else value
        )
        for key, value in (gpaw_settings or {}).items()
    }


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
                Dictionary of GPAW settings, which override the defaults in
                |GPAW_DefectSet.yaml| (:data:`default_gpaw_settings`); e.g. set
                ``"occupations": None`` for GPAW's own default smearing
                (Fermi-Dirac smearing of 0.1 eV for periodic systems, intended
                for metals) rather than Gaussian smearing of 0.05 eV.
                Relaxations use the ``"fmax"`` and ``"optimizer"`` settings;
                any ``ase.optimize`` (or ``ase.optimize.precon``) optimizer, as
                a name or a dict of its name and keyword arguments, with the
                default ``PreconLBFGS`` settings tuned for defect relaxations
                in https://github.com/ddmms/ml-peg/pull/863. GPAW parameter
                objects (e.g. ``PW(400)``) are converted to their ``todict()``
                forms. Initial magnetic moments can be set with
                ``"initial_magnetic_moments"``, as one value per site in a list
                matching the structure order, or as a ``{element: moment}``
                dict (unlisted elements set to zero).
            calculation_type (str):
                Type of calculation script to generate. Supported values are
                "relax" (default) and "singlepoint". The relaxation script
                (``relax.py``) continues from the last frame of ``relax.traj``
                if present (e.g. when resubmitting a timed-out calculation),
                otherwise starting from ``structure.cif``.
        """
        # (settings which would be repeated in the generated ``GPAW()`` & optimizer calls):
        if "charge" in (gpaw_settings or {}):
            raise ValueError("Set the charge with `charge_state`, not gpaw_settings['charge'].")
        optimizer = (gpaw_settings or {}).get("optimizer")
        if isinstance(optimizer, dict) and {"trajectory", "append_trajectory"} & optimizer.keys():
            raise ValueError(
                "The relaxation trajectory (relax.traj, which is continued if present) is set by doped, "
                "so cannot be set in gpaw_settings['optimizer']."
            )
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
        settings = default_gpaw_settings | self.gpaw_settings | {"charge": self.charge_state}
        # settings used by the script itself, with the rest passed directly to ``GPAW()``:
        optimizer, fmax = settings.pop("optimizer"), settings.pop("fmax")
        magmoms = settings.pop("initial_magnetic_moments", None)
        if isinstance(magmoms, dict):  # {element: moment}, expanded for this supercell
            magmoms = [magmoms.get(site.specie.symbol, 0.0) for site in self.structure]
        if magmoms is not None and len(magmoms) != len(self.structure):
            raise ValueError(
                f"initial_magnetic_moments must contain one value per atom ({len(self.structure)} values "
                f"required)"
            )
        magmoms_line = (
            "" if magmoms is None else f"atoms.set_initial_magnetic_moments({list(magmoms)!r})\n"
        )
        gpaw_kwargs = ",\n    ".join(f"{key}={value!r}" for key, value in settings.items())

        # GPAW only supports these hybrid functionals in plane-wave mode (failing at run time otherwise, or
        # silently omitting the exact exchange in FD mode, as of GPAW 26.7.1b1):
        xc, mode = (  # (names, or dicts of their names & parameters)
            value if isinstance(value, dict) else {"name": value}
            for value in (settings.get("xc"), settings.get("mode"))
        )
        # GPAW uses its PW-only hybrid implementation for these, or any with the ``"pw"`` xc backend:
        needs_pw_mode = xc.get("name") in {"HSE03", "HSE06", "YS-PBE0"} or xc.get("backend") == "pw"
        if needs_pw_mode and mode.get("name") != "pw":
            raise ValueError(
                f"GPAW only supports the {xc.get('name')} hybrid functional in plane-wave mode, so set "
                f"'mode' to 'pw', or 'xc' to a non-hybrid functional (e.g. 'PBE'; the default is HSE06), "
                f"in gpaw_settings."
            )

        if self.calculation_type == "relax":
            optimizer = optimizer.copy() if isinstance(optimizer, dict) else {"name": optimizer}
            name = optimizer.pop("name")
            module = "ase.optimize.precon" if name.startswith("Precon") else "ase.optimize"
            extra_imports = f"from {module} import {name}\n"
            optimizer_kwargs = "".join(f"{key}={value!r}, " for key, value in optimizer.items())
            dyn = f"dyn = {name}(atoms, {optimizer_kwargs}trajectory='relax.traj', append_trajectory=True)"
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
{dyn}
try:
    dyn.run(fmax={fmax})
except RuntimeError as exc:  # e.g. a failed line search (from SCF noise), so restart the optimizer
    print(f"{{exc}}; restarting the optimizer")
    {dyn}
    dyn.run(fmax={fmax})

# Save the final state
energy = atoms.get_potential_energy()
calc.write('relaxed.gpw')
"""
        else:
            extra_imports = ""
            read_structure = f"atoms = read('{structure_filename}')"
            calculation_block = """
# Static single-point calculation
energy = atoms.get_potential_energy()
calc.write('singlepoint.gpw')
"""

        return f"""
import os

from ase.io import read
from gpaw import GPAW
{extra_imports}

# Read structure
{read_structure}
{magmoms_line}

# Setup calculator
calc = GPAW(
    {gpaw_kwargs},
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
