"""
Quantum ESPRESSO (``pw.x``) calculation input file generation for ``doped``.

Note that all structures are written with ``ibrav = 0`` and the subsequent ``CELL_PARAMETERS``
card is used to write the lattice vectors.
"""
#TODO: Input files for multi-oxidation states
import copy
import os
import warnings
from functools import lru_cache

import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.io.espresso.inputs.pwin import (
    AtomicPositionsCard,
    AtomicSpeciesCard,
    CellNamelist,
    CellParametersCard,
    ControlNamelist,
    ElectronsNamelist,
    IonsNamelist,
    KPointsCard,
    PWin,
    SystemNamelist,
)
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.util.typing import PathLike

from doped import _ignore_pmg_warnings
from doped.core import DefectEntry
from doped.generation import DefectsGenerator, _custom_formatwarning
from doped.io.inputs import DefectsSetBase
from doped.utils.efficiency import Element, Structure
from doped.utils.parsing import _get_defect_supercell

_ignore_pmg_warnings()
warnings.formatwarning = _custom_formatwarning

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
QE_SETS_DIR = os.path.join(MODULE_DIR, "QE_sets")

# placeholder written to ``&CONTROL.pseudo_dir`` until the user sets it:
DEFAULT_PSEUDO_DIR = "./pseudo_folder_name/"


# espresso ``pw.x`` input defaults, loaded from the YAML/JSON sets in ``doped/io/espresso/QE_sets``
default_qe_set: dict = loadfn(os.path.join(QE_SETS_DIR, "QE_Convergence_set.yaml"))
default_qe_HSE_set: dict = loadfn(os.path.join(QE_SETS_DIR, "HSE_set.yaml"))
# ``{element: UPF filename}`` defaults for the ``ATOMIC_SPECIES`` card. ``doped`` does not bundle or
# require any particular pseudopotential library -- these are just the default filenames to look for
# in ``pseudo_dir``, and ``pseudo_map`` overrides them for any element.
QE_PSEUDO_LIBRARY: dict = {
    element: metadata["filename"]
    for element, metadata in loadfn(
        os.path.join(QE_SETS_DIR, "QE_PSEUDO_LIBRARY.json")
    ).items()
}


SUBFOLDER_PRIORITY: list[str] = [
    "espresso_hybrid",  # hybrid DFT (HSE06 by default); highest accuracy
    "espresso_ncl",  # spin-orbit coupling (``noncolin``/``lspinorb``; i.e. spin-orbit/'spinorb')
    "espresso_std",  # standard (semi-local, k-point mesh) fixed-cell relaxation
    "espresso_gamma",  # Γ-point-only fixed-cell relaxation; lowest accuracy
]
"""
The ``pw.x`` calculation subfolder names written by :class:`DefectRelaxSetQE` /
:class:`DefectsSetQE`, in decreasing order of accuracy -- and so the priority
order used when auto-detecting which subfolder to parse.
"""


DEFAULT_STARTING_MAGNETIZATION: float = 0.1


def _set_starting_magnetization(system_settings: dict, ntyp: int) -> dict:
    """
    Write ``&SYSTEM.starting_magnetization`` in espresso's per-type indexed form
    -- ``starting_magnetization(i)`` for ``i = 1,...,ntyp``, indexing the
    ``ATOMIC_SPECIES`` entries -- with a value set for every atomic type.

    A single (scalar) ``starting_magnetization`` is applied to every type, while
    a list/tuple is taken as the per-type values in ``ATOMIC_SPECIES`` order.
    Any ``starting_magnetization(i)`` keys already present are kept as-is, and
    any types left without a value (e.g. from a list shorter than ``ntyp``) are
    set to :data:`DEFAULT_STARTING_MAGNETIZATION` (0.1), so that spin
    polarisation is seeded on all species.

    Args:
        system_settings: The ``&SYSTEM`` namelist settings dict, updated in place.
        ntyp: Number of atomic types (``&SYSTEM.ntyp``) in the structure.
    """
    starting_magnetization = system_settings.pop("starting_magnetization", None)
    if not isinstance(starting_magnetization, list | tuple):  # scalar; applied to every type
        starting_magnetization = [starting_magnetization] * ntyp

    for idx in range(1, ntyp + 1):
        value = starting_magnetization[idx - 1] if idx <= len(starting_magnetization) else None
        if value is None:  # no per-type value given; seed all species by default
            value = DEFAULT_STARTING_MAGNETIZATION
        system_settings.setdefault(f"starting_magnetization({idx})", value)

    return system_settings


def _set_ecutrho(system_settings: dict) -> dict:
    """
    Set ``ecutrho`` to ``4 * ecutwfc`` if not already set, and warn if a
    supplied value is below that. These defaults are placeholders --
    the user must set the cutoffs according to the pseudopotentials used.
    """
    ecutwfc = system_settings.get("ecutwfc")
    if ecutwfc is None:
        return system_settings

    ecutrho = system_settings.get("ecutrho")
    if ecutrho is None:
        system_settings["ecutrho"] = 4 * ecutwfc

    elif ecutrho < 4 * ecutwfc:
        warnings.warn(
            f"`ecutrho` ({ecutrho} Ry) is less than 4x `ecutwfc` ({ecutwfc} Ry) "
        )

    return system_settings


def _kpoints_grid_from_reciprocal_density(
    structure: Structure, reciprocal_density: float
) -> list[int]:
    """``[kx, ky, kz]`` Monkhorst-Pack grid at the given k-points-per-Å^-3 density."""
    kpoints_obj = Kpoints.automatic_density_by_vol(structure, kppvol=reciprocal_density)
    return [int(k) for k in kpoints_obj.kpts[0]]


def _build_qe_base_settings(
    structure: Structure,
    pseudo_dir: str,
    is_metal: bool,
    user_control_settings: dict | None,
    user_system_settings: dict | None,
    user_electron_settings: dict | None,
    use_hse: bool = False,
    user_ions_settings: dict | None = None,
    user_cell_settings: dict | None = None,
    soc: bool = False,
) -> dict:
    """
    Build the per-structure base namelist dict from the default convergence
    YAML defaults (or the HSE06 hybrid-DFT defaults if ``use_hse``): sets
    ``ibrav=0``, ``nat``, ``ntyp``, ``pseudo_dir``, optional metallic
    smearing / spin-orbit coupling (``soc``), and merges any user overrides.

    Note that ``&IONS``/``&CELL`` overrides (``user_ions_settings`` /
    ``user_cell_settings``) only affect relaxation inputs: for the SCF
    convergence calculations espresso ignores ``&IONS`` and ``&CELL`` is not
    written, and ``&CELL`` is also dropped for fixed-cell (``relax``) inputs.
    """
    base = copy.deepcopy(default_qe_HSE_set if use_hse else default_qe_set)
    base["control"]["pseudo_dir"] = pseudo_dir
    base["control"].update(user_control_settings or {})
    base["system"].update(user_system_settings or {})
    base["electrons"].update(user_electron_settings or {})
    base["ions"].update(user_ions_settings or {})
    base["cell"].update(user_cell_settings or {})

    base["system"]["ibrav"] = 0
    base["system"]["nat"] = len(structure)
    base["system"]["ntyp"] = len(set(structure.species))

    if is_metal:
        base["system"].setdefault("occupations", "smearing")
        base["system"].setdefault("smearing", "gaussian")
        base["system"].setdefault("degauss", 0.005)

    if soc:  # spin-orbit coupling: noncolinear magnetisation with fully-relativistic pseudopotentials.
        base["system"].setdefault("noncolin", True)
        base["system"].setdefault("lspinorb", True)

    _set_ecutrho(base["system"])

    return base


class DopedDictSetQE(MSONable):
    r"""
    Input set for a single Quantum ESPRESSO (``pw.x``) calculation on a given
    structure.

    Holds the ``&CONTROL``/``&SYSTEM``/``&ELECTRONS``/``&IONS``/``&CELL``
    namelists (assembled from the bundled YAML sets plus user overrides, via
    :func:`_build_qe_base_settings`), the k-point grid and the pseudopotential
    map for one calculation, and writes them as a ``pw.in`` file.
    """

    def __init__(
        self,
        structure: Structure,
        calculation: str = "scf",
        ecutwfc: int | None = None,
        ecutrho: int | None = None,
        kpoints: list[int] | None = None,
        kpoint_density: float | None = None,
        kpoints_shift: tuple[int, int, int] = (0, 0, 0),
        pseudo_dir: str = DEFAULT_PSEUDO_DIR,
        pseudo_map: dict | None = None,
        is_metal: bool = False,
        use_hse: bool = False,
        soc: bool = False,
        starting_magnetization: float | None = None,
        user_control_settings: dict | None = None,
        user_system_settings: dict | None = None,
        user_electron_settings: dict | None = None,
        user_ions_settings: dict | None = None,
        user_cell_settings: dict | None = None,
    ):
        r"""
        Args:
            structure (Structure):
                ``pymatgen`` ``Structure`` object to write the input file for.
            calculation (str):
                espresso ``&CONTROL.calculation`` mode; ``"scf"`` (default),
                ``"relax"`` (fixed cell) or ``"vc-relax"`` (variable cell).
            ecutwfc (int):
                Plane-wave cutoff (Ry). If ``None`` (default), the YAML set
                default (60 Ry) is kept.
            ecutrho (int):
                Charge-density cutoff (Ry). If ``None`` (default), set to
                ``4 * ecutwfc``. Warns if set below ``4 * ecutwfc``.

                Both defaults are placeholders and the user must set cutoffs
                according to the pseudopotentials used.
            kpoints (list[int]):
                Explicit ``[kx, ky, kz]`` Monkhorst-Pack grid. Takes precedence
                over ``kpoint_density``; if both are ``None``, Γ-only sampling
                is used (``K_POINTS gamma``).
            kpoint_density (float):
                Reciprocal k-point density (Å^-3) from which to generate the
                grid, if ``kpoints`` is not given.
            kpoints_shift (tuple):
                ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
                ``K_POINTS automatic`` card. Default ``(0, 0, 0)``.
            pseudo_dir (str):
                Path written to ``&CONTROL.pseudo_dir``.
            pseudo_map (dict):
                ``{element: UPF filename}`` overrides on top of the bundled
                :data:`QE_PSEUDO_LIBRARY` defaults.
            is_metal (bool):
                If ``True``, set ``occupations='smearing'``,
                ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``.
            use_hse (bool):
                If ``True``, use the HSE06 hybrid-DFT defaults
                (``QE_sets/HSE_set.yaml``) as the base set instead of the (GGA)
                default (GGA) set. Default ``False``.
            soc (bool):
                If ``True``, set ``noncolin=.true.`` and ``lspinorb=.true.``
                (requires fully-relativistic pseudopotentials). Default
                ``False``.
            starting_magnetization (float):
                If not ``None``, set ``&SYSTEM.starting_magnetization`` for
                every species (and ``nspin=2``), seeding a spin-polarised
                calculation. Default ``None``.
            user_control_settings (dict):
                ``&CONTROL`` overrides merged on top of the YAML defaults.
            user_system_settings (dict):
                ``&SYSTEM`` overrides merged on top of the YAML defaults.
            user_electron_settings (dict):
                ``&ELECTRONS`` overrides merged on top of the YAML defaults.
            user_ions_settings (dict):
                ``&IONS`` overrides; only affects relaxation inputs.
            user_cell_settings (dict):
                ``&CELL`` overrides; only affects variable-cell (``vc-relax``)
                inputs, as ``&CELL`` is not written for fixed-cell runs.
        """
        self.structure = structure
        self.calculation = calculation
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self._kpoints = kpoints
        self.kpoint_density = kpoint_density
        self.kpoints_shift = kpoints_shift
        self.pseudo_dir = pseudo_dir
        self.pseudo_map = pseudo_map
        self.is_metal = is_metal
        self.use_hse = use_hse
        self.soc = soc
        self.starting_magnetization = starting_magnetization
        self.user_control_settings = user_control_settings or {}
        self.user_system_settings = user_system_settings or {}
        self.user_electron_settings = user_electron_settings or {}
        self.user_ions_settings = user_ions_settings or {}
        self.user_cell_settings = user_cell_settings or {}

    @property
    def namelists(self) -> dict:
        """
        The espresso namelist settings (``{namelist: {key: value}}``) for this
        calculation: the YAML set defaults merged with the user overrides (via
        :func:`_build_qe_base_settings`), then ``calculation``, the cutoffs and
        the spin seeding applied.
        """
        namelists = _build_qe_base_settings(
            self.structure,
            self.pseudo_dir,
            self.is_metal,
            self.user_control_settings,
            self.user_system_settings,
            self.user_electron_settings,
            use_hse=self.use_hse,
            user_ions_settings=self.user_ions_settings,
            user_cell_settings=self.user_cell_settings,
            soc=self.soc,
        )
        namelists["control"]["calculation"] = self.calculation
        if self.ecutwfc is not None:
            namelists["system"]["ecutwfc"] = self.ecutwfc
            if self.ecutrho is None and "ecutrho" not in self.user_system_settings:
                namelists["system"].pop("ecutrho", None)
        if self.ecutrho is not None:
            namelists["system"]["ecutrho"] = self.ecutrho
        _set_ecutrho(namelists["system"])
        if self.starting_magnetization is not None:
            namelists["system"]["starting_magnetization"] = self.starting_magnetization

        return namelists

    @property
    def kpoints(self) -> list[int] | None:
        """
        The ``[kx, ky, kz]`` Monkhorst-Pack grid for this calculation, or
        ``None`` for Γ-only sampling (written as ``K_POINTS gamma``).
        """
        if self._kpoints is not None:
            return list(self._kpoints)
        if self.kpoint_density is not None:
            return _kpoints_grid_from_reciprocal_density(self.structure, self.kpoint_density)

        return None

    @property
    def pseudopotentials(self) -> dict:
        """
        The ``{element: UPF filename}`` mapping written to the
        ``ATOMIC_SPECIES`` card: the :data:`QE_PSEUDO_LIBRARY` defaults, with
        ``pseudo_map`` overrides applied and a ``"{element}.upf"`` fallback
        for elements absent from it.

        Warns if these files are not present in ``pseudo_dir`` -- no
        pseudopotential library is bundled with ``doped``, so any library of
        your choosing needs to be downloaded separately, or ``pseudo_map`` used
        to point at pseudopotentials you already have.
        """
        species = sorted({str(el) for el in self.structure.species}, key=lambda s: Element(s).Z)
        pseudos = {sp: QE_PSEUDO_LIBRARY.get(sp, f"{sp}.upf") for sp in species}
        pseudos.update(self.pseudo_map or {})
        self._warn_if_pseudos_missing(self.pseudo_dir, tuple(pseudos[sp] for sp in species))

        return pseudos

    @staticmethod
    @lru_cache(maxsize=None)  # only warn once per (``pseudo_dir``, pseudopotential set); the
    # ``pseudopotentials`` property is re-evaluated for every defect in a ``DefectsSetQE``
    def _warn_if_pseudos_missing(pseudo_dir: str, pseudo_filenames: tuple[str, ...]) -> None:
        """
        Warn if the UPF pseudopotential files to be written to ``ATOMIC_SPECIES``
        are not actually present in ``pseudo_dir``.

        :data:`QE_PSEUDO_LIBRARY` only supplies the *filenames* espresso should
        look for; no pseudopotential library is shipped with ``doped``, so this
        catches the common case of writing inputs that ``pw.x`` will then fail to
        run.

        Cached on its arguments, so each distinct directory/pseudopotential
        combination warns only once per session (call
        ``DopedDictSetQE._warn_if_pseudos_missing.cache_clear()`` to reset, e.g.
        in tests).

        Args:
            pseudo_dir: The ``&CONTROL.pseudo_dir`` path to check.
            pseudo_filenames: UPF filenames expected in ``pseudo_dir``. Hashable
                (tuple) so the result can be cached.
        """
        resolved_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(pseudo_dir)))
        default_note = (
            f" Note that `pseudo_dir` is still the default placeholder ({DEFAULT_PSEUDO_DIR!r}); set "
            f"it to the directory containing your pseudopotentials."
            if pseudo_dir == DEFAULT_PSEUDO_DIR
            else ""
        )
        if not os.path.isdir(resolved_dir):
            warnings.warn(
                f"The pseudopotential directory `pseudo_dir` ({pseudo_dir!r}, resolved to "
                f"{resolved_dir!r}) does not exist, so the pseudopotentials required by these espresso "
                f"inputs could not be found. No pseudopotential library is bundled with `doped` -- "
                f"point `pseudo_dir` at a directory of pseudopotentials you already have, or download "
                f"a pseudopotential library."
            )
            return

        missing = [
            filename
            for filename in pseudo_filenames
            if not os.path.isfile(os.path.join(resolved_dir, filename))
        ]
        if missing:
            shown = ", ".join(missing[:10]) + (
                f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            )
            warnings.warn(
                f"{len(missing)} of the {len(pseudo_filenames)} pseudopotential file(s) required by "
                f"these espresso inputs were not found in `pseudo_dir` ({resolved_dir!r}): {shown}. "
                f"These are the default filenames from `QE_PSEUDO_LIBRARY`, so they will not match if "
                f"you are using a different pseudopotential library -- use `pseudo_map` to point the "
                f"affected element(s) at the files you have, or download a pseudopotential library "
            )

    @staticmethod
    def write_qe_pw_input(
        filepath: str,
        structure: Structure,
        namelist_settings: dict[str, dict],
        kpoints: list[int] | None,
        pseudo_map: dict[str, str] | None = None,
        kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """
        Write a espresso ``pw.in`` for ``structure``.

        Args:
            filepath: Destination path for ``pw.in``.
            structure: Structure to write.
            namelist_settings: ``{namelist: {key: value, ...}}`` for the espresso
                ``control``/``system``/``electrons``/``ions``/``cell`` namelists.
            kpoints: ``[kx, ky, kz]`` Monkhorst-Pack grid, or ``None`` for
                Γ-only sampling.
            pseudo_map: ``{element: UPF filename}`` overrides on top of the
                :data:`QE_PSEUDO_LIBRARY` defaults; missing elements fall back
                to ``"{element}.upf"``.
            kpoints_shift: ``(sx, sy, sz)`` grid offset (each 0 or 1) written as
                the second line of the ``K_POINTS automatic`` card, e.g.
                ``(1, 1, 1)`` for a half-grid (Γ-shifted) Monkhorst-Pack mesh.
                Default ``(0, 0, 0)`` (no shift). Ignored for Γ-only sampling
                (``kpoints=None``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # strip any oxidation states (e.g. on ``DefectsGenerator`` supercells) so that
        # species labels are plain element symbols (For example:"O", not "O2-"), as espresso expects.
        # Note the ``is not None`` test: a *zero* oxidation state is still a decorated ``Species``
        # which stringifies with a suffix ("Ge0+", not "Ge"), and guessed oxidation states are zero
        # for elemental / intermetallic hosts (e.g. ``DefectsGenerator(Ge).bulk_supercell``), so a
        # truthiness test here would leave those labels in place and break the ``Element`` lookups
        # (and ``ATOMIC_POSITIONS``/``ATOMIC_SPECIES`` labels) below:
        oxi_states = sorted(
            {str(sp) for sp in structure.species if getattr(sp, "oxi_state", None) is not None}
        )
        if oxi_states:
            print(
                f"Removing oxidation states ({', '.join(oxi_states)}) from the structure when "
                f"writing espresso input to {filepath}."
            )
            structure = structure.copy()
            structure.remove_oxidation_states()
        unique_species = sorted(
            {str(el) for el in structure.species}, key=lambda s: Element(s).Z
        )
        resolved_pseudos = {
            sp: QE_PSEUDO_LIBRARY.get(sp, f"{sp}.upf") for sp in unique_species
        }
        resolved_pseudos.update(pseudo_map or {})
        # this is a ``staticmethod`` so it re-resolves the pseudopotentials rather than using the
        # ``pseudopotentials`` property; check here too, as this is the path that actually writes files:
        DopedDictSetQE._warn_if_pseudos_missing(
            (namelist_settings.get("control") or {}).get("pseudo_dir", DEFAULT_PSEUDO_DIR),
            tuple(resolved_pseudos[sp] for sp in unique_species),
        )

        input_system_settings = namelist_settings.get("system") or {}
        if input_system_settings.get("starting_magnetization") is not None or any(
            key.startswith("starting_magnetization(") for key in input_system_settings
        ):
            namelist_settings = {nl: dict(settings) for nl, settings in namelist_settings.items()}
            system_settings = namelist_settings["system"]
            _set_starting_magnetization(system_settings, len(unique_species))
            if not system_settings.get("noncolin"):  # nspin is not allowed with noncolinear (SOC) calcs
                system_settings.setdefault("nspin", 2)

        fixed_cell_calcs = {"relax", "scf", "nscf", "bands"}
        calc_type = namelist_settings.get("control", {}).get("calculation", "vc-relax")

        namelist_classes = {
            "control": ControlNamelist,
            "system": SystemNamelist,
            "electrons": ElectronsNamelist,
            "ions": IonsNamelist,
            "cell": CellNamelist,
        }
        invalid_namelists = [nl_name for nl_name in namelist_settings if nl_name not in namelist_classes]
        if invalid_namelists:
            warnings.warn(
                f"Ignoring unrecognised espresso namelist(s) {invalid_namelists} in `namelist_settings` when "
                f"writing {filepath}; valid namelists are {list(namelist_classes)}. Check for typos "
                f"(e.g. 'electron' vs 'electrons')."
            )
        namelists: dict = {}
        for nl_name, nl_cls in namelist_classes.items():
            if nl_name not in namelist_settings:
                continue
            if nl_name == "cell" and calc_type in fixed_cell_calcs:
                continue
            namelists[nl_name] = nl_cls(namelist_settings[nl_name])

        if kpoints is None:
            k_points_card = KPointsCard("gamma", [], [], [], [], [])
        else:
            kx, ky, kz = kpoints
            k_points_card = KPointsCard("automatic", [kx, ky, kz], list(kpoints_shift), [], [], [])

        atomic_positions_card = AtomicPositionsCard(
            "angstrom",
            [site.species_string for site in structure],
            np.array([site.coords for site in structure]),
            None,
        )
        atomic_positions_card.force_multipliers = None

        cards = {
            "atomic_species": AtomicSpeciesCard(
                None,
                unique_species,
                [float(Element(sp).atomic_mass) for sp in unique_species],
                [resolved_pseudos[sp] for sp in unique_species],
            ),
            "atomic_positions": atomic_positions_card,
            "k_points": k_points_card,
            "cell_parameters": CellParametersCard(
                "angstrom",
                structure.lattice.matrix[0],
                structure.lattice.matrix[1],
                structure.lattice.matrix[2],
            ),
        }

        pwin = PWin(namelists, cards)
        pwin._indent = 2  # matches PWin.to_file's default indent
        # strip the trailing whitespace left on the ATOMIC_POSITIONS lines by the dropped `if_pos`:
        pw_str = "\n".join(line.rstrip() for line in str(pwin).splitlines()) + "\n"
        with open(filepath, "w", encoding="ascii") as f:
            f.write(pw_str)

    def __repr__(self):
        """
        Returns a string representation of the ``DopedDictSetQE`` object.
        """
        return (
            f"doped {type(self).__name__} with {self.calculation!r} calculation for "
            f"{self.structure.composition.reduced_formula} ({len(self.structure)} atoms), k-points: "
            f"{self.kpoints or 'gamma'}"
        )



class DefectDictSetQE(DopedDictSetQE):
    r"""
    Input set for a single Quantum ESPRESSO (``pw.x``) defect supercell
    calculation; the espresso analogue of
    ``doped.io.vasp.inputs.DefectDictSet``.

    Extends :class:`DopedDictSetQE` with the defect-specific settings: the
    supercell charge state (espresso's ``tot_charge``), a fixed-cell ``relax``
    by default (the defect supercell keeps the bulk supercell volume) and a
    spin-polarised starting point.
    """

    def __init__(
        self,
        structure: Structure,
        charge_state: int = 0,
        calculation: str = "relax",
        starting_magnetization: float | None = 0.1,
        **kwargs,
    ):
        r"""
        Args:
            structure (Structure):
                ``pymatgen`` ``Structure`` object of the defect supercell.
            charge_state (int):
                Charge state of the defect, written to espresso's
                ``&SYSTEM.tot_charge`` (``doped``/espresso convention:
                ``tot_charge`` = electrons removed = positive charge state).
                Default is 0.
            calculation (str):
                espresso ``&CONTROL.calculation`` mode. Default is ``"relax"``
                (fixed cell), as the defect supercell keeps the volume of the
                relaxed bulk supercell.
            starting_magnetization (float):
                ``&SYSTEM.starting_magnetization`` for every species (with
                ``nspin=2``), giving a spin-polarised starting point. Default
                is 0.1; set to ``None`` for a non-spin-polarised calculation.
                Applied to the reference bulk supercell as well as the defect
                supercells.
            **kwargs:
                Additional keyword arguments for :class:`DopedDictSetQE`
                (cutoffs, k-points, pseudopotentials, per-namelist overrides
                etc.).
        """
        self.charge_state = charge_state
        # ``tot_charge`` is merged into the ``&SYSTEM`` overrides (rather than set afterwards) so that
        # it is applied with the other user settings in ``_build_qe_base_settings``:
        user_system_settings = {
            **(kwargs.pop("user_system_settings", None) or {}),
            "tot_charge": charge_state,
        }
        super().__init__(
            structure,
            calculation=calculation,
            starting_magnetization=starting_magnetization,
            user_system_settings=user_system_settings,
            **kwargs,
        )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectDictSetQE`` object.
        """
        return (
            f"doped {type(self).__name__} with {self.calculation!r} calculation for "
            f"{self.structure.composition.reduced_formula} ({len(self.structure)} atoms) in charge "
            f"state {'+' if self.charge_state > 0 else ''}{self.charge_state}, k-points: "
            f"{self.kpoints or 'gamma'}"
        )


class DefectRelaxSetQE(MSONable):
    r"""
    Input sets for the Quantum ESPRESSO (``pw.x``) calculations of a single
    defect (and, optionally, its reference bulk supercell); the espresso
    analogue of ``doped.io.vasp.inputs.DefectRelaxSet``.
    """

    def __init__(
        self,
        defect_entry: DefectEntry | Structure,
        charge_state: int | None = None,
        soc: bool | None = None,
        bulk_supercell: Structure | None = None,
        **kwargs,
    ):
        r"""
        Args:
            defect_entry (DefectEntry or Structure):
                ``doped`` ``DefectEntry`` object for the defect, or a
                ``Structure`` object of the defect supercell (in which case
                ``charge_state`` must be given).
            charge_state (int):
                Charge state of the defect. If ``None`` (default), taken from
                ``defect_entry.charge_state``.
            soc (bool):
                Whether to include spin-orbit coupling. If ``None`` (default),
                set to ``True`` if the defect supercell contains an element
                with atomic number >= 31 (Ga and heavier), matching the VASP
                ``DefectRelaxSet`` behaviour.
            bulk_supercell (Structure):
                Reference bulk supercell, for the bulk input set. If ``None``
                (default), taken from ``defect_entry.bulk_supercell`` when a
                ``DefectEntry`` was provided.
            **kwargs:
                Keyword arguments for :class:`DefectDictSetQE` (cutoffs,
                k-points, pseudopotentials, per-namelist overrides etc.),
                applied to both the defect and bulk input sets.
        """
        self.defect_entry = defect_entry
        self.dict_set_kwargs = kwargs or {}

        if isinstance(defect_entry, Structure):
            self.defect_supercell = defect_entry
            self.bulk_supercell = bulk_supercell
            if charge_state is None:
                raise ValueError(
                    "`charge_state` must be specified when initialising `DefectRelaxSetQE` with a "
                    "`Structure` object rather than a `DefectEntry`!"
                )
            self.charge_state = charge_state
        else:
            self.defect_supercell = _get_defect_supercell(defect_entry)
            if self.defect_supercell is None:
                raise ValueError(
                    f"Could not determine the defect supercell for {defect_entry!r}, so cannot "
                    f"generate its espresso input files."
                )
            self.bulk_supercell = bulk_supercell if bulk_supercell is not None else getattr(
                defect_entry, "bulk_supercell", None
            )
            self.charge_state = charge_state if charge_state is not None else defect_entry.charge_state

        self.soc = soc if soc is not None else max(self.defect_supercell.atomic_numbers) >= 31

    def _dict_set(self, bulk: bool = False, **overrides) -> DefectDictSetQE:
        """
        Build the :class:`DefectDictSetQE` for one calculation stage.

        Args:
            bulk: If ``True``, build the set for the reference bulk supercell
                (a neutral single-point ``scf``, with the same
                ``starting_magnetization`` as the defect supercells) rather
                than the defect supercell.
            **overrides: Stage-specific :class:`DefectDictSetQE` settings (e.g.
                ``calculation``, ``soc``, ``use_hse``), applied on top of the
                ``dict_set_kwargs`` given when this set was initialised.
        """
        kwargs: dict = {**self.dict_set_kwargs}
        kwargs.setdefault("soc", self.soc)
        if bulk: 
            kwargs["calculation"] = "scf"  # single-point, on the already-relaxed bulk supercell
        kwargs.update(overrides)

        return DefectDictSetQE(
            self.bulk_supercell if bulk else self.defect_supercell,
            charge_state=0 if bulk else self.charge_state,
            **kwargs,
        )

    def _check_bulk_supercell_and_warn(self) -> Structure | None:
        """
        The reference bulk supercell, or ``None`` (with a warning) if this set
        was initialised with a ``Structure`` and no ``bulk_supercell``.
        """
        if self.bulk_supercell is None:
            warnings.warn(
                "`DefectRelaxSetQE.bulk_supercell` is None (because a `Structure` object rather than "
                "a `DefectEntry` was provided, without `bulk_supercell`), so the bulk espresso input "
                "files cannot be generated!"
            )
            return None

        return self.bulk_supercell

    @property
    def espresso_gamma(self) -> DefectDictSetQE:
        """
        :class:`DefectDictSetQE` for a Γ-point-only (``K_POINTS gamma``) fixed-
        cell (``relax``) calculation of the defect supercell in its charge
        state, spin-polarised; the espresso analogue of ``vasp_gam``.

        Any ``kpoints``/``kpoint_density`` settings given when initialising this
        set are overridden, so this is always Γ-only. Usually only needed for
        initial relaxations / defect structure-searching (e.g. with
        |ShakeNBreak|); note that espresso's default sampling is already Γ-only
        unless ``kpoints`` or ``kpoint_density`` is set, in which case
        :attr:`espresso_std` is the same calculation with a k-point mesh.
        """
        return self._dict_set(kpoints=None, kpoint_density=None)

    @property
    def espresso_std(self) -> DefectDictSetQE:
        """
        :class:`DefectDictSetQE` for the fixed-cell (``relax``) calculation of
        the defect supercell in its charge state, spin-polarised; the espresso
        analogue of ``vasp_std``.

        Uses whatever ``kpoints``/``kpoint_density`` were given when
        initialising this set. Unlike ``vasp_std`` -- which returns ``None`` for
        Γ-only meshes, because VASP needs a different binary -- this is never
        ``None``: espresso runs every k-point sampling through ``pw.x``, and its
        default sampling is Γ-only, so ``espresso_std`` with no k-point settings
        is the same calculation as :attr:`espresso_gamma`.
        """
        return self._dict_set()

    @property
    def espresso_ncl(self) -> DefectDictSetQE | None:
        """
        :class:`DefectDictSetQE` for the fixed-cell (``relax``) calculation of
        the defect supercell with spin-orbit coupling included (``noncolin =
        .true.``, ``lspinorb = .true.``); the espresso analogue of ``vasp_ncl``.

        Note that this is a relaxation, rather than the single-point (static)
        calculation, and that it is written from the *unrelaxed* defect
        supercell. Requires fully-relativistic pseudopotentials -- use
        ``pseudo_map`` to point at these, as the :data:`QE_PSEUDO_LIBRARY`
        defaults are scalar-relativistic.

        ``None`` (with a warning) if ``DefectRelaxSetQE.soc`` is ``False``. If
        ``soc`` was not set when initialising this set, it defaults to ``True``
        for defect supercells with a max atomic number (Z) >= 31 (i.e. further
        down the periodic table than Zn), matching the VASP ``DefectRelaxSet``.
        """
        if not self.soc:
            warnings.warn(
                "`DefectRelaxSetQE.soc` is False, so the spin-orbit coupling (`espresso_ncl`) input "
                "files cannot be generated! Set `soc=True` when initialising `DefectRelaxSetQE` if "
                "SOC effects should be included."
            )
            return None

        return self._dict_set(soc=True)

    @property
    def espresso_hybrid(self) -> DefectDictSetQE:
        """
        :class:`DefectDictSetQE` for a hybrid-DFT (HSE06 by default) fixed-cell
        (``relax``) calculation of the defect supercell, using the
        ``QE_sets/HSE_set.yaml`` defaults (``input_dft = HSE``,
        ``exx_fraction``, ``screening_parameter``, ``nqx1/2/3``).

        The espresso analogue of the ``vasp_(nkred_)std`` hybrid stage; there is
        no ``NKRED`` equivalent, as espresso instead controls the exact-exchange
        cost through the ``nqx1``/``nqx2``/``nqx3`` q-grid (1x1x1 by default,
        i.e. Γ-only exact exchange). Check ``exxdiv_treatment`` suits your cell.
        """
        return self._dict_set(use_hse=True)

    @property
    def bulk_espresso_gamma(self) -> DefectDictSetQE | None:
        """
        :class:`DefectDictSetQE` for the Γ-point-only single-point (``scf``)
        reference bulk supercell calculation: neutral, and spin-polarised as
        the defect supercells are. ``None`` (with a warning) if no bulk
        supercell is available.
        """
        if self._check_bulk_supercell_and_warn() is None:
            return None

        return self._dict_set(bulk=True, kpoints=None, kpoint_density=None)

    @property
    def bulk_espresso_std(self) -> DefectDictSetQE | None:
        """
        :class:`DefectDictSetQE` for the single-point (``scf``) reference bulk
        supercell calculation: neutral and, unlike the defect supercells, not
        relaxed. The same ``starting_magnetization`` (0.1 by default) is used..

        ``None`` if no bulk supercell is available (i.e. this set was
        initialised with a ``Structure`` and no ``bulk_supercell``).
        """
        if self._check_bulk_supercell_and_warn() is None:
            return None

        return self._dict_set(bulk=True)

    @property
    def bulk_espresso_ncl(self) -> DefectDictSetQE | None:
        """
        :class:`DefectDictSetQE` for the single-point (``scf``) spin-orbit-
        coupled reference bulk supercell calculation. ``None`` (with a warning)
        if ``soc`` is ``False`` or no bulk supercell is available.
        """
        if not self.soc or self._check_bulk_supercell_and_warn() is None:
            if not self.soc:
                warnings.warn(
                    "`DefectRelaxSetQE.soc` is False, so the bulk spin-orbit coupling "
                    "(`bulk_espresso_ncl`) input files cannot be generated!"
                )
            return None

        return self._dict_set(bulk=True, soc=True)

    @property
    def bulk_espresso_hybrid(self) -> DefectDictSetQE | None:
        """
        :class:`DefectDictSetQE` for the single-point (``scf``) hybrid-DFT
        reference bulk supercell calculation. ``None`` (with a warning) if no
        bulk supercell is available.
        """
        if self._check_bulk_supercell_and_warn() is None:
            return None

        return self._dict_set(bulk=True, use_hse=True)

    def _get_output_path(
        self, defect_dir: PathLike | None = None, subfolder: PathLike | None = None
    ) -> str:
        """
        ``<defect_dir>/<subfolder>``, with ``defect_dir`` defaulting to the
        ``DefectEntry`` name and no subfolder appended if ``subfolder`` is
        ``None``.
        """
        if defect_dir is None:
            no_name = getattr(self.defect_entry, "name", None) is None
            if isinstance(self.defect_entry, Structure) or no_name:
                raise ValueError(
                    "`defect_dir` must be specified if `DefectRelaxSetQE.defect_entry` is a `Structure` "
                    "object or has no `name` attribute set!"
                )
            defect_dir = self.defect_entry.name

        return os.path.join(str(defect_dir), subfolder) if subfolder else str(defect_dir)

    @staticmethod
    def _write_dict_set(dict_set: DefectDictSetQE, filepath: str) -> None:
        """
        Write the ``pw.in`` for one :class:`DefectDictSetQE` to ``filepath``.
        """
        dict_set.write_qe_pw_input(
            filepath=filepath,
            structure=dict_set.structure,
            namelist_settings=dict_set.namelists,
            kpoints=dict_set.kpoints,
            pseudo_map=dict_set.pseudo_map,
            kpoints_shift=dict_set.kpoints_shift,
        )

    def _write_espresso_xxx_files(
        self,
        defect_dir: PathLike | None,
        subfolder: PathLike | None,
        espresso_xxx_attribute: DefectDictSetQE,
    ) -> str:
        """
        Write the ``pw.in`` for one calculation stage
        (``espresso_xxx_attribute``) to ``<defect_dir>/<subfolder>/pw.in``,
        returning the written path; the espresso analogue of
        ``doped.vasp.DefectRelaxSet._write_vasp_xxx_files``, and the shared
        implementation behind :meth:`write_gamma`, :meth:`write_std`,
        :meth:`write_ncl` and :meth:`write_hybrid` (each of which calls it once
        for the defect supercell and, if ``bulk``, once for the reference bulk
        supercell).

        The ``DefectEntry`` is also serialised to
        ``<defect_dir>/<subfolder>/<name>.json.gz`` (as with VASP), to aid
        calculation provenance -- but not for bulk supercell folders, which have
        no corresponding ``DefectEntry``.
        """
        defect_dir = self._get_output_path(defect_dir)  # resolved folder, without subfolder
        output_path = self._get_output_path(defect_dir, subfolder)
        filepath = os.path.join(output_path, "pw.in")
        self._write_dict_set(espresso_xxx_attribute, filepath)

        if (  # not a bulk supercell folder, and a named ``DefectEntry`` to serialise:
            "bulk" not in os.path.basename(defect_dir)
            and not isinstance(self.defect_entry, Structure)
            and getattr(self.defect_entry, "name", None)
        ):
            self.defect_entry.to_json(os.path.join(output_path, f"{self.defect_entry.name}.json.gz"))

        return filepath

    def _get_bulk_dir(self, defect_dir: PathLike | None, bulk_dir: PathLike | None = None) -> str | None:
        """
        The folder for the reference bulk supercell calculation: ``bulk_dir`` if
        given, otherwise ``{host formula}_bulk`` alongside ``defect_dir`` (i.e.
        the bulk folder sits *beside* the defect folders, not inside them).

        ``None`` (with a warning) if no bulk supercell is available, i.e. this
        set was initialised with a ``Structure`` and no ``bulk_supercell``.
        """
        if self._check_bulk_supercell_and_warn() is None:
            return None
        if bulk_dir is not None:
            return str(bulk_dir)

        formula = self.bulk_supercell.composition.reduced_formula
        defect_dir = self._get_output_path(defect_dir)
        return os.path.join(os.path.dirname(defect_dir.rstrip("/")), f"{formula}_bulk")

    def write_gamma(
        self,
        defect_dir: PathLike | None = None,
        subfolder: str = "espresso_gamma",
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_gamma` (Γ-point-only relaxation) ``pw.in``
        input file, to ``<defect_dir>/<subfolder>/pw.in``.

        See :meth:`write_std` for the shared argument descriptions; ``subfolder``
        defaults to ``"espresso_gamma"`` here.
        """
        written = {"defect": self._write_espresso_xxx_files(defect_dir, subfolder, self.espresso_gamma)}
        if bulk:
            bulk_dir = self._get_bulk_dir(defect_dir, bulk_dir)  # None (& warns) if no bulk supercell
            if bulk_dir is not None:
                written["bulk"] = self._write_espresso_xxx_files(
                    bulk_dir, subfolder, self.bulk_espresso_gamma
                )

        return written

    def write_std(
        self,
        defect_dir: PathLike | None = None,
        subfolder: str = "espresso_std",
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_std` ``pw.in`` input file for the defect
        supercell calculation, to ``<defect_dir>/<subfolder>/pw.in``.

        The ``DefectEntry`` is also serialised to
        ``<defect_dir>/<subfolder>/<name>.json.gz`` (as with VASP), to aid
        calculation provenance.

        Args:
            defect_dir (PathLike):
                Folder for this defect. If ``None`` (default), the
                ``DefectEntry`` name is used.
            subfolder (str):
                Calculation subfolder name. Default is ``"espresso_std"``, the
                layout ``doped`` parses.
            bulk (bool):
                Whether to also write the reference bulk supercell input file.
                Default is ``False``.
            bulk_dir (PathLike):
                Folder for the bulk supercell calculation. If ``None``
                (default), written to ``{host formula}_bulk`` alongside
                ``defect_dir``.

        Returns:
            dict[str, str]: ``{name: pw.in path}`` for the written input files;
            ``"defect"`` and, if ``bulk``, ``"bulk"``.
        """
        written = {"defect": self._write_espresso_xxx_files(defect_dir, subfolder, self.espresso_std)}
        if bulk:
            bulk_dir = self._get_bulk_dir(defect_dir, bulk_dir)  # None (& warns) if no bulk supercell
            if bulk_dir is not None:
                written["bulk"] = self._write_espresso_xxx_files(
                    bulk_dir, subfolder, self.bulk_espresso_std
                )

        return written

    def write_ncl(
        self,
        defect_dir: PathLike | None = None,
        subfolder: str = "espresso_ncl",
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_ncl` (spin-orbit-coupled fixed-cell
        relaxation) ``pw.in`` input file, to ``<defect_dir>/<subfolder>/pw.in``.
        """
        espresso_ncl = self.espresso_ncl  # warns if `soc` is False
        if espresso_ncl is None:
            return {}

        written = {"defect": self._write_espresso_xxx_files(defect_dir, subfolder, espresso_ncl)}
        if bulk:
            bulk_dir = self._get_bulk_dir(defect_dir, bulk_dir)  # None (& warns) if no bulk supercell
            if bulk_dir is not None:
                written["bulk"] = self._write_espresso_xxx_files(
                    bulk_dir, subfolder, self.bulk_espresso_ncl
                )

        return written

    def write_hybrid(
        self,
        defect_dir: PathLike | None = None,
        subfolder: str = "espresso_hybrid",
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_hybrid` (HSE06 hybrid-DFT relaxation)
        ``pw.in`` input file, to ``<defect_dir>/<subfolder>/pw.in``.
        """
        written = {"defect": self._write_espresso_xxx_files(defect_dir, subfolder, self.espresso_hybrid)}
        if bulk:
            bulk_dir = self._get_bulk_dir(defect_dir, bulk_dir)  # None (& warns) if no bulk supercell
            if bulk_dir is not None:
                written["bulk"] = self._write_espresso_xxx_files(
                    bulk_dir, subfolder, self.bulk_espresso_hybrid
                )

        return written

    def write_all(
        self,
        defect_dir: PathLike | None = None,
        gamma: bool = False,
        hybrid: bool = False,
        bulk: bool | str = False,
        bulk_dir: PathLike | None = None,
    ) -> dict[str, dict[str, str]]:
        r"""
        Write the ``pw.in`` input files for every applicable calculation stage,
        each to its own subfolder of ``defect_dir``:

        - ``espresso_std``: the fixed-cell defect supercell relaxation. Always
          written.
        - ``espresso_ncl``: the spin-orbit-coupled fixed-cell relaxation.
          Written only if ``DefectRelaxSetQE.soc`` is ``True`` (which, if
          ``soc`` was not set explicitly, is the case for supercells with a max
          atomic number (Z) >= 31).
        - ``espresso_gamma``: the Γ-point-only relaxation. Written only if
          ``gamma=True``, as espresso's default sampling is already Γ-only and
          Γ-only relaxations are usually handled by defect structure-searching
          (e.g. |ShakeNBreak|).
        - ``espresso_hybrid``: the HSE06 hybrid-DFT relaxation. Written only if
          ``hybrid=True``.

        Args:
            defect_dir (PathLike):
                Folder for this defect. If ``None`` (default), the
                ``DefectEntry`` name is used.
            gamma (bool):
                Whether to also write the ``espresso_gamma`` folder. Default is
                ``False``.
            hybrid (bool):
                Whether to also write the ``espresso_hybrid`` folder. Default is
                ``False``.
            bulk (bool, str):
                If ``True``, the reference bulk supercell input file is also
                written, to the subfolder of the final (highest accuracy)
                calculation in the workflow (i.e. ``espresso_ncl`` if
                ``DefectRelaxSetQE.soc`` is ``True``, otherwise
                ``espresso_std``). If ``bulk = "all"``, it is written to the
                subfolder of every stage written here. Default is ``False``.
            bulk_dir (PathLike):
                Folder for the bulk supercell calculations. If ``None``
                (default), written to ``{host formula}_bulk`` alongside
                ``defect_dir``.

        Returns:
            dict[str, dict[str, str]]: ``{stage: {name: pw.in path}}`` for the
            written input files.
        """
        bulk_espresso: list[str] = []
        if isinstance(bulk, str):
            if bulk.lower() != "all":
                raise ValueError(
                    f"Unrecognised input for `bulk` argument: {bulk!r}. Must be True, False, or 'all'."
                )
            bulk_espresso = list(SUBFOLDER_PRIORITY)
        elif bulk:  # final (highest accuracy) calculation in the workflow:
            bulk_espresso = ["espresso_ncl" if self.soc else "espresso_std"]
            if hybrid:  # hybrid-DFT defects need a hybrid-DFT bulk reference
                bulk_espresso.append("espresso_hybrid")

        written = {
            "espresso_std": self.write_std(
                defect_dir, bulk="espresso_std" in bulk_espresso, bulk_dir=bulk_dir
            )
        }
        if gamma:
            written["espresso_gamma"] = self.write_gamma(
                defect_dir, bulk="espresso_gamma" in bulk_espresso, bulk_dir=bulk_dir
            )
        if hybrid:
            written["espresso_hybrid"] = self.write_hybrid(
                defect_dir, bulk="espresso_hybrid" in bulk_espresso, bulk_dir=bulk_dir
            )
        if self.soc:
            written["espresso_ncl"] = self.write_ncl(
                defect_dir, bulk="espresso_ncl" in bulk_espresso, bulk_dir=bulk_dir
            )

        return written

    def __repr__(self):
        """
        Returns a string representation of the ``DefectRelaxSetQE`` object.
        """
        formula = self.defect_supercell.composition.reduced_formula
        stages = [  # in priority (decreasing accuracy) order; `espresso_ncl` only if `soc`:
            stage for stage in SUBFOLDER_PRIORITY if self.soc or stage != "espresso_ncl"
        ]
        return (
            f"doped {type(self).__name__} for bulk composition {formula}, and defect in charge state "
            f"{'+' if self.charge_state > 0 else ''}{self.charge_state}, with espresso input sets "
            f"{', '.join(f'`{stage}`' for stage in stages)} (and their `bulk_`-prefixed equivalents)"
        )


class DefectsSetQE(DefectsSetBase):
    r"""
    Input sets for the Quantum ESPRESSO (``pw.x``) calculations of a set of
    defects (e.g. from a |DefectsGenerator|).

    Builds a :class:`DefectRelaxSetQE` per defect (``self.defect_sets``) and
    writes the ``<output_path>/<defect name>/<subfolder>/pw.in`` folder
    structure for every applicable calculation stage (``espresso_std``,
    ``espresso_ncl`` if ``soc``, and ``espresso_gamma``/``espresso_hybrid`` if
    requested), plus the ``{host formula}_bulk`` reference calculation.
    """

    _input_set_name = "DefectRelaxSetQE"

    def __init__(self, defect_entries, **kwargs):
        r"""
        As ``DefectsSetBase``, additionally setting the reference-bulk
        attributes: ``bulk_supercell`` and the single-point (``scf``)
        ``bulk_espresso_gamma``/``bulk_espresso_std``/``bulk_espresso_hybrid``/
        ``bulk_espresso_ncl`` input sets (the last of which is ``None`` if
        ``soc`` is ``False``).
        """
        super().__init__(defect_entries, **kwargs)

        # the bulk supercell is the same for every defect, so take the bulk sets from (any) one of
        # the per-defect sets, as with the VASP ``DefectsSet``:
        defect_relax_set = list(self.defect_sets.values())[-1]
        self.bulk_supercell = defect_relax_set.bulk_supercell
        with warnings.catch_warnings():  # `self.soc` already records this; don't warn on construction
            warnings.filterwarnings("ignore", "`DefectRelaxSetQE.soc` is False")
            self.bulk_espresso_gamma = defect_relax_set.bulk_espresso_gamma
            self.bulk_espresso_std = defect_relax_set.bulk_espresso_std
            self.bulk_espresso_hybrid = defect_relax_set.bulk_espresso_hybrid
            self.bulk_espresso_ncl = defect_relax_set.bulk_espresso_ncl

    def _setup(self) -> None:
        """
        Set ``self.soc`` for the whole set (and apply it to every
        :class:`DefectRelaxSetQE` built from ``self.kwargs``), determined -- if
        ``soc`` was not given -- by whether *any* defect supercell in the set
        has a max atomic number (Z) >= 31.
        """
        if self.kwargs.get("soc") is None:
            self.kwargs["soc"] = (
                max(
                    max(self._check_and_warn_defect_supercell(name, entry).atomic_numbers)
                    for name, entry in self.defect_entries.items()
                )
                >= 31
            )

        self.soc: bool = self.kwargs["soc"]

    @staticmethod
    def _check_and_warn_defect_supercell(name: str, defect_entry: DefectEntry) -> Structure:
        """
        The defect supercell of ``defect_entry``, erroring if undetermined.
        """
        defect_supercell = _get_defect_supercell(defect_entry)
        if defect_supercell is None:
            raise ValueError(
                f"Could not determine the defect supercell for {name!r}, so cannot generate its "
                f"espresso input files."
            )

        return defect_supercell

    def _defect_input_set(self, defect_entry: DefectEntry) -> DefectRelaxSetQE:
        """
        Build the :class:`DefectRelaxSetQE` for a single defect entry, with the
        set-wide ``soc`` setting determined in :meth:`_setup`.
        """
        return DefectRelaxSetQE(defect_entry, **self.kwargs)

    @staticmethod
    def _write_defect(args: tuple) -> None:
        """
        Write the espresso input files for a single defect (all applicable
        calculation stages, via ``DefectRelaxSetQE.write_all``), from one item
        of the ``args_list`` built in :meth:`write_files`.
        """
        defect_species, defect_relax_set, output_path, bulk, kwargs = args
        defect_relax_set.write_all(
            defect_dir=os.path.join(str(output_path), defect_species), bulk=bulk, **kwargs
        )

    def write_files(
        self,
        output_path: PathLike = ".",
        gamma: bool = False,
        hybrid: bool = False,
        bulk: bool | str = True,
        processes: int | None = None,
        **kwargs,
    ):
        r"""
        Write the espresso ``pw.in`` input files to folders for all defects in
        ``self.defect_entries``, in the ``<output_path>/<defect
        name>/<subfolder>`` folder structure, where ``defect name`` is the key
        of the :class:`DefectRelaxSetQE` in ``self.defect_sets`` (same as the
        ``self.defect_entries`` keys).

        For each defect folder, the following subfolders are generated:

        - ``espresso_std``: the fixed-cell (``relax``) defect supercell
          relaxation. Always written.
        - ``espresso_ncl``: the spin-orbit-coupled fixed-cell (``relax``)
          calculation. Written only if ``soc`` is ``True`` (which, if ``soc``
          was not set explicitly, is the case for supercells with a max atomic
          number (Z) >= 31).
        - ``espresso_gamma``: the Γ-point-only relaxation. Written only if
          ``gamma=True``, as espresso's default sampling is already Γ-only
          (i.e. ``espresso_std`` with no k-point settings is the same
          calculation) and Γ-only relaxations are usually handled by defect
          structure-searching (e.g. |ShakeNBreak|).
        - ``espresso_hybrid``: the HSE06 hybrid-DFT relaxation. Written only if
          ``hybrid=True``.

        The reference bulk supercell input file is written once, to
        ``<output_path>/{host formula}_bulk/<subfolder>``, alongside the defect
        folders (see ``bulk``).

        The |DefectEntry| objects are also serialised to ``json.gz`` files in
        the defect folders, as well as ``self.defect_entries``
        (``self.json_obj``) in ``output_path``, to aid calculation provenance.

        Args:
            output_path (PathLike):
                Folder in which to create the espresso defect calculation
                folders. Default is the current directory (".").
            gamma (bool):
                Whether to also write the ``espresso_gamma`` folders. Default is
                ``False``.
            hybrid (bool):
                Whether to also write the ``espresso_hybrid`` folders. Default
                is ``False``.
            bulk (bool, str):
                If ``True`` (default), the reference bulk supercell input file
                is also written, to the subfolder of the final (highest
                accuracy) calculation in the workflow (i.e. ``espresso_ncl`` if
                ``soc``, otherwise ``espresso_std``). If ``bulk = "all"``, it is
                written to the subfolder of every stage written here, and if
                ``bulk = False``, no bulk folder is created.
            processes (int):
                Number of processes to use for ``multiprocessing`` for file
                writing. If ``None`` (default), then is dynamically set to the
                optimal value for the number of folders to write.
            **kwargs:
                Additional keyword arguments for
                :meth:`DefectRelaxSetQE.write_all` (e.g. ``bulk_dir``).
        """
        return super().write_files(
            output_path=output_path,
            bulk=bulk,
            processes=processes,
            gamma=gamma,
            hybrid=hybrid,
            **kwargs,
        )


def qe_convergence_setup_from_structure(
    structure: Structure,
    output_dir: PathLike | None = None,
    kpoint_density_range: tuple = (20, 200, 20),
    kpoint_sweep_ecutwfc: int | None = None,
    ecut_range: tuple = (20, 90, 10),
    ecut_sweep_kpoint_density: int = 100,
    is_metal: bool = False,
    soc: bool = False,
    pseudo_dir: str = DEFAULT_PSEUDO_DIR,
    pseudo_map: dict | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_control_settings: dict | None = None,
    user_system_settings: dict | None = None,
    user_electron_settings: dict | None = None,
) -> dict[str, list[str]]:
    """
    Generate espresso ``pw.in`` files for k-point and plane-wave cutoff
    (``ecutwfc``) convergence testing from a single ``pymatgen``
    ``Structure``.

    Elements are read from ``structure.species`` and matched against the
    bundled :data:`QE_PSEUDO_LIBRARY` defaults (``doped/io/espresso/QE_sets``); the
    resolved UPF filenames are written into the ``ATOMIC_SPECIES`` card of
    every ``pw.in``. ``pseudo_map`` can override individual entries or
    supply pseudos for elements absent from it.

    Two sub-trees are written under ``output_dir``:

    - ``kpoint_converge/k<kx>_<ky>_<kz>/pw.in`` — ``ecutwfc`` held at
      ``kpoint_sweep_ecutwfc`` (or the set default if ``None``), k-grid
      swept over ``kpoint_density_range``. Duplicate grids produced by
      nearby densities are skipped.
    - ``ecut_convergence/ecutwfc_<N>/pw.in`` — k-grid held at
      ``ecut_sweep_kpoint_density``, ``ecutwfc`` swept over ``ecut_range``.
      ``ecutrho`` is rescaled to ``4 * ecutwfc`` at each step of the sweep.

    After running these and choosing converged values, call
    :func:`qe_relax_setup_from_structure` (with those converged values
    and, ideally, the relaxed structure) to write the final ``vc-relax``.

    Args:
        structure: Input structure (no MP lookup is performed).
        output_dir: Root folder for the two sub-trees. If ``None`` (default),
            written to a host-named parent folder as
            ``"{host_formula}_QE/Bulk_convergence"`` (e.g.
            ``"MgO_QE/Bulk_convergence"``), so all espresso inputs for a given host
            live under a single ``{host}_QE`` folder (matching the VASP
            example layout).
        kpoint_density_range: ``(min, max, step)`` reciprocal k-point
            density (Å^-3) sweep for the k-grid test (``max`` exclusive,
            matching ``range``). Default ``(20, 200, 20)``.
        kpoint_sweep_ecutwfc: ``ecutwfc`` (Ry) held fixed while the
            k-grid is swept. ``None`` (default) keeps the YAML set default
            (60 Ry).
        ecut_range: ``(min, max, step)`` ``ecutwfc`` (Ry) sweep for the
            cutoff test (``max`` inclusive). Default ``(20, 90, 10)``.
        ecut_sweep_kpoint_density: k-point density (Å^-3) held fixed
            while ``ecutwfc`` is swept. Default 100.
        is_metal: If ``True``, set ``occupations='smearing'``,
            ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``.
        soc: If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            (requires fully-relativistic pseudopotentials). Default ``False``.
        pseudo_dir: Path written to ``&CONTROL.pseudo_dir``.
        pseudo_map: ``{element: UPF filename}`` overrides on top of
            :data:`QE_PSEUDO_LIBRARY`.
        kpoints_shift: ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a half-grid
            (Γ-shifted) Monkhorst-Pack mesh. Default ``(0, 0, 0)`` (no shift).
        user_control_settings, user_system_settings, user_electron_settings:
            Per-namelist overrides merged on top of the YAML defaults.

    Returns:
        ``{"kpoint_converge": [...], "ecut_convergence": [...]}`` listing
        every ``pw.in`` path written.
    """
    if output_dir is None:
        output_dir = os.path.join(f"{structure.composition.reduced_formula}_QE", "Bulk_convergence")

    base = _build_qe_base_settings(
        structure,
        pseudo_dir,
        is_metal,
        user_control_settings,
        user_system_settings,
        user_electron_settings,
        soc=soc,
    )

    written: dict[str, list[str]] = {"kpoint_converge": [], "ecut_convergence": []}

    # ── k-point convergence: vary k-grid at a fixed ecutwfc ──
    kp_min, kp_max, kp_step = kpoint_density_range
    kpoint_scf = copy.deepcopy(base)
    kpoint_scf["control"]["calculation"] = "scf"
    if kpoint_sweep_ecutwfc is not None:
        kpoint_scf["system"]["ecutwfc"] = kpoint_sweep_ecutwfc
    seen_kgrids: set[tuple[int, int, int]] = set()
    for density in np.arange(kp_min, kp_max, kp_step):
        kgrid = _kpoints_grid_from_reciprocal_density(structure, density)
        kgrid_tuple = (kgrid[0], kgrid[1], kgrid[2])
        if kgrid_tuple in seen_kgrids:
            continue
        seen_kgrids.add(kgrid_tuple)
        kname = "k" + "_".join(str(k) for k in kgrid)
        filepath = os.path.join(str(output_dir), "kpoint_converge", kname, "pw.in")
        DopedDictSetQE.write_qe_pw_input(
            filepath=filepath,
            structure=structure,
            namelist_settings=kpoint_scf,
            kpoints=kgrid,
            pseudo_map=pseudo_map,
            kpoints_shift=kpoints_shift,
        )
        written["kpoint_converge"].append(filepath)

    # ── ecutwfc convergence: vary ecutwfc at a fixed k-grid ──
    ecut_kgrid = _kpoints_grid_from_reciprocal_density(structure, ecut_sweep_kpoint_density)
    ecut_min, ecut_max, ecut_step = ecut_range
    for ecut in range(ecut_min, ecut_max + 1, ecut_step):
        ecut_scf = copy.deepcopy(base)
        ecut_scf["control"]["calculation"] = "scf"
        ecut_scf["system"]["ecutwfc"] = ecut
        if "ecutrho" not in (user_system_settings or {}):
            ecut_scf["system"].pop("ecutrho", None)  
        _set_ecutrho(ecut_scf["system"])
        filepath = os.path.join(str(output_dir), "ecut_convergence", f"ecutwfc_{ecut}", "pw.in")
        DopedDictSetQE.write_qe_pw_input(
            filepath=filepath,
            structure=structure,
            namelist_settings=ecut_scf,
            kpoints=ecut_kgrid,
            pseudo_map=pseudo_map,
            kpoints_shift=kpoints_shift,
        )
        written["ecut_convergence"].append(filepath)

    return written

