"""
Quantum ESPRESSO (``pw.x``) calculation input file generation for ``doped``.

Note that all structures are written with ``ibrav = 0`` and the subsequent
``CELL_PARAMETERS`` card is used to write the lattice vectors.
"""

import copy
import os
import warnings
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core import Composition
from pymatgen.core.entries import ComputedEntry, ComputedStructureEntry
from pymatgen.core.units import Ry_to_eV
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

from doped.core import DefectEntry, _get_defect_supercell
from doped.generation import _custom_formatwarning
from doped.io.inputs import (
    DefectsSetBase,
    _get_competing_phase_magnetization,
    _get_constrained_total_magnetization,
)
from doped.utils import _ignore_pmg_warnings
from doped.utils.efficiency import Element, Structure

if TYPE_CHECKING:
    from doped.chemical_potentials import CompetingPhases

_ignore_pmg_warnings()
warnings.formatwarning = _custom_formatwarning

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
QE_SETS_DIR = os.path.join(MODULE_DIR, "QE_sets")

# placeholder written to ``&CONTROL.pseudo_dir`` until the user sets it:
DEFAULT_PSEUDO_DIR = "./pseudo_folder_name/"


# espresso ``pw.x`` input defaults, loaded from the YAML/JSON sets in ``doped/io/espresso/QE_sets``
default_qe_set: dict = loadfn(os.path.join(QE_SETS_DIR, "QE_Convergence_set.yaml"))
default_qe_HSE_set: dict = loadfn(os.path.join(QE_SETS_DIR, "HSE_set.yaml"))
# aliases used by the competing-phase input generation functions below (moved from ``outputs.py``):
_qe_convergence_defaults = default_qe_set
_qe_hse_relax_defaults = default_qe_HSE_set
_QE_PSEUDO_LIBRARY_METADATA: dict = loadfn(os.path.join(QE_SETS_DIR, "QE_PSEUDO_LIBRARY.json"))
QE_PSEUDO_LIBRARY: dict = {
    element: metadata["filename"] for element, metadata in _QE_PSEUDO_LIBRARY_METADATA.items()
}
"""
``{element: UPF filename}`` defaults for the ``ATOMIC_SPECIES`` card.

``doped`` does not bundle or require any particular pseudopotential library --
these are just the default filenames to look for in ``pseudo_dir``, and
``pseudo_map`` overrides them for any element.
"""
# the default pseudopotentials which are ultrasoft/PAW, rather than norm-conserving (``SG15``/``Dojo``):
_US_PAW_PSEUDOS: set[str] = {
    metadata["filename"]
    for metadata in _QE_PSEUDO_LIBRARY_METADATA.values()
    if not metadata["pseudopotential"].startswith(("SG15", "Dojo"))
}


SUBFOLDER_PRIORITY: list[str] = [
    "espresso_ncl",  # spin-orbit coupling (``noncolin``/``lspinorb``) single-point; highest accuracy
    "espresso_std",  # standard (k-point mesh) fixed-cell relaxation
    "espresso_gamma",  # Γ-point-only fixed-cell relaxation; lowest accuracy
]
"""
The ``pw.x`` calculation subfolder names written by :class:`DefectRelaxSet` /
:class:`DefectsSet`, in decreasing order of priority/accuracy.
"""


DEFAULT_STARTING_MAGNETIZATION: float = 0.1


def _set_starting_magnetization(system_settings: dict, ntyp: int) -> dict:
    """
    Write ``&SYSTEM.starting_magnetization`` in espresso's per-type indexed
    form -- ``starting_magnetization(i)`` for ``i = 1,...,ntyp``, indexing the
    ``ATOMIC_SPECIES`` entries -- with a value set for every atomic type.

    A single (scalar) ``starting_magnetization`` is applied to every type,
    while a list/tuple is taken as the per-type values in ``ATOMIC_SPECIES``
    order. Any ``starting_magnetization(i)`` keys already present are kept
    as-is, and any types left without a value (e.g. from a list shorter than
    ``ntyp``) are set to :data:`DEFAULT_STARTING_MAGNETIZATION` (0.1), so that
    spin polarisation is seeded on all species.

    Args:
        system_settings (dict):
            The ``&SYSTEM`` namelist settings dict, updated in place.
        ntyp (int):
            Number of atomic types (``&SYSTEM.ntyp``) in the structure.
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


@cache  # only warn once per pseudopotential set (as for ``DopedDictSet._warn_if_pseudos_missing``)
def _warn_if_hybrid_with_us_paw_pseudos(pseudo_filenames: tuple[str, ...]) -> None:
    """
    Warn if any of ``pseudo_filenames``, for a hybrid DFT relaxation or
    noncollinear (SOC) calculation, are ultrasoft/PAW pseudopotentials (from
    the :data:`QE_PSEUDO_LIBRARY` defaults), which ``pw.x`` does not support
    ('forces for hybrid functionals + US/PAW not implemented', 'noncollinear
    hybrid XC for USPP not implemented').
    """
    # if a recognised US/PAW (UPF) ``pseudo_type`` is being used:
    if unsupported := sorted(_US_PAW_PSEUDOS.intersection(pseudo_filenames)):
        warnings.warn(
            f"`pw.x` does not support relaxations or noncollinear (SOC) calculations with hybrid "
            f"functionals for ultrasoft/PAW pseudopotentials, which the default (`QE_PSEUDO_LIBRARY`) "
            f"pseudopotentials used here are: {', '.join(unsupported)}. Norm-conserving pseudopotentials "
            f"should be used for these (set with `pseudo_map`), or semi-local DFT (`use_hse=False`)."
        )


def _set_ecutrho(system_settings: dict) -> dict:
    """
    Set ``ecutrho`` to ``4 * ecutwfc`` if not already set, and warn if a
    supplied value is below that.

    These defaults are placeholders -- the user must set the cutoffs according
    to the pseudopotentials used.
    """
    ecutwfc = system_settings.get("ecutwfc")
    if ecutwfc is None:
        return system_settings

    ecutrho = system_settings.get("ecutrho")
    if ecutrho is None:
        system_settings["ecutrho"] = 4 * ecutwfc

    elif ecutrho < 4 * ecutwfc:
        warnings.warn(f"`ecutrho` ({ecutrho} Ry) is less than 4x `ecutwfc` ({ecutwfc} Ry) ")

    return system_settings


def _kpoints_grid_from_reciprocal_density(structure: Structure, reciprocal_density: float) -> list[int]:
    """
    ``[kx, ky, kz]`` Monkhorst-Pack grid at the given k-points-per-Å^-3
    density.
    """
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
    ``ibrav=0``, ``nat``, ``pseudo_dir``, optional metallic smearing / SOC
    (``soc``), and merges any user overrides (``ntyp`` is set on writing, in
    ``write_qe_pw_input``).

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

    if is_metal:
        _set_default_smearing(base["system"])

    if soc:  # spin-orbit coupling: noncollinear magnetisation with fully-relativistic pseudopotentials.
        base["system"].setdefault("noncolin", True)
        base["system"].setdefault("lspinorb", True)

    _set_ecutrho(base["system"])

    return base


class _KPointsCard(KPointsCard):
    """
    ``pymatgen-io-espresso`` ``KPointsCard`` which can also write the Γ-only
    ``K_POINTS gamma`` card (for which the upstream ``get_body`` leaves the
    card body unset, raising an ``UnboundLocalError``).
    """

    def get_body(self, indent: str) -> str:  # ponytail: drop once fixed in pymatgen-io-espresso
        return "" if self.option == "gamma" else super().get_body(indent)


def _get_seed_magmoms(structure: Structure, system_settings: dict) -> list[float] | None:
    """
    Get the ``magmom`` site properties of ``structure`` (zeroing those below
    0.1 μB, as for competing phase structures) to use as per-site
    ``starting_magnetization`` seeds (splitting sites into separate atomic
    species by magnetic moment), if the calculation is spin-polarised
    (``nspin=2`` or ``noncolin``) and ``structure`` has non-zero collinear
    (scalar) ``magmom`` site properties.

    These are used whether or not the total magnetization is fixed
    (``tot_magnetization``; where seeds are only needed to break the symmetry
    between sublattices for antiferro-/ferrimagnetic orderings), so that the
    same atomic species are written for collinear and noncollinear calculations
    of a structure (e.g. a relaxation and subsequent SOC single-point
    calculation, to which the relaxed structure is added).

    Note that the moments are written as-is, and ``pw.x`` only reads
    ``starting_magnetization`` values as moments in μB if any has a magnitude
    of 1 or more (from v7.3; earlier versions clip them to ±1); otherwise they
    are read as fractional polarisations of the valence charge, giving stronger
    seeds (with the same signs).
    """
    # seeds are the moments as-is; divide by the UPF ``z_valence`` values (fractional polarisations) if
    # exact / ``pw.x``-version-independent seeds are needed
    magmoms = structure.site_properties.get("magmom")
    if magmoms is None or not all(isinstance(m, int | float | np.number) for m in magmoms):
        return None  # no (collinear) site moments
    magmoms = [float(m) if abs(m) >= 0.1 else 0.0 for m in magmoms]  # (numerical noise removed)
    if not any(magmoms) or not (
        system_settings.get("noncolin") or int(system_settings.get("nspin", 1)) == 2
    ):
        return None
    return magmoms


def _get_atomic_species(
    structure: Structure, magmoms: list[float] | None = None
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """
    Get the espresso atomic species label for each site in ``structure`` and
    the corresponding (ordered) ``ATOMIC_SPECIES`` list.

    If ``magmoms`` (per-site magnetic moments in μB) are given, sites of the
    same element are split into separate atomic species by magnetic moment
    (as espresso sets ``starting_magnetization`` per species, so this is
    required for e.g. antiferromagnetic orderings), grouping moments with the
    same sign and within 0.5 μB of the largest (as in AiiDA's
    ``create_magnetic_configuration``), and labelling them ``X``, ``X1``,
    ``X2``... (as in ASE).

    Args:
        structure (Structure):
            The structure (without oxidation states).
        magmoms (list[float] or None):
            Per-site magnetic moments (in μB) to split species by, or
            ``None`` (default) for one species per element.

    Returns:
        tuple[list[str], list[tuple[str, str, float]]]:
            The species label of each site, and the ordered list of
            ``(label, element, magnetic moment)`` for each species (with
            moments of zero if ``magmoms`` is ``None``).
    """
    moments = magmoms or [0.0] * len(structure)
    site_labels: list[str] = [""] * len(structure)
    species: list[tuple[str, str, float]] = []
    site_elements = [site.specie.symbol for site in structure]
    for element in sorted(set(site_elements), key=lambda el: Element(el).Z):
        indices = [i for i, el in enumerate(site_elements) if el == element]
        groups: list[tuple[float, list[int]]] = []  # (largest moment, site indices)
        for i in sorted(indices, key=lambda i: -abs(moments[i])):
            group = next(
                (
                    g
                    for g in groups
                    if np.sign(g[0]) == np.sign(moments[i]) and abs(g[0] - moments[i]) <= 0.5
                ),
                None,
            )
            if group is None:
                groups.append((moments[i], [i]))
            else:
                group[1].append(i)
        for n, (moment, group_indices) in enumerate(groups):
            species.append((f"{element}{n or ''}", element, round(moment, 3) + 0.0))
            for i in group_indices:
                site_labels[i] = species[-1][0]

    if len(species) > 10:  # ``ntypx`` in ``pw.x``; also keeps labels within the 3 characters of < v7.3
        warnings.warn(
            f"{len(species)} atomic species are needed for {structure.composition.reduced_formula} "
            f"(with elements split into species by magnetic moment where these differ), but `pw.x` "
            f"only allows 10 (unless compiled with a larger `ntypx`), so will reject this input."
        )

    return site_labels, species


class DopedDictSet(MSONable):
    r"""
    Input set for a single Quantum ESPRESSO (``pw.x``) calculation on a given
    structure.

    Holds the ``&CONTROL``/``&SYSTEM``/``&ELECTRONS``/``&IONS``/``&CELL``
    namelists (assembled from the bundled YAML sets plus user overrides, via
    ``_build_qe_base_settings``), the k-point grid and the pseudopotential
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
                is used (``K_POINTS gamma``, or a 1x1x1 ``automatic`` grid for
                noncollinear calculations (as required by ``pw.x``).
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
                ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``
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
                calculation. Gaussian smearing is then used if neither
                ``occupations`` nor ``tot_magnetization`` are set (as
                ``pw.x`` requires ``tot_magnetization`` for spin-polarised
                calculations with fixed occupations). Default ``None``.
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
        ``_build_qe_base_settings``), then ``calculation``, the cutoffs and
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
        if self.use_hse and "vc-" not in self.calculation and "tstress" not in self.user_control_settings:
            # (costly) exact-exchange stresses are not needed at fixed cell, and ``pw.x`` doesn't support
            # them for noncollinear calculations or with k-point pools:
            namelists["control"]["tstress"] = False
        if self.ecutwfc is not None:
            namelists["system"]["ecutwfc"] = self.ecutwfc
            if self.ecutrho is None and "ecutrho" not in self.user_system_settings:
                namelists["system"].pop("ecutrho", None)
        if self.ecutrho is not None:
            namelists["system"]["ecutrho"] = self.ecutrho
        _set_ecutrho(namelists["system"])
        if self.starting_magnetization is not None:
            namelists["system"]["starting_magnetization"] = self.starting_magnetization
            if not namelists["system"].get("noncolin") and "tot_magnetization" not in namelists["system"]:
                # ``pw.x`` requires ``tot_magnetization`` for spin-polarised calculations with fixed
                # occupations (its default), so use Gaussian smearing (unless ``occupations`` is set):
                _set_default_smearing(namelists["system"])

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
        ``pseudo_map`` overrides applied and a ``"{element}.upf"`` fallback for
        elements absent from it.

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
    @cache  # only warn once per (``pseudo_dir``, pseudopotential set); the
    # ``pseudopotentials`` property is re-evaluated for every defect in a ``DefectsSet``
    def _warn_if_pseudos_missing(pseudo_dir: str, pseudo_filenames: tuple[str, ...]) -> None:
        """
        Warn if the UPF pseudopotential files to be written to
        ``ATOMIC_SPECIES`` are not actually present in ``pseudo_dir``.

        :data:`QE_PSEUDO_LIBRARY` only supplies the *filenames* espresso should
        look for; no pseudopotential library is shipped with ``doped``, so this
        catches the common case of writing inputs that ``pw.x`` will then fail
        to run.

        Cached on its arguments, so each distinct directory/pseudopotential
        combination warns only once per session (call
        ``DopedDictSet._warn_if_pseudos_missing.cache_clear()`` to reset, e.g.
        in tests).

        Args:
            pseudo_dir (str):
                The ``&CONTROL.pseudo_dir`` path to check.
            pseudo_filenames (tuple):
                UPF filenames expected in ``pseudo_dir``. Hashable (tuple) so
                the result can be cached.
        """
        resolved_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(pseudo_dir)))
        default_note = (
            f"Note that `pseudo_dir` is still the default placeholder ({DEFAULT_PSEUDO_DIR!r}); set "
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
                f"a pseudopotential library. {default_note}"
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
                f"affected element(s) at the files you have, or download a pseudopotential library."
                f" {default_note}"
            )

    @staticmethod
    def write_qe_pw_input(
        filepath: str,
        structure: Structure,
        namelist_settings: dict[str, dict],
        kpoints: list[int] | None,
        pseudo_map: dict[str, str] | None = None,
        kpoints_shift: tuple[int, int, int] = (0, 0, 0),
        positions: bool = True,
    ) -> None:
        """
        Write a espresso ``pw.in`` for ``structure``.

        Args:
            filepath (str):
                Destination path for ``pw.in``.
            structure (Structure):
                Structure to write.
            namelist_settings (dict):
                ``{namelist: {key: value, ...}}`` for the espresso
                ``control``/``system``/``electrons``/``ions``/``cell``
                namelists.
            kpoints (list[int]):
                ``[kx, ky, kz]`` Monkhorst-Pack grid, or ``None`` for Γ-only
                sampling.
            pseudo_map (dict):
                ``{element: UPF filename}`` overrides on top of the
                :data:`QE_PSEUDO_LIBRARY` defaults; missing elements fall back
                to ``"{element}.upf"``.
            kpoints_shift (tuple):
                ``(sx, sy, sz)`` grid offset (each 0 or 1) written as the
                second line of the ``K_POINTS automatic`` card, e.g.
                ``(1, 1, 1)`` for a half-grid (Γ-shifted) Monkhorst-Pack mesh.
                Default ``(0, 0, 0)`` (no shift). Ignored for Γ-only sampling
                (``kpoints=None``).
            positions (bool):
                Whether to write the structure (the ``CELL_PARAMETERS`` and
                ``ATOMIC_POSITIONS`` cards). If ``False``, these are left to be
                added to the written file, e.g. with the relaxed structure from
                a previous relaxation for a single-point calculation (``pw.x``
                exits with an error without them). Default ``True``.
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
        namelist_settings = {nl: dict(settings or {}) for nl, settings in namelist_settings.items()}
        system_settings = namelist_settings.setdefault("system", {})
        seeded = system_settings.get("starting_magnetization") is not None or any(
            key.startswith("starting_magnetization(") for key in system_settings
        )  # (then used for all sites of each element, rather than any ``magmom`` site properties)
        magmoms = None if seeded else _get_seed_magmoms(structure, system_settings)
        site_labels, species = _get_atomic_species(structure, magmoms)
        labels, elements, seeds = (list(values) for values in zip(*species, strict=True))
        resolved_pseudos = {el: QE_PSEUDO_LIBRARY.get(el, f"{el}.upf") for el in elements}
        resolved_pseudos.update(pseudo_map or {})
        # this is a ``staticmethod`` so it re-resolves the pseudopotentials rather than using the
        # ``pseudopotentials`` property; check here too, as this is the path that actually writes files:
        DopedDictSet._warn_if_pseudos_missing(
            (namelist_settings.get("control") or {}).get("pseudo_dir", DEFAULT_PSEUDO_DIR),
            tuple(resolved_pseudos[el] for el in dict.fromkeys(elements)),
        )

        system_settings["ntyp"] = len(species)  # (number of written ``ATOMIC_SPECIES`` labels)
        if magmoms is not None:  # per-species moment seeds
            system_settings["starting_magnetization"] = seeds
        if seeded or magmoms is not None:
            _set_starting_magnetization(system_settings, len(species))
            if not system_settings.get("noncolin"):  # nspin is not allowed with noncollinear (SOC) calcs
                system_settings.setdefault("nspin", 2)

        fixed_cell_calcs = {"relax", "scf", "nscf", "bands"}
        calc_type = namelist_settings.get("control", {}).get("calculation", "vc-relax")
        if "exx_fraction" in system_settings and (  # hybrid DFT (``HSE_set.yaml`` defaults)
            calc_type in {"relax", "vc-relax", "md", "vc-md"} or system_settings.get("noncolin")
        ):
            _warn_if_hybrid_with_us_paw_pseudos(tuple(resolved_pseudos[el] for el in elements))

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
                f"Ignoring unrecognised espresso namelist(s) {invalid_namelists} in `namelist_settings` "
                f"when writing {filepath}; valid namelists are {list(namelist_classes)}. Check for typos "
                f"(e.g. 'electron' vs 'electrons')."
            )
        namelists: dict = {}
        for nl_name, nl_cls in namelist_classes.items():
            if nl_name not in namelist_settings:
                continue
            if nl_name == "cell" and calc_type in fixed_cell_calcs:
                continue
            namelists[nl_name] = nl_cls(namelist_settings[nl_name])

        if kpoints is None and system_settings.get("noncolin"):
            # Γ-point only, but ``pw.x`` doesn't allow the ``K_POINTS gamma`` (``gamma_only``) tricks
            # with noncollinear calculations ('gamma_only and noncolin not allowed'):
            k_points_card = _KPointsCard("automatic", [1, 1, 1], [0, 0, 0], [], [], [])
        elif kpoints is None:
            k_points_card = _KPointsCard("gamma", [], [], [], [], [])
        else:
            kx, ky, kz = kpoints
            k_points_card = _KPointsCard("automatic", [kx, ky, kz], list(kpoints_shift), [], [], [])

        atomic_positions_card = AtomicPositionsCard(
            "angstrom",
            site_labels,
            np.array([site.coords for site in structure]),
            None,
        )
        atomic_positions_card.force_multipliers = None

        cards = {
            "atomic_species": AtomicSpeciesCard(
                None,
                labels,
                [float(Element(el).atomic_mass) for el in elements],
                [resolved_pseudos[el] for el in elements],
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

        if not positions:  # structure left to be added to the written file
            del cards["cell_parameters"], cards["atomic_positions"]

        pwin = PWin(namelists, cards)
        pwin._indent = 2  # matches PWin.to_file's default indent
        # strip the trailing whitespace left on the ATOMIC_POSITIONS lines by the dropped `if_pos`:
        pw_str = "\n".join(line.rstrip() for line in str(pwin).splitlines()) + "\n"
        if not positions:
            pw_str += (
                "! CELL_PARAMETERS & ATOMIC_POSITIONS to be added here (e.g. for the relaxed structure)\n"
            )
        with open(filepath, "w", encoding="ascii") as f:
            f.write(pw_str)

    def write_input(
        self,
        output_dir: PathLike,
        filename: str = "pw.in",
        positions: bool = True,
        rattle: bool = False,
        stdev: float | None = None,
        d_min: float | None = None,
    ) -> None:
        """
        Write the ``pw.x`` input file for this input set to
        ``<output_dir>/<filename>`` (creating ``output_dir`` if needed).

        Args:
            output_dir (PathLike):
                Directory to write the input file to.
            filename (str):
                Name of the ``pw.x`` input file. Default is ``"pw.in"``.
            positions (bool):
                Whether to write the structure (the ``CELL_PARAMETERS`` and
                ``ATOMIC_POSITIONS`` cards). If ``False``, these are left to
                be added to the written file, e.g. with the relaxed structure
                from a previous relaxation for a single-point calculation
                (``pw.x`` exits with an error without them). Default is
                ``True``.
            rattle (bool):
                If writing the structure for a `relaxation` (not single-point),
                apply random displacements to all atomic positions using the
                |ShakeNBreak| rattling algorithm; i.e. with the displacement
                distances randomly drawn from a Gaussian distribution of
                standard deviation equal to 10% of the nearest neighbour
                distance and using a Monte Carlo algorithm to penalise
                displacements that bring atoms closer than 80% of the nearest
                neighbour distance. This is intended to partially aid location
                of lower-energy broken-symmetry structures, but is far less
                effective than |ShakeNBreak| or more comprehensive energy
                landscape exploration strategies. Default is ``False``.
            stdev (float):
                Standard deviation (in Å) of the Gaussian distribution for the
                ``rattle`` displacements. Default (``None``) is 10% of the
                nearest neighbour distance.
            d_min (float):
                Minimum interatomic distance (in Å) for the ``rattle``
                displacements. Default (``None``) is 80% of the nearest
                neighbour distance.
        """
        structure = self.structure
        if (  # (never for single-point calculations)
            positions
            and rattle
            and "relax" in self.calculation
            and int(self.user_control_settings.get("nstep", 2)) > 1
        ):
            try:
                from shakenbreak.distortions import rattle as SnB_rattle
            except ImportError as e:
                raise ImportError(
                    "ShakeNBreak must be installed (pip install shakenbreak) to use the rattle option!"
                ) from e
            structure = SnB_rattle(structure, stdev=stdev, d_min=d_min)

        self.write_qe_pw_input(
            filepath=os.path.join(str(output_dir), filename),
            structure=structure,
            namelist_settings=self.namelists,
            kpoints=self.kpoints,
            pseudo_map=self.pseudo_map,
            kpoints_shift=self.kpoints_shift,
            positions=positions,
        )

    def __repr__(self):
        """
        Returns a string representation of the ``DopedDictSet`` object.
        """
        return (
            f"doped {type(self).__name__} with {self.calculation!r} calculation for "
            f"{self.structure.composition.reduced_formula} ({len(self.structure)} atoms), k-points: "
            f"{self.kpoints or 'gamma'}"
        )


class DefectDictSet(DopedDictSet):
    r"""
    Input set for a single Quantum ESPRESSO (``pw.x``) defect supercell
    calculation.

    Extends :class:`DopedDictSet` with the defect-specific settings: the
    supercell charge state (espresso's ``tot_charge``), a fixed-cell ``relax``
    by default (the defect supercell keeps the bulk supercell volume), no
    symmetry (``nosym``), a spin-polarised starting point, and Gaussian
    smearing (``degauss`` = 0.05 eV) with the spin state set by the
    electron-count parity (``tot_magnetization`` = 0 or 1 for an even or odd
    number of electrons) for collinear spin-polarised calculations, unless set
    in ``user_system_settings`` (e.g. ``tot_magnetization`` for magnetic
    hosts). The functional of the pseudopotentials (i.e. semi-local DFT) is
    used by default; set ``use_hse=True`` for HSE06 hybrid DFT (the default
    with the other ``doped`` calculator backends, and typically recommended for
    accurate defect calculations in insulating systems), for which ``pw.x``
    requires norm-conserving pseudopotentials for relaxations and noncollinear
    (SOC) calculations.
    """

    def __init__(
        self,
        structure: Structure,
        charge_state: int = 0,
        calculation: str = "relax",
        starting_magnetization: float | None = 0.1,
        use_hse: bool = False,
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
                is 0.1; set to ``None`` for a non-spin-polarised calculation
                (zero ``starting_magnetization`` if noncollinear).
                Applied to the reference bulk supercell as well as the defect
                supercells.
            use_hse (bool):
                Whether to use HSE06 hybrid DFT (the ``QE_sets/HSE_set.yaml``
                defaults: ``input_dft = HSE``, ``exx_fraction``,
                ``screening_parameter``, ``nqx1/2/3``; check that
                ``exxdiv_treatment`` suits your cell), as recommended for
                accurate defect calculations, rather than the functional of the
                pseudopotentials. Default is ``False``, as ``pw.x`` only
                supports relaxations and noncollinear (SOC) calculations with
                hybrid functionals for norm-conserving pseudopotentials (to be
                set with ``pseudo_map``), unlike most of the
                :data:`QE_PSEUDO_LIBRARY` defaults.
            **kwargs:
                Additional keyword arguments for :class:`DopedDictSet`
                (cutoffs, k-points, pseudopotentials, per-namelist overrides
                etc.).
        """
        self.charge_state = charge_state
        # ``tot_charge`` and the defect defaults are merged into the ``&SYSTEM`` overrides (rather than set
        # afterwards) so that they are applied with the other user settings in ``_build_qe_base_settings``:
        user_system_settings = {
            **(kwargs.pop("user_system_settings", None) or {}),
            "tot_charge": charge_state,
        }
        user_system_settings.setdefault("nosym", True)  # symmetry breaking extremely likely for defects
        # Gaussian smearing (as ``ISMEAR = 0``, ``SIGMA = 0.05`` eV for VASP defect calculations), and
        # the spin state set by the electron-count parity for collinear spin-polarised calculations (as
        # ``NUPDOWN`` = 0/1 for VASP); user settings take precedence:
        _set_default_smearing(user_system_settings, degauss=round(0.05 / Ry_to_eV, 5))
        if (
            starting_magnetization is not None
            and not (kwargs.get("soc") or user_system_settings.get("noncolin"))
            and int(user_system_settings.get("nspin", 2)) == 2
        ):
            # ponytail: parity from sum(Z) (frozen cores are closed-shell); wrong for f-in-core
            # pseudopotentials, read the UPF ``z_valence`` values if these are needed
            user_system_settings.setdefault(
                "tot_magnetization", (sum(structure.atomic_numbers) - charge_state) % 2
            )
        elif starting_magnetization is None and (
            kwargs.get("soc") or user_system_settings.get("noncolin")
        ):
            # not spin-polarised, so no seeds for noncollinear calculations either (else taken from any
            # ``magmom`` site properties, splitting sites into atomic species unlike the collinear stages):
            user_system_settings.setdefault("starting_magnetization", 0.0)
        super().__init__(
            structure,
            calculation=calculation,
            starting_magnetization=starting_magnetization,
            use_hse=use_hse,
            user_system_settings=user_system_settings,
            **kwargs,
        )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectDictSet`` object.
        """
        return (
            f"doped {type(self).__name__} with {self.calculation!r} calculation for "
            f"{self.structure.composition.reduced_formula} ({len(self.structure)} atoms) in charge "
            f"state {'+' if self.charge_state > 0 else ''}{self.charge_state}, k-points: "
            f"{self.kpoints or 'gamma'}"
        )


class DefectRelaxSet(MSONable):
    r"""
    Input sets for the Quantum ESPRESSO (``pw.x``) calculations of a single
    defect (and, optionally, its reference bulk supercell).
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
                Whether to include the spin-orbit coupling (``espresso_ncl``)
                calculations (the other stages are collinear, as for
                ``vasp_std``/``vasp_gam`` with VASP). If ``None`` (default),
                set to ``True`` if the defect supercell contains an element
                with atomic number >= 31 (Ga and heavier).
            bulk_supercell (Structure):
                Reference bulk supercell, for the bulk input set. If ``None``
                (default), taken from ``defect_entry.bulk_supercell`` when a
                ``DefectEntry`` was provided.
            **kwargs:
                Keyword arguments for :class:`DefectDictSet` (cutoffs,
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
                    "`charge_state` must be specified when initialising `DefectRelaxSet` with a "
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
            self.bulk_supercell = (
                bulk_supercell
                if bulk_supercell is not None
                else getattr(defect_entry, "bulk_supercell", None)
            )
            self.charge_state = charge_state if charge_state is not None else defect_entry.charge_state

        self.soc = soc if soc is not None else max(self.defect_supercell.atomic_numbers) >= 31

    def _dict_set(self, bulk: bool = False, **overrides) -> DefectDictSet:
        """
        Build the :class:`DefectDictSet` for one calculation stage.

        Args:
            bulk (bool):
                If ``True``, build the set for the reference bulk supercell (a
                neutral single-point ``scf``, with the same
                ``starting_magnetization`` as the defect supercells) rather
                than the defect supercell.
            **overrides:
                Stage-specific :class:`DefectDictSet` settings (e.g.
                ``calculation``, ``soc``), applied on top of the
                ``dict_set_kwargs`` given when this set was initialised.
        """
        kwargs: dict = {**self.dict_set_kwargs}  # (``soc`` only for the ``espresso_ncl`` stages)
        if bulk:
            kwargs["calculation"] = "scf"  # single-point, on the already-relaxed bulk supercell
        kwargs.update(overrides)

        return DefectDictSet(
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
                "`DefectRelaxSet.bulk_supercell` is None (because a `Structure` object rather than "
                "a `DefectEntry` was provided, without `bulk_supercell`), so the bulk espresso input "
                "files cannot be generated!"
            )
            return None

        return self.bulk_supercell

    @property
    def espresso_gamma(self) -> DefectDictSet:
        """
        :class:`DefectDictSet` for a Γ-only (``K_POINTS gamma``) fixed-cell
        (``relax``) calculation of the defect supercell in its charge state,
        spin-polarised.

        Any ``kpoints``/``kpoint_density`` settings given when initialising
        this set are overridden, so this is always Γ-only. Usually only needed
        for initial relaxations / defect structure-searching (e.g. with
        |ShakeNBreak|); note that espresso's default sampling is already Γ-only
        unless ``kpoints`` or ``kpoint_density`` is set, in which case
        :attr:`espresso_std` is the same calculation with a k-point mesh.
        """
        return self._dict_set(kpoints=None, kpoint_density=None)

    @property
    def espresso_std(self) -> DefectDictSet:
        """
        :class:`DefectDictSet` for the fixed-cell (``relax``) calculation of
        the defect supercell in its charge state, spin-polarised.

        Uses whatever ``kpoints``/``kpoint_density`` were given when
        initialising this set (``pw.x`` default sampling is Γ-only, so
        ``espresso_std`` with no k-point settings is the same as
        :attr:`espresso_gamma`).
        """
        return self._dict_set()

    @property
    def espresso_ncl(self) -> DefectDictSet | None:
        """
        :class:`DefectDictSet` for the single-point (``scf``) calculation of
        the defect supercell with spin-orbit coupling (SOC) included
        (``noncolin = .true.``, ``lspinorb = .true.``), to be run on the
        relaxed structure from the :attr:`espresso_std` (or
        :attr:`espresso_gamma`) relaxation (as for ``vasp_ncl`` with VASP).

        Requires fully-relativistic pseudopotentials -- use ``pseudo_map`` to
        point at these, as the :data:`QE_PSEUDO_LIBRARY` defaults are
        scalar-relativistic.

        ``None`` (with a warning) if ``DefectRelaxSet.soc`` is ``False``. If
        ``soc`` was not set when initialising this set, it defaults to ``True``
        for defect supercells with a max atomic number (Z) >= 31 (i.e. further
        down the periodic table than Zn).
        """
        if not self.soc:
            warnings.warn(
                "`DefectRelaxSet.soc` is False, so the spin-orbit coupling (`espresso_ncl`) input "
                "files cannot be generated! Set `soc=True` when initialising `DefectRelaxSet` if "
                "SOC effects should be included."
            )
            return None

        return self._dict_set(soc=True, calculation="scf")

    @property
    def bulk_espresso_gamma(self) -> DefectDictSet | None:
        """
        :class:`DefectDictSet` for the Γ-point-only single-point (``scf``)
        reference bulk supercell calculation: neutral, and spin-polarised as
        the defect supercells are.

        ``None`` (with a warning) if no bulk supercell is available.
        """
        if self._check_bulk_supercell_and_warn() is None:
            return None

        return self._dict_set(bulk=True, kpoints=None, kpoint_density=None)

    @property
    def bulk_espresso_std(self) -> DefectDictSet | None:
        """
        :class:`DefectDictSet` for the single-point (``scf``) reference bulk
        supercell calculation: neutral and, unlike the defect supercells, not
        relaxed. The same ``starting_magnetization`` (0.1 by default) is used.

        ``None`` if no bulk supercell is available.
        """
        if self._check_bulk_supercell_and_warn() is None:
            return None

        return self._dict_set(bulk=True)

    @property
    def bulk_espresso_ncl(self) -> DefectDictSet | None:
        """
        :class:`DefectDictSet` for the single-point (``scf``) spin-orbit-
        coupled reference bulk supercell calculation.

        ``None`` (with a warning) if ``soc`` is ``False`` or no bulk supercell
        is available.
        """
        if not self.soc or self._check_bulk_supercell_and_warn() is None:
            if not self.soc:
                warnings.warn(
                    "`DefectRelaxSet.soc` is False, so the bulk spin-orbit coupling "
                    "(`bulk_espresso_ncl`) input files cannot be generated!"
                )
            return None

        return self._dict_set(bulk=True, soc=True)

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
                    "`defect_dir` must be specified if `DefectRelaxSet.defect_entry` is a `Structure` "
                    "object or has no `name` attribute set!"
                )
            defect_dir = self.defect_entry.name

        return os.path.join(str(defect_dir), subfolder) if subfolder else str(defect_dir)

    @staticmethod
    def _write_dict_set(dict_set: DefectDictSet, filepath: str, **kwargs) -> None:
        """
        Write the ``pw.in`` for one :class:`DefectDictSet` to ``filepath``,
        with ``kwargs`` (``positions``, ``rattle``...) passed to
        :meth:`DopedDictSet.write_input`.
        """
        dict_set.write_input(*os.path.split(filepath), **kwargs)

    def _write_espresso_xxx_files(
        self,
        defect_dir: PathLike | None,
        subfolder: PathLike | None,
        espresso_xxx_attribute: DefectDictSet | None,
        **kwargs,
    ) -> str:
        """
        Write the ``pw.in`` for one calculation stage
        (``espresso_xxx_attribute``) to ``<defect_dir>/<subfolder>/pw.in``,
        returning the written path; this is a shared implementation behind
        :meth:`write_gamma`, :meth:`write_std` and :meth:`write_ncl` (each of
        which calls it once for the defect supercell and, if ``bulk``, once for
        the reference bulk supercell).

        The ``DefectEntry`` is also serialised to
        ``<defect_dir>/<subfolder>/<name>.json.gz``, to aid calculation
        provenance. ``kwargs`` (``positions``, ``rattle``...) are passed to
        :meth:`DopedDictSet.write_input`.
        """
        if espresso_xxx_attribute is None:  # bulk sets are None without a bulk supercell (callers check)
            raise ValueError(f"No input set available to write to {defect_dir} ({subfolder}).")
        defect_dir = self._get_output_path(defect_dir)  # resolved folder, without subfolder
        output_path = self._get_output_path(defect_dir, subfolder)
        filepath = os.path.join(output_path, "pw.in")
        self._write_dict_set(espresso_xxx_attribute, filepath, **kwargs)

        if (  # not a bulk supercell folder, and a named ``DefectEntry`` to serialise:
            "bulk" not in os.path.basename(defect_dir)
            and not isinstance(self.defect_entry, Structure)
            and getattr(self.defect_entry, "name", None)
        ):
            self.defect_entry.to_json(os.path.join(output_path, f"{self.defect_entry.name}.json.gz"))

        return filepath

    def _get_bulk_dir(self, defect_dir: PathLike | None, bulk_dir: PathLike | None = None) -> str | None:
        """
        The folder for the reference bulk supercell calculation: ``bulk_dir``
        if given, otherwise ``{host formula}_bulk`` alongside ``defect_dir``
        (i.e. the bulk folder sits *beside* the defect folders, not inside
        them).

        ``None`` (with a warning) if no bulk supercell is available, i.e. this
        set was initialised with a ``Structure`` and no ``bulk_supercell``.
        """
        if (bulk_supercell := self._check_bulk_supercell_and_warn()) is None:
            return None
        if bulk_dir is not None:
            return str(bulk_dir)

        formula = bulk_supercell.composition.reduced_formula
        defect_dir = self._get_output_path(defect_dir)
        return os.path.join(os.path.dirname(defect_dir.rstrip("/")), f"{formula}_bulk")

    def write_gamma(
        self,
        defect_dir: PathLike | None = None,
        subfolder: str = "espresso_gamma",
        rattle: bool = True,
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
        **kwargs,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_gamma` (Γ-point-only relaxation) ``pw.in``
        input file, to ``<defect_dir>/<subfolder>/pw.in``.

        See :meth:`write_std` for the shared argument descriptions;
        ``subfolder`` defaults to ``"espresso_gamma"`` here.
        """
        written = {
            "defect": self._write_espresso_xxx_files(
                defect_dir, subfolder, self.espresso_gamma, rattle=rattle, **kwargs
            )
        }
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
        rattle: bool = True,
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
        **kwargs,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_std` ``pw.in`` input file for the defect
        supercell calculation, to ``<defect_dir>/<subfolder>/pw.in``.

        The ``DefectEntry`` is also serialised to
        ``<defect_dir>/<subfolder>/<name>.json.gz`` , to aid calculation
        provenance.

        Args:
            defect_dir (PathLike):
                Folder for this defect. If ``None`` (default), the
                ``DefectEntry`` name is used.
            subfolder (str):
                Calculation subfolder name. Default is ``"espresso_std"``, the
                layout ``doped`` parses.
            rattle (bool):
                If writing the structure for a `relaxation` (not single-point),
                apply random displacements to all atomic positions using the
                |ShakeNBreak| rattling algorithm; i.e. with the displacement
                distances randomly drawn from a Gaussian distribution of
                standard deviation equal to 10% of the nearest neighbour
                distance and using a Monte Carlo algorithm to penalise
                displacements that bring atoms closer than 80% of the nearest
                neighbour distance. This is intended to partially aid location
                of lower-energy broken-symmetry structures, but is far less
                effective than |ShakeNBreak| or more comprehensive energy
                landscape exploration strategies. Default is ``True``. The
                reference bulk supercell is never rattled.
            bulk (bool):
                Whether to also write the reference bulk supercell input file.
                Default is ``False``.
            bulk_dir (PathLike):
                Folder for the bulk supercell calculation. If ``None``
                (default), written to ``{host formula}_bulk`` alongside
                ``defect_dir``.
            **kwargs:
                Keyword arguments for :meth:`DopedDictSet.write_input`, such
                as ``stdev`` and ``d_min`` to control the magnitude of the
                ``rattle`` displacements.

        Returns:
            dict[str, str]: ``{name: pw.in path}`` for the written input files;
            ``"defect"`` and, if ``bulk``, ``"bulk"``.
        """
        written = {
            "defect": self._write_espresso_xxx_files(
                defect_dir, subfolder, self.espresso_std, rattle=rattle, **kwargs
            )
        }
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
        positions: bool = False,
        bulk: bool = False,
        bulk_dir: PathLike | None = None,
        **kwargs,
    ) -> dict[str, str]:
        r"""
        Write the :attr:`espresso_ncl` (spin-orbit-coupled single-point)
        ``pw.in`` input file, to ``<defect_dir>/<subfolder>/pw.in``.

        By default, the structure (the ``CELL_PARAMETERS`` and
        ``ATOMIC_POSITIONS`` cards) is not written to the defect ``pw.in`` (as
        for ``POSCAR`` files with ``vasp_ncl``), as the relaxed structure from
        the :attr:`espresso_std` (or :attr:`espresso_gamma`) relaxation should
        be added (``pw.x`` exits with an error without it). If it is desired,
        set ``positions=True``.

        Args:
            defect_dir (PathLike):
                Folder for this defect. If ``None`` (default), the
                ``DefectEntry`` name is used.
            subfolder (str):
                Calculation subfolder name. Default is ``"espresso_ncl"``, the
                layout ``doped`` parses.
            positions (bool):
                If ``True``, the unperturbed structure of the defect supercell
                (the ``CELL_PARAMETERS`` and ``ATOMIC_POSITIONS`` cards) is
                also written to the defect ``pw.in`` (not rattled, as this is
                a single-point calculation). Default is ``False``. It is
                always written for the reference bulk supercell.
            bulk (bool):
                Whether to also write the reference bulk supercell input file.
                Default is ``False``.
            bulk_dir (PathLike):
                Folder for the bulk supercell calculation. If ``None``
                (default), written to ``{host formula}_bulk`` alongside
                ``defect_dir``.
            **kwargs:
                Keyword arguments for :meth:`DopedDictSet.write_input` (e.g.
                ``filename``).

        Returns:
            dict[str, str]: ``{name: pw.in path}`` for the written input files;
            ``"defect"`` and, if ``bulk``, ``"bulk"`` (empty if ``soc`` is
            ``False``).
        """
        espresso_ncl = self.espresso_ncl  # warns if `soc` is False
        if espresso_ncl is None:
            return {}

        written = {
            "defect": self._write_espresso_xxx_files(
                defect_dir, subfolder, espresso_ncl, positions=positions, **kwargs
            )
        }
        if bulk:
            bulk_dir = self._get_bulk_dir(defect_dir, bulk_dir)  # None (& warns) if no bulk supercell
            if bulk_dir is not None:
                written["bulk"] = self._write_espresso_xxx_files(
                    bulk_dir, subfolder, self.bulk_espresso_ncl
                )

        return written

    def write_all(
        self,
        defect_dir: PathLike | None = None,
        positions: bool = False,
        rattle: bool = True,
        gamma: bool = False,
        bulk: bool | str = False,
        bulk_dir: PathLike | None = None,
        **kwargs,
    ) -> dict[str, dict[str, str]]:
        r"""
        Write the ``pw.in`` input files for every applicable calculation stage,
        each to its own subfolder of ``defect_dir``.

        The calculation stages are:

        - ``espresso_std``: the fixed-cell defect supercell relaxation. Always
          written.
        - ``espresso_ncl``: the spin-orbit-coupled single-point calculation,
          for the relaxed structure from ``espresso_std`` (so written without
          the structure; see :meth:`write_ncl`). Written only if
          ``DefectRelaxSet.soc`` is ``True`` (which, if ``soc`` was not set
          explicitly, is the case for supercells with a max atomic number (Z)
          >= 31).
        - ``espresso_gamma``: the Γ-point-only relaxation. Written only if
          ``gamma=True``, as espresso's default sampling is already Γ-only and
          Γ-only relaxations are usually handled by defect structure-searching
          (e.g. |ShakeNBreak|).

        Args:
            defect_dir (PathLike):
                Folder for this defect. If ``None`` (default), the
                ``DefectEntry`` name is used.
            positions (bool):
                If ``True``, the (unrelaxed) structure of the defect supercell
                is also written to the ``espresso_ncl`` defect ``pw.in`` (see
                :meth:`write_ncl`); the ``espresso_std``/``espresso_gamma``
                relaxation inputs always include it. Default is ``False``.
            rattle (bool):
                Whether to apply random displacements to all atomic positions
                in the written defect structures for relaxations, using the
                |ShakeNBreak| rattling algorithm (see :meth:`write_std`).
                Default is ``True``. Single-point calculations
                (``espresso_ncl``, and the reference bulk supercell) are never
                rattled.
            gamma (bool):
                Whether to also write the ``espresso_gamma`` folder. Default is
                ``False``.
            bulk (bool, str):
                If ``True``, the reference bulk supercell input file is also
                written, to the subfolder of the final (highest accuracy)
                calculation in the workflow (i.e. ``espresso_ncl`` if
                ``DefectRelaxSet.soc`` is ``True``, otherwise
                ``espresso_std``). If ``bulk = "all"``, it is written to the
                subfolder of every stage.
            bulk_dir (PathLike):
                Folder for the bulk supercell calculations. If ``None``
                (default), written to ``{host formula}_bulk`` alongside
                ``defect_dir``.
            **kwargs:
                Keyword arguments for :meth:`DopedDictSet.write_input`, such
                as ``stdev`` and ``d_min`` to control the magnitude of the
                ``rattle`` displacements.

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

        written = {
            "espresso_std": self.write_std(
                defect_dir,
                rattle=rattle,
                bulk="espresso_std" in bulk_espresso,
                bulk_dir=bulk_dir,
                **kwargs,
            )
        }
        if gamma:
            written["espresso_gamma"] = self.write_gamma(
                defect_dir,
                rattle=rattle,
                bulk="espresso_gamma" in bulk_espresso,
                bulk_dir=bulk_dir,
                **kwargs,
            )
        if self.soc:
            written["espresso_ncl"] = self.write_ncl(  # (single-point, so never rattled)
                defect_dir,
                positions=positions,
                bulk="espresso_ncl" in bulk_espresso,
                bulk_dir=bulk_dir,
                **kwargs,
            )

        return written

    def __repr__(self):
        """
        Returns a string representation of the ``DefectRelaxSet`` object.
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


class DefectsSet(DefectsSetBase):
    r"""
    Input sets for the Quantum ESPRESSO (``pw.x``) calculations of a set of
    defects (e.g. from a |DefectsGenerator|).

    Builds a :class:`DefectRelaxSet` per defect (``self.defect_sets``) and
    writes the ``<output_path>/<defect name>/<subfolder>/pw.in`` folder
    structure for every applicable calculation stage (``espresso_std``,
    ``espresso_ncl`` if ``soc``, and ``espresso_gamma`` if requested), plus the
    ``{host formula}_bulk`` reference calculation.
    """

    _input_set_name = "DefectRelaxSet"

    def __init__(self, defect_entries, **kwargs):
        r"""
        As ``DefectsSetBase``, additionally setting the reference-bulk
        attributes: ``bulk_supercell`` and the single-point (``scf``)
        ``bulk_espresso_gamma``/``bulk_espresso_std``/``bulk_espresso_ncl``
        input sets (the last of which is ``None`` if ``soc`` is ``False``).
        """
        super().__init__(defect_entries, **kwargs)

        # the bulk supercell is the same for every defect, so take the bulk sets from (any) one of
        # the per-defect sets:
        defect_relax_set = list(self.defect_sets.values())[-1]
        self.bulk_supercell = defect_relax_set.bulk_supercell
        with warnings.catch_warnings():  # `self.soc` already records this; don't warn on construction
            warnings.filterwarnings("ignore", "`DefectRelaxSet.soc` is False")
            self.bulk_espresso_gamma = defect_relax_set.bulk_espresso_gamma
            self.bulk_espresso_std = defect_relax_set.bulk_espresso_std
            self.bulk_espresso_ncl = defect_relax_set.bulk_espresso_ncl

    def _setup(self) -> None:
        """
        Set ``self.soc`` for the whole set (and apply it to every
        :class:`DefectRelaxSet` built from ``self.kwargs``), determined -- if
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

    def _defect_input_set(self, defect_entry: DefectEntry) -> DefectRelaxSet:
        """
        Build the :class:`DefectRelaxSet` for a single defect entry, with the
        set-wide ``soc`` setting determined in :meth:`_setup`.
        """
        return DefectRelaxSet(defect_entry, **self.kwargs)

    @staticmethod
    def _write_defect(args: tuple) -> None:
        """
        Write the espresso input files for a single defect (all applicable
        calculation stages, via ``DefectRelaxSet.write_all``), from one item of
        the ``args_list`` built in :meth:`write_files`.
        """
        defect_species, defect_relax_set, output_path, bulk, kwargs = args
        defect_relax_set.write_all(
            defect_dir=os.path.join(str(output_path), defect_species), bulk=bulk, **kwargs
        )

    def write_files(  # type: ignore[override]  # deliberately extends the base signature
        self,
        output_path: PathLike = ".",
        positions: bool = False,
        rattle: bool = True,
        gamma: bool = False,
        bulk: bool | str = True,
        processes: int | None = None,
        **kwargs,
    ):
        r"""
        Write the espresso ``pw.in`` input files to folders for all defects in
        ``self.defect_entries``, in the ``<output_path>/<defect
        name>/<subfolder>`` folder structure, where ``defect name`` is the key
        of the :class:`DefectRelaxSet` in ``self.defect_sets`` (same as the
        ``self.defect_entries`` keys).

        For each defect folder, the following subfolders are generated:

        - ``espresso_std``: the fixed-cell (``relax``) defect supercell
          relaxation. Always written.
        - ``espresso_ncl``: the spin-orbit-coupled single-point (``scf``)
          calculation, for the relaxed structure from ``espresso_std`` (so
          written without the structure; see :meth:`DefectRelaxSet.write_ncl`).
          Written only if ``soc`` is ``True`` (which, if ``soc`` was not set
          explicitly, is the case for supercells with a max atomic number (Z)
          >= 31).
        - ``espresso_gamma``: the Γ-point-only relaxation. Written only if
          ``gamma=True``, as espresso's default sampling is already Γ-only
          (i.e. ``espresso_std`` with no k-point settings is the same
          calculation) and Γ-only relaxations are usually handled by defect
          structure-searching (e.g. |ShakeNBreak|).

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
            positions (bool):
                If ``True``, the (unrelaxed) structures of the defect
                supercells are also written to the ``espresso_ncl`` defect
                ``pw.in``\s (see :meth:`DefectRelaxSet.write_ncl`); the
                ``espresso_std``/``espresso_gamma`` relaxation inputs always
                include them. Default is ``False``.
            rattle (bool):
                Whether to apply random displacements to all atomic positions
                in the written defect structures for relaxations, using the
                |ShakeNBreak| rattling algorithm (see
                :meth:`DefectRelaxSet.write_std`). Default is ``True``.
                Single-point calculations (``espresso_ncl``, and the reference
                bulk supercell) are never rattled.
            gamma (bool):
                Whether to also write the ``espresso_gamma`` folders. Default
                is ``False``.
            bulk (bool, str):
                If ``True`` (default), the reference bulk supercell input file
                is also written, to the subfolder of the final (highest
                accuracy) calculation in the workflow (i.e. ``espresso_ncl`` if
                ``soc``, otherwise ``espresso_std``). If ``bulk = "all"``, it
                is written to the subfolder of every stage written here, and if
                ``bulk = False``, no bulk folder is created.
            processes (int):
                Number of processes to use for ``multiprocessing`` for file
                writing. If ``None`` (default), then is dynamically set to the
                optimal value for the number of folders to write.
            **kwargs:
                Additional keyword arguments for
                :meth:`DefectRelaxSet.write_all` (e.g. ``bulk_dir``, or
                ``stdev`` and ``d_min`` to control the magnitude of the
                ``rattle`` displacements).
        """
        return super().write_files(
            output_path=output_path,
            bulk=bulk,
            processes=processes,
            positions=positions,
            rattle=rattle,
            gamma=gamma,
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
    (``ecutwfc``) convergence testing from a single ``pymatgen`` ``Structure``.

    Elements are read from ``structure.species`` and matched against the
    bundled :data:`QE_PSEUDO_LIBRARY` defaults (``doped/io/espresso/QE_sets``);
    the resolved UPF filenames are written into the ``ATOMIC_SPECIES`` card of
    every ``pw.in``. ``pseudo_map`` can override individual entries or supply
    pseudos for elements absent from it.

    Two sub-trees are written under ``output_dir``:

    - ``kpoint_converge/k<kx>_<ky>_<kz>/pw.in`` — ``ecutwfc`` held at
      ``kpoint_sweep_ecutwfc`` (or the set default if ``None``), k-grid
      swept over ``kpoint_density_range``. Duplicate grids produced by
      nearby densities are skipped.
    - ``ecut_convergence/ecutwfc_<N>/pw.in`` — k-grid held at
      ``ecut_sweep_kpoint_density``, ``ecutwfc`` swept over ``ecut_range``.
      ``ecutrho`` is rescaled to ``4 * ecutwfc`` at each step of the sweep.

    After running these and choosing converged values, feed them to
    :func:`get_relaxation_sets` / :func:`write_relaxation_files` -- the cutoffs
    via ``user_system_settings`` (e.g. ``{"ecutwfc": 80, "ecutrho": 320}``) and
    the k-point densities via ``kpoints_metals``/``kpoints_nonmetals`` -- to
    write the final ``vc-relax`` calculations of the competing phases.

    Args:
        structure (Structure):
            Input structure (no MP lookup is performed).
        output_dir (PathLike or None):
            Root folder for the two sub-trees. If ``None`` (default), written
            to a host-named parent folder as
            ``"{host_formula}_QE/Bulk_convergence"`` (e.g.
            ``"MgO_QE/Bulk_convergence"``), so all espresso inputs for a given
            host live under a single ``{host}_QE`` folder.
        kpoint_density_range (tuple):
            ``(min, max, step)`` reciprocal k-point density (Å^-3) sweep for
            the k-grid test (``max`` exclusive, matching ``range``). Default
            ``(20, 200, 20)``.
        kpoint_sweep_ecutwfc (int or None):
            ``ecutwfc`` (Ry) held fixed while the k-grid is swept. ``None``
            (default) keeps the YAML set default (60 Ry).
        ecut_range (tuple):
            ``(min, max, step)`` ``ecutwfc`` (Ry) sweep for the cutoff test
            (``max`` inclusive). Default ``(20, 90, 10)``.
        ecut_sweep_kpoint_density (int):
            k-point density (Å^-3) held fixed while ``ecutwfc`` is swept.
            Default 100.
        is_metal (bool):
            If ``True``, set ``occupations='smearing'``,
            ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM`` (unless
            ``occupations`` is given in ``user_system_settings``).
        soc (bool):
            If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            (requires fully-relativistic pseudopotentials). Default ``False``.
        pseudo_dir (str):
            Path written to ``&CONTROL.pseudo_dir``.
        pseudo_map (dict or None):
            ``{element: UPF filename}`` overrides on top of
            :data:`QE_PSEUDO_LIBRARY`.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a half-grid
            (Γ-shifted) Monkhorst-Pack mesh. Default ``(0, 0, 0)`` (no shift).
        user_control_settings (dict or None):
            ``&CONTROL`` namelist overrides, merged on top of the YAML
            defaults.
        user_system_settings (dict or None):
            ``&SYSTEM`` namelist overrides, merged on top of the YAML defaults.
        user_electron_settings (dict or None):
            ``&ELECTRONS`` namelist overrides, merged on top of the YAML
            defaults.

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
        DopedDictSet.write_qe_pw_input(
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
        DopedDictSet.write_qe_pw_input(
            filepath=filepath,
            structure=structure,
            namelist_settings=ecut_scf,
            kpoints=ecut_kgrid,
            pseudo_map=pseudo_map,
            kpoints_shift=kpoints_shift,
        )
        written["ecut_convergence"].append(filepath)

    return written


# ──────────────────────────────────────────────────────────────────────────
# competing-phase input generation (formerly ``CompetingPhasesQE`` methods)
# ──────────────────────────────────────────────────────────────────────────


def _default_qe_competing_phases_path(composition: str | Composition) -> str:
    """
    Get the default host-compound folder for QE competing phase calculations:
    ``"{host}_QE/CompetingPhases_{host}_QE"`` (e.g.
    ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a ``{host}_QE`` parent folder
    keyed on the host composition's reduced formula.

    Used as the default ``output_path`` for :func:`get_kpoint_convergence_sets`
    / :func:`get_relaxation_sets` / :func:`get_singlepoint_sets` (and the
    corresponding ``write_*_files`` functions), so that generated inputs can be
    found again (by passing the same path to
    :class:`~doped.chemical_potentials.CompetingPhasesAnalyzer` with
    ``calculator="espresso"``).

    Args:
        composition (str or ``Composition``):
            Composition of the host material (e.g. ``"MgO"``).

    Returns:
        str:
            The default competing phases folder for this host composition.
    """
    host = Composition(composition).reduced_formula
    return os.path.join(f"{host}_QE", f"CompetingPhases_{host}_QE")


def _competing_phase_system_settings(
    entry: ComputedEntry | ComputedStructureEntry,
    user_system_settings: dict | None,
    soc: bool,
    structure: Structure,
    is_metal: bool,
) -> dict:
    """
    ``&SYSTEM`` namelist overrides for the calculation of one competing phase
    (``entry``, with ``structure``): the user settings, plus spin polarisation
    for magnetic phases (via :func:`_set_spin_polarisation`, which depends on
    whether the calculation is noncollinear (``soc``; hence the
    ``noncolin``/``lspinorb`` flags are set here too) and on the occupations
    (fixed for non-metals by default, i.e. ``is_metal = False``)).
    """
    system_settings = copy.deepcopy(user_system_settings or {})
    if soc:  # spin-orbit coupling: noncollinear magnetisation with fully-relativistic pseudos
        system_settings.setdefault("noncolin", True)
        system_settings.setdefault("lspinorb", True)
    _set_spin_polarisation(system_settings, entry, structure, is_metal)

    return system_settings


def get_kpoint_convergence_sets(
    competing_phases: "CompetingPhases",
    output_path: PathLike | None = None,
    kpoints_metals: tuple[float, float, float] = (40, 1000, 5),
    kpoints_nonmetals: tuple[float, float, float] = (5, 120, 5),
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    soc: bool = False,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_system_settings: dict | None = None,
    user_control_settings: dict | None = None,
    user_electron_settings: dict | None = None,
) -> dict[str, DopedDictSet]:
    r"""
    Generates a dictionary of :class:`DopedDictSet`\s for `k`-point convergence
    testing of competing phases, as single-point (``scf``) ``pw.x``
    calculations using the :data:`QE_PSEUDO_LIBRARY` pseudopotential filenames
    by default, written to
    ``<output_path>/<phase>/kpoint_converge/k<kx>_<ky>_<kz>``.

    Diatomic molecules are skipped, as Γ-only sampling is sufficient for
    molecules-in-a-box. Plane-wave cutoffs (``ecutwfc``/``ecutrho``) are kept
    fixed for all competing-phase, bulk and defect calculations.

    Automatically sets Gaussian smearing (``occupations = 'smearing'``) for
    metallic phases, and spin polarisation for magnetic phases (``nspin = 2``,
    with ``tot_magnetization`` for non-metals if consistent with any ``magmom``
    site properties, which are used as ``starting_magnetization`` seeds,
    splitting sites into separate atomic species by magnetic moment; or only
    ``starting_magnetization`` if ``soc``).

    Args:
        competing_phases (CompetingPhases):
            The :class:`~doped.chemical_potentials.CompetingPhases` object
            whose entries to generate calculation inputs for.
        output_path (PathLike or None):
            Host-compound parent folder under which all competing-phase
            sub-folders are written. If ``None`` (default), this is set to
            ``"{host_compound}_QE/CompetingPhases_{host_compound}_QE"``
            (e.g. ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a
            ``{host}_QE`` parent folder keyed on the host composition's
            reduced formula.
        kpoints_metals (tuple[float, float, float]):
            ``(min, max, step)`` reciprocal k-point density (Å^-3) range
            to test for metallic phases (``max`` exclusive).
            Default ``(40, 1000, 5)``. Note that only
            unique k-point grids are generated, so small step sizes (as
            default) just result in each k-points choice between ``min``
            and ``max`` being included.
        kpoints_nonmetals (tuple[float, float, float]):
            ``(min, max, step)`` reciprocal k-point density (Å^-3) range
            to test for non-metallic phases (``max`` exclusive).
            Default ``(5, 120, 5)``. Note that
            only unique k-point grids are generated, so small step sizes
            (as default) just result in each k-points choice between
            ``min`` and ``max`` being included.
        pseudo_dir (str):
            Path to the directory containing UPF pseudopotential files,
            written into ``&CONTROL``. Default ``"./pseudo_folder_name/"``.
        pseudo_map (dict or None):
            ``{element_symbol: pseudo_filename}`` mapping, e.g.
            ``{"O": "O.pbe-n-kjpaw_psl.0.1.UPF"}``. Defaults to the
            :data:`QE_PSEUDO_LIBRARY` filename for each element; any entries
            given here override that default per element.
        soc (bool):
            If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            for every phase (requires fully-relativistic
            pseudopotentials). Magnetic phases are then seeded via
            ``starting_magnetization`` instead of
            ``nspin``/``tot_magnetization`` (which espresso does not allow
            for noncollinear calculations). Should be consistent with the
            defect supercell calculations. Default ``False``.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a
            half-grid (Γ-shifted) Monkhorst-Pack mesh. Default
            ``(0, 0, 0)`` (no shift).
        user_system_settings (dict or None):
            Override default ``&SYSTEM`` namelist settings, e.g.
            ``{"ecutwfc": 80, "ecutrho": 640}``.
        user_control_settings (dict or None):
            Override default ``&CONTROL`` namelist settings.
        user_electron_settings (dict or None):
            Override default ``&ELECTRONS`` namelist settings.

    Returns:
        dict[str, DopedDictSet]:
            Mapping of output folder paths to generated
            :class:`DopedDictSet`\s.
    """
    if output_path is None:
        output_path = _default_qe_competing_phases_path(competing_phases.composition)

    kpoints_by_metallicity = {"non-metals": kpoints_nonmetals, "metals": kpoints_metals}
    dict_sets: dict[str, DopedDictSet] = {}

    for entry, category, structure, folder_name in competing_phases._iter_entries_with_categories():
        if category == "molecules":  # Γ-only sampling is sufficient for a molecule-in-a-box
            continue

        dict_set_kwargs: dict[str, Any] = {  # settings shared by all this phase's convergence calculations
            "calculation": "scf",  # convergence testing uses single-point SCF
            "kpoints_shift": kpoints_shift,
            "pseudo_dir": pseudo_dir,
            "pseudo_map": pseudo_map,
            "is_metal": category == "metals",  # only metallic phases get Gaussian smearing
            "soc": soc,
            "user_control_settings": user_control_settings,
            "user_system_settings": _competing_phase_system_settings(
                entry, user_system_settings, soc, structure, is_metal=category == "metals"
            ),
            "user_electron_settings": user_electron_settings,
        }
        min_k, max_k, step_k = kpoints_by_metallicity[category]
        seen_kgrids: set[tuple[int, ...]] = set()
        for kpoint in np.arange(min_k, max_k, step_k):
            kgrid = _kpoints_grid_from_reciprocal_density(structure, kpoint)
            if tuple(kgrid) in seen_kgrids:
                continue  # this k-point grid already generated, bump reciprocal density more
            seen_kgrids.add(tuple(kgrid))
            kname = "k" + "_".join(str(k) for k in kgrid)
            dict_sets[os.path.join(str(output_path), folder_name, "kpoint_converge", kname)] = (
                DopedDictSet(structure, kpoints=kgrid, **dict_set_kwargs)
            )

    if competing_phases.molecular_entries:
        print(
            f"Note that diatomic molecular phases, calculated as molecules-in-a-box "
            f"({', '.join([e.name for e in competing_phases.molecular_entries])} in this case), do not "
            f"require k-point convergence testing, as Γ-only sampling is sufficient."
        )

    return dict_sets


def write_input_sets(
    input_sets: dict[str, DopedDictSet], positions: bool = True, **kwargs
) -> dict[str, DopedDictSet]:
    r"""
    Write a dictionary of :class:`DopedDictSet`\s (``{output folder: input
    set}``, as generated by :func:`get_kpoint_convergence_sets` /
    :func:`get_relaxation_sets` / :func:`get_singlepoint_sets`) to their
    corresponding output folders (as ``<output folder>/pw.in``), warning if any
    already exist.

    Args:
        input_sets (dict[str, DopedDictSet]):
            Mapping of output folder paths to :class:`DopedDictSet`\s.
        positions (bool):
            Whether to write the structures (the ``CELL_PARAMETERS`` and
            ``ATOMIC_POSITIONS`` cards; see :meth:`DopedDictSet.write_input`).
            Default is ``True``.
        **kwargs:
            Additional keyword arguments for
            :meth:`DopedDictSet.write_input` (e.g. ``filename``).

    Returns:
        dict[str, DopedDictSet]:
            The written ``input_sets``.
    """
    for folder, input_set in input_sets.items():
        if os.path.exists(folder):
            warnings.warn(f"Output folder {folder} already exists. Overwriting files.")
        input_set.write_input(folder, positions=positions, **kwargs)

    return input_sets


def write_kpoint_convergence_files(
    competing_phases: "CompetingPhases",
    output_path: PathLike | None = None,
    kpoints_metals: tuple[float, float, float] = (40, 1000, 5),
    kpoints_nonmetals: tuple[float, float, float] = (5, 120, 5),
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    soc: bool = False,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_system_settings: dict | None = None,
    user_control_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    **kwargs,
) -> dict[str, DopedDictSet]:
    r"""
    Generates and writes ``pw.x`` input files for `k`-point convergence testing
    of competing phases (as single-point ``scf`` calculations), to
    ``<output_path>/<phase>/kpoint_converge/k<kx>_<ky>_<kz>/pw.in``.

    The input sets are generated with :func:`get_kpoint_convergence_sets`
    (see there for details) and written with :func:`write_input_sets`,
    returning the corresponding dictionary of :class:`DopedDictSet`\s.

    Args:
        competing_phases (CompetingPhases):
            The :class:`~doped.chemical_potentials.CompetingPhases` object
            whose entries to generate calculation inputs for.
        output_path (PathLike or None):
            Host-compound parent folder under which all competing-phase
            sub-folders are written. If ``None`` (default), this is set to
            ``"{host_compound}_QE/CompetingPhases_{host_compound}_QE"``
            (e.g. ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a
            ``{host}_QE`` parent folder keyed on the host composition's
            reduced formula.
        kpoints_metals (tuple[float, float, float]):
            ``(min, max, step)`` reciprocal k-point density (Å^-3) range
            to test for metallic phases (``max`` exclusive).
            Default ``(40, 1000, 5)``. Note that only
            unique k-point grids are generated, so small step sizes (as
            default) just result in each k-points choice between ``min``
            and ``max`` being included.
        kpoints_nonmetals (tuple[float, float, float]):
            ``(min, max, step)`` reciprocal k-point density (Å^-3) range
            to test for non-metallic phases (``max`` exclusive).
            Default ``(5, 120, 5)``. Note that
            only unique k-point grids are generated, so small step sizes
            (as default) just result in each k-points choice between
            ``min`` and ``max`` being included.
        pseudo_dir (str):
            Path to the directory containing UPF pseudopotential files,
            written into ``&CONTROL``. Default ``"./pseudo_folder_name/"``.
        pseudo_map (dict or None):
            ``{element_symbol: pseudo_filename}`` mapping, e.g.
            ``{"O": "O.pbe-n-kjpaw_psl.0.1.UPF"}``. Defaults to the
            :data:`QE_PSEUDO_LIBRARY` filename for each element; any entries
            given here override that default per element.
        soc (bool):
            If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            for every phase (requires fully-relativistic
            pseudopotentials). Magnetic phases are then seeded via
            ``starting_magnetization`` instead of
            ``nspin``/``tot_magnetization`` (which espresso does not allow
            for noncollinear calculations). Should be consistent with the
            defect supercell calculations. Default ``False``.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a
            half-grid (Γ-shifted) Monkhorst-Pack mesh. Default
            ``(0, 0, 0)`` (no shift).
        user_system_settings (dict or None):
            Override default ``&SYSTEM`` namelist settings, e.g.
            ``{"ecutwfc": 80, "ecutrho": 640}``.
        user_control_settings (dict or None):
            Override default ``&CONTROL`` namelist settings.
        user_electron_settings (dict or None):
            Override default ``&ELECTRONS`` namelist settings.
        **kwargs:
            Additional keyword arguments for :func:`write_input_sets` /
            :meth:`DopedDictSet.write_input` (e.g. ``filename``).

    Returns:
        dict[str, DopedDictSet]:
            Mapping of output folder paths to the written
            :class:`DopedDictSet`\s.
    """
    input_sets = get_kpoint_convergence_sets(
        competing_phases,
        output_path=output_path,
        kpoints_metals=kpoints_metals,
        kpoints_nonmetals=kpoints_nonmetals,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        soc=soc,
        kpoints_shift=kpoints_shift,
        user_system_settings=user_system_settings,
        user_control_settings=user_control_settings,
        user_electron_settings=user_electron_settings,
    )
    return write_input_sets(input_sets, **kwargs)


def get_relaxation_sets(
    competing_phases: "CompetingPhases",
    output_path: PathLike | None = None,
    kpoints_metals: float = 200,
    kpoints_nonmetals: float = 64,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    use_hse: bool = False,
    soc: bool = False,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_system_settings: dict | None = None,
    user_control_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    user_ions_settings: dict | None = None,
    user_cell_settings: dict | None = None,
    subfolder: PathLike = "espresso_std",
) -> dict[str, DopedDictSet]:
    r"""
    Generates a dictionary of :class:`DopedDictSet`\s for standard (variable-
    cell) ``pw.x`` relaxations of competing phases, using the functional of
    your chosen pseudopotentials by default (consistent with the default
    settings for defect calculations), or HSE06 hybrid DFT if ``use_hse=True``.

    Solid phases are set up as variable-cell (``vc-relax``) relaxations,
    while diatomic molecules (molecules-in-a-box) are relaxed at fixed cell
    (``relax``) with Γ-only sampling. If ``calculation`` is given in
    ``user_control_settings`` it is used instead (as done by
    :func:`get_singlepoint_sets` for ``scf`` calculations), with
    variable-cell modes still replaced by ``relax`` for molecules.

    Automatically sets tetrahedron-method occupations
    (``occupations = 'tetrahedra'``) for `metallic` phases only. Override via
    ``user_system_settings``.

    Each phase input set is keyed by its output folder,
    ``<output_path>/<phase>/<subfolder>`` (``espresso_std`` by default), i.e.
    under a single host-compound parent folder (the layout expected by
    :class:`~doped.chemical_potentials.CompetingPhasesAnalyzer` with
    ``calculator="espresso"``).

    Note that this function uses a single k-point density setting each for
    metals (``kpoints_metals``), non-metals (``kpoints_nonmetals``) and
    molecules (Γ-only), while one will often want to specify custom k-point
    settings for each material individually based on convergence testing
    (e.g. using :func:`get_kpoint_convergence_sets`) to minimise cost.


    Args:
        competing_phases (CompetingPhases):
            The :class:`~doped.chemical_potentials.CompetingPhases` object
            whose entries to generate calculation inputs for.
        output_path (PathLike or None):
            Host-compound parent folder under which all competing-phase
            sub-folders are written. If ``None`` (default), this is set to
            ``"{host_compound}_QE/CompetingPhases_{host_compound}_QE"``
            (e.g. ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a
            ``{host}_QE`` parent folder keyed on the host composition's
            reduced formula.
        kpoints_metals (float):
            Reciprocal k-point density (Å^-3) for metallic phases.
            Default 200.
        kpoints_nonmetals (float):
            Reciprocal k-point density (Å^-3) for non-metallic phases.
            Default 64.
        pseudo_dir (str):
            Path to the directory containing UPF pseudopotential files.
            Default ``"./pseudo_folder_name/"``.
        pseudo_map (dict or None):
            ``{element_symbol: pseudo_filename}`` mapping. Defaults to the
            :data:`~doped.io.espresso.inputs.QE_PSEUDO_LIBRARY` filename
            for each element; any
            entries given here override that default per element.
        use_hse (bool):
            If ``True``, use HSE06 hybrid DFT settings (recommended for
            accurate formation energies). If ``False`` (default), use the
            functional of the pseudopotentials. Note that the functional
            should be consistent with that used for the defect supercell
            calculations, and that ``pw.x`` only supports relaxations and
            noncollinear (SOC) calculations with hybrid functionals for
            norm-conserving pseudopotentials (and their variable-cell
            relaxations cannot be run with k-point pools).
        soc (bool):
            If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            for every phase (requires fully-relativistic
            pseudopotentials). Magnetic phases are then seeded via
            ``starting_magnetization`` instead of
            ``nspin``/``tot_magnetization`` (which espresso does not allow
            for noncollinear calculations). Should be consistent with the
            defect supercell calculations. Default ``False``.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a
            half-grid (Γ-shifted) Monkhorst-Pack mesh. Default
            ``(0, 0, 0)`` (no shift). Ignored for molecules (Γ-only).
        user_system_settings (dict or None):
            Override default ``&SYSTEM`` namelist settings.
        user_control_settings (dict or None):
            Override default ``&CONTROL`` namelist settings.
        user_electron_settings (dict or None):
            Override default ``&ELECTRONS`` namelist settings.
        user_ions_settings (dict or None):
            Override default ``&IONS`` namelist settings (e.g.
            ``{"ion_dynamics": "bfgs"}``).
        user_cell_settings (dict or None):
            Override default ``&CELL`` namelist settings (e.g.
            ``{"cell_dofree": "all", "press": 0.0}``). These apply to the
            variable-cell relaxation performed for solid phases. Ignored
            for molecules, which are relaxed at fixed cell (so ``&CELL`` is
            not written).
        subfolder (PathLike):
            Output folder structure is
            ``<output_path>/<competing_phase_dir>/<subfolder>``. Default is
            ``"espresso_std"``. Set to ``"."`` to write input files directly
            to ``<output_path>/<competing_phase_dir>``, with no subfolders
            created.

    Returns:
        dict[str, DopedDictSet]:
            Mapping of output folder paths to generated
            :class:`DopedDictSet`\s.
    """
    if output_path is None:
        output_path = _default_qe_competing_phases_path(competing_phases.composition)

    # std setup does full cell relaxation, unless the user overrides the ``calculation`` mode (e.g.
    # ``scf`` for the single-point sets of ``get_singlepoint_sets``). Molecules-in-a-box are always
    # fixed-cell (the box is not physical), so any variable-cell mode is replaced by ``relax`` for them:
    user_calculation = (user_control_settings or {}).get("calculation")
    solid_calculation = user_calculation or "vc-relax"
    molecule_calculation = (
        "relax" if user_calculation is None or user_calculation.startswith("vc-") else user_calculation
    )
    kpoints_by_metallicity = {"non-metals": kpoints_nonmetals, "metals": kpoints_metals}
    dict_sets: dict[str, DopedDictSet] = {}

    for entry, category, structure, folder_name in competing_phases._iter_entries_with_categories():
        system_settings = _competing_phase_system_settings(
            entry, user_system_settings, soc, structure, is_metal=category == "metals"
        )
        if category == "molecules":  # Γ-only sampling for molecules-in-a-box
            calculation, kgrid = molecule_calculation, None
        else:
            calculation = solid_calculation
            kgrid = _kpoints_grid_from_reciprocal_density(structure, kpoints_by_metallicity[category])
            if category == "metals":  # tetrahedron method (no smearing) for metals; user setting wins
                system_settings.setdefault("occupations", "tetrahedra")

        dict_sets[os.path.join(str(output_path), folder_name, str(subfolder))] = DopedDictSet(
            structure,
            calculation=calculation,
            kpoints=kgrid,
            kpoints_shift=kpoints_shift,
            pseudo_dir=pseudo_dir,
            pseudo_map=pseudo_map,
            is_metal=category == "metals",
            use_hse=use_hse,
            soc=soc,
            user_control_settings=user_control_settings,
            user_system_settings=system_settings,
            user_electron_settings=user_electron_settings,
            user_ions_settings=user_ions_settings,
            user_cell_settings=user_cell_settings,
        )

    return dict_sets


def write_relaxation_files(
    competing_phases: "CompetingPhases",
    output_path: PathLike | None = None,
    kpoints_metals: float = 200,
    kpoints_nonmetals: float = 64,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    use_hse: bool = False,
    soc: bool = False,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_system_settings: dict | None = None,
    user_control_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    user_ions_settings: dict | None = None,
    user_cell_settings: dict | None = None,
    subfolder: PathLike = "espresso_std",
    **kwargs,
) -> dict[str, DopedDictSet]:
    r"""
    Generates and writes espresso ``pw.in`` input files for standard (variable-
    cell) relaxations of competing phases, to
    ``<output_path>/<phase>/<subfolder>/pw.in`` (``espresso_std`` subfolder by
    default), using the functional of your chosen pseudopotentials by default
    (consistent with the default settings for defect calculations), or HSE06
    hybrid DFT if ``use_hse=True``.

    The input sets are generated with :func:`get_relaxation_sets` (see there
    for details) and written with :func:`write_input_sets`, returning the
    corresponding dictionary of :class:`DopedDictSet`\s.

    Args:
        competing_phases (CompetingPhases):
            The :class:`~doped.chemical_potentials.CompetingPhases` object
            whose entries to generate calculation inputs for.
        output_path (PathLike or None):
            Host-compound parent folder under which all competing-phase
            sub-folders are written. If ``None`` (default), this is set to
            ``"{host_compound}_QE/CompetingPhases_{host_compound}_QE"``
            (e.g. ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a
            ``{host}_QE`` parent folder keyed on the host composition's
            reduced formula.
        kpoints_metals (float):
            Reciprocal k-point density (Å^-3) for metallic phases.
            Default 200.
        kpoints_nonmetals (float):
            Reciprocal k-point density (Å^-3) for non-metallic phases.
            Default 64.
        pseudo_dir (str):
            Path to the directory containing UPF pseudopotential files.
            Default ``"./pseudo_folder_name/"``.
        pseudo_map (dict or None):
            ``{element_symbol: pseudo_filename}`` mapping. Defaults to the
            :data:`~doped.io.espresso.inputs.QE_PSEUDO_LIBRARY` filename
            for each element; any
            entries given here override that default per element.
        use_hse (bool):
            If ``True``, use HSE06 hybrid DFT settings (recommended for
            accurate formation energies). If ``False`` (default), use the
            functional of the pseudopotentials. Note that the functional
            should be consistent with that used for the defect supercell
            calculations, and that ``pw.x`` only supports relaxations and
            noncollinear (SOC) calculations with hybrid functionals for
            norm-conserving pseudopotentials (and their variable-cell
            relaxations cannot be run with k-point pools).
        soc (bool):
            If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            for every phase (requires fully-relativistic
            pseudopotentials). Magnetic phases are then seeded via
            ``starting_magnetization`` instead of
            ``nspin``/``tot_magnetization`` (which espresso does not allow
            for noncollinear calculations). Should be consistent with the
            defect supercell calculations. Default ``False``.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a
            half-grid (Γ-shifted) Monkhorst-Pack mesh. Default
            ``(0, 0, 0)`` (no shift). Ignored for molecules (Γ-only).
        user_system_settings (dict or None):
            Override default ``&SYSTEM`` namelist settings.
        user_control_settings (dict or None):
            Override default ``&CONTROL`` namelist settings.
        user_electron_settings (dict or None):
            Override default ``&ELECTRONS`` namelist settings.
        user_ions_settings (dict or None):
            Override default ``&IONS`` namelist settings (e.g.
            ``{"ion_dynamics": "bfgs"}``).
        user_cell_settings (dict or None):
            Override default ``&CELL`` namelist settings (e.g.
            ``{"cell_dofree": "all", "press": 0.0}``). These apply to the
            variable-cell relaxation performed for solid phases. Ignored
            for molecules, which are relaxed at fixed cell (so ``&CELL`` is
            not written).
        subfolder (PathLike):
            Output folder structure is
            ``<output_path>/<competing_phase_dir>/<subfolder>``. Default is
            ``"espresso_std"``. Set to ``"."`` to write input files directly
            to ``<output_path>/<competing_phase_dir>``, with no subfolders
            created.
        **kwargs:
            Additional keyword arguments for :func:`write_input_sets` /
            :meth:`DopedDictSet.write_input` (e.g. ``filename``).

    Returns:
        dict[str, DopedDictSet]:
            Mapping of output folder paths to the written
            :class:`DopedDictSet`\s.
    """
    input_sets = get_relaxation_sets(
        competing_phases,
        output_path=output_path,
        kpoints_metals=kpoints_metals,
        kpoints_nonmetals=kpoints_nonmetals,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        use_hse=use_hse,
        soc=soc,
        kpoints_shift=kpoints_shift,
        user_system_settings=user_system_settings,
        user_control_settings=user_control_settings,
        user_electron_settings=user_electron_settings,
        user_ions_settings=user_ions_settings,
        user_cell_settings=user_cell_settings,
        subfolder=subfolder,
    )
    return write_input_sets(input_sets, **kwargs)


def get_singlepoint_sets(
    competing_phases: "CompetingPhases",
    output_path: PathLike | None = None,
    kpoints_metals: float = 200,
    kpoints_nonmetals: float = 64,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    use_hse: bool = False,
    soc: bool | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_system_settings: dict | None = None,
    user_control_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    subfolder: PathLike = "SinglePoint",
) -> dict[str, DopedDictSet]:
    r"""
    Generates a dictionary of :class:`DopedDictSet`\s for single-point
    (``scf``) energy calculations of the competing phases, using the functional
    of your chosen pseudopotentials by default (consistent with the default
    settings for defect calculations), or HSE06 hybrid DFT if ``use_hse=True``.

    These are expected to be used as the final energy calculation after
    geometry relaxation (from :func:`write_relaxation_files`), to obtain
    accurate total energies with tight convergence settings (and settings
    consistent with the defect supercell calculations). All phases
    (including molecules-in-a-box, with Γ-only sampling) are set up as
    ``scf`` calculations, generated via :func:`get_relaxation_sets` with the
    ``calculation`` mode overridden -- so the same k-point density,
    pseudopotential, spin-polarisation and metal-occupations handling
    applies (see there).

    If ``soc=True``, spin-orbit coupling (SOC) is included (by setting
    ``noncolin`` and ``lspinorb`` in ``&SYSTEM``). If ``soc`` is not explicitly
    set (i.e. is ``None``), it defaults to ``True`` for systems where the max
    atomic number across all species (host and extrinsic) is Z >= 31 (heavier
    than Zn). Note that SOC calculations in espresso require fully-relativistic
    pseudopotentials, so ``pseudo_map`` should then point at these (the
    :data:`QE_PSEUDO_LIBRARY` defaults are scalar-relativistic).

    For metals, the tetrahedron method (``occupations = 'tetrahedra'``) is
    used by default (as in :func:`get_relaxation_sets`; override via
    ``user_system_settings``).

    Note that this function uses a single k-point density setting each for
    metals (``kpoints_metals``), non-metals (``kpoints_nonmetals``) and
    molecules (Γ-only), while one will often want to specify custom k-point
    settings for each material individually based on convergence testing
    to minimise cost.

    Each phase input set is keyed by its output folder,
    ``<output_path>/<phase>/<subfolder>``, i.e. under a single host-compound
    parent folder.

    Args:
        competing_phases (CompetingPhases):
            The :class:`~doped.chemical_potentials.CompetingPhases` object
            whose entries to generate calculation inputs for.
        output_path (PathLike or None):
            Host-compound parent folder under which all competing-phase
            sub-folders are written. If ``None`` (default), this is set to
            ``"{host_compound}_QE/CompetingPhases_{host_compound}_QE"``
            (e.g. ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a
            ``{host}_QE`` parent folder keyed on the host composition's
            reduced formula.
        kpoints_metals (float):
            Reciprocal k-point density (Å^-3) for metallic phases.
            Default 200.
        kpoints_nonmetals (float):
            Reciprocal k-point density (Å^-3) for non-metallic phases.
            Default 64.
        pseudo_dir (str):
            Path to the directory containing UPF pseudopotential files.
            Default ``"./pseudo_folder_name/"``.
        pseudo_map (dict or None):
            ``{element_symbol: pseudo_filename}`` mapping. Defaults to the
            :data:`~doped.io.espresso.inputs.QE_PSEUDO_LIBRARY` filename
            for each element; any entries given here override that default
            per element. Required to point at fully-relativistic
            pseudopotentials if ``soc`` is ``True``.
        use_hse (bool):
            If ``True``, use HSE06 hybrid DFT settings (recommended for
            accurate formation energies). If ``False`` (default), use the
            functional of the pseudopotentials. Note that the functional
            should be consistent with that used for the defect supercell
            calculations, and that ``pw.x`` only supports relaxations and
            noncollinear (SOC) calculations with hybrid functionals for
            norm-conserving pseudopotentials (and their variable-cell
            relaxations cannot be run with k-point pools).
        soc (bool or None):
            Whether to include spin-orbit coupling (SOC), by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            for every phase (requires fully-relativistic pseudopotentials;
            see ``pseudo_map``). If not set (``soc = None``; default), SOC
            is enabled when the max atomic number across all species (host
            and extrinsic) is Z >= 31. Magnetic phases are then seeded via
            ``starting_magnetization`` instead of
            ``nspin``/``tot_magnetization`` (which espresso does not allow
            for noncollinear calculations). Should be consistent with the
            defect supercell calculations.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a
            half-grid (Γ-shifted) Monkhorst-Pack mesh. Default
            ``(0, 0, 0)`` (no shift). Ignored for molecules (Γ-only).
        user_system_settings (dict or None):
            Override default ``&SYSTEM`` namelist settings.
        user_control_settings (dict or None):
            Override default ``&CONTROL`` namelist settings (take
            precedence over the ``calculation = 'scf'`` setting here).
        user_electron_settings (dict or None):
            Override default ``&ELECTRONS`` namelist settings (e.g. a
            tighter ``{"conv_thr": 1e-10}``).
        subfolder (PathLike):
            Output folder structure is
            ``<output_path>/<competing_phase_dir>/<subfolder>``. Default is
            ``"SinglePoint"``. Set to ``"."`` to write input files directly to
            ``<output_path>/<competing_phase_dir>``, with no subfolders
            created.

    Returns:
        dict[str, DopedDictSet]:
            Mapping of output folder paths to generated
            :class:`DopedDictSet`\s.
    """
    if soc is None:
        all_elements = competing_phases.intrinsic_elements + getattr(
            competing_phases, "extrinsic_elements", []
        )
        soc = max(Element(el).Z for el in all_elements) >= 31

        if soc:  # if SOC being automatically determined, print an info message
            print(
                "Spin-orbit coupling (SOC) is being used by default for competing phase single-point "
                "calculations, as the heaviest element present (across intrinsic and extrinsic species) "
                "has an atomic number Z >= 31"
            )

    # single-point (``scf``) calculations for all phases; user ``&CONTROL`` settings take precedence:
    sp_control_settings = {"calculation": "scf", **(user_control_settings or {})}

    # reuse the relaxation set generation with the single-point ``calculation`` override
    return get_relaxation_sets(
        competing_phases,
        output_path=output_path,
        kpoints_metals=kpoints_metals,
        kpoints_nonmetals=kpoints_nonmetals,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        use_hse=use_hse,
        soc=soc,
        kpoints_shift=kpoints_shift,
        user_system_settings=user_system_settings,
        user_control_settings=sp_control_settings,
        user_electron_settings=user_electron_settings,
        subfolder=subfolder,
    )


def write_singlepoint_files(
    competing_phases: "CompetingPhases",
    output_path: PathLike | None = None,
    kpoints_metals: float = 200,
    kpoints_nonmetals: float = 64,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    use_hse: bool = False,
    soc: bool | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_system_settings: dict | None = None,
    user_control_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    subfolder: PathLike = "SinglePoint",
    positions: bool = False,
    **kwargs,
) -> dict[str, DopedDictSet]:
    r"""
    Generates and writes espresso ``pw.in`` input files for single-point
    (``scf``) energy calculations of the competing phases, to
    ``<output_path>/<phase>/<subfolder>/pw.in`` (``SinglePoint`` subfolder by
    default), using the functional of your chosen pseudopotentials by default
    (consistent with the default settings for defect calculations), or HSE06
    hybrid DFT if ``use_hse=True``.

    If ``soc`` is not explicitly set, SOC is enabled when the max atomic number
    across all species (host and extrinsic) is Z >= 31 (requiring
    fully-relativistic pseudopotentials via ``pseudo_map``).

    The input sets are generated with :func:`get_singlepoint_sets` (see
    there for details) and written with :func:`write_input_sets`, returning
    the corresponding dictionary of :class:`DopedDictSet`\s.

    Args:
        competing_phases (CompetingPhases):
            The :class:`~doped.chemical_potentials.CompetingPhases` object
            whose entries to generate calculation inputs for.
        output_path (PathLike or None):
            Host-compound parent folder under which all competing-phase
            sub-folders are written. If ``None`` (default), this is set to
            ``"{host_compound}_QE/CompetingPhases_{host_compound}_QE"``
            (e.g. ``"MgO_QE/CompetingPhases_MgO_QE"``), i.e. a
            ``{host}_QE`` parent folder keyed on the host composition's
            reduced formula.
        kpoints_metals (float):
            Reciprocal k-point density (Å^-3) for metallic phases.
            Default 200.
        kpoints_nonmetals (float):
            Reciprocal k-point density (Å^-3) for non-metallic phases.
            Default 64.
        pseudo_dir (str):
            Path to the directory containing UPF pseudopotential files.
            Default ``"./pseudo_folder_name/"``.
        pseudo_map (dict or None):
            ``{element_symbol: pseudo_filename}`` mapping. Defaults to the
            :data:`~doped.io.espresso.inputs.QE_PSEUDO_LIBRARY` filename
            for each element; any entries given here override that default
            per element. Required to point at fully-relativistic
            pseudopotentials if ``soc`` is ``True``.
        use_hse (bool):
            If ``True``, use HSE06 hybrid DFT settings (recommended for
            accurate formation energies). If ``False`` (default), use the
            functional of the pseudopotentials. Note that the functional
            should be consistent with that used for the defect supercell
            calculations, and that ``pw.x`` only supports relaxations and
            noncollinear (SOC) calculations with hybrid functionals for
            norm-conserving pseudopotentials (and their variable-cell
            relaxations cannot be run with k-point pools).
        soc (bool or None):
            Whether to include spin-orbit coupling (SOC), by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            for every phase (requires fully-relativistic pseudopotentials;
            see ``pseudo_map``). If not set (``soc = None``; default), SOC
            is enabled when the max atomic number across all species (host
            and extrinsic) is Z >= 31. Should be consistent with the defect
            supercell calculations.
        kpoints_shift (tuple):
            ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a
            half-grid (Γ-shifted) Monkhorst-Pack mesh. Default
            ``(0, 0, 0)`` (no shift). Ignored for molecules (Γ-only).
        user_system_settings (dict or None):
            Override default ``&SYSTEM`` namelist settings.
        user_control_settings (dict or None):
            Override default ``&CONTROL`` namelist settings (take
            precedence over the ``calculation = 'scf'`` setting here).
        user_electron_settings (dict or None):
            Override default ``&ELECTRONS`` namelist settings (e.g. a
            tighter ``{"conv_thr": 1e-10}``).
        subfolder (PathLike):
            Output folder structure is
            ``<output_path>/<competing_phase_dir>/<subfolder>``. Default is
            ``"SinglePoint"``. Set to ``"."`` to write input files directly to
            ``<output_path>/<competing_phase_dir>``, with no subfolders
            created.
        positions (bool):
            Whether to write the structures (the ``CELL_PARAMETERS`` and
            ``ATOMIC_POSITIONS`` cards). Default is ``False`` (as for
            ``POSCAR`` files with VASP), as single-point calculations are
            intended to be run on the relaxed structures from
            :func:`write_relaxation_files`, which should be added to the
            written ``pw.in`` files (``pw.x`` exits with an error without
            them).
        **kwargs:
            Additional keyword arguments for :func:`write_input_sets` /
            :meth:`DopedDictSet.write_input` (e.g. ``filename``).

    Returns:
        dict[str, DopedDictSet]:
            Mapping of output folder paths to the written
            :class:`DopedDictSet`\s.
    """
    input_sets = get_singlepoint_sets(
        competing_phases,
        output_path=output_path,
        kpoints_metals=kpoints_metals,
        kpoints_nonmetals=kpoints_nonmetals,
        pseudo_dir=pseudo_dir,
        pseudo_map=pseudo_map,
        use_hse=use_hse,
        soc=soc,
        kpoints_shift=kpoints_shift,
        user_system_settings=user_system_settings,
        user_control_settings=user_control_settings,
        user_electron_settings=user_electron_settings,
        subfolder=subfolder,
    )
    return write_input_sets(input_sets, positions=positions, **kwargs)


def _set_spin_polarisation(
    system_settings: dict,
    entry: ComputedEntry | ComputedStructureEntry,
    structure: Structure,
    is_metal: bool,
) -> None:
    r"""
    Set the spin polarisation settings in the espresso ``&SYSTEM`` namelist
    (``system_settings``, including any user settings, which take precedence)
    for the calculation of ``entry`` with ``structure`` (from
    ``_iter_entries_with_categories()``, which can be a smaller cell than that
    of the Materials Project calculation); the espresso analogue of
    :func:`doped.io.vasp.inputs._set_spin_polarisation`.

    If ``structure`` has non-zero ``magmom`` site properties, the entry has a
    total magnetization (scaled to ``structure``) greater than 0.1 μB, or
    ``nspin = 2`` is set by the user, the calculation is spin-polarised:

    - Collinear calculations: ``nspin = 2``. With fixed occupations
      (non-metals and molecules by default), ``pw.x`` requires
      ``tot_magnetization``, which is set to the integer-rounded total
      magnetization (as ``NUPDOWN`` with VASP). If this is unknown or
      inconsistent with the ``magmom`` site properties (e.g. for user-supplied
      host structures with a different magnetic ordering), the magnetization
      is instead left unconstrained (as with VASP), using Gaussian smearing.
    - noncollinear (``soc``) calculations: ``nspin``/``tot_magnetization`` are
      not allowed by ``pw.x``, so the magnetization is unconstrained.

    Any ``magmom`` site properties are used as ``starting_magnetization`` seeds
    when writing the ``pw.in`` (with sites split into separate atomic species
    by magnetic moment; see :meth:`DopedDictSet.write_qe_pw_input`), also with
    a constrained magnetization, so that the same atomic species are written
    for all calculations of a phase. Without these, an unconstrained
    magnetization (also the case for metals) is seeded with
    :data:`DEFAULT_STARTING_MAGNETIZATION` for all species.
    """
    magnetization = _get_competing_phase_magnetization(entry, structure)
    magmoms = structure.site_properties.get("magmom", [])
    has_magmoms = bool(np.any(np.abs(magmoms) >= 0.1))  # (as for the seeds in ``_get_seed_magmoms``)
    if (
        not has_magmoms
        and (magnetization is None or magnetization <= 0.1)
        and int(system_settings.get("nspin", 1)) != 2
    ):
        return  # ``nspin`` not set, so no spin polarisation

    if not system_settings.get("noncolin"):
        if int(system_settings.setdefault("nspin", 2)) != 2 or "tot_magnetization" in system_settings:
            return  # (set by the user)
        occupations = system_settings.get("occupations", "smearing" if is_metal else "fixed")
        if str(occupations).lower() == "fixed":  # ``pw.x`` then requires ``tot_magnetization``
            tot_magnetization = (
                None
                if magnetization is None
                else _get_constrained_total_magnetization(magnetization, magmoms)
            )
            if tot_magnetization is not None:
                system_settings["tot_magnetization"] = tot_magnetization
                return
            _set_default_smearing(system_settings)  # unconstrained magnetization

    if not has_magmoms:  # unconstrained magnetization, to be seeded (else by the site ``magmom``\s)
        system_settings.setdefault("starting_magnetization", DEFAULT_STARTING_MAGNETIZATION)


def _set_default_smearing(system_settings: dict, degauss: float = 0.005) -> None:
    """
    Set ``occupations = 'smearing'`` (if ``occupations`` is not already set),
    with ``smearing = 'gaussian'`` and ``degauss`` (in Ry; 0.005 by default, as
    used for metallic phases), in the espresso ``&SYSTEM`` namelist.

    If ``occupations`` is already set to anything other than ``'smearing'``
    (e.g. ``'tetrahedra'``, as used for metals by :func:`get_relaxation_sets`
    / :func:`get_singlepoint_sets`), no smearing parameters are added.
    """
    system_settings.setdefault("occupations", "smearing")
    if str(system_settings["occupations"]).lower() == "smearing":
        system_settings.setdefault("smearing", "gaussian")
        system_settings.setdefault("degauss", degauss)
