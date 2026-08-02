"""
Code to generate Quantum ESPRESSO (``pw.x``) defect calculation input files.

Please note that all structures are written with 'ibrav = 0' and the subsequent CELL_PARAMETERS card
is used to write lattice vectors.
"""
#TODO: Input files for multi-oxidation states
import copy
import os
import warnings

import numpy as np
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
from doped.generation import DefectsGenerator, _custom_formatwarning
from doped.utils.efficiency import Element, Structure
from doped.utils.parsing import _get_defect_supercell

_ignore_pmg_warnings()
warnings.formatwarning = _custom_formatwarning

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# QE ``pw.x`` input defaults, loaded from the YAML/JSON sets in ``doped/QE_sets``
# Please cite the SSSP workflow in the way described here:
# https://legacy.materialscloud.org/discover/sssp/table/efficiency
default_qe_SSSP_set: dict = loadfn(os.path.join(MODULE_DIR, "QE_sets/SSSP_Convergence_set.yaml"))
default_qe_HSE_set: dict = loadfn(os.path.join(MODULE_DIR, "QE_sets/HSE_set.yaml"))
# If you use the SSSP pseudopotentials, please cite in the way described here:
# https://legacy.materialscloud.org/discover/sssp/table/efficiency
qe_SSSP_pseudo_filenames: dict = {
    element: metadata["filename"]
    for element, metadata in loadfn(
        os.path.join(MODULE_DIR, "QE_sets/SSSP_1.3.0_PBE_efficiency.json")
    ).items()
}


def _kpoints_grid_from_reciprocal_density(
    structure: Structure, reciprocal_density: float
) -> list[int]:
    """``[kx, ky, kz]`` Monkhorst-Pack grid at the given k-points-per-Å^-3 density."""
    kpoints_obj = Kpoints.automatic_density_by_vol(structure, kppvol=reciprocal_density)
    return [int(k) for k in kpoints_obj.kpts[0]]


def _write_qe_pw_input(
    filepath: str,
    structure: Structure,
    namelist_settings: dict[str, dict],
    kpoints: list[int] | None,
    pseudo_map: dict[str, str] | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Write a QE ``pw.in`` for ``structure``.

    Args:
        filepath: Destination path for ``pw.in``.
        structure: Structure to write.
        namelist_settings: ``{namelist: {key: value, ...}}`` for the QE
            ``control``/``system``/``electrons``/``ions``/``cell`` namelists.
        kpoints: ``[kx, ky, kz]`` Monkhorst-Pack grid, or ``None`` for
            Γ-only sampling.
        pseudo_map: ``{element: UPF filename}`` overrides on top of the SSSP
            defaults; missing elements fall back to ``"{element}.upf"``.
        kpoints_shift: ``(sx, sy, sz)`` grid offset (each 0 or 1) written as
            the second line of the ``K_POINTS automatic`` card, e.g.
            ``(1, 1, 1)`` for a half-grid (Γ-shifted) Monkhorst-Pack mesh.
            Default ``(0, 0, 0)`` (no shift). Ignored for Γ-only sampling
            (``kpoints=None``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    # strip any oxidation states (e.g. on ``DefectsGenerator`` supercells) so that
    # species labels are plain element symbols (For example:"O", not "O2-"), as QE expects.
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
            f"writing QE input to {filepath}."
        )
        structure = structure.copy()
        structure.remove_oxidation_states()
    unique_species = sorted(
        {str(el) for el in structure.species}, key=lambda s: Element(s).Z
    )
    resolved_pseudos = {
        sp: qe_SSSP_pseudo_filenames.get(sp, f"{sp}.upf") for sp in unique_species
    }
    resolved_pseudos.update(pseudo_map or {})

    if (namelist_settings.get("system") or {}).get("starting_magnetization") is not None:
        namelist_settings = {nl: dict(settings) for nl, settings in namelist_settings.items()}
        system_settings = namelist_settings["system"]
        if not isinstance(system_settings["starting_magnetization"], list | tuple):
            system_settings["starting_magnetization"] = [
                system_settings["starting_magnetization"]
            ] * len(unique_species)
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
            f"Ignoring unrecognised QE namelist(s) {invalid_namelists} in `namelist_settings` when "
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
    Build the per-structure base namelist dict from the SSSP convergence
    YAML defaults (or the HSE06 hybrid-DFT defaults if ``use_hse``): sets
    ``ibrav=0``, ``nat``, ``ntyp``, ``pseudo_dir``, optional metallic
    smearing / spin-orbit coupling (``soc``), and merges any user overrides.

    Note that ``&IONS``/``&CELL`` overrides (``user_ions_settings`` /
    ``user_cell_settings``) only affect relaxation inputs: for the SCF
    convergence calculations QE ignores ``&IONS`` and ``&CELL`` is not
    written, and ``&CELL`` is also dropped for fixed-cell (``relax``) inputs.
    """
    base = copy.deepcopy(default_qe_HSE_set if use_hse else default_qe_SSSP_set)
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

    return base


def qe_convergence_setup_from_structure(
    structure: Structure,
    output_dir: PathLike | None = None,
    kpoint_density_range: tuple = (20, 200, 20),
    kpoint_sweep_ecutwfc: int | None = None,
    ecut_range: tuple = (20, 90, 10),
    ecut_sweep_kpoint_density: int = 100,
    is_metal: bool = False,
    soc: bool = False,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_control_settings: dict | None = None,
    user_system_settings: dict | None = None,
    user_electron_settings: dict | None = None,
) -> dict[str, list[str]]:
    """
    Generate QE ``pw.in`` files for k-point and plane-wave cutoff
    (``ecutwfc``) convergence testing from a single ``pymatgen``
    ``Structure``.

    Elements are read from ``structure.species`` and matched against the
    bundled SSSP 1.3.0 PBE Efficiency library (``doped/QE_sets``); the
    resolved UPF filenames are written into the ``ATOMIC_SPECIES`` card of
    every ``pw.in``. ``pseudo_map`` can override individual entries or
    supply pseudos for elements absent from SSSP.

    Two sub-trees are written under ``output_dir``:

    - ``kpoint_converge/k<kx>_<ky>_<kz>/pw.in`` — ``ecutwfc`` held at
      ``kpoint_sweep_ecutwfc`` (or the set default if ``None``), k-grid
      swept over ``kpoint_density_range``. Duplicate grids produced by
      nearby densities are skipped.
    - ``ecut_convergence/ecutwfc_<N>/pw.in`` — k-grid held at
      ``ecut_sweep_kpoint_density``, ``ecutwfc`` swept over ``ecut_range``.
      ``ecutrho`` is left at the set default.

    After running these and choosing converged values, call
    :func:`qe_relax_setup_from_structure` (with those converged values
    and, ideally, the relaxed structure) to write the final ``vc-relax``.

    Args:
        structure: Input structure (no MP lookup is performed).
        output_dir: Root folder for the two sub-trees. If ``None`` (default),
            written to a host-named parent folder as
            ``"{host_formula}_QE/Bulk_convergence"`` (e.g.
            ``"MgO_QE/Bulk_convergence"``), so all QE inputs for a given host
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
        pseudo_map: ``{element: UPF filename}`` overrides on top of SSSP.
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
        kname = "k" + ("_" * (kgrid[0] // 10)) + ",".join(str(k) for k in kgrid)
        filepath = os.path.join(str(output_dir), "kpoint_converge", kname, "pw.in")
        _write_qe_pw_input(
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
        filepath = os.path.join(str(output_dir), "ecut_convergence", f"ecutwfc_{ecut}", "pw.in")
        _write_qe_pw_input(
            filepath=filepath,
            structure=structure,
            namelist_settings=ecut_scf,
            kpoints=ecut_kgrid,
            pseudo_map=pseudo_map,
            kpoints_shift=kpoints_shift,
        )
        written["ecut_convergence"].append(filepath)

    return written


def qe_relax_setup_from_structure(
    structure: Structure,
    ecutwfc: int,
    kpoint_density: int,
    output_dir: PathLike | None = None,
    ecutrho: int | None = None,
    calculation: str = "vc-relax",
    is_metal: bool = False,
    use_hse: bool = False,
    soc: bool = False,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_control_settings: dict | None = None,
    user_system_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    user_ions_settings: dict | None = None,
    user_cell_settings: dict | None = None,
    starting_magnetization: float | None = None,
) -> str:
    """
    Generate a QE ``pw.in`` for a final relaxation, using the converged
    ``ecutwfc`` and k-point density obtained from
    :func:`qe_convergence_setup_from_structure`.

    Pass the structure here once convergence
    is done — element identification and pseudopotential lookup use the
    same SSSP 1.3.0 PBE Efficiency map as the convergence helper.

    Args:
        structure: Structure to relax (ideally the one returned by the
            converged SCF/relax workflow).
        ecutwfc: Converged plane-wave cutoff (Ry) from the ``ecutwfc``
            sweep. Required.
        kpoint_density: Converged reciprocal k-point density (Å^-3) from
            the k-grid sweep. Required.
        output_dir: Calculation folder; ``pw.in`` is written into an
            ``espresso_std`` subfolder of it (``output_dir/espresso_std/pw.in``,
            matching the layout ``doped`` parses). If ``None`` (default),
            written to a host-named parent folder as
            ``"{host_formula}_QE/Bulk_relax"`` (e.g. ``"MgO_QE/Bulk_relax"``),
            so all QE inputs for a given host live under a single ``{host}_QE``
            folder (matching the VASP example layout).
        ecutrho: Optional ``ecutrho`` (Ry); ``None`` keeps the set default.
        calculation: ``"vc-relax"`` (default, full cell+ions) or
            ``"relax"`` (ions only, fixed cell).
        is_metal: If ``True``, set ``occupations='smearing'``,
            ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``.
        use_hse: If ``True``, use the HSE06 hybrid-DFT ``&SYSTEM`` defaults
            (``doped/QE_sets/HSE_set.yaml``) as the base set instead of the
            (GGA) SSSP set. Should be consistent with the defect supercell
            calculations. Default ``False``.
        soc: If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM``
            (requires fully-relativistic pseudopotentials). Should be
            consistent with the convergence tests / defect supercell
            calculations. Default ``False``.
        pseudo_dir: Path written to ``&CONTROL.pseudo_dir``.
        pseudo_map: ``{element: UPF filename}`` overrides on top of SSSP.
        kpoints_shift: ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a half-grid
            (Γ-shifted) Monkhorst-Pack mesh. Default ``(0, 0, 0)`` (no shift).
        user_control_settings, user_system_settings, user_electron_settings:
            Per-namelist overrides merged on top of the YAML defaults.
        user_ions_settings: Overrides for the ``&IONS`` namelist (e.g.
            ``{"ion_dynamics": "bfgs"}``).
        user_cell_settings: Overrides for the ``&CELL`` namelist (e.g.
            ``{"cell_dofree": "all"}``), applied to the variable-cell
            relaxation. Ignored for ``calculation="relax"`` (fixed cell), for
            which ``&CELL`` is not written.
        starting_magnetization: If not ``None``, set the QE ``&SYSTEM``
            ``starting_magnetization`` to this value for every atomic species
            (and ``nspin=2``), seeding a spin-polarised calculation. Default
            ``None`` (no spin polarisation).

    Returns:
        Path of the written ``pw.in`` (``output_dir/espresso_std/pw.in``).
    """
    if output_dir is None:
        output_dir = os.path.join(f"{structure.composition.reduced_formula}_QE", "Bulk_relax")

    base = _build_qe_base_settings(
        structure,
        pseudo_dir,
        is_metal,
        user_control_settings,
        user_system_settings,
        user_electron_settings,
        use_hse=use_hse,
        user_ions_settings=user_ions_settings,
        user_cell_settings=user_cell_settings,
        soc=soc,
    )
    base["control"]["calculation"] = calculation
    base["system"]["ecutwfc"] = ecutwfc
    if ecutrho is not None:
        base["system"]["ecutrho"] = ecutrho
    if starting_magnetization is not None:
        base["system"]["starting_magnetization"] = starting_magnetization

    kgrid = _kpoints_grid_from_reciprocal_density(structure, kpoint_density)
    filepath = os.path.join(str(output_dir), "espresso_std", "pw.in")
    _write_qe_pw_input(
        filepath=filepath,
        structure=structure,
        namelist_settings=base,
        kpoints=kgrid,
        pseudo_map=pseudo_map,
        kpoints_shift=kpoints_shift,
    )
    return filepath


def qe_defect_setup_from_generator(
    defect_generator: DefectsGenerator,
    ecutwfc: int,
    kpoint_density: int,
    output_dir: PathLike | None = None,
    ecutrho: int | None = None,
    is_metal: bool = False,
    use_hse: bool = False,
    soc: bool = False,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    kpoints_shift: tuple[int, int, int] = (0, 0, 0),
    user_control_settings: dict | None = None,
    user_system_settings: dict | None = None,
    user_electron_settings: dict | None = None,
    user_ions_settings: dict | None = None,
    starting_magnetization: float | None = 0.1,
    include_bulk: bool = True,
) -> dict[str, str]:
    """
    Generate fixed-cell QE ``pw.in`` files for every defect supercell in a
    :class:`~doped.generation.DefectsGenerator`.

    For each entry in ``defect_generator.defect_entries`` a ``relax`` ( fixed cell relaxation)
    ``pw.in`` is written to ``output_dir/<defect_name>/espresso_std/pw.in``, with QE's
    ``tot_charge`` set from the entry's ``charge_state`` (``doped``/QE convention:
    ``tot_charge`` = electrons removed = positive charge state).
     The neutral bulk supercell reference is also written to
    ``output_dir/{host_formula}_bulk/espresso_std/pw.in`` (e.g.
    ``MgO_bulk``) when ``include_bulk`` is ``True``.
    Each calculation thus lives in its own ``espresso_std`` subfolder, matching
    the layout ``doped`` parses.

    The defect supercell keeps the bulk supercell volume, so a fixed-cell
    ``relax`` is used. Element identification and pseudopotential
     lookup use the same SSSP 1.3.0 PBE Efficiency map as :func:
     `qe_relax_setup_from_structure`, which this delegates to per entry.

    Run :func:`qe_convergence_setup_from_structure` first (on the primitive
    cell) to obtain the converged ``ecutwfc`` and ``kpoint_density`` passed
    here.

    Args:
        defect_generator: A ``doped`` :class:`~doped.generation.DefectsGenerator` instance.
        ecutwfc: Converged plane-wave cutoff (Ry). Required.
        kpoint_density: Converged reciprocal k-point density (Å^-3). Required.
        output_dir: Root folder for the per-defect (and ``{host}_bulk``)
            sub-folders. If ``None`` (default), written to a host-named parent
            folder as ``"{host_formula}_QE/Defects"`` (e.g.
            ``"MgO_QE/Defects"``), so all QE inputs for a given host live under
            a single ``{host}_QE`` folder (matching the VASP example layout).
        ecutrho: Optional ``ecutrho`` (Ry); ``None`` keeps the set default.
        is_metal: If ``True``, set ``occupations='smearing'``,
            ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``.
        use_hse: If ``True``, use the HSE06 hybrid-DFT ``&SYSTEM`` defaults
            (``doped/QE_sets/HSE_set.yaml``) as the base set for all defect
            (and bulk) inputs instead of the (GGA) SSSP set. Should be
            consistent with the bulk relaxation. Default ``False``.
        soc: If ``True``, include spin-orbit coupling by setting
            ``noncolin=.true.`` and ``lspinorb=.true.`` in ``&SYSTEM`` of
            every defect (and bulk) input (requires fully-relativistic
            pseudopotentials). Note that ``nspin`` is not written for
            noncolinear calculations (not allowed by QE); the
            ``starting_magnetization`` spin seeding still applies. Should be
            consistent with the bulk relaxation. Default ``False``.
        pseudo_dir: Path written to ``&CONTROL.pseudo_dir``.
        pseudo_map: ``{element: UPF filename}`` overrides on top of SSSP.
        kpoints_shift: ``(sx, sy, sz)`` grid offset (each 0 or 1) for the
            ``K_POINTS automatic`` card, e.g. ``(1, 1, 1)`` for a half-grid
            (Γ-shifted) Monkhorst-Pack mesh. Default ``(0, 0, 0)`` (no shift).
        user_control_settings, user_system_settings, user_electron_settings:
            Per-namelist overrides merged on top of the YAML defaults. Note
            that ``tot_charge`` in ``user_system_settings`` is overridden
            per defect by its charge state.
        user_ions_settings: Overrides for the ``&IONS`` namelist (e.g.
            ``{"ion_dynamics": "bfgs"}``), applied to the fixed-cell ionic
            relaxation of each defect (and bulk) supercell. There is no
            ``&CELL`` override here as the defect supercells are relaxed at
            fixed cell (``relax``), for which ``&CELL`` is not written.
        starting_magnetization: Default value for the QE ``&SYSTEM``
            ``starting_magnetization`` (applied to every atomic species, with
            ``nspin=2``) in every defect ``pw.in``, giving a spin-polarised
            starting point. The neutral bulk reference is non spin-polarised. Default ``0.1``.
        include_bulk: If ``True`` (default), also write the neutral bulk
            supercell reference to
            ``output_dir/{host_formula}_bulk/espresso_std/pw.in``.

    Returns:
        ``{name: pw.in path}`` for every written input file (defect names,
        plus ``"{host_formula}_bulk"`` (e.g. ``"MgO_bulk"``) when
        ``include_bulk`` is ``True``).
    """
    host_formula = defect_generator.bulk_supercell.composition.reduced_formula
    if output_dir is None:
        output_dir = os.path.join(f"{host_formula}_QE", "Defects")
    bulk_name = f"{host_formula}_bulk"

    structures: dict[str, Structure] = {}
    charges: dict[str, int] = {}
    for name, defect_entry in defect_generator.defect_entries.items():
        supercell = _get_defect_supercell(defect_entry)
        if supercell is None:
            raise ValueError(
                f"Could not determine the defect supercell for {name!r} from the "
                f"`DefectsGenerator`, so cannot write its QE input file."
            )
        structures[name] = supercell
        charges[name] = defect_entry.charge_state

    if include_bulk:
        structures[bulk_name] = defect_generator.bulk_supercell
        charges[bulk_name] = 0

    written: dict[str, str] = {}
    for name, structure in structures.items():
        written[name] = qe_relax_setup_from_structure(
            structure=structure,
            ecutwfc=ecutwfc,
            kpoint_density=kpoint_density,
            output_dir=os.path.join(str(output_dir), name),
            ecutrho=ecutrho,
            calculation="relax",  # fixed cell relaxation of the defect supercell based on the optimized bulk structure
            is_metal=is_metal,
            use_hse=use_hse,
            soc=soc,
            pseudo_dir=pseudo_dir,
            pseudo_map=pseudo_map,
            kpoints_shift=kpoints_shift,
            user_control_settings=user_control_settings,
            user_system_settings={**(user_system_settings or {}), "tot_charge": charges[name]},
            user_electron_settings=user_electron_settings,
            user_ions_settings=user_ions_settings,
            # the neutral bulk reference is left non-spin-polarised:
            starting_magnetization=None if name == bulk_name else starting_magnetization,
        )

    return written