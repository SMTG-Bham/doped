"""
Quantum ESPRESSO (``pw.x`` and pp.x(.cube files)) calculation output parsing for ``doped``.
"""

import contextlib
import inspect
import os
import warnings
from pathlib import Path
from typing import Any, Literal  # noqa: F401  (kept for the moved annotations)
from xml.parsers.expat import ExpatError

import numpy as np
from ase.io.cube import read_cube_data
from pymatgen.core.units import Ry_to_eV
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.espresso.outputs.pwxml import PWxml
from pymatgen.io.espresso.utils import parse_pwvals
from pymatgen.io.vasp import VolumetricData
from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.util.typing import PathLike
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from doped.io import utils as _io_utils
from doped.io.espresso.inputs import SUBFOLDER_PRIORITY  # noqa: F401  (re-exported backend constant)
from doped.io.outputs import CalculationOutputs
from doped.io.utils import _multiple_files_warning, find_archived_fname
from doped.utils.parsing import (  # VASP-only helpers; no ``doped.io`` home outside ``doped/io/vasp/``
    get_locpot,
    get_magnetization_from_vasprun,
)

BOHR_TO_ANGSTROM = 0.529177


CALC_OUTPUT_MASK = ("espresso.xml", ".xml", "espresso.xml.gz", ".xml.gz")
# unlike VASP's LOCPOT/OUTCAR split, one ``pp.x`` cube serves both FNV and eFNV corrections:
PLANAR_POTENTIALS_FILE = ".cube"
SITE_POTENTIALS_FILE = ".cube"
PROJECTIONS_FILE = ".projwfc_up"

_CALC_OUTPUT_PARSING_ACTION = "parse the calculation energy and metadata."
_POTENTIALS_PARSING_ACTION = (
    "parse the electrostatic potentials (planar-averaged and/or atomic-site) and compute the "
    "charged-defect finite-size corrections."
)
_PROJECTIONS_PARSING_ACTION = (
    "parse the orbital projections (projected eigenvalues), for eigenvalue & perturbed host state "
    "(PHS) analysis."
)
FILE_PARSING_ACTIONS: dict[str, str] = {
    **dict.fromkeys(CALC_OUTPUT_MASK, _CALC_OUTPUT_PARSING_ACTION),
    PLANAR_POTENTIALS_FILE: _POTENTIALS_PARSING_ACTION,
    SITE_POTENTIALS_FILE: _POTENTIALS_PARSING_ACTION,
    "LOCPOT": _POTENTIALS_PARSING_ACTION,
    PROJECTIONS_FILE: _PROJECTIONS_PARSING_ACTION,
}
"""
``{file pattern: action description}`` for the files parsed by this backend, used in the "multiple
matching files found" warnings emitted by ``_multiple_files_warning`` (part of the ``doped.io``
backend protocol). Without an entry for each pattern, these warnings error out rather than warn.
"""


def _make_pwxml_property_settable(prop_name: str) -> None:
    """
    Replace a read-only ``PWxml`` property with a read-write version, storing
    any assigned value in a hidden instance attribute which then
    short-circuits the original getter.

    Calculator-agnostic ``doped`` code nulls out the eigenvalue arrays on
    parsed run objects (``raw["vasprun"]``) once used, to reduce memory demand
    (e.g. ``analysis.py`` / ``core.py`` memory-cleanup blocks setting
    ``projected_eigenvalues`` / ``projected_magnetization`` / ``eigenvalues``
    to ``None``). On ``Vasprun`` these are all plain attributes, but on
    ``PWxml`` the first two are read-only properties, so assignment raises
    ``AttributeError`` -- this patch makes them settable. No-op if the
    attribute is not a read-only property (e.g. if ``pymatgen-io-espresso``
    later adds a setter or converts it to a plain attribute).
    """
    prop = getattr(PWxml, prop_name, None)
    if not isinstance(prop, property) or prop.fset is not None:
        return  # plain attribute, or already settable; nothing to patch

    fget = prop.fget
    backing_attr = f"_{prop_name}_override"

    def getter(self):
        if backing_attr in self.__dict__:  # only fall back to the original (possibly
            return self.__dict__[backing_attr]  # expensive) getter if never assigned
        return fget(self)

    def setter(self, value):
        self.__dict__[backing_attr] = value

    setattr(PWxml, prop_name, property(getter, setter, doc=prop.__doc__))


# ``eigenvalues`` is a plain ``PWxml`` attribute and needs no patch; these two are
# read-only properties (unlike on ``Vasprun``), patched at import time to accept assignment:
for _prop_name in ("projected_eigenvalues", "projected_magnetization"):
    _make_pwxml_property_settable(_prop_name)





def _file_parsing_action(file_type: PathLike) -> str:
    """
    The action description for ``file_type`` (see :data:`FILE_PARSING_ACTIONS`),
    for use in ``_multiple_files_warning``, with a generic fallback for any
    unlisted file type.
    """
    return FILE_PARSING_ACTIONS.get(str(file_type), "Need to parse the calculation outputs.")




def get_espresso_run(
    espressorun_path: PathLike, parse_mag: bool = False, **kwargs
) -> PWxml:
    """
    Parse the espresso ``.xml(.gz)`` output at ``espressorun_path`` into a
    ``PWxml`` object; the espresso analogue of ``get_vasprun`` for VASP.

    ``kwargs`` are passed to ``PWxml`` (with ``parse_dos = False`` by default).
    Note that ``parse_projected_eigen = True`` additionally requires a
    ``projwfc.x`` ``filproj`` file, and ``parse_dos = True`` a ``dos.x``
    ``fildos``, as ``pw.x`` writes neither to its XML.

    ``PWxml.__init__`` ends in ``**_`` ("ignored arguments for compatibility
    with ``Vasprun``"), so ``Vasprun``-only keywords are silently swallowed
    rather than rejected. Two consequences worth knowing:

    - There is no ``parse_eigen`` analogue -- eigenvalues are always parsed.
    - There is no ``exception_on_bad_xml`` analogue -- a truncated/corrupted
      XML raises rather than warning-and-continuing, since ``PWxml`` reads the
      whole file through ``xmltodict`` in one go and so has no partial-data
      state to fall back on. That is re-raised below with the filename
      attached, as ``expat``'s own message carries only a line/column.
    """
    espressorun_path = str(espressorun_path)  # convert to string if Path object
    warnings.filterwarnings(
        "ignore", category=UnknownPotcarWarning
    )  # Ignore unknown POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings(
        "ignore", message="No POTCAR file with matching TITEL fields"
    )  # `message` only needs to match start of message
    default_kwargs = {"parse_dos": False}  # ``PWxml`` has no ``exception_on_bad_xml``; see below
    default_kwargs.update(kwargs)
    #TODO: Devise a test for working with projected eigenvalues: Currently untested with doped examples.

    try:
        vasprun = PWxml(find_archived_fname(espressorun_path), **default_kwargs)

        # ``PWxml`` never calls ``Vasprun.__init__``, so these two can be missing entirely rather
        # than ``None`` (pymatgen-io-espresso#27): ``atomic_states`` whenever projections were not
        # requested, and ``kpoints_opt_props`` unconditionally. Both are read unguarded downstream
        # (``projected_eigenvalues`` tests ``atomic_states is None``, so raises instead of returning
        # ``None``; ``as_dict()`` needs ``kpoints_opt_props``), so default them -- but only *where
        # missing*: assigning unconditionally destroyed the projections ``PWxml`` had just parsed
        # into ``atomic_states``.
        for _attr in ("atomic_states", "kpoints_opt_props"):
            if not hasattr(vasprun, _attr):
                setattr(vasprun, _attr, None)

    except ExpatError as exc:
        raise ExpatError(
            f"espresso.xml file at {espressorun_path} is corrupted/incomplete, and could not be "
            f"parsed as XML ({exc}). `pw.x` was likely interrupted before writing the closing tags; "
            f"re-run the calculation, or parse a complete output file."
        ) from exc

    except FileNotFoundError as exc:
        if _archived_exists(espressorun_path):
            # the calculation XML is right there, so this is ``PWxml``'s ``FileGuesser`` failing to
            # find a ``filproj`` (projections were requested), not a missing calculation output.
            # Reporting it as "espresso.xml not found" would name a file that exists, for a failure
            # about a different file entirely:
            raise FileNotFoundError(
                f"No `projwfc.x` `filproj` file could be found for the calculation at "
                f"{espressorun_path} (`PWxml` searched its default `filproj` locations), so orbital "
                f"projections could not be parsed."
            ) from exc

        raise FileNotFoundError(
            f"espresso.xml not found at {espressorun_path}. Needed for parsing calculation output!"
        ) from exc
    return vasprun


def _parse_run_and_poss_projwfc(
    vr_path: PathLike,
    parse_projected_eigen: bool | None = None,
    output_path: PathLike | None = None,
    label: str = "bulk",
    **kwargs,
):
    r"""
    :func:`get_espresso_run` plus orbital projections where available; the espresso
    analogue of ``_parse_vr_and_poss_procar`` for VASP.

    ``pw.x`` writes no orbital projections to its XML, so these come from a
    separate ``projwfc.x`` ``filproj`` output -- the espresso analogue of VASP's
    ``PROCAR``. ``PWxml`` guesses and parses a ``filproj`` itself when
    constructed with ``parse_projected_eigen = True``, but only from a fixed set
    of guessed filenames & directories (raising ``FileNotFoundError`` when none
    matches), so any ``*.projwfc_up`` file in ``output_path`` is additionally
    searched for here and parsed with ``PWxml._parse_projected_eigen``, which
    also handles the paired ``.projwfc_down`` channel of spin-polarised
    (``lsda``) calculations and validates the projections against the XML.

    """
    projwfc = None
    if output_path is None:
        output_path = os.path.dirname(str(vr_path)) or "."

    failed_eig_parsing_warning_message = (
        f"Could not parse projected eigenvalue data for the {label} calculation at {output_path}"
    )

    try:
        vr = get_espresso_run(
            vr_path, parse_projected_eigen=parse_projected_eigen is not False, **kwargs
        )

    except Exception as vr_exc:  # retry without projections
        vr = get_espresso_run(vr_path, parse_projected_eigen=False, **kwargs)
        failed_eig_parsing_warning_message += f", got error:\n{vr_exc}"

    if parse_projected_eigen is not False and getattr(vr, "atomic_states", None) is None:

        filproj_path, multiple = _io_utils._get_output_files_and_check_if_multiple(
            PROJECTIONS_FILE, output_path
        )

        if _archived_exists(filproj_path):
            if multiple:
                _multiple_files_warning(
                    PROJECTIONS_FILE,
                    output_path,
                    filproj_path,
                    action=_file_parsing_action(PROJECTIONS_FILE),
                    dir_type=label,
                )
            try:
                vr.atomic_states = vr._parse_projected_eigen(
                    str(filproj_path).removesuffix(PROJECTIONS_FILE)
                )

            except Exception as projwfc_exc:
                failed_eig_parsing_warning_message += (
                    f"\nThen got the following error when attempting to parse projected eigenvalues "
                    f"from the {label} projwfc.x filproj file ({filproj_path}):\n{projwfc_exc}"
                )

        else:
            failed_eig_parsing_warning_message += (
                f"\nNo projwfc.x filproj file (`*{PROJECTIONS_FILE}`) was found in the {label} folder "
                f"at {output_path} either, from which the projected eigenvalues could otherwise be "
                f"parsed (`pw.x` writes no orbital projections to its XML, so a separate `projwfc.x` "
                f"calculation is required for eigenvalue analysis with espresso)."
            )

    if getattr(vr, "atomic_states", None) is None and projwfc is None and parse_projected_eigen is True:
        warnings.warn(failed_eig_parsing_warning_message)

    return vr, projwfc


def ensure_band_edges(vasprun_obj, occu_tol=1e-8, backend="doped"):
    """
    Ensure that the Vasprun object has VBM, CBM, and band_gap set.

    Note that ``occu_tol`` is only used by the ``"pymatgen"`` backend, and that
    metallic calculations are left with these attributes unset.
    """
    if backend == "pymatgen":
        vasprun_obj.occu_tol = occu_tol
        band_gap, cbm, vbm, _ = vasprun_obj.eigenvalue_band_properties
        vasprun_obj.vbm = vbm
        vasprun_obj.cbm = cbm
        vasprun_obj.band_gap = band_gap

    elif backend == "doped":
        from doped.utils.eigenvalues import band_edge_properties_from_vasprun

        if (
            not hasattr(vasprun_obj, "vbm")
            or vasprun_obj.vbm is None
            or not hasattr(vasprun_obj, "cbm")
            or vasprun_obj.cbm is None
            or not hasattr(vasprun_obj, "band_gap")
            or vasprun_obj.band_gap is None
        ):
            band_edge_prop = band_edge_properties_from_vasprun(vasprun_obj)

            if not band_edge_prop.is_metal:
                vasprun_obj.vbm = band_edge_prop.vbm_info.as_dict()["energy"]
                vasprun_obj.cbm = band_edge_prop.cbm_info.as_dict()["energy"]
                vasprun_obj.band_gap = vasprun_obj.cbm - vasprun_obj.vbm
    else:
        raise ValueError("Use doped or pymatgen for finding band_gap")
    return vasprun_obj


def get_cube(cube_path: PathLike) -> VolumetricData:
    """
    Read a ``pp.x`` cube file as a ``pymatgen`` ``VolumetricData`` object.
    """
    cube_path = str(cube_path)  # convert to string if Path object

    try:
        cube = VolumetricData.from_cube(cube_path)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cube file not found at {cube_path}(.gz/.xz/.bz/.lzma). Needed for calculating the "
            f"Freysoldt (FNV) image charge correction!"
        ) from None
    return cube


def _get_bulk_cube_dict(bulk_path: PathLike, quiet: bool = False, filename: PathLike = ".cube") -> dict:
    """
    ``{axis (str): planar-averaged potential}`` from the bulk ``pp.x`` cube in
    ``bulk_path``, for the Freysoldt (FNV) charge correction.
    """
    bulk_cube_path, multiple = _io_utils._get_output_files_and_check_if_multiple(filename, bulk_path)
    if multiple and not quiet:
        _multiple_files_warning(
            filename,
            bulk_path,
            bulk_cube_path,
            action=_file_parsing_action(filename),
            dir_type="bulk",
        )

    bulk_cube = get_cube(bulk_cube_path)
    return {str(k): bulk_cube.get_average_along_axis(k) for k in [0, 1, 2]}


def _handle_infinite_band_gap(
    band_gap: float | None,
    cbm: float | None,
    calc_path: PathLike | None = None,
) -> tuple[float | None, float | None]:
    """
    Guard against the infinite band gaps / CBMs returned by
    ``eigenvalue_band_properties`` when no empty bands were calculated,
    returning ``(None, None)`` (and warning) in this case.

    ``Vasprun.eigenvalue_band_properties`` (inherited by ``PWxml``) takes the
    CBM as the lowest eigenvalue with an occupancy below ``occu_tol``, leaving
    it at its ``float("inf")`` initial value if there are no unoccupied
    eigenvalues -- giving an infinite band gap (and CBM), which can cause
    downstream failures. This is common with espresso, where the default ``nbnd``
    with ``occupations = 'fixed'`` is exactly ``nelec/2`` (i.e. no empty
    bands), unlike VASP's default ``NBANDS`` which always includes some.

    Args:
        band_gap (float):
            Band gap (in eV), as returned by ``eigenvalue_band_properties``.
        cbm (float):
            CBM eigenvalue (in eV), as returned by
            ``eigenvalue_band_properties``.
        calc_path (PathLike):
            Path to the parsed calculation, only used to identify it in the
            warning message. (Default: None)

    Returns:
        tuple[float | None, float | None]:
            ``(band_gap, cbm)`` unchanged, or ``(None, None)`` if either was
            infinite.
    """
    if band_gap is None or cbm is None or not (np.isinf(band_gap) or np.isinf(cbm)):
        return band_gap, cbm

    calc_info = f" for the calculation in {calc_path}" if calc_path else ""
    warnings.warn(
        f"An infinite band gap and CBM were parsed{calc_info}, which indicates that no empty bands "
        f"were calculated (espresso's default `nbnd` with `occupations = 'fixed'` is `nelec/2`) and can "
        f"cause downstream failures, so these are set to `None`. Add extra bands using the `nbnd` "
        f"parameter for espresso if the band gap of this calculation is needed."
    )
    return None, None


def get_magnetization_from_espressorun(pwxml: PWxml) -> int | float | np.ndarray | None:
    r"""
    Determine the total magnetization of an espresso calculation, in Bohr
    magnetons (i.e. electronic units -- the number of unpaired electrons, as
    used by ``VASP`` and by ``doped``\'s spin-degeneracy analysis); the espresso
    analogue of ``get_magnetization_from_vasprun`` for VASP.

    - ``total``: the scalar magnetization of a collinear (``lsda``)
      calculation.
    - ``total_vec``: the magnetization vector of a non-collinear
      (``noncolin``/``lspinorb``) calculation, returned as a length-3 array
      (the vector norm is taken downstream, by the spin-degeneracy analysis).

    Args:
        pwxml (PWxml):
            ``pymatgen-io-espresso`` ``PWxml`` object for the calculation.

    Returns:
        int, float, np.ndarray or None:
            The total magnetization; a length-3 array for non-collinear
            calculations, ``0`` for non-spin-polarised calculations, or
            ``None`` if it could not be determined.
    """
    magnetization_block = ((getattr(pwxml, "_raw_dict", None) or {}).get("output") or {}).get(
        "magnetization"
    ) or {}

    if parse_pwvals(magnetization_block.get("do_magnetization", "true")):
        for key in ("total_vec", "total"):  # ``total_vec`` (non-collinear) takes precedence
            if (raw_value := magnetization_block.get(key)) is None:
                continue
            with contextlib.suppress(TypeError, ValueError):
                magnetization = parse_pwvals(raw_value)
                if isinstance(magnetization, list | tuple | np.ndarray):
                    return np.array(magnetization, dtype=float)
                return float(magnetization)

    with contextlib.suppress(RuntimeError, TypeError):
        return get_magnetization_from_vasprun(pwxml)

    return None


def get_atomic_site_potentials(volumetric_data_path: PathLike | VolumetricData, beta: float = 0.5):
    """
    Calculate the Gaussian-averaged site potentials, for the Kumagai (eFNV) finite-size charge correction.


    Args:
        volumetric_data_path (PathLike | VolumetricData):
            Path to a ``pp.x`` ``.cube(.gz)`` file or a VASP ``LOCPOT``, or an
            already-parsed ``VolumetricData`` object.
        beta (float):
            Gaussian broadening factor at the atomic sites; **in bohr** for
            ``.cube`` inputs (whose potentials are in Ry, i.e. atomic units)
            and **in Å** for ``LOCPOT`` / ``VolumetricData`` inputs (already
            in eV). Default is 0.5.

    Returns:
        dict:
            ``{"positions": Cartesian site coordinates (Å), "site_potentials":
            potentials at those sites (eV), "atoms": site element symbols}``,
            each ordered as ``VolumetricData.structure.sites``.
    """
    if isinstance(volumetric_data_path, VolumetricData):
        volumetric_data = volumetric_data_path
        is_cube = False
    elif str(volumetric_data_path).endswith((".cube", ".cube.gz")):
        volumetric_data = VolumetricData.from_cube(volumetric_data_path)
        is_cube = True
    else:
        volumetric_data = get_locpot(volumetric_data_path)
        is_cube = False

    nx, ny, nz = volumetric_data.dim
    lattice = volumetric_data.structure.lattice

    reci_latt = lattice.reciprocal_lattice
    # integer Miller indices along each reciprocal axis, FFT-ordered:
    nx_idx = np.roll(np.arange(-nx // 2, nx // 2, 1, dtype=int), int(nx // 2))
    ny_idx = np.roll(np.arange(-ny // 2, ny // 2, 1, dtype=int), int(ny // 2))
    nz_idx = np.roll(np.arange(-nz // 2, nz // 2, 1, dtype=int), int(nz // 2))

    Nx, Ny, Nz = np.meshgrid(nx_idx, ny_idx, nz_idx, indexing="ij")
    # G = n1*b1 + n2*b2 + n3*b3; compute |G|^2 using the reciprocal metric tensor to correctly handle
    # non-orthorhombic cells:
    recip_matrix = reci_latt.matrix  # rows are b1, b2, b3
    metric = recip_matrix @ recip_matrix.T  # G_ij = bi . bj
    g2 = (
            Nx ** 2 * metric[0, 0]
            + Ny ** 2 * metric[1, 1]
            + Nz ** 2 * metric[2, 2]
            + 2 * Nx * Ny * metric[0, 1]
            + 2 * Nx * Nz * metric[0, 2]
            + 2 * Ny * Nz * metric[1, 2]
    )
    if is_cube:  # espresso cube: potential in Ry, beta given in bohr (atomic units)
        pot = volumetric_data.data["total"] * -Ry_to_eV
        beta_angstrom = beta * BOHR_TO_ANGSTROM
    else:  # VASP LOCPOT: potential already in eV, and beta given directly in angstroms
        pot = -volumetric_data.data["total"]
        beta_angstrom = beta

    v_G = np.fft.fftn(pot)
    v_G *= np.exp(-0.5 * (beta_angstrom ** 2) * g2)  # Gaussian broadening in reciprocal space
    v_R = np.real(np.fft.ifftn(v_G))

    v_R_atomic_sites = interpolate_potentials_at_atomic_sites(v_R, volumetric_data)

    sites = volumetric_data.structure.sites
    coords = np.array([site.coords for site in sites])

    efnv_plot_data_dict = {"positions": [], "site_potentials": [], "atoms": []}

    efnv_plot_data_dict["site_potentials"].extend(v_R_atomic_sites)
    efnv_plot_data_dict["positions"].extend(coords)
    efnv_plot_data_dict["atoms"].extend(site.specie.symbol for site in sites)

    return efnv_plot_data_dict




def interpolate_potentials_at_atomic_sites(
    smoothed_potential: np.ndarray,
    volumetric_data: VolumetricData,
):
    """
    Interpolate ``smoothed_potential`` (using the ``volumetric_data`` grid) at the
    atomic sites of its structure.

    Args:
        smoothed_potential (np.ndarray):
            Potential on the ``(nx, ny, nz)`` grid of ``volumetric_data``.
        volumetric_data (VolumetricData):
            Supplies the grid dimensions and the structure whose sites are
            sampled.

    Returns:
        np.ndarray:
            The potential at each site, ordered as
            ``volumetric_data.structure.sites``.
    """
    nx, ny, nz = volumetric_data.dim

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

    # TODO: Will want to revisit this implementation; currently quite slow with cubic interpolation, but
    # dropping to linear significantly worsens accuracy...
    interpolator = RegularGridInterpolator(
        (xpoints_padded, ypoints_padded, zpoints_padded),
        padded,
        method="cubic",  # 'linear' is faster, but 'cubic' is more accurate for interpolation
        bounds_error=True,
    )
    frac_coords = np.mod(volumetric_data.structure.frac_coords, 1.0)

    return interpolator(frac_coords)


def _resolve_calc_output_path(path: PathLike) -> PathLike:
    """
    Resolve ``path`` to an espresso XML output file.

    Args:
        path (PathLike):
            Path to the calculation directory, or directly to the espresso
            ``.xml(.gz)`` output file.

    Returns:
        PathLike: Path to the espresso XML output file.
    """
    if os.path.isfile(path) or not os.path.isdir(path):
        return path

    for pattern in CALC_OUTPUT_MASK:
        xml_path, multiple = _io_utils._get_output_files_and_check_if_multiple(pattern, path)
        if _archived_exists(xml_path):
            if multiple:
                _multiple_files_warning(
                    pattern, path, xml_path, action=_file_parsing_action(pattern), dir_type="calculation"
                )
            return xml_path

    return os.path.join(path, CALC_OUTPUT_MASK[0])


def _archived_exists(path: PathLike) -> bool:
    """
    Whether ``path`` exists in some (possibly archived) form.
    """
    with contextlib.suppress(FileNotFoundError):
        return bool(find_archived_fname(path, raise_error=False))
    return False


def get_calculation_outputs(
    path: PathLike,
    load_planar_averaged_potentials: bool = False,
    load_site_potentials: bool = False,
    parse_projected_eigen: bool | None = None,
    label: str = "calculation",
    beta: float = 0.5,
    **kwargs,
) -> CalculationOutputs:
    """
    Parse the outputs of a Quantum ESPRESSO (``pw.x``) supercell calculation in
    ``path`` to a (calculator-agnostic)
    :class:`~doped.io.outputs.CalculationOutputs` object.

    The espresso ``.xml(.gz)`` file in ``path`` is parsed for energies,
    structures, eigenvalue data and calculation metadata, with optional parsing
    of the ``pp.x`` cube file (``plot_num = 11``) for the planar-averaged
    electrostatic potentials (Freysoldt (FNV) charge corrections) and the
    atomic-site potentials (Kumagai (eFNV) charge corrections).
    The parsed ``PWxml`` object is retained in the (non-serialised) ``CalculationOutputs.raw`` dict.


    Args:
        path (PathLike):
            Path to the espresso calculation directory (containing the
            ``.xml(.gz)`` output file to parse), or directly to the
            ``.xml(.gz)`` file.
        load_planar_averaged_potentials (bool):
            Whether to also parse the planar-averaged electrostatic potentials
            from the ``pp.x`` cube file in ``path``. Default is ``False``.
        load_site_potentials (bool):
            Whether to also parse the atomic-site potentials from the ``pp.x``
            cube file in ``path``. Default is ``False``.
        parse_projected_eigen (bool):
            Whether to parse orbital projections . If ``None`` (default), tries
            to parse projections but with no warning if this fails; if ``True``,
            warns on failure; if ``False``, skips projection parsing.

            Note that ``pw.x`` writes no projections to its XML, so this
            requires a ``projwfc.x`` ``filproj`` file; without one, ``PWxml``
            raises ``FileNotFoundError`` .
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings and parsing efficiency
            choices. Default is ``"calculation"``.
        beta (float):
            Gaussian broadening factor (**in bohr** for espresso cubes) used
            when sampling the atomic-site potentials, only used if
            ``load_site_potentials``. Default is 0.5.
        **kwargs:
            Additional keyword arguments to pass to
            :func:`get_espresso_run`.

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    xml_path = _resolve_calc_output_path(path)
    directory = path if os.path.isdir(path) else os.path.dirname(str(path))

    pwxml, projections = _parse_run_and_poss_projwfc(
        xml_path,
        parse_projected_eigen=parse_projected_eigen,
        output_path=directory,
        label=label,
        **kwargs,
    )

    planar_averaged_potentials = site_potentials = None
    if load_planar_averaged_potentials or load_site_potentials:
        cube_path, multiple = _io_utils._get_output_files_and_check_if_multiple(
            PLANAR_POTENTIALS_FILE, directory
        )
        if multiple:
            _multiple_files_warning(
                PLANAR_POTENTIALS_FILE,
                directory,
                cube_path,
                action=_file_parsing_action(PLANAR_POTENTIALS_FILE),
                dir_type=label,
            )

        if load_planar_averaged_potentials:
            cube = get_cube(cube_path)
            # int axis keys, matching the ``CalculationOutputs.planar_averaged_potentials`` annotation:
            planar_averaged_potentials = {axis: cube.get_average_along_axis(axis) for axis in [0, 1, 2]}
        if load_site_potentials:
            site_potentials = get_atomic_site_potentials(cube_path, beta=beta)["site_potentials"]

    return calculation_outputs_from_pwxml(
        pwxml,
        projections=projections,
        path=directory,
        planar_averaged_potentials=planar_averaged_potentials,
        site_potentials=site_potentials,
    )


def calculation_outputs_from_pwxml(
    pwxml: PWxml,
    projections: PathLike | None = None,
    path: PathLike | None = None,
    planar_averaged_potentials: dict[int, np.ndarray] | None = None,
    site_potentials: list | np.ndarray | None = None,
) -> CalculationOutputs:
    """
    Build a (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs`
    object from an already-parsed ``PWxml`` object.

    The input ``pwxml`` (and ``projections``, if provided) objects are kept in
    the (non-serialised) ``CalculationOutputs.raw`` dict, for reuse without
    re-parsing: the ``PWxml`` under ``"vasprun"`` (the run-object key read by
    calculator-agnostic ``doped`` code) and its espresso-side alias
    ``"pwxml"``, and the projections under ``"procar"``.
    ``raw["computed_entry"]`` is deliberately **not** set, so that
    ``CalculationOutputs.get_computed_entry()`` falls back to building a bare
    ``ComputedStructureEntry(structure, energy)`` -- ``PWxml.get_computed_entry``
    requires a ``generator`` attribute which ``PWxml`` does not provide.


    Args:
        pwxml (PWxml):
            ``pymatgen-io-espresso`` ``PWxml`` object for the calculation.
        projections (PathLike):
            Path to a ``projwfc.x`` file or parsed projections
            object for the calculation, if parsed (stored in ``raw["procar"]``
            for eigenvalue analyses when the ``pwxml`` lacks orbital
            projections). Default is ``None``.
        path (PathLike):
            Directory from which the outputs were parsed, if applicable.
        planar_averaged_potentials (dict[int, np.ndarray]):
            Planar-averaged electrostatic potentials (from the ``pp.x`` cube),
            if already parsed. Default is ``None``.
        site_potentials (list | np.ndarray):
            Atomic-site potentials (from the ``pp.x`` cube), if already parsed.
            Default is ``None``.

    Returns:
        CalculationOutputs: The calculation outputs.
    """
    band_gap, cbm, vbm, _direct = pwxml.eigenvalue_band_properties if pwxml.eigenvalues else (None,) * 4
    band_gap, cbm = _handle_infinite_band_gap(band_gap, cbm, path)

    charge = None
    with contextlib.suppress(AttributeError):
        charge = pwxml.total_charge

    projected_eigenvalues = pwxml.projected_eigenvalues
    if projected_eigenvalues is None and projections is not None:
        projected_eigenvalues = getattr(projections, "data", None)

    spin_params = pwxml.parameters.get("spin", {}) if isinstance(pwxml.parameters, dict) else {}
    noncollinear = bool(spin_params.get("noncolin") or spin_params.get("spinorbit"))

    return CalculationOutputs(
        structure=pwxml.final_structure,
        energy=float(pwxml.final_energy),
        calculator="espresso",
        directory=path,
        converged_electronic=pwxml.converged_electronic,
        converged_ionic=pwxml.converged_ionic,
        efermi=pwxml.efermi,
        eigenvalues=pwxml.eigenvalues,
        projected_eigenvalues=projected_eigenvalues,
        projected_magnetisation=getattr(pwxml, "projected_magnetization", None),
        kpoint_coords=np.array(pwxml.actual_kpoints),
        kpoint_weights=np.array(pwxml.actual_kpoints_weights),
        nelect=getattr(pwxml, "nelec", None),
        charge=charge,
        magnetization=get_magnetization_from_espressorun(pwxml),
        noncollinear=noncollinear,
        vbm=vbm,
        cbm=cbm,
        band_gap=band_gap,
        planar_averaged_potentials=planar_averaged_potentials,
        site_potentials=site_potentials,
        run_metadata={  # espresso namelists + pseudopotential filenames
            "parameters": pwxml.parameters,
            "pseudo_filenames": pwxml.potcar_spec,
            "kpoints": _get_qe_kpoints_grid(pwxml),
            "kpoints_shift": _get_qe_kpoints_shift(pwxml),
            "actual_kpoints": pwxml.actual_kpoints,
        },
        raw={"vasprun": pwxml, "pwxml": pwxml, "procar": projections},
    )


def load_eigenvalue_outputs(
    path: PathLike | None = None,
    outputs: PathLike | PWxml | None = None,
    projections: PathLike | None = None,
    label: str = "bulk",
    run_metadata: dict | None = None,
) -> CalculationOutputs:
    """
    Load espresso calculation outputs `with orbital projections` for eigenvalue
    / band-edge analysis.

    Args:
        path (PathLike):
            Path to the calculation directory (e.g.
            ``DefectEntry.calculation_metadata["bulk_path"]``), or directly to
            the espresso ``.xml(.gz)`` output file, to load outputs from if
            ``outputs`` is not provided / lacks orbital projections.
        outputs (PathLike | PWxml):
            Path to an espresso ``.xml(.gz)`` output file (or its calculation
            directory), or a ``PWxml`` object, if already loaded. Default is
            ``None``.
        projections (PathLike):
            Path to a ``projwfc.x`` ``filproj`` output, or an
            already-parsed projections object, for the orbital projections. If
            ``None`` (default).
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings/errors. Default is
            ``"bulk"``.
        run_metadata (dict):
            The ``DefectEntry.calculation_metadata["run_metadata"]`` dict, to
            re-hydrate the ``PWxml`` from its serialised ``{label}_vasprun_dict``,
            when projections are supplied.

    Returns:
        CalculationOutputs:
            The parsed calculation outputs, with ``projected_eigenvalues``.
    """

    pwxml = outputs if isinstance(outputs, PWxml) else None
    projwfc = directory = None

    for source in (outputs, path):
        if source is None or isinstance(source, PWxml):
            continue
        if getattr(pwxml, "atomic_states", None) is not None:
            break

        with contextlib.suppress(Exception):
            xml_path = _resolve_calc_output_path(source)
            source_dir = source if os.path.isdir(source) else (os.path.dirname(str(xml_path)) or ".")
            source_pwxml, source_projwfc = _parse_run_and_poss_projwfc(
                xml_path, parse_projected_eigen=True, output_path=source_dir, label=label
            )
            if pwxml is None or getattr(source_pwxml, "atomic_states", None) is not None:
                pwxml, projwfc, directory = source_pwxml, source_projwfc, source_dir

    if pwxml is None and projections is not None:
        with contextlib.suppress(Exception):
            pwxml = PWxml.from_dict((run_metadata or {})[f"{label}_vasprun_dict"])

    if not isinstance(pwxml, PWxml):
        raise FileNotFoundError(
            f"No {label} espresso '.xml(.gz)' output file found (and successfully parsed) in path: "
            f"{path}. Required for eigenvalue analysis!"
        )

    if projections is not None and getattr(pwxml, "atomic_states", None) is None:
        if isinstance(projections, str | os.PathLike):
            pwxml.atomic_states = pwxml._parse_projected_eigen(
                str(projections).removesuffix(PROJECTIONS_FILE)
            )
        else:
            projwfc = projections

    if getattr(pwxml, "atomic_states", None) is None and projwfc is None:
        raise FileNotFoundError(
            f"No {label} `projwfc.x` `filproj` file (`*{PROJECTIONS_FILE}`) found (and successfully parsed) "
            f"in path: {path}. Required for eigenvalue analysis! (`pw.x` writes no orbital projections to its XML, so a "
            f"separate `projwfc.x` calculation is required for eigenvalue analysis with espresso)"
        )

    return calculation_outputs_from_pwxml(pwxml, projections=projwfc, path=directory or path)


HARTREE_TO_RY = 2.0
#TODO: Add this bulit into pymatgen.io.espresso

def _compare_qe_input_parameters(
    ref_params: dict,
    entry_params: dict,
    ignore_params: set[str] | None = None,
    ref_name: str = "reference",
    entry_name: str = "entry",
    warn: bool = True,
) -> list | bool:
    """
    Check espresso input parameters that can affect energies are consistent between
    two calculations.  Analogous to ``_compare_incar_tags`` for VASP.

    The ``ref_params`` and ``entry_params`` dicts are the nested parameter
    dicts stored in ``entry.data["qe_input"]`` (i.e. ``PWxml.parameters``).

    Returns ``True`` if no critical mismatches, otherwise a list of
    ``(parameter, value_in_entry, value_in_reference)`` tuples.
    """

    fatal_qe_params = {
        "ecutwfc": None,
        "ecutrho": None,
        "hybrid": False,
        "exx_fraction": 0.25,
        "screening_parameter": 0.106,
        "nqx": None,
        "noncolin": False,
        "spinorbit": False,
        "occupations": "fixed",
        "smearing": "gaussian",
        "degauss": 0.0,
    }
    if ignore_params:
        fatal_qe_params = {k: v for k, v in fatal_qe_params.items() if k not in ignore_params}

    def _flatten(params: dict) -> dict:
        basis = params.get("basis") or {}
        dft = params.get("dft") or {}
        hybrid = dft.get("hybrid") or {}
        spin = params.get("spin") or {}
        bands = params.get("bands") or {}

        smearing = bands.get("smearing")
        smearing_type = smearing.get("#text") if isinstance(smearing, dict) else smearing
        degauss = smearing.get("@degauss") if isinstance(smearing, dict) else None
        ecutwfc, ecutrho = basis.get("ecutwfc"), basis.get("ecutrho")

        flat: dict = {}
        # energies are stored in Ha in the espresso XML; converted to Ry (as set in ``pw.in``) here:
        flat["ecutwfc"] = ecutwfc * HARTREE_TO_RY if ecutwfc is not None else None
        flat["ecutrho"] = ecutrho * HARTREE_TO_RY if ecutrho is not None else None
        flat["degauss"] = degauss * HARTREE_TO_RY if degauss is not None else None
        flat["hybrid"] = bool(hybrid)  
        flat["exx_fraction"] = hybrid.get("exx_fraction")
        flat["screening_parameter"] = hybrid.get("screening_parameter")
        qpoint_grid = hybrid.get("qpoint_grid") or {}
        with contextlib.suppress(KeyError, TypeError, ValueError):
            flat["nqx"] = tuple(int(qpoint_grid[f"@nqx{i}"]) for i in (1, 2, 3)) or None
        flat["noncolin"] = spin.get("noncolin")
        flat["spinorbit"] = spin.get("spinorbit")
        flat["occupations"] = bands.get("occupations")
        flat["smearing"] = smearing_type
        return flat

    def _vals_match(v1, v2) -> bool:
        if v1 is None and v2 is None:
            return True
        if v1 is None or v2 is None:
            return False
        if isinstance(v1, bool) or isinstance(v2, bool):
            return bool(v1) == bool(v2)
        if isinstance(v1, int | float) and isinstance(v2, int | float):
            return bool(np.isclose(v1, v2, rtol=1e-3))
        if isinstance(v1, str):
            return v1.lower() == str(v2).lower()
        return v1 == v2

    ref_flat = _flatten(ref_params or {})
    entry_flat = _flatten(entry_params or {})

    mismatch_list = []
    for key, default in fatal_qe_params.items():
        ref_val = default if ref_flat.get(key) is None else ref_flat[key]
        entry_val = default if entry_flat.get(key) is None else entry_flat[key]
        if not _vals_match(ref_val, entry_val):
            mismatch_list.append((key, entry_val, ref_val))

    if mismatch_list and warn:
        warnings.warn(
            f"There are mismatching espresso input parameters for your {entry_name} and {ref_name} "
            f"calculations which are likely to cause errors in the parsed results (energies). "
            f"Found the following differences:\n"
            f"(in the format: (espresso parameter, value in {entry_name}, value in {ref_name}); "
            f"cutoffs in Ry):"
            f"\n{mismatch_list}\n"
            f"In general, the same espresso input settings should be used in all final calculations "
            f"for parameters which can affect energies!"
        )
    return mismatch_list if mismatch_list else True


def _compare_pseudo_symbols(
    ref_pseudos: list[str],
    entry_pseudos: list[str],
    only_matching_elements: bool = False,
) -> list | bool:
    """
    Check pseudopotential filenames are consistent between two calculations.

    Returns ``True`` if all pseudopotentials match, otherwise a list of
    ``(entry_pseudo, ref_pseudo)`` mismatching pairs.
    """

    def _elem_from_pseudo(fname: str) -> str:
        return os.path.basename(fname).split(".")[0].split("_")[0].capitalize()

    ref_list = list(ref_pseudos or [])
    entry_list = list(entry_pseudos or [])

    if only_matching_elements:
        entry_elements = {_elem_from_pseudo(p) for p in entry_list}
        ref_to_check = [p for p in ref_list if _elem_from_pseudo(p) in entry_elements]
    else:
        ref_to_check = ref_list

    mismatches = []
    for ref_p in ref_to_check:
        if ref_p not in entry_list:
            entry_match = next(
                (p for p in entry_list if _elem_from_pseudo(p) == _elem_from_pseudo(ref_p)),
                None,
            )
            mismatches.append((entry_match, ref_p))

    return mismatches if mismatches else True


def _qe_input_parameters(pwxml_or_parameters) -> dict:
    """
    The espresso input-parameter dict (``PWxml.parameters``) from either a
    ``PWxml`` object or an already-extracted parameters dict (as stored in
    ``CalculationOutputs.run_metadata["parameters"]``, which -- unlike the raw
    ``PWxml`` -- survives serialisation).
    """
    if isinstance(pwxml_or_parameters, dict):
        return pwxml_or_parameters

    return getattr(pwxml_or_parameters, "parameters", None) or {}


def _get_qe_kpoints_grid(pwxml, actual_kpoints=None) -> list[int] | None:
    """
    Get the ``[nk1, nk2, nk3]`` Monkhorst-Pack grid used for a espresso calculation,
    from the ``k_points_IBZ`` input block of its ``.xml`` output.

    This is the espresso analogue of ``Vasprun.kpoints.kpts[0]`` (the `input` grid),
    as opposed to ``PWxml.kpoints_frac`` (the `irreducible` k-points actually
    sampled, which is what espresso reports in its output and is typically much
    longer than 3 entries).

    Γ-only calculations (written by ``doped`` as ``K_POINTS gamma`` for
    molecules-in-a-box) record a single explicit k-point at the origin rather
    than a grid, and are reported as ``[1, 1, 1]``.

    Args:
        pwxml (``PWxml`` or dict):
            Parsed espresso ``.xml`` output, or its already-extracted
            ``PWxml.parameters`` dict (as stored in
            ``CalculationOutputs.run_metadata["parameters"]``).
        actual_kpoints (np.ndarray or list):
            The irreducible k-points of the calculation, used only for the
            Γ-only check below. If ``None`` (default), taken from
            ``PWxml.kpoints_frac`` when ``pwxml`` is a ``PWxml`` object.

    Returns:
        list[int] or None:
            The ``[nk1, nk2, nk3]`` grid, or ``None`` if an explicit k-point
            list was used (for which there is no regular grid).
    """
    k_points_ibz = _qe_input_parameters(pwxml).get("k_points_IBZ") or {}

    if mp_grid := k_points_ibz.get("monkhorst_pack"):
        with contextlib.suppress(KeyError, TypeError, ValueError):
            return [int(mp_grid[f"@nk{i}"]) for i in (1, 2, 3)]

    if k_points_ibz.get("nk") == 1:
        if actual_kpoints is None:
            actual_kpoints = getattr(pwxml, "kpoints_frac", None)
        with contextlib.suppress(Exception):
            if np.allclose(actual_kpoints[0], 0):
                return [1, 1, 1]

    return None


def _get_qe_kpoints_shift(pwxml) -> list[int] | None:
    """
    Get the ``[k1, k2, k3]`` Monkhorst-Pack grid offset (each 0 or 1) used for
    a espresso calculation, from the ``k_points_IBZ`` input block of its
    ``.xml`` output; the espresso analogue of ``Vasprun.kpoints.kpts_shift``.

    As with :func:`_get_qe_kpoints_grid`, ``pwxml`` may be either a ``PWxml``
    object or its already-extracted ``PWxml.parameters`` dict.

    Returns ``None`` if an explicit k-point list was used (see
    :func:`_get_qe_kpoints_grid`).
    """
    mp_grid = (_qe_input_parameters(pwxml).get("k_points_IBZ") or {}).get("monkhorst_pack")
    if mp_grid:
        with contextlib.suppress(KeyError, TypeError, ValueError):
            return [int(mp_grid[f"@k{i}"]) for i in (1, 2, 3)]

    return None


def _compare_qe_kpoints(
    ref_kpoints_grid: list[int] | None,
    entry_kpoints_grid: list[int] | None,
    ref_kpoints_shift: list[int] | None = None,
    entry_kpoints_shift: list[int] | None = None,
    ref_actual_kpoints=None,
    entry_actual_kpoints=None,
    ref_name: str = "bulk",
    entry_name: str = "defect",
    warn: bool = True,
) -> list | bool:
    """
    Check k-point sampling is consistent between two calculations.  Analogous
    to ``_compare_kpoints`` for VASP, but using the input Monkhorst-Pack grid &
    shift from ``PWxml.parameters`` (see :func:`_get_qe_kpoints_grid` /
    :func:`_get_qe_kpoints_shift`) rather than ``PWxml.kpoints``, which is an
    empty placeholder object for espresso -- so the VASP comparator always
    short-circuits to a match and can never detect a QE mismatch.

    The input grids are compared in preference to the irreducible k-points
    (``PWxml.actual_kpoints``, only used when no grid was set, e.g. for
    explicit k-point lists), as a defect breaks symmetry and so the same input
    grid routinely folds to more irreducible k-points in the defect supercell
    than in the bulk (e.g. 8 vs 4 for a 2x2x2 grid).

    Returns ``True`` if the sampling matches, otherwise
    ``[entry_kpoints, ref_kpoints]``.
    """
    if ref_kpoints_grid is not None and entry_kpoints_grid is not None:
        entry_kpoints = [list(entry_kpoints_grid), list(entry_kpoints_shift or [0, 0, 0])]
        ref_kpoints = [list(ref_kpoints_grid), list(ref_kpoints_shift or [0, 0, 0])]
        match = entry_kpoints == ref_kpoints
        formatted = (
            f"(in the format: ([grid], [shift])):"
            f"\n{entry_name}: {entry_kpoints}\n{ref_name}: {ref_kpoints}\n"
        )
    elif ref_actual_kpoints is None or entry_actual_kpoints is None:
        return True

    else:
        entry_kpoints = sorted(np.array(entry_actual_kpoints).tolist())
        ref_kpoints = sorted(np.array(ref_actual_kpoints).tolist())
        match = len(entry_kpoints) == len(ref_kpoints) and np.allclose(entry_kpoints, ref_kpoints)
        formatted = f"\n{entry_name}: {entry_kpoints}\n{ref_name}: {ref_kpoints}\n"

    if match:
        return True

    if warn:
        warnings.warn(
            f"The k-point sampling for your {entry_name} and {ref_name} calculations does not match, "
            f"which is likely to cause errors in the parsed results (energies). Found the following "
            f"differences:\n{formatted}"
            f"In general, the same k-point settings should be used for all final calculations for "
            f"accurate results!"
        )

    return [entry_kpoints, ref_kpoints]


def _format_mismatching_qe_input_warning(
    mismatching_qe_input_warnings: list[tuple[str, set]],
) -> str:
    """
    Generate a formatted warning string for mismatching espresso input parameters.
    """
    mismatching_tags_name_list_dict = {
        tuple(sorted(mismatching_set)): sorted(
            [
                name
                for name, other_mismatching_set in mismatching_qe_input_warnings
                if other_mismatching_set == mismatching_set
            ]
        )
        for mismatching_set in [mismatching for name, mismatching in mismatching_qe_input_warnings]
    }
    return "\n".join(
        f"Entries {names}: {list(tags)}" for tags, names in mismatching_tags_name_list_dict.items()
    )


def _get_pwxml_dict_without_heavy_arrays(pwxml: PWxml) -> dict:
    """
    Get the ``PWxml.as_dict()`` representation, with the (large) projected
    eigenvalue/magnetisation and atomic-state data excluded (as these are not
    needed in later stages of ``doped`` analysis workflows).

    """
    attributes_to_cut = [
        "projected_eigenvalues",
        "projected_magnetisation",
        "kpoints_opt_props",
        "atomic_states",
    ]
    pwxml_dict = pwxml.as_dict()
    output = {k: v for k, v in pwxml_dict.get("output", {}).items() if k not in attributes_to_cut}
    output.update(dict.fromkeys(attributes_to_cut))

    return {**{k: v for k, v in pwxml_dict.items() if k != "output"}, "output": output}


def get_planar_averaged_potentials(
    path: PathLike, dir_type: str = "bulk", quiet: bool = False
) -> dict[str, np.ndarray]:
    """
    Get the planar-averaged electrostatic potentials along each lattice vector
    for the calculation in ``path`` (from the ``pp.x`` cube file with
    espresso), needed for Freysoldt (FNV) finite-size charge corrections.

    Args:
        path (PathLike):
            Path to the calculation directory.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative warnings/errors. Default is
            ``"bulk"``.
        quiet (bool):
            Whether to skip the multiple-files warning if several matching
            files are found. Default is ``False``.

    Returns:
        dict[str, np.ndarray]:
            The planar-averaged potentials, as ``{axis index (str): 1D
            array}``.
    """
    cube_path, multiple = _io_utils._get_output_files_and_check_if_multiple(
        PLANAR_POTENTIALS_FILE, path
    )
    if multiple and not quiet:
        _multiple_files_warning(
            PLANAR_POTENTIALS_FILE,
            path,
            cube_path,
            action="parse the electrostatic potential and compute the FNV charge "
            "correction.",
            dir_type=dir_type,
        )
    cube = get_cube(cube_path)

    return {str(k): cube.get_average_along_axis(k) for k in [0, 1, 2]}


def get_site_potentials(
    path: PathLike,
    dir_type: str = "bulk",
    quiet: bool = False,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Get the atomic-site electrostatic potentials for the calculation in
    ``path``, needed for Kumagai (eFNV) finite-size charge corrections.


    Args:
        path (PathLike):
            Path to the calculation directory.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative warnings/errors. Default is
            ``"bulk"``.
        quiet (bool):
            Whether to skip the multiple-files warning if several matching
            files are found. Default is ``False``.
        beta (float):
            Gaussian broadening factor (**in bohr**) used when sampling the
            site potentials. Default is 0.5. Note that the calculator-agnostic
            parsing code calls this function with the backend-protocol
            arguments only, so uses this default.

    Returns:
        np.ndarray: The atomic-site electrostatic potentials.
    """
    cube_path, multiple = _io_utils._get_output_files_and_check_if_multiple(SITE_POTENTIALS_FILE, path)
    if multiple and not quiet:
        _multiple_files_warning(
            SITE_POTENTIALS_FILE,
            path,
            cube_path,
            action="parse the electrostatic potential and compute the Kumagai (eFNV) charge "
            "correction.",
            dir_type=dir_type,
        )

    return get_atomic_site_potentials(cube_path, beta=beta)["site_potentials"]


def check_run_compatibility(
    defect_outputs: CalculationOutputs | PWxml,
    bulk_outputs: CalculationOutputs | PWxml,
    warn: bool = True,
) -> dict:
    """
    Check the compatibility of the calculation settings of a defect & bulk
    supercell calculation pair (espresso input parameters, pseudopotentials and
    k-point sampling), returning the run metadata and any mismatches for
    storage in ``DefectEntry.calculation_metadata``.

    Args:
        defect_outputs (CalculationOutputs | PWxml):
            Parsed outputs of the defect supercell calculation; either a
            :class:`~doped.io.outputs.CalculationOutputs` object (whose
            ``run_metadata`` supplies the compared settings, and whose
            ``raw["pwxml"]`` is used only for the extras noted below) or a
            ``PWxml`` object directly.
        bulk_outputs (CalculationOutputs | PWxml):
            Parsed outputs of the reference bulk supercell calculation.
        warn (bool):
            Whether to warn about any found mismatches. Default is ``True``.

    Returns:
        dict:
            ``calculation_metadata`` updates: the ``"run_metadata"`` dict
            (espresso input parameters, pseudopotential filenames & k-points,
            plus serialised ``PWxml`` dicts where the parsed ``PWxml`` objects
            are available), and ``"mismatching_QE_input_params"``,
            ``"mismatching_pseudo_filenames"`` & ``"mismatching_kpoints"``
            entries (``False``, or the mismatching values).
    """
    run_metadata: dict[str, Any] = {}
    for label, outputs in [("defect", defect_outputs), ("bulk", bulk_outputs)]:

        is_outputs = isinstance(outputs, CalculationOutputs)
        stored: dict = (outputs.run_metadata or {}) if is_outputs else {}
        pwxml = (outputs.raw or {}).get("pwxml") if is_outputs else outputs

        parameters = stored.get("parameters") or _qe_input_parameters(pwxml)
        actual_kpoints = stored.get("actual_kpoints")
        if actual_kpoints is None:
            actual_kpoints = getattr(pwxml, "actual_kpoints", None)
        kpoints = stored.get("kpoints")
        if not isinstance(kpoints, list | tuple):
            kpoints = _get_qe_kpoints_grid(parameters, actual_kpoints)
        kpoints_shift = stored.get("kpoints_shift")
        if kpoints_shift is None:
            kpoints_shift = _get_qe_kpoints_shift(parameters)

        run_metadata[f"{label}_qe_input"] = parameters
        run_metadata[f"{label}_pseudo_filenames"] = list(
            stored.get("pseudo_filenames") or getattr(pwxml, "potcar_spec", None) or []
        )
        run_metadata[f"{label}_kpoints"] = list(kpoints) if kpoints is not None else None
        run_metadata[f"{label}_kpoints_shift"] = (
            list(kpoints_shift) if kpoints_shift is not None else None
        )
        run_metadata[f"{label}_actual_kpoints"] = actual_kpoints


        if pwxml is not None:
            try:
                run_metadata[f"{label}_vasprun_dict"] = _get_pwxml_dict_without_heavy_arrays(pwxml)

            except Exception as exc:
                calc_dir = outputs.directory if is_outputs else None
                warnings.warn(
                    f"The {label} calculation metadata could not be serialised from the parsed "
                    f"espresso XML" + (f" in {calc_dir}" if calc_dir else "") + f", so "
                    f"`run_metadata['{label}_vasprun_dict']` is missing and any later analysis "
                    f"reading it may fail. Got error:\n  {type(exc).__name__}: {exc}\n"
                )

    qe_input_mismatches = _compare_qe_input_parameters(
        run_metadata["bulk_qe_input"],
        run_metadata["defect_qe_input"],
        ref_name="bulk",
        entry_name="defect",
        warn=warn,
    )
    pseudo_mismatches = _compare_pseudo_symbols(
        run_metadata["bulk_pseudo_filenames"],
        run_metadata["defect_pseudo_filenames"],
        only_matching_elements=True,
    )
    if warn and not isinstance(pseudo_mismatches, bool):
        warnings.warn(
            f"The pseudopotentials for your defect and bulk calculations do not match, which is "
            f"likely to cause errors in the parsed results. Found the following differences:\n"
            f"(in the format: (pseudopotential in defect calculation, pseudopotential in bulk "
            f"calculation)):"
            f"\n{pseudo_mismatches}\n"
            f"The same pseudopotentials should be used for all calculations for accurate results!"
        )
    kpoint_mismatches = _compare_qe_kpoints(
        run_metadata["bulk_kpoints"],
        run_metadata["defect_kpoints"],
        run_metadata["bulk_kpoints_shift"],
        run_metadata["defect_kpoints_shift"],
        run_metadata["bulk_actual_kpoints"],
        run_metadata["defect_actual_kpoints"],
        warn=warn,
    )

    return {
        "mismatching_QE_input_params": (
            qe_input_mismatches if not isinstance(qe_input_mismatches, bool) else False
        ),
        "mismatching_pseudo_filenames": (
            pseudo_mismatches if not isinstance(pseudo_mismatches, bool) else False
        ),
        "mismatching_kpoints": kpoint_mismatches if not isinstance(kpoint_mismatches, bool) else False,
        "run_metadata": run_metadata,
    }


MISMATCH_WARNING_SPECS = {
    "mismatching_QE_input_params": {
        "object_name": "espresso input parameters",
        "per_defect_warning_prefix": "There are mismatching espresso input parameters",
        "transform": set,
        "message": lambda lst: (
            "'Defects: (espresso parameter, value in defect calculation, value in bulk "
            "calculation)'; cutoffs & smearing widths in Ry):\n"
            f"{_format_mismatching_qe_input_warning(lst)}\n"
            "In general, the same espresso input settings should be used in all final calculations "
            "for parameters which can affect energies!"
        ),
    },
    "mismatching_pseudo_filenames": {
        "object_name": "pseudopotentials",
        "per_defect_warning_prefix": "The pseudopotentials",
        "transform": lambda pseudo_mismatches: pseudo_mismatches,
        "message": lambda lst: (
            "(pseudopotential in defect calculation, pseudopotential in bulk calculation)):\n"
            + "\n".join(f"{name}: {mismatching}" for name, mismatching in lst)
            + "\nThe same pseudopotentials should be used for all calculations for accurate results!"
        ),
    },
    "mismatching_kpoints": {
        "object_name": "k-point settings",
        "per_defect_warning_prefix": "The k-point sampling",
        "transform": lambda kpoint_mismatches: kpoint_mismatches,
        "message": lambda lst: (
            "(defect k-points, bulk k-points)):\n"
            + "\n".join(f"{name}: {mismatching}" for name, mismatching in lst)
            + "\nThe same k-point settings should be used for all final calculations for "
            "accurate results!"
        ),
    },
}


