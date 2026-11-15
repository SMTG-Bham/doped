"""
Quantum ESPRESSO (``pw.x`` and pp.x(.cube files)) calculation output parsing
for ``doped``.
"""

import contextlib
import os
import warnings
from typing import Any, Literal  # noqa: F401  (kept for the moved annotations)
from xml.parsers.expat import ExpatError

import numpy as np
from pymatgen.core.units import Ry_to_eV
from pymatgen.electronic_structure.dos import Dos, FermiDos
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.espresso.outputs.pwxml import PWxml
from pymatgen.io.espresso.utils import parse_pwvals
from pymatgen.io.vasp import VolumetricData
from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.util.typing import PathLike
from scipy.ndimage import map_coordinates

from doped.io import utils as _io_utils
from doped.io.espresso.inputs import SUBFOLDER_PRIORITY  # noqa: F401  (re-exported backend constant)
from doped.io.outputs import CalculationOutputs
from doped.io.utils import _multiple_files_warning, find_archived_fname
from doped.io.vasp.outputs import (  # ``LOCPOT``/|Vasprun| helpers, reused for the shared code paths
    get_locpot,
    get_magnetization_from_vasprun,
)

BOHR_TO_ANGSTROM = 0.529177

# fmt: off
PAW_COMPENSATION_RADII: dict[str, float] = {
    "H": 0.2846, "He": 0.4743, "Li": 0.6325, "Be": 0.4743,
    "B": 0.3795, "C": 0.3795, "N": 0.3447, "O": 0.3984,
    "F": 0.3795, "Ne": 0.5692, "Na": 0.7052, "Mg": 0.6166,
    "Al": 0.6483, "Si": 0.6325, "P": 0.5692, "S": 0.4522,
    "Cl": 0.4743, "Ar": 0.5060, "K": 0.6609, "Ca": 0.6957,
    "Sc": 0.7336, "Ti": 0.7273, "V": 0.7084, "Cr": 0.6641,
    "Mn": 0.7242, "Fe": 0.6325, "Co": 0.6008, "Ni": 0.6894,
    "Cu": 0.6325, "Zn": 0.6957, "Ga": 0.6957, "Ge": 0.6008,
    "As": 0.6325, "Se": 0.5060, "Br": 0.6641, "Kr": 0.6957,
    "Rb": 0.7273, "Sr": 0.7495, "Y": 0.7621, "Zr": 0.7495,
    "Nb": 0.7589, "Mo": 0.7178, "Ru": 0.7368, "Rh": 0.7242,
    "Pd": 0.7115, "Ag": 0.7495, "Cd": 0.7147, "In": 0.6799,
    "Sn": 0.7147, "Sb": 0.6894, "Te": 0.6831, "I": 0.6957,
    "Xe": 0.6356, "Cs": 0.6325, "Ba": 0.6957, "La": 0.6594,
    "Ce": 0.6594, "Pr": 0.6594, "Nd": 0.6594, "Pm": 0.6594,
    "Sm": 0.6594, "Eu": 0.6594, "Gd": 0.6594, "Tb": 0.6594,
    "Dy": 0.6594, "Ho": 0.6594, "Er": 0.6594, "Tm": 0.6594,
    "Yb": 0.6594, "Lu": 0.6594, "Hf": 0.7748, "Ta": 0.7874,
    "W": 0.7621, "Re": 0.7906, "Os": 0.7463, "Ir": 0.7558,
    "Pt": 0.7716, "Au": 0.7906, "Hg": 0.7684, "Tl": 0.7115,
    "Pb": 0.7242, "Bi": 0.7242, "Rn": 0.6388,
}
# fmt: on
"""
Gaussian compensation-charge radii (``rc``, in bohr) of the default ``GPAW``
PAW datasets (``gpaw-data`` 1.1.0).

These are used directly as the default per-element Gaussian smearing width when
sampling atomic site potentials from a volumetric potential file, i.e.
``beta = rc`` (see :func:`get_atomic_site_potentials`). Note that it is
``rc / sqrt(2)`` which exactly reproduces ``GPAW``'s own compensation-charge
weighting (its shape function is ``exp(-(r/rc)^2)``, i.e. ``exp(-G^2 rc^2/4)``
in reciprocal space), but the quantity being matched here is the core
electrostatic potential (following the VASP convention, as expected by
``pydefect``'s eFNV implementation), which is a wider average; ``beta = rc``.
"""
_FALLBACK_SMEARING_WIDTH = float(np.median(list(PAW_COMPENSATION_RADII.values())))  # bohr, ~0.66


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
``{file pattern: action description}`` for the files parsed by this backend,
used in the "multiple matching files found" warnings emitted by
``_multiple_files_warning`` (part of the ``doped.io`` backend protocol).

Without an entry for each pattern, these warnings error out rather than warn.
"""


def _make_pwxml_property_settable(prop_name: str) -> None:
    """
    Replace a read-only ``PWxml`` property with a read-write version, storing
    any assigned value in a hidden instance attribute which then short-circuits
    the original getter.

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
    The action description for ``file_type`` (see
    :data:`FILE_PARSING_ACTIONS`), for use in ``_multiple_files_warning``, with
    a generic fallback for any unlisted file type.
    """
    return FILE_PARSING_ACTIONS.get(str(file_type), "Need to parse the calculation outputs.")


def get_espresso_run(espressorun_path: PathLike, **kwargs) -> PWxml:
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
    # TODO: Devise a test for working with projected eigenvalues: Currently untested with doped examples.

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
    :func:`get_espresso_run` plus orbital projections where available; the
    espresso analogue of ``_parse_vr_and_poss_procar`` for VASP.

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
        vr = get_espresso_run(vr_path, parse_projected_eigen=parse_projected_eigen is not False, **kwargs)

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
    # Ry -> eV as in ``get_planar_averaged_potentials``; see the note there:
    return {str(k): bulk_cube.get_average_along_axis(k) * Ry_to_eV for k in [0, 1, 2]}


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
    used by ``VASP`` and by ``doped``\'s spin-degeneracy analysis); the
    espresso analogue of ``get_magnetization_from_vasprun`` for VASP.

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


def get_atomic_site_potentials(
    volumetric_data_path: PathLike | VolumetricData, beta: float | dict[str, float] | None = None
):
    """
    Calculate the Gaussian-averaged site potentials, for the Kumagai (eFNV)
    finite-size charge correction.

    Args:
        volumetric_data_path (PathLike | VolumetricData):
            Path to a ``pp.x`` ``.cube(.gz)`` file or a VASP ``LOCPOT``, or an
            already-parsed ``VolumetricData`` object.
        beta (float | dict[str, float]):
            Gaussian broadening width at the atomic sites, in Å. If ``None``
            (default), a per-element width is used for each site: ``rc``, that
            element's PAW compensation-charge radius
            (:data:`PAW_COMPENSATION_RADII`). Can be set to a single float
            (used for every site) or to a ``{element symbol: width}`` dict,
            with any elements not given taking the default ``rc`` value.
            Elements missing from the table take the median tabulated radius
            (~0.66 bohr = 0.35 Å), with a warning.

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
    # integer Miller indices along each reciprocal axis, in the same order as ``np.fft.fftn`` output
    nx_idx, ny_idx, nz_idx = (np.fft.fftfreq(n, 1 / n) for n in (nx, ny, nz))

    Nx, Ny, Nz = np.meshgrid(nx_idx, ny_idx, nz_idx, indexing="ij")
    # G = n1*b1 + n2*b2 + n3*b3; compute |G|^2 using the reciprocal metric tensor to correctly handle
    # non-orthorhombic cells:
    recip_matrix = reci_latt.matrix  # rows are b1, b2, b3
    metric = recip_matrix @ recip_matrix.T  # G_ij = bi . bj
    g2 = (
        Nx**2 * metric[0, 0]
        + Ny**2 * metric[1, 1]
        + Nz**2 * metric[2, 2]
        + 2 * Nx * Ny * metric[0, 1]
        + 2 * Nx * Nz * metric[0, 2]
        + 2 * Ny * Nz * metric[1, 2]
    )
    # espresso cubes give the potential in Ry; VASP LOCPOTs give it in eV already. ``beta`` is in Å
    # for both, so the broadening below needs no unit handling (``g2`` is in Å^-2):
    pot = volumetric_data.data["total"] * -Ry_to_eV if is_cube else -volumetric_data.data["total"]

    sites = volumetric_data.structure.sites
    atoms = [site.specie.symbol for site in sites]
    betas = _site_smearing_widths(atoms, beta)

    v_G = np.fft.fftn(pot)  # forward transform shared by all widths
    frac_coords = np.mod(volumetric_data.structure.frac_coords, 1.0)
    v_R_atomic_sites = np.empty(len(atoms))
    for width in np.unique(betas):  # one broadening + inverse FFT + interpolation per distinct width
        sampled = betas == width
        v_R = np.real(np.fft.ifftn(v_G * np.exp(-0.5 * width**2 * g2)))  # Gaussian broadening
        v_R_atomic_sites[sampled] = interpolate_potentials_at_atomic_sites(
            v_R, volumetric_data, frac_coords=frac_coords[sampled]
        )

    return {
        "positions": np.array([site.coords for site in sites]),  # (N, 3) array, not 1D-array list
        "site_potentials": v_R_atomic_sites,
        "atoms": atoms,
    }


def _site_smearing_widths(atoms: list[str], beta: float | dict[str, float] | None) -> np.ndarray:
    """
    Gaussian smearing width (in Å) for each site, for sampling atomic-site
    potentials.

    Args:
        atoms (list[str]):
            Element symbol of each site.
        beta (float | dict[str, float]):
            A single width (Å) to use for every site, or a
            ``{element symbol: width (Å)}`` dict setting the width for some
            or all elements. Elements not given -- and every element if
            ``None`` -- take the default ``rc``, from
            :data:`PAW_COMPENSATION_RADII`, or its median (with a warning)
            for elements absent from that table.

    Returns:
        np.ndarray: The smearing width (Å) for each site.
    """
    if beta is not None and not isinstance(beta, dict):
        return np.full(len(atoms), float(beta))

    # note no warning for ``beta`` keys absent from ``atoms``: the same dict is used for the bulk and every
    # defect supercell, so an element missing from `one` of them is normal:
    beta = beta or {}
    defaulted = {atom for atom in atoms if atom not in beta}
    if missing := sorted(defaulted.difference(PAW_COMPENSATION_RADII)):
        warnings.warn(
            f"No PAW compensation-charge radius is tabulated for {missing}, so the median tabulated "
            f"radius ({_FALLBACK_SMEARING_WIDTH:.2f} bohr) is used as the Gaussian smearing width for "
            f"these sites when sampling the atomic site potentials. Set `beta` explicitly to override "
            f"this."
        )

    # ``PAW_COMPENSATION_RADII`` is tabulated in bohr (atomic units, as PAW/pseudopotential datasets give
    # it) while ``beta`` is in Å throughout the public API, so the defaults are converted here:
    widths = []
    for atom in atoms:
        if atom in beta:
            widths.append(beta[atom])
        elif atom in PAW_COMPENSATION_RADII:
            widths.append(PAW_COMPENSATION_RADII[atom] * BOHR_TO_ANGSTROM)
        else:
            widths.append(_FALLBACK_SMEARING_WIDTH * BOHR_TO_ANGSTROM)

    return np.array(widths)


def interpolate_potentials_at_atomic_sites(
    smoothed_potential: np.ndarray,
    volumetric_data: VolumetricData,
    frac_coords: np.ndarray | None = None,
):
    """
    Interpolate ``smoothed_potential`` (using the ``volumetric_data`` grid) at
    the atomic sites of its structure.

    Args:
        smoothed_potential (np.ndarray):
            Potential on the ``(nx, ny, nz)`` grid of ``volumetric_data``.
        volumetric_data (VolumetricData):
            Supplies the grid dimensions and the structure whose sites are
            sampled.
        frac_coords (np.ndarray):
            Fractional coordinates to sample at, if only a subset of the
            sites is wanted (e.g. those sharing one smearing width). Default
            is ``None``; i.e. all sites in ``volumetric_data.structure``.

    Returns:
        np.ndarray:
            The potential at each site, ordered as ``frac_coords`` if given
            and otherwise as ``volumetric_data.structure.sites``.
    """
    if frac_coords is None:
        frac_coords = volumetric_data.structure.frac_coords

    return map_coordinates(  # ``"grid-wrap"`` applies PBC
        smoothed_potential, (frac_coords * volumetric_data.dim).T, order=3, mode="grid-wrap"
    )  # ``map_coordinates`` takes coord in grid-index space (so multiply by dim and transpose)


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
    beta: float | dict[str, float] | None = None,
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
            Gaussian broadening width (in Å) used when sampling the atomic site
            potentials, only used if ``load_site_potentials``. Default is
            ``None``; i.e. per-element widths from
            :data:`PAW_COMPENSATION_RADII`.
        **kwargs:
            Additional keyword arguments to pass to :func:`get_espresso_run`.

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
            # int axis keys, matching the ``CalculationOutputs.planar_averaged_potentials`` annotation;
            # Ry -> eV as in ``get_planar_averaged_potentials``, to match the ``LOCPOT`` convention
            # expected by the FNV correction:
            planar_averaged_potentials = {
                axis: cube.get_average_along_axis(axis) * Ry_to_eV for axis in [0, 1, 2]
            }
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

    nelect = getattr(pwxml, "nelec", None)

    # ``pw.x`` writes the ``k``-point weights (``wk``) summing to 2 for non-spin-polarised
    # calculations, folding the spin degeneracy factor of 2 into the weights
    # themselves, but summing to 1 for non-collinear (``noncolin``) and (per spin channel) collinear
    # spin-polarised (``nspin = 2``) calculations. ``CalculationOutputs.kpoint_weights`` follows the
    # VASP convention of weights summing to 1, so normalising the weighted sum for espresso:
    kpoint_weights = np.array(pwxml.actual_kpoints_weights, dtype=float)
    if (total_weight := kpoint_weights.sum()) > 0:
        kpoint_weights /= total_weight

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
        kpoint_coords=np.array(pwxml.actual_kpoints),
        kpoint_weights=kpoint_weights,
        nelect=nelect,
        charge=charge,
        magnetization=get_magnetization_from_espressorun(pwxml),
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
            f"No {label} `projwfc.x` `filproj` file (`*{PROJECTIONS_FILE}`) found (and successfully "
            f"parsed) in path: {path}. Required for eigenvalue analysis! (`pw.x` writes no orbital "
            f"projections to its XML, so a separate `projwfc.x` calculation is required for eigenvalue "
            f"analysis with espresso)"
        )

    return calculation_outputs_from_pwxml(pwxml, projections=projwfc, path=directory or path)


HARTREE_TO_RY = 2.0
# TODO: Add this built into pymatgen.io.espresso


def _compare_qe_input_parameters(
    ref_params: dict,
    entry_params: dict,
    ignore_params: set[str] | None = None,
    ref_name: str = "reference",
    entry_name: str = "entry",
    warn: bool = True,
) -> list | bool:
    """
    Check espresso input parameters that can affect energies are consistent
    between two calculations.  Analogous to ``_compare_incar_tags`` for VASP.

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
    Get the ``[nk1, nk2, nk3]`` Monkhorst-Pack grid used for a espresso
    calculation, from the ``k_points_IBZ`` input block of its ``.xml`` output.

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
    Generate a formatted warning string for mismatching espresso input
    parameters.
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
    eigenvalue/magnetization and atomic-state data excluded (as these are not
    needed in later stages of ``doped`` analysis workflows).
    """
    attributes_to_cut = [
        "projected_eigenvalues",
        "projected_magnetization",
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
    cube_path, multiple = _io_utils._get_output_files_and_check_if_multiple(PLANAR_POTENTIALS_FILE, path)
    if multiple and not quiet:
        _multiple_files_warning(
            PLANAR_POTENTIALS_FILE,
            path,
            cube_path,
            action="parse the electrostatic potential and compute the FNV charge correction.",
            dir_type=dir_type,
        )
    cube = get_cube(cube_path)

    # ``pp.x`` writes the potential in Ry, and ``VolumetricData.from_cube`` converts only the lattice
    # (bohr -> Å), reading the volumetric data verbatim -- so convert to eV here, to match the ``LOCPOT``
    # convention expected by the FNV correction. The cube sign convention already matches ``LOCPOT``, so
    # no sign flip (unlike ``get_atomic_site_potentials``, which negates both for the eFNV convention):
    return {str(k): cube.get_average_along_axis(k) * Ry_to_eV for k in [0, 1, 2]}


def get_site_potentials(
    path: PathLike,
    dir_type: str = "bulk",
    quiet: bool = False,
    outputs: CalculationOutputs | None = None,
    total_energy: list | float | None = None,
    beta: float | dict[str, float] | None = None,
) -> np.ndarray:
    r"""
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
        outputs (CalculationOutputs):
            Accepted for backend-protocol compatibility (the agnostic parsing
            code passes it) and otherwise **unused** -- see the note below.
            Default is ``None``.
        total_energy (list | float):
            Accepted for backend-protocol compatibility and otherwise
            **unused** -- see the note below. Default is ``None``.
        beta (float):
            Gaussian broadening width (in Å) used when sampling the site
            potentials. Default is ``None``; i.e. per-element widths from
            :data:`PAW_COMPENSATION_RADII`. Can be set through
            :class:`~doped.parsing.DefectsParser` /
            :class:`~doped.parsing.DefectParser`, which forward it here.

    Returns:
        np.ndarray: The atomic-site electrostatic potentials.

    Note:
        With VASP, ``outputs``/``total_energy`` cross-check the total energy
        recorded in the ``OUTCAR`` that supplies the site potentials against
        the parsed calculation, catching inconsistent file combinations. There
        is no espresso analogue: the site potentials come from a ``pp.x``
        ``.cube``, a volumetric file which records no total energy (its header
        is two fixed comment lines). This mirrors VASP's own treatment of its
        volumetric file -- ``get_planar_averaged_potentials`` does not check
        the ``LOCPOT`` either, for exactly the same reason.
    """
    cube_path, multiple = _io_utils._get_output_files_and_check_if_multiple(SITE_POTENTIALS_FILE, path)
    if multiple and not quiet:
        _multiple_files_warning(
            SITE_POTENTIALS_FILE,
            path,
            cube_path,
            action="parse the electrostatic potential and compute the Kumagai (eFNV) charge correction.",
            dir_type=dir_type,
        )

    return get_atomic_site_potentials(cube_path, beta=beta)["site_potentials"]


def get_potentials_from_input(
    potentials_input: PathLike | VolumetricData | dict | list | np.ndarray,
    potential_type: str = "planar",
    dir_type: str = "",
    entry_energy: float | None = None,
    run_metadata: dict | None = None,
    beta: float | dict[str, float] | None = None,
):
    r"""
    Get planar-averaged (``potential_type="planar"``) or atomic-site
    (``"site"``) electrostatic potentials from calculator-native inputs, for
    finite-size charge corrections; the espresso analogue of
    ``doped.io.vasp.outputs.get_potentials_from_input``.

    Accepts a path to a ``pp.x`` ``.cube(.gz)`` file (or to a calculation
    directory containing one), an already-parsed ``VolumetricData`` object, or
    already-parsed potentials (dict/list/array), which are returned as-is.

    Unlike VASP -- where the planar-averaged and atomic-site potentials come
    from separate files (``LOCPOT`` and ``OUTCAR``) -- one ``pp.x`` cube
    (``plot_num = 11``) serves both corrections here, so ``potential_type``
    alone selects how it is reduced.


    Args:
        potentials_input (PathLike | VolumetricData | dict | list | np.ndarray):
            The calculator-native potentials input (see above).
        potential_type (str):
            ``"planar"`` for planar-averaged potentials (Freysoldt/FNV
            correction) or ``"site"`` for atomic-site potentials
            (Kumagai/eFNV correction). Default is ``"planar"``.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative warnings/errors.
        entry_energy (float):
            Accepted for protocol compatibility and **ignored**. With VASP this
            cross-checks a supplied ``OUTCAR``\'s final energy against the
            parsed calculation energy, to catch mismatched calculations; a
            ``pp.x`` cube records no total energy, so there is nothing to
            compare against here.
        run_metadata (dict):
            Accepted for protocol compatibility and **ignored**; see
            ``entry_energy``.
        beta (float):
            Gaussian broadening width in Å, used when sampling the atomic site
            potentials (only used for ``potential_type="site"``). Default is
            ``None``; i.e. per-element widths from
            :data:`PAW_COMPENSATION_RADII`.

    Returns:
        The planar-averaged potentials (``{axis index (str): 1D array}``, in
        eV) or the atomic-site potentials (array, in eV), depending on
        ``potential_type``.
    """
    if isinstance(potentials_input, PathLike):
        if os.path.isdir(potentials_input):  # calculation directory; locate the cube within it
            if potential_type == "planar":
                return get_planar_averaged_potentials(potentials_input, dir_type=dir_type)
            return get_site_potentials(potentials_input, dir_type=dir_type, beta=beta)

        if potential_type == "planar":  # direct path to a ``.cube(.gz)`` file
            cube = get_cube(potentials_input)
            # Ry -> eV, to match the ``LOCPOT`` convention expected by the FNV correction (as in
            # ``get_planar_averaged_potentials``); no sign flip, as the cube sign already matches:
            return {str(k): cube.get_average_along_axis(k) * Ry_to_eV for k in [0, 1, 2]}
        return get_atomic_site_potentials(potentials_input, beta=beta)["site_potentials"]

    if isinstance(potentials_input, VolumetricData):
        # Note that an already-parsed ``VolumetricData`` is taken to be in the ``LOCPOT`` convention
        # (potentials already in eV, ``beta`` in Å), matching ``get_atomic_site_potentials``. A cube
        # parsed by the user with ``VolumetricData.from_cube`` is still in Ry, so should be supplied
        # as a *path* here instead, so that the Ry -> eV conversion is applied:
        if potential_type == "planar":
            return {str(k): potentials_input.get_average_along_axis(k) for k in [0, 1, 2]}
        return get_atomic_site_potentials(potentials_input, beta=beta)["site_potentials"]

    if not isinstance(potentials_input, dict | list | np.ndarray):
        raise TypeError(
            f"`{'planar-averaged' if potential_type == 'planar' else 'atomic-site'} potentials` input "
            f"must be either a path to a `pp.x` `.cube(.gz)` file (or its calculation directory), a "
            f"`pymatgen` `VolumetricData` object, or already-parsed potentials "
            f"({'dict' if potential_type == 'planar' else 'list/array'}), but got "
            f"{type(potentials_input)} instead."
        )

    return potentials_input


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
        run_metadata[f"{label}_kpoints_shift"] = list(kpoints_shift) if kpoints_shift is not None else None
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

# =====================================================================================================
# Quantum ESPRESSO utilities for thermodynamics.py
# (helpers for using Quantum ESPRESSO outputs with the thermodynamic / concentration analyses above)
# =====================================================================================================


def get_fermi_dos_from_espresso_dos(
    dos: PathLike,
    bulk_pwxml: PathLike | None = None,
    structure=None,
    nelecs: float | None = None,
) -> FermiDos:
    r"""
    Create a ``pymatgen`` ``FermiDos`` object from a Quantum ESPRESSO density
    of states (DOS), for use as the ``bulk_dos`` input to
    ``DefectThermodynamics`` (for Fermi level / carrier concentration
    analyses).

    ``doped``\'s ``bulk_dos`` handling natively supports ``pymatgen``
    ``FermiDos``/|Vasprun| inputs, but a Quantum ESPRESSO ``EspressoDos`` (from
    ``pymatgen.io.espresso``) carries no crystal structure, which is needed for
    the volume -> cm^-3 normalisation of carrier concentrations. The structure
    is therefore supplied here -- most conveniently via the bulk supercell
    ``espresso.xml`` (``bulk_pwxml``), i.e. the cell the DOS was computed on.

    The returned ``FermiDos`` can be passed directly as the ``bulk_dos`` input
    to ``DefectThermodynamics``.

    Args:
        dos (PathLike | EspressoDos):
            A ``pymatgen`` ``EspressoDos`` object, or a path to a Quantum
            ESPRESSO ``dos.x`` output file (``fildos``), parsed with
            ``EspressoDos.from_fildos``.
        bulk_pwxml (PathLike | PWxml | None):
            A ``pymatgen`` ``PWxml`` object, or a path to the bulk supercell
            ``espresso.xml`` file (i.e. the cell the DOS was computed on), used
            to obtain the ``structure`` for the volume normalisation and the
            number of valence electrons (``nelec``) for the DOS normalisation.
        structure (Structure | None):
            The structure of the cell the DOS was computed on (usually the bulk
            supercell), used for the volume -> cm^-3 normalisation. Taken from
            ``bulk_pwxml`` if not provided directly.
        nelecs (float | None):
            The number of valence electrons in the cell the DOS was computed on
            (Quantum ESPRESSO's ``nelec``), used to normalise the DOS. Taken
            from ``bulk_pwxml`` if not provided directly. If neither is given,
            ``FermiDos`` falls back to the all-electron count (with a warning),
            which inflates carrier concentrations.

    Returns:
        FermiDos: The ``FermiDos`` object.
    """
    from pymatgen.io.espresso.outputs.dos import EspressoDos
    from pymatgen.io.espresso.outputs.pwxml import PWxml

    if not isinstance(dos, EspressoDos):
        dos = EspressoDos.from_fildos(dos)

    if bulk_pwxml is not None and not isinstance(bulk_pwxml, PWxml):
        bulk_pwxml = PWxml(find_archived_fname(str(bulk_pwxml)))  # resolves ``.xml`` -> ``.xml.gz``

    if structure is None and isinstance(bulk_pwxml, PWxml):
        structure = bulk_pwxml.final_structure

    if structure is None:
        raise ValueError(
            "The bulk structure is required to build a `FermiDos` from a Quantum ESPRESSO DOS (for the "
            "volume -> cm^-3 normalisation), but was not provided. Either supply `bulk_pwxml` (the bulk "
            "supercell `espresso.xml`) or `structure` directly."
        )

    if nelecs is None and isinstance(bulk_pwxml, PWxml):
        nelecs = bulk_pwxml.nelec

    if nelecs is None:
        warnings.warn(
            "The number of valence electrons (`nelec`) could not be determined (no `bulk_pwxml` or "
            "`nelecs` is provided), so the `FermiDos` will normalise the DOS to the all-electron count "
            "from the structure composition. This results in the wrong carrier concentrations. "
            "Please provide `bulk_pwxml` (the bulk supercell `espresso.xml`) or set `nelecs` explicitly "
            "(Quantum ESPRESSO's `nelec`)."
        )

    return FermiDos(
        Dos(efermi=dos.efermi, energies=dos.energies, densities=dos.tdos),
        structure=structure,
        nelecs=nelecs,
    )


DOS_FILE = ".dos"
"""
Filename pattern identifying a ``dos.x`` density-of-states output (``fildos``)
with espresso, for locating one within a calculation directory.
"""


def get_fermi_dos(
    dos_path: PathLike,
    bulk_pwxml: PathLike | PWxml | None = None,
    structure=None,
    nelecs: float | None = None,
) -> tuple[FermiDos, float, float]:
    r"""
    Create a ``pymatgen`` ``FermiDos`` object from the outputs of a bulk
    espresso DOS calculation, for calculating Fermi level positions and
    defect/carrier concentrations.

    Unlike a VASP ``vasprun.xml``, a ``dos.x`` ``fildos`` file carries neither
    the crystal structure (needed for the volume -> cm^-3 normalisation of
    carrier concentrations) nor the valence electron count, and no band
    eigenvalues. These are therefore taken from the espresso ``.xml(.gz)``
    output of the DOS calculation, which is searched for alongside the
    ``fildos`` file when ``bulk_pwxml`` is not given explicitly. If neither is
    available, ``structure``/``nelecs`` must be supplied directly (else
    :func:`get_fermi_dos_from_espresso_dos` raises).

    Args:
        dos_path (PathLike):
            Path to a ``dos.x`` output (``fildos``) file, or to a directory
            containing one (``*.dos``).
        bulk_pwxml (PathLike | PWxml):
            The espresso ``.xml(.gz)`` output of the cell the DOS was computed
            on (or an already-parsed ``PWxml``), supplying the structure,
            ``nelec`` and band edges. If ``None`` (default), an espresso XML
            alongside the ``fildos`` file is used if one is found.
        structure (Structure):
            The structure of the cell the DOS was computed on, if
            ``bulk_pwxml`` is unavailable. Default is ``None``.
        nelecs (float):
            The valence electron count (espresso's ``nelec``) of that cell, if
            ``bulk_pwxml`` is unavailable. Default is ``None``.

    Returns:
        tuple[FermiDos, float, float]:
            The ``FermiDos`` object, along with the VBM eigenvalue and band gap
            (in eV) of the DOS calculation -- used by ``doped`` to check
            consistency with the bulk supercell calculation.
    """
    if os.path.isdir(dos_path):  # locate the ``fildos`` file within the directory
        dos_path, multiple = _io_utils._get_output_files_and_check_if_multiple(DOS_FILE, dos_path)
        if multiple:
            _multiple_files_warning(
                DOS_FILE,
                os.path.dirname(str(dos_path)),
                dos_path,
                action="parse the density of states, for Fermi level & carrier concentration analysis.",
                dir_type="bulk DOS",
            )

    if bulk_pwxml is None:  # look for the DOS calculation's espresso XML alongside the ``fildos``
        xml_path = _resolve_calc_output_path(os.path.dirname(str(dos_path)) or ".")
        if _archived_exists(xml_path):
            bulk_pwxml = xml_path

    if bulk_pwxml is not None and not isinstance(bulk_pwxml, PWxml):
        bulk_pwxml = get_espresso_run(bulk_pwxml, parse_projected_eigen=False)

    fermi_dos = get_fermi_dos_from_espresso_dos(
        dos_path, bulk_pwxml=bulk_pwxml, structure=structure, nelecs=nelecs
    )

    # Band edges from the DOS calculation's own eigenvalues where available, which are more robust
    # than reading them off the (smeared / tetrahedron-broadened) DOS -- whose in-gap tails can make
    # ``get_gap()`` return zero for a real insulator. Fall back to the DOS itself otherwise, using the
    # same settings as ``DefectThermodynamics._parse_fermi_dos`` applies to a ``FermiDos`` input:
    if isinstance(bulk_pwxml, PWxml) and bulk_pwxml.eigenvalues:
        band_gap, cbm, vbm, _ = bulk_pwxml.eigenvalue_band_properties
        band_gap, _cbm = _handle_infinite_band_gap(band_gap, cbm, dos_path)
        if band_gap is not None:
            dos_band_gap = fermi_dos.get_gap(tol=1e-4, abs_tol=True)
            if band_gap - dos_band_gap > 0.1:
                warnings.warn(
                    f"The band gap of this DOS ({dos_band_gap:.2f} eV) is much smaller than that of "
                    f"the eigenvalues of the same calculation ({band_gap:.2f} eV)."
                )
            return fermi_dos, vbm, band_gap

    return (
        fermi_dos,
        fermi_dos.get_cbm_vbm(tol=1e-4, abs_tol=True)[1],
        fermi_dos.get_gap(tol=1e-4, abs_tol=True),
    )


# =====================================================================================================
# Quantum ESPRESSO utilities for chemical_potentials.py
# (competing-phase input generation & output parsing for chemical potential analysis; moved here from
# ``doped/chemical_potentials.py``, with the former ``CompetingPhasesQE``/``CompetingPhasesAnalyzerQE``
# classes converted to module-level functions per the PR #154 ``doped.io`` backend layout)
# =====================================================================================================

# ──────────────────────────────────────────────────────────────────────────
# espresso competing-phase parsing helpers
# ──────────────────────────────────────────────────────────────────────────


def get_competing_phase_entry(path: PathLike, **kwargs) -> ComputedStructureEntry:
    """
    Parse the outputs of a competing phase espresso calculation in ``path`` to
    a ``pymatgen`` ``ComputedStructureEntry``.

    Args:
        path (PathLike):
            Path to the calculation directory, or directly to an espresso
            ``.xml(.gz)`` file.
        **kwargs:
            Additional keyword arguments to pass to :func:`get_espresso_run`.

    Returns:
        ComputedStructureEntry:
            The parsed entry, with ``entry.data`` containing the calculation
            summary info (``"summary"``; band gap & total magnetization),
            settings (``"qe_input"``, ``"kpoints"``, ``"pseudo_filenames"``),
            convergence (``"converged_electronic"``, ``"converged_ionic"``)
            and the folder it was parsed from (``"folder"``).
    """
    # TODO: a directory ``path`` raises ``IsADirectoryError`` here (only a file path works);
    # resolve it with ``_resolve_calc_output_path`` first, as ``get_calculation_outputs`` does
    pwxml = get_espresso_run(path, **kwargs)
    # ``PWxml`` (pymatgen-io-espresso) subclasses ``Vasprun`` but never sets the
    # ``generator`` attribute (espresso's espresso.xml has no ``<generator>`` block). Recent
    # pymatgen versions access ``self.generator["DATE"]`` when building the default
    # ``entry_id`` in the inherited ``Vasprun.get_computed_entry()``, raising
    # ``AttributeError: 'PWxml' object has no attribute 'generator'``. Pass an explicit
    # ``entry_id`` to bypass that branch:
    entry = pwxml.get_computed_entry(entry_id=f"pwxml-{path}")
    unique_symbols = sorted(set(pwxml.atomic_symbols))
    kpoints_grid = _get_qe_kpoints_grid(pwxml)
    summary_dict: dict[str, Any] = {}  # band gap (float | None) and magnetisation (float | ndarray)
    with contextlib.suppress(Exception):  # non-essential properties; can fail with incomplete XML
        band_gap, cbm, *_ = pwxml.eigenvalue_band_properties
        # guard against infinite band gaps (common with band-gap materials in espresso; no empty
        # bands are calculated without setting ``nbnd``), which would otherwise break entry
        # classification and JSON serialisation:
        summary_dict["band_gap"] = _handle_infinite_band_gap(band_gap, cbm, path)[0]
        summary_dict["total_magnetization"] = get_magnetization_from_vasprun(pwxml)

    entry.data.update(
        {
            "formula_pretty": entry.composition.reduced_formula,
            "nsites": len(entry.structure),
            "volume": entry.structure.volume,
            "energy_per_atom": entry.energy_per_atom,
            "elements": unique_symbols,
            "nelements": len(unique_symbols),
            "kpoints": ([kpoints_grid] if kpoints_grid is not None else list(pwxml.kpoints_frac)),
            "qe_input": pwxml.parameters,
            "pseudo_filenames": list(pwxml.potcar_spec),
            "summary": summary_dict,
            "converged_electronic": pwxml.converged_electronic,
            "converged_ionic": pwxml.converged_ionic,
            "folder": os.path.dirname(str(path)),
        }
    )
    return entry


# ──────────────────────────────────────────────────────────────────────────
# competing-phase entry compatibility checks
# ──────────────────────────────────────────────────────────────────────────


def check_entry_compatibility(entries, template_candidates=None) -> None:
    r"""
    Check the compatibility of parsed competing phase entries, by comparing
    their espresso input parameters and pseudopotential filenames against those
    of a reference (template) entry, and warning about any mismatches (which
    can cause errors in the parsed energies, and thus the chemical potential
    limits).

    Mismatches are also recorded in ``entry.data`` under
    ``"mismatching_QE_input_params"`` / ``"mismatching_pseudo_filenames"``,
    matching the corresponding keys set by :func:`check_run_compatibility` for
    defect & bulk supercell calculations. Entries without ``"qe_input"`` /
    ``"pseudo_filenames"`` data (e.g. ``ComputedEntry``\s supplied directly by
    the user, rather than parsed from calculation outputs) are simply skipped
    for the corresponding check.

    Args:
        entries (list[ComputedEntry]):
            The competing phase entries to check.
        template_candidates (list[ComputedEntry]):
            Priority-ordered candidate entries from which to take the
            reference (template) calculation settings; the first with the
            relevant data is used. Default is ``None``, in which case
            ``entries`` is used.
    """
    if template_candidates is None:
        template_candidates = entries

    # espresso input parameter consistency check (analogous to the VASP ``INCAR`` check):
    entries_with_qe_input = [entry for entry in template_candidates if entry.data.get("qe_input")]
    if entries_with_qe_input:
        qe_input_template_entry = entries_with_qe_input[0]
        for entry in entries:
            if not entry.data.get("qe_input"):  # no settings data for this entry (e.g. a user-supplied
                continue  # ``ComputedEntry``); skip rather than compare against nothing
            qe_input_mismatches = _compare_qe_input_parameters(
                qe_input_template_entry.data["qe_input"],
                entry.data["qe_input"],
                ignore_params={"occupations", "smearing", "degauss"},  # set per-phase; see docstring
                warn=False,
            )  # warned collectively below if any mismatches
            entry.data["mismatching_QE_input_params"] = (
                qe_input_mismatches if not isinstance(qe_input_mismatches, bool) else False
            )

        mismatching_qe_input_warnings = sorted(
            [
                (entry.name, set(entry.data.get("mismatching_QE_input_params")))
                for entry in entries
                if entry.data.get("mismatching_QE_input_params")
            ],
            key=lambda x: (len(x[1]), x[0]),
            reverse=True,
        )  # sort by number of mismatches, reversed
        if mismatching_qe_input_warnings:
            warnings.warn(
                f"There are mismatching espresso input parameters for (some of) your competing phases "
                f"calculations which are likely to cause errors in the parsed results (energies & thus "
                f"chemical potential limits). Found the following differences:\n"
                f"(in the format: 'Entries: (espresso parameter, value in entry calculation, value in "
                f"reference calculation))'; cutoffs in Ry:"
                f"\n{_format_mismatching_qe_input_warning(mismatching_qe_input_warnings)}\n"
                f"Where {qe_input_template_entry.name} was used as the reference entry calculation.\n"
                f"In general, the same espresso input settings should be used in all final calculations "
                f"for parameters which can affect energies!"
            )

    # pseudopotential consistency check (analogous to the VASP ``POTCAR`` check):
    entries_with_pseudos = [entry for entry in template_candidates if entry.data.get("pseudo_filenames")]
    if entries_with_pseudos:
        pseudo_template_entry = entries_with_pseudos[0]
        for entry in entries:
            if not entry.data.get("pseudo_filenames"):  # no settings data for this entry; skip
                continue
            pseudo_mismatches = _compare_pseudo_symbols(
                pseudo_template_entry.data["pseudo_filenames"],
                entry.data["pseudo_filenames"],
                only_matching_elements=True,
            )  # warned collectively below if any mismatches
            entry.data["mismatching_pseudo_filenames"] = (
                pseudo_mismatches if not isinstance(pseudo_mismatches, bool) else False
            )

        mismatching_pseudo_warnings = sorted(
            [
                (entry.name, entry.data.get("mismatching_pseudo_filenames"))
                for entry in entries
                if entry.data.get("mismatching_pseudo_filenames")
            ],
            key=lambda x: (len(x[1]), x[0]),
            reverse=True,
        )  # sort by number of mismatches, reversed
        if mismatching_pseudo_warnings:
            joined_info_string = "\n".join(
                [f"{name}: {mismatching}" for name, mismatching in mismatching_pseudo_warnings]
            )
            warnings.warn(
                f"There are mismatching pseudopotential filenames for (some of) your competing phases "
                f"calculations which are likely to cause errors in the parsed results (energies & thus "
                f"chemical potential limits). Found the following differences:\n"
                f"(in the format: (entry pseudopotential, reference pseudopotential)):"
                f"\n{joined_info_string}\n"
                f"Where {pseudo_template_entry.name} was used as the reference entry calculation.\n"
                f"The same pseudopotentials should be used in all final calculations!"
            )
