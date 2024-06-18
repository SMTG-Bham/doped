"""
Code to analyse VASP defect calculations.

These functions are built from a combination of useful modules from ``pymatgen``,
alongside substantial modification, in the efforts of making an efficient,
user-friendly package for managing and analysing defect calculations, with
publication-quality outputs.
"""

import contextlib
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from filelock import FileLock
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.sites import PeriodicSite
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Procar, Vasprun
from tqdm import tqdm

from doped import _doped_obj_properties_methods, _ignore_pmg_warnings
from doped.core import DefectEntry, guess_and_set_oxi_states_with_timeout
from doped.generation import get_defect_name_from_defect, get_defect_name_from_entry, name_defect_entries
from doped.thermodynamics import DefectThermodynamics
from doped.utils.parsing import (
    _compare_incar_tags,
    _compare_kpoints,
    _compare_potcar_symbols,
    _defect_spin_degeneracy_from_vasprun,
    _get_bulk_locpot_dict,
    _get_bulk_site_potentials,
    _get_defect_supercell_bulk_site_coords,
    _get_output_files_and_check_if_multiple,
    _multiple_files_warning,
    _vasp_file_parsing_action_dict,
    check_atom_mapping_far_from_defect,
    defect_charge_from_vasprun,
    get_defect_site_idxs_and_unrelaxed_structure,
    get_defect_type_and_composition_diff,
    get_locpot,
    get_orientational_degeneracy,
    get_outcar,
    get_procar,
    get_vasprun,
)
from doped.utils.plotting import format_defect_name
from doped.utils.symmetry import (
    _frac_coords_sort_func,
    _get_all_equiv_sites,
    _get_sga,
    point_symmetry_from_defect_entry,
)

if TYPE_CHECKING:
    from easyunfold.procar import Procar as EasyunfoldProcar


def _custom_formatwarning(
    message: Union[Warning, str],
    category: type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    """
    Reformat warnings to just print the warning message.
    """
    return f"{message}\n"


warnings.formatwarning = _custom_formatwarning

_ANGSTROM = "\u212B"  # unicode symbol for angstrom to print in strings
_ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings

_aniso_dielectric_but_outcar_problem_warning = (
    "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to compute the "
    "_anisotropic_ Kumagai eFNV charge correction) "
)
# Neither new nor old pymatgen FNV correction can do anisotropic dielectrics (while new sxdefectalign can)
_aniso_dielectric_but_using_locpot_warning = (
    "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) charge "
    "correction developed for _isotropic_ materials will be applied here, which corresponds to using the "
    "effective isotropic average of the supplied anisotropic dielectric. This could lead to significant "
    "errors for very anisotropic systems and/or relatively small supercells!"
)


def _convert_dielectric_to_tensor(dielectric):
    # check if dielectric in required 3x3 matrix format
    if not isinstance(dielectric, (float, int)):
        dielectric = np.array(dielectric)
        if dielectric.shape == (3,):
            dielectric = np.diag(dielectric)
        elif dielectric.shape != (3, 3):
            raise ValueError(
                f"Dielectric constant must be a float/int or a 3x1 matrix or 3x3 matrix, "
                f"got type {type(dielectric)} and shape {dielectric.shape}"
            )

    else:
        dielectric = np.eye(3) * dielectric

    return dielectric


def check_and_set_defect_entry_name(
    defect_entry: DefectEntry, possible_defect_name: str = "", bulk_symm_ops: Optional[list] = None
) -> None:
    """
    Check that ``possible_defect_name`` is a recognised format by doped (i.e.
    in the format "{defect_name}_{optional_site_info}_{charge_state}").

    If the DefectEntry.name attribute is not defined or does not end with the
    charge state, then the entry will be renamed with the doped default name
    for the `unrelaxed` defect (i.e. using the point symmetry of the defect
    site in the bulk cell).

    Args:
        defect_entry (DefectEntry): DefectEntry object.
        possible_defect_name (str):
            Possible defect name (usually the folder name) to check if
            recognised by ``doped``, otherwise defect name is re-determined.
        bulk_symm_ops (list):
            List of symmetry operations of the defect_entry.bulk_supercell
            structure (used in determining the `unrelaxed` point symmetry), to
            avoid re-calculating. Default is None (recalculates).
    """
    formatted_defect_name = None
    charge_state = defect_entry.charge_state
    # check if defect folder name ends with charge state:
    defect_name_w_charge_state = (
        possible_defect_name
        if (possible_defect_name.endswith((f"_{charge_state}", f"_{charge_state:+}")))
        else f"{possible_defect_name}_{'+' if charge_state > 0 else ''}{charge_state}"
    )

    with contextlib.suppress(Exception):  # check if defect name is recognised
        formatted_defect_name = format_defect_name(
            defect_name_w_charge_state, include_site_info_in_name=True
        )  # tries without site_info if with site_info fails

    # (re-)determine doped defect name and store in metadata, regardless of whether folder name is
    # recognised:
    if "full_unrelaxed_defect_name" not in defect_entry.calculation_metadata:
        defect_entry.calculation_metadata["full_unrelaxed_defect_name"] = (
            f"{get_defect_name_from_entry(defect_entry, symm_ops=bulk_symm_ops, relaxed=False)}_"
            f"{'+' if charge_state > 0 else ''}{charge_state}"
        )

    if formatted_defect_name is not None:
        defect_entry.name = defect_name_w_charge_state
    else:  # otherwise use default doped name
        defect_entry.name = defect_entry.calculation_metadata["full_unrelaxed_defect_name"]


def defect_from_structures(
    bulk_supercell, defect_supercell, return_all_info=False, bulk_voronoi_node_dict=None, oxi_state=None
):
    """
    Auto-determines the defect type and defect site from the supplied bulk and
    defect structures, and returns a corresponding ``Defect`` object.

    If ``return_all_info`` is set to true, then also returns:

    - `relaxed` defect site in the defect supercell
    - the defect site in the bulk supercell
    - defect site index in the defect supercell
    - bulk site index (index of defect site in bulk supercell)
    - guessed initial defect structure (before relaxation)
    - 'unrelaxed defect structure' (also before relaxation, but with interstitials at their
      final `relaxed` positions, and all bulk atoms at their unrelaxed positions).

    Args:
        bulk_supercell (Structure):
            Bulk supercell structure.
        defect_supercell (Structure):
            Defect structure to use for identifying the defect site and type.
        return_all_info (bool):
            If True, returns additional python objects related to the
            site-matching, listed above. (Default = False)
        bulk_voronoi_node_dict (dict):
            Dictionary of bulk supercell Voronoi node information, for
            expedited site-matching. If None, will be re-calculated.
        oxi_state (int, float, str):
            Oxidation state of the defect site. If not provided, will be
            automatically determined from the defect structure.

    Returns:
        defect (Defect):
            doped Defect object.

        If ``return_all_info`` is True, then also:

        defect_site (Site):
            pymatgen Site object of the `relaxed` defect site in the defect supercell.
        defect_site_in_bulk (Site):
            pymatgen Site object of the defect site in the bulk supercell
            (i.e. unrelaxed vacancy/substitution site, or final `relaxed` interstitial
            site for interstitials).
        defect_site_index (int):
            index of defect site in defect supercell (None for vacancies)
        bulk_site_index (int):
            index of defect site in bulk supercell (None for interstitials)
        guessed_initial_defect_structure (Structure):
            pymatgen Structure object of the guessed initial defect structure.
        unrelaxed_defect_structure (Structure):
            pymatgen Structure object of the unrelaxed defect structure.
        bulk_voronoi_node_dict (dict):
            Dictionary of bulk supercell Voronoi node information, for
            further expedited site-matching.
    """
    warnings.filterwarnings("ignore", "dict interface")  # ignore spglib warning from v2.4.1
    try:
        def_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, defect_supercell)
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_supercell)} in bulk vs. {len(defect_supercell)} in defect?"
        ) from exc

    # Try automatic defect site detection - this gives us the "unrelaxed" defect structure
    try:
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_supercell, defect_supercell, def_type, comp_diff
        )

    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not identify {def_type} defect site in defect structure. Try supplying the initial "
            f"defect structure to DefectParser.from_paths()."
        ) from exc

    if def_type == "vacancy":
        defect_site_in_bulk = defect_site = bulk_supercell[bulk_site_idx]
    elif def_type == "substitution":
        defect_site = defect_supercell[defect_site_idx]
        site_in_bulk = bulk_supercell[bulk_site_idx]  # this is with orig (substituted) element
        defect_site_in_bulk = PeriodicSite(
            defect_site.species, site_in_bulk.frac_coords, site_in_bulk.lattice
        )
    else:
        defect_site_in_bulk = defect_site = defect_supercell[defect_site_idx]

    check_atom_mapping_far_from_defect(bulk_supercell, defect_supercell, defect_site_in_bulk.frac_coords)

    if unrelaxed_defect_structure:
        if def_type == "interstitial":
            # get closest Voronoi site in bulk supercell to final interstitial site as this is likely
            # the _initial_ interstitial site
            try:
                if bulk_voronoi_node_dict is not None and not StructureMatcher(
                    stol=0.05,
                    primitive_cell=False,
                    scale=False,
                    attempt_supercell=False,
                    allow_subset=False,
                    comparator=ElementComparator(),
                ).fit(bulk_voronoi_node_dict["bulk_supercell"], bulk_supercell):
                    warnings.warn(
                        "Previous bulk voronoi_nodes.json detected, but does not match current bulk "
                        "supercell. Recalculating Voronoi nodes."
                    )
                    raise FileNotFoundError

                voronoi_frac_coords = bulk_voronoi_node_dict["Voronoi nodes"]

            except Exception:  # first time parsing
                from shakenbreak.input import _get_voronoi_nodes

                voronoi_frac_coords = [site.frac_coords for site in _get_voronoi_nodes(bulk_supercell)]
                bulk_voronoi_node_dict = {
                    "bulk_supercell": bulk_supercell,
                    "Voronoi nodes": voronoi_frac_coords,
                }

            closest_node_frac_coords = min(
                voronoi_frac_coords,
                key=lambda node: defect_site.distance_and_image_from_frac_coords(node)[0],
            )
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()
            int_site = guessed_initial_defect_structure[defect_site_idx]
            guessed_initial_defect_structure.remove_sites([defect_site_idx])
            guessed_initial_defect_structure.insert(
                defect_site_idx,  # Place defect at same position as in DFT calculation
                int_site.species_string,
                closest_node_frac_coords,
                coords_are_cartesian=False,
                validate_proximity=True,
            )

        else:
            guessed_initial_defect_structure = unrelaxed_defect_structure.copy()

    else:
        warnings.warn(
            "Cannot determine the unrelaxed `initial_defect_structure`. Please ensure the "
            "`initial_defect_structure` is indeed unrelaxed."
        )

    for_monty_defect = {  # initialise doped Defect object, needs to use defect site in bulk (which for
        # substitutions differs from defect_site)
        "@module": "doped.core",
        "@class": def_type.capitalize(),
        "structure": bulk_supercell,
        "site": defect_site_in_bulk,
        "oxi_state": oxi_state,
    }  # note that we now define the Defect in the bulk supercell, rather than the primitive structure
    # as done during generation. Future work could try mapping the relaxed defect site back to the
    # primitive cell, however interstitials will be very tricky for this...
    if def_type == "interstitial":
        for_monty_defect["multiplicity"] = 1  # multiplicity needed for interstitial initialisation with
        # pymatgen-analysis-defects, so set to 1 here. Set later for interstitials during parsing anyway
        # (see below)
    defect = MontyDecoder().process_decoded(for_monty_defect)

    if not return_all_info:
        return defect

    return (
        defect,
        defect_site,
        defect_site_in_bulk,
        defect_site_idx,
        bulk_site_idx,
        guessed_initial_defect_structure,
        unrelaxed_defect_structure,
        bulk_voronoi_node_dict,
    )


def defect_name_from_structures(bulk_structure, defect_structure):
    """
    Get the doped/SnB defect name using the bulk and defect structures.

    Args:
        bulk_structure (Structure):
            Bulk (pristine) structure.
        defect_structure (Structure):
            Defect structure.

    Returns:
        str: Defect name.
    """
    # set oxi_state to avoid wasting time trying to auto-determine when unnecessary here
    defect = defect_from_structures(bulk_structure, defect_structure, oxi_state="Undetermined")

    # note that if the symm_op approach fails for any reason here, the defect-supercell expansion
    # approach will only be valid if the defect structure is a diagonal expansion of the primitive...

    return get_defect_name_from_defect(defect)


def defect_entry_from_paths(
    defect_path: str,
    bulk_path: str,
    dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
    charge_state: Optional[int] = None,
    initial_defect_structure_path: Optional[str] = None,
    skip_corrections: bool = False,
    error_tolerance: float = 0.05,
    bulk_band_gap_vr: Optional[Union[str, Vasprun]] = None,
    **kwargs,
):
    """
    Parse the defect calculation outputs in ``defect_path`` and return the
    parsed ``DefectEntry`` object.

    By default, the ``DefectEntry.name`` attribute (later used to label the
    defects in plots) is set to the defect_path folder name (if it is a
    recognised defect name), else it is set to the default ``doped`` name for
    that defect (using the estimated `unrelaxed` defect structure, for the point
    group and neighbour distances).

    Note that the bulk and defect supercells should have the same definitions/basis
    sets (for site-matching and finite-size charge corrections to work appropriately).

    Args:
        defect_path (str):
            Path to defect supercell folder (containing at least vasprun.xml(.gz)).
        bulk_path (str):
            Path to bulk supercell folder (containing at least vasprun.xml(.gz)).
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Ionic + static contributions to the dielectric constant, in the same xyz
            Cartesian basis as the supercell calculations. If not provided, charge
            corrections cannot be computed and so ``skip_corrections`` will be set to
            true.
        charge_state (int):
            Charge state of defect. If not provided, will be automatically determined
            from the defect calculation outputs.
        initial_defect_structure_path (str):
            Path to the initial/unrelaxed defect structure. Only recommended for use
            if structure matching with the relaxed defect structure(s) fails (rare).
            Default is None.
        skip_corrections (bool):
            Whether to skip the calculation and application of finite-size charge
            corrections to the defect energy (not recommended in most cases).
            Default = False.
        error_tolerance (float):
            If the estimated error in the defect charge correction, based on the
            variance of the potential in the sampling region is greater than this
            value (in eV), then a warning is raised. (default: 0.05 eV)
        bulk_band_gap_vr (str or Vasprun):
            Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` ``Vasprun``
            object, from which to determine the bulk band gap and band edge positions.
            If the VBM/CBM occur at `k`-points which are not included in the bulk
            supercell calculation, then this parameter should be used to provide the
            output of a bulk bandstructure calculation so that these are correctly
            determined.
            Alternatively, you can edit/add the ``"gap"`` and ``"vbm"`` entries in
            ``self.defect_entry.calculation_metadata`` to match the correct
            (eigen)values.
            If None, will use ``DefectEntry.calculation_metadata["bulk_path"]`` (i.e.
            the bulk supercell calculation output).

            Note that the ``"gap"`` and ``"vbm"`` values should only affect the
            reference for the Fermi level values output by ``doped`` (as this VBM
            eigenvalue is used as the zero reference), thus affecting the position of
            the band edges in the defect formation energy plots and doping window /
            dopability limit functions, and the reference of the reported Fermi levels.
        **kwargs:
            Keyword arguments to pass to ``DefectParser()`` methods
            (``load_FNV_data()``, ``load_eFNV_data()``, ``load_bulk_gap_data()``)
            ``point_symmetry_from_defect_entry()`` or ``defect_from_structures``,
            including ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
            ``mpid``, ``api_key``, ``symprec`` or ``oxi_state``.

    Return:
        Parsed ``DefectEntry`` object.
    """
    dp = DefectParser.from_paths(
        defect_path,
        bulk_path,
        dielectric=dielectric,
        charge_state=charge_state,
        initial_defect_structure_path=initial_defect_structure_path,
        skip_corrections=skip_corrections,
        error_tolerance=error_tolerance,
        bulk_band_gap_vr=bulk_band_gap_vr,
        **kwargs,
    )
    return dp.defect_entry


class DefectsParser:
    def __init__(
        self,
        output_path: str = ".",
        dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
        subfolder: Optional[str] = None,
        bulk_path: Optional[str] = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        bulk_band_gap_vr: Optional[Union[str, Vasprun]] = None,
        processes: Optional[int] = None,
        json_filename: Optional[Union[str, bool]] = None,
        parse_projected_eigen: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        A class for rapidly parsing multiple VASP defect supercell calculations
        for a given host (bulk) material.

        Loops over calculation directories in ``output_path`` (likely the same
        ``output_path`` used with ``DefectsSet`` for file generation in
        ``doped.vasp``) and parses the defect calculations into a dictionary of:
        ``{defect_name: DefectEntry}``, where the ``defect_name`` is set to the
        defect calculation folder name (`if it is a recognised defect name`),
        else it is set to the default ``doped`` name for that defect (using the
        estimated `unrelaxed` defect structure, for the point group and neighbour
        distances). By default, searches for folders in ``output_path`` with
        ``subfolder`` containing ``vasprun.xml(.gz)`` files, and tries to parse
        them as ``DefectEntry``\s.

        By default, tries multiprocessing to speed up defect parsing, which can be
        controlled with ``processes``. If parsing hangs, this may be due to memory
        issues, in which case you should reduce ``processes`` (e.g. 4 or less).

        Defect charge states are automatically determined from the defect
        calculation outputs if ``POTCAR``\s are set up with ``pymatgen`` (see docs
        Installation page), or if that fails, using the defect folder name (must
        end in "_+X" or "_-X" where +/-X is the defect charge state).

        Uses the (single) ``DefectParser`` class to parse the individual defect
        calculations. Note that the bulk and defect supercells should have the
        same definitions/basis sets (for site-matching and finite-size charge
        corrections to work appropriately).

        Args:
            output_path (str):
                Path to the output directory containing the defect calculation
                folders (likely the same ``output_path`` used with ``DefectsSet``
                for file generation in ``doped.vasp``). Default = current directory.
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Ionic + static contributions to the dielectric constant, in the same
                xyz Cartesian basis as the supercell calculations. If not provided,
                charge corrections cannot be computed and so ``skip_corrections``
                will be set to ``True``.
            subfolder (str):
                Name of subfolder(s) within each defect calculation folder (in the
                ``output_path`` directory) containing the VASP calculation files to
                parse (e.g. ``vasp_ncl``, ``vasp_std``, ``vasp_gam`` etc.). If not
                specified, ``doped`` checks first for ``vasp_ncl``, ``vasp_std``,
                ``vasp_gam`` subfolders with calculation outputs
                (``vasprun.xml(.gz)`` files) and uses the highest level VASP type
                (ncl > std > gam) found as ``subfolder``, otherwise uses the defect
                calculation folder itself with no subfolder (set
                ``subfolder = "."`` to enforce this).
            bulk_path (str):
                Path to bulk supercell reference calculation folder. If not
                specified, searches for folder with name "X_bulk" in the
                ``output_path`` directory (matching the default ``doped`` name for
                the bulk supercell reference folder).
            skip_corrections (bool):
                Whether to skip the calculation & application of finite-size charge
                corrections to the defect energies (not recommended in most cases).
                Default = False.
            error_tolerance (float):
                If the estimated error in any charge correction, based on the
                variance of the potential in the sampling region, is greater than
                this value (in eV), then a warning is raised. (default: 0.05 eV)
            bulk_band_gap_vr (str or Vasprun):
                Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` ``Vasprun``
                object, from which to determine the bulk band gap and band edge
                positions. If the VBM/CBM occur at `k`-points which are not included
                in the bulk supercell calculation, then this parameter should be used
                to provide the output of a bulk bandstructure calculation so that
                these are correctly determined.
                Alternatively, you can edit/add the ``"gap"`` and ``"vbm"`` entries in
                ``self.defect_entry.calculation_metadata`` to match the correct
                (eigen)values.
                If None, will use ``DefectEntry.calculation_metadata["bulk_path"]``
                (i.e. the bulk supercell calculation output).

                Note that the ``"gap"`` and ``"vbm"`` values should only affect the
                reference for the Fermi level values output by ``doped`` (as this VBM
                eigenvalue is used as the zero reference), thus affecting the position
                of the band edges in the defect formation energy plots and doping
                window / dopability limit functions, and the reference of the reported
                Fermi levels.
            processes (int):
                Number of processes to use for multiprocessing for expedited parsing.
                If not set, defaults to one less than the number of CPUs available.
            json_filename (str):
                Filename to save the parsed defect entries dict
                (``DefectsParser.defect_dict``) to in ``output_path``, to avoid
                having to re-parse defects when later analysing further and aiding
                calculation provenance. Can be reloaded using the ``loadfn`` function
                from ``monty.serialization`` (and then input to ``DefectThermodynamics``
                etc.). If ``None`` (default), set as
                ``{Host Chemical Formula}_defect_dict.json``.
                If ``False``, no json file is saved.
            parse_projected_eigen (bool):
                Whether to parse the projected eigenvalues & orbitals from the bulk and
                defect calculations (so ``DefectEntry.get_eigenvalue_analysis()`` can
                then be used with no further parsing). Will initially try to load orbital
                projections from ``vasprun.xml(.gz)`` files (slightly slower but more
                accurate), or failing that from ``PROCAR(.gz)`` files if present in the
                bulk/defect directories. Parsing this data can increase total parsing time
                by anywhere from ~5-25%, so set to ``False`` if parsing speed is crucial.
                Default is ``None``, which will attempt to load this data but with no
                warning if it fails (otherwise if ``True`` a warning will be printed).
            **kwargs:
                Keyword arguments to pass to ``DefectParser()`` methods
                (``load_FNV_data()``, ``load_eFNV_data()``, ``load_bulk_gap_data()``)
                ``point_symmetry_from_defect_entry()`` or ``defect_from_structures``,
                including ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
                ``mpid``, ``api_key``, ``symprec`` or ``oxi_state``. Primarily used by
                ``DefectsParser`` to expedite parsing by avoiding reloading bulk data
                for each defect.

        Attributes:
            defect_dict (dict):
                Dictionary of parsed defect calculations in the format:
                ``{"defect_name": DefectEntry}`` where the defect_name is set to the
                defect calculation folder name (`if it is a recognised defect name`),
                else it is set to the default ``doped`` name for that defect (using
                the estimated `unrelaxed` defect structure, for the point group and
                neighbour distances).
        """
        self.output_path = output_path
        self.dielectric = dielectric
        self.skip_corrections = skip_corrections
        self.error_tolerance = error_tolerance
        self.bulk_path = bulk_path
        self.subfolder = subfolder
        self.bulk_band_gap_vr = bulk_band_gap_vr
        self.processes = processes
        self.json_filename = json_filename
        self.parse_projected_eigen = parse_projected_eigen
        self.bulk_vr = None  # loaded later
        self.kwargs = kwargs

        possible_defect_folders = [
            dir
            for dir in os.listdir(self.output_path)
            if any(
                "vasprun" in file and ".xml" in file
                for file_list in [tup[2] for tup in os.walk(os.path.join(self.output_path, dir))]
                for file in file_list
            )
            and dir not in (self.bulk_path.split("/") if self.bulk_path else [])
        ]

        if not possible_defect_folders:  # user may have specified the defect folder directly, so check
            # if we can dynamically determine the defect folder:
            possible_defect_folders = [
                dir
                for dir in os.listdir(os.path.join(self.output_path, os.pardir))
                if any(
                    "vasprun" in file and ".xml" in file
                    for file_list in [
                        tup[2] for tup in os.walk(os.path.join(self.output_path, os.pardir, dir))
                    ]
                    for file in file_list
                )
                and (
                    os.path.basename(self.output_path) in dir  # only that defect directory
                    or "bulk" in str(dir).lower()  # or a bulk directory, for later
                )
                and dir not in (self.bulk_path.split("/") if self.bulk_path else [])
            ]
            if possible_defect_folders:  # update output path (otherwise will crash with informative error)
                self.output_path = os.path.join(self.output_path, os.pardir)

        if self.subfolder is None:  # determine subfolder to use
            vasp_subfolders = [
                subdir
                for possible_defect_folder in possible_defect_folders
                for subdir in os.listdir(os.path.join(self.output_path, possible_defect_folder))
                if os.path.isdir(os.path.join(self.output_path, possible_defect_folder, subdir))
                and "vasp_" in subdir
            ]
            vasp_type_count_dict = {  # Count Dik
                i: len([subdir for subdir in vasp_subfolders if i in subdir])
                for i in ["vasp_ncl", "vasp_std", "vasp_nkred_std", "vasp_gam"]
            }
            # take first entry with non-zero count, else use defect folder itself:
            self.subfolder = next((subdir for subdir, count in vasp_type_count_dict.items() if count), ".")

        possible_bulk_folders = [dir for dir in possible_defect_folders if "bulk" in str(dir).lower()]
        if self.bulk_path is None:  # determine bulk_path to use
            if len(possible_bulk_folders) == 1:
                self.bulk_path = os.path.join(self.output_path, possible_bulk_folders[0])
            elif len([dir for dir in possible_bulk_folders if dir.endswith("_bulk")]) == 1:
                self.bulk_path = os.path.join(
                    self.output_path,
                    next(iter(dir for dir in possible_bulk_folders if str(dir).lower().endswith("_bulk"))),
                )
            else:
                raise ValueError(
                    f"Could not automatically determine bulk supercell calculation folder in "
                    f"{self.output_path}, found {len(possible_bulk_folders)} folders containing "
                    f"`vasprun.xml(.gz)` files (in subfolders) and 'bulk' in the folder name. Please "
                    f"specify `bulk_path` manually."
                )

        self.defect_folders = [
            dir
            for dir in possible_defect_folders
            if dir not in possible_bulk_folders
            and (
                self.subfolder in os.listdir(os.path.join(self.output_path, dir)) or self.subfolder == "."
            )
        ]

        # add subfolder to bulk_path if present with vasprun.xml(.gz), otherwise use bulk_path as is:
        if os.path.isdir(os.path.join(self.bulk_path, self.subfolder)) and any(
            "vasprun" in file and ".xml" in file
            for file in os.listdir(os.path.join(self.bulk_path, self.subfolder))
        ):
            self.bulk_path = os.path.join(self.bulk_path, self.subfolder)
        elif all("vasprun" not in file or ".xml" not in file for file in os.listdir(self.bulk_path)):
            possible_bulk_subfolders = [
                dir
                for dir in os.listdir(self.bulk_path)
                if os.path.isdir(os.path.join(self.bulk_path, dir))
                and any(
                    "vasprun" in file and ".xml" in file
                    for file in os.listdir(os.path.join(self.bulk_path, dir))
                )
            ]
            if len(possible_bulk_subfolders) == 1 and subfolder is None:
                # if only one subfolder with a vasprun.xml file in it, and `subfolder` wasn't explicitly
                # set by the user, then use this
                self.bulk_path = os.path.join(self.bulk_path, possible_bulk_subfolders[0])
            else:
                raise FileNotFoundError(
                    f"`vasprun.xml(.gz)` files (needed for defect parsing) not found in bulk folder at: "
                    f"{self.bulk_path} or subfolder: {self.subfolder} - please ensure `vasprun.xml(.gz)` "
                    f"files are present and/or specify `bulk_path` manually."
                )

        # remove trailing '/.' from bulk_path if present:
        self.bulk_path = self.bulk_path.rstrip("/.")
        bulk_vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", self.bulk_path)
        if multiple:
            _multiple_files_warning(
                "vasprun.xml",
                self.bulk_path,
                bulk_vr_path,
                dir_type="bulk",
            )

        self.bulk_vr, self.bulk_procar = _parse_vr_and_poss_procar(
            bulk_vr_path,
            parse_projected_eigen=self.parse_projected_eigen,
            output_path=self.bulk_path,
            label="bulk",
            parse_procar=True,
        )
        self.parse_projected_eigen = (
            self.bulk_vr.projected_eigenvalues is not None or self.bulk_procar is not None
        )

        # try parsing the bulk oxidation states first, for later assigning defect "oxi_state"s (i.e.
        # fully ionised charge states):
        self._bulk_oxi_states: Union[dict, bool] = False
        if bulk_struct_w_oxi := guess_and_set_oxi_states_with_timeout(
            self.bulk_vr.final_structure, break_early_if_expensive=True
        ):
            self.bulk_vr.final_structure = bulk_struct_w_oxi
            self._bulk_oxi_states = {
                el.symbol: el.oxi_state for el in self.bulk_vr.final_structure.composition.elements  # type: ignore
            }

        self.defect_dict = {}
        self.bulk_corrections_data = {  # so we only load and parse bulk data once
            "bulk_locpot_dict": None,
            "bulk_site_potentials": None,
        }
        parsed_defect_entries = []
        parsing_warnings = []

        if self.processes is None:  # multiprocessing?
            self.processes = min(max(1, cpu_count() - 1), len(self.defect_folders) - 1)  # only
            # multiprocess as much as makes sense, if only a handful of defect folders

        if self.processes <= 1:  # no multiprocessing
            with tqdm(self.defect_folders, desc="Parsing defect calculations") as pbar:
                for defect_folder in pbar:
                    # set tqdm progress bar description to defect folder being parsed:
                    pbar.set_description(f"Parsing {defect_folder}/{self.subfolder}".replace("/.", ""))
                    parsed_defect_entry, warnings_string = self._parse_defect_and_handle_warnings(
                        defect_folder
                    )
                    parsing_warning = self._parse_parsing_warnings(
                        warnings_string, defect_folder, f"{defect_folder}/{self.subfolder}"
                    )

                    parsing_warnings.append(parsing_warning)
                    if parsed_defect_entry is not None:
                        parsed_defect_entries.append(parsed_defect_entry)

        else:  # otherwise multiprocessing:
            with FileLock("voronoi_nodes.json.lock"):  # avoid reading/writing simultaneously
                pass  # create and release lock, to be used in multiprocessing parsing

            # guess a charged defect in defect_folders, to try initially check if dielectric and
            # corrections correctly set, before multiprocessing with the same settings for all folders:
            charged_defect_folder = None
            for possible_charged_defect_folder in self.defect_folders:
                with contextlib.suppress(Exception):
                    if abs(int(possible_charged_defect_folder[-1])) > 0:  # likely charged defect
                        charged_defect_folder = possible_charged_defect_folder

            pbar = tqdm(total=len(self.defect_folders))
            try:
                if charged_defect_folder is not None:
                    # will throw warnings if dielectric is None / charge corrections not possible,
                    # and set self.skip_corrections appropriately
                    pbar.set_description(  # set this first as desc is only set after parsing in function
                        f"Parsing {charged_defect_folder}/{self.subfolder}".replace("/.", "")
                    )
                    parsed_defect_entry, warnings_string = self._parse_defect_and_handle_warnings(
                        charged_defect_folder
                    )
                    parsing_warning = self._update_pbar_and_return_warnings_from_parsing(
                        (parsed_defect_entry, warnings_string),
                        pbar,
                    )
                    parsing_warnings.append(parsing_warning)
                    if parsed_defect_entry is not None:
                        parsed_defect_entries.append(parsed_defect_entry)

                # also load the other bulk corrections data if possible:
                for k, v in self.bulk_corrections_data.items():
                    if v is None:
                        with contextlib.suppress(Exception):
                            if k == "bulk_locpot_dict":
                                self.bulk_corrections_data[k] = _get_bulk_locpot_dict(
                                    self.bulk_path, quiet=True
                                )
                            elif k == "bulk_site_potentials":
                                self.bulk_corrections_data[k] = _get_bulk_site_potentials(
                                    self.bulk_path, quiet=True
                                )

                folders_to_process = [
                    folder for folder in self.defect_folders if folder != charged_defect_folder
                ]
                pbar.set_description("Setting up multiprocessing")
                if self.processes > 1:
                    with Pool(processes=self.processes) as pool:  # result is parsed_defect_entry, warnings
                        results = pool.imap_unordered(
                            self._parse_defect_and_handle_warnings, folders_to_process
                        )
                        for result in results:
                            parsing_warning = self._update_pbar_and_return_warnings_from_parsing(
                                result, pbar
                            )
                            parsing_warnings.append(parsing_warning)
                            if result[0] is not None:
                                parsed_defect_entries.append(result[0])

            except Exception as exc:
                pbar.close()
                raise exc

            finally:
                pbar.close()

            if os.path.exists("voronoi_nodes.json.lock"):  # remove lock file
                os.remove("voronoi_nodes.json.lock")

        if parsing_warnings := [
            warning for warning in parsing_warnings if warning  # remove empty strings
        ]:
            split_parsing_warnings = [warning.split("\n\n") for warning in parsing_warnings]

            def _mention_bulk_path_subfolder_for_correction_warnings(warning: str) -> str:
                if "defect & bulk" in warning or "defect or bulk" in warning:
                    # charge correction file warning, print subfolder and bulk_path:
                    if self.subfolder == ".":
                        warning += f"\n(using bulk path: {self.bulk_path} and without defect subfolders)"
                    else:
                        warning += (
                            f"\n(using bulk path {self.bulk_path} and {self.subfolder} defect subfolders)"
                        )

                return warning

            split_parsing_warnings = [
                [_mention_bulk_path_subfolder_for_correction_warnings(warning) for warning in warning_list]
                for warning_list in split_parsing_warnings
            ]
            flattened_warnings_list = [
                warning for warning_list in split_parsing_warnings for warning in warning_list
            ]
            duplicate_warnings: dict[str, list[str]] = {
                warning: []
                for warning in set(flattened_warnings_list)
                if flattened_warnings_list.count(warning) > 1
            }
            new_parsing_warnings = []
            parsing_errors_dict: dict[str, list[str]] = {
                message.split("got error: ")[1]: []
                for message in set(flattened_warnings_list)
                if "Parsing failed for " in message
            }
            multiple_files_warning_dict: dict[str, list[tuple]] = {
                "vasprun.xml": [],
                "OUTCAR": [],
                "LOCPOT": [],
            }

            for warnings_list in split_parsing_warnings:
                failed_warnings = [
                    warning_message
                    for warning_message in warnings_list
                    if "Parsing failed for " in warning_message
                ]
                if failed_warnings:
                    defect_name = failed_warnings[0].split("Parsing failed for ")[1].split(", got ")[0]
                    error = failed_warnings[0].split("got error: ")[1]
                    parsing_errors_dict[error].append(defect_name)
                elif "Warning(s) encountered" in warnings_list[0]:
                    defect_name = warnings_list[0].split("when parsing ")[1].split(" at")[0]
                else:
                    defect_name = None

                new_warnings_list = []
                for warning in warnings_list:
                    if warning.startswith("Multiple"):
                        file_type = warning.split("`")[1]
                        directory = warning.split("directory: ")[1].split(". Using")[0]
                        chosen_file = warning.split("Using ")[1].split(" to")[0]
                        multiple_files_warning_dict[file_type].append((directory, chosen_file))

                    elif warning in duplicate_warnings:
                        duplicate_warnings[warning].append(defect_name)

                    else:
                        new_warnings_list.append(warning)

                if [  # if we still have other warnings, keep them for parsing_warnings list
                    warning
                    for warning in new_warnings_list
                    if "Warning(s) encountered" not in warning and "Parsing failed for " not in warning
                ]:
                    new_parsing_warnings.append("\n".join(new_warnings_list))

            for error, defect_list in parsing_errors_dict.items():
                if defect_list:
                    if len(defect_list) > 1:
                        warnings.warn(
                            f"Parsing failed for defects: {defect_list} with the same error:\n{error}"
                        )
                    else:
                        warnings.warn(f"Parsing failed for defect {defect_list[0]} with error:\n{error}")

            for file_type, directory_file_list in multiple_files_warning_dict.items():
                if directory_file_list:
                    joined_info_string = "\n".join(
                        [f"{directory}: {file}" for directory, file in directory_file_list]
                    )
                    warnings.warn(
                        f"Multiple `{file_type}` files found in certain defect directories:\n"
                        f"(directory: chosen file for parsing):\n"
                        f"{joined_info_string}\n"
                        f"{file_type} files are used to "
                        f"{_vasp_file_parsing_action_dict[file_type]}"
                    )

            parsing_warnings = new_parsing_warnings
            if parsing_warnings:
                warnings.warn("\n".join(parsing_warnings))

            for warning, defect_name_list in duplicate_warnings.items():
                defect_list = [
                    defect_name
                    for defect_name in defect_name_list
                    if defect_name
                    and all(
                        defect_name not in defects_with_errors
                        for defects_with_errors in parsing_errors_dict.values()
                    )
                ]  # remove None and don't warn if later encountered parsing error (already warned)
                if defect_list:
                    warnings.warn(f"Defects: {defect_list} each encountered the same warning:\n{warning}")

        if not parsed_defect_entries:
            subfolder_string = f" and `subfolder`: '{self.subfolder}'" if self.subfolder != "." else ""
            raise ValueError(
                f"No defect calculations in `output_path` '{self.output_path}' were successfully parsed, "
                f"using `bulk_path`: {self.bulk_path}{subfolder_string}. Please check the correct "
                f"defect/bulk paths and subfolder are being set, and that the folder structure is as "
                f"expected (see `DefectsParser` docstring)."
            )

        # get any defect entries in parsed_defect_entries that share the same name (without charge):
        # first get any entries with duplicate names:
        entries_to_rename = [
            defect_entry
            for defect_entry in parsed_defect_entries
            if len(
                [
                    defect_entry
                    for other_defect_entry in parsed_defect_entries
                    if defect_entry.name == other_defect_entry.name
                ]
            )
            > 1
        ]
        # then get all entries with the same name(s), ignoring charge state (in case e.g. only duplicate
        # for one charge state etc):
        entries_to_rename = [
            defect_entry
            for defect_entry in parsed_defect_entries
            if any(
                defect_entry.name.rsplit("_", 1)[0] == other_defect_entry.name.rsplit("_", 1)[0]
                for other_defect_entry in entries_to_rename
            )
        ]

        self.defect_dict = {
            defect_entry.name: defect_entry
            for defect_entry in parsed_defect_entries
            if defect_entry not in entries_to_rename
        }

        with contextlib.suppress(AttributeError, TypeError):  # sort by supercell frac cooords,
            # to aid deterministic naming:
            entries_to_rename.sort(
                key=lambda x: _frac_coords_sort_func(_get_defect_supercell_bulk_site_coords(x))
            )

        new_named_defect_entries_dict = name_defect_entries(entries_to_rename)
        # set name attribute: (these are names without charges!)
        for defect_name_wout_charge, defect_entry in new_named_defect_entries_dict.items():
            defect_entry.name = (
                f"{defect_name_wout_charge}_{'+' if defect_entry.charge_state > 0 else ''}"
                f"{defect_entry.charge_state}"
            )

        if duplicate_names := [  # if any duplicate names, crash (and burn, b...)
            defect_entry.name
            for defect_entry in entries_to_rename
            if defect_entry.name in self.defect_dict
        ]:
            raise ValueError(
                f"Some defect entries have the same name, due to mixing of doped-named and unnamed "
                f"defect folders. This would cause defect entries to be overwritten. Please check "
                f"your defect folder names in `output_path`!\nDuplicate defect names:\n"
                f"{duplicate_names}"
            )

        self.defect_dict.update(
            {defect_entry.name: defect_entry for defect_entry in new_named_defect_entries_dict.values()}
        )

        FNV_correction_errors = []
        eFNV_correction_errors = []
        for name, defect_entry in self.defect_dict.items():
            if (
                defect_entry.corrections_metadata.get("freysoldt_charge_correction_error", 0)
                > error_tolerance
            ):
                FNV_correction_errors.append(
                    (name, defect_entry.corrections_metadata["freysoldt_charge_correction_error"])
                )
            if (
                defect_entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
                > error_tolerance
            ):
                eFNV_correction_errors.append(
                    (name, defect_entry.corrections_metadata["kumagai_charge_correction_error"])
                )

        def _call_multiple_corrections_tolerance_warning(correction_errors, type="FNV"):
            long_name = "Freysoldt" if type == "FNV" else "Kumagai"
            if error_tolerance >= 0.01:  # if greater than 10 meV, round energy values to meV:
                error_tol_string = f"{error_tolerance:.3f}"
                correction_errors_string = "\n".join(
                    f"{name}: {error:.3f} eV" for name, error in correction_errors
                )
            else:  # else give in scientific notation:
                error_tol_string = f"{error_tolerance:.2e}"
                correction_errors_string = "\n".join(
                    f"{name}: {error:.2e} eV" for name, error in correction_errors
                )

            warnings.warn(
                f"Estimated error in the {long_name} ({type}) charge correction for certain "
                f"defects is greater than the `error_tolerance` (= {error_tol_string} eV):"
                f"\n{correction_errors_string}\n"
                f"You may want to check the accuracy of the corrections by plotting the site "
                f"potential differences (using `defect_entry.get_{long_name.lower()}_correction()`"
                f" with `plot=True`). Large errors are often due to unstable or shallow defect "
                f"charge states (which can't be accurately modelled with the supercell "
                f"approach). If these errors are not acceptable, you may need to use a larger "
                f"supercell for more accurate energies."
            )

        if FNV_correction_errors:
            _call_multiple_corrections_tolerance_warning(FNV_correction_errors, type="FNV")
        if eFNV_correction_errors:
            _call_multiple_corrections_tolerance_warning(eFNV_correction_errors, type="eFNV")

        # check if same type of charge correction was used in each case or not:
        if (
            len(
                {
                    k
                    for defect_entry in self.defect_dict.values()
                    for k in defect_entry.corrections
                    if k.endswith("_charge_correction")
                }
            )
            > 1
        ):
            warnings.warn(
                "Beware: The Freysoldt (FNV) charge correction scheme has been used for some defects, "
                "while the Kumagai (eFNV) scheme has been used for others. For _isotropic_ materials, "
                "this should be fine, and the results should be the same regardless (assuming a "
                "relatively well-converged supercell size), while for _anisotropic_ materials this could "
                "lead to some quantitative inaccuracies. You can use the "
                "`DefectThermodynamics.get_formation_energies()` method to print out the calculated "
                "charge corrections for all defects, and/or visualise the charge corrections using "
                "`defect_entry.get_freysoldt_correction`/`get_kumagai_correction` with `plot=True` to "
                "check."
            )  # either way have the error analysis for the charge corrections so in theory should be grand
        # note that we also check if multiple charge corrections have been applied to the same defect
        # within the charge correction functions (with self._check_if_multiple_finite_size_corrections())

        mismatching_INCAR_warnings = [
            (name, defect_entry.calculation_metadata.get("mismatching_INCAR_tags"))
            for name, defect_entry in self.defect_dict.items()
            if defect_entry.calculation_metadata.get("mismatching_INCAR_tags", True) is not True
        ]
        if mismatching_INCAR_warnings:
            joined_info_string = "\n".join(
                [f"{name}: {mismatching}" for name, mismatching in mismatching_INCAR_warnings]
            )
            warnings.warn(
                f"There are mismatching INCAR tags for (some of) your bulk and defect calculations which "
                f"are likely to cause errors in the parsed results (energies). Found the following "
                f"differences:\n"
                f"(in the format: (INCAR tag, value in bulk calculation, value in defect calculation)):"
                f"\n{joined_info_string}\n"
                f"In general, the same INCAR settings should be used in all final calculations for these "
                f"tags which can affect energies!"
            )

        mismatching_kpoints_warnings = [
            (name, defect_entry.calculation_metadata.get("mismatching_KPOINTS"))
            for name, defect_entry in self.defect_dict.items()
            if defect_entry.calculation_metadata.get("mismatching_KPOINTS", True) is not True
        ]
        if mismatching_kpoints_warnings:
            joined_info_string = "\n".join(
                [f"{name}: {mismatching}" for name, mismatching in mismatching_kpoints_warnings]
            )
            warnings.warn(
                f"There are mismatching KPOINTS for (some of) your bulk and defect calculations which "
                f"are likely to cause errors in the parsed results (energies). Found the following "
                f"differences:\n"
                f"(in the format: (bulk kpoints, defect kpoints)):"
                f"\n{joined_info_string}\n"
                f"In general, the same KPOINTS settings should be used for all final calculations for "
                f"accurate results!"
            )

        mismatching_potcars_warnings = [
            (name, defect_entry.calculation_metadata.get("mismatching_POTCAR_symbols"))
            for name, defect_entry in self.defect_dict.items()
            if defect_entry.calculation_metadata.get("mismatching_POTCAR_symbols", True) is not True
        ]
        if mismatching_potcars_warnings:
            joined_info_string = "\n".join(
                [f"{name}: {mismatching}" for name, mismatching in mismatching_potcars_warnings]
            )
            warnings.warn(
                f"There are mismatching POTCAR symbols for (some of) your bulk and defect calculations "
                f"which are likely to cause severe errors in the parsed results (energies). Found the "
                f"following differences:\n"
                f"(in the format: (bulk POTCARs, defect POTCARs)):"
                f"\n{joined_info_string}\n"
                f"In general, the same POTCAR settings should be used for all calculations for accurate "
                f"results!"
            )

        if self.json_filename is not False:  # save to json unless json_filename is False:
            if self.json_filename is None:
                formula = next(
                    iter(self.defect_dict.values())
                ).defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
                self.json_filename = f"{formula}_defect_dict.json"

            dumpfn(self.defect_dict, os.path.join(self.output_path, self.json_filename))  # type: ignore

    def _parse_parsing_warnings(self, warnings_string, defect_folder, defect_path):
        if warnings_string:
            if "Parsing failed for " in warnings_string:
                return warnings_string
            return (
                f"Warning(s) encountered when parsing {defect_folder} at {defect_path}:\n\n"
                f"{warnings_string}"
            )

        return ""

    def _update_pbar_and_return_warnings_from_parsing(self, result, pbar):
        pbar.update()

        if result[0] is not None:
            i = -1 if self.subfolder == "." else -2
            defect_folder = result[0].calculation_metadata["defect_path"].replace("/.", "").split("/")[i]
            pbar.set_description(f"Parsing {defect_folder}/{self.subfolder}".replace("/.", ""))

            if result[1]:
                return self._parse_parsing_warnings(
                    result[1], defect_folder, result[0].calculation_metadata["defect_path"]
                )

        return result[1] or ""  # failed parsing warning if result[0] is None

    def _parse_defect_and_handle_warnings(self, defect_folder):
        """
        Process defect and catch warnings along the way, so we can print which
        warnings came from which defect together at the end, in a summarised
        output.
        """
        with warnings.catch_warnings(record=True) as captured_warnings:
            parsed_defect_entry = self._parse_single_defect(defect_folder)

        ignore_messages = [
            "Estimated error",
            "There are mismatching",
            "The KPOINTS",
            "The POTCAR",
        ]  # collectively warned later
        warnings_string = "\n\n".join(
            str(warning.message)
            for warning in captured_warnings
            if not any(warning.message.args[0].startswith(i) for i in ignore_messages)
        )

        return parsed_defect_entry, warnings_string

    def _parse_single_defect(self, defect_folder):
        try:
            self.kwargs.update(self.bulk_corrections_data)  # update with bulk corrections data
            dp = DefectParser.from_paths(
                defect_path=os.path.join(self.output_path, defect_folder, self.subfolder),
                bulk_path=self.bulk_path,
                bulk_vr=self.bulk_vr,
                bulk_procar=self.bulk_procar,
                dielectric=self.dielectric,
                skip_corrections=self.skip_corrections,
                error_tolerance=self.error_tolerance,
                bulk_band_gap_vr=self.bulk_band_gap_vr,
                oxi_state=self.kwargs.get("oxi_state") if self._bulk_oxi_states else "Undetermined",
                parse_projected_eigen=self.parse_projected_eigen,
                **self.kwargs,
            )

            if dp.skip_corrections and dp.defect_entry.charge_state != 0 and self.dielectric is None:
                self.skip_corrections = dp.skip_corrections  # set skip_corrections to True if
                # dielectric is None and there are charged defects present (shows dielectric warning once)

            if (
                dp.defect_entry.calculation_metadata.get("bulk_locpot_dict") is not None
                and self.bulk_corrections_data.get("bulk_locpot_dict") is None
            ):
                self.bulk_corrections_data["bulk_locpot_dict"] = dp.defect_entry.calculation_metadata[
                    "bulk_locpot_dict"
                ]

            if (
                dp.defect_entry.calculation_metadata.get("bulk_site_potentials") is not None
                and self.bulk_corrections_data.get("bulk_site_potentials") is None
            ):
                self.bulk_corrections_data["bulk_site_potentials"] = (
                    dp.defect_entry.calculation_metadata
                )["bulk_site_potentials"]

        except Exception as exc:
            warnings.warn(
                f"Parsing failed for "
                f"{defect_folder if self.subfolder == '.' else f'{defect_folder}/{self.subfolder}'}, "
                f"got error: {exc!r}"
            )
            return None

        return dp.defect_entry

    def get_defect_thermodynamics(
        self,
        chempots: Optional[dict] = None,
        el_refs: Optional[dict] = None,
        vbm: Optional[float] = None,
        band_gap: Optional[float] = None,
        dist_tol: float = 1.5,
        check_compatibility: bool = True,
    ) -> DefectThermodynamics:
        r"""
        Generates a DefectThermodynamics object from the parsed ``DefectEntry``
        objects in self.defect_dict, which can then be used to analyse and plot
        the defect thermodynamics (formation energies, transition levels,
        concentrations etc).

        Note that the DefectEntry.name attributes (rather than the defect_name key
        in the defect_dict) are used to label the defects in plots.

        Args:
            chempots (dict):
                Dictionary of chemical potentials to use for calculating the defect
                formation energies. This can have the form of
                ``{"limits": [{'limit': [chempot_dict]}]}`` (the format generated by
                ``doped``\'s chemical potential parsing functions (see tutorials)) which
                allows easy analysis over a range of chemical potentials - where limit(s)
                (chemical potential limit(s)) to analyse/plot can later be chosen using
                the ``limits`` argument.

                Alternatively this can be a dictionary of chemical potentials for a
                single limit (limit), in the format: ``{element symbol: chemical potential}``.
                If manually specifying chemical potentials this way, you can set the
                ``el_refs`` option with the DFT reference energies of the elemental phases
                in order to show the formal (relative) chemical potentials above the
                formation energy plot, in which case it is the formal chemical potentials
                (i.e. relative to the elemental references) that should be given here,
                otherwise the absolute (DFT) chemical potentials should be given.

                If None (default), sets all chemical potentials to zero. Chemical
                potentials can also be supplied later in each analysis function.
                (Default: None)
            el_refs (dict):
                Dictionary of elemental reference energies for the chemical potentials
                in the format:
                ``{element symbol: reference energy}`` (to determine the formal chemical
                potentials, when ``chempots`` has been manually specified as
                ``{element symbol: chemical potential}``). Unnecessary if ``chempots`` is
                provided in format generated by ``doped`` (see tutorials).
                (Default: None)
            vbm (float):
                VBM eigenvalue to use as Fermi level reference point for analysis.
                If None (default), will use ``"vbm"`` from the ``calculation_metadata``
                dict attributes of the parsed ``DefectEntry`` objects, which by default
                is taken from the bulk supercell VBM (unless ``bulk_band_gap_vr`` is set
                during parsing).
                Note that ``vbm`` should only affect the reference for the Fermi level
                values output by ``doped`` (as this VBM eigenvalue is used as the zero
                reference), thus affecting the position of the band edges in the defect
                formation energy plots and doping window / dopability limit functions,
                and the reference of the reported Fermi levels.
            band_gap (float):
                Band gap of the host, to use for analysis.
                If None (default), will use "gap" from the calculation_metadata
                dict attributes of the parsed DefectEntry objects.
            dist_tol (float):
                Threshold for the closest distance (in Å) between equivalent
                defect sites, for different species of the same defect type,
                to be grouped together (for plotting and transition level
                analysis). If the minimum distance between equivalent defect
                sites is less than ``dist_tol``, then they will be grouped
                together, otherwise treated as separate defects.
                (Default: 1.5)
            check_compatibility (bool):
                Whether to check the compatibility of the bulk entry for each defect
                entry (i.e. that all reference bulk energies are the same).
                (Default: True)

        Returns:
            doped DefectThermodynamics object (``DefectThermodynamics``)
        """
        if not self.defect_dict or self.defect_dict is None:
            raise ValueError(
                "No defects found in `defect_dict`. DefectThermodynamics object can only be generated "
                "when defects have been parsed and are present as `DefectEntry`s in "
                "`DefectsParser.defect_dict`."
            )

        return DefectThermodynamics(
            list(self.defect_dict.values()),
            chempots=chempots,
            el_refs=el_refs,
            vbm=vbm,
            band_gap=band_gap,
            dist_tol=dist_tol,
            check_compatibility=check_compatibility,
        )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectsParser`` object.
        """
        formula = next(
            iter(self.defect_dict.values())
        ).defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectsParser for bulk composition {formula}, with {len(self.defect_dict)} parsed "
            f"defect entries in self.defect_dict. Available attributes:\n{properties}\n\n"
            f"Available methods:\n{methods}"
        )


def _parse_vr_and_poss_procar(
    vr_path: str,
    parse_projected_eigen: Optional[bool] = None,
    output_path: Optional[str] = None,
    label: str = "bulk",
    parse_procar: bool = True,
):
    procar = None

    failed_eig_parsing_warning_message = (
        f"Could not parse eigenvalue data from vasprun.xml.gz files in {label} folder at {output_path}"
    )

    try:
        vr = get_vasprun(
            vr_path,
            parse_projected_eigen=parse_projected_eigen is not False,
            parse_eigen=(parse_projected_eigen is not False or label == "bulk"),
        )  # vr.eigenvalues not needed for defects except for vr-only eigenvalue analysis
    except Exception as vr_exc:
        vr = get_vasprun(vr_path, parse_projected_eigen=False, parse_eigen=label == "bulk")
        failed_eig_parsing_warning_message += f", got error:\n{vr_exc}"

        if parse_procar:
            procar_path, multiple = _get_output_files_and_check_if_multiple("PROCAR", output_path)
            if "PROCAR" in procar_path and parse_projected_eigen is not False:
                try:
                    procar = get_procar(procar_path)

                except Exception as procar_exc:
                    failed_eig_parsing_warning_message += (
                        f"\nThen got the following error when attempting to parse projected eigenvalues "
                        f"from the defect PROCAR(.gz):\n{procar_exc}"
                    )

    if vr.projected_eigenvalues is None and procar is None and parse_projected_eigen is True:
        # only warn if parse_projected_eigen is set to True (not None)
        warnings.warn(failed_eig_parsing_warning_message)

    return vr, procar if parse_procar else vr


class DefectParser:
    def __init__(
        self,
        defect_entry: DefectEntry,
        defect_vr: Optional[Vasprun] = None,
        bulk_vr: Optional[Vasprun] = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        parse_projected_eigen: Optional[bool] = None,
        **kwargs,
    ):
        """
        Create a ``DefectParser`` object, which has methods for parsing the
        results of defect supercell calculations.

        Direct initiation with ``DefectParser()`` is typically not recommended. Rather
        ``DefectParser.from_paths()`` or ``defect_entry_from_paths()`` are preferred as
        shown in the doped parsing tutorials.

        Args:
            defect_entry (DefectEntry):
                doped ``DefectEntry``
            defect_vr (Vasprun):
                ``pymatgen`` ``Vasprun`` object for the defect supercell calculation
            bulk_vr (Vasprun):
                ``pymatgen`` ``Vasprun`` object for the reference bulk supercell calculation
            skip_corrections (bool):
                Whether to skip calculation and application of finite-size charge
                corrections to the defect energy (not recommended in most cases).
                Default = False.
            error_tolerance (float):
                If the estimated error in the defect charge correction, based on the
                variance of the potential in the sampling region is greater than this
                value (in eV), then a warning is raised. (default: 0.05 eV)
            parse_projected_eigen (bool):
                Whether to parse the projected eigenvalues & orbitals from the bulk and
                defect calculations (so ``DefectEntry.get_eigenvalue_analysis()`` can
                then be used with no further parsing). Will initially try to load orbital
                projections from ``vasprun.xml(.gz)`` files (slightly slower but more
                accurate), or failing that from ``PROCAR(.gz)`` files if present in the
                bulk/defect directories. Parsing this data can increase total parsing time
                by anywhere from ~5-25%, so set to ``False`` if parsing speed is crucial.
                Default is ``None``, which will attempt to load this data but with no
                warning if it fails (otherwise if ``True`` a warning will be printed).
            **kwargs:
                Keyword arguments to pass to ``DefectParser()`` methods
                (``load_FNV_data()``, ``load_eFNV_data()``, ``load_bulk_gap_data()``)
                ``point_symmetry_from_defect_entry()`` or ``defect_from_structures``,
                including ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
                ``mpid``, ``api_key``, ``symprec`` or ``oxi_state``. Primarily used by
                ``DefectsParser`` to expedite parsing by avoiding reloading bulk data
                for each defect.
        """
        self.defect_entry: DefectEntry = defect_entry
        self.defect_vr = defect_vr
        self.bulk_vr = bulk_vr
        self.skip_corrections = skip_corrections
        self.error_tolerance = error_tolerance
        self.kwargs = kwargs or {}
        self.parse_projected_eigen = parse_projected_eigen

    @classmethod
    def from_paths(
        cls,
        defect_path: str,
        bulk_path: Optional[str] = None,
        bulk_vr: Optional[Vasprun] = None,
        bulk_procar: Optional[Union["EasyunfoldProcar", Procar]] = None,
        dielectric: Optional[Union[float, int, np.ndarray, list]] = None,
        charge_state: Optional[int] = None,
        initial_defect_structure_path: Optional[str] = None,
        skip_corrections: bool = False,
        error_tolerance: float = 0.05,
        bulk_band_gap_vr: Optional[Union[str, Vasprun]] = None,
        parse_projected_eigen: Optional[bool] = None,
        **kwargs,
    ):
        """
        Parse the defect calculation outputs in ``defect_path`` and return the
        ``DefectParser`` object. By default, the
        ``DefectParser.defect_entry.name`` attribute (later used to label
        defects in plots) is set to the defect_path folder name (if it is a
        recognised defect name), else it is set to the default doped name for
        that defect (using the estimated `unrelaxed` defect structure, for the
        point group and neighbour distances).

        Note that the bulk and defect supercells should have the same definitions/basis
        sets (for site-matching and finite-size charge corrections to work appropriately).

        Args:
            defect_path (str):
                Path to defect supercell folder (containing at least ``vasprun.xml(.gz)``).
            bulk_path (str):
                Path to bulk supercell folder (containing at least ``vasprun.xml(.gz)``).
                Not required if ``bulk_vr`` is provided.
            bulk_vr (Vasprun):
                ``pymatgen`` ``Vasprun`` object for the reference bulk supercell
                calculation, if already loaded (can be supplied to expedite parsing).
                Default is ``None``.
            bulk_procar (Procar):
                ``easyunfold``/``pymatgen`` ``Procar`` object, for the reference bulk
                supercell calculation if already loaded (can be supplied to expedite
                parsing). Default is ``None``.
            dielectric (float or int or 3x1 matrix or 3x3 matrix):
                Ionic + static contributions to the dielectric constant. If not provided,
                charge corrections cannot be computed and so ``skip_corrections`` will be
                set to true.
            charge_state (int):
                Charge state of defect. If not provided, will be automatically determined
                from defect calculation outputs, or if that fails, using the defect folder
                name (must end in "_+X" or "_-X" where +/-X is the defect charge state).
            initial_defect_structure_path (str):
                Path to the initial/unrelaxed defect structure. Only recommended for use
                if structure matching with the relaxed defect structure(s) fails (rare).
                Default is ``None``.
            skip_corrections (bool):
                Whether to skip the calculation and application of finite-size charge
                corrections to the defect energy (not recommended in most cases).
                Default = ``False``.
            error_tolerance (float):
                If the estimated error in the defect charge correction, based on the
                variance of the potential in the sampling region, is greater than this
                value (in eV), then a warning is raised. (default: 0.05 eV)
            bulk_band_gap_vr (str or Vasprun):
                Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` ``Vasprun``
                object, from which to determine the bulk band gap and band edge positions.
                If the VBM/CBM occur at `k`-points which are not included in the bulk
                supercell calculation, then this parameter should be used to provide the
                output of a bulk bandstructure calculation so that these are correctly
                determined.
                Alternatively, you can edit/add the ``"gap"`` and ``"vbm"`` entries in
                ``self.defect_entry.calculation_metadata`` to match the correct
                (eigen)values.
                If None, will use ``DefectEntry.calculation_metadata["bulk_path"]`` (i.e.
                the bulk supercell calculation output).

                Note that the ``"gap"`` and ``"vbm"`` values should only affect the
                reference for the Fermi level values output by ``doped`` (as this VBM
                eigenvalue is used as the zero reference), thus affecting the position of
                the band edges in the defect formation energy plots and doping window /
                dopability limit functions, and the reference of the reported Fermi levels.
            parse_projected_eigen (bool):
                Whether to parse the projected eigenvalues & orbitals from the bulk and
                defect calculations (so ``DefectEntry.get_eigenvalue_analysis()`` can
                then be used with no further parsing). Will initially try to load orbital
                projections from ``vasprun.xml(.gz)`` files (slightly slower but more
                accurate), or failing that from ``PROCAR(.gz)`` files if present in the
                bulk/defect directories. Parsing this data can increase total parsing time
                by anywhere from ~5-25%, so set to ``False`` if parsing speed is crucial.
                Default is ``None``, which will attempt to load this data but with no
                warning if it fails (otherwise if ``True`` a warning will be printed).
            **kwargs:
                Keyword arguments to pass to ``DefectParser()`` methods
                (``load_FNV_data()``, ``load_eFNV_data()``, ``load_bulk_gap_data()``)
                ``point_symmetry_from_defect_entry()`` or ``defect_from_structures``,
                including ``bulk_locpot_dict``, ``bulk_site_potentials``, ``use_MP``,
                ``mpid``, ``api_key``, ``symprec`` or ``oxi_state``. Primarily used by
                ``DefectsParser`` to expedite parsing by avoiding reloading bulk data
                for each defect.

        Return:
            ``DefectParser`` object.
        """
        _ignore_pmg_warnings()  # ignore unnecessary pymatgen warnings

        calculation_metadata = {
            "bulk_path": os.path.abspath(bulk_path) if bulk_path else "bulk Vasprun supplied",
            "defect_path": os.path.abspath(defect_path),
        }

        if bulk_path is not None and bulk_vr is None:
            # add bulk simple properties
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", bulk_path)
            if multiple:
                _multiple_files_warning(
                    "vasprun.xml",
                    bulk_path,
                    bulk_vr_path,
                    dir_type="bulk",
                )
            bulk_vr, reparsed_bulk_procar = _parse_vr_and_poss_procar(
                bulk_vr_path,
                parse_projected_eigen,
                bulk_path,
                label="bulk",
                parse_procar=bulk_procar is None,
            )
            if bulk_procar is None and reparsed_bulk_procar is not None:
                bulk_procar = reparsed_bulk_procar
            parse_projected_eigen = bulk_vr.projected_eigenvalues is not None or bulk_procar is not None

        elif bulk_vr is None:
            raise ValueError("Either `bulk_path` or `bulk_vr` must be provided!")
        bulk_supercell = bulk_vr.final_structure.copy()

        # add defect simple properties
        (
            defect_vr_path,
            multiple,
        ) = _get_output_files_and_check_if_multiple("vasprun.xml", defect_path)
        if multiple:
            _multiple_files_warning(
                "vasprun.xml",
                defect_path,
                defect_vr_path,
                dir_type="defect",
            )

        defect_vr, defect_procar = _parse_vr_and_poss_procar(
            defect_vr_path, parse_projected_eigen, defect_path, label="defect"
        )
        parse_projected_eigen = defect_procar is not None or defect_vr.projected_eigenvalues is not None

        possible_defect_name = os.path.basename(
            defect_path.rstrip("/.").rstrip("/")  # remove any trailing slashes to ensure correct name
        )  # set equal to folder name
        if "vasp" in possible_defect_name:  # get parent directory name:
            possible_defect_name = os.path.basename(os.path.dirname(defect_path))

        try:
            parsed_charge_state: int = defect_charge_from_vasprun(defect_vr, charge_state)
        except RuntimeError as orig_exc:  # auto charge guessing failed and charge_state not provided,
            # try to determine from folder name - must have "-" or "+" at end of name for this
            try:
                charge_state_suffix = possible_defect_name.rsplit("_", 1)[-1]
                if charge_state_suffix[0] not in ["-", "+"]:
                    raise ValueError(
                        f"Could not guess charge state from folder name ({possible_defect_name}), must "
                        f"end in '_+X' or '_-X' where +/-X is the charge state."
                    )

                parsed_charge_state = int(charge_state_suffix)
                if abs(parsed_charge_state) >= 7:
                    raise ValueError(
                        f"Guessed charge state from folder name was {parsed_charge_state:+} which is "
                        f"almost certainly unphysical"
                    )
            except Exception as next_exc:
                raise orig_exc from next_exc

        degeneracy_factors = {
            "spin degeneracy": _defect_spin_degeneracy_from_vasprun(
                defect_vr, charge_state=parsed_charge_state
            )
        }

        if dielectric is None and not skip_corrections and parsed_charge_state != 0:
            warnings.warn(
                "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
                "corrections, but none was provided, so charge corrections will be skipped "
                "(`skip_corrections = True`). Formation energies and transition levels of charged "
                "defects will likely be very inaccurate without charge corrections!"
            )
            skip_corrections = True

        if dielectric is not None:
            dielectric = _convert_dielectric_to_tensor(dielectric)
            calculation_metadata["dielectric"] = dielectric

        # Add defect structure to calculation_metadata, so it can be pulled later on (e.g. for eFNV)
        defect_structure = defect_vr.final_structure.copy()
        calculation_metadata["defect_structure"] = defect_structure

        # check if the bulk and defect supercells are the same size:
        if not np.isclose(defect_structure.volume, bulk_supercell.volume, rtol=1e-2):
            warnings.warn(
                f"The defect and bulk supercells are not the same size, having volumes of "
                f"{defect_structure.volume:.1f} and {bulk_supercell.volume:.1f} Å^3 respectively. This "
                f"may cause errors in parsing and/or output energies. In most cases (unless looking at "
                f"extremely high doping concentrations) the same fixed supercell (ISIF = 2) should be "
                f"used for both the defect and bulk calculations! (i.e. assuming the dilute limit)"
            )

        # identify defect site, structural information, and create defect object:
        # try load previous bulk_voronoi_node_dict if present:
        def _read_bulk_voronoi_node_dict(bulk_path):
            if os.path.exists(os.path.join(bulk_path, "voronoi_nodes.json")):
                return loadfn(os.path.join(bulk_path, "voronoi_nodes.json"))
            return {}

        if os.path.exists("voronoi_nodes.json.lock"):
            with FileLock("voronoi_nodes.json.lock"):
                bulk_voronoi_node_dict = _read_bulk_voronoi_node_dict(bulk_path)
        else:
            bulk_voronoi_node_dict = _read_bulk_voronoi_node_dict(bulk_path)

        # Can specify initial defect structure (to help find the defect site if we have a very distorted
        # final structure), but regardless try using the final structure (from defect OUTCAR) first:
        try:
            (
                defect,
                defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
                defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
                # w/interstitials
                defect_site_index,
                bulk_site_index,
                guessed_initial_defect_structure,
                unrelaxed_defect_structure,
                bulk_voronoi_node_dict,
            ) = defect_from_structures(
                bulk_supercell,
                defect_structure.copy(),
                return_all_info=True,
                bulk_voronoi_node_dict=bulk_voronoi_node_dict,
                oxi_state=kwargs.get("oxi_state"),
            )

        except RuntimeError:
            if not initial_defect_structure_path:
                raise

            defect_structure_for_ID = Poscar.from_file(initial_defect_structure_path).structure.copy()
            (
                defect,
                defect_site_in_initial_struct,
                defect_site_in_bulk,  # bulk site for vac/sub, relaxed defect site w/interstitials
                defect_site_index,  # in this initial_defect_structure
                bulk_site_index,
                guessed_initial_defect_structure,
                unrelaxed_defect_structure,
                bulk_voronoi_node_dict,
            ) = defect_from_structures(
                bulk_supercell,
                defect_structure_for_ID,
                return_all_info=True,
                bulk_voronoi_node_dict=bulk_voronoi_node_dict,
                oxi_state=kwargs.get("oxi_state"),
            )

            # then try get defect_site in final structure:
            # need to check that it's the correct defect site and hasn't been reordered/changed compared to
            # the initial_defect_structure used here -> check same element and distance reasonable:
            defect_site = defect_site_in_initial_struct

            if defect.defect_type != core.DefectType.Vacancy:
                final_defect_site = defect_structure[defect_site_index]
                if (
                    defect_site_in_initial_struct.specie.symbol == final_defect_site.specie.symbol
                ) and final_defect_site.distance(defect_site_in_initial_struct) < 2:
                    defect_site = final_defect_site

        calculation_metadata["guessed_initial_defect_structure"] = guessed_initial_defect_structure
        calculation_metadata["defect_site_index"] = defect_site_index
        calculation_metadata["bulk_site_index"] = bulk_site_index

        # add displacement from (guessed) initial site to final defect site:
        if defect_site_index is not None:  # not a vacancy
            guessed_initial_site = guessed_initial_defect_structure[defect_site_index]
            final_site = defect_vr.final_structure[defect_site_index]
            guessed_displacement = final_site.distance(guessed_initial_site)
            calculation_metadata["guessed_initial_defect_site"] = guessed_initial_site
            calculation_metadata["guessed_defect_displacement"] = guessed_displacement
        else:  # vacancy
            calculation_metadata["guessed_initial_defect_site"] = bulk_supercell[bulk_site_index]
            calculation_metadata["guessed_defect_displacement"] = None  # type: ignore

        calculation_metadata["unrelaxed_defect_structure"] = unrelaxed_defect_structure
        if bulk_site_index is None:  # interstitial
            calculation_metadata["bulk_site"] = defect_site_in_bulk
        else:
            calculation_metadata["bulk_site"] = bulk_supercell[bulk_site_index]

        defect_entry = DefectEntry(
            # pmg attributes:
            defect=defect,  # this corresponds to _unrelaxed_ defect
            charge_state=parsed_charge_state,
            sc_entry=defect_vr.get_computed_entry(),
            sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
            bulk_entry=bulk_vr.get_computed_entry(),
            # doped attributes:
            name=possible_defect_name,  # set later, so set now to avoid guessing in ``__post_init__()``
            defect_supercell_site=defect_site,  # _relaxed_ defect site
            defect_supercell=defect_vr.final_structure,
            bulk_supercell=bulk_vr.final_structure,
            calculation_metadata=calculation_metadata,
            degeneracy_factors=degeneracy_factors,
        )

        bulk_supercell_symm_ops = _get_sga(
            defect_entry.defect.structure, symprec=0.01
        ).get_symmetry_operations()
        if defect.defect_type == core.DefectType.Interstitial:
            # site multiplicity is automatically computed for vacancies and substitutions (much easier),
            # but not interstitials
            defect_entry.defect.multiplicity = len(
                _get_all_equiv_sites(
                    _get_defect_supercell_bulk_site_coords(defect_entry),
                    defect_entry.defect.structure,
                    symm_ops=bulk_supercell_symm_ops,
                    symprec=0.01,
                    dist_tol=0.01,
                )
            )

        # get orientational degeneracy
        relaxed_point_group, periodicity_breaking = point_symmetry_from_defect_entry(
            defect_entry,
            relaxed=True,
            verbose=False,
            return_periodicity_breaking=True,
            symprec=kwargs.get("symprec"),
        )  # relaxed so defect symm_ops
        bulk_site_point_group = point_symmetry_from_defect_entry(
            defect_entry,
            symm_ops=bulk_supercell_symm_ops,  # unrelaxed so bulk symm_ops
            relaxed=False,
            symprec=0.01,  # same symprec used w/interstitial multiplicity for consistency
        )
        with contextlib.suppress(ValueError):
            defect_entry.degeneracy_factors["orientational degeneracy"] = get_orientational_degeneracy(
                relaxed_point_group=relaxed_point_group,
                bulk_site_point_group=bulk_site_point_group,
            )
        defect_entry.calculation_metadata["relaxed point symmetry"] = relaxed_point_group
        defect_entry.calculation_metadata["bulk site symmetry"] = bulk_site_point_group
        defect_entry.calculation_metadata["periodicity_breaking_supercell"] = periodicity_breaking

        if bulk_voronoi_node_dict and bulk_path:  # save to bulk folder for future expedited parsing:
            if os.path.exists("voronoi_nodes.json.lock"):
                with FileLock("voronoi_nodes.json.lock"):
                    dumpfn(bulk_voronoi_node_dict, os.path.join(bulk_path, "voronoi_nodes.json"))
            else:
                dumpfn(bulk_voronoi_node_dict, os.path.join(bulk_path, "voronoi_nodes.json"))

        check_and_set_defect_entry_name(
            defect_entry, possible_defect_name, bulk_symm_ops=bulk_supercell_symm_ops
        )

        dp = cls(
            defect_entry,
            defect_vr=defect_vr,
            bulk_vr=bulk_vr,
            skip_corrections=skip_corrections,
            error_tolerance=error_tolerance,
            parse_projected_eigen=parse_projected_eigen,
            **kwargs,
        )

        if parse_projected_eigen is not False:
            try:
                dp.defect_entry._load_and_parse_eigenvalue_data(
                    bulk_vr=bulk_vr,
                    bulk_procar=bulk_procar,
                    defect_vr=defect_vr,
                    defect_procar=defect_procar,
                )
            except Exception as exc:
                if parse_projected_eigen is True:  # otherwise no warning
                    warnings.warn(f"Projected eigenvalues/orbitals parsing failed with error: {exc!r}")

        defect_vr.projected_eigenvalues = None  # no longer needed, delete to reduce memory demand
        defect_vr.eigenvalues = None  # no longer needed, delete to reduce memory demand
        dp.load_and_check_calculation_metadata()  # Load standard defect metadata
        dp.load_bulk_gap_data(bulk_band_gap_vr=bulk_band_gap_vr)  # Load band gap data

        if not skip_corrections and defect_entry.charge_state != 0:
            # no finite-size charge corrections by default for neutral defects
            skip_corrections = dp._check_and_load_appropriate_charge_correction()

        if not skip_corrections and defect_entry.charge_state != 0:
            try:
                dp.apply_corrections()
            except Exception as exc:
                warnings.warn(
                    f"Got this error message when attempting to apply finite-size charge corrections:"
                    f"\n{exc}\n"
                    f"-> Charge corrections will not be applied for this defect."
                )

            # check that charge corrections are not negative
            summed_corrections = sum(
                val
                for key, val in dp.defect_entry.corrections.items()
                if any(i in key.lower() for i in ["freysoldt", "kumagai", "fnv", "charge"])
            )
            if summed_corrections < -0.08:
                # usually unphysical for _isotropic_ dielectrics (suggests over-delocalised charge,
                # affecting the potential alignment)
                # how anisotropic is the dielectric?
                how_aniso = np.diag(
                    (dielectric - np.mean(np.diag(dielectric))) / np.mean(np.diag(dielectric))
                )
                if np.allclose(how_aniso, 0, atol=0.05):
                    warnings.warn(
                        f"The calculated finite-size charge corrections for defect at {defect_path} and "
                        f"bulk at {bulk_path} sum to a _negative_ value of {summed_corrections:.3f}. For "
                        f"relatively isotropic dielectrics (as is the case here) this is usually "
                        f"unphyical, and can indicate 'false charge state' behaviour (with the supercell "
                        f"charge occupying the band edge states and not localised at the defect), "
                        f"affecting the potential alignment, or some error/mismatch in the defect and "
                        f"bulk calculations. If this defect species is not stable in the formation "
                        f"energy diagram then this warning can usually be ignored, but if it is, "
                        f"you should double-check your calculations and parsed results!"
                    )

        return dp

    def _check_and_load_appropriate_charge_correction(self):
        skip_corrections = False
        dielectric = self.defect_entry.calculation_metadata["dielectric"]
        bulk_path = self.defect_entry.calculation_metadata["bulk_path"]
        defect_path = self.defect_entry.calculation_metadata["defect_path"]

        # determine charge correction to use, based on what output files are available (`LOCPOT`s or
        # `OUTCAR`s), and whether the supplied dielectric is isotropic or not
        def _check_folder_for_file_match(folder, filename):
            return any(
                filename.lower() in folder_filename.lower() for folder_filename in os.listdir(folder)
            )

        def _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(
            aniso_dielectric,
        ):
            return 3 / sum(1 / diagonal_elt for diagonal_elt in np.diag(aniso_dielectric))

        # check if dielectric (3x3 matrix) has diagonal elements that differ by more than 20%
        isotropic_dielectric = all(np.isclose(i, dielectric[0, 0], rtol=0.2) for i in np.diag(dielectric))

        # regardless, try parsing OUTCAR files first (quickest, more robust for cases where defect
        # charge is localised somewhat off the (auto-determined) defect site (e.g. split-interstitials
        # etc) and also works regardless of isotropic/anisotropic)
        if _check_folder_for_file_match(defect_path, "OUTCAR") and _check_folder_for_file_match(
            bulk_path, "OUTCAR"
        ):
            try:
                self.load_eFNV_data()
            except Exception as kumagai_exc:
                if _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
                    bulk_path, "LOCPOT"
                ):
                    try:
                        if not isotropic_dielectric:
                            # convert anisotropic dielectric to harmonic mean of the diagonal:
                            # (this is a better approximation than the pymatgen default of the
                            # standard arithmetic mean of the diagonal)
                            self.defect_entry.calculation_metadata["dielectric"] = (
                                _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                            )
                        self.load_FNV_data()
                        if not isotropic_dielectric:
                            warnings.warn(
                                _aniso_dielectric_but_outcar_problem_warning
                                + "in the defect or bulk folder were unable to be parsed, giving the "
                                "following error message:"
                                + f"\n{kumagai_exc}\n"
                                + _aniso_dielectric_but_using_locpot_warning
                            )
                    except Exception as freysoldt_exc:
                        warnings.warn(
                            f"Got this error message when attempting to parse defect & bulk `OUTCAR` "
                            f"files to compute the Kumagai (eFNV) charge correction:"
                            f"\n{kumagai_exc}\n"
                            f"Then got this error message when attempting to parse defect & bulk "
                            f"`LOCPOT` files to compute the Freysoldt (FNV) charge correction:"
                            f"\n{freysoldt_exc}\n"
                            f"-> Charge corrections will not be applied for this defect."
                        )
                        if not isotropic_dielectric:
                            # reset dielectric to original anisotropic value if FNV failed as well:
                            self.defect_entry.calculation_metadata["dielectric"] = dielectric
                        skip_corrections = True

                else:
                    warnings.warn(
                        f"`OUTCAR` files (needed to compute the Kumagai eFNV charge correction for "
                        f"_anisotropic_ and isotropic systems) in the defect or bulk folder were unable "
                        f"to be parsed, giving the following error message:"
                        f"\n{kumagai_exc}\n"
                        f"-> Charge corrections will not be applied for this defect."
                    )
                    skip_corrections = True

        elif _check_folder_for_file_match(defect_path, "LOCPOT") and _check_folder_for_file_match(
            bulk_path, "LOCPOT"
        ):
            try:
                if not isotropic_dielectric:
                    # convert anisotropic dielectric to harmonic mean of the diagonal:
                    # (this is a better approximation than the pymatgen default of the
                    # standard arithmetic mean of the diagonal)
                    self.defect_entry.calculation_metadata["dielectric"] = (
                        _convert_anisotropic_dielectric_to_isotropic_harmonic_mean(dielectric)
                    )
                self.load_FNV_data()
                if not isotropic_dielectric:
                    warnings.warn(
                        _aniso_dielectric_but_outcar_problem_warning
                        + "are missing from the defect or bulk folder.\n"
                        + _aniso_dielectric_but_using_locpot_warning
                    )
            except Exception as freysoldt_exc:
                warnings.warn(
                    f"Got this error message when attempting to parse defect & bulk `LOCPOT` files to "
                    f"compute the Freysoldt (FNV) charge correction:"
                    f"\n{freysoldt_exc}\n"
                    f"-> Charge corrections will not be applied for this defect."
                )
                if not isotropic_dielectric:
                    # reset dielectric to original anisotropic value if FNV failed as well:
                    self.defect_entry.calculation_metadata["dielectric"] = dielectric
                skip_corrections = True

        else:
            if int(self.defect_entry.charge_state) != 0:
                warnings.warn(
                    "`LOCPOT` or `OUTCAR` files are missing from the defect or bulk folder. "
                    "These are needed to perform the finite-size charge corrections. "
                    "Charge corrections will not be applied for this defect."
                )
                skip_corrections = True

        return skip_corrections

    def load_FNV_data(self, bulk_locpot_dict=None):
        """
        Load metadata required for performing Freysoldt correction (i.e. LOCPOT
        planar-averaged potential dictionary).

        Requires "bulk_path" and "defect_path" to be present in
        DefectEntry.calculation_metadata, and VASP LOCPOT files to be
        present in these directories. Can read compressed "LOCPOT.gz"
        files. The bulk_locpot_dict can be supplied if already parsed,
        for expedited parsing of multiple defects.

        Saves the ``bulk_locpot_dict`` and ``defect_locpot_dict`` dictionaries
        (containing the planar-averaged electrostatic potentials along each
        axis direction) to the DefectEntry.calculation_metadata dict, for
        use with DefectEntry.get_freysoldt_correction().

        Args:
            bulk_locpot_dict (dict): Planar-averaged potential dictionary
                for bulk supercell, if already parsed. If ``None`` (default),
                will load from ``LOCPOT(.gz)`` file in
                ``defect_entry.calculation_metadata["bulk_path"]``

        Returns:
            bulk_locpot_dict for reuse in parsing other defect entries
        """
        if not self.defect_entry.charge_state:
            # no charge correction if charge is zero
            return None

        bulk_locpot_dict = (
            bulk_locpot_dict
            or self.kwargs.get("bulk_locpot_dict", None)
            or _get_bulk_locpot_dict(self.defect_entry.calculation_metadata["bulk_path"])
        )

        defect_locpot_path, multiple = _get_output_files_and_check_if_multiple(
            "LOCPOT", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            _multiple_files_warning(
                "LOCPOT",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_locpot_path,
                dir_type="defect",
            )
        defect_locpot = get_locpot(defect_locpot_path)
        defect_locpot_dict = {str(k): defect_locpot.get_average_along_axis(k) for k in [0, 1, 2]}

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_locpot_dict": bulk_locpot_dict,
                "defect_locpot_dict": defect_locpot_dict,
            }
        )

        return bulk_locpot_dict

    def load_eFNV_data(self, bulk_site_potentials=None):
        """
        Load metadata required for performing Kumagai correction (i.e. atomic
        site potentials from the OUTCAR files).

        Requires "bulk_path" and "defect_path" to be present in
        DefectEntry.calculation_metadata, and VASP OUTCAR files to be
        present in these directories. Can read compressed "OUTCAR.gz"
        files. The bulk_site_potentials can be supplied if already
        parsed, for expedited parsing of multiple defects.

        Saves the ``bulk_site_potentials`` and ``defect_site_potentials``
        lists (containing the atomic site electrostatic potentials, from
        -1*np.array(Outcar.electrostatic_potential)) to
        DefectEntry.calculation_metadata, for use with
        DefectEntry.get_kumagai_correction().

        Args:
            bulk_site_potentials (dict): Atomic site potentials for the
                bulk supercell, if already parsed. If None (default), will
                load from OUTCAR(.gz) file in
                defect_entry.calculation_metadata["bulk_path"]

        Returns:
            bulk_site_potentials for reuse in parsing other defect entries
        """
        from doped.corrections import _raise_incomplete_outcar_error  # avoid circular import

        if not self.defect_entry.charge_state:
            # don't need to load outcars if charge is zero
            return None

        bulk_site_potentials = bulk_site_potentials or self.kwargs.get("bulk_site_potentials", None)
        if bulk_site_potentials is None:
            bulk_site_potentials = _get_bulk_site_potentials(
                self.defect_entry.calculation_metadata["bulk_path"]
            )

        defect_outcar_path, multiple = _get_output_files_and_check_if_multiple(
            "OUTCAR", self.defect_entry.calculation_metadata["defect_path"]
        )
        if multiple:
            _multiple_files_warning(
                "OUTCAR",
                self.defect_entry.calculation_metadata["defect_path"],
                defect_outcar_path,
                dir_type="defect",
            )
        defect_outcar = get_outcar(defect_outcar_path)

        if defect_outcar.electrostatic_potential is None:
            _raise_incomplete_outcar_error(defect_outcar_path, dir_type="defect")

        defect_site_potentials = -1 * np.array(defect_outcar.electrostatic_potential)

        self.defect_entry.calculation_metadata.update(
            {
                "bulk_site_potentials": bulk_site_potentials,
                "defect_site_potentials": defect_site_potentials,
            }
        )

        return bulk_site_potentials

    def load_and_check_calculation_metadata(self):
        """
        Pull metadata about the defect supercell calculations from the outputs,
        and check if the defect and bulk supercell calculations settings are
        compatible.
        """
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                _multiple_files_warning(
                    "vasprun.xml",
                    self.defect_entry.calculation_metadata["bulk_path"],
                    bulk_vr_path,
                    dir_type="bulk",
                )
            self.bulk_vr = _parse_vr_and_poss_procar(
                bulk_vr_path,
                parse_projected_eigen=False,  # not needed for DefectEntry metadata
                label="bulk",
                parse_procar=False,
            )

        if not self.defect_vr:
            defect_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["defect_path"]
            )
            if multiple:
                _multiple_files_warning(
                    "vasprun.xml",
                    self.defect_entry.calculation_metadata["defect_path"],
                    defect_vr_path,
                    dir_type="defect",
                )
            self.defect_vr = _parse_vr_and_poss_procar(
                defect_vr_path,
                parse_projected_eigen=False,  # not needed for DefectEntry metadata
                label="defect",
                parse_procar=False,
            )

        def _get_vr_dict_without_proj_eigenvalues(vr):
            proj_eigen = vr.projected_eigenvalues
            vr.projected_eigenvalues = None
            vr_dict = vr.as_dict()  # only call once
            vr_dict_wout_proj = {  # projected eigenvalue data might be present, but not needed (v slow
                # and data-heavy)
                **{k: v for k, v in vr_dict.items() if k != "output"},
                "output": {
                    k: v
                    for k, v in vr_dict["output"].items()
                    if k != "projected_eigenvalues"  # reduce memory demand
                },
            }
            vr_dict_wout_proj["output"]["projected_eigenvalues"] = None
            vr.projected_eigenvalues = proj_eigen  # reset to original value
            return vr_dict_wout_proj

        run_metadata = {
            # incars need to be as dict without module keys otherwise not JSONable:
            "defect_incar": {k: v for k, v in self.defect_vr.incar.as_dict().items() if "@" not in k},
            "bulk_incar": {k: v for k, v in self.bulk_vr.incar.as_dict().items() if "@" not in k},
            "defect_kpoints": self.defect_vr.kpoints,
            "bulk_kpoints": self.bulk_vr.kpoints,
            "defect_actual_kpoints": self.defect_vr.actual_kpoints,
            "bulk_actual_kpoints": self.bulk_vr.actual_kpoints,
            "defect_potcar_symbols": self.defect_vr.potcar_spec,
            "bulk_potcar_symbols": self.bulk_vr.potcar_spec,
            "defect_vasprun_dict": _get_vr_dict_without_proj_eigenvalues(self.defect_vr),
            "bulk_vasprun_dict": _get_vr_dict_without_proj_eigenvalues(self.bulk_vr),
        }

        self.defect_entry.calculation_metadata["mismatching_INCAR_tags"] = _compare_incar_tags(
            run_metadata["bulk_incar"], run_metadata["defect_incar"]
        )
        self.defect_entry.calculation_metadata["mismatching_POTCAR_symbols"] = _compare_potcar_symbols(
            run_metadata["bulk_potcar_symbols"], run_metadata["defect_potcar_symbols"]
        )
        self.defect_entry.calculation_metadata["mismatching_KPOINTS"] = _compare_kpoints(
            run_metadata["bulk_actual_kpoints"],
            run_metadata["defect_actual_kpoints"],
            run_metadata["bulk_kpoints"],
            run_metadata["defect_kpoints"],
        )

        self.defect_entry.calculation_metadata.update({"run_metadata": run_metadata.copy()})

    def load_bulk_gap_data(self, bulk_band_gap_vr=None, use_MP=False, mpid=None, api_key=None):
        r"""
        Load the ``"gap"`` and ``"vbm"`` values for the parsed
        ``DefectEntry``\s.

        If ``bulk_band_gap_vr`` is provided, then these values are parsed from it,
        else taken from the parsed bulk supercell calculation.

        Alternatively, one can specify query the Materials Project (MP) database
        for the bulk gap data, using ``use_MP = True``, in which case the MP entry
        with the lowest number ID and composition matching the bulk will be used,
        or the MP ID (``mpid``) of the bulk material to use can be specified. This
        is not recommended as it will correspond to a severely-underestimated GGA DFT
        bandgap!

        Args:
            bulk_band_gap_vr (str or Vasprun):
                Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` ``Vasprun``
                object, from which to determine the bulk band gap and band edge
                positions. If the VBM/CBM occur at `k`-points which are not included
                in the bulk supercell calculation, then this parameter should be used
                to provide the output of a bulk bandstructure calculation so that
                these are correctly determined.
                Alternatively, you can edit/add the ``"gap"`` and ``"vbm"`` entries in
                ``self.defect_entry.calculation_metadata`` to match the correct
                (eigen)values.
                If None, will use ``DefectEntry.calculation_metadata["bulk_path"]``
                (i.e. the bulk supercell calculation output).

                Note that the ``"gap"`` and ``"vbm"`` values should only affect the
                reference for the Fermi level values output by ``doped`` (as this VBM
                eigenvalue is used as the zero reference), thus affecting the position
                of the band edges in the defect formation energy plots and doping
                window / dopability limit functions, and the reference of the reported
                Fermi levels.
            use_MP (bool):
                If True, will query the Materials Project database for the bulk gap data.
            mpid (str):
                If provided, will query the Materials Project database for the bulk gap
                data, using this Materials Project ID.
            api_key (str):
                Materials API key to access database.
        """
        if not self.bulk_vr:
            bulk_vr_path, multiple = _get_output_files_and_check_if_multiple(
                "vasprun.xml", self.defect_entry.calculation_metadata["bulk_path"]
            )
            if multiple:
                warnings.warn(
                    f"Multiple `vasprun.xml` files found in bulk directory: "
                    f"{self.defect_entry.calculation_metadata['bulk_path']}. Using "
                    f"{os.path.basename(bulk_vr_path)} to {_vasp_file_parsing_action_dict['vasprun.xml']}."
                )
            self.bulk_vr = _parse_vr_and_poss_procar(
                bulk_vr_path,
                parse_projected_eigen=self.parse_projected_eigen,
                label="bulk",
                parse_procar=False,
            )

        bulk_sc_structure = self.bulk_vr.initial_structure
        band_gap, cbm, vbm, _ = self.bulk_vr.eigenvalue_band_properties
        gap_calculation_metadata = {}

        use_MP = use_MP or self.kwargs.get("use_MP", False)
        mpid = mpid or self.kwargs.get("mpid", None)
        api_key = api_key or self.kwargs.get("api_key", None)

        if use_MP and mpid is None:
            try:
                with MPRester(api_key=api_key) as mp:
                    tmp_mplist = mp.get_entries_in_chemsys(list(bulk_sc_structure.symbol_set))
                mplist = [
                    mp_ent.entry_id
                    for mp_ent in tmp_mplist
                    if mp_ent.composition.reduced_composition
                    == bulk_sc_structure.composition.reduced_composition
                ]
            except Exception as exc:
                raise ValueError(
                    f"Error with querying MPRester for"
                    f" {bulk_sc_structure.composition.reduced_formula}:"
                ) from exc

            mpid_fit_list = []
            for trial_mpid in mplist:
                with MPRester(api_key=api_key) as mp:
                    mpstruct = mp.get_structure_by_material_id(trial_mpid)
                if StructureMatcher(
                    primitive_cell=True,
                    scale=False,
                    attempt_supercell=True,
                    allow_subset=False,
                ).fit(bulk_sc_structure, mpstruct):
                    mpid_fit_list.append(trial_mpid)

            if len(mpid_fit_list) == 1:
                mpid = mpid_fit_list[0]
                print(f"Single mp-id found for bulk structure:{mpid}.")
            elif len(mpid_fit_list) > 1:
                num_mpid_list = [int(mp.split("-")[1]) for mp in mpid_fit_list]
                num_mpid_list.sort()
                mpid = f"mp-{num_mpid_list[0]!s}"
                print(
                    f"Multiple mp-ids found for bulk structure:{mpid_fit_list}. Will use lowest "
                    f"number mpid for bulk band structure = {mpid}."
                )
            else:
                print(
                    "Could not find bulk structure in MP database after tying the following "
                    f"list:\n{mplist}"
                )
                mpid = None

        if mpid is not None:
            print(f"Using user-provided mp-id for bulk structure: {mpid}.")
            with MPRester(api_key=api_key) as mp:
                bs = mp.get_bandstructure_by_material_id(mpid)
            if bs:
                cbm = bs.get_cbm()["energy"]
                vbm = bs.get_vbm()["energy"]
                band_gap = bs.get_band_gap()["energy"]
                gap_calculation_metadata["MP_gga_BScalc_data"] = bs.get_band_gap().copy()

        if (vbm is None or band_gap is None or cbm is None or not bulk_band_gap_vr) and (
            mpid and band_gap is None
        ):
            warnings.warn(
                f"MPID {mpid} was provided, but no bandstructure entry currently exists for it. "
                f"Reverting to use of bulk supercell calculation for band edge extrema."
            )
            gap_calculation_metadata["MP_gga_BScalc_data"] = None  # to signal no MP BS is used

        if bulk_band_gap_vr:
            if not isinstance(bulk_band_gap_vr, Vasprun):
                bulk_band_gap_vr = get_vasprun(bulk_band_gap_vr, parse_projected_eigen=False)

            band_gap, cbm, vbm, _ = bulk_band_gap_vr.eigenvalue_band_properties

        gap_calculation_metadata = {
            "mpid": mpid,
            "cbm": cbm,
            "vbm": vbm,
            "gap": band_gap,
        }
        self.defect_entry.calculation_metadata.update(gap_calculation_metadata)

    def apply_corrections(self):
        """
        Get defect corrections and warn if likely to be inappropriate.
        """
        if not self.defect_entry.charge_state:  # no charge correction if charge is zero
            return

        # try run Kumagai (eFNV) correction if required info available:
        if (
            self.defect_entry.calculation_metadata.get("bulk_site_potentials", None) is not None
            and self.defect_entry.calculation_metadata.get("defect_site_potentials", None) is not None
        ):
            self.defect_entry.get_kumagai_correction(verbose=False, error_tolerance=self.error_tolerance)

        elif self.defect_entry.calculation_metadata.get(
            "bulk_locpot_dict"
        ) and self.defect_entry.calculation_metadata.get("defect_locpot_dict"):
            self.defect_entry.get_freysoldt_correction(verbose=False, error_tolerance=self.error_tolerance)

        else:
            raise ValueError(
                "No charge correction performed! Missing required metadata in "
                "defect_entry.calculation_metadata ('bulk/defect_site_potentials' for Kumagai ("
                "eFNV) correction, or 'bulk/defect_locpot_dict' for Freysoldt (FNV) correction) - these "
                "are loaded with either the load_eFNV_data() or load_FNV_data() methods for "
                "DefectParser."
            )

        if (
            self.defect_entry.charge_state != 0
            and (not self.defect_entry.corrections or sum(self.defect_entry.corrections.values())) == 0
        ):
            warnings.warn(
                f"No charge correction computed for {self.defect_entry.name} with charge"
                f" {self.defect_entry.charge_state:+}, indicating problems with the required data for "
                f"the charge correction (i.e. dielectric constant, LOCPOT files for Freysoldt "
                f"correction, OUTCAR (with ICORELEVEL = 0) for Kumagai correction etc)."
            )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectParser`` object.
        """
        formula = self.bulk_vr.final_structure.composition.get_reduced_formula_and_factor(
            iupac_ordering=True
        )[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectParser for bulk composition {formula}. "
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )
