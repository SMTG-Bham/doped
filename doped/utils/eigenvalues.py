"""
Helper functions for setting up PHS analysis.

Contains modified versions of functions from pydefect (https://github.com/kumagai-group/pydefect)
and vise (https://github.com/kumagai-group/vise), to avoid requiring additional files (i.e. ``PROCAR``s).
"""

# suppress pydefect INFO messages
import logging
import os
import warnings
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Element, Species
from pymatgen.electronic_structure.core import Spin
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Procar, Vasprun
from shakenbreak.plotting import _install_custom_font

from doped import _ignore_pmg_warnings
from doped.core import DefectEntry
from doped.utils.parsing import get_magnetization_from_vasprun, get_nelect_from_vasprun, get_procar
from doped.utils.plotting import _get_backend

if TYPE_CHECKING:
    from easyunfold.procar import Procar as EasyunfoldProcar
    from pydefect.analyzer.make_defect_structure_info import DefectStructureInfo

try:
    import pydefect.analyzer.make_band_edge_states
    import pydefect.cli.vasp.make_band_edge_orbital_infos as make_bes
    from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, OrbitalInfo, PerfectBandEdgeState
    from pydefect.analyzer.eigenvalue_plotter import EigenvalueMplPlotter
    from pydefect.analyzer.make_band_edge_states import make_band_edge_states
    from pydefect.analyzer.make_defect_structure_info import MakeDefectStructureInfo
    from pydefect.cli.vasp.make_perfect_band_edge_state import get_edge_info
    from pydefect.defaults import defaults
    from pydefect.util.structure_tools import Coordination, Distances
    from vise import user_settings
    from vise.analyzer.vasp.band_edge_properties import BandEdgeProperties, eigenvalues_from_vasprun

    user_settings.logger.setLevel(logging.CRITICAL)

except ImportError as exc:
    raise ImportError(
        "To perform eigenvalue & orbital analysis, you need to install pydefect. "
        "You can do this by running `pip install pydefect`."
    ) from exc

# vise suppresses `UserWarning`s, so need to reset
warnings.simplefilter("default")
warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`")
warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")
_ignore_pmg_warnings()


def _coordination(self, include_on_site=True, cutoff_factor=None) -> "Coordination":
    cutoff_factor = cutoff_factor or defaults.cutoff_distance_factor
    cutoff = self.shortest_distance * cutoff_factor
    elements = [element.specie.name for element in self.structure]
    e_d = zip(elements, self.distances(remove_self=False))

    unsorted_distances = defaultdict(list)
    neighboring_atom_indices = []
    for i, (element, distance) in enumerate(e_d):
        if distance < cutoff and include_on_site:
            unsorted_distances[element].append(round(distance, 2))
            neighboring_atom_indices.append(i)

    distance_dict = {element: sorted(distances) for element, distances in unsorted_distances.items()}
    return Coordination(distance_dict, round(cutoff, 3), neighboring_atom_indices)


def _distances(self, remove_self=True, specie=None) -> list[float]:
    result = []
    lattice = self.structure.lattice
    if isinstance(specie, Element):
        el = specie.symbol
    elif specie is None:
        el = None
    elif isinstance(specie, Species):
        el = specie.element
    for site in self.structure:
        site_specie = site.specie.element if isinstance(site.specie, Species) else site.specie
        if el and Element(el) != site_specie:
            result.append(float("inf"))
            continue
        distance, _ = lattice.get_distance_and_image(site.frac_coords, self.coord)
        if remove_self and distance < 1e-5:
            continue
        result.append(distance)
    return result


def band_edge_properties_from_vasprun(
    vasprun: Vasprun, integer_criterion: float = 0.1
) -> BandEdgeProperties:
    """
    Create a ``pydefect`` ``BandEdgeProperties`` object from a ``Vasprun``
    object.

    Args:
        vasprun (Vasprun): ``Vasprun`` object.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or
            fully occupied (> 1 - ``integer_criterion``).
            Default is 0.1.

    Returns:
        ``BandEdgeProperties`` object.
    """
    band_edge_prop = BandEdgeProperties(
        eigenvalues=eigenvalues_from_vasprun(vasprun),
        nelect=get_nelect_from_vasprun(vasprun),
        magnetization=get_magnetization_from_vasprun(vasprun),
        kpoint_coords=vasprun.actual_kpoints,
        integer_criterion=integer_criterion,
    )
    band_edge_prop.structure = vasprun.final_structure
    return band_edge_prop


def make_perfect_band_edge_state_from_vasp(
    procar: Procar, vasprun: Vasprun, integer_criterion: float = 0.1
) -> PerfectBandEdgeState:
    """
    Create a ``pydefect`` ``PerfectBandEdgeState`` object from just a
    ``Vasprun`` and ``Procar`` object, without the need for the ``Outcar``
    input (as in ``pydefect``).

    Args:
        procar (Procar): ``Procar`` object.
        vasprun (Vasprun): ``Vasprun`` object.
        integer_criterion (float):
            Threshold criterion for determining if a band is unoccupied
            (< ``integer_criterion``), partially occupied (between
            ``integer_criterion`` and 1 - ``integer_criterion``), or
            fully occupied (> 1 - ``integer_criterion``).
            Default is 0.1.

    Returns:
        ``PerfectBandEdgeState`` object.
    """
    band_edge_prop = band_edge_properties_from_vasprun(vasprun, integer_criterion)
    orbs, s = procar.data, vasprun.final_structure
    vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, vasprun)
    cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, vasprun)
    return PerfectBandEdgeState(vbm_info, cbm_info)


def _make_band_edge_orbital_infos_vr(
    defect_vr: Vasprun, vbm: float, cbm: float, str_info: "DefectStructureInfo", eigval_shift: float = 0.0
):
    """
    Make ``BandEdgeOrbitalInfos`` from a ``Vasprun`` object.

    Modified from ``pydefect`` to use projected orbitals
    stored in the ``Vasprun`` object.

    Args:
        defect_vr (Vasprun): Defect ``Vasprun`` object.
        vbm (float): VBM eigenvalue in eV.
        cbm (float): CBM eigenvalue in eV.
        str_info (DefectStructureInfo): ``pydefect`` ``DefectStructureInfo``.
        eigval_shift (float): Shift eigenvalues down by this value (to set VBM at 0 eV).

    Returns:
        ``BandEdgeOrbitalInfos `` object
    """
    eigval_range = defaults.eigval_range
    kpt_coords = [tuple(coord) for coord in defect_vr.actual_kpoints]
    max_energy_by_spin, min_energy_by_spin = [], []
    neighbors = str_info.neighbor_atom_indices

    for e in defect_vr.eigenvalues.values():
        max_energy_by_spin.append(np.amax(e[:, :, 0], axis=0))
        min_energy_by_spin.append(np.amin(e[:, :, 0], axis=0))

    max_energy_by_band = np.amax(np.vstack(max_energy_by_spin), axis=0)
    min_energy_by_band = np.amin(np.vstack(min_energy_by_spin), axis=0)

    lower_idx = np.argwhere(max_energy_by_band > vbm - eigval_range)[0][0]
    upper_idx = np.argwhere(min_energy_by_band < cbm + eigval_range)[-1][-1]

    orbs, s = defect_vr.projected_eigenvalues, defect_vr.final_structure
    orb_infos: list[Any] = []
    for spin, eigvals in defect_vr.eigenvalues.items():
        orb_infos.append([])
        for k_idx in range(len(kpt_coords)):
            orb_infos[-1].append([])
            for b_idx in range(lower_idx, upper_idx + 1):
                e, occ = eigvals[k_idx, b_idx, :]
                orbitals = make_bes.calc_orbital_character(orbs, s, spin, k_idx, b_idx)
                if neighbors:
                    p_ratio = make_bes.calc_participation_ratio(orbs, spin, k_idx, b_idx, neighbors)
                else:
                    p_ratio = None
                orb_infos[-1][-1].append(OrbitalInfo(e, orbitals, occ, p_ratio))

    return BandEdgeOrbitalInfos(
        orbital_infos=orb_infos,
        kpt_coords=kpt_coords,
        kpt_weights=defect_vr.actual_kpoints_weights,
        lowest_band_index=int(lower_idx),
        fermi_level=defect_vr.efermi,
        eigval_shift=eigval_shift,
    )


def _parse_procar(procar: Union[str, "Path", "EasyunfoldProcar", Procar]):
    """
    Parse a ``procar`` input to a ``Procar`` object in the correct format.

    Args:
        procar (str, Path, EasyunfoldProcar, Procar):
            Either a path to the ``VASP`` ``PROCAR``` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/``pymatgen``
            ``Procar`` object.
    """
    if not hasattr(procar, "data"):  # not a parsed Procar object
        if hasattr(procar, "proj_data") and not isinstance(procar, (str, Path, Procar)):
            if procar._is_soc:
                procar.data = {Spin.up: procar.proj_data[0]}
            else:
                procar.data = {Spin.up: procar.proj_data[0], Spin.down: procar.proj_data[1]}
            del procar.proj_data

        else:  # path to PROCAR file
            procar = get_procar(procar)

    return procar


def get_band_edge_info(
    bulk_vr: Vasprun,
    defect_vr: Vasprun,
    bulk_procar: Optional[Union[str, "Path", "EasyunfoldProcar", Procar]] = None,
    defect_procar: Optional[Union[str, "Path", "EasyunfoldProcar", Procar]] = None,
):
    """
    Generate metadata required for performing eigenvalue & orbital analysis,
    specifically ``pydefect`` ``BandEdgeOrbitalInfos``, and ``EdgeInfo``
    objects for the bulk VBM and CBM.

    See https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states.

    Args:
        bulk_vr (Vasprun):
            ``Vasprun`` object of the bulk supercell calculation.
            If ``bulk_procar`` is not provided, then this must have the
            ``projected_eigenvalues`` attribute (i.e. from a calculation
            with ``LORBIT > 10`` in the ``INCAR`` and parsed with
            ``parse_projected_eigen = True``).
        defect_vr (Vasprun):
            ``Vasprun`` object of the defect supercell calculation.
            If ``defect_procar`` is not provided, then this must have the
            ``projected_eigenvalues`` attribute (i.e. from a calculation
            with ``LORBIT > 10`` in the ``INCAR`` and parsed with
            ``parse_projected_eigen = True``).
        bulk_procar (str, Path, EasyunfoldProcar, Procar):
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the bulk supercell
            calculation. Not required if the supplied ``bulk_vr`` was
            parsed with ``parse_projected_eigen = True``.
            Default is ``None``.
        defect_procar (str, Path, EasyunfoldProcar, Procar):
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the defect supercell
            calculation. Not required if the supplied ``bulk_vr`` was
            parsed with ``parse_projected_eigen = True``.
            Default is ``None``.

    Returns:
        ``pydefect`` ``BandEdgeOrbitalInfos``, and ``EdgeInfo`` objects
        for the bulk VBM and CBM.
    """
    band_edge_prop = BandEdgeProperties(
        eigenvalues=eigenvalues_from_vasprun(bulk_vr),
        nelect=get_nelect_from_vasprun(bulk_vr),
        magnetization=get_magnetization_from_vasprun(bulk_vr),
        kpoint_coords=bulk_vr.actual_kpoints,
        integer_criterion=0.1,  # default value in vise
    )
    band_edge_prop.structure = bulk_vr.final_structure

    if bulk_procar is not None:
        bulk_procar = _parse_procar(bulk_procar)
        pbes = make_perfect_band_edge_state_from_vasp(bulk_procar, bulk_vr)

    _orig_method_coor = Distances.coordination
    Distances.coordination = _coordination
    _orig_method_dist = Distances.distances
    Distances.distances = _distances

    dsinfo = MakeDefectStructureInfo(  # Using default values suggested by pydefect
        bulk_vr.final_structure,
        defect_vr.final_structure,
        defect_vr.final_structure,
        symprec=0.1,
        dist_tol=1.0,
        neighbor_cutoff_factor=1.3,  # Neighbors are sites within min_dist * neighbor_cutoff_factor
    )

    # Undo monkey patch in case used in other parts of the code:
    Distances.coordination = _orig_method_coor
    Distances.distances = _orig_method_dist

    from pydefect.analyzer.band_edge_states import logger

    logger.setLevel(logging.CRITICAL)  # quieten unnecessary eigenvalue shift INFO message

    if bulk_procar is not None:
        vbm_info, cbm_info = pbes.vbm_info, pbes.cbm_info
    else:
        orbs, s = bulk_vr.projected_eigenvalues, bulk_vr.final_structure
        vbm_info = get_edge_info(band_edge_prop.vbm_info, orbs, s, bulk_vr)
        cbm_info = get_edge_info(band_edge_prop.cbm_info, orbs, s, bulk_vr)

    if defect_procar is not None:
        defect_procar = _parse_procar(defect_procar)
        band_orb = make_bes.make_band_edge_orbital_infos(
            defect_procar,
            defect_vr,
            vbm_info.orbital_info.energy,
            cbm_info.orbital_info.energy,
            eigval_shift=-vbm_info.orbital_info.energy,
            str_info=dsinfo.defect_structure_info,
        )

    else:
        band_orb = _make_band_edge_orbital_infos_vr(
            defect_vr,
            vbm_info.orbital_info.energy,
            cbm_info.orbital_info.energy,
            eigval_shift=-vbm_info.orbital_info.energy,
            str_info=dsinfo.defect_structure_info,
        )

    return band_orb, vbm_info, cbm_info


def get_eigenvalue_analysis(
    defect_entry: Optional[DefectEntry] = None,
    plot: bool = True,
    filename: Optional[str] = None,
    ks_labels: bool = False,
    style_file: Optional[str] = None,
    bulk_vr: Optional[Union[str, "Path", Vasprun]] = None,
    bulk_procar: Optional[Union[str, "Path", "EasyunfoldProcar", Procar]] = None,
    defect_vr: Optional[Union[str, "Path", Vasprun]] = None,
    defect_procar: Optional[Union[str, "Path", "EasyunfoldProcar", Procar]] = None,
    force_reparse: bool = False,
):
    r"""
    Get eigenvalue & orbital info (with automated classification of PHS states)
    for the band edge and in-gap electronic states for the input defect entry /
    calculation outputs, as well as a plot of the single-particle electronic
    eigenvalues and their occupation (if ``plot=True``).

    Can be used to determine if a defect is adopting a perturbed host
    state (PHS / shallow state), see
    https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states.

    Either a ``doped`` ``DefectEntry`` object can be provided, or the required
    VASP output files/objects for the bulk and defect supercell calculations
    (``Vasprun``\s and ``Procar``\s).
    If a ``DefectEntry`` is provided but eigenvalue data has not already been
    parsed (default in ``doped`` is to parse this data with ``DefectsParser``/
    ``DefectParser``, as controlled by the ``parse_projected_eigen`` flag),
    then this function will attempt to load the eigenvalue data from either
    the input ``Vasprun``/``PROCAR`` objects or files, or from the
    ``bulk/defect_path``\s in ``defect_entry.calculation_metadata``.
    If so, will initially try to load orbital projections from ``PROCAR(.gz)``
    files if present, otherwise will attempt to load from ``vasprun.xml(.gz)``
    (typically slower).

    This function uses code from ``pydefect``:
    Citation: https://doi.org/10.1103/PhysRevMaterials.5.123803.

    Args:
        defect_entry (DefectEntry):
            ``doped`` ``DefectEntry`` object. Default is ``None``.
        plot (bool):
            Whether to plot the single-particle eigenvalues.
            (Default: True)
        filename (str):
            Filename to save the eigenvalue plot to (if ``plot = True``).
            If ``None`` (default), plots are not saved.
        ks_labels (bool):
            Whether to add band index labels to the KS levels.
            (Default: False)
        style_file (str):
            Path to a ``mplstyle`` file to use for the plot. If None
            (default), uses the ``doped`` displacement plot style
            (``doped/utils/displacement.mplstyle``).
        bulk_vr (str, Path, Vasprun):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``vasprun.xml(.gz)`` output file
            or a ``pymatgen`` ``Vasprun`` object, for the reference bulk
            supercell calculation. If ``None`` (default), tries to load
            the ``Vasprun`` object from
            ``defect_entry.calculation_metadata["run_metadata"]["bulk_vasprun_dict"]``,
            or, failing that, from a ``vasprun.xml(.gz)`` file at
            ``defect_entry.calculation_metadata["bulk_path"]``.
        bulk_procar (str, Path, EasyunfoldProcar, Procar):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the reference bulk supercell
            calculation. Not required, but speeds up parsing (~50%) if present.
            If ``None`` (default), tries to load from a ``PROCAR(.gz)``
            file at ``defect_entry.calculation_metadata["bulk_path"]``.
        defect_vr (str, Path, Vasprun):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``vasprun.xml(.gz)`` output file
            or a ``pymatgen`` ``Vasprun`` object, for the defect supercell
            calculation. If ``None`` (default), tries to load the ``Vasprun``
            object from
            ``defect_entry.calculation_metadata["run_metadata"]["defect_vasprun_dict"]``,
            or, failing that, from a ``vasprun.xml(.gz)`` file at
            ``defect_entry.calculation_metadata["defect_path"]``.
        defect_procar (str, Path, EasyunfoldProcar, Procar):
            Not required if ``defect_entry`` provided and eigenvalue data
            already parsed (default behaviour when parsing with ``doped``,
            data in ``defect_entry.calculation_metadata["eigenvalue_data"]``).
            Either a path to the ``VASP`` ``PROCAR`` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or an ``easyunfold``/
            ``pymatgen`` ``Procar`` object, for the defect supercell calculation.
            Not required, but speeds up parsing (~50%) if present. If ``None``
            (default), tries to load from a ``PROCAR(.gz)`` file at
            ``defect_entry.calculation_metadata["defect_path"]``.
        force_reparse (bool):
            Whether to force re-parsing of the eigenvalue data, even if
            already present in the ``calculation_metadata``.

    Returns:
        ``pydefect`` ``PerfectBandEdgeState`` class
    """
    if defect_entry is None:
        if not all([bulk_vr, defect_vr, bulk_procar, defect_procar]):
            raise ValueError(
                "If `defect_entry` is not provided, then all of `bulk_vr`, `defect_vr`, "
                "`bulk_procar`, and `defect_procar` must be provided!"
            )
        from doped.analysis import defect_from_structures

        bulk_vr = bulk_vr if isinstance(bulk_vr, Vasprun) else Vasprun(bulk_vr)
        defect_vr = defect_vr if isinstance(defect_vr, Vasprun) else Vasprun(defect_vr)

        (
            defect,
            defect_site,
            defect_site_in_bulk,  # bulk site for vac/sub, relaxed defect site w/interstitials
            defect_site_index,  # in this initial_defect_structure
            bulk_site_index,
            guessed_initial_defect_structure,
            unrelaxed_defect_structure,
            bulk_voronoi_node_dict,
        ) = defect_from_structures(
            bulk_vr.final_structure,
            defect_vr.final_structure,
            oxi_state="Undetermined",
            return_all_info=True,
        )
        defect_entry = DefectEntry(
            # pmg attributes:
            defect=defect,  # this corresponds to _unrelaxed_ defect
            charge_state=0,
            sc_entry=ComputedStructureEntry(
                structure=defect_vr.final_structure,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
            bulk_entry=None,
            # doped attributes:
            defect_supercell_site=defect_site,  # _relaxed_ defect site
            defect_supercell=defect_vr.final_structure,
            bulk_supercell=bulk_vr.final_structure,
        )

    defect_entry._load_and_parse_eigenvalue_data(
        bulk_vr=bulk_vr,
        defect_vr=defect_vr,
        bulk_procar=bulk_procar,
        defect_procar=defect_procar,
        force_reparse=force_reparse,
    )

    band_orb = defect_entry.calculation_metadata["eigenvalue_data"]["band_orb"]
    vbm_info = defect_entry.calculation_metadata["eigenvalue_data"]["vbm_info"]
    cbm_info = defect_entry.calculation_metadata["eigenvalue_data"]["cbm_info"]

    # Ensures consistent number of sig. fig. so test work 100% of the time
    def _orbital_diff(orbital_1: dict, orbital_2: dict) -> float:
        element_set = set(list(orbital_1.keys()) + list(orbital_2.keys()))
        orb_1, orb_2 = defaultdict(list, orbital_1), defaultdict(list, orbital_2)
        result = 0
        for e in element_set:
            result += sum(abs(i - j) for i, j in zip_longest(orb_1[e], orb_2[e], fillvalue=0))
        return round(result, 3)

    pydefect.analyzer.make_band_edge_states.orbital_diff = _orbital_diff

    perfect = PerfectBandEdgeState(vbm_info, cbm_info)
    bes = make_band_edge_states(band_orb, perfect)

    if not plot:
        return bes

    vbm = vbm_info.orbital_info.energy + band_orb.eigval_shift
    cbm = cbm_info.orbital_info.energy + band_orb.eigval_shift

    _install_custom_font()  # in case not installed already
    style_file = style_file or f"{os.path.dirname(__file__)}/displacement.mplstyle"
    plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter

    emp = EigenvalueMplPlotter(
        title="Eigenvalues",
        band_edge_orb_infos=band_orb,
        supercell_vbm=vbm,
        supercell_cbm=cbm,
        y_range=[vbm - 3, cbm + 3],
    )

    with plt.style.context(style_file):
        plt.rcParams["axes.titlesize"] = 12
        plt.rc("axes", unicode_minus=False)

        emp.construct_plot()
        partial = None

        # Change colors to match sumo and doped conventions
        for a in range(len(emp.axs)):
            for i in range(len(emp.axs[a].get_children())):
                if hasattr(emp.axs[a].get_children()[i], "get_facecolor"):
                    if np.array_equal(
                        emp.axs[a].get_children()[i].get_facecolor(), [[1, 0, 0, 1]]
                    ):  # Check for red color
                        emp.axs[a].get_children()[i].set_facecolor((0.22, 0.325, 0.643))
                        emp.axs[a].get_children()[i].set_edgecolor((0.22, 0.325, 0.643))
                    elif np.array_equal(emp.axs[a].get_children()[i].get_facecolor(), [[0, 0, 1, 1]]):
                        emp.axs[a].get_children()[i].set_facecolor((0.98, 0.639, 0.086))
                        emp.axs[a].get_children()[i].set_edgecolor((0.98, 0.639, 0.086))
                    elif np.array_equal(emp.axs[a].get_children()[i].get_facecolor(), [[0, 0.5, 0, 1]]):
                        partial = True

        for axes in emp.axs:
            annotations = [child for child in axes.get_children() if isinstance(child, plt.Annotation)]
            for annotation in annotations:
                if (ks_labels and annotation.get_position()[0] > 1) or not ks_labels:
                    annotation.remove()

        if len(emp.axs) > 1:
            emp.axs[0].set_title("Spin Down")
            emp.axs[1].set_title("Spin Up")
        else:
            emp.axs[0].set_title("KS levels")

        ymin, ymax = 0, 0
        for spin in emp._energies_and_occupations:
            for kpoint in spin:
                ymin = min(ymin, *(x[0] for x in kpoint))
                ymax = max(ymax, *(x[0] for x in kpoint))

        gamma_check = "\N{GREEK CAPITAL LETTER GAMMA}"
        for ax in emp.axs:
            labels = ax.get_xticklabels()
            labels = [label.get_text() for label in labels]
            for i, label in enumerate(labels):
                if gamma_check in label:
                    labels[i] = r"$\Gamma$"
            ax.set_xticklabels(labels)

        fig = emp.plt.gcf()
        ax = fig.gca()

        ax.set_ylim([ymin - 0.25, ymax + 0.75])
        # add a point at 0,-5 with the color range and label unoccopied states
        ax.scatter(0, -5, label="Unoccupied", color=(0.98, 0.639, 0.086))
        ax.scatter(0, -5, label="Occupied", color=(0.22, 0.325, 0.643))
        if partial:
            ax.scatter(0, -5, label="Partially Occupied", color=(0, 0.5, 0))
        ax.axhline(-5, 0, 1, color="black", linewidth=0.5, linestyle="-.", label="Band Edges")
        ax.legend(loc="upper right", fontsize=7)

    if filename:
        emp.plt.savefig(filename, bbox_inches="tight", transparent=True, backend=_get_backend(filename))

    return bes, fig
