"""
Deprecated module; ``doped.utils.parsing`` has been dissolved.

Its contents now live in: ``doped.io.vasp.outputs`` (VASP calculation output
parsing), ``doped.utils.mappings`` (defect identification & site-mapping
utilities), ``doped.core`` (``DefectEntry`` structure accessors) and
``doped.utils.symmetry`` (electron-count / spin degeneracy helpers). This
shim forwards the old names with deprecation warnings, and will be removed in
a future release.
"""

import warnings
from importlib import import_module
from typing import Any

_MOVED = {
    **dict.fromkeys(
        (
            "_get_potcar_summary_stats",
            "find_archived_fname",
            "parse_projected_eigen",
            "get_vasprun",
            "get_locpot",
            "_get_outcar_path",
            "get_outcar",
            "get_core_potentials_from_outcar",
            "_get_final_energy_from_outcar",
            "_get_core_potentials_from_outcar_obj",
            "_check_outcar_energy",
            "_raise_incomplete_outcar_error",
            "get_procar",
            "_get_output_files_and_check_if_multiple",
            "_dataframe_of_files",
            "_get_calc_files_df",
            "_determine_subfolder",
            "_find_calc_outputs",
            "_compare_potcar_symbols",
            "_compare_kpoints",
            "_compare_incar_tags",
            "_format_mismatching_incar_warning",
            "get_magnetization_from_vasprun",
            "get_nelect_from_vasprun",
            "get_neutral_nelect_from_vasprun",
            "spin_degeneracy_from_vasprun",
            "total_charge_from_vasprun",
            "_get_bulk_locpot_dict",
            "_get_bulk_site_potentials",
            "_vasp_file_parsing_action_dict",
            "_multiple_files_warning",
        ),
        "doped.io.vasp.outputs",
    ),
    **dict.fromkeys(
        (
            "get_defect_type_and_composition_diff",
            "get_defect_type_and_site_indices",
            "get_coords_and_idx_of_species",
            "get_matching_site",
            "_create_unrelaxed_defect_structure",
            "get_wigner_seitz_radius",
            "check_atom_mapping_far_from_defect",
            "_get_site_mapping_from_coords_and_indices",
            "get_site_mappings",
            "reorder_s2_like_s1",
            "get_dimer_bonds",
        ),
        "doped.utils.mappings",
    ),
    **dict.fromkeys(
        (
            "_get_bulk_supercell",
            "_get_defect_supercell",
            "_get_defect_supercell_frac_coords",
            "_get_defect_supercell_site",
            "_update_defect_entry_structure_metadata",
        ),
        "doped.core",
    ),
    **dict.fromkeys(
        ("_num_electrons_from_charge_state", "_simple_spin_degeneracy_from_num_electrons"),
        "doped.utils.symmetry",
    ),
}


def __getattr__(name: str) -> Any:
    """
    Forward the dissolved ``doped.utils.parsing`` names to their new homes,
    with a deprecation warning.
    """
    if target := _MOVED.get(name):
        warnings.warn(
            f"{name} has moved to {target}; import it from there instead. This deprecated alias will "
            "be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )  # TODO: Remove this deprecation shim in future
        return getattr(import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
