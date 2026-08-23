"""
Deprecated module; ``doped.utils.parsing`` has been dissolved.

Its contents now live in: ``doped.io.vasp.outputs`` (VASP calculation output
parsing), ``doped.utils.mappings`` (defect identification & site-mapping
utilities), ``doped.core`` (``DefectEntry`` structure accessors) and
``doped.utils.symmetry`` (electron-count / spin degeneracy helpers). This
shim forwards the old public names with deprecation warnings (private
helpers are not aliased), and will be removed in a future release.

Note that these names did `not` move to the (unrelated, top-level)
``doped.parsing`` module, which is the renamed ``doped.analysis`` (defect
calculation parsing).
"""

import warnings
from importlib import import_module
from typing import Any

_MOVED = {
    **dict.fromkeys(
        (
            "find_archived_fname",
            "parse_projected_eigen",
            "get_vasprun",
            "get_locpot",
            "get_outcar",
            "get_core_potentials_from_outcar",
            "get_procar",
            "get_magnetization_from_vasprun",
            "get_nelect_from_vasprun",
            "get_neutral_nelect_from_vasprun",
            "spin_degeneracy_from_vasprun",
            "total_charge_from_vasprun",
        ),
        "doped.io.vasp.outputs",
    ),
    **dict.fromkeys(
        (
            "get_defect_type_and_composition_diff",
            "get_defect_type_and_site_indices",
            "get_coords_and_idx_of_species",
            "get_matching_site",
            "get_wigner_seitz_radius",
            "check_atom_mapping_far_from_defect",
            "get_site_mappings",
            "reorder_s2_like_s1",
            "get_dimer_bonds",
        ),
        "doped.utils.mappings",
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
