"""
Tests for the ``doped.utils.stenciling`` module.
"""

import os
import unittest
import warnings
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.structure_matcher import ElementComparator
from pymatgen.core.structure import PeriodicSite
from test_utils import (
    EXAMPLE_DIR,
    STYLE,
    _potcars_available,
    _print_warning_info,
    custom_mpl_image_compare,
)

from doped.analysis import defect_site_from_structures
from doped.core import DefectEntry
from doped.thermodynamics import DefectThermodynamics
from doped.utils.displacements import plot_site_displacements

# use doped efficiency functions for speed in structure-matching testing
from doped.utils.efficiency import (
    Structure,
    StructureMatcher_scan_stol,
    get_element_min_max_bond_length_dict,
)
from doped.utils.parsing import check_atom_mapping_far_from_defect, get_defect_type_and_composition_diff
from doped.utils.stenciling import get_defect_in_supercell
from doped.utils.supercells import min_dist

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally

_DISP_STYLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../doped/utils/displacement.mplstyle"
)

# TODO: Decide optimal dist tol factor choices
# TODO: Add / run some quick energy tests (e.g. Madelung, from split-vac code); or some invariant test?
# TODO: After stenciling test updates; redo pytest split timings
# TODO: Useful test case could be trying to stencil split vacancies in new supercells...


def _get_sorted_nn_distances(structure, frac_coords, n_neighbours=12):
    """
    Get the sorted nearest-neighbour distances around a given fractional
    coordinate position in a structure.

    Returns the ``n_neighbours`` shortest distances from the position to
    atoms in the structure (accounting for PBC).
    """
    all_dists = structure.lattice.get_all_distances(structure.frac_coords, [frac_coords]).ravel()
    # exclude zero distances (for interstitial/substitution defects)
    return np.sort(all_dists[all_dists > 1e-2])[:n_neighbours]


def _plot_stenciled_vs_original_displacements(stenciled_entry: DefectEntry, original_entry: DefectEntry):
    """
    Create a 1x2 figure comparing site displacements of a stenciled defect
    supercell (left) against the original DFT defect entry (right).
    """
    with plt.style.context(_DISP_STYLE):
        styled_fig_size = plt.rcParams["figure.figsize"]
        styled_font_size = plt.rcParams["font.size"]
        fig, axes = plt.subplots(
            1, 2, figsize=(2 * styled_fig_size[0], styled_fig_size[1]), sharey=True, sharex=False
        )
        for ax, entry, title in zip(
            axes, [stenciled_entry, original_entry], ["Stenciled", "Original DFT"], strict=False
        ):
            plot_site_displacements(entry, ax=ax, style_file=_DISP_STYLE)
            ax.set_title(title, fontsize=styled_font_size)
        axes[1].set_ylabel("")
        axes[1].get_legend().remove()
        fig.subplots_adjust(wspace=0.15)
    return fig


def _make_stenciled_defect_entry(
    defect_entry: DefectEntry, stenciled_supercell: Structure, corresponding_bulk: Structure
) -> DefectEntry:
    """
    Create a copy of ``defect_entry`` with its ``defect_supercell`` and
    ``bulk_supercell`` replaced by the stenciled structures, for use with
    displacement plotting functions.

    Args:
        defect_entry (DefectEntry):
            Original ``DefectEntry`` (with DFT structures).
        stenciled_supercell (Structure):
            Stenciled defect supercell from ``get_defect_in_supercell``.
        corresponding_bulk (Structure):
            Corresponding bulk supercell from ``get_defect_in_supercell``.

    Returns:
        A deepcopy of ``defect_entry`` with updated supercell structures and
        defect site fractional coordinates.
    """
    stenciled_entry = deepcopy(defect_entry)
    stenciled_entry.defect_supercell = stenciled_supercell
    stenciled_entry.bulk_supercell = corresponding_bulk
    (
        defect_site,
        _defect_type,
        defect_site_in_bulk,
        _defect_site_index,
        bulk_site_index,
        _unrelaxed_defect_structure,
    ) = defect_site_from_structures(corresponding_bulk, stenciled_supercell, return_all_info=True)
    stenciled_entry.defect_supercell_site = defect_site
    stenciled_entry.sc_defect_frac_coords = defect_site.frac_coords
    # pop any previously-calculated site displacement data:
    stenciled_entry.calculation_metadata.pop("site_displacements", None)
    stenciled_entry.calculation_metadata["bulk_site"] = (
        defect_site_in_bulk
        if bulk_site_index is None  # interstitial
        else corresponding_bulk[bulk_site_index]
    )
    return stenciled_entry


def _validate_stenciled_supercell(
    stenciled_supercell: Structure,
    defect_entry: DefectEntry,
    target_supercell: Structure,
    corresponding_bulk: Structure,
    min_dist_tol_factor: float = 0.85,
    check_exact_bulk_match: bool = False,
):
    """
    Validate the physical correctness of a stenciled defect supercell using
    checks that are invariant to the structural coordinate basis (if
    ``check_exact_bulk_match`` is ``False``).

    Args:
        stenciled_supercell (Structure):
            The generated stenciled defect supercell.
        defect_entry (DefectEntry):
            The original ``DefectEntry``.
        target_supercell (Structure):
            The target bulk supercell structure.
        corresponding_bulk (Structure):
            The bulk supercell, corresponding to the stenciled supercell,
            returned by ``get_defect_in_supercell``.
        min_dist_tol_factor (float):
            Tolerance factor for minimum bond length. Default = 0.85.
        check_exact_bulk_match (bool):
            Whether to check for an exact match (without reduction to the
            primitive cell) between the output bulk supercell and the target
            supercell. Default = ``False``.
    """
    orig_supercell = defect_entry.defect_supercell
    orig_defect_frac_coords = defect_entry.sc_defect_frac_coords
    bulk_min_bond_length = min_dist(defect_entry.bulk_supercell)
    bulk_min_dist_tol = bulk_min_bond_length * min_dist_tol_factor

    # 1. Lattice check: stenciled supercell lattice matches target
    np.testing.assert_allclose(
        stenciled_supercell.lattice.matrix,
        target_supercell.lattice.matrix,
        atol=1e-4,
        err_msg="Stenciled supercell lattice does not match target supercell lattice",
    )

    # 2. Composition check: defect type and composition difference are correct
    defect_type, comp_diff = get_defect_type_and_composition_diff(target_supercell, stenciled_supercell)
    orig_defect_type, orig_comp_diff = get_defect_type_and_composition_diff(
        defect_entry.bulk_supercell, orig_supercell
    )
    assert defect_type == orig_defect_type, f"Defect type mismatch: {defect_type} != {orig_defect_type}"
    assert comp_diff == orig_comp_diff, f"Composition diff mismatch: {comp_diff} != {orig_comp_diff}"

    # 3. Min bond length preservation: stenciled min dist >= original * tolerance
    stenciled_min_dist = min_dist(stenciled_supercell)
    orig_defect_min_dist = min_dist(orig_supercell)
    assert stenciled_min_dist >= min(orig_defect_min_dist, bulk_min_dist_tol) * 0.999, (
        f"Stenciled min dist ({stenciled_min_dist:.3f} Å) < original "
        f"({orig_defect_min_dist:.3f} Å) and bulk min dist tol "
        f"({bulk_min_bond_length:.3f} Å * {min_dist_tol_factor})"
    )  # we multiply min by 0.999 to account for small numerical differences

    stenciled_defect_site = defect_site_from_structures(corresponding_bulk, stenciled_supercell)
    assert isinstance(stenciled_defect_site, PeriodicSite)  # typing
    stenciled_defect_frac_coords = stenciled_defect_site.frac_coords

    # 4. Check atom displacements vs corresponding_bulk, away from defect site:
    assert check_atom_mapping_far_from_defect(
        bulk_supercell=corresponding_bulk,
        defect_supercell=stenciled_supercell,
        defect_coords=stenciled_defect_frac_coords,
        displacement_tol=0.25,  # less than 0.25 Å displacement from bulk site, outside of defect WZ region
    )

    # 5. Defect nearest-neighbour distances preserved:
    # Find the actual defect position in the stenciled supercell by comparing with the corresponding
    # bulk supercell:
    orig_nn_dists = _get_sorted_nn_distances(orig_supercell, orig_defect_frac_coords)
    stenciled_nn_dists = _get_sorted_nn_distances(stenciled_supercell, stenciled_defect_frac_coords)
    # The first few NN (=12 by default here) distances from the defect site should be preserved,
    # as this local geometry should be effectively fixed (assuming sufficiently large target supercell)
    np.testing.assert_allclose(
        stenciled_nn_dists,
        orig_nn_dists,
        atol=0.01,
        err_msg="Core defect nearest-neighbour distances not preserved",
    )

    # 6. Bulk supercell validation: corresponding_bulk should match target
    assert StructureMatcher_scan_stol(
        target_supercell,
        corresponding_bulk,
        "fit",
        max_stol=0.02,
        comparator=ElementComparator(),
        primitive_cell=True,  # should always match when allowing reduction to primitive cell
    ), "Corresponding bulk does not match target bulk supercell"

    if check_exact_bulk_match:  # check if the bulk supercells are the exact same (same basis definition)
        assert StructureMatcher_scan_stol(
            target_supercell,
            corresponding_bulk,
            "fit",
            max_stol=0.02,
            comparator=ElementComparator(),
            primitive_cell=False,  # check match of primitive cell tiling in the supercells
        ), "Corresponding bulk does not _exactly_ match target bulk supercell"


class DefectStencilingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # don't run heavy tests on GH Actions, these are run locally
        cls.heavy_tests = bool(_potcars_available())
        cls.Se_example_dir = os.path.join(EXAMPLE_DIR, "Se")
        cls.Se_20A_bulk_supercell = Structure.from_file(f"{cls.Se_example_dir}/Se_20Å_Supercell_POSCAR")
        cls.Se_222_expanded_supercell = Structure.from_file(
            f"{cls.Se_example_dir}/Se_222_Expanded_Supercell_POSCAR"
        )
        cls.Se_intrinsic_thermo = DefectThermodynamics.from_json(
            f"{cls.Se_example_dir}/Se_Intrinsic_Thermo.json.gz"
        )
        cls.Se_extrinsic_thermo = DefectThermodynamics.from_json(
            f"{cls.Se_example_dir}/Se_Amalgamated_Extrinsic_Thermo.json.gz"
        )
        cls.Se_old_new_names_dict = {
            "vac_1_Se": "v_Se",
            "Int_Se_1": "Se_i_C2",
            "sub_1_H_on_Se": "H_Se",
            "inter_1_F": "F_i_C2",
        }
        for thermo in [cls.Se_intrinsic_thermo, cls.Se_extrinsic_thermo]:
            for old_name in list(thermo.defect_entries.keys()):
                name = old_name
                for key, val in cls.Se_old_new_names_dict.items():
                    name = name.replace(key, val)
                thermo.defect_entries[name] = thermo.defect_entries.pop(old_name)

        # TODO: Test "Generated structure has a minimum interatomic" warnings

    def test_Se_20_A_supercell(self):
        """
        Tests stenciling from the original 13.0 x 13.0 x 14.9 Å 81-atom Se
        supercell to a 20.5 x 20.0 x 20.3 Å 234-atom Se supercell.

        234-atom supercell was generated from
        ``DefectsGenerator(prim_Se, supercell_gen_kwargs={"min_dist":20})``.
        """
        # these supercells were explicitly tested by performing hybrid DFT relaxations from these
        # starting points and comparing to results of unperturbed/rattled supercell relaxations of these
        # defects directly generated (with ``DefectsGenerator``) in this 20Å supercell
        # (for the work described in: https://doi.org/10.1039/D4EE04647A)
        # these defects are good test cases as some are not so trivial; e.g. v_Se_+2 has two inter-chain
        # bridging bonds; see https://doi.org/10.1039/D4EE04647A SI.
        Se_20A_test_supercells = [i for i in os.listdir(self.Se_example_dir) if "20Å_Stenciled" in i]

        previous_bulk = None
        for name, defect_entry in (
            self.Se_intrinsic_thermo.defect_entries | self.Se_extrinsic_thermo.defect_entries
        ).items():
            if name in [i.split("_20Å")[0] for i in Se_20A_test_supercells]:
                print(f"Testing {name}")
                with warnings.catch_warnings(record=True) as w:
                    expanded_defect_supercell, corresponding_bulk = get_defect_in_supercell(
                        defect_entry,
                        self.Se_20A_bulk_supercell,
                    )
                _print_warning_info(w)
                assert not any(
                    "Note that the atomic position basis of the generated defect/bulk supercell differs"
                    in str(warning.message)
                    for warning in w
                )  # previously we got non-tile-matching supercells for this 20Å target_supercell (throwing
                # these warnings), but updated pre-stenciling re-orientation of
                # ``(oriented_)big_{bulk,defect}_supercell`` now returns tile-matching supercells

                # invariant validation tests:
                _validate_stenciled_supercell(
                    expanded_defect_supercell,
                    defect_entry,
                    self.Se_20A_bulk_supercell,
                    corresponding_bulk,
                    check_exact_bulk_match=True,  # we now get tiling match with latest stenciling code
                )

                if previous_bulk is not None:  # check same bulk structure output in each case here
                    assert StructureMatcher_scan_stol(
                        previous_bulk, corresponding_bulk, "fit", primitive_cell=False, max_stol=0.02
                    )
                previous_bulk = corresponding_bulk

                # Note: These direct structure comparisons are the most sensitive tests here, and can break
                # with updated edge-site handling -- which may be perfectly fine, if the other
                # validation tests above pass:
                # expanded_defect_supercell.to(  # uncomment to update reference structures
                #     f"{self.Se_example_dir}/{name}_20Å_Stenciled_POSCAR"
                # )
                reference_struct = Structure.from_file(
                    f"{self.Se_example_dir}/{name}_20Å_Stenciled_POSCAR"
                )
                assert StructureMatcher_scan_stol(reference_struct, expanded_defect_supercell, "fit")

    def test_Se_222_expanded_supercell(self):
        """
        Tests stenciling from the original 13.0 x 13.0 x 14.9 Å 81-atom Se
        supercell to a 2x2x2 expansion of this cell; 26.0 x 26.0 x 29.8 Å
        648-atom supercell.
        """
        # these supercells were explicitly tested by performing hybrid DFT relaxations from these
        # starting points and comparing to results of unperturbed/rattled supercell relaxations of these
        # defects directly generated (with ``DefectsGenerator``) in this 222-expanded supercell
        # (for the work described in: https://doi.org/10.1039/D4EE04647A)
        Se_222_exp_test_supercells = [
            i for i in os.listdir(self.Se_example_dir) if "222_Exp_Stenciled" in i
        ]

        previous_bulk = None
        for name, defect_entry in (
            self.Se_intrinsic_thermo.defect_entries | self.Se_extrinsic_thermo.defect_entries
        ).items():
            if name in [i.split("_222")[0] for i in Se_222_exp_test_supercells]:
                print(f"Testing {name}")
                with warnings.catch_warnings(record=True) as w:
                    expanded_defect_supercell, corresponding_bulk = get_defect_in_supercell(
                        defect_entry,
                        self.Se_222_expanded_supercell,
                    )
                _print_warning_info(w)
                assert not any("Note that the atomic position" in str(warning.message) for warning in w)

                # invariant validation tests:
                _validate_stenciled_supercell(
                    expanded_defect_supercell,
                    defect_entry,
                    self.Se_222_expanded_supercell,
                    corresponding_bulk,
                    check_exact_bulk_match=True,
                )

                if previous_bulk is not None:  # check same bulk structure output in each case here
                    assert StructureMatcher_scan_stol(
                        previous_bulk, corresponding_bulk, "fit", primitive_cell=False, max_stol=0.02
                    )
                previous_bulk = corresponding_bulk

                # Note: These direct structure comparisons are the most sensitive tests here, and can break
                # with updated edge-site handling -- which may be perfectly fine, if the other
                # validation tests above pass:
                # expanded_defect_supercell.to(  # uncomment to update reference structures
                #     f"{self.Se_example_dir}/{name}_222_Exp_Stenciled_POSCAR"
                # )
                reference_struct = Structure.from_file(
                    f"{self.Se_example_dir}/{name}_222_Exp_Stenciled_POSCAR"
                )
                ref_elt_min_max_bond_length_dict = get_element_min_max_bond_length_dict(reference_struct)
                stenciled_elt_min_max_bond_length_dict = get_element_min_max_bond_length_dict(
                    expanded_defect_supercell
                )
                for elt in ref_elt_min_max_bond_length_dict:
                    assert np.allclose(
                        ref_elt_min_max_bond_length_dict[elt],
                        stenciled_elt_min_max_bond_length_dict[elt],
                        atol=1e-3,
                    )

    @custom_mpl_image_compare(filename="Se_v_Se_0_stenciled_vs_original_displacements.png", style=STYLE)
    def test_stenciling_displacement_plot_v_Se_0_20A(self):
        """
        Test 1x2 displacement comparison plot for ``v_Se_0`` stenciled to 20Å
        supercell.

        Left panel: site displacements of the stenciled defect supercell
        relative to the corresponding bulk. Right panel: original DFT defect
        entry displacements.
        The two panels should show very similar patterns, validating the
        stenciling algorithm.
        """
        defect_entry = self.Se_intrinsic_thermo.defect_entries.get("v_Se_0")
        stenciled_supercell, corresponding_bulk = get_defect_in_supercell(
            defect_entry, self.Se_20A_bulk_supercell
        )
        stenciled_entry = _make_stenciled_defect_entry(
            defect_entry, stenciled_supercell, corresponding_bulk
        )
        return _plot_stenciled_vs_original_displacements(stenciled_entry, defect_entry)

    @custom_mpl_image_compare(filename="Se_i_C2_0_stenciled_vs_original_displacements.png", style=STYLE)
    def test_stenciling_displacement_plot_Se_i_C2_0_20A(self):
        """
        Test 1x2 displacement comparison plot for ``Se_i_C2_0`` stenciled to
        20Å supercell.
        """
        defect_entry = self.Se_intrinsic_thermo.defect_entries.get("Se_i_C2_0")
        stenciled_supercell, corresponding_bulk = get_defect_in_supercell(
            defect_entry, self.Se_20A_bulk_supercell
        )
        stenciled_entry = _make_stenciled_defect_entry(
            defect_entry, stenciled_supercell, corresponding_bulk
        )
        return _plot_stenciled_vs_original_displacements(stenciled_entry, defect_entry)

    @custom_mpl_image_compare(filename="H_Se_-1_stenciled_vs_original_displacements.png", style=STYLE)
    def test_stenciling_displacement_plot_H_Se_m1_20A(self):
        """
        Test 1x2 displacement comparison plot for ``H_Se_-1`` stenciled to 20Å
        supercell.
        """
        defect_entry = self.Se_extrinsic_thermo.defect_entries.get("H_Se_-1")
        stenciled_supercell, corresponding_bulk = get_defect_in_supercell(
            defect_entry, self.Se_20A_bulk_supercell
        )
        stenciled_entry = _make_stenciled_defect_entry(
            defect_entry, stenciled_supercell, corresponding_bulk
        )
        return _plot_stenciled_vs_original_displacements(stenciled_entry, defect_entry)
