"""
Tests for the ``doped.utils.stenciling`` module.
"""

import os
import shutil
import unittest
import warnings

from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Composition, IStructure, PeriodicSite, Structure

from doped.thermodynamics import DefectThermodynamics

# use doped efficiency functions for speed in structure-matching testing
from doped.utils.efficiency import Composition as doped_Composition
from doped.utils.efficiency import IStructure as doped_IStructure
from doped.utils.efficiency import PeriodicSite as doped_PeriodicSite
from doped.utils.stenciling import _check_min_dist, get_defect_in_supercell
from doped.utils.supercells import min_dist

Composition.__instances__ = {}
Composition.__eq__ = doped_Composition.__eq__
Composition.__hash__ = doped_Composition.__hash__
PeriodicSite.__eq__ = doped_PeriodicSite.__eq__
PeriodicSite.__hash__ = doped_PeriodicSite.__hash__
IStructure.__instances__ = {}
IStructure.__eq__ = doped_IStructure.__eq__


def if_present_rm(path):
    """
    Remove the file/folder if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _potcars_available() -> bool:
    """
    Check if the POTCARs are available for the tests (i.e. testing locally).
    """
    from doped.vasp import _test_potcar_functional_choice

    try:
        _test_potcar_functional_choice("PBE")
        return True
    except ValueError:
        return False


class DefectStencilingTest(unittest.TestCase):
    def setUp(self):
        # don't run heavy tests on GH Actions, these are run locally
        self.heavy_tests = bool(_potcars_available())
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.CdTe_data_dir = os.path.join(self.data_dir, "CdTe")
        self.example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        self.Se_example_dir = os.path.join(self.example_dir, "Se")
        self.Se_20A_bulk_supercell = Structure.from_file(f"{self.Se_example_dir}/Se_20Å_Supercell_POSCAR")
        self.Se_222_expanded_supercell = Structure.from_file(
            f"{self.Se_example_dir}/Se_222_Expanded_Supercell_POSCAR"
        )
        self.Se_intrinsic_thermo = DefectThermodynamics.from_json(
            f"{self.Se_example_dir}/Se_Intrinsic_Thermo.json.gz"
        )
        self.Se_old_new_names_dict = {"vac_1_Se": "v_Se", "Int_Se_1": "Se_i_C2"}
        self.tight_sm = StructureMatcher(stol=0.02, comparator=ElementComparator(), primitive_cell=False)
        # TODO: Do one of the Se extrinsic substitutions/interstitials as a test too

    def test_Se_20_Å_supercell(self):
        """
        Tests stenciling from the original 13.0 x 13.0 x 14.9 Å 81-atom Se
        supercell to a 20.5 x 20.0 x 20.3 Å 234-atom Se supercell.

        234-atom supercell was generated from
        ``DefectsGenerator(prim_Se, supercell_gen_kwargs={"min_dist":20})``.
        """
        # these supercells were explicitly tested by performing hybrid DFT relaxations from these
        # starting points and comparing to results of unperturbed/rattled supercell relaxations of these
        # defects directly generated (with ``DefectsGenerator``) in this 20Å supercell
        # (for the work described in: https://doi.org/10.26434/chemrxiv-2024-91h02)
        # these defects are good test cases as some are not so trivial; e.g. v_Se_+2 has two inter-chain
        # bridging bonds; see https://doi.org/10.26434/chemrxiv-2024-91h02 SI.
        Se_20A_test_supercells = [i for i in os.listdir(self.Se_example_dir) if "20Å_Stenciled" in i]

        previous_bulk = None
        for old_name, defect_entry in self.Se_intrinsic_thermo.defect_entries.items():
            name = old_name
            for key, val in self.Se_old_new_names_dict.items():
                name = name.replace(key, val)
            if name in [i.split("_20Å")[0] for i in Se_20A_test_supercells]:
                print(f"Testing {name}")
                reference_struct = Structure.from_file(
                    f"{self.Se_example_dir}/{name}_20Å_Stenciled_POSCAR"
                )
                with warnings.catch_warnings(record=True) as w:
                    expanded_defect_supercell, corresponding_bulk = get_defect_in_supercell(
                        defect_entry,
                        self.Se_20A_bulk_supercell,
                    )
                print([str(warning.message) for warning in w])  # for debugging
                assert all(
                    "Note that the atomic position basis of the generated defect/bulk supercell differs"
                    in str(warning.message)
                    for warning in w
                )  # also checks no min_dist warnings

                _check_min_dist(
                    expanded_defect_supercell,
                    orig_min_dist=min_dist(defect_entry.defect_supercell),
                    warning=False,
                )  # check min_dist, error if minimum distance has changed for some reason
                assert self.tight_sm.fit(reference_struct, expanded_defect_supercell)

                # below we also directly compare structures, but note that this may change with updates to
                # the code (and still be fine, there may be multiple quasi-equivalent stenciling choices
                # depending on site selection around the boundary region); so mainly just for tracking
                # changes which affect the final outputs:
                # for comparing the structures, we compared their reloaded POSCARs, as POSCAR
                # writing/reading can introduce some negligible rounding differences in coordinates,
                # so compare like-for-like:
                expanded_defect_supercell.to(  # debugging and like-for-like, removed later if tests pass
                    fmt="vasp", filename=f"{self.Se_example_dir}/{name}_gen_20Å_POSCAR"
                )
                expanded_defect_supercell_from_poscar = Structure.from_file(
                    f"{self.Se_example_dir}/{name}_gen_20Å_POSCAR"
                )
                assert expanded_defect_supercell_from_poscar == reference_struct

                if_present_rm(f"{self.Se_example_dir}/{name}_gen_20Å_POSCAR")  # tests passed, can remove

                if previous_bulk is not None:  # check same bulk structure output in each case here
                    assert previous_bulk == corresponding_bulk

    def test_Se_222_expanded_supercell(self):
        """
        Tests stenciling from the original 13.0 x 13.0 x 14.9 Å 81-atom Se
        supercell to a 2x2x2 expansion of this cell; 26.0 x 26.0 x 29.8 Å
        648-atom supercell.
        """
        # these supercells were explicitly tested by performing hybrid DFT relaxations from these
        # starting points and comparing to results of unperturbed/rattled supercell relaxations of these
        # defects directly generated (with ``DefectsGenerator``) in this 222-expanded supercell
        # (for the work described in: https://doi.org/10.26434/chemrxiv-2024-91h02)
        Se_222_exp_test_supercells = [
            i for i in os.listdir(self.Se_example_dir) if "222_Exp_Stenciled" in i
        ]

        previous_bulk = None
        for old_name, defect_entry in self.Se_intrinsic_thermo.defect_entries.items():
            name = old_name
            for key, val in self.Se_old_new_names_dict.items():
                name = name.replace(key, val)
            if name in [i.split("_222")[0] for i in Se_222_exp_test_supercells]:
                print(f"Testing {name}")
                reference_struct = Structure.from_file(
                    f"{self.Se_example_dir}/{name}_222_Exp_Stenciled_POSCAR"
                )
                with warnings.catch_warnings(record=True) as w:
                    expanded_defect_supercell, corresponding_bulk = get_defect_in_supercell(
                        defect_entry,
                        self.Se_222_expanded_supercell,
                    )
                print([str(warning.message) for warning in w])  # for debugging
                assert not any("Note that the atomic position" in str(warning.message) for warning in w)

                _check_min_dist(
                    expanded_defect_supercell,
                    orig_min_dist=min_dist(defect_entry.defect_supercell),
                    warning=False,
                )  # check min_dist, error if minimum distance has changed for some reason

                expanded_defect_supercell.to(  # for debugging, removed later if tests pass
                    fmt="vasp", filename=f"{self.Se_example_dir}/{name}_gen_222_exp_POSCAR"
                )

                assert self.tight_sm.fit(reference_struct, expanded_defect_supercell)
                # also directly compare structures, but note that this may change with updates to the
                # code (and still be fine, there may be multiple quasi-equivalent stenciling choices
                # depending on site selection around the boundary region); so mainly just for tracking
                # changes which affect the final outputs:
                # compare Structures from POSCARs to compare like for like:
                expanded_defect_supercell_from_poscar = Structure.from_file(
                    f"{self.Se_example_dir}/{name}_gen_222_exp_POSCAR"
                )
                assert expanded_defect_supercell_from_poscar == reference_struct
                if_present_rm(
                    f"{self.Se_example_dir}/{name}_gen_222_exp_POSCAR"
                )  # tests passed, can remove

                if previous_bulk is not None:  # check same bulk structure output in each case here
                    assert previous_bulk == corresponding_bulk

                assert self.tight_sm.fit(self.Se_222_expanded_supercell, corresponding_bulk)
